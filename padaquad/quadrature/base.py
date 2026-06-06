"""
Parallel adaptive-stepsize integration solvers.

This module contains the core parallel integration engine. Unlike traditional
sequential integrators that must evaluate one step at a time (because each step
depends on the previous result), this solver evaluates many integration steps
simultaneously in a batch. This is possible because the integrand f(x) depends
only on x, not on accumulated state -- each step's contribution to the integral
is independent and can be computed in parallel.

The integration domain [mesh_init, mesh_final] is divided by "barriers" into steps.
Within each step, C quadrature points are placed (per the RK tableau). The
solver adaptively refines this mesh: steps with too much error are split into
smaller steps; consecutive steps with very little error are merged.

Key concepts:

- **mesh**: Boundary points dividing [mesh_init, mesh_final] into steps.
  Shape: [M, T] where M is the number of barriers. Step i spans from
  barrier[i] to barrier[i+1].

- **mesh_trackers**: Boolean array of length M. True means the step starting
  at that barrier still needs to be evaluated (or re-evaluated after splitting).

- **Batching**: When there are more steps than fit in GPU memory, the solver
  processes them in batches. Batch size is determined dynamically by measuring
  the memory footprint of the integrand function.

Class hierarchy (defined here):

- ``AdaptiveQuadrature``: Abstract base with the main integrate()
  loop, adaptive step management, error computation, and memory management.

- ``_UniformAdaptiveQuadratureBase``: Concrete subclass for methods with
  fixed tableau c values (quadrature points at constant fractional positions).

- ``_VariableAdaptiveQuadratureBase``: Concrete subclass for methods
  where quadrature points can be at arbitrary positions within each step.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
import gc
from abc import abstractmethod
from typing import TYPE_CHECKING

import psutil
import torch

from padaquad.base import SolverBase
from padaquad.results import IntegrationResult, MethodOutput

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


class AdaptiveQuadrature(SolverBase):
    """
    Base class for parallel adaptive-stepsize numerical integration.

    Implements the main integration loop that:
    1. Initializes a mesh of quadrature step barriers across the integration domain.
    2. Evaluates the integrand at quadrature points within each step (in batches).
    3. Computes integral contributions and error estimates per step using RK methods.
    4. Adaptively refines the mesh: splits high-error steps, merges low-error pairs.
    5. Repeats until all steps meet the error tolerance.
    6. Computes an optimal mesh for potential reuse.

    Subclasses must implement:
        - ``_compute_nodes(mesh_left, mesh_right)``: Place quadrature points within steps.
        - ``_evaluate_adaptive_nodes(...)``: Evaluate integrand at refined points.
        - ``_merge_excess_nodes(...)``: Merge consecutive low-error steps.
        - ``_calculate_integral(t, y, y0)``: Compute RK integral + error for a batch.

    Attributes:
        remove_cut: Error ratio threshold for merging steps (default 0.1).
        max_batch: Maximum number of integrand evaluations per batch. If None,
            determined dynamically from available memory.
        total_mem_usage: Fraction of total memory to use (0 < value <= 1).
        max_path_change: If set, stops integration when the fraction of failing
            steps exceeds this value (used with pre-specified meshes).
        use_absolute_error_ratio: If True, uses the total integral for error
            normalization. If False, uses cumulative integral up to each step.
        error_norm: Vector-error reduction / acceptance scheme (one of "2",
            "max", "rms", "failure_fraction", or a callable). See __init__.
        mesh_failure_tolerance: Fraction of output elements allowed to fail when
            error_norm == "failure_fraction".
        method: The RK method object (set by subclass).
        order: Convergence order of the RK method (set by subclass).
        C: Number of quadrature points per integration step (set by subclass).
        Cm1: C - 1, used frequently in index calculations (set by subclass).
    """

    def __init__(
        self,
        remove_cut: float = 0.1,
        max_batch: int | None = None,
        total_mem_usage: float = 0.9,
        max_path_change: float | None = None,
        max_adaptive_splits: int | None = None,
        use_absolute_error_ratio: bool = True,
        error_norm: str | Callable = "2",
        mesh_failure_tolerance: float = 0.0,
        error_on_nonfinite: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the parallel adaptive solver.

        Args:
            remove_cut: Error ratio threshold below which consecutive step pairs
                are merged. Must be < 1. Lower values keep more steps (more
                conservative). Default: 0.1.
            max_batch: Maximum number of integrand evaluations per batch. If None,
                batch size is determined dynamically based on available memory.
            total_mem_usage: Fraction of total device memory the solver may use
                for batched evaluations (0 < value <= 1). Default: 0.9.
            max_path_change: If set, and the user provides a mesh (mesh is not
                None), integration stops early if this fraction of steps fail
                the error tolerance. Useful for iterative optimization.
            max_adaptive_splits: Maximum number of times a panel may be split
                during adaptive refinement. A panel that has been split this
                many times is accepted even if it still fails the error
                tolerance, instead of being split further. If None (default),
                refinement is uncapped. Can be overridden per call via
                ``integrate(max_adaptive_splits=...)``.
            use_absolute_error_ratio: If True, error ratios use the total
                (converging) integral value. If False, uses cumulative sum up
                to each step. Default: True.
            error_norm: Selects how a vector-valued (D > 1) per-step error is
                reduced to a single accept/reject decision. One of:

                  - ``"2"`` (default, matches ``scipy.integrate.quad_vec``):
                    L2 norm ``sqrt(sum(e**2))`` of the error vector, compared
                    against ``atol + rtol * L2(integral)`` (reduce-then-compare).
                  - ``"max"``: L-infinity norm ``max(|e|)`` (reduce-then-compare).
                  - ``"rms"``: root-mean-square ``sqrt(mean(e**2))``
                    (reduce-then-compare; padaquad's historical reduction).
                  - a callable ``norm(x)`` that reduces the last (D) axis of a
                    ``[..., D]`` tensor to ``[...]`` (reduce-then-compare).
                  - ``"failure_fraction"``: per-component control. Each output
                    element is compared against its own tolerance
                    ``atol + rtol * |integral_d|``; a panel is accepted when the
                    fraction of failing elements is ``<= mesh_failure_tolerance``.

                For D == 1 every option reduces to ``|e| / (atol + rtol*|I|)``.
            mesh_failure_tolerance: Used only when ``error_norm ==
                "failure_fraction"``. Fraction in ``[0, 1]`` of output elements
                that may exceed tolerance while a panel is still accepted.
                Default 0.0 (every element must pass). Can be overridden per
                call via ``integrate(mesh_failure_tolerance=...)``.
            error_on_nonfinite: If True (default), raise a ValueError naming the
                offending ``t`` when the integrand returns NaN/Inf. If False,
                such panels are accepted and the non-finite value propagates
                into the result (the run never hangs in either case). Can be
                overridden per call via ``integrate(error_on_nonfinite=...)``.
            *args: Forwarded to SolverBase (and DistributedEnvironment).
            **kwargs: Forwarded to SolverBase (and DistributedEnvironment).
        """

        super().__init__(*args, **kwargs)
        assert remove_cut < 1.0
        assert max_adaptive_splits is None or max_adaptive_splits >= 0
        self.remove_cut = remove_cut
        
        # Memory handling variables
        self._oom_max_batch = None
        self.previous_max_batch = None
        
        # Construction-time defaults; each integrate() call falls back to these
        # when its corresponding argument is None.
        self.init_max_batch = max_batch
        self.max_path_change = max_path_change
        self.init_max_adaptive_splits = max_adaptive_splits
        self.use_absolute_error_ratio = use_absolute_error_ratio

        self.method = None
        self.order = None
        self.C = None
        self.Cm1 = None
        # Construction-time defaults; the active values live in self.error_norm
        # and self.mesh_failure_tolerance and are (re)set each integrate() call.
        self._check_error_norm(error_norm)
        self._check_mesh_failure_tolerance(mesh_failure_tolerance)
        self.init_error_norm = error_norm
        self.error_norm = error_norm
        self.init_mesh_failure_tolerance = mesh_failure_tolerance
        self.mesh_failure_tolerance = mesh_failure_tolerance
        # Construction-time default; the active value lives in
        # self.error_on_nonfinite and is (re)set each integrate() call.
        self.init_error_on_nonfinite = error_on_nonfinite
        self.error_on_nonfinite = error_on_nonfinite
        self.init_total_mem_usage = total_mem_usage

    #: String values accepted by ``error_norm`` (in addition to a callable).
    _VALID_ERROR_NORMS = ("2", "max", "rms", "failure_fraction")

    @classmethod
    def _check_error_norm(cls, error_norm) -> None:
        """Validate the ``error_norm`` selector (a known string or a callable)."""
        if callable(error_norm):
            return
        if error_norm not in cls._VALID_ERROR_NORMS:
            raise ValueError(
                f"error_norm must be a callable or one of "
                f"{cls._VALID_ERROR_NORMS}, got {error_norm!r}"
            )

    @staticmethod
    def _check_mesh_failure_tolerance(mesh_failure_tolerance) -> None:
        """Validate that ``mesh_failure_tolerance`` is a fraction in [0, 1]."""
        if not (0.0 <= mesh_failure_tolerance <= 1.0):
            raise ValueError(
                "mesh_failure_tolerance must be in [0, 1], got "
                f"{mesh_failure_tolerance!r}"
            )

    # -------------------------------------------------------------------------------- #
    #                                 ABSTRACT METHODS                                 #
    # -------------------------------------------------------------------------------- #

    @abstractmethod
    def _evaluate_adaptive_nodes(
        self,
        f: Callable,
        idxs_add: torch.Tensor,
        y: torch.Tensor,
        nodes: torch.Tensor,
        f_args: tuple = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the integrand at new points created by splitting failed steps.

        When steps fail the error tolerance, they are split into two smaller
        steps. This method evaluates the integrand at the new quadrature points
        needed for the smaller steps. The implementation differs between uniform
        and variable solvers.

        Args:
            f: The integrand function.
            idxs_add: Indices of steps that need to be split. Shape: [n_add].
            y: Current integrand evaluations for all steps. Shape: [N, C, D].
            nodes: Current quadrature point positions for all steps.
                Shape: [N, C, T].
            f_args: Extra arguments passed to f.

        Returns:
            Tuple of (y_new, nodes_new): integrand evaluations and quadrature
            point positions for the replacement (split) steps.
        """

    @abstractmethod
    def _merge_excess_nodes(
        self, nodes, mesh_quadratures, mesh_quadrature_errors, remove_idxs
    ):
        """
        Merges neighboring quadrature steps or removes and one quadtrature steps
        and extends its neighbor to cover the same range.

        Args:
            nodes (Tensor): Per-step quadrature point positions.
            remove_idxs (Tensor): First index of neighboring steps needed to be
                merged, or remove at given index and extend the following step

        Shapes:
            nodes : [N, C, T]
            removed_idxs : [n]
        """

    # -------------------------------------------------------------------------------- #
    #                            PRIMARY INTEGRATION METHOD                            #
    # -------------------------------------------------------------------------------- #

    def integrate(
        self,
        f: Callable | None = None,
        y0: torch.Tensor | None = None,
        mesh: torch.Tensor | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        reuse_mesh: bool = False,
        random_initial_mesh: bool = True,
        N_init_steps: int = 13,
        f_args: tuple = (),
        take_gradient: bool = True,
        total_mem_usage: float | None = None,
        loss_fxn: Callable | None = None,
        max_batch: int | None = None,
        max_adaptive_splits: int | None = None,
        error_norm: str | Callable | None = None,
        mesh_failure_tolerance: float | None = None,
        error_on_nonfinite: bool | None = None,
    ) -> IntegrationResult:
        """
        Perform parallel adaptive numerical integration of f.

        This is the main integration loop. It divides [mesh_init, mesh_final] into
        steps using barrier points, evaluates the integrand in parallel batches,
        and adaptively refines the mesh until all steps meet the error tolerance.

        The algorithm:
        1. Initialize barriers (random mesh or user-provided).
        2. While unevaluated steps remain:
           a. Select a batch of steps that fits in memory.
           b. Place C quadrature points in each step.
           c. Evaluate the integrand at all points in parallel.
           d. Compute integral contributions and error estimates via RK.
           e. Accept steps with error_ratio < 1; split the rest.
           f. Record accepted results.
        3. Optimize the final mesh (prune over-resolved + refine under-resolved).
        4. Return the integral, error, and diagnostics.

        Args:
            f: The integrand f(t). Takes shape [N, T], returns [N, D].
                If None, uses the function from construction.
            y0: Initial integral accumulator value. Shape: [D].
            mesh: Optional initial step barriers. If provided, these are the
                starting mesh. If None, a random mesh is generated. Shape: [N, T].
            mesh_init: Lower integration bound. Shape: [T].
            mesh_final: Upper integration bound. Shape: [T].
            N_init_steps: Approximate number of initial steps when mesh is None.
                The actual count is ~sqrt(N_init_steps) segments with
                ~sqrt(N_init_steps)+1 random sub-barriers each.
            f_args: Extra arguments passed to f.
            take_gradient: If True, calls loss.backward() after each batch
                to compute gradients through the integration.
            total_mem_usage: Fraction of memory to use for batching. Overrides
                the value from construction if provided.
            loss_fxn: Custom loss function. Takes an IntegrationResult, returns a
                scalar tensor. If None, uses the integral value itself.
            max_batch: Maximum evaluations per batch. Overrides dynamic memory
                calculation if provided.
            max_adaptive_splits: Maximum number of times a panel may be split
                during adaptive refinement; a panel split this many times is
                accepted even if it still fails the tolerance. If None
                (default), falls back to the value from construction (also
                None ⇒ uncapped). When both are set, this per-call value takes
                priority.
            error_norm: Vector-error reduction / acceptance scheme (see
                __init__). If None (default), falls back to the value from
                construction. When set, this per-call value takes priority.
            mesh_failure_tolerance: Fraction of output elements allowed to fail
                when ``error_norm == "failure_fraction"`` (see __init__). If
                None (default), falls back to the value from construction. When
                set, this per-call value takes priority.
            error_on_nonfinite: If True, raise a ValueError naming the offending
                ``t`` when the integrand returns NaN/Inf; if False, accept those
                panels and let the non-finite value propagate into the result.
                If None (default), falls back to the value from construction
                (which defaults to True). When both are set, this per-call value
                takes priority.
            random_initial_mesh: When True (default), the fresh initial
                mesh is built with random sub-barrier offsets within each
                top-level segment. Randomness is essential here, not
                cosmetic: when the integrand has features at uniformly-
                spaced positions (e.g. zeros of ``sin(2*pi*k*t)``,
                polynomial extrema), an evenly-spaced mesh can align
                with those features in a way the adaptive controller
                cannot recover from. Random offsets break this
                alignment. Set to False only for debugging or for
                integrands you have separately verified to be safe
                against uniform-mesh aliasing; reproducibility is
                better achieved via ``torch.manual_seed`` before the
                call.
            reuse_mesh: When True, seed the integration from the optimal mesh
                cached by the previous successful call (warm start). Default
                False. The cached mesh is the *optimal* mesh produced after
                prune-and-refine on the previous call; reusing it across
                training-loop iterations where the integrand changes only
                slightly between calls saves substantial adaptive-refinement
                cost. If reuse_mesh=True but no cache exists, falls back to
                a fresh initial mesh and emits a warning. If the cached mesh
                was produced for a different integrand (id mismatch), emits
                a warning but proceeds. Ignored when ``mesh`` is provided
                explicitly (the explicit ``mesh`` always takes precedence).

        Returns:
            IntegrationResult with the computed integral, error estimates, the
            optimized mesh (mesh_optimal), per-step nodes, and diagnostics.

        Note:
            If mesh is provided, the solver uses these as initial barriers.
            Steps may be split or merged, but the bounds [mesh[0], mesh[-1]]
            are preserved. If mesh is None and reuse_mesh is False (default),
            a random initial mesh is generated in [mesh_init, mesh_final].
        """
        # Set dtype based on input
        self.set_dtype_by_input(mesh=mesh, mesh_init=mesh_init, mesh_final=mesh_init)

        # If mesh is given set mesh_init and mesh_final, else use input, else use saved values
        mesh_init, mesh_final = self._setup_integral_bounds(mesh, mesh_init, mesh_final)

        # Replace max_batch if default it given
        force_max_batch = self.init_max_batch if max_batch is None else max_batch

        # Per-call max_adaptive_splits takes priority over the constructor value.
        max_adaptive_splits = (
            self.init_max_adaptive_splits
            if max_adaptive_splits is None
            else max_adaptive_splits
        )

        # Per-call error_on_nonfinite takes priority over the constructor value.
        # Stored on self so the evaluation helpers can read it without threading
        # it through every signature.
        self.error_on_nonfinite = (
            self.init_error_on_nonfinite
            if error_on_nonfinite is None
            else error_on_nonfinite
        )

        # Per-call error_norm / mesh_failure_tolerance take priority over the
        # constructor values. Stored on self so the error-ratio helpers read
        # them without threading them through every signature.
        self.error_norm = self.init_error_norm if error_norm is None else error_norm
        self._check_error_norm(self.error_norm)
        self.mesh_failure_tolerance = (
            self.init_mesh_failure_tolerance
            if mesh_failure_tolerance is None
            else mesh_failure_tolerance
        )
        self._check_mesh_failure_tolerance(self.mesh_failure_tolerance)

        # Get variables or populate with default values, send to correct device
        f, mesh_init, mesh_final, y0 = self._check_variables(
            f, mesh_init, mesh_final, y0
        )
        # Coerce any tensor in f_args onto the solver device too (it is forwarded
        # to f(t, *f_args)); DistributedEnvironment owns the device and every
        # input is moved onto it.
        f_args = tuple(a.to(self.device) if torch.is_tensor(a) else a for a in f_args)
        total_mem_usage = (
            self.init_total_mem_usage if total_mem_usage is None else total_mem_usage
        )
        MEM_ERROR = "total_mem_usage is a ratio and must be 0 < total_mem_usage <= 1"
        assert total_mem_usage <= 1.0, MEM_ERROR
        assert total_mem_usage > 0, MEM_ERROR
        # Has memory been benchmarked for this integrand yet? Skip the
        # benchmark if id(f) matches what we've already measured.
        # Using id() avoids the lambda-collision bug present in earlier
        # versions which compared f.__name__ (every lambda has
        # __name__ == "<lambda>").
        same_integrand_fxn = (
            self.previous_f_id is not None and id(f) == self.previous_f_id
        )
        # Use the previous max_batch, running memory check everytime is slow
        if force_max_batch is None:
            force_max_batch = self.previous_max_batch
            
        # Benchmark memory footprint on first call with a new integrand
        if not same_integrand_fxn and force_max_batch is None:
            self._setup_memory_checks(
                f, mesh_init, take_gradient=take_gradient, f_args=f_args
            )
        # From previous version
        # assert self._get_max_f_evals(total_mem_usage) > (2 * self.Cm1 + 1), (
        #    "Not enough free memory to run 2 integration steps, consider increasing total_mem_usage"
        # )
        loss_fxn = loss_fxn if loss_fxn is not None else self._integral_loss

        # Make sure f exists and provides the correct output
        assert f is not None, "Must specify f or pass it during class initialization."
        test_output = f(
            torch.tensor([[mesh_init]], dtype=self.dtype, device=self.device), *f_args
        )
        test_integrand, _ = self._split_f_output(test_output)
        assert len(test_integrand.shape) >= 2
        del test_output, test_integrand

        # Decide initial mesh:
        #   - explicit mesh passed in: use it (always takes precedence);
        #   - reuse_mesh=True with a populated cache: warm-start from the
        #     cached optimal mesh, snapping its endpoints to [mesh_init, mesh_final];
        #   - otherwise: generate a fresh random initial mesh.
        mesh, mesh_trackers, mesh_is_given = self._setup_initial_mesh(
            mesh,
            mesh_init,
            mesh_final,
            reuse_mesh,
            same_integrand_fxn,
            random_initial_mesh,
            N_init_steps,
        )

        record = {}
        split_node_state = (None, None, None, None)
        # Per-panel split counter, parallel to mesh/mesh_trackers. Only
        # maintained when a depth cap is active (None ⇒ zero overhead, no
        # behavior change).
        split_counts = (
            torch.zeros(len(mesh), dtype=torch.long, device=self.device)
            if max_adaptive_splits is not None
            else None
        )
        # === Main integration loop ===
        # Continues until all steps have been evaluated and accepted
        # (mesh_trackers[i] == False for all i)
        while torch.any(mesh_trackers):
            # From earlier debugging
            # if y is not None:
            #    assert max_steps >= len(y), f"{max_steps}  {len(y)}"

            (
                nodes,
                y_step_eval,
                tracked_step_eval,
                step_idxs,
                split_node_state,
            ) = self._evaluate_f_on_mesh(
                f,
                f_args,
                mesh,
                mesh_trackers,
                take_gradient,
                force_max_batch,
                total_mem_usage,
                split_node_state,
            )

            # --- Step 3: Compute integral contributions via qudrature formula ---
            t0 = time.time()
            method_output = self._calculate_integral(
                nodes,
                y_step_eval,
                y0=torch.zeros(1, device=self.device, dtype=self.dtype),
            )
            if len(record) == 0:
                # First batch: integral is just this batch's contribution
                current_integral = method_output.integral.detach()
                all_mesh_quadratures = method_output.mesh_quadratures.detach()
                cum_mesh_quadratures = torch.cumsum(all_mesh_quadratures, 0)
            else:
                # Subsequent batches: add to previously recorded integral
                current_integral = record["integral"] + method_output.integral.detach()
                # Merge new steps into the sorted record to compute cumulative sums
                idxs_keep, idxs_input = self._get_sorted_indices(
                    record["nodes"][:, 0, 0], nodes[:, 0, 0]
                )
                all_mesh_quadratures = self._insert_sorted_results(
                    record["mesh_quadratures"],
                    idxs_keep,
                    method_output.mesh_quadratures,
                    idxs_input,
                )
                cum_mesh_quadratures = torch.cumsum(all_mesh_quadratures, 0)[idxs_input]
            if self.speed_logger:
                self.speed_logger.debug("calc integrals: %s", time.time() - t0)

            # --- Step 4: Compute error ratios for each step ---
            t0 = time.time()
            error_ratios, error_ratios_2steps, error_ratios_per_dim = (
                self._compute_error_ratios(
                    mesh_quadrature_errors=method_output.mesh_quadrature_errors,
                    mesh_quadratures=method_output.mesh_quadratures,
                    cum_mesh_quadratures=cum_mesh_quadratures,
                    integral=current_integral,
                )
            )
            keep_mask, remove_mask = self._accept_reject_masks(
                error_ratios, error_ratios_per_dim
            )
            if self.speed_logger:
                self.speed_logger.debug("calculate errors: %s", time.time() - t0)
            assert len(y_step_eval) == len(error_ratios)
            assert len(y_step_eval) - 1 == len(error_ratios_2steps), (
                f" y: {y_step_eval.shape} | ratios: {error_ratios_2steps.shape} | nodes: {nodes.shape}"
            )
            logger.debug("error_ratios: %s", error_ratios)
            logger.debug("error_ratios_2steps: %s", error_ratios_2steps)

            # Early exit if too many steps fail and user-provided mesh is given.
            # Bug B6 fix: previously returned bare `None`, breaking the
            # documented return-type contract. Now returns an
            # IntegrationResult with converged=False populated from the
            # most-recent batch's intermediate result so callers can
            # inspect partial state instead of having to special-case
            # None.
            if mesh_is_given and self.max_path_change is not None:
                # A "failed step" is one that would be split under the active
                # error_norm scheme (consistent with the accept/reject masks).
                fail_ratio = remove_mask.to(float).sum() / len(remove_mask)
                if fail_ratio >= self.max_path_change:
                    logger.warning(
                        "%.1f%% of integration steps failed error requirements, "
                        "which is greater than max_path_change (%s), now exiting.",
                        fail_ratio * 100,
                        self.max_path_change,
                    )
                    return IntegrationResult(
                        integral=method_output.integral,
                        integral_error=method_output.integral_error,
                        mesh_optimal=mesh,
                        mesh_init=mesh_init,
                        mesh_final=mesh_final,
                        nodes=nodes,
                        h=method_output.h,
                        y=y_step_eval,
                        tracked_variables=tracked_step_eval,
                        mesh_quadratures=method_output.mesh_quadratures,
                        mesh_quadrature_errors=torch.abs(
                            method_output.mesh_quadrature_errors
                        ),
                        error_ratios=error_ratios,
                        loss=None,
                        gradient_taken=take_gradient,
                        y0=y0,
                        converged=False,
                    )

            # --- Step 5: Adaptive refinement ---
            # Split steps with error_ratio >= 1, keep steps with error_ratio < 1,
            # and update barriers/trackers accordingly
            (
                method_output,
                y_step_eval,
                tracked_step_eval,
                nodes,
                mesh,
                mesh_trackers,
                error_ratios,
                split_counts,
            ) = self._adaptively_increase_mesh(
                method_output=method_output,
                error_ratios=error_ratios,
                y_step_eval=y_step_eval,
                nodes=nodes,
                mesh=mesh,
                mesh_idxs=step_idxs,
                mesh_trackers=mesh_trackers,
                tracked_step_eval=tracked_step_eval,
                split_counts=split_counts,
                max_adaptive_splits=max_adaptive_splits,
                keep_mask=keep_mask,
                remove_mask=remove_mask,
            )
            # Verify barrier ordering after adaptive refinement
            mesh_diff = mesh[1:, 0] - mesh[:-1, 0]
            assert torch.all(mesh_diff + self.atol_assert > 0) or torch.all(
                mesh_diff - self.atol_assert < 0
            )

            # --- Step 6: Record accepted results and handle gradients ---
            if nodes.shape[0] > 0:
                # take_gradient = take_gradient or (
                #    self.training and (torch.any(mesh_trackers) or take_gradient)
                # )
                intermediate_results = IntegrationResult(
                    integral=method_output.integral,
                    integral_error=method_output.integral_error,
                    nodes=nodes,
                    h=method_output.h,
                    y=y_step_eval,
                    tracked_variables=tracked_step_eval,
                    mesh_quadratures=method_output.mesh_quadratures,
                    mesh_quadrature_errors=torch.abs(
                        method_output.mesh_quadrature_errors
                    ),
                    error_ratios=error_ratios,
                    loss=None,
                    gradient_taken=take_gradient,
                    mesh_init=mesh_init,
                    mesh_final=mesh_final,
                    y0=0,
                )

                # TODO make sure growing string loss center is a time not the number of evals because eval number is meaningless here.
                # Compute loss and accumulate into the record
                loss = loss_fxn(intermediate_results)
                intermediate_results.loss = loss
                record = self._record_results(
                    record=record,
                    take_gradient=take_gradient,
                    results=intermediate_results,
                )

                # Backpropagate gradients through the integration if requested
                if take_gradient and loss.requires_grad:
                    loss.backward()
            del y_step_eval

        # === Post-convergence: sort results and optimize the mesh ===
        record = self._sort_record(record)
        # Prune over-resolved steps and refine under-resolved ones
        mesh_optimal = self._get_optimal_mesh(record, mesh)
        # Cache results for warm-starting subsequent calls with the same integrand
        self.mesh_previous = mesh_optimal
        self.previous_f_id = id(f)

        # Tracked variables are stored in the record as a list; expose a tuple.
        record_tracked: tuple[torch.Tensor, ...] | None = None
        if "tracked_variables" in record:
            record_tracked = tuple(record["tracked_variables"])

        return IntegrationResult(
            integral=record["integral"] + y0,
            integral_error=record["integral_error"],
            mesh_optimal=mesh_optimal,
            mesh_init=mesh_init,
            mesh_final=mesh_final,
            nodes=record["nodes"],
            h=record["h"],
            y=record["y"],
            tracked_variables=record_tracked,
            mesh_quadratures=record["mesh_quadratures"],
            mesh_quadrature_errors=torch.abs(record["mesh_quadrature_errors"]),
            error_ratios=record["error_ratios"],
            loss=record["loss"],
            gradient_taken=take_gradient,
            y0=y0,
        )

    # -------------------------------------------------------------------------------- #
    #                             ADAPTIVE MESH REFINEMENT                             #
    # -------------------------------------------------------------------------------- #

    def _adaptively_increase_mesh(
        self,
        method_output: MethodOutput | None,
        error_ratios: torch.Tensor,
        y_step_eval: torch.Tensor | None,
        nodes: torch.Tensor | None,
        mesh: torch.Tensor,
        mesh_idxs: torch.Tensor,
        mesh_trackers: torch.Tensor,
        tracked_step_eval: tuple[torch.Tensor, ...] | None = None,
        split_counts: torch.Tensor | None = None,
        max_adaptive_splits: int | None = None,
        keep_mask: torch.Tensor | None = None,
        remove_mask: torch.Tensor | None = None,
    ) -> tuple[
        MethodOutput | None,
        torch.Tensor | None,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Accept accurate steps and split inaccurate ones.

        This is the core adaptive refinement operation. For each evaluated step:
        - If error_ratio < 1.0: ACCEPT the step. Mark it as done in mesh_trackers.
          Keep its integral contribution.
        - If error_ratio >= 1.0: REJECT the step. Insert a new midpoint barrier
          between its start and end, splitting it into two smaller steps. These
          new steps will be evaluated in the next iteration.
        - If error_ratio is non-finite (NaN/Inf): ACCEPT the step. Splitting
          cannot reduce a non-finite error (the integrand is singular at a
          node), so such steps are accepted rather than split forever. The
          non-finite value propagates into the recorded integral.

        The midpoint barrier is placed at the average of the two neighboring
        barriers: mesh_new = (mesh_left + mesh_right) / 2.

        When ``split_counts`` and ``max_adaptive_splits`` are provided, a step
        whose split count has already reached ``max_adaptive_splits`` is
        accepted even if its error_ratio >= 1.0 (it is not split further). Each
        split increments the count: both children of a depth-N step are depth
        N+1.

        Args:
            method_output: RK results from the current batch (may be None when
                called during post-convergence optimization). If present, rejected
                steps are removed from it.
            error_ratios: Per-step error ratios. Shape: [N_batch].
            y_step_eval: Integrand evaluations for current batch. Shape: [N_batch, C, D].
            nodes: Quadrature points for current batch. Shape: [N_batch, C, T].
            mesh: All barrier positions. Shape: [M, T].
            mesh_idxs: Indices into mesh for the steps
                in the current batch. Shape: [N_batch].
            mesh_trackers: Boolean array tracking which steps need evaluation.
                Shape: [M].
            tracked_step_eval: Optional tuple of tracked-variable tensors for
                the current batch, each shape [N_batch, C, *var_dims]. Filtered
                to accepted steps alongside y_step_eval. None when the integrand
                emits no tracked variables.
            split_counts: Optional per-panel split-count array aligned with
                mesh. Shape: [M]. None disables depth tracking (the
                post-convergence caller passes None).
            max_adaptive_splits: Optional cap on the per-panel split count.
                Panels at or above this count are accepted instead of split.
                None disables the cap.
            keep_mask: Optional precomputed accept mask (True = accept), shape
                [N_batch], from ``_accept_reject_masks``. When provided together
                with ``remove_mask`` the scheme-aware decision is used as-is;
                when omitted the legacy ``error_ratios < 1.0`` rule is applied
                (kept for direct/unit callers).
            remove_mask: Optional precomputed reject (split) mask, the
                complement of ``keep_mask``.

        Returns:
            Tuple of (method_output, y_step_eval, tracked_step_eval, nodes,
            mesh_new, mesh_trackers_new, error_ratios_kept, split_counts_new):
                - method_output: Updated with rejected steps removed.
                - y_step_eval: Kept evaluations only.
                - tracked_step_eval: Kept tracked variables only (or None).
                - nodes: Kept quadrature points only.
                - mesh_new: Barriers with new midpoints inserted.
                - mesh_trackers_new: Updated tracker with new steps marked True.
                - error_ratios_kept: Error ratios for accepted steps only.
                - split_counts_new: Per-panel split counts for mesh_new, with
                  split children incremented (or None if split_counts is None).
        """
        if keep_mask is None or remove_mask is None:
            # Legacy/direct-call path: derive the masks from a scalar error
            # ratio. Non-finite error ratios (NaN/Inf) cannot be reduced by
            # splitting: the integrand returned NaN/Inf at a node, and a midpoint
            # split reuses the same boundary node, so the value regenerates.
            # Accept such panels unconditionally so the main loop always
            # terminates -- with the default uncapped max_adaptive_splits,
            # routing them to the split branch would refine forever (and a +Inf
            # ratio is >= 1.0, so it would otherwise be split forever too).
            nonfinite = ~torch.isfinite(error_ratios)
            keep_mask = (error_ratios < 1.0) | nonfinite
            remove_mask = (error_ratios >= 1.0) & ~nonfinite
        # Panels that have already been split the maximum number of times are
        # accepted as-is rather than split further.
        if max_adaptive_splits is not None and split_counts is not None:
            at_max = remove_mask & (split_counts[mesh_idxs] >= max_adaptive_splits)
            keep_mask = keep_mask | at_max
            remove_mask = remove_mask & ~at_max
        mesh_trackers[mesh_idxs[keep_mask]] = False

        N_t_add = torch.sum(remove_mask)
        # Allocate new barriers array with room for inserted midpoints
        mesh_new = torch.nan * torch.ones(
            (N_t_add + len(mesh), mesh.shape[-1]),
            dtype=self.dtype,
            device=self.device,
        )
        mesh_trackers_new = torch.ones(
            N_t_add + len(mesh), dtype=bool, device=self.device
        )

        # Transfer existing barriers to their new positions in the expanded array.
        # Each rejected step causes a +1 offset for all subsequent barriers
        # (because a midpoint is being inserted). idx_offset tracks this shift.
        idx_offset = torch.zeros(len(mesh), dtype=torch.long, device=self.device)
        idx_offset[mesh_idxs[remove_mask] + 1] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idxs_transfer = idx_offset + torch.arange(len(mesh), device=self.device)
        mesh_new[idxs_transfer] = mesh.clone()
        mesh_trackers_new[idxs_transfer] = mesh_trackers.clone()

        # Insert new midpoint barriers between the start and end of rejected steps.
        # The midpoint is placed at (left_barrier + right_barrier) / 2.
        idxs_new = (
            mesh_idxs[remove_mask] + torch.arange(N_t_add, device=self.device) + 1
        )
        t_add_barriers = 0.5 * (mesh_new[idxs_new - 1] + mesh_new[idxs_new + 1])
        mesh_new[idxs_new] = t_add_barriers
        assert torch.sum(torch.isnan(mesh_new)) == 0
        assert len(idxs_new) + len(idxs_transfer) == len(mesh_new)

        # Expand the per-panel split counts the same way as the barriers: carry
        # existing counts to their new positions, then bump both children of
        # each split panel to parent_count + 1.
        if split_counts is not None:
            split_counts_new = torch.zeros(
                len(mesh_new), dtype=torch.long, device=self.device
            )
            split_counts_new[idxs_transfer] = split_counts.clone()
            parent_counts = split_counts[mesh_idxs[remove_mask]]
            # left child = transferred parent barrier; right child = new midpoint
            split_counts_new[idxs_transfer[mesh_idxs[remove_mask]]] = parent_counts + 1
            split_counts_new[idxs_new] = parent_counts + 1
        else:
            split_counts_new = None

        if method_output is not None:
            method_output.mesh_quadratures = method_output.mesh_quadratures[keep_mask]
            method_output.mesh_quadrature_errors = method_output.mesh_quadrature_errors[
                keep_mask
            ]
            method_output.h = method_output.h[keep_mask]
            method_output.integral = torch.sum(method_output.mesh_quadratures, 0)
            method_output.integral_error = torch.sum(
                method_output.mesh_quadrature_errors, 0
            )
        if y_step_eval is not None:
            y_step_eval = y_step_eval[keep_mask]
        if tracked_step_eval is not None:
            tracked_step_eval = tuple(tv[keep_mask] for tv in tracked_step_eval)
        if nodes is not None:
            nodes = nodes[keep_mask]
        return (
            method_output,
            y_step_eval,
            tracked_step_eval,
            nodes,
            mesh_new,
            mesh_trackers_new,
            error_ratios[keep_mask],
            split_counts_new,
        )

    @staticmethod
    def _split_f_output(
        out,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """
        Normalize the integrand's output into (integrand, tracked_variables).

        The integrand ``f`` may return either:
        - a bare integrand tensor (existing contract) -> tracked is None, or
        - a 2-tuple ``(integrand, tracked_variables)`` where
          ``tracked_variables`` is a tuple of tensors, a single tensor (wrapped
          into a 1-tuple), or ``None``.

        Args:
            out: The raw return value of a call to ``f``.

        Returns:
            Tuple of (integrand_tensor, tracked_tuple_or_None).
        """
        if isinstance(out, (tuple, list)) and len(out) == 2:
            integrand, tracked = out
            if tracked is None:
                return integrand, None
            if torch.is_tensor(tracked):
                tracked = (tracked,)
            return integrand, tuple(tracked)
        return out, None

    def _check_f_output_finite(self, integrand, nodes_flat, max_report=5):
        """
        Raise a located ``ValueError`` if an integrand batch is non-finite.

        A NaN/Inf integrand value poisons the panel's error ratio. Detecting it
        at the source (rather than letting it silently propagate into a NaN
        result, or stall the adaptive loop) gives the user an actionable error
        naming the offending ``t``. Gated on ``self.error_on_nonfinite`` (set
        from the ``integrate(error_on_nonfinite=...)`` argument).

        Args:
            integrand: The integrand tensor for this batch (shape ``[B, D]``,
                with ``B = N*C`` for full nodes or the slice size for split
                nodes). May also be a bare Python scalar if ``f`` returns a
                number, in which case finiteness is checked with
                ``math.isfinite`` and no node localization is reported.
            nodes_flat: The flattened node coordinates fed to ``f`` for this
                batch. Shape ``[B, T]``. Used only to report offending ``t``.
            max_report: Maximum number of offending ``t`` values to list.

        Raises:
            ValueError: If ``self.error_on_nonfinite`` and any value is
                non-finite. When ``self.error_on_nonfinite`` is False this is a
                no-op; the always-on ``_adaptively_increase_mesh`` guard keeps
                the run alive instead.
        """
        if not self.error_on_nonfinite:
            return
        if not torch.is_tensor(integrand):
            # Bare scalar (e.g. examples.identity returns int 1). math.isfinite
            # handles int/float; there is no node to localize to.
            if not math.isfinite(integrand):
                raise ValueError(
                    f"Integrand f returned a non-finite scalar value "
                    f"({integrand}). Pass error_on_nonfinite=False to return a "
                    f"NaN-containing result instead of raising."
                )
            return
        with torch.no_grad():
            finite = torch.isfinite(integrand)
            if finite.all():
                return
            # Reduce over the output (D) dimension: a node is bad if ANY of its
            # output components is non-finite. [B, D] -> [B].
            bad_rows = ~finite.reshape(integrand.shape[0], -1).all(dim=1)
            bad_idx = torch.nonzero(bad_rows, as_tuple=False).flatten()
            n_bad = int(bad_idx.numel())
            bad_t = nodes_flat[bad_idx[:max_report]].detach().cpu().tolist()
        suffix = "" if n_bad <= max_report else f" (+{n_bad - max_report} more)"
        raise ValueError(
            f"Integrand f returned non-finite values (NaN/Inf) at {n_bad} of "
            f"{integrand.shape[0]} evaluation node(s). First offending t "
            f"value(s): {bad_t}{suffix}. This usually means f is singular there "
            f"(e.g. a boundary singularity or degenerate parameters). Pass "
            f"error_on_nonfinite=False to accept the affected panels and return "
            f"a NaN-containing result instead of raising."
        )

    def _evaluate_f_on_mesh(
        self,
        f,
        f_args,
        mesh,
        mesh_trackers,
        take_gradient,
        force_max_batch,
        total_mem_usage,
        split_node_state,
    ):
        # Determine how many f evaluations fit in one batch based on memory
        if force_max_batch is not None:
            max_batch = force_max_batch
        else:
            max_batch = self._get_max_f_evals(total_mem_usage)
        self.previous_max_batch = max_batch

        # max_batch should not exceed the remaining number of f evaluations
        batches_left = torch.sum(mesh_trackers) * self.C
        assert batches_left > 0
        max_batch = max_batch if max_batch < batches_left else batches_left
        if self._oom_max_batch is not None and max_batch > self._oom_max_batch:
            max_batch = self._oom_max_batch
        max_mesh_steps = max_batch // self.C

        step_idxs = torch.arange(len(mesh), device=self.device)
        step_idxs = step_idxs[mesh_trackers]

        if take_gradient:
            return self._evaluate_f_on_full_nodes(
                f, f_args, mesh, step_idxs, max_mesh_steps
            )
        else:
            return self._evaluate_f_on_split_nodes(
                f,
                f_args,
                mesh,
                step_idxs,
                max_batch,
                max_mesh_steps,
                split_node_state,
            )

    def _evaluate_f_on_full_nodes(self, f, f_args, mesh, step_idxs, max_mesh_steps):
        assert max_mesh_steps >= 1, (
            "Not enough free memory to run 1 integration steps for take_gradient=True. Set take_gradient=False if the gradient of the integral is not needed and consider increasing total_mem_usage"
        )
        # Find barrier indices where mesh_trackers is True, take up to max_steps
        step_idxs = step_idxs[:max_mesh_steps]
        # Place C quadrature points within each selected step
        nodes = self._compute_nodes(mesh[step_idxs], mesh[step_idxs + 1])

        # Flatten [N, C, T] -> [N*C, T] for batch evaluation, then reshape back
        nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=-2)
        assert torch.all(nodes_flat[1:] - nodes_flat[:-1] + self.atol_assert >= 0)
        node_evals, tracked = self._split_f_output(f(nodes_flat, *f_args))
        self._check_f_output_finite(node_evals, nodes_flat)

        # Reshape nodes and evaluations before returning
        unflatten = (len(step_idxs), self.C, -1)
        nodes = torch.reshape(nodes_flat, unflatten)
        f_evals = torch.reshape(node_evals, unflatten)

        # Tracked variables ride alongside the integrand: reshape each to
        # [N, C, *var_dims] and detach (diagnostic-only, no autograd).
        tracked_out = None
        if tracked is not None:
            tracked_out = tuple(
                tv.reshape(len(step_idxs), self.C, *tv.shape[1:]).detach()
                for tv in tracked
            )

        return nodes, f_evals, tracked_out, step_idxs, (None, None, None, None)

    def _evaluate_f_on_split_nodes(
        self,
        f,
        f_args,
        mesh,
        step_idxs,
        max_batch,
        max_mesh_steps,
        split_node_state,
    ):
        # Gather nodes and evaluations from previous split. split_mesh_idx
        # stores the residual panel's left-barrier *coordinate* (not an
        # integer index), so it survives mesh refinement: the calling
        # integrate loop inserts midpoints between iterations, shifting
        # any cached integer index. Barriers themselves are never removed
        # mid-loop, so a stored coordinate stays resolvable.
        split_nodes, split_f_evals, split_tracked, split_mesh_idx = split_node_state
        num_split_nodes = 0 if split_mesh_idx is None else len(split_nodes)
        num_remaining_split_nodes = self.C - num_split_nodes

        if split_mesh_idx is not None:
            # Translate the cached barrier coordinate back to an index in
            # the current (possibly refined) mesh.
            split_mesh_idx = torch.where((mesh == split_mesh_idx).all(dim=-1))[0][0]

        if split_mesh_idx is None:
            evaluate_all = (max_mesh_steps * self.C) % max_batch == 0
            num_mesh_steps = max_mesh_steps if evaluate_all else max_mesh_steps + 1

            step_idxs = step_idxs[:num_mesh_steps]

            # Place C quadrature points within each selected step
            nodes = self._compute_nodes(mesh[step_idxs], mesh[step_idxs + 1])

            # Flatten [N, C, T] -> [N*C, T] for batch evaluation, then reshape back
            nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=-2)

            # Determine the number of evaluation iterations based on split
            # if evaluate_all:
            #     num_accumulation_iters = (num_mesh_steps * self.C) // max_batch
            # else:
            #     num_accumulation_iters = (max_mesh_steps * self.C) // max_batch + 1
            #     num_residual_nodes = max_batch - (max_mesh_steps * self.C % max_batch)
            num_nodes_to_eval = max_mesh_steps * self.C 
        else:
            num_mesh_steps = max_mesh_steps + 1
            # Add another mesh step if number of saved split nodes is too small
            if (
                num_remaining_split_nodes
                < ((max_mesh_steps * self.C) % max_batch) - max_batch
            ):
                num_mesh_steps += 1

            # Get mesh step indices
            step_idxs = step_idxs[:num_mesh_steps]

            # Move the previously split mesh index to the front
            split_idx = torch.where(step_idxs == split_mesh_idx)[0]
            if len(split_idx):
                split_idx = split_idx[0]
                step_idxs[1 : split_idx + 1] = step_idxs[:split_idx].clone()
            else:
                step_idxs[1:] = step_idxs[:-1].clone()
            step_idxs[0] = split_mesh_idx

            # Place C quadrature points within each selected step
            nodes = self._compute_nodes(mesh[step_idxs], mesh[step_idxs + 1])

            # Flatten [N, C, T] -> [N*C, T] for batch evaluation, then reshape back
            nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=-2)
            nodes_flat = nodes_flat[len(split_nodes) :]
            num_eval_nodes = len(nodes_flat)

            # One batch of up to max_batch new evals per call. After
            # completing the carry-over panel (num_remaining_split_nodes
            # evals), the rest splits into full new panels plus a partial
            # residual; if num_eval_nodes < max_batch the layout's tail
            # already ends cleanly, so the residual is zero.
            # num_accumulation_iters = 1
            # actual_evals = min(max_batch, num_eval_nodes)
            # num_residual_nodes = (actual_evals - num_remaining_split_nodes) % self.C
            # evaluate_all = num_residual_nodes == 0

            num_nodes_to_eval = min(
                max_mesh_steps * self.C + num_remaining_split_nodes,
                num_eval_nodes,
            )

        # Evaluate the integrand over all batches, splitting each batch's
        # output into the integrand value and any tracked variables.
        f_evals = []
        # tracked_lists[k] is the list (one per accumulation batch) of the
        # k-th tracked variable; stays None when f emits no tracked variables.
        tracked_lists = None
        num_nodes_evaluated = 0
        while num_nodes_evaluated < num_nodes_to_eval:
            try:
                f_output = f(
                    nodes_flat[num_nodes_evaluated : num_nodes_evaluated + max_batch],
                    *f_args
                )
            except torch.OutOfMemoryError as e:  # Use RuntimeError for PyTorch < 2.0
                if max_batch == 1:
                    raise torch.OutOfMemoryError(
                        f"{e}\n\nSingle integrand (f) evaluation failed to fit in memory, batch size cannot be reduced."
                    )
                free_mem, total_mem = self._get_memory()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Small grace period for any async CUDA cleanup to land.
                    # Avoiding `torch.cuda.synchronize()` here on purpose: post-
                    # CUDA-OOM the default stream can be in an error state, and
                    # synchronize on a faulted stream has been observed to
                    # block indefinitely — matching the futex-wait-on-all-threads
                    # signature of the prior deadlock.
                    time.sleep(0.05)
                self._oom_max_batch = torch.int(
                    torch.round(0.75*self._oom_max_batch)
                )
                logger.warning(
                    f"Caught OOM with {free_mem} GB free of {total_mem} GB. "
                    + f"Reducing max_batch from {max_batch} to {self._oom_max_batch}."
                    + "If this warning appears often, consider reducing max_batch to "
                    + "avoid deleting and rerunning evaluations."
                )
                max_batch = self._oom_max_batch
                self.previous_max_batch = self._oom_max_batch
                continue
            
            integrand, tracked = self._split_f_output(f_output)
            self._check_f_output_finite(
                integrand,
                nodes_flat[num_nodes_evaluated : num_nodes_evaluated + max_batch],
            )
            f_evals.append(integrand)
            if tracked is not None:
                if tracked_lists is None:
                    tracked_lists = [[] for _ in tracked]
                for k, tv in enumerate(tracked):
                    tracked_lists[k].append(tv)
            
            num_nodes_evaluated += len(integrand)
            del integrand

        assert num_nodes_evaluated >= num_nodes_to_eval
        num_residual_nodes = num_nodes_evaluated - num_nodes_to_eval
        # Get the residual evaluations of the last mesh step
        if num_residual_nodes == 0:
            residual_f_evals = None
            residual_nodes = None
            residual_tracked = None
            residual_mesh_idx = None
        else:
            residual_nodes = nodes_flat[-self.C : -self.C + num_residual_nodes]
            nodes_flat = nodes_flat[: -self.C]
            residual_f_evals = f_evals[-1][-num_residual_nodes:]
            f_evals[-1] = f_evals[-1][:-num_residual_nodes]
            residual_tracked = None
            if tracked_lists is not None:
                residual_tracked = [
                    tl[-1][-num_residual_nodes:].detach() for tl in tracked_lists
                ]
                for tl in tracked_lists:
                    tl[-1] = tl[-1][:-num_residual_nodes]
            # Store the residual panel's barrier coordinate, not its
            # integer index — see the note at the top of this method.
            residual_mesh_idx = mesh[step_idxs[-1]].clone()
            step_idxs = step_idxs[:-1]

        # Combine split evaluations and nodes
        if split_mesh_idx is not None:
            nodes_flat = torch.concatenate([split_nodes, nodes_flat], dim=0)
            f_evals = torch.concatenate([split_f_evals, *f_evals], dim=0)
        else:
            f_evals = torch.concatenate(f_evals, dim=0)

        # Combine tracked variables the same way (prepending the carried split).
        tracked_combined = None
        if tracked_lists is not None:
            tracked_combined = []
            for k, tl in enumerate(tracked_lists):
                if split_mesh_idx is not None:
                    tracked_combined.append(
                        torch.concatenate([split_tracked[k], *tl], dim=0)
                    )
                else:
                    tracked_combined.append(torch.concatenate(tl, dim=0))

        # Reshape and combine outputs
        nodes = torch.reshape(nodes_flat, (-1, self.C, nodes_flat.shape[-1]))
        f_evals = torch.reshape(f_evals, (-1, self.C, f_evals.shape[-1]))
        if tracked_combined is not None:
            tracked_combined = [
                tv.reshape(-1, self.C, *tv.shape[1:]) for tv in tracked_combined
            ]

        # Path B may have laid out panels with the residual panel first
        # regardless of its position in time order — fine for the
        # split-prefix bookkeeping above, but the caller's
        # _adaptively_increase_mesh requires step_idxs sorted ascending
        # (its barrier-insertion offset scan walks left-to-right). Sort
        # step_idxs and permute the per-panel outputs to match.
        if split_mesh_idx is not None and len(step_idxs) > 1:
            sort_perm = torch.argsort(step_idxs)
            step_idxs = step_idxs[sort_perm]
            nodes = nodes[sort_perm]
            f_evals = f_evals[sort_perm]
            if tracked_combined is not None:
                tracked_combined = [tv[sort_perm] for tv in tracked_combined]

        tracked_out = (
            tuple(tv.detach() for tv in tracked_combined)
            if tracked_combined is not None
            else None
        )
        split_node_state = (
            residual_nodes,
            residual_f_evals,
            residual_tracked,
            residual_mesh_idx,
        )

        return nodes, f_evals, tracked_out, step_idxs, split_node_state

        # node_split_state = (split_nodes)
        # return torch.flatten(nodes, start_dim=0, end_dim=-2), step_idxs,
        # # Find barrier indices where mesh_trackers is True, take up to max_steps
        # step_idxs = torch.arange(len(mesh), device=self.device)
        # step_idxs = step_idxs[mesh_trackers]
        # step_idxs = step_idxs[:max_steps]
        # # Place C quadrature points within each selected step
        # nodes = self._compute_nodes(mesh[step_idxs], mesh[step_idxs + 1])

        # # --- Step 2: Evaluate the integrand at all quadrature points ---
        # # Flatten [N, C, T] -> [N*C, T] for batch evaluation, then reshape back
        # # shape = nodes.shape
        # return torch.flatten(nodes, start_dim=0, end_dim=-2), step_idxs, node_bucket

    def _sort_evals_into_mesh(
        self, take_gradient, nodes_flat, y_step_eval, y_step_bucket
    ):
        if take_gradient:
            return self._sort_evals_into_mesh_full_nodes(
                nodes_flat, y_step_eval, y_step_bucket
            )
        else:
            return self._sort_evals_into_mesh_full_nodes(
                nodes_flat, y_step_eval, y_step_bucket
            )

    def _sort_evals_into_mesh_full_nodes(self, nodes_flat, y_step_eval, y_step_bucket):
        assert len(nodes_flat) % self.C == 0
        N = len(nodes_flat) // self.C
        y_step_eval = torch.reshape(y_step_eval, (N, self.C, -1))
        nodes = torch.reshape(nodes_flat, (N, self.C, -1))
        return nodes, y_step_eval, y_step_bucket

    def _prune_excess_mesh(
        self, nodes, mesh_quadratures, mesh_quadrature_errors, error_ratios_2steps
    ):
        """
        Remove a single integration mesh step where
        error_ratios_2steps < remove_cut by merging two neighboring mesh steps,
        error_ratios_2steps corresponds to the first mesh step of the pair. This
        function only alters nodes, where remove_fxn merges the two mesh steps.

        Args:
            nodes (Tensor): Per-step quadrature point positions.
            error_ratios_2steps (Tensor): The merged errors of neighboring mesh
                steps, these indices align with the first step of the pair
                (error_ratios_2steps[i] -> nodes[i])

        Shapes:
            nodes: [N, C, T]
            error_ratios_2steps: [N-1]
        """

        if len(error_ratios_2steps) == 0:
            return nodes, mesh_quadratures, mesh_quadrature_errors
        # Since error ratios encompasses 2 RK steps each neighboring element shares
        # a step, we cannot remove that same step twice and therefore remove the
        # first in pair of steps that it appears in
        ratio_idxs_cut = torch.where(
            self._rec_remove(error_ratios_2steps < self.remove_cut)
        )[0]  # Index for first interval of 2
        assert not torch.any(ratio_idxs_cut[:-1] + 1 == ratio_idxs_cut[1:])

        if len(ratio_idxs_cut) == 0:
            return nodes, mesh_quadratures, mesh_quadrature_errors

        return self._merge_excess_nodes(
            nodes, mesh_quadratures, mesh_quadrature_errors, ratio_idxs_cut
        )

    def _get_optimal_mesh(
        self, record: dict[str, torch.Tensor], mesh: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute an optimized mesh from the converged integration results.

        After the integration loop converges, this method produces a refined
        mesh that can be reused for subsequent integrations of the same function.
        It does two things:
        1. Prunes: merges consecutive step pairs whose combined error ratio
           is below remove_cut (they were over-resolved).
        2. Adds: inserts midpoints for steps that still have high error ratios
           relative to the final integral value.

        This produces a mesh tailored to the difficulty of the integrand at
        different positions along the integration domain.

        Args:
            record: Dictionary of converged results including 't', 'mesh_quadratures',
                'mesh_quadrature_errors', and 'integral'.
            mesh: Current barrier positions. Shape: [M, T].

        Returns:
            Optimized barrier positions. Shape: [M_opt, T].
        """
        # Prune steps with excess accuracy (over-resolved regions)
        _, error_ratios_2steps, _ = self._compute_error_ratios(
            mesh_quadrature_errors=record["mesh_quadrature_errors"],
            mesh_quadratures=record["mesh_quadratures"],
            integral=record["integral"].detach(),
        )
        nodes_pruned, mesh_quadratures_pruned, mesh_quadrature_errors_pruned = (
            self._prune_excess_mesh(
                record["nodes"],
                record["mesh_quadratures"],
                record["mesh_quadrature_errors"],
                error_ratios_2steps,
            )
        )
        mesh_pruned = torch.concatenate(
            [nodes_pruned[:, 0, :], mesh[-1].unsqueeze(0)], dim=0
        )

        # Add new t steps using converged integral value
        error_ratios, error_ratios_2steps, error_ratios_per_dim = (
            self._compute_error_ratios(
                mesh_quadrature_errors=mesh_quadrature_errors_pruned,
                mesh_quadratures=mesh_quadratures_pruned,
                integral=record["integral"].detach(),
            )
        )
        keep_mask, remove_mask = self._accept_reject_masks(
            error_ratios, error_ratios_per_dim
        )
        adaptive_step = self._adaptively_increase_mesh(
            method_output=None,
            error_ratios=error_ratios,
            y_step_eval=None,
            nodes=None,
            mesh=mesh_pruned,
            mesh_idxs=torch.arange(len(mesh_pruned) - 1, device=self.device),
            mesh_trackers=torch.zeros(len(mesh_pruned), dtype=bool, device=self.device),
            keep_mask=keep_mask,
            remove_mask=remove_mask,
        )
        _, _, _, _, mesh_optimal, _, _, _ = adaptive_step

        return mesh_optimal

    def _rec_remove(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Ensure no two adjacent True values exist in a boolean mask.

        When merging step pairs, two adjacent steps cannot both be merged
        (they share a boundary). This recursively resolves conflicts by
        keeping the first flagged step in any adjacent pair and un-flagging
        the second.

        Example: [True, True, False, True] -> [True, False, False, True]

        Args:
            mask: Boolean mask where True = flagged for removal. Shape: [N].

        Returns:
            Modified mask with no adjacent True values. Shape: [N].
        """

        mask2 = mask[:-1] * mask[1:]
        if not torch.any(mask2):
            return mask

        # Must keep the first integration step
        if mask2[0]:
            mask[1] = False

        # Mask is too small to remove points
        if len(mask) <= 2:
            return mask

        return self._rec_remove(
            torch.concatenate(
                [mask[:2], mask2[1:] * mask[:-2] + (~mask2[1:]) * mask[2:]]
            )
        )

        # if torch.any(mask2):
        #     if mask2[0]:
        #         mask[1] = False
        #     if len(mask) > 2:
        #         return self._rec_remove(torch.concatenate(
        #             [
        #                 mask[:2],
        #                 mask2[1:]*mask[:-2] + (~mask2[1:])*mask[2:]
        #             ]
        #         ))
        #     else:
        #         return mask
        # else:
        #     return mask

    def _setup_integral_bounds(self, mesh, mesh_init, mesh_final):
        if mesh is not None:
            assert len(mesh.shape) == 2
            mesh = mesh.to(self.dtype).to(self.device)
            if mesh_init is not None:
                mesh_init = mesh_init.to(self.dtype).to(self.device)
                assert torch.allclose(
                    mesh[0], mesh_init, atol=self.atol_assert, rtol=self.rtol_assert
                )
            if mesh_final is not None:
                mesh_final = mesh_final.to(self.dtype).to(self.device)
                assert torch.allclose(
                    mesh[-1], mesh_final, atol=self.atol_assert, rtol=self.rtol_assert
                )
            mesh_init = mesh[0]
            mesh_final = mesh[-1]
            assert mesh_init < mesh_final, (
                "Integrator requires mesh_init < mesh_final, consider switching them and multiplying the integral by -1. Please also consider effects to your f."
            )
        else:
            mesh_init = self.init_mesh_init if mesh_init is None else mesh_init
            mesh_final = self.init_mesh_final if mesh_final is None else mesh_final
        mesh_init = mesh_init.to(self.dtype).to(self.device)
        mesh_final = mesh_final.to(self.dtype).to(self.device)

        assert mesh_init < mesh_final, (
            "Integrator requires mesh_init < mesh_final, consider switching them and multiplying the integral by -1. Please also consider the effects your loss function if one is provided."
        )
        return mesh_init, mesh_final

    def _setup_initial_mesh(
        self,
        mesh,
        mesh_init,
        mesh_final,
        reuse_mesh,
        same_integrand_fxn,
        random_initial_mesh,
        N_init_steps,
    ):
        if mesh is not None:
            # The user-provided mesh enters the loop here; _setup_integral_bounds
            # only moved a local copy for its assertions, so coerce it onto the
            # solver's device (DistributedEnvironment owns the device; inputs are
            # moved to it). Without this, the loop indexes a CPU mesh with
            # device-side indices and crashes on GPU.
            mesh = mesh.to(self.dtype).to(self.device)
            mesh_is_given = True
        elif reuse_mesh and self.mesh_previous is not None:
            mesh_is_given = False
            # Warn if the cached mesh was produced for a different integrand;
            # the user has opted into reuse but we should flag the mismatch.
            if not same_integrand_fxn:
                warnings.warn(
                    "reuse_mesh=True but f id differs from the cached "
                    "integrand; warm-started mesh may be poorly tuned for "
                    "this f.",
                    stacklevel=2,
                )
            # Filter cached barriers to within the new [mesh_init, mesh_final].
            # TODO: CHECK THIS PART WITH MULTI DIM T
            mask = (self.mesh_previous[:, 0] <= mesh_final[0]) & (
                self.mesh_previous[:, 0] >= mesh_init[0]
            )
            mesh = self.mesh_previous[mask]
            # Ensure the warm-started mesh starts at mesh_init.
            if len(mesh) == 0 or not torch.all(mesh[0] == mesh_init):
                mesh = torch.concatenate([mesh_init.unsqueeze(0), mesh], dim=0)
            # Ensure the warm-started mesh ends at mesh_final.
            # (Bug B1 fix: previously concatenated mesh_init here, producing a
            # non-monotone mesh whenever the cached endpoint did not match
            # the new mesh_final.)
            if not torch.all(mesh[-1] == mesh_final):
                mesh = torch.concatenate([mesh, mesh_final.unsqueeze(0)], dim=0)
        else:
            mesh_is_given = False
            if reuse_mesh:
                warnings.warn(
                    "reuse_mesh=True but no cached mesh is available "
                    "(first call, or after solver state was reset). "
                    "Falling back to a fresh initial mesh.",
                    stacklevel=2,
                )
            # Generate a fresh initial mesh of barriers across [mesh_init, mesh_final].
            # Layout: sqrt(N_init_steps) evenly-spaced top-level segments, each
            # subdivided into sqrt(N_init_steps)+1 sub-barriers. The total
            # barrier count is ~N_init_steps. Sub-barriers are placed
            # randomly (default) or uniformly within each segment.
            N_even_t = torch.sqrt(torch.tensor(N_init_steps, dtype=torch.float)).to(
                torch.int
            )
            dt = (mesh_final - mesh_init) / N_even_t
            mesh = (
                mesh_init
                + dt * torch.arange(N_even_t, device=self.device)[:, None, None]
            )  # TODO: this assumes the mesh is 1d

            n_sub = N_even_t + 1  # sub-barriers per segment
            if random_initial_mesh:
                # Random sub-barrier offsets within each segment, sorted.
                # Default. Random offsets break alignment between the
                # mesh and any uniformly-spaced features of the
                # integrand (e.g. zeros of a sinusoid, polynomial
                # extrema): on such integrands, a uniform mesh can
                # produce step errors the adaptive controller cannot
                # recover from. For deterministic reproducibility,
                # call ``torch.manual_seed`` before integrate().
                random_ts = dt * torch.rand((N_even_t, n_sub, 1), device=self.device)
                random_ts = torch.sort(random_ts, dim=1)[0]
                mesh = mesh + random_ts
            else:
                # Deterministic uniformly-spaced sub-barriers within
                # each segment. Available for debugging and for
                # integrands separately verified safe against uniform-
                # mesh aliasing; not the default because uniform meshes
                # fail to integrate certain test cases due to feature
                # alignment.
                #
                # Sub-barrier offsets are in [0, dt) — excluding dt to
                # avoid duplicating the top-level segment boundary
                # (segment k's last sub-barrier would otherwise
                # coincide with segment k+1's first sub-barrier and
                # the strict monotonicity assertion below would fail).
                offsets = (
                    dt
                    * torch.arange(n_sub, dtype=self.dtype, device=self.device)
                    / n_sub
                )
                mesh = mesh + offsets[None, :, None]
            # Enforce exact start and end points
            mesh[0] += mesh_init - mesh[0, 0]
            mesh[-1] += mesh_final - mesh[-1, -1]
            # Flatten segments into a single sorted barrier array
            mesh = torch.flatten(mesh, start_dim=0, end_dim=1)
            mesh[0] = mesh_init
            mesh[-1] = mesh_final
            assert torch.all(mesh[1:] - mesh[:-1] > 0)
        mesh_trackers = torch.ones(len(mesh), device=self.device).to(bool)
        mesh_trackers[-1] = False  # mesh_final cannot be a step starting point

        return mesh, mesh_trackers, mesh_is_given

    # -------------------------------------------------------------------------------- #
    #                           ADAPTIVE ERROR CALCULATIONS                            #
    # -------------------------------------------------------------------------------- #

    def _reduce_norm(self, x: torch.Tensor, error_norm=None) -> torch.Tensor:
        """
        Reduce the last (output dimension D) axis of ``x`` to one scalar per
        step using the configured ``error_norm``.

        Mirrors the ``norm`` argument of ``scipy.integrate.quad_vec``:

          - ``"2"``   → L2 norm ``sqrt(sum(x**2))``     (scipy ``np.linalg.norm``)
          - ``"max"`` → L-infinity norm ``max(|x|)``    (scipy ``_max_norm``)
          - ``"rms"`` → root-mean-square ``sqrt(mean(x**2))`` (padaquad legacy)
          - callable  → ``error_norm(x)`` reducing the last axis

        For 1D integrands every option equals ``torch.abs(x)``.

        Args:
            x: Per-dimension values. Shape: [..., D].
            error_norm: Override for the configured norm (defaults to
                ``self.error_norm``). ``"failure_fraction"`` is not a norm and
                must not be passed here.

        Returns:
            Reduced values, shape [...].
        """
        error_norm = self.error_norm if error_norm is None else error_norm
        if callable(error_norm):
            return error_norm(x)
        if error_norm == "2":
            return torch.sqrt(torch.sum(x**2, dim=-1))
        if error_norm == "max":
            return torch.amax(torch.abs(x), dim=-1)
        if error_norm == "rms":
            return torch.sqrt(torch.mean(x**2, dim=-1))
        raise ValueError(f"_reduce_norm cannot reduce with error_norm={error_norm!r}")

    def _round_floor(self, mesh_quadratures: torch.Tensor) -> torch.Tensor:
        """
        Machine-precision floor on the per-element error estimate.

        Mirrors ``scipy.integrate.quad_vec``'s ``round_err = norm(50*eps*h*s_k)``
        guard: it keeps the controller from splitting a panel forever chasing an
        error that is already at/below floating-point round-off. ``s_k`` is the
        panel's own integral contribution (``mesh_quadratures``).

        Args:
            mesh_quadratures: Per-step integral contributions. Shape: [N, D].

        Returns:
            Per-element rounding floor. Shape: [N, D].
        """
        eps = torch.finfo(self.dtype).eps
        return 50.0 * eps * torch.abs(mesh_quadratures)

    def _compute_error_ratios(
        self,
        mesh_quadrature_errors,
        mesh_quadratures=None,
        cum_mesh_quadratures=None,
        integral=None,
    ):
        """
        Compute per-step error ratios, dispatching on ``self.error_norm``.

        The error estimate is the difference between the order-p method and the
        embedded order-(p-1) method. How the D output dimensions are turned into
        a single accept/reject quantity depends on the scheme:

          - Norm family (``"2"``/``"max"``/``"rms"``/callable): scipy-style
            reduce-then-compare, in ``_error_ratios_norm``.
          - ``"failure_fraction"``: per-component comparison, in
            ``_error_ratios_failure_fraction``.

        The absolute-vs-cumulative-mode axis (``use_absolute_error_ratio``) is
        orthogonal and handled inside each family helper (it only changes the
        tolerance denominator).

        Args:
            mesh_quadrature_errors (Tensor): Per-step error estimates. [N, D].
            mesh_quadratures (Tensor): Per-step integral contributions (``s_k``),
                used for the rounding floor and (in cumulative mode) the
                cumulative sum. [N, D]. May be None.
            cum_mesh_quadratures (Tensor): Pre-computed cumulative sum [N, D]
                (cumulative mode only).
            integral (Tensor): Current total integral estimate [D]
                (absolute mode only).

        Returns:
            Tuple ``(error_ratios, error_ratios_2steps, error_ratios_per_dim)``:
                - error_ratios: per-step accept/reject quantity [N] (reduced
                  ratio for the norm family; failure fraction for
                  ``"failure_fraction"``).
                - error_ratios_2steps: per-pair merge/prune indicator [N-1].
                - error_ratios_per_dim: per-element ratio [N, D] for
                  ``"failure_fraction"`` (so non-finite panels can be detected),
                  else None.
        """
        abs_err = torch.abs(mesh_quadrature_errors)
        if (not callable(self.error_norm)) and self.error_norm == "failure_fraction":
            return self._error_ratios_failure_fraction(
                abs_err, mesh_quadratures, cum_mesh_quadratures, integral
            )
        return self._error_ratios_norm(
            abs_err, mesh_quadratures, cum_mesh_quadratures, integral
        )

    def _error_ratios_norm(
        self, abs_err, mesh_quadratures, cum_mesh_quadratures, integral
    ):
        """
        Norm family (reduce-then-compare), matching ``scipy.integrate.quad_vec``.

        The error vector is reduced to a scalar with ``self.error_norm`` and
        compared to a single tolerance built from the *same* norm of the
        integral::

            error_ratio[k] = R(|e[k]|) / (atol + rtol * R(I))      (absolute)
            error_ratio[k] = R(|e[k]|) / (atol + rtol * R(cum[k])) (cumulative)

        where ``R`` is the configured norm. A panel is accepted when
        ``error_ratio < 1`` (see ``_accept_reject_masks``).

        The rounding floor enters as a *lower bound on the tolerance*
        (``effective_tol = max(tol, R(round_floor))``), not as an inflation of
        the error. This realizes scipy's "stop once rounding-limited" behavior:
        a panel whose error is already at/below round-off has ``error_ratio <=
        ~1`` and is accepted instead of being split forever -- important when
        the requested tolerance is below what the working dtype can resolve.

        Shapes: abs_err [N, D]; integral [D]; cum [N, D]; returns [N], [N-1], None.
        """
        err = self._reduce_norm(abs_err)
        floor = (
            self._reduce_norm(self._round_floor(mesh_quadratures))
            if mesh_quadratures is not None
            else None
        )

        if self.use_absolute_error_ratio:
            error_tol = self.atol + self.rtol * self._reduce_norm(integral)
            error_tol_2steps = error_tol
        else:
            if cum_mesh_quadratures is not None:
                cum_steps = cum_mesh_quadratures
            elif mesh_quadratures is not None:
                cum_steps = torch.cumsum(mesh_quadratures, dim=0)
            else:
                raise ValueError("Must give mesh_quadratures or cum_mesh_quadratures")
            error_tol = self.atol + self.rtol * self._reduce_norm(torch.abs(cum_steps))
            error_tol_2steps = self.atol + self.rtol * torch.maximum(
                self._reduce_norm(torch.abs(cum_steps[:-1])),
                self._reduce_norm(torch.abs(cum_steps[1:])),
            )

        if floor is not None:
            # Broadcasts a scalar (absolute mode) or per-step (cumulative) tol.
            error_tol = torch.maximum(error_tol, floor)
            error_tol_2steps = torch.maximum(error_tol_2steps, floor[:-1] + floor[1:])

        error_ratio = err / error_tol
        err_2steps = self._reduce_norm(abs_err[:-1] + abs_err[1:])
        error_ratio_2steps = err_2steps / error_tol_2steps
        return error_ratio, error_ratio_2steps, None

    def _error_ratios_failure_fraction(
        self, abs_err, mesh_quadratures, cum_mesh_quadratures, integral
    ):
        """
        Per-component family: each output element keeps its own relative
        tolerance, and a panel's accept/reject quantity is the *fraction* of
        elements whose ratio exceeds 1::

            r[k, d] = |e[k, d]| / (atol + rtol * |I[d]|)        (absolute)
            r[k, d] = |e[k, d]| / (atol + rtol * |cum[k, d]|)   (cumulative)
            failure_fraction[k] = mean_d( r[k, d] >= 1 )

        A panel is accepted when ``failure_fraction <= mesh_failure_tolerance``
        (applied in ``_accept_reject_masks``). The raw per-element ratio is also
        returned so non-finite (NaN/Inf) panels can be detected downstream.

        The rounding floor enters per element as a lower bound on the tolerance
        (``effective_tol = max(tol, round_floor)``), so an element whose error is
        already at/below round-off does not count as a failure and the panel is
        not split chasing sub-precision accuracy.

        Shapes: abs_err [N, D]; integral [D]; cum [N, D];
        returns failure_fraction [N], error_ratio_2steps [N-1], ratio [N, D].
        """
        if self.use_absolute_error_ratio:
            error_tol = self.atol + self.rtol * torch.abs(integral)  # [D]
            error_tol_2steps = error_tol
        else:
            if cum_mesh_quadratures is not None:
                cum_steps = cum_mesh_quadratures
            elif mesh_quadratures is not None:
                cum_steps = torch.cumsum(mesh_quadratures, dim=0)
            else:
                raise ValueError("Must give mesh_quadratures or cum_mesh_quadratures")
            error_tol = self.atol + self.rtol * torch.abs(cum_steps)  # [N, D]
            error_tol_2steps = self.atol + self.rtol * torch.maximum(
                torch.abs(cum_steps[:-1]), torch.abs(cum_steps[1:])
            )

        if mesh_quadratures is not None:
            floor = self._round_floor(mesh_quadratures)  # [N, D]
            error_tol = torch.maximum(error_tol * torch.ones_like(floor), floor)
            error_tol_2steps = torch.maximum(
                error_tol_2steps * torch.ones_like(floor[:-1]),
                floor[:-1] + floor[1:],
            )

        ratio = abs_err / error_tol  # [N, D]
        finite = torch.isfinite(ratio)
        failed = (ratio >= 1.0) & finite
        failure_fraction = failed.to(self.dtype).mean(dim=-1)  # [N]

        # Merge/prune indicator: conservative max-norm of the combined 2-step
        # ratio (compared against remove_cut < 1, so a magnitude is needed here
        # rather than a fraction).
        ratio_2steps = (abs_err[:-1] + abs_err[1:]) / error_tol_2steps
        error_ratio_2steps = torch.amax(ratio_2steps, dim=-1)  # [N-1]
        return failure_fraction, error_ratio_2steps, ratio

    def _accept_reject_masks(self, error_ratios, error_ratios_per_dim):
        """
        Turn per-step error quantities into accept (keep) / reject (split) masks.

        Both families accept a panel whose error is non-finite (NaN/Inf): a
        midpoint split reuses the boundary node, so it can never reduce a
        non-finite value, and routing it to the split branch would loop forever.

          - ``"failure_fraction"``: keep when the failure fraction is
            ``<= mesh_failure_tolerance`` (plus a float-noise epsilon), or when
            any element of the panel is non-finite.
          - Norm family: keep when ``error_ratios < 1`` or ``error_ratios`` is
            non-finite.

        Args:
            error_ratios: Per-step accept/reject quantity. Shape: [N].
            error_ratios_per_dim: Per-element ratio [N, D] for the failure
                family (used for the non-finite check), else None.

        Returns:
            Tuple ``(keep_mask, remove_mask)``, each boolean shape [N].
        """
        if (not callable(self.error_norm)) and self.error_norm == "failure_fraction":
            panel_nonfinite = (~torch.isfinite(error_ratios_per_dim)).any(dim=-1)
            eps_frac = 8.0 * torch.finfo(self.dtype).eps
            keep_mask = (
                error_ratios <= self.mesh_failure_tolerance + eps_frac
            ) | panel_nonfinite
        else:
            keep_mask = (error_ratios < 1.0) | ~torch.isfinite(error_ratios)
        return keep_mask, ~keep_mask

    # -------------------------------------------------------------------------------- #
    #                                    RECORDING                                     #
    # -------------------------------------------------------------------------------- #

    # Record dict keys that are cumulative scalars (sum across batches),
    # not per-step arrays that need re-sorting.
    _RECORD_SCALAR_KEYS = ("integral", "integral_error", "loss")

    def _get_sorted_indices(
        self, record: torch.Tensor, result: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute indices for merging new results into an existing sorted record.

        Uses binary search (searchsorted) to find where new results should be
        inserted, then computes the indices for both the existing and new entries
        in the merged array.

        Args:
            record: Sorted 1D tensor of existing values (e.g. start node of
                recorded steps). Shape: [N_record].
            result: New values to insert. Shape: [N_result].

        Returns:
            Tuple of (idxs_keep, idxs_input):
                - idxs_keep: Where existing record entries go in the merged array.
                - idxs_input: Where new result entries go in the merged array.
        """
        idxs_sorted = torch.searchsorted(record, result)
        idxs_input = idxs_sorted + torch.arange(len(result), device=self.device)
        idxs_keep = torch.arange(len(result) + len(record), device=self.device)
        keep_mask = torch.ones(len(idxs_keep), device=self.device).to(bool)
        keep_mask[idxs_input] = False
        idxs_keep = idxs_keep[keep_mask]
        return idxs_keep, idxs_input

    def _insert_sorted_results(
        self,
        record: torch.Tensor,
        record_idxs: torch.Tensor,
        result: torch.Tensor,
        result_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge new results into an existing record at pre-computed sorted positions.

        Creates a new tensor large enough for both, places existing entries at
        record_idxs and new entries at result_idxs.

        Args:
            record: Existing recorded data. Shape: [N_record, ...].
            record_idxs: Positions for existing data in merged array. Shape: [N_record].
            result: New data to insert. Shape: [N_result, ...].
            result_idxs: Positions for new data in merged array. Shape: [N_result].

        Returns:
            Merged tensor with both record and result entries in sorted order.
            Shape: [N_record + N_result, ...].
        """
        add_shape = (len(record) + len(result), *record.shape[1:])
        old_record = record.clone()
        # Allocate with the input tensor's own dtype/device so this works for
        # non-float records (e.g. integer/bool tracked variables), not just
        # self.dtype. Every position is overwritten below, so zeros is safe.
        record = torch.zeros(add_shape, device=record.device, dtype=old_record.dtype)
        record[record_idxs] = old_record
        record[result_idxs] = result
        return record

    def _record_results(
        self,
        record: dict[str, torch.Tensor],
        take_gradient: bool,
        results: IntegrationResult,
    ) -> dict[str, torch.Tensor]:
        """
        Add a batch of accepted step results to the running record.

        On the first batch, initializes the record dict. On subsequent batches,
        inserts new results in sorted order and accumulates the integral
        and loss. When take_gradient is True, detaches results to prevent
        the computation graph from growing across batches.

        Args:
            record: Running record dict. Empty dict {} on the first call.
            take_gradient: Whether gradients are being computed. If True,
                detaches tensors before storing to keep graph manageable.
            results: IntegrationResult from the current accepted batch.

        Returns:
            Updated record dict with the new results merged in. Dict keys
            match IntegrationResult field names so getattr-based merge
            below can iterate without translation.
        """
        if len(record) == 0 and not take_gradient:
            record["integral"] = results.integral
            record["nodes"] = results.nodes
            record["h"] = results.h
            record["y"] = results.y
            record["mesh_quadratures"] = results.mesh_quadratures
            record["mesh_quadrature_errors"] = results.mesh_quadrature_errors
            record["integral_error"] = results.integral_error
            record["error_ratios"] = results.error_ratios
            record["loss"] = results.loss
            if results.tracked_variables is not None:
                # Tracked variables are already detached at evaluation time.
                record["tracked_variables"] = list(results.tracked_variables)
            return record
        elif len(record) == 0 and take_gradient:
            record["integral"] = results.integral.detach()
            record["nodes"] = results.nodes.detach()
            record["h"] = results.h.detach()
            record["y"] = results.y.detach()
            record["mesh_quadratures"] = results.mesh_quadratures.detach()
            record["mesh_quadrature_errors"] = results.mesh_quadrature_errors.detach()
            record["integral_error"] = results.integral_error.detach()
            record["error_ratios"] = results.error_ratios.detach()
            record["loss"] = results.loss.detach()
            if results.tracked_variables is not None:
                # Tracked variables are already detached at evaluation time.
                record["tracked_variables"] = list(results.tracked_variables)
            return record

        idxs_keep, idxs_input = self._get_sorted_indices(
            record["nodes"][:, 0, 0].detach(), results.nodes[:, 0, 0].detach()
        )
        for key, value in record.items():
            if key in self._RECORD_SCALAR_KEYS:
                record[key] = value + getattr(results, key).detach()
            elif key == "tracked_variables":
                # A list of per-variable tensors: insert each independently.
                record[key] = [
                    self._insert_sorted_results(v, idxs_keep, r, idxs_input)
                    for v, r in zip(value, results.tracked_variables, strict=True)
                ]
            else:
                record[key] = self._insert_sorted_results(
                    value, idxs_keep, getattr(results, key), idxs_input
                )
        assert torch.all(record["nodes"][1:, 0, 0] - record["nodes"][:-1, 0, 0] > 0)

        return record

    def _sort_record(self, record: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Sort all per-step entries in the record by ascending order.

        The integration loop may process batches in any order, so the record
        needs to be sorted before final output. Scalar values (integral, loss)
        are not reordered since they are cumulative sums.

        Args:
            record: Record dict with per-step tensors.

        Returns:
            Record with per-step tensors sorted by start node of each step.
        """
        sorted_idxs = torch.argsort(record["nodes"][:, 0, 0], dim=0)
        for key, value in record.items():
            if key in self._RECORD_SCALAR_KEYS:
                continue
            if key == "tracked_variables":
                record[key] = [tv[sorted_idxs] for tv in value]
            else:
                record[key] = value[sorted_idxs]
        all_ascending = torch.all(
            record["nodes"][1:, 0, 0] - record["nodes"][:-1, 0, 0] > 0
        )
        all_descending = torch.all(
            record["nodes"][1:, 0, 0] - record["nodes"][:-1, 0, 0] < 0
        )
        assert all_ascending or all_descending, (
            "Nodes are required to be either in ascending or descending order"
        )
        return record

    # -------------------------------------------------------------------------------- #
    #                                MEMORY MANAGEMENT                                 #
    # -------------------------------------------------------------------------------- #

    def _get_cpu_memory(self) -> tuple[float, float]:
        """
        Query available and total CPU (system) memory in GB.

        Returns:
            Tuple of (free_gb, total_gb).
        """
        mem = psutil.virtual_memory()
        free = mem.available / 1024**3
        total = mem.total / 1024**3
        return free, total

    def _get_cuda_memory(self) -> tuple[float, float]:
        """
        Query available and total CUDA (GPU) memory in GB.

        Accounts for both free system GPU memory and unused PyTorch cache
        memory (reserved but not allocated). This gives a more accurate
        picture of how much memory is truly available for new allocations.

        Returns:
            Tuple of (free_gb, total_gb).
        """
        mem_info = torch.cuda.mem_get_info(self.device)
        # Total memory on the GPU
        total_gpu = mem_info[1] / 1024**3
        # Memory that is free outside of the PyTorch cache
        free_gpu = mem_info[0] / 1024**3
        # Memory reserved for the PyTorch cache
        torch_cache = torch.cuda.memory_reserved(self.device) / 1024**3
        # Cache memory being used by tensors
        torch_cache_used = torch.cuda.memory_allocated(self.device) / 1024**3
        # Total free amount of memory that can be used
        total_free = free_gpu + (torch_cache - torch_cache_used)

        return total_free, total_gpu

    def _get_memory(self) -> tuple[float, float]:
        """
        Query available and total memory in GB for the active device type.

        Dispatches to _get_cuda_memory() or _get_cpu_memory() based on
        self.device_type.

        Returns:
            Tuple of (free_gb, total_gb).
        """
        if self.device_type == "cuda":
            return self._get_cuda_memory()
        else:
            return self._get_cpu_memory()

    def _setup_memory_checks(
        self,
        f: Callable,
        node_test: torch.Tensor,
        take_gradient: bool,
        f_args: tuple = (),
    ) -> None:
        """
        Benchmark the integrand's memory footprint to determine batch sizes.

        Runs the integrand with increasing batch sizes (10, 100, 1000, ...)
        and measures the memory consumed per evaluation. This per-evaluation
        memory cost (f_unit_mem_size) is then used throughout integration
        to dynamically compute how many steps can fit in one batch.

        When take_gradient=True, a 2.1x safety factor is applied to the measured
        memory to account for intermediate allocations during integration (RK
        computation, error estimation, etc.).

        Args:
            f: The integrand function to benchmark.
            node_test: A sample node point for benchmarking. Shape: [T] or [1, T].
            f_args: Extra arguments passed to f.
        """
        assert len(node_test.shape) <= 2
        if len(node_test.shape) == 2:
            node_test = node_test[0]
        node_test = node_test.unsqueeze(0)
        self.f_unit_mem_size = None

        N = 1
        max_evals = 2 * N
        eval_time = 0
        mem_scale = 2.1 if take_gradient else 1.0
        while eval_time < 0.1 and N < 1e9 and max_evals > N:
            t0 = time.time()
            t_input = torch.tile(node_test, (N, 1))
            mem_before = self._get_memory()
            if (
                self.f_unit_mem_size is not None
                and self.f_unit_mem_size * N > mem_before[0]
            ):
                return
            
            # Catch OOM errors
            try:
                result = f(t_input, *f_args)
            except torch.OutOfMemoryError as e:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Small grace period for any async CUDA cleanup to land.
                    # Avoiding `torch.cuda.synchronize()` here on purpose: post-
                    # CUDA-OOM the default stream can be in an error state, and
                    # synchronize on a faulted stream has been observed to
                    # block indefinitely — matching the futex-wait-on-all-threads
                    # signature of the prior deadlock.
                    time.sleep(0.05)
                return
            
            mem_after = self._get_memory()
            del result
            self.f_unit_mem_size = mem_scale * max(
                0, (mem_before[0] - mem_after[0]) / float(N)
            )
            eval_time = time.time() - t0
            N = 10 * N
            max_evals = self._get_max_f_evals(0.8)
        logger.debug("Ending unit memory search")

    def _get_usable_memory(self, total_mem_usage: float) -> float:
        """
        Compute how much memory (in GB) can be used for integrand evaluations.

        Reserves a buffer of (1 - total_mem_usage) * total_memory to avoid
        out-of-memory errors from other system/PyTorch allocations.

        Args:
            total_mem_usage: Fraction of total memory allowed (0 < value <= 1).

        Returns:
            Usable memory in GB (non-negative).
        """
        free, total = self._get_memory()
        buffer = (1 - total_mem_usage) * total
        return max(0, free - buffer)

    def _get_max_f_evals(self, total_mem_usage: float) -> int:
        """
        Compute the maximum number of integrand evaluations that fit in memory.

        Divides usable memory by the per-evaluation memory cost (measured by
        _setup_memory_checks). A small epsilon (1e-12) prevents division by zero.

        Args:
            total_mem_usage: Fraction of total memory allowed.

        Returns:
            Maximum number of evaluations (integer).
        """
        usable = self._get_usable_memory(total_mem_usage)
        return int(usable // (1e-12 + self.f_unit_mem_size))
