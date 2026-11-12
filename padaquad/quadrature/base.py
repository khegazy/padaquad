"""
Adaptive quadrature engine that evaluates many subintervals in parallel.

An ODE solver must advance one step at a time, because each step depends on
the previous result. Here the integrand ``f(t)`` depends only on the
integration variable ``t``, so the contribution of every subinterval is
independent of the others, and many subintervals are evaluated together in
one batch.

The mesh divides ``[mesh_init, mesh_final]`` into subintervals, and the
method places C nodes in each one. A subinterval whose error estimate exceeds
the tolerance is split in half and both halves are evaluated again. After the
integral has converged, neighboring subintervals with very small errors are
merged to build the optimal mesh, which a later call can start from.

Key concepts:

- **mesh**: the mesh points, shape [M, T]. Subinterval i runs from mesh
  point i to mesh point i + 1.
- **mesh_trackers**: boolean array of length M. True means the subinterval
  that starts at that mesh point still has to be evaluated (or evaluated
  again after a split).
- **Batching**: when the subintervals do not fit in memory at once, they are
  evaluated in batches. The batch size comes from a measurement of the
  memory one evaluation of ``f`` uses (see ``memory.py``).

Classes:

- ``AdaptiveQuadrature`` (here): the abstract base class with the
  integration loop, the splitting and merging of subintervals, and the error
  ratios. Memory management (the memory benchmark, the batch size and the
  recognition of out-of-memory errors) comes from a mixin in ``memory.py``.
- ``UniformAdaptiveQuadrature`` (``uniform.py``): the solver for methods whose
  nodes sit at fixed fractions of each subinterval.
- ``VariableAdaptiveQuadrature`` (``variable.py``): the solver for methods
  whose weights are computed from where the nodes actually are.

The quadrature method is a ``MethodClass`` (``padaquad/methods``); the engine
uses it only through ``node_fractions()`` and ``weights()``.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
import numbers
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from padaquad.base import DEFAULT_METHOD, SolverBase, steps
from padaquad.distributed import DistributedEnvironment, _resolve_device
from padaquad.methods import _get_method
from padaquad.quadrature.memory import _is_out_of_memory, _MemoryManagement
from padaquad.results import IntegrationResult, MethodOutput

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from padaquad.methods import MethodClass

#: State carried across batches by ``conserve_memory=True`` when one step's
#: evaluation is split between batches: (nodes, f evaluations, tracked
#: variables) of its evaluated part, and the step's left-barrier coordinate
#: (shape [T], not an index, so it survives mesh refinement). All None when
#: nothing is carried.
_SplitState = tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    list[torch.Tensor] | None,
    torch.Tensor | None,
]


def _RK_integral(
    t: torch.Tensor,
    y: torch.Tensor,
    tableau_b: torch.Tensor,
    y0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the method's weights to a batch of subintervals and sum the results.

    For every subinterval i, the integrand values are multiplied by the
    weights, summed over the C nodes and multiplied by the width ``h_i``:
    ``R_i = h_i * sum_j(b_ij * y_ij)``, and ``integral = y0 + sum_i(R_i)``.
    The function is used for every method, not only the Runge-Kutta ones.

    Parameters
    ----------
    t : torch.Tensor
        Nodes of each subinterval. The width is ``t[:, -1] - t[:, 0]``,
        because every method's first and last node are the subinterval's
        endpoints. Shape: [N, C, T].
    y : torch.Tensor
        Integrand values at the nodes. Shape: [N, C, D].
    tableau_b : torch.Tensor
        Weights of each node, from ``MethodClass.weights``. Shape: [N, C, 1].
    y0 : torch.Tensor
        Value added to the total. Shape: [D].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple ``(integral, RK_steps, h)``: the total ``y0`` plus all
        contributions [D], the contribution of each subinterval [N, D], and
        the width of each subinterval [N, T].
    """
    # Step size: difference between last and first time point in each step
    h = t[:, -1] - t[:, 0]
    logger.debug("H shape=%s values=%s", h.shape, h)

    # Weighted sum of integrand evaluations within each step, scaled by step size.
    # tableau_b*y: weight each evaluation, dim=1 sums over C quadrature points
    RK_steps = h * torch.sum(tableau_b * y, dim=1)
    logger.debug("RK_steps shape=%s values=%s", RK_steps.shape, RK_steps)
    # Sum all step contributions to get the total integral
    integral = y0 + torch.sum(RK_steps, dim=0)
    return integral, RK_steps, h


# Multi-GPU keyword options the solver forwards to ``DistributedEnvironment``,
# read from its signature so the names and defaults live only there.
_DISTRIBUTED_OPTIONS = frozenset(
    inspect.signature(DistributedEnvironment.__init__).parameters
) - {"self", "device"}


class AdaptiveQuadrature(_MemoryManagement, SolverBase):
    """
    Base class of the adaptive solvers, which integrate f in parallel batches.

    The integration loop:

    1. Builds a starting mesh on ``[mesh_init, mesh_final]``.
    2. Evaluates the integrand at the nodes of many subintervals at once.
    3. Computes each subinterval's contribution and error estimate with the
       method's weights.
    4. Accepts the subintervals whose error ratio is below 1 and splits the
       others in half.
    5. Repeats steps 2 to 4 until every subinterval is accepted.
    6. Builds the optimal mesh, which a later call with ``same_f=True`` starts
       from.

    Subclasses set ``sampling_type`` and define how a split subinterval gets
    its nodes and how two neighboring subintervals are merged.

    Attributes
    ----------
    method : MethodClass
        The quadrature method.
    C : int
        Number of nodes per subinterval.
    remove_cut : float
        Largest predicted error ratio of a subinterval made by merging two
        neighbors when the optimal mesh is built.
    init_max_batch : int or None
        The constructor's ``max_batch``.
    init_total_mem_usage : float
        The constructor's ``total_mem_usage``.
    use_absolute_error_ratio : bool
        True to judge the relative tolerance against the whole integral,
        False to judge it against the running integral.
    error_norm : str or Callable
        The ``error_norm`` in use: the one given to the current or last
        ``integrate`` call, or else the constructor's.
    mesh_failure_tolerance : float
        The ``mesh_failure_tolerance`` in use, chosen the same way.
    """

    #: Sampling mode of the concrete solver; selects the method registry.
    sampling_type: ClassVar[steps]
    # Set by ``_set_solver_dtype``, which ``SolverBase.__init__`` reaches
    # through ``_set_dtype``.
    method: MethodClass
    C: int

    def __init__(
        self,
        *,
        f: Callable[..., Any] | None = None,
        method: str = DEFAULT_METHOD,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
        remove_cut: float = 0.1,
        max_batch: int | None = None,
        total_mem_usage: float = 0.9,
        max_adaptive_splits: int | None = None,
        use_absolute_error_ratio: bool = True,
        error_norm: str | Callable[[torch.Tensor], torch.Tensor] = "2",
        mesh_failure_tolerance: float = 0.0,
        error_on_nonfinite: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device | None = None,
        output_speed_info: bool = False,
        **distributed_options: Any,
    ) -> None:
        """
        Create a solver and store the defaults that ``integrate`` falls back on.

        Every argument is keyword-only.

        Parameters
        ----------
        f : Callable or None
            The integrand, used by ``integrate`` calls that pass none. It takes
            points of shape [N, T] and returns values of shape [N, D]
            (:ref:`ug-shapes`). Default None.
        method : str
            Name of the quadrature method, a key of ``UNIFORM_METHODS`` (uniform
            solver) or ``VARIABLE_METHODS`` (variable solver); case does not
            matter. Default ``"gk21"``.
        atol : float
            Absolute tolerance. A subinterval is accepted when its error
            estimate is below ``max(atol, rtol * |I|)``, with ``I`` the integral
            (:ref:`ug-error-control`). Default 1e-5.
        rtol : float
            Relative tolerance; see ``atol``. Default 1e-5.
        mesh_init : torch.Tensor or None
            Default lower integration bound, shape [T]. None means 0.
        mesh_final : torch.Tensor or None
            Default upper integration bound, shape [T]. None means 1.
        y0 : torch.Tensor or None
            Default value added to the returned integral, shape [D]. None means 0.
        remove_cut : float
            Largest predicted error ratio of a subinterval made by merging two
            neighbors when the optimal mesh is built; must be below 1. Lower
            values merge less. Default 0.1 (:ref:`ug-optimal-mesh`).
        max_batch : int or None
            Default for the largest number of integrand evaluations per call of
            ``f``. None (default) sets the batch size from the free memory,
            measured by a memory benchmark (:ref:`ug-memory`).
        total_mem_usage : float
            Fraction of the device's total memory, in (0, 1], that batches may
            fill: the batch size leaves ``(1 - total_mem_usage)`` of it free.
            Used only when no ``max_batch`` is set. Default 0.9.
        max_adaptive_splits : int or None
            Default cap on how many times a subinterval may have been halved
            since the starting mesh; at the cap it is accepted even if it fails
            the tolerance. None (default) means no cap (:ref:`ug-convergence`).
        use_absolute_error_ratio : bool
            If True (default), the relative tolerance uses the whole integral;
            if False, the running integral from ``mesh_init`` through each
            subinterval (:ref:`ug-error-reference`).
        error_norm : str or Callable
            How the error of a vector-valued integrand becomes one accept-or-split
            decision: ``"2"`` (default), ``"max"``, ``"rms"``,
            ``"failure_fraction"`` or a callable norm (:ref:`ug-error-norm`).
        mesh_failure_tolerance : float
            With ``error_norm="failure_fraction"``, the fraction of output
            components, in [0, 1], that may fail while a subinterval is still
            accepted. Default 0.0 (every component must pass).
        error_on_nonfinite : bool
            If True (default), raise ``ValueError`` naming the points where
            ``f`` returned NaN or Inf. If False, accept those subintervals and
            let the non-finite value reach the result.
        dtype : torch.dtype
            ``torch.float64`` (default) or ``torch.float32``; ``torch.float16``
            is refused. ``integrate`` switches to the dtype of the mesh or the
            bounds it is given.
        device : str, torch.device or None
            ``"cpu"``, ``"cuda"``, ``"cuda:N"`` or a ``torch.device``. None
            (default) picks CUDA if it is available, else the CPU.
        output_speed_info : bool
            If True, log the time of each stage of the loop to the file
            ``padaquad_speed.log``. Default False.
        **distributed_options : Any
            Options for multi-process (multi-GPU) runs only.

        Raises
        ------
        TypeError
            If an argument is passed by position, or a keyword is not an option
            of the solver.
        ValueError
            If ``remove_cut`` is not below 1, or another option is invalid.
        """
        unknown = set(distributed_options) - _DISTRIBUTED_OPTIONS
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument(s) "
                f"{sorted(unknown)}"
            )
        if not remove_cut < 1.0:
            raise ValueError(f"remove_cut must be below 1, got {remove_cut!r}")
        super().__init__(
            method=method,
            atol=atol,
            rtol=rtol,
            y0=y0,
            f=f,
            mesh_init=mesh_init,
            mesh_final=mesh_final,
            output_speed_info=output_speed_info,
            dtype=dtype,
            device=device,
            **distributed_options,
        )
        self._check_max_adaptive_splits(max_adaptive_splits)
        self.remove_cut = remove_cut

        # Memory handling variables
        # max_batch of the last integrate() call that returned; reused by
        # same_f=True.
        self.previous_max_batch: int | None = None
        # Per-evaluation memory cost; set by each _setup_memory_checks run and
        # used by get_max_batch to size batches from live free memory.
        self.f_unit_mem_size = None
        # Cap discovered after an OOM during a take_gradient=False evaluation;
        # None until one occurs. Cleared at the start of every integrate() call
        # (same_f=True carries it over through previous_max_batch).
        self._oom_max_batch = None

        # Construction-time defaults; each integrate() call falls back to these
        # when its corresponding argument is None.
        self.init_max_batch = max_batch
        self.init_max_adaptive_splits = max_adaptive_splits
        self.use_absolute_error_ratio = use_absolute_error_ratio

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
    def _check_error_norm(cls, error_norm: object) -> None:
        """
        Validate the ``error_norm`` selector (a known string or a callable).

        Parameters
        ----------
        error_norm : object
            The value to check.

        Raises
        ------
        ValueError
            If ``error_norm`` is neither callable nor in ``_VALID_ERROR_NORMS``.
        """
        if callable(error_norm):
            return
        if error_norm not in cls._VALID_ERROR_NORMS:
            raise ValueError(
                f"error_norm must be a callable or one of "
                f"{cls._VALID_ERROR_NORMS}, got {error_norm!r}"
            )

    @staticmethod
    def _check_mesh_failure_tolerance(mesh_failure_tolerance: float) -> None:
        """
        Validate that ``mesh_failure_tolerance`` is a fraction in [0, 1].

        Parameters
        ----------
        mesh_failure_tolerance : float
            The value to check.

        Raises
        ------
        ValueError
            If the value is outside [0, 1].
        """
        if not (0.0 <= mesh_failure_tolerance <= 1.0):
            raise ValueError(
                "mesh_failure_tolerance must be in [0, 1], got "
                f"{mesh_failure_tolerance!r}"
            )

    @staticmethod
    def _check_max_adaptive_splits(max_adaptive_splits: object) -> None:
        """
        Validate that ``max_adaptive_splits`` is None or a non-negative integer.

        The type check also catches a float such as 0.1 meant for the removed
        ``max_path_change``, which would otherwise act as a cap.

        Parameters
        ----------
        max_adaptive_splits : object
            The value to check.

        Raises
        ------
        TypeError
            If the value is not None and not an integer
            (``numbers.Integral``, so numpy integers pass; bool rejected).
        ValueError
            If the value is negative.
        """
        if max_adaptive_splits is None:
            return
        if isinstance(max_adaptive_splits, bool) or not isinstance(
            max_adaptive_splits, numbers.Integral
        ):
            raise TypeError(
                "max_adaptive_splits must be None or an integer, got "
                f"{type(max_adaptive_splits).__name__} {max_adaptive_splits!r}"
            )
        if max_adaptive_splits < 0:
            raise ValueError(
                f"max_adaptive_splits must be >= 0, got {max_adaptive_splits}"
            )

    # -------------------------------------------------------------------------------- #
    #                        OPERATIONS ON THE QUADRATURE RULE                         #
    # -------------------------------------------------------------------------------- #

    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """
        Build the quadrature method at ``dtype`` on the solver's device.

        Called from ``SolverBase._set_dtype``, which runs at the end of
        ``SolverBase.__init__`` (the method's first build) and on every later
        dtype change. ``_get_method`` clones the named method from the registry of this
        solver's ``sampling_type``.

        Parameters
        ----------
        dtype : torch.dtype
            Floating-point dtype for the method's tensors.

        Raises
        ------
        ValueError
            If ``method_name`` is not in that registry.
        """
        self.method = _get_method(
            self.sampling_type, self.method_name, self.device, dtype
        )
        self.C = self.method.n_nodes

    def _compute_nodes(
        self, mesh_left: torch.Tensor, mesh_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Place the C nodes of each subinterval between its two mesh points.

        Node i of a subinterval is ``mesh_left + c_i * (mesh_right - mesh_left)``,
        with c the method's ``node_fractions()``: the tableau fractions for a
        uniform method, the method's own ``c`` for a ``VariableMethodClass``
        method (evenly spaced for ``interpolatory3_variable``). For example,
        with c = [0, 0.5, 1] the nodes are the start, the midpoint and the end.

        Parameters
        ----------
        mesh_left : torch.Tensor
            Left mesh point of each subinterval. Shape: [N, T].
        mesh_right : torch.Tensor
            Right mesh point of each subinterval. Shape: [N, T].

        Returns
        -------
        torch.Tensor
            Node positions. Shape: [N, C, T].
        """
        dt = (mesh_right - mesh_left).unsqueeze(1)
        return mesh_left.unsqueeze(1) + self.method.node_fractions().unsqueeze(-1) * dt

    def _calculate_integral(
        self, t: torch.Tensor, y: torch.Tensor, y0: torch.Tensor
    ) -> MethodOutput:
        """
        Compute the integral and the error estimate for a batch of subintervals.

        Runs the weighted sum twice: with the method's weights b for the
        integral, and with b_error for the error estimate (b_error is the
        difference between the weights of the method and of its embedded
        lower-degree method). The error estimates are detached from the
        autograd graph: they only drive the accept-or-split decisions, so no
        gradient flows through them and the graph they would hold is not kept.

        Parameters
        ----------
        t : torch.Tensor
            Nodes of each subinterval. Shape: [N, C, T].
        y : torch.Tensor
            Integrand values at the nodes. Shape: [N, C, D].
        y0 : torch.Tensor
            Value added to the integral. Shape: [D].

        Returns
        -------
        MethodOutput
            The integral, its error estimate, and the contribution, error
            estimate and width of each subinterval.
        """
        b, b_error = self.method.weights(t)
        integral, mesh_quadratures, h = _RK_integral(t, y, b, y0=y0)
        integral_error, mesh_quadrature_errors, _ = _RK_integral(t, y, b_error, y0=y0)
        return MethodOutput(
            integral=integral,
            integral_error=integral_error.detach(),
            mesh_quadratures=mesh_quadratures,
            mesh_quadrature_errors=mesh_quadrature_errors.detach(),
            h=h,
        )

    # -------------------------------------------------------------------------------- #
    #                                 ABSTRACT METHODS                                 #
    # -------------------------------------------------------------------------------- #

    @abstractmethod
    def _evaluate_adaptive_nodes(
        self,
        f: Callable[..., Any],
        idxs_add: torch.Tensor,
        y: torch.Tensor,
        nodes: torch.Tensor,
        f_args: tuple[Any, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the integrand at the nodes of the halves of split subintervals.

        A subinterval that fails the tolerance is split into two halves. This
        method evaluates the integrand at the nodes the halves need. Uniform
        and variable sampling do this differently.

        Parameters
        ----------
        f : Callable[..., Any]
            The integrand.
        idxs_add : torch.Tensor
            Indices of the subintervals to split. Shape: [n_add].
        y : torch.Tensor
            Integrand values of all subintervals. Shape: [N, C, D].
        nodes : torch.Tensor
            Nodes of all subintervals. Shape: [N, C, T].
        f_args : tuple[Any, ...]
            Extra arguments passed to f.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple ``(y_new, nodes_new)``: the integrand values and the nodes of
            the halves that replace the split subintervals.
        """

    @abstractmethod
    def _merge_excess_nodes(
        self,
        nodes: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        mesh_quadrature_errors: torch.Tensor,
        remove_idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge each pair of neighboring subintervals into one subinterval.

        Pair i is subintervals ``remove_idxs[i]`` and ``remove_idxs[i] + 1``.

        Parameters
        ----------
        nodes : torch.Tensor
            Nodes of all subintervals. Shape: [N, C, T].
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh_quadrature_errors : torch.Tensor
            Error estimate of each subinterval. Shape: [N, D].
        remove_idxs : torch.Tensor
            Index of the first subinterval of each pair to merge. Shape: [R].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple ``(nodes, mesh_quadratures, mesh_quadrature_errors)`` with
            each merged pair replaced by one subinterval (N - R in all); a
            merged subinterval's contribution and error are the pair's sums.
        """

    # -------------------------------------------------------------------------------- #
    #                            PRIMARY INTEGRATION METHOD                            #
    # -------------------------------------------------------------------------------- #

    def integrate(
        self,
        f: Callable[..., Any] | None = None,
        y0: torch.Tensor | None = None,
        mesh: torch.Tensor | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        same_f: bool = False,
        random_initial_mesh: bool = True,
        N_init_steps: int = 13,
        f_args: tuple[Any, ...] = (),
        take_gradient: bool = True,
        total_mem_usage: float | None = None,
        loss_fxn: Callable[[IntegrationResult], torch.Tensor] | None = None,
        max_batch: int | None = None,
        max_adaptive_splits: int | None = None,
        error_norm: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
        error_integral_reference: float | torch.Tensor | None = None,
        mesh_failure_tolerance: float | None = None,
        error_on_nonfinite: bool | None = None,
        conserve_memory: bool = False,
        result_device: str | torch.device | None = "cpu",
    ) -> IntegrationResult:
        """
        Integrate f over [mesh_init, mesh_final] with adaptive subintervals.

        The integrand is called as ``f(t, *f_args)`` with points ``t`` of shape
        [N, T] and must return values of shape [N, D], or a 2-tuple of those
        values and the tracked variables (:ref:`ug-shapes`,
        :ref:`ug-tracked-variables`). The loop evaluates many subintervals in
        one batch, accepts those whose error ratio is below 1 and splits the
        others in half, until every subinterval is accepted. It then builds
        the optimal mesh, which a later call with ``same_f=True`` starts from.

        Every argument left as None falls back to the constructor's value.

        Parameters
        ----------
        f : Callable or None
            The integrand. None uses the integrand given to the constructor.
        y0 : torch.Tensor or None
            Value added to the returned integral, shape [D]. None uses the
            constructor's ``y0`` (0 by default).
        mesh : torch.Tensor or None
            Starting mesh, shape [M, T], increasing; its first and last points
            are the integration bounds. None builds a fresh starting mesh
            (:ref:`ug-mesh`). Replaced by the stored mesh under ``same_f``.
        mesh_init : torch.Tensor or None
            Lower integration bound, shape [T]. If ``mesh`` is given, it must
            equal the mesh's first point. None uses ``mesh[0]`` or the
            constructor's value.
        mesh_final : torch.Tensor or None
            Upper integration bound, shape [T]. If ``mesh`` is given, it must
            equal the mesh's last point. None uses ``mesh[-1]`` or the
            constructor's value.
        same_f : bool
            True when f is the same integrand as in the previous call, or
            changed only slightly: start from the stored mesh and ``max_batch``
            and skip the memory benchmark. Default False (:ref:`ug-same-f`).
        random_initial_mesh : bool
            If True (default), the points of a fresh starting mesh are placed at
            random, so the mesh does not line up with evenly spaced features of
            f; if False, they are evenly spaced (:ref:`ug-mesh`).
        N_init_steps : int
            Approximate number of subintervals of a fresh starting mesh. The
            default 13 gives 11 subintervals (:ref:`ug-mesh`).
        f_args : tuple
            Extra positional arguments passed to f; tensors are moved to the
            solver's device. Default ``()``.
        take_gradient : bool
            If True (default), backpropagate each batch's loss right after it,
            so the whole autograd graph is never held; if False, keep the graph
            so the returned integral can be differentiated (:ref:`ug-autodiff`).
        total_mem_usage : float or None
            Fraction of the device's total memory, in (0, 1], that batches may
            fill. Used only when no ``max_batch`` is set (:ref:`ug-memory`).
        loss_fxn : Callable or None
            Maps one batch's ``IntegrationResult`` to a scalar loss, which
            ``take_gradient=True`` backpropagates; ``result.loss`` is the sum
            over batches. None uses the integral (:ref:`ug-autodiff`).
        max_batch : int or None
            Largest number of integrand evaluations per call of f. If this and
            the constructor's value are None, the memory benchmark sets it
            (:ref:`ug-memory`). Replaced by the stored value under ``same_f``.
        max_adaptive_splits : int or None
            Cap on how many times a subinterval may have been halved since the
            starting mesh; at the cap it is accepted even if it fails the
            tolerance (:ref:`ug-convergence`).
        error_norm : str, Callable or None
            How the error of a vector-valued integrand becomes one
            accept-or-split decision (:ref:`ug-error-norm`).
        error_integral_reference : float, torch.Tensor or None
            A value close to the integral, used in place of the running integral
            in the relative tolerance ``rtol * |reference|``. Only with
            ``use_absolute_error_ratio=True`` (:ref:`ug-error-reference`).
        mesh_failure_tolerance : float or None
            With ``error_norm="failure_fraction"``, the fraction of output
            components that may fail while a subinterval is still accepted
            (:ref:`ug-error-norm`).
        error_on_nonfinite : bool or None
            If True, raise ``ValueError`` when f returns NaN or Inf; if False,
            accept those subintervals and let the value reach the result.
        conserve_memory : bool
            With ``take_gradient=False``, evaluate one batch of ``max_batch``
            points per pass instead of every waiting subinterval at once: less
            memory, more passes. Default False (:ref:`ug-memory`).
        result_device : str, torch.device or None
            Device of the per-subinterval arrays of the result (default
            ``"cpu"``); None keeps them on the solver's device. The integral and
            ``mesh_optimal`` stay on the solver's device.

        Returns
        -------
        IntegrationResult
            The integral (plus ``y0``), its error estimate, whether it
            ``converged``, the optimal mesh, the nodes and integrand values of
            every subinterval, and split statistics (:ref:`ug-convergence`).
        """
        # Set dtype based on input
        self.set_dtype_by_input(mesh=mesh, mesh_init=mesh_init, mesh_final=mesh_final)

        # Setup results device
        result_device = _resolve_device(
            result_device or self.device, self.device.index or 0
        )

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
        self._check_max_adaptive_splits(max_adaptive_splits)

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
        if error_integral_reference is not None:
            if not isinstance(error_integral_reference, torch.Tensor):
                error_integral_reference = torch.tensor(error_integral_reference)
            error_integral_reference = error_integral_reference.to(self.device)
            error_integral_reference = error_integral_reference.to(self.dtype)
        # A fixed reference sets every absolute-mode tolerance, so the in-loop
        # tolerance does not drift and accepted steps are not re-checked.
        reference_pinned = (
            self.use_absolute_error_ratio and error_integral_reference is not None
        )

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
        # The OOM cap lasts one call; same_f=True carries it over
        # through the stored previous_max_batch, which the cap clamps.
        self._oom_max_batch = None

        # Make sure f exists and provides the correct output. Running this
        # before the memory benchmark also warms f up, so one-time costs (CUDA
        # start-up, lazy model loading, cuBLAS workspace) stay out of it.
        # One point, shaped [1, T] like every batch f receives. mesh_init is
        # already on the solver's dtype and device (_check_variables).
        test_output = f(mesh_init.reshape(1, -1), *f_args)
        test_integrand, _ = self._split_f_output(test_output)
        assert len(test_integrand.shape) >= 2
        del test_output, test_integrand

        # same_f=True: the stored max_batch replaces the caller's and
        # skips the memory benchmark.
        if same_f and self.previous_max_batch is not None:
            force_max_batch = self.previous_max_batch
        if force_max_batch is None:
            self._setup_memory_checks(
                f,
                mesh_init,
                take_gradient=take_gradient,
                total_mem_usage=total_mem_usage,
                f_args=f_args,
            )
        loss_fxn = loss_fxn if loss_fxn is not None else self._integral_loss

        # Decide initial mesh:
        #   - same_f=True with a stored mesh: warm-start from it,
        #     snapping its endpoints to [mesh_init, mesh_final];
        #   - explicit mesh passed in: use it;
        #   - otherwise: generate a fresh random initial mesh.
        mesh, mesh_trackers, mesh_indices = self._setup_initial_mesh(
            mesh,
            mesh_init,
            mesh_final,
            same_f,
            random_initial_mesh,
            N_init_steps,
        )
        n_start_steps = len(mesh) - 1

        record: dict[str, torch.Tensor] = {}
        split_node_state: _SplitState = (None, None, None, None)
        # Per-panel split counter, parallel to mesh/mesh_trackers: how many
        # times each step was halved (0 = a step of the starting mesh). It
        # enforces max_adaptive_splits and gives the split statistics returned
        # with the result. Typed Optional because _adaptively_increase_mesh
        # returns it as Optional (its post-convergence caller passes None).
        split_counts: torch.Tensor | None = torch.zeros(
            len(mesh), dtype=torch.long, device=self.device
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
                conserve_memory,
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
                # Merge new steps into the sorted record to compute cumulative
                # sums. Placement is by mesh order (each panel's left barrier
                # looked up in mesh_indices), robust for any time dimension.
                idxs_keep, idxs_input = self._merge_positions(
                    self._mesh_order(record["nodes"][:, 0, :], mesh_indices),
                    self._mesh_order(nodes[:, 0, :], mesh_indices),
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
            # The tolerance denominator is max(atol, rtol*|reference_integral|)
            # (atol + rtol*|reference_integral| for failure_fraction). By
            # default the reference is the running integral, which is
            # incomplete in the early batches; a user-supplied
            # error_integral_reference (≈ the true total integral) replaces it
            # so rtol is judged against the right scale from the first batch.
            reference_integral = current_integral
            if error_integral_reference is not None:
                reference_integral = error_integral_reference
            error_ratios, error_ratios_2steps, error_ratios_per_dim = (
                self._compute_error_ratios(
                    mesh_quadrature_errors=method_output.mesh_quadrature_errors,
                    mesh_quadratures=method_output.mesh_quadratures,
                    cum_mesh_quadratures=cum_mesh_quadratures,
                    integral=reference_integral,
                )
            )
            keep_mask, remove_mask, _ = self._accept_reject_masks(
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

            # --- Step 4b: Re-check already-accepted (recorded) panels ---
            # The accept/reject denominator (built from reference_integral, Step 4)
            # drifts as the running integral changes, so a panel accepted early
            # can violate the *current* tolerance. Re-check the recorded panels
            # against the SAME reference_integral the batch just used (so a panel
            # accepted this iteration -- not yet in the record -- cannot
            # immediately re-fail) and split any that now fail in the same mesh
            # update below (extra_split_idxs). Only when not taking gradients:
            # per-batch backward has not run, so removing/re-splitting a recorded
            # panel cannot double-count gradients. Skipped when a fixed
            # error_integral_reference pins the denominator (no drift).
            recheck_active = (
                not take_gradient and len(record) > 0 and not reference_pinned
            )
            extra_split_idxs: torch.Tensor | None = None
            record_keep_mask: torch.Tensor | None = None
            if recheck_active:
                # All inputs are integration-device-resident (mesh_quadratures
                # and mesh_quadrature_errors are in _DEVICE_RESIDENT_KEYS;
                # reference_integral is on self.device) -- no host<->device copy.
                # The record is non-empty here, so Step 3 took its merge branch
                # and set idxs_keep. Cumulative mode judges each recorded step
                # against the running integral through it, which includes this
                # batch's steps to its left: the merged record+batch cumsum,
                # restricted to the recorded positions (a cumsum over the record
                # alone would leave the batch out).
                _, rec_remove, _ = self._record_error_check(
                    record,
                    torch.cumsum(all_mesh_quadratures.detach(), 0)[idxs_keep],
                    reference_integral,
                )
                rec_positions = self._mesh_order(record["nodes"][:, 0, :], mesh_indices)
                # Respect the split cap so the record stays consistent with the
                # mesh: a failing panel already at the cap is not split, so it
                # must stay recorded.
                splittable = rec_remove
                if max_adaptive_splits is not None:
                    assert split_counts is not None  # always kept in the loop
                    splittable = rec_remove & (
                        split_counts[rec_positions] < max_adaptive_splits
                    )
                extra_split_idxs = rec_positions[splittable]
                record_keep_mask = ~splittable

            # --- Step 5: Adaptive refinement ---
            # Split steps with error_ratio >= 1, keep steps with error_ratio < 1,
            # and update barriers/trackers accordingly. extra_split_idxs adds the
            # re-checked recorded panels that failed, so both the new batch and
            # the stale recorded panels are bisected in this single mesh update.
            (
                kept_output,
                kept_y,
                tracked_step_eval,
                kept_nodes,
                mesh,
                mesh_trackers,
                mesh_indices,
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
                extra_split_idxs=extra_split_idxs,
            )
            # The batch data passed in comes back filtered to the accepted steps
            # (None comes back only for None inputs, as in _get_optimal_mesh).
            assert kept_output is not None
            assert kept_y is not None
            assert kept_nodes is not None
            assert split_counts is not None  # returned non-None for non-None input
            method_output, y_step_eval, nodes = kept_output, kept_y, kept_nodes
            # Verify barrier ordering after adaptive refinement.
            # Any-T invariant: consecutive barriers are distinct (no
            # zero-length panels), which is all that is meaningful for a
            # multi-dimensional path.
            assert torch.all(torch.any(mesh[1:] != mesh[:-1], dim=-1))
            # For 1-D time the barriers must additionally be monotonic.
            if mesh.shape[-1] == 1:
                mesh_diff = mesh[1:, 0] - mesh[:-1, 0]
                assert torch.all(mesh_diff + self.atol_assert > 0) or torch.all(
                    mesh_diff - self.atol_assert < 0
                )

            # --- Step 5b: Drop re-split recorded panels from the record ---
            # The recorded panels flagged above were just bisected in the mesh
            # update; remove their now-stale entries (their children will be
            # re-evaluated as fresh panels). Done before _record_results so the
            # remaining record stays strictly mesh-ordered for the merge.
            # record_keep_mask is set exactly when recheck_active.
            if record_keep_mask is not None and not bool(torch.all(record_keep_mask)):
                record = self._remove_failed_record_panels(
                    record, record_keep_mask, loss_fxn
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
                    # Per-batch integrals exclude y0; the caller's y0 is added
                    # once to the returned integral.
                    y0=torch.zeros_like(method_output.integral),
                )

                # TODO make sure growing string loss center is a time not the number of evals because eval number is meaningless here.
                # Compute loss
                loss = loss_fxn(intermediate_results)

                # Backpropagate gradients through the integration if requested
                if take_gradient and loss.requires_grad:
                    loss.backward()

                # Accumulate into the record
                intermediate_results.loss = loss
                record = self._record_results(
                    record=record,
                    take_gradient=take_gradient,
                    results=intermediate_results,
                    mesh_indices=mesh_indices,
                    result_device=result_device,
                )

            del y_step_eval

        # === Post-convergence: sort results and optimize the mesh ===
        record = self._sort_record(record, mesh_indices)

        # Split statistics. A step of the starting mesh has split count 0 and
        # every half of a split step has count >= 1, so the steps still at 0
        # are the starting steps that were never split. The last barrier
        # starts no step. Splits made below by _get_optimal_mesh are not
        # counted, and max_adaptive_splits truncates the depth.
        assert split_counts is not None  # always kept in the loop
        step_split_counts = split_counts[:-1]
        n_start_steps_split = n_start_steps - int((step_split_counts == 0).sum())
        max_split_depth = int(step_split_counts.max())

        # Final error check: does the returned integral meet the tolerance?
        # Every recorded step is judged again against the final integral (in
        # cumulative mode, the final running integral through the step), the
        # reference _get_optimal_mesh below also uses. A step can fail here
        # although it passed when accepted, because it was judged against the
        # running integral and, with take_gradient=True, nothing re-checked it.
        # A step accepted at max_adaptive_splits usually fails, but can pass.
        # A non-finite error ratio never meets a tolerance; _accept_reject_masks
        # accepts it only because splitting cannot reduce it. In absolute mode
        # mesh_optimal splits every finite failing step; in cumulative mode
        # _prune_excess_mesh can merge one with a right neighbour whose running
        # integral, and so its tolerance, is much larger. Runs before the record
        # is moved to result_device, so all inputs are on the integration device.
        final_ratios, final_fails, nonfinite = self._record_error_check(
            record,
            torch.cumsum(record["mesh_quadratures"].detach(), 0),
            record["integral"].detach(),
        )
        final_fails = final_fails | nonfinite
        converged = not bool(final_fails.any())
        if not converged:
            self._warn_not_converged(
                record,
                final_ratios,
                final_fails,
                nonfinite,
                max_adaptive_splits,
                take_gradient,
                reference_pinned,
            )

        # Prune over-resolved steps and refine under-resolved ones. This re-derives
        # node positions through the method tableau, so it must run with the
        # tableau and device-bound allocations co-located with its inputs. The
        # bulky per-step field nodes already lives on result_device, so we
        # evaluate the whole computation there -- moving the small tableau to
        # result_device for the duration -- rather than copying it back to the
        # integration device. The device-resident fields (mesh_quadratures,
        # mesh_quadrature_errors, integral; see _DEVICE_RESIDENT_KEYS) are the
        # ones that must be moved to result_device to join it.
        with self._integration_on_device(result_device):
            mesh_optimal = self._get_optimal_mesh(
                nodes=record["nodes"],
                mesh_quadratures=record["mesh_quadratures"].to(result_device),
                mesh_quadrature_errors=record["mesh_quadrature_errors"].to(
                    result_device
                ),
                integral=record["integral"].to(result_device),
                mesh=mesh.to(result_device),
            )
        # mesh_optimal is fed back into the integrator as the warm-start mesh on
        # the next call, so it must match the integration device (its endpoints
        # are mesh_init/mesh_final, which live on self.device).
        mesh_optimal = mesh_optimal.to(self.device)
        # Stored together for same_f=True; only a returning call writes
        # them (converged or not), so both come from one call.
        self.mesh_previous = mesh_optimal
        self.previous_max_batch = self.get_max_batch(total_mem_usage, force_max_batch)

        # Move the resident record fields to result_device so the rest of the
        # returned IntegrationResult lives on result_device. The integration-
        # output scalars (integral, integral_error) and mesh_optimal stay on the
        # integration device -- the scalars are the primary answer and
        # mesh_optimal is reused as the warm-start mesh.
        for key in self._DEVICE_RESIDENT_KEYS:
            if key in record and key not in self._RESULT_INTEGRATION_KEYS:
                record[key] = record[key].to(result_device)

        # Each tracked variable is its own record key (see _record_fields), in
        # the order the integrand returns them.
        record_tracked = tuple(
            value
            for key, value in record.items()
            if key.startswith(self._TRACKED_KEY_PREFIX)
        )

        return IntegrationResult(
            # integral, integral_error and y0 stay on the integration device.
            integral=record["integral"] + y0,
            integral_error=record["integral_error"],
            # The mesh family (mesh_optimal/init/final) stays on the integration
            # device: mesh_optimal is the warm-start mesh and its endpoints are
            # mesh_init/mesh_final, so they must share a device.
            mesh_optimal=mesh_optimal,
            mesh_init=mesh_init,
            mesh_final=mesh_final,
            nodes=self._flatten_output(record["nodes"]),
            h=record["h"],
            y=self._flatten_output(record["y"]),
            # Flatten tracked variables the same way so they stay aligned with
            # the flattened nodes/y.
            tracked_variables=tuple(self._flatten_output(tv) for tv in record_tracked)
            if record_tracked
            else None,
            mesh_quadratures=record["mesh_quadratures"],
            mesh_quadrature_errors=torch.abs(record["mesh_quadrature_errors"]),
            error_ratios=record["error_ratios"],
            loss=record["loss"],
            gradient_taken=take_gradient,
            y0=y0,
            converged=converged,
            n_start_steps=n_start_steps,
            n_start_steps_split=n_start_steps_split,
            max_split_depth=max_split_depth,
        )

    def _record_error_check(
        self,
        record: dict[str, torch.Tensor],
        cum_mesh_quadratures: torch.Tensor,
        integral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Judge every recorded (accepted) subinterval against a given reference.

        The one definition of "a recorded subinterval fails", used by the
        in-loop re-check of accepted subintervals (Step 4b of ``integrate``)
        and by the final ``converged`` check. It uses the same error ratios and
        accept rule as a freshly evaluated batch (``_compute_error_ratios``,
        ``_accept_reject_masks``).

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            The running record, in mesh order, with ``mesh_quadratures`` and
            ``mesh_quadrature_errors`` [N, D] on the integration device.
        cum_mesh_quadratures : torch.Tensor
            Running integral through each recorded subinterval [N, D] (used in
            cumulative mode only).
        integral : torch.Tensor
            Reference integral [D] for the absolute-mode tolerance.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple ``(error_ratios, reject_mask, nonfinite_mask)``: the error
            ratio of each subinterval [N], and the split and non-finite masks
            [N] of ``_accept_reject_masks`` (which accepts non-finite
            subintervals).
        """
        error_ratios, _, error_ratios_per_dim = self._compute_error_ratios(
            mesh_quadrature_errors=record["mesh_quadrature_errors"].detach(),
            mesh_quadratures=record["mesh_quadratures"].detach(),
            cum_mesh_quadratures=cum_mesh_quadratures,
            integral=integral,
        )
        _, reject_mask, nonfinite_mask = self._accept_reject_masks(
            error_ratios, error_ratios_per_dim
        )
        return error_ratios, reject_mask, nonfinite_mask

    def _warn_not_converged(
        self,
        record: dict[str, torch.Tensor],
        final_ratios: torch.Tensor,
        final_fails: torch.Tensor,
        nonfinite: torch.Tensor,
        max_adaptive_splits: int | None,
        take_gradient: bool,
        reference_pinned: bool,
    ) -> None:
        """
        Log how far a returned integral is from meeting the tolerance.

        Called by ``integrate`` when the final error check fails. The sizes
        reported are error estimates (the difference between the method and
        its embedded lower-degree method), not the true error. For D > 1 they
        are reported for one output component: the one whose failing
        subintervals' summed error is largest relative to the absolute value
        of its integral.

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            The sorted record on the integration device.
        final_ratios : torch.Tensor
            Error ratio of each subinterval from the final check [N] (error
            over tolerance for the norm family; fraction of failing output
            components for ``"failure_fraction"``).
        final_fails : torch.Tensor
            Subintervals that fail the final check, including the non-finite
            ones [N].
        nonfinite : torch.Tensor
            Subintervals whose error ratio is NaN or Inf [N] (a non-finite
            error estimate, or a non-finite tolerance from the integral).
        max_adaptive_splits : int | None
            The split cap used by the call, or None.
        take_gradient : bool
            Whether the call backpropagated per batch (then accepted
            subintervals were not re-checked).
        reference_pinned : bool
            Whether a fixed ``error_integral_reference`` set the absolute-mode
            tolerance (then accepted subintervals were not re-checked, and the
            loop judged them against that reference).
        """
        finite_fails = final_fails & ~nonfinite
        # Per-component sums of |per-step error estimate| [D], over the finite
        # failing steps and over all finite steps. Absolute values, unlike
        # integral_error, whose signed per-step errors can cancel.
        abs_errors = torch.abs(record["mesh_quadrature_errors"])
        fail_error = abs_errors[finite_fails].sum(0)
        all_error = abs_errors[~nonfinite].sum(0)
        abs_integral = torch.abs(record["integral"])
        relative = torch.where(
            fail_error > 0, fail_error / abs_integral, torch.zeros_like(fail_error)
        )
        comp = int(torch.argmax(relative))
        if bool(finite_fails.any()):
            worst = float(final_ratios[finite_fails].max())
            fail_value = float(fail_error[comp])
        else:
            worst = fail_value = float("nan")
        component = (
            f" (output component {comp} of {len(abs_integral)})"
            if len(abs_integral) > 1
            else ""
        )
        remedies = []
        if max_adaptive_splits is not None:
            remedies.append("raise max_adaptive_splits")
        if take_gradient and not reference_pinned:
            remedies.append(
                "pass take_gradient=False (accepted steps are then re-checked)"
            )
        if reference_pinned:
            remedies.append(
                "use an error_integral_reference no larger than |integral| "
                "(converged is judged against the final integral)"
            )
        remedies.append("call again with same_f=True to start from mesh_optimal")
        logger.warning(
            "The integral did not converge: %d of %d accepted steps (%.1f%%) do "
            "not meet the tolerance (atol=%.3g, rtol=%.3g) judged against %s. "
            "Failing steps with a non-finite error ratio (a NaN/Inf error "
            "estimate or tolerance): %d. Largest error ratio among the finite "
            "failing steps: "
            "%.3g (%s). Their summed |error estimate| is %.3g, of %.3g summed "
            "over all finite steps, for |integral| = %.3g%s (estimates, not the "
            "true error). To refine further: %s.",
            int(final_fails.sum()),
            len(final_fails),
            100.0 * float(final_fails.float().mean()),
            float(self.atol),
            float(self.rtol),
            "the final integral"
            if self.use_absolute_error_ratio
            else "the final running integral through each step",
            int((final_fails & nonfinite).sum()),
            worst,
            "fraction of output elements over tolerance"
            if self._is_failure_fraction
            else "error / tolerance",
            fail_value,
            float(all_error[comp]),
            float(abs_integral[comp]),
            component,
            "; ".join(remedies),
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
        extra_split_idxs: torch.Tensor | None = None,
    ) -> tuple[
        MethodOutput | None,
        torch.Tensor | None,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        dict[tuple[float, ...], int],
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Accept the accurate subintervals of a batch and split the others in half.

        For each evaluated subinterval, with the masks from
        ``_accept_reject_masks`` (or, when they are not given, from the error
        ratio alone):

        - error ratio below 1: accept. Its tracker is set False and its
          contribution is kept.
        - error ratio 1 or above: split. A new mesh point is inserted at the
          midpoint ``(mesh_left + mesh_right) / 2``, and both halves are
          evaluated in a later pass.
        - non-finite error ratio (NaN or Inf): accept. Splitting cannot be
          relied on to remove a non-finite value, so splitting until it is
          gone might never end. The non-finite value reaches the recorded
          integral.

        When ``split_counts`` and ``max_adaptive_splits`` are given, a
        subinterval whose split count has reached ``max_adaptive_splits`` is
        accepted even if its error ratio is 1 or above. Both halves of a
        subinterval with split count n get split count n + 1.

        Parameters
        ----------
        method_output : MethodOutput | None
            Integral and error output of the current batch (None in the
            optimal-mesh pass). If given, the split subintervals are removed
            from it.
        error_ratios : torch.Tensor
            Error ratio of each subinterval. Shape: [N_batch].
        y_step_eval : torch.Tensor | None
            Integrand values of the batch. Shape: [N_batch, C, D].
        nodes : torch.Tensor | None
            Nodes of the batch. Shape: [N_batch, C, T].
        mesh : torch.Tensor
            All mesh points. Shape: [M, T].
        mesh_idxs : torch.Tensor
            Mesh index of the left end of each subinterval of the batch.
            Shape: [N_batch].
        mesh_trackers : torch.Tensor
            True for the subintervals that still have to be evaluated.
            Shape: [M].
        tracked_step_eval : tuple[torch.Tensor, ...] | None
            Tracked variables of the batch, each of shape
            [N_batch, C, ...]. Filtered to the accepted subintervals like
            ``y_step_eval``. None when the integrand returns none.
        split_counts : torch.Tensor | None
            Split count of the subinterval starting at each mesh point.
            Shape: [M]. None disables split counting (the optimal-mesh pass
            passes None).
        max_adaptive_splits : int | None
            Cap on the split count; a subinterval at or above it is accepted
            instead of split. None disables the cap.
        keep_mask : torch.Tensor | None
            Accept mask (True = accept) from ``_accept_reject_masks``, shape
            [N_batch]. When it and ``remove_mask`` are given they are used
            as they are; otherwise the masks come from ``error_ratios < 1``
            with non-finite ratios accepted (for direct callers and tests).
        remove_mask : torch.Tensor | None
            Split mask, the complement of ``keep_mask``.
        extra_split_idxs : torch.Tensor | None
            Further mesh indices to split in the same update, besides the
            batch's rejected subintervals: recorded subintervals that failed
            the re-check after the running integral changed. Must be
            ascending and disjoint from ``mesh_idxs[remove_mask]``. Their
            trackers are set True so both halves are evaluated. None splits
            only the batch's rejected subintervals.

        Returns
        -------
        tuple[MethodOutput | None, torch.Tensor | None, tuple[torch.Tensor, ...] | None, torch.Tensor | None, torch.Tensor, torch.Tensor, dict[tuple[float, ...], int], torch.Tensor, torch.Tensor | None]
            Tuple ``(method_output, y_step_eval, tracked_step_eval, nodes,
            mesh_new, mesh_trackers_new, mesh_indices_new, error_ratios_kept,
            split_counts_new)``:

            - method_output: with the split subintervals removed.
            - y_step_eval: integrand values of the accepted subintervals.
            - tracked_step_eval: their tracked variables (or None).
            - nodes: their nodes.
            - mesh_new: the mesh with the midpoints inserted.
            - mesh_trackers_new: the trackers, True for the new halves.
            - mesh_indices_new: ``{tuple(mesh point): index}`` for mesh_new
              (``_get_mesh_indices``).
            - error_ratios_kept: error ratios of the accepted subintervals.
            - split_counts_new: split counts for mesh_new, with both halves
              of every split subinterval incremented (or None if
              ``split_counts`` is None).
        """
        if keep_mask is None or remove_mask is None:
            # Legacy/direct-call path: derive the masks from a scalar error
            # ratio. Splitting cannot be relied on to remove a non-finite error
            # ratio (NaN/Inf): a NaN/Inf at a mesh point comes back as an
            # endpoint of a half, and a NaN/Inf running integral makes every
            # tolerance non-finite.
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

        # Mesh positions to bisect this update: the batch's rejected steps plus
        # any extra positions (already-accepted panels that failed the re-check
        # against the drifted tolerance). The recorded panels were accepted
        # (tracker False), so re-activate the whole split set to ensure both
        # children of every bisected panel are re-evaluated. With no extras this
        # is exactly the previous batch-only behaviour (and trackers untouched
        # here, matching the original).
        split_idxs = mesh_idxs[remove_mask]
        if extra_split_idxs is not None and extra_split_idxs.numel() > 0:
            split_idxs = torch.sort(torch.cat([split_idxs, extra_split_idxs])).values
            # Batch (tracker True) and recorded (tracker False) positions are
            # disjoint, so the combined set must be strictly ascending; a
            # duplicate would corrupt the midpoint-offset math below.
            assert torch.all(split_idxs[1:] > split_idxs[:-1])
            mesh_trackers[split_idxs] = True

        N_t_add = len(split_idxs)
        # Allocate new barriers array with room for inserted midpoints. All
        # allocations follow ``mesh.device`` so this runs on the integration
        # device in the loop and on result_device in the post-convergence pass.
        mesh_new = torch.nan * torch.ones(
            (N_t_add + len(mesh), mesh.shape[-1]),
            dtype=self.dtype,
            device=mesh.device,
        )
        mesh_trackers_new = torch.ones(
            N_t_add + len(mesh), dtype=torch.bool, device=mesh.device
        )

        # Transfer existing barriers to their new positions in the expanded array.
        # Each rejected step causes a +1 offset for all subsequent barriers
        # (because a midpoint is being inserted). idx_offset tracks this shift.
        idx_offset = torch.zeros(len(mesh), dtype=torch.long, device=mesh.device)
        idx_offset[split_idxs + 1] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idxs_transfer = idx_offset + torch.arange(len(mesh), device=mesh.device)
        mesh_new[idxs_transfer] = mesh.clone()
        mesh_trackers_new[idxs_transfer] = mesh_trackers.clone()

        # Insert new midpoint barriers between the start and end of rejected steps.
        # The midpoint is placed at (left_barrier + right_barrier) / 2.
        idxs_new = split_idxs + torch.arange(N_t_add, device=mesh.device) + 1
        t_add_barriers = 0.5 * (mesh_new[idxs_new - 1] + mesh_new[idxs_new + 1])
        mesh_new[idxs_new] = t_add_barriers
        assert torch.sum(torch.isnan(mesh_new)) == 0
        assert len(idxs_new) + len(idxs_transfer) == len(mesh_new)

        # Expand the per-panel split counts the same way as the barriers: carry
        # existing counts to their new positions, then bump both children of
        # each split panel to parent_count + 1.
        if split_counts is not None:
            split_counts_new = torch.zeros(
                len(mesh_new), dtype=torch.long, device=mesh_new.device
            )
            split_counts_new[idxs_transfer] = split_counts.clone()
            parent_counts = split_counts[split_idxs]
            # left child = transferred parent barrier; right child = new midpoint
            split_counts_new[idxs_transfer[split_idxs]] = parent_counts + 1
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
            self._get_mesh_indices(mesh_new),
            error_ratios[keep_mask],
            split_counts_new,
        )

    @staticmethod
    def _split_f_output(
        out: Any,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """
        Normalize the integrand's output into (integrand, tracked_variables).

        The integrand ``f`` may return either:

        - the integrand values alone; tracked is then None, or
        - a 2-tuple (or 2-element list) ``(integrand, tracked_variables)``,
          where ``tracked_variables`` is a tuple of tensors, a single tensor
          (wrapped into a 1-tuple), or None (an empty tuple counts as None).

        Parameters
        ----------
        out : Any
            The raw return value of a call to ``f``.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]
            Tuple of (integrand_tensor, tracked_tuple_or_None).
        """
        if isinstance(out, (tuple, list)) and len(out) == 2:
            integrand, tracked = out
            if tracked is None:
                return integrand, None
            if torch.is_tensor(tracked):
                tracked = (tracked,)
            # An empty tuple means no tracked variables, like None. The record
            # stores one key per tracked variable, so it could not tell the two
            # apart at the end of the run.
            return integrand, tuple(tracked) or None
        return out, None

    def _check_f_output_finite(
        self, integrand: torch.Tensor, nodes_flat: torch.Tensor, max_report: int = 5
    ) -> None:
        """
        Raise a located ``ValueError`` if an integrand batch is non-finite.

        A NaN or Inf integrand value makes the subinterval's error ratio
        non-finite. Detecting it where it arises, instead of letting it reach a
        NaN result, gives the user an error that names the offending ``t``.
        Runs only when ``self.error_on_nonfinite`` is True (set from the
        ``integrate(error_on_nonfinite=...)`` argument or the constructor).

        Parameters
        ----------
        integrand : torch.Tensor
            The integrand values of this batch, shape ``[B, D]`` (``B = N*C``
            for whole subintervals, or the batch size when a batch holds part
            of the nodes). May also be a Python number if ``f`` returns one;
            it is then checked with ``math.isfinite`` and no ``t`` is reported.
        nodes_flat : torch.Tensor
            The nodes passed to ``f`` for this batch. Shape ``[B, T]``. Used
            only to report the offending ``t``.
        max_report : int
            Largest number of offending ``t`` values to list.

        Raises
        ------
        ValueError
            If ``self.error_on_nonfinite`` is True and any value is NaN or
            Inf. When it is False this check does nothing, and
            ``_accept_reject_masks`` accepts the non-finite subintervals so
            the loop still ends.
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
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        mesh: torch.Tensor,
        mesh_trackers: torch.Tensor,
        take_gradient: bool,
        conserve_memory: bool,
        force_max_batch: int | None,
        total_mem_usage: float,
        split_node_state: _SplitState,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor,
        _SplitState,
    ]:
        """
        Evaluate the integrand at the nodes of the next unevaluated subintervals.

        Chooses the batch size with ``get_max_batch`` (raised to 1 with a warning
        if it is 0, capped at the number of evaluations left) and dispatches to
        ``_evaluate_f_on_full_nodes`` when ``take_gradient`` is True, else to
        ``_evaluate_f_on_split_residual_nodes`` when ``conserve_memory`` is True,
        else to ``_evaluate_f_on_split_nodes``.

        Parameters
        ----------
        f : callable
            The integrand.
        f_args : tuple
            Extra arguments passed to ``f`` after the nodes.
        mesh : torch.Tensor
            The current mesh. Shape: [M, T].
        mesh_trackers : torch.Tensor
            Boolean, shape [M]: True at the left mesh point of every subinterval
            that still needs evaluation.
        take_gradient : bool
            If True, evaluate whole subintervals with one call of ``f`` per batch.
        conserve_memory : bool
            If True (and ``take_gradient`` is False), evaluate about one batch and
            carry a partly evaluated subinterval to the next call.
        force_max_batch : int or None
            Batch size that takes precedence over the memory estimate; None uses
            the estimate.
        total_mem_usage : float
            Fraction of the device's memory that may be in use.
        split_node_state : _SplitState
            The partly evaluated subinterval carried from the previous call; all
            None when nothing is carried.

        Returns
        -------
        tuple
            ``(nodes, f_evals, tracked, step_idxs, split_node_state)``: the nodes
            [N, C, T] and integrand values [N, C, D] of the N evaluated
            subintervals, their detached tracked variables (each [N, C, ...]) or
            None, the indices of their left mesh points [N], ascending, and the
            state carried to the next call (all None unless ``conserve_memory``
            is True).
        """
        # Determine how many f evaluations fit in one batch based on memory
        max_batch = self.get_max_batch(total_mem_usage, force_max_batch)
        # integrate() runs the memory benchmark whenever no max_batch is
        # forced, so a budget always exists here.
        assert max_batch is not None

        # Rescue a zero budget to the minimum workable value of 1 before
        # dispatching to either evaluation path. Placed ahead of the
        # take_gradient branch so neither _evaluate_f_on_full_nodes nor
        # _evaluate_f_on_split_nodes is ever entered with max_batch=0. A 0 here
        # comes from the dynamic memory estimate collapsing (or an explicit
        # max_batch=0); it is always a misconfiguration, hence the warning.
        if max_batch == 0:
            logger.warning(
                "max_batch is 0, increasing to 1. This is a significant issue and "
                "should be addressed. Either your memory is too small and you will "
                "get an OOM error, or your system's memory is being significantly "
                "underutilized and removing the benefits of padaquad's "
                "parallelization. For the latter case, increase max_batch or "
                "total_mem_usage to be closer to 1.0"
            )
            max_batch = 1

        # max_batch should not exceed the remaining number of f evaluations
        batches_left = torch.sum(mesh_trackers) * self.C
        assert batches_left > 0
        # int(): keep max_batch a Python int (batches_left is a 0-d tensor; the
        # assert above already synchronized on it).
        max_batch = min(max_batch, int(batches_left))
        max_mesh_steps = max_batch // self.C

        step_idxs = torch.arange(len(mesh), device=self.device)
        step_idxs = step_idxs[mesh_trackers]

        if take_gradient:
            return self._evaluate_f_on_full_nodes(
                f, f_args, mesh, step_idxs, max_mesh_steps
            )
        else:
            if conserve_memory:
                return self._evaluate_f_on_split_residual_nodes(
                    f,
                    f_args,
                    mesh,
                    step_idxs,
                    max_batch,
                    max_mesh_steps,
                    split_node_state,
                )
            else:
                return self._evaluate_f_on_split_nodes(
                    f,
                    f_args,
                    mesh,
                    step_idxs,
                    max_batch,
                )

    def _evaluate_f_on_full_nodes(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        mesh: torch.Tensor,
        step_idxs: torch.Tensor,
        max_mesh_steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor,
        _SplitState,
    ]:
        """
        Evaluate the integrand on the first max_mesh_steps subintervals in one call.

        Used when ``take_gradient`` is True. No out-of-memory retry is made.

        Parameters
        ----------
        f : callable
            The integrand.
        f_args : tuple
            Extra arguments passed to ``f`` after the nodes.
        mesh : torch.Tensor
            The current mesh. Shape: [M, T].
        step_idxs : torch.Tensor
            Indices of the left mesh points of the subintervals that need
            evaluation, ascending. Shape: [K].
        max_mesh_steps : int
            Largest number of subintervals to evaluate, at least 1.

        Returns
        -------
        tuple
            ``(nodes, f_evals, tracked, step_idxs, split_node_state)``: the nodes
            [N, C, T] and integrand values [N, C, D] of the N evaluated
            subintervals, their detached tracked variables (each [N, C, ...]) or
            None, the indices of their left mesh points [N], ascending, and the
            state carried to the next call (all None).
        """
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

    def _evaluate_f_in_batches(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        nodes_flat: torch.Tensor,
        num_nodes_to_eval: int,
        max_batch: int,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]] | None, int]:
        """
        Evaluate f on nodes_flat in batches of at most max_batch nodes.

        Used by both evaluation paths of ``take_gradient=False``. Batches run
        until at least num_nodes_to_eval nodes are evaluated; the last batch
        may run past it, and _evaluate_f_on_split_residual_nodes keeps that
        overshoot as the residual. On an out-of-memory error the failed batch
        is retried with a smaller max_batch, cached in _oom_max_batch.

        Parameters
        ----------
        f : callable
            The integrand.
        f_args : tuple
            Extra arguments passed to ``f`` after the nodes.
        nodes_flat : torch.Tensor
            The nodes to evaluate, in order. Shape: [K, T].
        num_nodes_to_eval : int
            Batches run until at least this many nodes are evaluated.
        max_batch : int
            Largest number of nodes per call of ``f``.

        Returns
        -------
        tuple[list[torch.Tensor], list[list[torch.Tensor]] | None, int]
            The per-batch integrand values ``f_evals``, each [batch, D]; the
            per-batch lists of tracked variables (entry k for the k-th tracked
            variable), or None when ``f`` returns none; and the number of nodes
            evaluated (at least num_nodes_to_eval).

        Raises
        ------
        torch.OutOfMemoryError
            If one node does not fit in memory (max_batch is already 1).
        """
        f_evals: list[torch.Tensor] = []
        tracked_lists: list[list[torch.Tensor]] | None = None
        num_nodes_evaluated = 0
        while num_nodes_evaluated < num_nodes_to_eval:
            out_of_memory = False
            try:
                f_output = f(
                    nodes_flat[num_nodes_evaluated : num_nodes_evaluated + max_batch],
                    *f_args,
                )
            except (RuntimeError, MemoryError) as e:
                if not _is_out_of_memory(e):
                    raise
                if max_batch == 1:
                    raise torch.OutOfMemoryError(
                        f"{e}\n\nSingle integrand (f) evaluation failed to fit in memory, batch size cannot be reduced."
                    ) from e
                out_of_memory = True
            if out_of_memory:
                # Cleaned up here, after the except block: inside it the error's
                # traceback still holds f's tensors, so they cannot be freed.
                free_mem, total_mem = self._get_memory()
                self._release_memory()
                # Seed from the live max_batch on the first OOM (_oom_max_batch
                # starts as None), then shrink. min(..., prev - 1) guarantees a
                # strict decrease -- plain round(0.75 * 2) == 2 would stick and
                # retry forever.
                prev = (
                    self._oom_max_batch
                    if self._oom_max_batch is not None
                    else max_batch
                )
                self._oom_max_batch = max(1, int(min(round(0.75 * prev), prev - 1)))
                logger.warning(
                    f"Caught OOM with {free_mem} GB free of {total_mem} GB. "
                    f"Reducing max_batch from {max_batch} to {self._oom_max_batch}."
                    "If this warning appears often, consider reducing max_batch to "
                    "avoid deleting and rerunning evaluations."
                )
                max_batch = self._oom_max_batch
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
        return f_evals, tracked_lists, num_nodes_evaluated

    def _evaluate_f_on_split_nodes(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        mesh: torch.Tensor,
        step_idxs: torch.Tensor,
        max_batch: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor,
        _SplitState,
    ]:
        """
        Evaluate the integrand on all the given subintervals, max_batch nodes per call.

        Used when ``take_gradient`` and ``conserve_memory`` are both False.

        Parameters
        ----------
        f : callable
            The integrand.
        f_args : tuple
            Extra arguments passed to ``f`` after the nodes.
        mesh : torch.Tensor
            The current mesh. Shape: [M, T].
        step_idxs : torch.Tensor
            Indices of the left mesh points of the subintervals that need
            evaluation, ascending. Shape: [K].
        max_batch : int
            Largest number of nodes per call of ``f``, at least 1.

        Returns
        -------
        tuple
            ``(nodes, f_evals, tracked, step_idxs, split_node_state)``: the nodes
            [N, C, T] and integrand values [N, C, D] of the N evaluated
            subintervals, their detached tracked variables (each [N, C, ...]) or
            None, the indices of their left mesh points [N], ascending, and the
            state carried to the next call (all None).
        """
        # A zero budget is unsatisfiable here and would divide-by-zero below
        # (the % max_batch layout math). Callers route through
        # _evaluate_f_on_mesh, which rescues max_batch=0 -> 1 before dispatch;
        # this assert guards direct/unit callers with an explicit message.
        assert max_batch > 0

        # Place C quadrature points within each selected step
        nodes = self._compute_nodes(mesh[step_idxs], mesh[step_idxs + 1])

        # Flatten [N, C, T] -> [N*C, T] for batch evaluation, then reshape back
        nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=-2)
        f_eval_batches, tracked_lists, _ = self._evaluate_f_in_batches(
            f, f_args, nodes_flat, len(nodes_flat), max_batch
        )

        # Combine split evaluations and nodes
        f_evals = torch.concatenate(f_eval_batches, dim=0)

        # Combine tracked variables the same way (prepending the carried split).
        tracked_combined: list[torch.Tensor] | None = None
        if tracked_lists is not None:
            tracked_combined = [torch.concatenate(tl, dim=0) for tl in tracked_lists]

        # Reshape and combine outputs
        nodes = torch.reshape(nodes_flat, (-1, self.C, nodes_flat.shape[-1]))
        f_evals = torch.reshape(f_evals, (-1, self.C, f_evals.shape[-1]))
        if tracked_combined is not None:
            tracked_combined = [
                tv.reshape(-1, self.C, *tv.shape[1:]) for tv in tracked_combined
            ]

        tracked_out = (
            tuple(tv.detach() for tv in tracked_combined)
            if tracked_combined is not None
            else None
        )

        return nodes, f_evals, tracked_out, step_idxs, (None, None, None, None)

    def _evaluate_f_on_split_residual_nodes(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        mesh: torch.Tensor,
        step_idxs: torch.Tensor,
        max_batch: int,
        max_mesh_steps: int,
        split_node_state: _SplitState,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor,
        _SplitState,
    ]:
        """
        Evaluate about one batch of nodes, carrying a partly evaluated subinterval.

        Used when ``conserve_memory`` is True. The remaining nodes of the
        subinterval carried in ``split_node_state`` (if any) are evaluated
        first, then max_mesh_steps whole subintervals. When the last batch ends
        inside one more subinterval, that subinterval is not returned: its
        evaluated nodes, values, tracked variables and left mesh point become
        the returned carried state.

        Parameters
        ----------
        f : callable
            The integrand.
        f_args : tuple
            Extra arguments passed to ``f`` after the nodes.
        mesh : torch.Tensor
            The current mesh. Shape: [M, T].
        step_idxs : torch.Tensor
            Indices of the left mesh points of the subintervals that need
            evaluation, ascending. Shape: [K].
        max_batch : int
            Largest number of nodes per call of ``f``, at least 1.
        max_mesh_steps : int
            Number of whole subintervals to evaluate (max_batch // C); 0 is
            raised to 1.
        split_node_state : _SplitState
            The partly evaluated subinterval carried from the previous call; all
            None when nothing is carried.

        Returns
        -------
        tuple
            ``(nodes, f_evals, tracked, step_idxs, split_node_state)``: the nodes
            [N, C, T] and integrand values [N, C, D] of the N evaluated
            subintervals, their detached tracked variables (each [N, C, ...]) or
            None, the indices of their left mesh points [N], ascending, and the
            state carried to the next call (all None when nothing is carried).
        """
        # A zero budget is unsatisfiable here and would divide-by-zero below
        # (the % max_batch layout math). Callers route through
        # _evaluate_f_on_mesh, which rescues max_batch=0 -> 1 before dispatch;
        # this assert guards direct/unit callers with an explicit message.
        assert max_batch > 0
        # When max_batch < C the budget cannot hold even one whole panel, so the
        # dispatcher's max_batch // C floors to 0. Unlike the full-nodes path
        # (which requires >= 1 whole panel per batch), the split path evaluates a
        # single panel's C nodes across several max_batch-sized batches and
        # carries the remainder forward in split_node_state. Force at least one
        # panel of work so that sub-C budgets still make progress.
        max_mesh_steps = 1 if max_mesh_steps == 0 else max_mesh_steps
        # Gather nodes and evaluations from previous split. split_mesh_idx
        # stores the residual panel's left-barrier *coordinate* (not an
        # integer index), so it survives mesh refinement: the calling
        # integrate loop inserts midpoints between iterations, shifting
        # any cached integer index. Barriers themselves are never removed
        # mid-loop, so a stored coordinate stays resolvable.
        split_nodes, split_f_evals, split_tracked, split_mesh_idx = split_node_state
        num_split_nodes = 0
        if split_mesh_idx is not None:
            # The carried state is written as one unit (see the end of this
            # method), so nodes and evaluations exist whenever the index does.
            assert split_nodes is not None
            assert split_f_evals is not None
            num_split_nodes = len(split_nodes)
        num_remaining_split_nodes = self.C - num_split_nodes

        if split_mesh_idx is not None:
            # Translate the cached barrier coordinate back to an index in
            # the current (possibly refined) mesh.
            split_mesh_idx = torch.where((mesh == split_mesh_idx).all(dim=-1))[0][0]

        if split_mesh_idx is None:
            num_mesh_steps = max_mesh_steps
            if (max_mesh_steps * self.C) % max_batch != 0:
                num_mesh_steps += 1

            step_idxs = step_idxs[:num_mesh_steps]

            # Place C quadrature points within each selected step
            nodes = self._compute_nodes(mesh[step_idxs], mesh[step_idxs + 1])

            # Flatten [N, C, T] -> [N*C, T] for batch evaluation, then reshape back
            nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=-2)
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
            nodes_flat = nodes_flat[num_split_nodes:]
            num_eval_nodes = len(nodes_flat)

            # One batch of up to max_batch new evals per call. After
            # completing the carry-over panel (num_remaining_split_nodes
            # evals), the rest splits into full new panels plus a partial
            # residual; if num_eval_nodes < max_batch the layout's tail
            # already ends cleanly, so the residual is zero.
            num_nodes_to_eval = min(
                max_mesh_steps * self.C + num_remaining_split_nodes,
                num_eval_nodes,
            )

        # The last batch may run past num_nodes_to_eval; that overshoot is the
        # residual carried to the next call.
        f_eval_batches, tracked_lists, num_nodes_evaluated = (
            self._evaluate_f_in_batches(
                f, f_args, nodes_flat, num_nodes_to_eval, max_batch
            )
        )
        num_residual_nodes = num_nodes_evaluated - num_nodes_to_eval
        # Get the residual evaluations of the last mesh step
        residual_nodes: torch.Tensor | None = None
        residual_f_evals: torch.Tensor | None = None
        residual_tracked: list[torch.Tensor] | None = None
        residual_mesh_idx: torch.Tensor | None = None
        if num_residual_nodes > 0:
            residual_nodes = nodes_flat[-self.C : -self.C + num_residual_nodes]
            nodes_flat = nodes_flat[: -self.C]
            residual_f_evals = f_eval_batches[-1][-num_residual_nodes:]
            f_eval_batches[-1] = f_eval_batches[-1][:-num_residual_nodes]
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
            assert split_nodes is not None
            assert split_f_evals is not None
            nodes_flat = torch.concatenate([split_nodes, nodes_flat], dim=0)
            f_evals = torch.concatenate([split_f_evals, *f_eval_batches], dim=0)
        else:
            f_evals = torch.concatenate(f_eval_batches, dim=0)

        # Combine tracked variables the same way (prepending the carried split).
        tracked_combined: list[torch.Tensor] | None = None
        if tracked_lists is not None:
            tracked_combined = []
            for k, tl in enumerate(tracked_lists):
                if split_mesh_idx is not None:
                    # Tracked variables were carried with the split whenever
                    # f emits them (every batch emits the same set).
                    assert split_tracked is not None
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

    def _prune_excess_mesh(
        self,
        nodes: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        mesh_quadrature_errors: torch.Tensor,
        error_ratios_2steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge pairs of neighboring subintervals whose union is predicted to pass.

        Pair i (subintervals i and i + 1) is merged when its summed error ratio
        ``S = error_ratios_2steps[i]`` satisfies
        ``S < remove_cut / 2**(k - 1)``, with ``k = method.error_exponent``
        (padaquad's own derivation):

        1. Two neighboring subintervals of width h have error ratios that sum
           to S, so each has a ratio of about S / 2.
        2. The error estimate scales as ``h**k``, so the merged subinterval of
           width 2h has an error about ``2**k`` times that of one half, and an
           error ratio of about ``2**k * S / 2 = 2**(k - 1) * S``.
        3. Requiring that predicted ratio to be below ``remove_cut`` gives the
           cutoff.

        The scaling ``h**k`` with ``k = d + 2``, for an error estimate whose
        less accurate method has degree d, is a derivation, not a cited
        result: that method integrates the degree-d Taylor polynomial of f
        exactly, so its error comes from the ``h**(d + 1)`` term integrated
        over a width h. :cite:p:`gonnet2012` shows this for Simpson's rule
        (Eqs. (4) and (5), Sec. 2.1) and, in footnote 20, treats the error
        estimate of an embedded pair as that of the lower-degree method. The
        prediction holds only when the subintervals are small enough for the
        leading Taylor term to dominate. When the error estimate is at
        rounding level (high k, tight tolerance, float32) it does not shrink
        as ``h**k``, so pairs rarely reach the cut.

        Of several adjacent qualifying pairs, ``_rec_remove`` keeps every other
        one, so no subinterval is merged twice; one call merges at most one
        level. The merge itself is ``_merge_excess_nodes`` (uniform or variable
        sampling).

        Parameters
        ----------
        nodes : torch.Tensor
            Nodes of each subinterval. Shape: [N, C, T].
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh_quadrature_errors : torch.Tensor
            Error estimate of each subinterval. Shape: [N, D].
        error_ratios_2steps : torch.Tensor
            Summed error ratio of each pair of neighboring subintervals,
            indexed by the pair's first subinterval. Shape: [N-1].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple ``(nodes, mesh_quadratures, mesh_quadrature_errors)`` with
            each merged pair replaced by one subinterval whose contribution
            and error are the pair's sums. Shapes: [N-R, C, T], [N-R, D],
            [N-R, D] for R merged pairs.
        """
        if len(error_ratios_2steps) == 0:
            return nodes, mesh_quadratures, mesh_quadrature_errors
        # Merge cutoff S < remove_cut / 2^(k-1), k = error_exponent; the
        # derivation (padaquad's own) is in the docstring. The merged
        # subinterval is never evaluated.
        merge_cut = self.remove_cut / 2 ** (self.method.error_exponent - 1)
        # Since error ratios encompasses 2 RK steps each neighboring element shares
        # a step, we cannot remove that same step twice and therefore remove the
        # first in pair of steps that it appears in
        ratio_idxs_cut = torch.where(self._rec_remove(error_ratios_2steps < merge_cut))[
            0
        ]  # Index for first interval of 2
        assert not torch.any(ratio_idxs_cut[:-1] + 1 == ratio_idxs_cut[1:])

        if len(ratio_idxs_cut) == 0:
            return nodes, mesh_quadratures, mesh_quadrature_errors

        return self._merge_excess_nodes(
            nodes, mesh_quadratures, mesh_quadrature_errors, ratio_idxs_cut
        )

    @contextlib.contextmanager
    def _integration_on_device(self, device: torch.device) -> Iterator[None]:
        """
        Temporarily run integration-device-bound operations on ``device``.

        ``_get_optimal_mesh`` (and the subclass node helpers it calls) allocate
        on ``self.device`` and read the method's tensors, which live on
        ``self.device``. To run the optimal-mesh computation on
        ``result_device``, so the large per-subinterval record tensors never
        have to be copied back to the integration device, this context manager
        sets ``self.device`` to ``device`` and moves the method's tensors there
        for the duration of the block, restoring both on exit. It does nothing
        when ``device`` already equals ``self.device``.

        Parameters
        ----------
        device : torch.device
            Device to run on for the duration of the block.
        """
        if device == self.device:
            yield
            return
        original_device = self.device
        self.device = device
        self.method.to_device(device)
        try:
            yield
        finally:
            self.device = original_device
            self.method.to_device(original_device)

    def _get_optimal_mesh(
        self,
        nodes: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        mesh_quadrature_errors: torch.Tensor,
        integral: torch.Tensor,
        mesh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the optimal mesh from the results of the integration loop.

        The optimal mesh is the starting mesh of a later call with
        ``same_f=True``. It is built in two stages:

        1. Merge: pairs of neighboring subintervals whose merged subinterval is
           predicted to have an error ratio below ``remove_cut`` are merged
           (``_prune_excess_mesh``).
        2. Split: subintervals whose error ratio against the final integral is
           still 1 or above get a midpoint. A merged subinterval carries the
           pair's summed error, so its ratio is the pair's (below the merge
           cutoff, so below 1) unless the rounding floor sets the tolerance
           (see ``_error_ratios_norm``): this stage does not split a merged
           subinterval, and it does not check the merge prediction.
           Subintervals fail here only when no re-check ran against the final
           integral (``take_gradient=True``, or a fixed
           ``error_integral_reference``) or when ``max_adaptive_splits``
           stopped their refinement.

        All inputs must share a device, and the returned mesh is on that
        device. (The caller runs this on ``result_device`` through
        ``_integration_on_device`` and moves the result back to the
        integration device.)

        Parameters
        ----------
        nodes : torch.Tensor
            Nodes of each subinterval. Shape: [N, C, T].
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh_quadrature_errors : torch.Tensor
            Error estimate of each subinterval. Shape: [N, D].
        integral : torch.Tensor
            The final integral. Shape: [D].
        mesh : torch.Tensor
            The mesh at the end of the loop. Shape: [M, T].

        Returns
        -------
        torch.Tensor
            The optimal mesh. Shape: [M_opt, T].
        """
        # Prune steps with excess accuracy (over-resolved regions)
        _, error_ratios_2steps, _ = self._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors,
            mesh_quadratures=mesh_quadratures,
            integral=integral.detach(),
        )
        nodes_pruned, mesh_quadratures_pruned, mesh_quadrature_errors_pruned = (
            self._prune_excess_mesh(
                nodes,
                mesh_quadratures,
                mesh_quadrature_errors,
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
                integral=integral.detach(),
            )
        )
        keep_mask, remove_mask, _ = self._accept_reject_masks(
            error_ratios, error_ratios_per_dim
        )
        adaptive_step = self._adaptively_increase_mesh(
            method_output=None,
            error_ratios=error_ratios,
            y_step_eval=None,
            nodes=None,
            mesh=mesh_pruned,
            mesh_idxs=torch.arange(len(mesh_pruned) - 1, device=mesh_pruned.device),
            mesh_trackers=torch.zeros(
                len(mesh_pruned), dtype=torch.bool, device=mesh_pruned.device
            ),
            keep_mask=keep_mask,
            remove_mask=remove_mask,
        )
        _, _, _, _, mesh_optimal, _, _, _, _ = adaptive_step

        return mesh_optimal

    def _rec_remove(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Ensure no two adjacent True values exist in a boolean mask.

        Entry i flags the pair of subintervals i and i + 1. Two adjacent
        flagged pairs share a subinterval, so both cannot be merged. Conflicts
        are resolved from left to right: the first flagged entry of a run is
        kept and its neighbor unflagged, so a run of Trues alternates
        (TTT -> TFT, TTTT -> TFTF).

        Example: [True, True, False, True] -> [True, False, False, True]

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask, True for a pair flagged for merging. Shape: [N].

        Returns
        -------
        torch.Tensor
            Modified mask with no adjacent True values. Shape: [N].
        """
        # Iterative form of what was a tail recursion: each pass un-flags the
        # second element of an adjacent True pair and rebuilds the mask, looping
        # until no adjacent pair remains. The recursive version overflowed
        # Python's stack on the large, finely-refined meshes the accepted-panel
        # re-check can produce; this loop is behaviourally identical (the prior
        # body's only recursive call was the final `return self._rec_remove(...)`)
        # but has no depth limit.
        while True:
            mask2 = mask[:-1] * mask[1:]
            if not torch.any(mask2):
                return mask

            # Must keep the first integration step
            if mask2[0]:
                mask[1] = False

            # Mask is too small to remove points
            if len(mask) <= 2:
                return mask

            mask = torch.concatenate(
                [mask[:2], mask2[1:] * mask[:-2] + (~mask2[1:]) * mask[2:]]
            )

    def _setup_integral_bounds(
        self,
        mesh: torch.Tensor | None,
        mesh_init: torch.Tensor | None,
        mesh_final: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve the integration bounds of a call in the solver's dtype and device.

        With a mesh, the bounds are its first and last points; a given
        ``mesh_init`` or ``mesh_final`` must match them. Without one, a missing
        bound falls back to the constructor's value.

        Parameters
        ----------
        mesh : torch.Tensor or None
            The caller's mesh. Shape: [M, T].
        mesh_init : torch.Tensor or None
            The caller's lower bound. Shape: [T].
        mesh_final : torch.Tensor or None
            The caller's upper bound. Shape: [T].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mesh_init, mesh_final)``, each of shape [T], with
            ``mesh_init < mesh_final`` asserted.
        """
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
        mesh: torch.Tensor | None,
        mesh_init: torch.Tensor,
        mesh_final: torch.Tensor,
        same_f: bool,
        random_initial_mesh: bool,
        N_init_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[float, ...], int]]:
        """
        Build the starting mesh of an integration.

        In order of precedence: the stored ``mesh_previous`` when ``same_f`` is
        True (restricted to [mesh_init, mesh_final], with both bounds added if
        missing), the caller's mesh, or a fresh mesh of about N_init_steps
        points: sqrt(N_init_steps) equal segments, each holding
        sqrt(N_init_steps) + 1 mesh points, placed at random or evenly.

        Parameters
        ----------
        mesh : torch.Tensor or None
            The caller's mesh. Shape: [M, T].
        mesh_init : torch.Tensor
            Lower bound. Shape: [T].
        mesh_final : torch.Tensor
            Upper bound. Shape: [T].
        same_f : bool
            If True and a mesh is stored, start from ``mesh_previous``.
        random_initial_mesh : bool
            If True, place the mesh points inside each segment at random;
            otherwise evenly.
        N_init_steps : int
            Approximate number of mesh points of a fresh mesh.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, dict[tuple[float, ...], int]]
            The mesh [M, T]; ``mesh_trackers`` [M], True at every mesh point
            but the last (every subinterval needs evaluation); and the
            ``{tuple(mesh point): index}`` map of ``_get_mesh_indices``.
        """
        if same_f and self.mesh_previous is not None:
            # The stored mesh replaces the caller's mesh.
            # Filter cached barriers to within the new [mesh_init, mesh_final]
            # box, comparing every time dimension (min/max handles either
            # integration direction). For 1-D forward integration this is the
            # same set as the previous coordinate-0 range test.
            lo = torch.minimum(mesh_init, mesh_final)
            hi = torch.maximum(mesh_init, mesh_final)
            mask = torch.all(
                (self.mesh_previous >= lo) & (self.mesh_previous <= hi), dim=-1
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
        elif mesh is not None:
            # The user-provided mesh enters the loop here; _setup_integral_bounds
            # only moved a local copy for its assertions, so coerce it onto the
            # solver's device (DistributedEnvironment owns the device; inputs are
            # moved to it). Without this, the loop indexes a CPU mesh with
            # device-side indices and crashes on GPU.
            mesh = mesh.to(self.dtype).to(self.device)
        else:
            # Generate a fresh initial mesh of barriers across [mesh_init, mesh_final].
            # Layout: sqrt(N_init_steps) evenly-spaced top-level segments, each
            # subdivided into sqrt(N_init_steps)+1 sub-barriers. The total
            # barrier count is ~N_init_steps. Sub-barriers are placed
            # randomly (default) or uniformly within each segment.
            # A Python int (computed on the CPU, as before) so it can size
            # torch.arange / torch.rand.
            N_even_t = int(
                torch.sqrt(torch.tensor(N_init_steps, dtype=torch.float)).to(torch.int)
            )
            dt = (mesh_final - mesh_init) / N_even_t
            # Top-level segment starts: mesh_init + k*dt, points along the
            # mesh_init -> mesh_final line (dt is the [T] segment vector, so
            # this is correct for any time dimensionality).
            mesh = (
                mesh_init
                + dt * torch.arange(N_even_t, device=self.device)[:, None, None]
            )

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
                # Fractions in [0, 1) along the segment vector dt (shape [T]),
                # so sub-barriers stay on the line for any T. For T == 1 this is
                # identical to the previous dt * arange(n_sub) / n_sub.
                fractions = (
                    torch.arange(n_sub, dtype=self.dtype, device=self.device) / n_sub
                )
                offsets = fractions[:, None] * dt[None, :]  # [n_sub, T]
                mesh = mesh + offsets[None]
            # Enforce exact start and end points
            mesh[0] += mesh_init - mesh[0, 0]
            mesh[-1] += mesh_final - mesh[-1, -1]
            # Flatten segments into a single sorted barrier array
            mesh = torch.flatten(mesh, start_dim=0, end_dim=1)
            mesh[0] = mesh_init
            mesh[-1] = mesh_final
            # Any-T: consecutive barriers are distinct (no zero-length panels).
            assert torch.all(torch.any(mesh[1:] != mesh[:-1], dim=-1))
            # 1-D: additionally strictly increasing.
            if mesh.shape[-1] == 1:
                assert torch.all(mesh[1:] - mesh[:-1] > 0)
        mesh_trackers = torch.ones(len(mesh), device=self.device).to(torch.bool)
        mesh_trackers[-1] = False  # mesh_final cannot be a step starting point
        mesh_indices = self._get_mesh_indices(mesh)

        return mesh, mesh_trackers, mesh_indices

    def _get_mesh_indices(self, mesh: torch.Tensor) -> dict[tuple[float, ...], int]:
        """
        Map each mesh point to its position in the mesh.

        Parameters
        ----------
        mesh : torch.Tensor
            The ordered mesh. Shape: [M, T].

        Returns
        -------
        dict[tuple[float, ...], int]
            ``{tuple(mesh point): index}``, keyed by coordinates so a
            subinterval's position can be found from its left mesh point.
        """
        # Map each barrier to its position in the ordered mesh. Keyed on the
        # full barrier coordinate as a hashable tuple (not the tensor object,
        # which hashes by identity and so cannot be looked up by value) so the
        # recording code can recover a panel's order from its left barrier for
        # any time dimensionality T.
        # Single device->host transfer of the whole [M, T] mesh, then build the
        # tuple keys in pure Python (one host sync instead of one per barrier).
        return {tuple(row): idx for idx, row in enumerate(mesh.tolist())}

    # -------------------------------------------------------------------------------- #
    #                           ADAPTIVE ERROR CALCULATIONS                            #
    # -------------------------------------------------------------------------------- #

    def _reduce_norm(
        self,
        x: torch.Tensor,
        error_norm: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Reduce the last (output, D) axis of ``x`` with the configured norm.

        The ``"2"`` and ``"max"`` options are the two named norms of the
        ``norm`` argument of ``scipy.integrate.quad_vec``
        :cite:p:`scipy_quadvec` (v1.18.1, line 115 and lines 288-295):

        - ``"2"``: the Euclidean norm ``sqrt(sum(x**2))`` (scipy uses
          ``np.linalg.norm``).
        - ``"max"``: the largest absolute value ``max(abs(x))`` (scipy's
          ``_max_norm``).
        - ``"rms"``: the root-mean-square ``sqrt(mean(x**2))``; scipy has no
          such option.
        - a callable: ``error_norm(x)``, which must reduce the last axis.

        For D = 1 every string option equals ``torch.abs(x)``.

        Parameters
        ----------
        x : torch.Tensor
            Per-dimension values. Shape: [..., D].
        error_norm : str | Callable[[torch.Tensor], torch.Tensor] | None
            Override for the configured norm (defaults to
            ``self.error_norm``). ``"failure_fraction"`` is not a norm and
            must not be passed here.

        Returns
        -------
        torch.Tensor
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
        Compute the rounding floor of each element of each subinterval's contribution.

        The floor is ``50 * eps * abs(s_k)``, with ``eps`` the machine epsilon
        of the solver's dtype and ``s_k`` the subinterval's signed contribution
        (``mesh_quadratures``). The error-ratio helpers use it as a lower bound
        on the tolerance, so a subinterval whose error is already at rounding
        level is accepted instead of being split forever.

        It is modeled on the rounding error of ``scipy.integrate.quad_vec``
        :cite:p:`scipy_quadvec` (``_quadrature_gk``, v1.18.1 lines 539-543),
        which differs in three ways:

        - scipy computes ``round_err = norm(50 * eps * h * s_k_abs)``, with
          ``h * s_k_abs`` the subinterval's quadrature of ``abs(f)``; padaquad
          uses the absolute value of the signed contribution, which is smaller
          where f changes sign.
        - scipy raises the error estimate to the floor,
          ``err = max(err, round_err)``, and only when ``round_err`` exceeds the
          smallest positive double; padaquad raises the tolerance instead and
          has no such underflow guard.
        - scipy first rescales the error estimate as QUADPACK does,
          ``resasc * min(1, (200 * err / resasc)**1.5)`` (:cite:p:`quadpack_netlib`,
          dqk21.f lines 177-178); padaquad does not.

        Parameters
        ----------
        mesh_quadratures : torch.Tensor
            Contribution of each subinterval. Shape: [N, D].

        Returns
        -------
        torch.Tensor
            Rounding floor of each element. Shape: [N, D].
        """
        eps = torch.finfo(self.dtype).eps
        return 50.0 * eps * torch.abs(mesh_quadratures)

    def _compute_error_ratios(
        self,
        mesh_quadrature_errors: torch.Tensor,
        mesh_quadratures: torch.Tensor | None = None,
        cum_mesh_quadratures: torch.Tensor | None = None,
        integral: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Compute the error ratio of each subinterval, choosing by ``self.error_norm``.

        The error estimate is the difference between the results of the method
        and of its embedded lower-degree method. How the D output components
        become one accept-or-split quantity depends on the scheme:

        - Norm family (``"2"``, ``"max"``, ``"rms"`` or a callable): reduce the
          error vector with the norm, then compare it to one tolerance, in
          ``_error_ratios_norm``.
        - ``"failure_fraction"``: compare each component to its own tolerance,
          in ``_error_ratios_failure_fraction``.

        Absolute versus cumulative mode (``use_absolute_error_ratio``) only
        changes the tolerance and is handled inside each of the two helpers.

        Parameters
        ----------
        mesh_quadrature_errors : torch.Tensor
            Error estimate of each subinterval. [N, D].
        mesh_quadratures : torch.Tensor or None
            Contribution of each subinterval (``s_k``), used for the rounding
            floor and, in cumulative mode, the running integral. [N, D].
        cum_mesh_quadratures : torch.Tensor or None
            Precomputed running integral through each subinterval [N, D]
            (cumulative mode only).
        integral : torch.Tensor or None
            Current total integral [D] (absolute mode only; required there).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
            Tuple ``(error_ratios, error_ratios_2steps, error_ratios_per_dim)``:

            - error_ratios: accept-or-split quantity of each subinterval [N]
              (the reduced ratio for the norm family; the failure fraction for
              ``"failure_fraction"``).
            - error_ratios_2steps: merge indicator of each pair of neighbors
              [N-1].
            - error_ratios_per_dim: ratio of each element [N, D] for
              ``"failure_fraction"`` (so non-finite subintervals can be
              found), else None.

        Raises
        ------
        ValueError
            If ``integral`` is missing in absolute mode, or both
            ``mesh_quadratures`` and ``cum_mesh_quadratures`` are missing
            in cumulative mode.
        """
        if self.use_absolute_error_ratio and integral is None:
            raise ValueError("Must give integral when use_absolute_error_ratio=True")
        abs_err = torch.abs(mesh_quadrature_errors)
        if self._is_failure_fraction:
            return self._error_ratios_failure_fraction(
                abs_err, mesh_quadratures, cum_mesh_quadratures, integral
            )
        return self._error_ratios_norm(
            abs_err, mesh_quadratures, cum_mesh_quadratures, integral
        )

    def _error_ratios_norm(
        self,
        abs_err: torch.Tensor,
        mesh_quadratures: torch.Tensor | None,
        cum_mesh_quadratures: torch.Tensor | None,
        integral: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Error ratios of the norm family: reduce the error vector, then compare.

        The error vector is reduced to a number with the configured norm R and
        compared to one tolerance built from the same norm of the integral::

            absolute:   error_ratio[k] = R(abs(e[k])) / max(atol, rtol * R(I))
            cumulative: error_ratio[k] = R(abs(e[k])) / max(atol, rtol * R(abs(cum[k])))

        where ``cum[k]`` is the running integral through subinterval k. A
        subinterval is accepted when ``error_ratio < 1`` (see
        ``_accept_reject_masks``).

        Relation to ``scipy.integrate.quad_vec`` :cite:p:`scipy_quadvec`
        (v1.18.1): the tolerance ``max(epsabs, epsrel * norm(I))`` is the same
        (lines 373 and 409), but the acceptance rule differs. quad_vec stops
        when the sum of the errors of all subintervals is below ``tol / 8``
        (line 410). padaquad accepts or splits each subinterval by its own
        error ratio.

        The pair ratio ``error_ratio_2steps[k] = R(abs(e[k]) + abs(e[k+1])) /
        tol_2steps[k]`` (used only to merge subintervals, in
        ``_prune_excess_mesh``) uses the tolerance the merged subinterval
        would get: the same ``tol`` in absolute mode, subinterval k+1's (the
        merged subinterval's right end) in cumulative mode. Its rounding floor
        is ``R(floor[k]) + R(floor[k+1])``, at least the merged subinterval's
        own floor (equal when both contributions have the same sign).

        The rounding floor (``_round_floor``) is a lower bound on the
        tolerance, ``effective_tol = max(tol, R(floor))``; the error itself is
        not raised. A subinterval whose error is at or below rounding level
        then has an error ratio of about 1 or less and is accepted instead of
        being split forever, which matters when the requested tolerance is
        below what the dtype can resolve.

        Parameters
        ----------
        abs_err : torch.Tensor
            Absolute error estimate of each subinterval. Shape: [N, D].
        mesh_quadratures : torch.Tensor | None
            Contribution of each subinterval (for the rounding floor). Shape:
            [N, D].
        cum_mesh_quadratures : torch.Tensor | None
            Running integral through each subinterval (cumulative mode).
            Shape: [N, D].
        integral : torch.Tensor | None
            Total integral (absolute mode). Shape: [D].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, None]
            ``(error_ratio, error_ratio_2steps, None)``, shapes [N], [N-1].
        """
        err = self._reduce_norm(abs_err)
        # Tolerances follow the inputs' device so this runs on the integration
        # device in the loop and on result_device in the post-convergence pass.
        atol = self.atol.to(abs_err.device)
        rtol = self.rtol.to(abs_err.device)
        floor = (
            self._reduce_norm(self._round_floor(mesh_quadratures))
            if mesh_quadratures is not None
            else None
        )

        if self.use_absolute_error_ratio:
            assert integral is not None  # checked in _compute_error_ratios
            error_tol = torch.maximum(atol, rtol * self._reduce_norm(integral))
            error_tol_2steps = error_tol
        else:
            if cum_mesh_quadratures is not None:
                cum_steps = cum_mesh_quadratures
            elif mesh_quadratures is not None:
                cum_steps = torch.cumsum(mesh_quadratures, dim=0)
            else:
                raise ValueError("Must give mesh_quadratures or cum_mesh_quadratures")
            error_tol = torch.maximum(
                atol, rtol * self._reduce_norm(torch.abs(cum_steps))
            )
            # A merged pair (i, i+1) ends where step i+1 ends, so it is judged
            # against step i+1's cumulative tolerance (not the larger of the
            # two, which under-predicts the merged ratio when |cum| falls).
            error_tol_2steps = error_tol[1:]

        if floor is not None:
            # Broadcasts a scalar (absolute mode) or per-step (cumulative) tol.
            error_tol = torch.maximum(error_tol, floor)
            error_tol_2steps = torch.maximum(error_tol_2steps, floor[:-1] + floor[1:])

        error_ratio = err / error_tol
        err_2steps = self._reduce_norm(abs_err[:-1] + abs_err[1:])
        error_ratio_2steps = err_2steps / error_tol_2steps
        return error_ratio, error_ratio_2steps, None

    def _error_ratios_failure_fraction(
        self,
        abs_err: torch.Tensor,
        mesh_quadratures: torch.Tensor | None,
        cum_mesh_quadratures: torch.Tensor | None,
        integral: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Error ratios of ``"failure_fraction"``: the fraction of failing components.

        Each output component has its own tolerance, and a subinterval's
        accept-or-split quantity is the fraction of components whose ratio is
        1 or above::

            r[k, d] = abs(e[k, d]) / (atol + rtol * abs(I[d]))        (absolute)
            r[k, d] = abs(e[k, d]) / (atol + rtol * abs(cum[k, d]))   (cumulative)
            failure_fraction[k] = mean over d of (r[k, d] >= 1)

        A non-finite ratio does not count as failing here. A subinterval is
        accepted when ``failure_fraction <= mesh_failure_tolerance`` (in
        ``_accept_reject_masks``). The ratio of each element is also returned,
        so non-finite (NaN or Inf) subintervals can be found later. The pair
        ratio (used only to merge subintervals) is the largest over components
        of ``(abs(e[k, d]) + abs(e[k+1, d])) / tol_2steps[k, d]``, with the
        merged subinterval's tolerance as in ``_error_ratios_norm``.

        The rounding floor is a lower bound on each component's tolerance,
        ``effective_tol = max(tol, floor)``, so a component whose error is at
        or below rounding level does not count as failing.

        Parameters
        ----------
        abs_err : torch.Tensor
            Absolute error estimate of each subinterval. Shape: [N, D].
        mesh_quadratures : torch.Tensor | None
            Contribution of each subinterval (for the rounding floor). Shape:
            [N, D].
        cum_mesh_quadratures : torch.Tensor | None
            Running integral through each subinterval (cumulative mode).
            Shape: [N, D].
        integral : torch.Tensor | None
            Total integral (absolute mode). Shape: [D].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(failure_fraction, error_ratio_2steps, ratio)``, shapes [N],
            [N-1], [N, D].
        """
        # Tolerances follow the inputs' device (see _error_ratios_norm).
        atol = self.atol.to(abs_err.device)
        rtol = self.rtol.to(abs_err.device)
        if self.use_absolute_error_ratio:
            assert integral is not None  # checked in _compute_error_ratios
            error_tol = atol + rtol * torch.abs(integral)  # [D]
            error_tol_2steps = error_tol
        else:
            if cum_mesh_quadratures is not None:
                cum_steps = cum_mesh_quadratures
            elif mesh_quadratures is not None:
                cum_steps = torch.cumsum(mesh_quadratures, dim=0)
            else:
                raise ValueError("Must give mesh_quadratures or cum_mesh_quadratures")
            error_tol = atol + rtol * torch.abs(cum_steps)  # [N, D]
            # Merged pair's tolerance: step i+1's (see _error_ratios_norm).
            error_tol_2steps = error_tol[1:]

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

    @property
    def _is_failure_fraction(self) -> bool:
        """
        Whether ``error_norm`` is the per-component ``"failure_fraction"``.

        Returns
        -------
        bool
            True for ``"failure_fraction"``, False for a norm or a callable.
        """
        return (not callable(self.error_norm)) and self.error_norm == "failure_fraction"

    def _accept_reject_masks(
        self, error_ratios: torch.Tensor, error_ratios_per_dim: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Turn the error ratios into accept (keep) and split (remove) masks.

        Both families accept a subinterval whose error ratio is non-finite (NaN
        or Inf). Splitting cannot be relied on to remove a non-finite value (a
        NaN or Inf at a mesh point comes back as an endpoint of a half, and a
        non-finite running integral makes every tolerance non-finite), so
        splitting until it is gone might never end.

        - ``"failure_fraction"``: keep when the failure fraction is at most
          ``mesh_failure_tolerance`` (plus ``8 * eps`` for rounding), or when
          any component of the subinterval is non-finite.
        - Norm family: keep when ``error_ratios < 1`` or when it is
          non-finite.

        Parameters
        ----------
        error_ratios : torch.Tensor
            Accept-or-split quantity of each subinterval. Shape: [N].
        error_ratios_per_dim : torch.Tensor | None
            Ratio of each element [N, D] for ``"failure_fraction"`` (used for
            the non-finite check), else None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple ``(keep_mask, remove_mask, nonfinite_mask)``, each boolean of
            shape [N]. ``nonfinite_mask`` marks the non-finite subintervals,
            which are kept (the final ``converged`` check counts them as
            failing).
        """
        if self._is_failure_fraction:
            if error_ratios_per_dim is None:
                raise ValueError(
                    'error_norm="failure_fraction" needs the per-element ratios '
                    "returned by _compute_error_ratios"
                )
            nonfinite_mask = (~torch.isfinite(error_ratios_per_dim)).any(dim=-1)
            eps_frac = 8.0 * torch.finfo(self.dtype).eps
            passes = error_ratios <= self.mesh_failure_tolerance + eps_frac
        else:
            nonfinite_mask = ~torch.isfinite(error_ratios)
            passes = error_ratios < 1.0
        keep_mask = passes | nonfinite_mask
        return keep_mask, ~keep_mask, nonfinite_mask

    # -------------------------------------------------------------------------------- #
    #                                    RECORDING                                     #
    # -------------------------------------------------------------------------------- #

    # Record dict keys that are cumulative scalars (sum across batches),
    # not per-step arrays that need re-sorting.
    _RECORD_SCALAR_KEYS = ("integral", "integral_error", "loss")

    # Record dict keys that must stay on the integration device because they are
    # read back during the loop (running integral and the per-step quadratures
    # merged + cumsum'd for the error ratios; the per-step errors are re-checked
    # against the drifting tolerance every iteration -- see the accepted-panel
    # re-check in ``integrate``). Everything else is detached and moved to
    # ``result_device`` as it is recorded, to keep the large per-step tensors
    # (nodes, y, tracked_variables, ...) off the GPU.
    _DEVICE_RESIDENT_KEYS = ("integral", "mesh_quadratures", "mesh_quadrature_errors")

    # Cumulative scalar keys whose final output stays on the integration device.
    # The integral and its error estimate are the primary answer, so they remain
    # on the compute device even when the bulky per-step arrays are offloaded to
    # ``result_device``. (``mesh_quadratures`` is resident only for loop reads
    # and is moved to ``result_device`` after convergence.)
    _RESULT_INTEGRATION_KEYS = ("integral", "integral_error")
    #: Per-step IntegrationResult fields kept in the record (besides the
    #: scalars in ``_RECORD_SCALAR_KEYS``).
    _RECORD_STEP_KEYS = (
        "nodes",
        "h",
        "y",
        "mesh_quadratures",
        "mesh_quadrature_errors",
        "error_ratios",
    )
    #: Tracked variable i is recorded under ``f"{_TRACKED_KEY_PREFIX}{i}"``, so
    #: every record value is a per-step tensor or a scalar accumulator.
    _TRACKED_KEY_PREFIX = "tracked_variables_"

    def _record_fields(self, results: IntegrationResult) -> dict[str, torch.Tensor]:
        """
        Map a batch result to the record's keys.

        Parameters
        ----------
        results : IntegrationResult
            IntegrationResult of one accepted batch; ``loss`` must be set.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{record key: tensor}`` for the scalar keys, the per-subinterval
            keys and one key per tracked variable.
        """
        assert results.loss is not None, "loss must be set before recording"
        fields: dict[str, torch.Tensor] = {
            key: getattr(results, key)
            for key in self._RECORD_SCALAR_KEYS + self._RECORD_STEP_KEYS
        }
        for i, tracked in enumerate(results.tracked_variables or ()):
            fields[f"{self._TRACKED_KEY_PREFIX}{i}"] = tracked
        return fields

    def _mesh_order(
        self, barriers: torch.Tensor, mesh_indices: dict[tuple[float, ...], int]
    ) -> torch.Tensor:
        """
        Look up the mesh index of each subinterval from its left mesh point.

        A subinterval's left mesh point is ``nodes[:, 0, :]``, its first node,
        which equals the mesh point bit for bit because every method's
        ``node_fractions()`` start at 0 (so ``mesh_left + 0 * (...) ==
        mesh_left`` exactly). ``mesh_indices`` maps each mesh point (keyed on
        its full coordinate tuple) to its index in the ordered mesh, so this
        gives an order for any number of coordinates T without comparing
        coordinates.

        Parameters
        ----------
        barriers : torch.Tensor
            Left mesh point of each subinterval. Shape: [N, T].
        mesh_indices : dict[tuple[float, ...], int]
            ``{tuple(mesh point): index}`` from ``_get_mesh_indices`` for the
            current mesh.

        Returns
        -------
        torch.Tensor
            Mesh index of each subinterval. Shape: [N], dtype long.
        """
        # Build keys from a SINGLE device->host transfer of the whole [N, T]
        # block (barriers.tolist()), then look each up in pure Python. Avoids
        # one host sync per panel, which matters on GPU for fine meshes.
        return torch.tensor(
            [mesh_indices[tuple(row)] for row in barriers.tolist()],
            dtype=torch.long,
            device=self.device,
        )

    def _merge_positions(
        self, rec_order: torch.Tensor, new_order: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute merged-array positions from two mesh-order arrays.

        Given the mesh-order indices of the existing record entries and of the
        new entries, return where each goes in the combined array in mesh
        order. The mesh indices are distinct integers, so the merge is a rank
        computation, with no comparison of coordinates.

        Parameters
        ----------
        rec_order : torch.Tensor
            Mesh positions of existing record entries. Shape: [N_record].
        new_order : torch.Tensor
            Mesh positions of new entries. Shape: [N_result].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (idxs_keep, idxs_input): positions in the merged array for
            the existing and the new entries respectively (consumed by
            ``_insert_sorted_results``).
        """
        orders = torch.cat([rec_order, new_order])
        positions = torch.empty(len(orders), dtype=torch.long, device=self.device)
        positions[torch.argsort(orders)] = torch.arange(len(orders), device=self.device)
        n = len(rec_order)
        return positions[:n], positions[n:]

    def _insert_sorted_results(
        self,
        record: torch.Tensor,
        record_idxs: torch.Tensor,
        result: torch.Tensor,
        result_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge new results into an existing record at precomputed positions.

        Creates a new tensor large enough for both, places existing entries at
        record_idxs and new entries at result_idxs.

        Parameters
        ----------
        record : torch.Tensor
            Existing recorded data. Shape: [N_record, ...].
        record_idxs : torch.Tensor
            Positions for existing data in merged array. Shape: [N_record].
        result : torch.Tensor
            New data to insert. Shape: [N_result, ...].
        result_idxs : torch.Tensor
            Positions for new data in merged array. Shape: [N_result].

        Returns
        -------
        torch.Tensor
            Merged tensor with both record and result entries in sorted order.
            Shape: [N_record + N_result, ...].
        """
        add_shape = (len(record) + len(result), *record.shape[1:])
        old_record = record.clone()
        # Index tensors come from _merge_positions on the integration device;
        # move them to the record's device so a record living on result_device
        # (CPU) merges with CPU indices while a GPU-resident record uses GPU
        # indices. ``result`` is already placed on record.device by the caller.
        record_idxs = record_idxs.to(record.device)
        result_idxs = result_idxs.to(record.device)
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
        mesh_indices: dict[tuple[float, ...], int],
        result_device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Add the accepted subintervals of a batch to the running record.

        On the first batch this creates the record. On later batches it
        inserts the new results in mesh order and adds to the integral, its
        error and the loss.

        Each field is placed on its device: the keys in
        ``_DEVICE_RESIDENT_KEYS`` and ``_RESULT_INTEGRATION_KEYS`` stay on the
        integration device (the first are read back during the loop), and all
        other fields, including the large ``nodes``, ``y`` and tracked
        variables, are moved to ``result_device`` so they do not accumulate on
        the GPU. When ``take_gradient`` is True the per-batch
        ``loss.backward()`` has already run, so tensors are detached first;
        when False the autograd graph is kept so the returned integral stays
        differentiable (the device move is itself differentiable).

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            Running record. An empty dict on the first call.
        take_gradient : bool
            If True, the per-batch backward already ran and tensors are
            detached before recording; if False the graph is kept.
        results : IntegrationResult
            Results of the accepted subintervals of the current batch.
        mesh_indices : dict[tuple[float, ...], int]
            ``{tuple(mesh point): index}`` for the current mesh, used to place
            the new subintervals in mesh order (for any T).
        result_device : str | torch.device | None
            Device for the large output tensors. None keeps everything on the
            integration device.

        Returns
        -------
        dict[str, torch.Tensor]
            Updated record dict with the new results merged in. Keys are the
            IntegrationResult field names, plus one key per tracked variable
            (see ``_record_fields``).
        """
        if result_device is None:
            result_device = self.device

        def target(key: str) -> str | torch.device:
            """
            Return the device a record field is kept on.

            Parameters
            ----------
            key : str
                The record field's name.

            Returns
            -------
            str or torch.device
                The integration device or ``result_device``.
            """
            # Resident keys (read back in the loop) and the integration-output
            # scalars stay on the integration device; everything else goes to
            # result_device.
            if (
                key in self._DEVICE_RESIDENT_KEYS
                or key in self._RESULT_INTEGRATION_KEYS
            ):
                return self.device
            return result_device

        def place(value: torch.Tensor, key: str) -> torch.Tensor:
            """
            Move a record field to its device, detached if take_gradient is True.

            Parameters
            ----------
            value : torch.Tensor
                The field's new values.
            key : str
                The record field's name.

            Returns
            -------
            torch.Tensor
                The value on the device ``target(key)`` gives.
            """
            # take_gradient=True does per-batch backward *before* recording, so
            # the graph is no longer needed and is detached. take_gradient=False
            # retains the full graph for a single backward through the returned
            # integral, so the grad-tracked tensor is kept (``.to`` is itself an
            # autograd op, so the device move stays differentiable).
            if take_gradient:
                value = value.detach()
            return value.to(target(key))

        fields = self._record_fields(results)
        if len(record) == 0:
            # Tracked variables are already detached at evaluation time, so
            # place() leaves them unchanged apart from the device move.
            for key, value in fields.items():
                record[key] = place(value, key)
            return record

        idxs_keep, idxs_input = self._merge_positions(
            self._mesh_order(record["nodes"][:, 0, :], mesh_indices),
            self._mesh_order(results.nodes[:, 0, :].detach(), mesh_indices),
        )
        if fields.keys() != record.keys():
            raise ValueError(
                "The integrand returned a different set of tracked variables "
                f"than in earlier batches: {sorted(fields)} vs {sorted(record)}."
            )
        for key, value in record.items():
            if key in self._RECORD_SCALAR_KEYS:
                record[key] = value + place(fields[key], key)
            else:
                record[key] = self._insert_sorted_results(
                    value,
                    idxs_keep,
                    place(fields[key], key),
                    idxs_input,
                )
        # General (any-T) invariant: the record is in strictly increasing mesh
        # order after the merge.
        merged_order = self._mesh_order(record["nodes"][:, 0, :], mesh_indices)
        assert torch.all(merged_order[1:] - merged_order[:-1] > 0)
        # For 1-D time the left-node coordinate must also be strictly ascending.
        if record["nodes"].shape[-1] == 1:
            assert torch.all(record["nodes"][1:, 0, 0] - record["nodes"][:-1, 0, 0] > 0)

        return record

    def _sort_record(
        self,
        record: dict[str, torch.Tensor],
        mesh_indices: dict[tuple[float, ...], int],
    ) -> dict[str, torch.Tensor]:
        """
        Sort the per-subinterval entries of the record by mesh order and check it.

        ``_record_results`` already inserts every batch in mesh order, so on
        the record ``integrate`` builds the sort leaves the order unchanged;
        this is a safeguard before the final output. The order is each
        subinterval's index in the mesh (its left mesh point looked up in
        ``mesh_indices``), which works for any number of coordinates T. The
        accumulated scalars (integral, its error, loss) are not reordered.

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            Record with per-subinterval tensors.
        mesh_indices : dict[tuple[float, ...], int]
            ``{tuple(mesh point): index}`` for the final mesh.

        Returns
        -------
        dict[str, torch.Tensor]
            The record with its per-subinterval tensors in mesh order.
        """
        sorted_idxs = torch.argsort(
            self._mesh_order(record["nodes"][:, 0, :], mesh_indices), dim=0
        )
        # Per-step fields may live on different devices (result_device vs the
        # integration device), so move the index to each field's device.
        for key, value in record.items():
            if key not in self._RECORD_SCALAR_KEYS:
                record[key] = value[sorted_idxs.to(value.device)]
        # General (any-T) invariant: strictly increasing mesh order.
        sorted_order = self._mesh_order(record["nodes"][:, 0, :], mesh_indices)
        assert torch.all(sorted_order[1:] - sorted_order[:-1] > 0)
        # For 1-D time the left-node coordinate must be strictly monotonic.
        if record["nodes"].shape[-1] == 1:
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

    def _remove_failed_record_panels(
        self,
        record: dict[str, torch.Tensor],
        record_keep_mask: torch.Tensor,
        loss_fxn: Callable[[IntegrationResult], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Remove from the record the subintervals that failed the re-check.

        They were just split in the mesh (their halves will be evaluated as
        new subintervals), so their entries must leave the record. The
        per-subinterval fields are filtered by ``record_keep_mask`` and the
        accumulated scalars are updated:

        - ``integral`` is updated exactly: it is the sum of the
          ``mesh_quadratures``, so subtracting the removed contributions is
          exact.
        - ``integral_error`` is a sum of signed errors, but the recorded
          per-subinterval errors are absolute values, so the removed absolute
          errors are subtracted; the result is an approximation of this
          diagnostic field.
        - ``loss`` with the default loss equals the integral, so it is set to
          the updated integral; a custom ``loss_fxn`` is left unchanged (its
          per-batch sum is already an approximation, and the re-check only
          runs with ``take_gradient=False``, so no per-batch backward depends
          on it).

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            Running record (per-subinterval fields in mesh order).
        record_keep_mask : torch.Tensor
            Boolean mask over the recorded subintervals; True keeps one.
        loss_fxn : Callable[[IntegrationResult], torch.Tensor]
            The active loss function (to detect the default loss).

        Returns
        -------
        dict[str, torch.Tensor]
            The record without the failed subintervals.
        """
        removed = ~record_keep_mask
        # mesh_quadratures / mesh_quadrature_errors are device-resident, as are
        # integral / integral_error; move the mask to each tensor's device.
        quad = record["mesh_quadratures"]
        err = record["mesh_quadrature_errors"]
        removed_quad_sum = quad[removed.to(quad.device)].sum(0)
        removed_err_sum = err[removed.to(err.device)].sum(0)
        record["integral"] = record["integral"] - removed_quad_sum.to(
            record["integral"].device
        )
        record["integral_error"] = record["integral_error"] - removed_err_sum.to(
            record["integral_error"].device
        )
        for key, value in record.items():
            if key not in self._RECORD_SCALAR_KEYS:
                record[key] = value[record_keep_mask.to(value.device)]
        # Default loss == integral, so keep it consistent with the rebuilt
        # value. Bound methods are not identity-stable (``obj.m is obj.m`` is
        # False), so compare the underlying function (the class attribute)
        # rather than the bound method object. A custom loss_fxn is left as-is
        # (its per-batch sum is already an approximation; the re-check only
        # runs with take_gradient=False, so no per-batch backward depends on it).
        is_default_loss = (
            getattr(loss_fxn, "__func__", None) is type(self)._integral_loss
        )
        if is_default_loss:
            # loss lives on result_device (see _record_results), integral on the
            # integration device; keep loss where the next batch's loss is added.
            record["loss"] = record["integral"].to(record["loss"].device)
        return record

    def _flatten_output(self, panel_array: torch.Tensor) -> torch.Tensor:
        """
        Join per-subinterval values into one sequence, dropping shared endpoints.

        Keeps all C entries of the first subinterval and drops the first entry
        of every later one, which repeats the previous subinterval's last entry
        (node fractions start at 0 and end at 1).

        Parameters
        ----------
        panel_array : torch.Tensor
            Values per subinterval and node, in mesh order. Shape: [N, C, ...].

        Returns
        -------
        torch.Tensor
            The joined values. Shape: [N * (C - 1) + 1, ...].
        """
        return torch.concatenate(
            [
                panel_array[0],
                torch.flatten(panel_array[1:, 1:], start_dim=0, end_dim=1),
            ],
            dim=0,
        )
