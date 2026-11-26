"""
Adaptive quadrature engine that evaluates many subintervals in parallel.

An ODE solver must advance one step at a time, because each step depends on
the previous result. Here the integrand ``f(t)`` depends only on the
integration variable ``t``, so the contribution of every subinterval is
independent of the others, and many subintervals are evaluated together in
one batch.

The mesh divides ``[mesh_init, mesh_final]`` into subintervals, and the
method places C nodes in each one. A subinterval whose error estimate exceeds
the tolerance is split in half (or in four, with ``split_rule="accelerate"``)
and the pieces are evaluated again. After the
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
import itertools
import logging
import math
import numbers
import operator
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from padaquad.base import DEFAULT_METHODS, SolverBase, steps
from padaquad.distributed import DistributedEnvironment, _resolve_device
from padaquad.methods import _get_method
from padaquad.quadrature.memory import (
    _GB,
    _GROWTH,
    _LARGEST_TEST_CALL,
    _OOM_BACKOFF,
    _check_evaluation_buffer,
    _is_out_of_memory,
    _MemoryManagement,
    _restart_after_backward_oom,
    _save_grad,
    _SavedGrads,
    _shrink_batch,
)
from padaquad.results import IntegrationResult, IntegrationState, MethodOutput

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from padaquad.methods import MethodClass

# Largest temporary (bytes) of the operations on integrand values done in
# blocks of rows: the weighted sum in ``_RK_integral`` and the selection of the
# accepted rows in ``_keep_rows``, so neither needs a second copy of all the
# values.
_BLOCK_BYTES = 2**27


def _RK_integral(
    h: torch.Tensor,
    y: torch.Tensor,
    tableau_b: torch.Tensor,
    y0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the method's weights to a batch of subintervals and sum the results.

    For every subinterval i, the integrand values are multiplied by the
    weights, summed over the C nodes and multiplied by the width ``h_i``:
    ``R_i = h_i * sum_j(b_ij * y_ij)``, and ``integral = y0 + sum_i(R_i)``.
    The function is used for every method, not only the Runge-Kutta ones.
    The products ``b_ij * y_ij`` are formed for blocks of subintervals of at
    most ``_BLOCK_BYTES`` (at least one subinterval), so the temporary
    stays small however large ``y`` is; each row of the result depends only on
    its own subinterval, so the blocks give the same bits as one sum.

    Parameters
    ----------
    h : torch.Tensor
        Width of each subinterval, the difference of its two mesh points (not
        of its first and last node: a method's nodes need not include the
        ends). Shape: [N, T].
    y : torch.Tensor
        Integrand values at the nodes. Shape: [N, C, D].
    tableau_b : torch.Tensor
        Weights of each node, from ``MethodClass.weights``. Shape: [N, C, 1].
    y0 : torch.Tensor
        Value added to the total. Shape: [D].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Tuple ``(integral, RK_steps)``: the total ``y0`` plus all
        contributions [D], and the contribution of each subinterval [N, D].
    """
    # Weighted sum over the C nodes (dim 1) of each subinterval, scaled by its
    # width, in blocks of rows (one block when N = 0, giving an empty sum).
    row_bytes = y.shape[1:].numel() * y.element_size()
    rows = max(1, _BLOCK_BYTES // max(1, row_bytes))
    sums = [
        torch.sum(tableau_b[i : i + rows] * y[i : i + rows], dim=1)
        for i in range(0, max(1, len(y)), rows)
    ]
    RK_steps = h * (sums[0] if len(sums) == 1 else torch.cat(sums))
    logger.debug("RK_steps shape=%s values=%s", RK_steps.shape, RK_steps)
    # Sum all step contributions to get the total integral
    integral = y0 + torch.sum(RK_steps, dim=0)
    return integral, RK_steps


def _keep_rows(x: torch.Tensor, idx: torch.Tensor, in_place: bool) -> torch.Tensor:
    """
    Return the rows ``idx`` of ``x``, in order.

    With ``in_place``, the kept rows are moved to the front of ``x`` and the
    first of them are returned, a view of ``x``: no second array as large as
    the kept rows is allocated, only blocks of at most ``_BLOCK_BYTES``.
    Kept row j comes from row ``idx[j] >= j`` (``idx`` is ascending), so a
    block never overwrites a row a later block reads. ``x`` is overwritten, so
    ``in_place`` is only for an array nothing else reads. Even then, an ``x``
    that requires grad is indexed (a copy), since its autograd graph may need
    the old rows, and so is a view of another tensor, such as f's own output
    kept as the pass's values when one call covers the pass (f may still hold
    it).

    Parameters
    ----------
    x : torch.Tensor
        Values of N subintervals. Shape: [N, ...].
    idx : torch.Tensor
        Positions of the rows to keep, ascending, on any device. Shape: [K].
    in_place : bool
        True to move the kept rows within ``x``; False to index ``x``.

    Returns
    -------
    torch.Tensor
        The kept rows. Shape: [K, ...].
    """
    idx = idx.to(x.device)
    if not in_place or x.requires_grad or x._base is not None:
        return x[idx]
    row_bytes = x.shape[1:].numel() * x.element_size()
    rows = max(1, _BLOCK_BYTES // max(1, row_bytes))
    for i in range(0, len(idx), rows):
        block = idx[i : i + rows]
        x[i : i + len(block)] = x[block]
    return x[: len(idx)]


def _midpoint(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """
    Return the mesh point that a split inserts between ``left`` and ``right``.

    The one formula of the midpoint: ``_splittable`` refuses a split whose
    midpoint rounds onto an end, which holds only if the mesh update inserts
    the same bits.

    Parameters
    ----------
    left : torch.Tensor
        Left ends of the subintervals. Shape: [N, T].
    right : torch.Tensor
        Right ends of the subintervals. Shape: [N, T].

    Returns
    -------
    torch.Tensor
        ``0.5 * (left + right)``. Shape: [N, T].
    """
    return 0.5 * (left + right)


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
       others in half (or in four, with ``split_rule="accelerate"``).
    5. Repeats steps 2 to 4 until every subinterval is accepted.
    6. Builds the optimal mesh, which a later call can start from by passing
       it as ``mesh``.

    Subclasses set ``sampling_type``, which selects the method registry.

    Attributes
    ----------
    method : MethodClass
        The quadrature method.
    C : int
        Number of nodes per subinterval.
    remove_cut : float
        Largest predicted error ratio of a subinterval made by merging two
        neighbors when the optimal mesh is built.
    merge_rule : str
        How that predicted error ratio is computed: ``"predicted"``,
        ``"power_law"`` or ``"sum"`` (see ``_merge_inputs``).
    init_max_batch : int or None
        The constructor's ``max_batch``.
    init_total_mem_usage : float
        The constructor's ``total_mem_usage``.
    limit_benchmark_time : bool
        The constructor's ``limit_benchmark_time``.
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
    # Weight vectors of _merged_pair_weights by quantized width ratio; reset by
    # ``_set_solver_dtype``.
    _merge_weight_cache: dict[int, torch.Tensor]

    def __init__(
        self,
        *,
        f: Callable[..., Any] | None = None,
        method: str | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        y0: torch.Tensor | None = None,
        remove_cut: float = 0.1,
        merge_rule: str = "predicted",
        max_batch: int | None = None,
        total_mem_usage: float = 0.9,
        limit_benchmark_time: bool = True,
        max_adaptive_splits: int | None = None,
        split_rule: str = "halve",
        use_absolute_error_ratio: bool = True,
        error_norm: str | Callable[[torch.Tensor], torch.Tensor] = "2",
        mesh_failure_tolerance: float = 0.0,
        runtime_checks: bool = True,
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
            (see :ref:`the user guide <ug-shapes>`). Default None.
        method : str or None
            Name of the quadrature method, a key of ``UNIFORM_METHODS`` (uniform
            solver) or ``VARIABLE_METHODS`` (variable solver); case does not
            matter. None (default) means ``"gk21"`` (uniform) or
            ``"interpolatory3_variable"`` (variable).
        atol : float
            Absolute tolerance, finite and >= 0. A subinterval is accepted when
            its error estimate is below ``max(atol, rtol * |I|)``, with ``I`` the
            integral (see :ref:`the user guide <ug-error-control>`). Default 1e-5.
        rtol : float
            Relative tolerance, finite and >= 0; see ``atol``. Default 1e-5.
        y0 : torch.Tensor or None
            Default value added to the returned integral, shape [D]. None means 0.
        remove_cut : float
            Largest predicted error ratio of a subinterval made by merging two
            neighbors when the optimal mesh is built; must be below 1. Lower
            values merge less. Default 0.1 (see :ref:`the user guide <ug-optimal-mesh>`).
        merge_rule : str
            How the merged subinterval's error is predicted when the optimal
            mesh is built (see :ref:`the user guide <ug-optimal-mesh>`):

            - ``"predicted"`` (default): computed from the integrand values
              already stored for the two neighbors, by interpolating them onto
              the merged subinterval's nodes. Use this one.
            - ``"power_law"``: the neighbors' summed error times
              ``2**(error_exponent - 1)``, the merge rule of PadaQuad 1.3.0
              and 1.4.0. It assumes the error shrinks like a power of the width
              and refuses many safe merges on wide subintervals.
            - ``"sum"``: the neighbors' summed error, the merge rule of
              PadaQuad 1.2.2. Merged subintervals often fail when evaluated
              again.

            The older two are kept to reproduce meshes of earlier versions.
        max_batch : int or None
            Default for the largest number of integrand evaluations per call of
            ``f``, used by calls given no ``max_batch`` and no ``state``; with
            ``take_gradient=True`` at least the C points of one subinterval.
            None (default): test calls of ``f`` find a batch size to start
            from, which then grows until ``f`` runs out of memory (see
            ``integrate`` and :ref:`the user guide <ug-memory>`).
        total_mem_usage : float
            Default fraction of the device's total memory, in (0, 1], that may
            be in use; see ``integrate``. Default 0.9.
        limit_benchmark_time : bool
            If True (default), the memory benchmark raises ``RuntimeError``
            when it cannot measure the memory of ``f`` within 60 s; False
            removes the limit, for an ``f`` so slow that the measurement needs
            longer (see :ref:`the user guide <ug-memory>`).
        max_adaptive_splits : int or None
            Default cap on how many times a subinterval may have been halved
            since the starting mesh; at the cap it is accepted even if it fails
            the tolerance. None (default) means no cap (see :ref:`the user guide <ug-convergence>`).
        split_rule : str
            Default for how a failing subinterval is split (see :ref:`the user
            guide <ug-four-pieces>`): ``"halve"`` (default) splits it in two;
            ``"accelerate"`` splits it in four when it is itself a piece of a
            split subinterval and its error ratio is at least 2**k, k the
            method's ``error_exponent`` (where its error estimate shrinks as
            h**k, both halves would fail again), which saves evaluations of f
            and passes on integrands that need many levels of refinement.
            ``"accelerate"`` with ``error_norm="failure_fraction"`` raises
            ``ValueError``.
        use_absolute_error_ratio : bool
            If True (default), the relative tolerance uses the whole integral;
            if False, the running integral from ``mesh_init`` through each
            subinterval (see :ref:`the user guide <ug-error-reference>`).
        error_norm : str or Callable
            How the error of a vector-valued integrand becomes one accept-or-split
            decision: ``"2"`` (default), ``"max"``, ``"rms"``,
            ``"failure_fraction"`` or a callable norm (see :ref:`the user guide <ug-error-norm>`).
        mesh_failure_tolerance : float
            With ``error_norm="failure_fraction"``, the fraction of output
            components, in [0, 1], that may fail while a subinterval is still
            accepted. Default 0.0 (every component must pass).
        runtime_checks : bool
            If True (default), check while integrating: raise ``ValueError``
            naming the points where ``f`` returned NaN or Inf, and check
            PadaQuad's own bookkeeping (that the accepted subintervals cover
            the final mesh once, each with its own width). If False, skip
            these checks to save time and memory; subintervals where ``f``
            returned NaN or Inf are then accepted and the non-finite value
            reaches the result. The mesh and the nodes are in order by
            construction either way.
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
            If an argument is passed by position, a keyword is not an option
            of the solver, or ``limit_benchmark_time``,
            ``use_absolute_error_ratio``, ``runtime_checks`` or
            ``output_speed_info`` is not a bool (Python or NumPy).
        ValueError
            If ``remove_cut`` is not below 1, ``merge_rule`` is not one of
            ``"predicted"``, ``"power_law"`` and ``"sum"``,
            ``split_rule="accelerate"`` is given with
            ``error_norm="failure_fraction"``, or another option is invalid.
        """
        unknown = set(distributed_options) - _DISTRIBUTED_OPTIONS
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument(s) "
                f"{sorted(unknown)}"
            )
        if not remove_cut < 1.0:
            raise ValueError(f"remove_cut must be below 1, got {remove_cut!r}")
        if merge_rule not in self._VALID_MERGE_RULES:
            raise ValueError(
                f"merge_rule must be one of {self._VALID_MERGE_RULES}, "
                f"got {merge_rule!r}"
            )
        self._check_bools(
            limit_benchmark_time=limit_benchmark_time,
            use_absolute_error_ratio=use_absolute_error_ratio,
            runtime_checks=runtime_checks,
            output_speed_info=output_speed_info,
        )
        super().__init__(
            method=DEFAULT_METHODS[self.sampling_type] if method is None else method,
            atol=atol,
            rtol=rtol,
            y0=y0,
            f=f,
            output_speed_info=output_speed_info,
            dtype=dtype,
            device=device,
            **distributed_options,
        )
        self._check_max_adaptive_splits(max_adaptive_splits)
        self.remove_cut = remove_cut
        self.merge_rule = merge_rule

        # Memory handling variables
        # Fitted memory of one call of f on B points, (a, b, d) of
        # a + b B + d B^2 in GB; set by each _setup_memory_checks run and used
        # by get_max_batch to size batches from live free memory.
        self.f_mem_coeffs = None
        # Cap discovered after an OOM during the memory benchmark of
        # get_max_batch(f=...); None until one occurs.
        self._oom_max_batch = None
        # The batch size of the running integrate() call (see _batch_for) and
        # whether it may still grow; set at the start of every call. The other
        # three are read by the check before a backward pass.
        self._batch, self._batch_converged = 1, True
        self._total_mem_usage = total_mem_usage
        self._forward_bytes, self._backward_ratio = 0.0, 1.0

        # Construction-time defaults; each integrate() call falls back to these
        # when its corresponding argument is None.
        self.init_max_batch = max_batch
        self.init_max_adaptive_splits = max_adaptive_splits
        self.init_split_rule = split_rule
        self.use_absolute_error_ratio = use_absolute_error_ratio

        # Construction-time defaults; the active values live in self.error_norm
        # and self.mesh_failure_tolerance and are (re)set each integrate() call.
        self._check_error_norm(error_norm)
        self._check_mesh_failure_tolerance(mesh_failure_tolerance)
        self.init_error_norm = error_norm
        self.error_norm = error_norm
        self._check_split_rule(split_rule)
        self.init_mesh_failure_tolerance = mesh_failure_tolerance
        self.mesh_failure_tolerance = mesh_failure_tolerance
        # Construction-time default; the active value lives in
        # self.runtime_checks and is (re)set each integrate() call.
        self.init_runtime_checks = runtime_checks
        self.runtime_checks = runtime_checks
        self._check_total_mem_usage(total_mem_usage)
        self.init_total_mem_usage = total_mem_usage
        self.limit_benchmark_time = limit_benchmark_time

    #: String values accepted by ``error_norm`` (in addition to a callable).
    _VALID_ERROR_NORMS = ("2", "max", "rms", "failure_fraction")
    #: Values accepted by ``merge_rule``.
    _VALID_MERGE_RULES = ("predicted", "power_law", "sum")

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

    @staticmethod
    def _check_bools(**options: object) -> None:
        """
        Validate options that must be True or False.

        A Python or NumPy bool passes. Anything else raises, including the
        values whose truth Python would take silently: a string such as
        ``"False"`` (true), None, 0 or ``[]`` (false), and a tensor (whose
        truth on a GPU is a read of the device). The options that also accept
        None (``runtime_checks`` and ``enforce_convergence`` of ``integrate``)
        are checked after None is replaced by its meaning.

        Parameters
        ----------
        **options : object
            The values to check, by option name.

        Raises
        ------
        TypeError
            If a value is not a bool, naming the option and the value.
        """
        for name, value in options.items():
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"{name} must be True or False, got "
                    f"{type(value).__name__} {value!r}"
                )

    #: Values accepted by ``split_rule``.
    _VALID_SPLIT_RULES = ("halve", "accelerate")

    def _check_split_rule(self, split_rule: object) -> None:
        """
        Validate ``split_rule`` against the active ``error_norm``.

        Called by the constructor and by ``integrate`` once ``self.error_norm``
        holds the value in force (the call's, else the constructor's).

        Parameters
        ----------
        split_rule : object
            The value to check.

        Raises
        ------
        ValueError
            If the value is not one of ``_VALID_SPLIT_RULES``, or if it is
            ``"accelerate"`` while ``error_norm`` is ``"failure_fraction"``.
        """
        if split_rule not in self._VALID_SPLIT_RULES:
            raise ValueError(
                f"split_rule must be one of {self._VALID_SPLIT_RULES}, "
                f"got {split_rule!r}"
            )
        if split_rule == "accelerate" and self._is_failure_fraction:
            raise ValueError(
                'split_rule="accelerate" splits a subinterval in four by its '
                'error ratio, which error_norm="failure_fraction" does not '
                "have (it compares the fraction of failing output components "
                'with mesh_failure_tolerance). Pass split_rule="halve", or a '
                'norm-family error_norm ("2", "max", "rms" or a callable).'
            )

    # -------------------------------------------------------------------------------- #
    #                        OPERATIONS ON THE QUADRATURE RULE                         #
    # -------------------------------------------------------------------------------- #

    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """
        Build the quadrature method at ``dtype`` on the solver's device.

        Called from ``SolverBase._set_dtype``, which runs at the end of
        ``SolverBase.__init__`` (the method's first build) and on every later
        dtype change. ``_get_method`` converts the named method of the registry of
        this solver's ``sampling_type``; the converted method is shared with every
        solver of the same method, device and dtype (``MethodClass.to``).

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
        self._merge_weight_cache = {}

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

        ``torch.lerp`` computes this as ``mesh_right - (1 - c_i) * (mesh_right -
        mesh_left)`` for ``c_i >= 0.5`` (``aten/src/ATen/native/Lerp.h``), so the
        nodes at c = 0 and c = 1 are the mesh points bit for bit. With the
        formula above, ``mesh_left + 1 * (mesh_right - mesh_left)`` is one rounding
        step past ``mesh_right`` when the difference rounds (float32, wide
        subintervals); the node then lay past the next subinterval's first node
        and outside the integration interval.

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
        return torch.lerp(
            mesh_left.unsqueeze(1),
            mesh_right.unsqueeze(1),
            self.method.node_fractions().unsqueeze(-1),
        )

    def _calculate_integral(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        mesh_left: torch.Tensor,
        mesh_right: torch.Tensor,
        y0: torch.Tensor | None = None,
    ) -> MethodOutput:
        """
        Compute the integral and the error estimate for a batch of subintervals.

        Runs the weighted sum twice: with the method's weights b for the
        integral, and with b_error for the error estimate (b_error is the
        difference between the weights of the method and of its embedded
        lower-degree method). The error estimates are detached from the
        autograd graph: they only drive the accept-or-split decisions, so no
        gradient flows through them and the graph they would hold is not kept.

        The widths come from the mesh points, not from the nodes, because a
        method's nodes need not include the ends of its subinterval. Pass the
        same mesh points the nodes were placed from (``_compute_nodes``), so
        that a bound that requires grad reaches the integral through both.

        Parameters
        ----------
        t : torch.Tensor
            Nodes of each subinterval. Shape: [N, C, T].
        y : torch.Tensor
            Integrand values at the nodes. Shape: [N, C, D].
        mesh_left : torch.Tensor
            Left mesh point of each subinterval. Shape: [N, T].
        mesh_right : torch.Tensor
            Right mesh point of each subinterval. Shape: [N, T].
        y0 : torch.Tensor or None
            Value added to the integral. Shape: [D]. None adds nothing (a
            zero of the solver's dtype and device).

        Returns
        -------
        MethodOutput
            The integral, its error estimate, and the contribution, error
            estimate and width of each subinterval.
        """
        if y0 is None:
            y0 = torch.zeros(1, device=self.device, dtype=self.dtype)
        b, b_error = self.method.weights(t)
        h = mesh_right - mesh_left
        integral, mesh_quadratures = _RK_integral(h, y, b, y0=y0)
        integral_error, mesh_quadrature_errors = _RK_integral(h, y, b_error, y0=y0)
        return MethodOutput(
            integral=integral,
            integral_error=integral_error.detach(),
            mesh_quadratures=mesh_quadratures,
            mesh_quadrature_errors=mesh_quadrature_errors.detach(),
            h=h,
        )

    # -------------------------------------------------------------------------------- #
    #                            PRIMARY INTEGRATION METHOD                            #
    # -------------------------------------------------------------------------------- #

    @_restart_after_backward_oom
    def integrate(
        self,
        f: Callable[..., Any] | None = None,
        y0: torch.Tensor | None = None,
        mesh: torch.Tensor | None = None,
        mesh_init: torch.Tensor | None = None,
        mesh_final: torch.Tensor | None = None,
        *,
        random_initial_mesh: bool = True,
        N_init_steps: int = 13,
        f_args: tuple[Any, ...] = (),
        take_gradient: bool = True,
        total_mem_usage: float | None = None,
        loss_fxn: Callable[[IntegrationResult], torch.Tensor] | None = None,
        max_batch: int | None = None,
        state: IntegrationState | None = None,
        parameters: Iterable[torch.Tensor] | torch.optim.Optimizer | None = None,
        max_adaptive_splits: int | None = None,
        split_rule: str | None = None,
        error_norm: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
        error_integral_reference: float | torch.Tensor | None = None,
        enforce_convergence: bool | None = None,
        mesh_failure_tolerance: float | None = None,
        runtime_checks: bool | None = None,
        trade_speed_for_memory: bool = False,
        evaluation_buffer: int | None = None,
        record_evaluations: bool = True,
        result_device: str | torch.device | None = "cpu",
        build_state_mesh: bool = True,
    ) -> tuple[IntegrationResult, IntegrationState]:
        """
        Integrate f over [mesh_init, mesh_final] with adaptive subintervals.

        The integrand is called as ``f(t, *f_args)`` with points ``t`` of shape
        [N, T] and must return values of shape [N, D], or a 2-tuple of those
        values and the tracked variables (:ref:`ug-shapes`,
        :ref:`ug-tracked-variables`). The loop evaluates many subintervals in
        one batch, accepts those whose error ratio is below 1 and splits the
        others (in half, or in four with ``split_rule="accelerate"``), until
        every subinterval is accepted. It then builds the optimal mesh, unless
        ``build_state_mesh=False``. It returns the result and a state, the
        optimal mesh and the batch size: ``result, state =
        solver.integrate(...)``. Pass
        ``state=state`` to a later call to start from both (see
        :ref:`the user guide <ug-reuse>`); the solver itself keeps nothing
        between calls.

        Every argument left as None falls back to the constructor's value,
        except the bounds, which have no default, and ``enforce_convergence``,
        whose default follows ``take_gradient``.

        Parameters
        ----------
        f : Callable or None
            The integrand. None uses the integrand given to the constructor.
        y0 : torch.Tensor or None
            Value added to the returned integral, shape [D]. None uses the
            constructor's ``y0`` (0 by default).
        mesh : torch.Tensor or None
            Starting mesh, shape [M, T], increasing, for example an earlier
            call's ``state.mesh``. Its first and last points are the
            integration bounds unless bounds are given: a bound inside the mesh
            cuts it, one outside extends it (the integration refines the new
            subinterval). None builds a fresh starting mesh on the bounds, which
            must then both be given (see :ref:`the user guide <ug-mesh>`).
        mesh_init : torch.Tensor or None
            Lower integration bound, a tensor of shape [T] (also when T is 1).
            With a ``mesh``, it becomes the starting mesh's first point, inside
            or outside the mesh (so it gets its gradient); None uses
            ``mesh[0]``. Without a mesh it is required.
        mesh_final : torch.Tensor or None
            Upper integration bound, a tensor of shape [T] (also when T is 1).
            With a ``mesh``, it becomes the starting mesh's last point, inside
            or outside the mesh (so it gets its gradient); None uses
            ``mesh[-1]``. Without a mesh it is required.
        random_initial_mesh : bool
            If True (default), the points of a fresh starting mesh are placed at
            random, so the mesh does not line up with evenly spaced features of
            f; if False, they are evenly spaced (see :ref:`the user guide <ug-mesh>`).
        N_init_steps : int
            Approximate number of subintervals of a fresh starting mesh. The
            default 13 gives 11 subintervals; points that rounding puts on or
            below an earlier point are dropped, so a large mesh, or one far
            from 0, can have fewer (see :ref:`the user guide <ug-mesh>`).
        f_args : tuple
            Extra positional arguments passed to f; tensors are moved to the
            solver's device. Default ``()``.
        take_gradient : bool
            If True (default), backpropagate each batch's loss right after it,
            so the whole autograd graph is never held; if False, keep the graph
            so the returned integral can be differentiated (see :ref:`the user guide <ug-autodiff>`).
        total_mem_usage : float or None
            Fraction of the device's total memory, in (0, 1], that may be in use.
            With ``take_gradient=True`` on a GPU, a batch whose backward pass is
            predicted not to fit in it is evaluated again, smaller. The calls of
            f themselves grow until they run out of memory and do not keep this
            share free (see :ref:`the user guide <ug-memory>`).
        loss_fxn : Callable or None
            Maps one batch's ``IntegrationResult`` to a scalar loss, which
            ``take_gradient=True`` backpropagates; ``result.loss`` is the sum
            over batches. None uses the integral (see :ref:`the user guide <ug-autodiff>`).
        max_batch : int or None
            Largest number of integrand evaluations per call of f, used as it is
            (it does not grow), also with a ``state``. With
            ``take_gradient=True`` it must be at least the C points of one
            subinterval (a call of f then evaluates whole subintervals), and so
            must a ``state``'s batch size that has converged, unless this is
            given; otherwise ``ValueError``. If this, the ``state`` and the
            constructor's value are all None, f is called on 10, 100 and 1,000
            copies of the first node of the first subinterval (stopping at the
            size of the first pass; the values are discarded), and the batch
            size starts at the largest that ran and grows by factors of 10
            until f runs out of memory. A test call that runs out of memory is
            repeated at 0.5 times its size up to 10 points and 0.75 above
            (with ``take_gradient=True`` not below one subinterval) until one
            runs, and the batch size then stops growing (see :ref:`the user
            guide <ug-memory>`).
        state : IntegrationState or None
            Where an earlier call ended (the second value it returned): its mesh (as
            ``mesh=``) and its batch size. Cannot be given with ``mesh``; bounds
            given with it cut or extend its mesh (see :ref:`the user guide <ug-reuse>`).
        parameters : iterable of torch.Tensor, torch.optim.Optimizer or None
            Needed only when f uses a custom autograd function (for example
            activation checkpointing with ``use_reentrant=True``, or a function
            compiled with ``torch.compile``) and ``take_gradient=True``: the tensors f's gradients reach, or their
            optimizer. After an out-of-memory error in a backward pass,
            PadaQuad puts back the gradients the call added and restarts the
            integral; it finds them by walking each batch's autograd graph,
            which a custom autograd function can hide them from. Without it,
            such a call raises ``BackwardOutOfMemoryError`` instead of
            restarting (see :ref:`the user guide <ug-memory>`).
        max_adaptive_splits : int or None
            Cap on how many times a subinterval may have been halved since the
            starting mesh; at the cap it is accepted even if it fails the
            tolerance (see :ref:`the user guide <ug-convergence>`).
        split_rule : str or None
            How a failing subinterval is split: ``"halve"`` or
            ``"accelerate"`` (into four pieces when it is a piece of a split
            subinterval with an error ratio of at least 2**error_exponent; see :ref:`the
            user guide <ug-four-pieces>`). ``"accelerate"`` with
            ``error_norm="failure_fraction"`` raises ``ValueError``.
        error_norm : str, Callable or None
            How the error of a vector-valued integrand becomes one
            accept-or-split decision (see :ref:`the user guide <ug-error-norm>`).
        error_integral_reference : float, torch.Tensor or None
            A value close to the integral, used in place of the running integral
            in the relative tolerance ``rtol * |reference|``. Only with
            ``use_absolute_error_ratio=True``. With ``enforce_convergence``, it
            is dropped once every subinterval is accepted, and the subintervals
            that fail against the integral are split, as below (see
            :ref:`the user guide <ug-error-reference>`).
        enforce_convergence : bool or None
            If True, judge every accepted subinterval again on every pass
            (with a fixed ``error_integral_reference``, only once the
            reference is dropped), against the same tolerance as the
            subintervals just evaluated, and
            split those that fail, so that the returned integral meets the
            tolerance of the final integral unless ``max_adaptive_splits`` or a
            non-finite value stops the refinement. If False, an accepted
            subinterval is never judged again. None (default) means True with
            ``take_gradient=False`` and False with ``take_gradient=True``;
            True with ``take_gradient=True`` raises ``ValueError``, because a
            batch's gradient has already been backpropagated when it is
            accepted (see :ref:`the user guide <ug-convergence>`).
        mesh_failure_tolerance : float or None
            With ``error_norm="failure_fraction"``, the fraction of output
            components that may fail while a subinterval is still accepted
            (see :ref:`the user guide <ug-error-norm>`).
        runtime_checks : bool or None
            The constructor's ``runtime_checks`` for this call: if True, raise
            ``ValueError`` when f returns NaN or Inf and check PadaQuad's own
            bookkeeping; if False, skip these checks. None (default) keeps the
            constructor's value.
        trade_speed_for_memory : bool
            Only with ``take_gradient=False``. If False (default), a pass's
            integrand values stay on the solver's device until all its
            subintervals are evaluated; their contributions and error estimates
            are then computed at once, and the accepted subintervals' values
            move to ``result_device`` after the accept-or-split decision. If
            True, the contributions and error estimates are computed as soon as
            ``evaluation_buffer`` bytes of values of whole subintervals are
            waiting, and those values move to ``result_device`` right away, so
            the solver's device holds about one call's values (plus the
            buffer) instead of the pass's, at a cost in time per call of f.
            The calls of f, the mesh and the integral are the same either way
            (see :ref:`the user guide <ug-trade-speed-for-memory>`).
        evaluation_buffer : int or None
            With ``trade_speed_for_memory=True``, how many bytes of integrand
            values of whole subintervals may wait on the solver's device before
            their contributions and error estimates are computed and they move
            to ``result_device``. None (default) computes them after every call
            of f. A larger buffer costs that much device memory and saves time
            per call of f.
        record_evaluations : bool
            If True (default), the result holds the integrand values and
            tracked variables at every node. If False, they are dropped as soon
            as their subintervals' contributions and error estimates are
            computed, so they are never moved or kept: ``y`` and
            ``tracked_variables`` are None in the result and in the per-batch
            results ``loss_fxn`` receives, the nodes stay on the solver's
            device until the end, and the state's mesh is built with
            ``merge_rule="power_law"``, because ``"predicted"`` reads the
            values.
        result_device : str, torch.device or None
            Device of the per-subinterval arrays of the result (default
            ``"cpu"``); None keeps them on the solver's device. The integral and
            the state's mesh stay on the solver's device.
        build_state_mesh : bool
            If True (default), build the optimal mesh after the loop and return
            it as ``state.mesh``. If False, skip it, which saves a few
            milliseconds per call when no later call starts from the mesh, and
            return ``state.mesh = None``: a later call given this state keeps
            its batch size but needs both bounds and starts from a fresh mesh
            (see :ref:`the user guide <ug-reuse>`).

        Returns
        -------
        result : IntegrationResult
            The integral (plus ``y0``), its error estimate, whether it
            ``converged``, the nodes and integrand values of every subinterval,
            and split statistics (see :ref:`the user guide <ug-convergence>`).
        state : IntegrationState
            The optimal mesh (None with ``build_state_mesh=False``) and the
            batch size, for a later call's ``state=`` (see :ref:`the user guide
            <ug-reuse>`).

        Raises
        ------
        ValueError
            If there is no ``mesh`` and a bound is missing; if the ``mesh`` is
            not [M, T] with M >= 2 increasing points; if a bound is not a
            tensor of shape [T] (with or without a mesh); if the bounds do not
            satisfy ``mesh_init < mesh_final``; if
            no integrand was given here or at construction, or it does not
            return values of shape [N, D]; if both ``mesh`` and ``state`` are
            given; if an option such as ``error_norm`` or
            ``total_mem_usage`` is invalid; if ``enforce_convergence`` is
            True with ``take_gradient=True``; if ``trade_speed_for_memory`` is
            True with ``take_gradient=True``; if ``split_rule="accelerate"``
            meets ``error_norm="failure_fraction"`` (from this call or the
            constructor); or if ``evaluation_buffer`` is given with
            ``trade_speed_for_memory=False`` or is not an int >= 1; or if,
            with ``take_gradient=True``, ``max_batch`` (or, without it, the
            batch size of a ``state`` that has converged) is below the C
            points of one subinterval.
        TypeError
            If ``take_gradient``, ``random_initial_mesh``,
            ``trade_speed_for_memory``, ``record_evaluations`` or
            ``build_state_mesh`` is not a bool (Python or NumPy), or
            ``runtime_checks`` or ``enforce_convergence`` is neither None nor
            a bool; checked before f is called.
        BackwardOutOfMemoryError
            If a backward pass ran out of memory and PadaQuad cannot restart
            the integral (see ``parameters``); its ``state`` is the state to
            call again with, after resetting the gradients.
        torch.OutOfMemoryError
            If one point does not fit in memory, or, with
            ``trade_speed_for_memory=False``, the integrand values of a pass
            do not fit in one array; the message names the remedies.

        Examples
        --------
        Start a second call from the first one's mesh and batch size:

        >>> import torch
        >>> from padaquad import adaptive_quadrature
        >>> _ = torch.manual_seed(0)  # the first starting mesh is random
        >>> solver = adaptive_quadrature("uniform", "gk21", device="cpu")
        >>> a = torch.tensor([0.0], dtype=torch.float64)
        >>> first, state = solver.integrate(torch.exp, mesh_init=a, mesh_final=a + 1)
        >>> second, state = solver.integrate(torch.exp, state=state)
        >>> print(round(float(second.integral), 6))
        1.718282
        """
        if state is not None:
            if mesh is not None:
                raise ValueError(
                    "Pass mesh= or state=, not both. To start a state from another "
                    "mesh, pass state=dataclasses.replace(state, mesh=mesh)."
                )
            mesh = state.mesh
        # A restart starts from the given mesh (it may require grad), else
        # from the fresh one (see the end of the loop).
        given_mesh = mesh

        # Set dtype based on input
        self.set_dtype_by_input(mesh=mesh, mesh_init=mesh_init, mesh_final=mesh_final)

        # Setup results device
        result_device = _resolve_device(
            result_device or self.device, self.device.index or 0
        )
        # Where a pass's integrand values and tracked variables go once their
        # subintervals are integrated (_evaluate_f_on_split_nodes): to
        # result_device right away with trade_speed_for_memory, else they stay
        # on the solver's device until the accept-or-split decision, after
        # which the record moves the accepted ones; None drops them.
        values_device: torch.device | None = None
        if record_evaluations:
            values_device = result_device if trade_speed_for_memory else self.device
        # Bytes of values of whole subintervals that wait on the solver's
        # device before they are integrated: the whole pass without
        # trade_speed_for_memory, else the buffer (0: after every call of f).
        buffer_bytes = (evaluation_buffer or 0) if trade_speed_for_memory else math.inf

        # The bounds: the given ones (they cut or extend a given mesh), else the
        # mesh's ends; without a mesh both must be given.
        mesh, mesh_init, mesh_final = self._setup_integral_bounds(
            mesh, mesh_init, mesh_final
        )

        # A given max_batch (this call's, else the constructor's unless a state
        # is given): an int (numpy ints too) of at least 1; a bool, float or
        # string raises here, before f is called. Used below the starting mesh.
        given = (
            self.init_max_batch if max_batch is None and state is None else max_batch
        )
        if given is not None and (isinstance(given, bool) or operator.index(given) < 1):
            raise ValueError(f"max_batch must be an int >= 1; got {given!r}.")

        # Per-call max_adaptive_splits takes priority over the constructor value.
        max_adaptive_splits = (
            self.init_max_adaptive_splits
            if max_adaptive_splits is None
            else max_adaptive_splits
        )
        self._check_max_adaptive_splits(max_adaptive_splits)

        # The re-check of accepted subintervals (Step 4b) is on by default only
        # without per-batch backward: with take_gradient=True a batch's
        # gradient is already backpropagated when the batch is accepted, so
        # splitting an accepted subinterval again would count its gradient
        # twice.
        if enforce_convergence is None:
            enforce_convergence = not take_gradient
        # Per-call runtime_checks takes priority over the constructor value.
        # Stored on self so the evaluation helpers can read it without threading
        # it through every signature.
        self.runtime_checks = (
            self.init_runtime_checks if runtime_checks is None else runtime_checks
        )
        self._check_bools(
            take_gradient=take_gradient,
            enforce_convergence=enforce_convergence,
            runtime_checks=self.runtime_checks,
            random_initial_mesh=random_initial_mesh,
            trade_speed_for_memory=trade_speed_for_memory,
            record_evaluations=record_evaluations,
            build_state_mesh=build_state_mesh,
        )
        if enforce_convergence and take_gradient:
            raise ValueError(
                "enforce_convergence=True cannot be used with take_gradient=True: "
                "each batch's gradient is backpropagated as soon as the batch is "
                "accepted, so an accepted subinterval that later fails its "
                "tolerance cannot be split again without counting its gradient "
                "twice. Pass take_gradient=False and differentiate "
                "result.integral, or leave enforce_convergence as None."
            )
        # trade_speed_for_memory changes when a pass's values are integrated
        # and moved; with per-batch backward each batch is one call of f,
        # integrated and moved after its backward pass, so it has no effect.
        if trade_speed_for_memory and take_gradient:
            raise ValueError(
                "trade_speed_for_memory=True cannot be used with "
                "take_gradient=True: with take_gradient=True each batch is one "
                "call of f on whole subintervals, integrated at once, whose values "
                "leave the solver's device after the batch's backward pass, so "
                "there is nothing to trade. Pass take_gradient=False; to integrate "
                "a gradient without an autograd graph, see the example "
                "examples/gradient_as_output.py."
            )
        if evaluation_buffer is not None:
            if not trade_speed_for_memory:
                raise ValueError(
                    "evaluation_buffer applies only with trade_speed_for_memory="
                    "True: with False a pass's values are integrated at once, "
                    "after its last call of f. Pass trade_speed_for_memory=True, "
                    "or leave evaluation_buffer as None."
                )
            _check_evaluation_buffer(evaluation_buffer)
        # Advice for a setting that works but may not do what it is meant to,
        # given once, on the first call of a series (no state yet): later calls
        # pass the state, and repeating it every optimizer step would bury it.
        guide = (
            "https://khegazy.github.io/padaquad/user_guide/devices_memory.html"
            "#ug-trade-speed-for-memory"
        )
        if state is None and trade_speed_for_memory and evaluation_buffer is None:
            logger.warning(
                "trade_speed_for_memory=True without evaluation_buffer integrates "
                "the values after every call of f, which costs about 0.1 ms per "
                "call on a CPU and 0.2 ms on a GPU (measured with cheap "
                "integrands). With many calls of f, or a fast f, pass "
                "evaluation_buffer (bytes): values of whole subintervals then "
                "wait on the solver's device until that many bytes are "
                "integrated together. See %s",
                guide,
            )
        if (
            state is None
            and build_state_mesh
            and not record_evaluations
            and self.merge_rule == "predicted"
        ):
            logger.warning(
                "record_evaluations=False: the state's mesh is built with "
                "merge_rule='power_law', because merge_rule='predicted' reads the "
                "integrand values, which are not kept. It refuses some merges of "
                "wide subintervals, so a later call starting from the state may "
                "begin with more subintervals. Construct the solver with "
                "merge_rule='power_law' to choose it explicitly."
            )

        # Per-call error_norm / mesh_failure_tolerance take priority over the
        # constructor values. Stored on self so the error-ratio helpers read
        # them without threading them through every signature.
        self.error_norm = self.init_error_norm if error_norm is None else error_norm
        self._check_error_norm(self.error_norm)
        split_rule = self.init_split_rule if split_rule is None else split_rule
        self._check_split_rule(split_rule)
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
        # tolerance does not drift and accepted steps are not re-checked; with
        # enforce_convergence it is dropped once every step is accepted
        # (Step 7 of the loop).
        reference_pinned = (
            self.use_absolute_error_ratio and error_integral_reference is not None
        )

        # Get variables or populate with default values, send to correct device
        f, y0 = self._check_variables(f, y0)
        # Coerce any tensor in f_args onto the solver device too (it is forwarded
        # to f(t, *f_args)); DistributedEnvironment owns the device and every
        # input is moved onto it.
        f_args = tuple(a.to(self.device) if torch.is_tensor(a) else a for a in f_args)
        total_mem_usage = (
            self.init_total_mem_usage if total_mem_usage is None else total_mem_usage
        )
        self._check_total_mem_usage(total_mem_usage)
        self._total_mem_usage = total_mem_usage
        self._backward_ratio = 1.0

        loss_fxn = loss_fxn if loss_fxn is not None else self._integral_loss

        # Starting mesh: the caller's mesh cut to the bounds, or a fresh one.
        mesh, mesh_trackers = self._setup_initial_mesh(
            mesh,
            mesh_init,
            mesh_final,
            random_initial_mesh,
            N_init_steps,
        )
        n_start_steps = len(mesh) - 1
        start_mesh = given_mesh if given_mesh is not None else mesh.detach().clone()
        # The batch size of this call (_batch_for): a given max_batch is fixed;
        # a state's may grow unless it has converged; with neither, the test
        # calls find where to start.
        if given is not None:
            self._batch, self._batch_converged = operator.index(given), True
        elif state is not None:
            self._batch = state.max_batch
            self._batch_converged = state.max_batch_converged
        else:
            # The test calls: f, forward only, on 10, 100 and 1,000 copies of
            # one point, stopping at the order of the first pass, only to see
            # that f runs at these sizes; the values are discarded. The batch
            # size starts at the largest that ran (1 below 10 points), so the
            # first pass's first call is at most 10 times it. A call that runs
            # out of memory is repeated smaller, at 0.5 times its size up to
            # 10 points and 0.75 above, until one runs; the batch size has then
            # converged. With take_gradient=True, where every batch holds whole
            # subintervals, no retry is smaller than one subinterval: C points
            # are tried before giving up, and _call_f raises if they do not
            # fit. The point is the first node of the first subinterval, which
            # the first pass evaluates: never a bound that a method without
            # end nodes leaves out (f may be singular there). Detached, so the
            # test calls build no graph of bounds that require grad.
            point = self._compute_nodes(mesh[:1], mesh[1:2])[0, :1].detach()
            smallest = self.C if take_gradient else 1
            self._batch, self._batch_converged, n = 1, False, _GROWTH
            while n <= min(n_start_steps * self.C, _LARGEST_TEST_CALL) and not (
                self._batch_converged
            ):
                while (
                    self._call_f(f, f_args, point.repeat(n, 1), n != smallest) is None
                ):
                    factor = 0.5 if n <= _GROWTH else _OOM_BACKOFF
                    n = max(smallest, _shrink_batch(n, factor))
                    self._batch_converged = True
                self._batch, n = n, n * _GROWTH
        # With take_gradient=True every call of f is whole subintervals, each
        # followed by its backward pass, so a batch size that does not grow
        # must hold one subinterval: a given one, or a state's converged one
        # (the largest known to fit; raising it could run out of memory).
        if take_gradient and self._batch_converged and self._batch < self.C:
            raise ValueError(
                f"{'max_batch' if given is not None else 'The state batch size'} "
                f"{self._batch} is below the {self.C} points of one subinterval of "
                f"{self.method_name}: with take_gradient=True every call of f "
                "evaluates whole subintervals, each followed by its backward "
                f"pass. Pass max_batch={self.C} or more (it replaces a state's "
                "batch size), or take_gradient=False, where a call of f may end "
                "inside a subinterval."
            )
        # Gradients to put back after an out-of-memory error in a backward
        # pass: the given parameters now, the rest as the graph walk finds them.
        saved: _SavedGrads = {}
        # A list by now (_restart_after_backward_oom), never an optimizer.
        assert not isinstance(parameters, torch.optim.Optimizer)
        for p in parameters or ():
            _save_grad(p, saved)
        unsafe: str | None = None
        failed_batch: int | None = None

        # The record of the accepted subintervals, in the order they were
        # accepted (_record_results): the sums, and the fields the loop reads,
        # one tensor each, among them each subinterval's left mesh index in
        # the current mesh (mesh_index); the other fields, one chunk per
        # accepted batch, in chunks. Put in mesh order after the loop
        # (_sort_record).
        record: dict[str, torch.Tensor] = {}
        chunks: dict[str, list[torch.Tensor]] = {}
        # Per-panel split counter, parallel to mesh/mesh_trackers: how many
        # times each step was halved (0 = a step of the starting mesh). It
        # enforces max_adaptive_splits and gives the split statistics returned
        # with the result. Typed Optional because _adaptively_increase_mesh
        # returns it as Optional (its post-convergence caller passes None).
        split_counts: torch.Tensor | None = torch.zeros(
            len(mesh), dtype=torch.long, device=self.device
        )
        # With take_gradient every batch runs its own backward, but all of them
        # reach a bound that requires grad through one autograd graph, the
        # mesh's (its ends and the midpoints split from them), whose saved
        # indices the first backward frees. A batch is therefore evaluated on a
        # detached copy of the mesh, and the copy's gradient is added to
        # mesh_grad, the gradient with respect to the current mesh (None until
        # a batch depends on the mesh), which is passed through the mesh's
        # graph once, after the loop. Passing it once per batch crossed the
        # whole graph, which grows with every split, each time: the time grew
        # with the square of the number of batches.
        collect_mesh_grad = take_gradient and mesh.requires_grad
        mesh_grad: torch.Tensor | None = None
        # === Main integration loop ===
        # Continues until all steps have been evaluated and accepted
        # (mesh_trackers[i] == False for all i). waiting, the left mesh
        # indices of the subintervals to evaluate (ascending), is the turn's
        # one read of the trackers: it gives the loop's condition and the
        # subintervals to evaluate.
        while len(waiting := torch.nonzero(mesh_trackers).flatten()) > 0:
            # The batch's mesh: a detached copy when mesh_grad collects the
            # gradient (see mesh_grad above).
            mesh_eval = mesh.detach().requires_grad_() if collect_mesh_grad else mesh
            # Steps 1-3: evaluate f on the waiting subintervals (all of them
            # with take_gradient=False; with take_gradient=True, the whole ones
            # that fit in one batch) and integrate each (method_output).
            (
                nodes,
                y_step_eval,
                tracked_step_eval,
                step_idxs,
                method_output,
            ) = self._evaluate_f_on_mesh(
                f,
                f_args,
                mesh_eval,
                waiting,
                take_gradient,
                values_device,
                buffer_bytes,
            )
            batch_size = len(nodes) * self.C

            # The running integral, and in cumulative mode the running sums.
            # Error control only decides; it reads detached values (the error
            # estimates are detached already), so the error ratios hold no
            # autograd graph, nor the arrays its nodes would save.
            t0 = time.time()
            quads = method_output.mesh_quadratures.detach()
            current_integral = method_output.integral.detach()
            if record:
                current_integral = record["integral"].detach() + current_integral
            # Absolute mode judges against the integral alone. Cumulative mode
            # judges each subinterval, evaluated or recorded, against the
            # running integral through it: the record and the batch merged in
            # mesh order by the left mesh index of each (integers, for any time
            # dimension; on the first turn the record is empty), one sum for
            # both.
            cum_mesh_quadratures = recorded_cum = None
            if not self.use_absolute_error_ratio:
                idxs_keep, idxs_input = self._merge_positions(
                    record.get("mesh_index", step_idxs[:0]), step_idxs
                )
                cum = torch.cumsum(
                    self._insert_sorted_results(
                        record.get("mesh_quadratures", quads[:0]).detach(),
                        idxs_keep,
                        quads,
                        idxs_input,
                    ),
                    0,
                )
                cum_mesh_quadratures, recorded_cum = cum[idxs_input], cum[idxs_keep]
            if self.speed_logger:
                self.speed_logger.debug("merge with record: %s", time.time() - t0)

            # --- Step 4: Compute error ratios for each step ---
            t0 = time.time()
            # The tolerance denominator is max(atol, rtol*|reference_integral|)
            # (atol + rtol*|reference_integral| for failure_fraction). By
            # default the reference is the running integral: with
            # take_gradient=True only part of a pass is known when a batch is
            # judged, and in the first passes the estimate is coarse. A
            # user-supplied error_integral_reference (≈ the true total integral)
            # replaces it so rtol is judged against the right scale from the
            # first batch.
            reference_integral = current_integral
            if error_integral_reference is not None:
                # The reference must match the integrand's D output components
                # (or be one value for all); another shape would broadcast
                # silently. Checked here, where D is first known: on the first
                # pass, before any subinterval is judged or backpropagated.
                D = current_integral.shape[-1]
                if error_integral_reference.dim() > 1 or (
                    error_integral_reference.numel() not in (1, D)
                ):
                    raise ValueError(
                        "error_integral_reference must be a number or a tensor of "
                        f"shape [], [1] or [D] = [{D}]; got shape "
                        f"{list(error_integral_reference.shape)}."
                    )
                reference_integral = error_integral_reference
            # The norm family judges a subinterval by its reduced error and
            # rounding floor, which never change once it is evaluated: the
            # record keeps them, so Step 4b, Step 7 and the final check judge
            # it by the numbers it was accepted with. "failure_fraction"
            # compares each output with its own tolerance and keeps none.
            reduced_errors = (
                None
                if self._is_failure_fraction
                else self._reduced_errors(method_output.mesh_quadrature_errors, quads)
            )
            error_ratios, _, error_ratios_per_dim = self._compute_error_ratios(
                mesh_quadrature_errors=method_output.mesh_quadrature_errors,
                mesh_quadratures=quads,
                cum_mesh_quadratures=cum_mesh_quadratures,
                integral=reference_integral,
                reduced_errors=reduced_errors,
            )
            keep_mask, _, _ = self._accept_reject_masks(
                error_ratios, error_ratios_per_dim
            )
            if self.speed_logger:
                self.speed_logger.debug("calculate errors: %s", time.time() - t0)
            assert len(nodes) == len(error_ratios)
            logger.debug("error_ratios: %s", error_ratios)

            # --- Step 4b: Re-check already-accepted (recorded) panels ---
            # The accept/reject denominator (built from reference_integral, Step 4)
            # drifts as the running integral changes, so a panel accepted early
            # can violate the *current* tolerance. Re-check the recorded panels
            # against the SAME reference_integral the batch just used (so a panel
            # accepted this iteration -- not yet in the record -- cannot
            # immediately re-fail) and split any that now fail in the same mesh
            # update below (extra_split_idxs). Only with enforce_convergence,
            # which is never True with take_gradient=True (per-batch backward
            # would count a re-split panel's gradient twice). Skipped while a
            # fixed error_integral_reference pins the denominator (no drift);
            # Step 7 drops it once every panel is accepted.
            recheck_active = (
                enforce_convergence and len(record) > 0 and not reference_pinned
            )
            extra_split_idxs: torch.Tensor | None = None
            record_keep_mask: torch.Tensor | None = None
            if recheck_active:
                # All inputs are integration-device-resident (mesh_quadratures
                # and mesh_quadrature_errors are in _RECORD_LOOP_KEYS;
                # reference_integral is on self.device) -- no host<->device copy.
                # Cumulative mode judges each recorded step against the running
                # integral through it, which includes this batch's steps to its
                # left (recorded_cum, from the sum above; None in absolute
                # mode).
                extra_split_idxs, record_keep_mask, _ = self._failed_record_positions(
                    record,
                    recorded_cum,
                    reference_integral,
                    mesh,
                    split_counts,
                    max_adaptive_splits,
                )

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
                idxs_transfer,
                keep_idx,
                split_counts,
            ) = self._adaptively_increase_mesh(
                method_output=method_output,
                keep_mask=keep_mask,
                y_step_eval=y_step_eval,
                nodes=nodes,
                mesh=mesh,
                mesh_idxs=step_idxs,
                mesh_trackers=mesh_trackers,
                tracked_step_eval=tracked_step_eval,
                split_counts=split_counts,
                max_adaptive_splits=max_adaptive_splits,
                extra_split_idxs=extra_split_idxs,
                # The split path's arrays are the solver's own (an f output
                # that requires grad is copied by _keep_rows); the full path's
                # values are f's output, which f may still hold.
                in_place=not take_gradient,
                # The ratios split some subintervals in four. Only a piece of a
                # split subinterval can be split in four: none exists until
                # the mesh first grows, so the rule's test (about 30 GPU
                # launches) is skipped until then.
                error_ratios=error_ratios
                if split_rule == "accelerate" and len(mesh) - 1 > n_start_steps
                else None,
            )
            # The batch data passed in comes back filtered to the accepted steps
            # (None comes back only for None inputs: no values are kept with
            # record_evaluations=False, and _get_optimal_mesh passes none).
            assert kept_output is not None
            assert kept_nodes is not None
            assert split_counts is not None  # returned non-None for non-None input
            method_output, y_step_eval, nodes = kept_output, kept_y, kept_nodes
            error_ratios = error_ratios[keep_idx]
            # The recorded subintervals' left mesh points in the new mesh.
            if record:
                record["mesh_index"] = idxs_transfer[record["mesh_index"]]

            # --- Step 5b: Drop re-split recorded panels from the record ---
            # The recorded panels flagged above were just bisected in the mesh
            # update; remove their now-stale entries (their children will be
            # re-evaluated as fresh panels). Done before _record_results, whose
            # rows record_keep_mask does not cover.
            # Both are set exactly when recheck_active, and record_keep_mask
            # drops exactly the subintervals of extra_split_idxs, whose size is
            # known on the host (no read of the device).
            if extra_split_idxs is not None and len(extra_split_idxs) > 0:
                assert record_keep_mask is not None
                record = self._remove_failed_record_panels(
                    record,
                    record_keep_mask,
                    chunks,
                    loss_fxn,
                    mesh_init,
                    mesh_final,
                    values_device,
                )

            # --- Step 6: Record accepted results and handle gradients ---
            if nodes.shape[0] > 0:
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
                # The loss, and its backward pass if requested. With
                # take_gradient=True an out-of-memory error in either restarts
                # the integral (below the loop).
                loss, batch_unsafe = self._loss_and_backward(
                    intermediate_results,
                    loss_fxn,
                    take_gradient,
                    saved,
                    parameters is not None,
                )
                unsafe = batch_unsafe or unsafe
                if loss is None:
                    failed_batch = batch_size
                    break

                # Accumulate into the record. The record keeps the signed error
                # estimates, so that removing a subinterval (Step 5b, Step 7)
                # subtracts exactly what it added to integral_error; loss_fxn
                # above and the returned result see their absolute values.
                intermediate_results.loss = loss
                intermediate_results.mesh_quadrature_errors = (
                    method_output.mesh_quadrature_errors
                )
                record = self._record_results(
                    record=record,
                    chunks=chunks,
                    take_gradient=take_gradient,
                    results=intermediate_results,
                    mesh_index=idxs_transfer[step_idxs[keep_idx]],
                    result_device=result_device,
                    reduced_errors=None
                    if reduced_errors is None
                    else reduced_errors[keep_idx],
                )
                del intermediate_results

            # Release the batch's values (the record holds what it keeps), so
            # they are not held during the next pass's calls of f or while the
            # record is sorted after the loop.
            del y_step_eval, kept_y, tracked_step_eval

            if collect_mesh_grad:
                grads = [g for g in (mesh_grad, mesh_eval.grad) if g is not None]
                if grads:
                    # Carry the gradient to the split mesh, whose points include
                    # every point of the batch's mesh, with the same values
                    # (Step 7, the other mesh update, never runs with
                    # take_gradient=True).
                    mesh_grad = torch.zeros_like(mesh.detach())
                    mesh_grad[idxs_transfer] = sum(grads)

            # --- Step 7: drop a fixed reference once every panel is accepted ---
            # With a fixed error_integral_reference the loop judged every panel
            # against it, not against the integral. The record now holds the
            # whole integral, so judge every recorded panel against it, from
            # the stored integrals and error estimates (no call of f), and split
            # the failures here (a failing panel at max_adaptive_splits is not
            # split and stays recorded). Only their halves are evaluated, on the
            # next pass, where the running integral is the reference: they and
            # the whole record are judged against it together (Step 4b), so the
            # check of every panel happens in the same pass as the split
            # decision until the integral converges. A reference is fixed only
            # in absolute mode, so no running sum is needed, and the record is
            # non-empty.
            if (
                enforce_convergence
                and reference_pinned
                and not torch.any(mesh_trackers)
            ):
                error_integral_reference, reference_pinned = None, False
                split_positions, record_keep_mask, fails = (
                    self._failed_record_positions(
                        record,
                        None,
                        record["integral"].detach(),
                        mesh,
                        split_counts,
                        max_adaptive_splits,
                    )
                )
                if bool(fails.any()):
                    # Explains the calls of f that follow the dropped reference.
                    logger.info(
                        "error_integral_reference dropped once every step was "
                        "accepted: %d of %d accepted steps fail against the "
                        "integral; %d of them (those below max_adaptive_splits "
                        "whose midpoint differs from both ends) are split.",
                        int(fails.sum()),
                        len(fails),
                        split_positions.numel(),
                    )
                if split_positions.numel() > 0:
                    # No batch: only the extra positions are split.
                    (
                        _,
                        _,
                        _,
                        _,
                        mesh,
                        mesh_trackers,
                        idxs_transfer,
                        _,
                        split_counts,
                    ) = self._adaptively_increase_mesh(
                        method_output=None,
                        keep_mask=torch.empty(0, dtype=torch.bool, device=self.device),
                        y_step_eval=None,
                        nodes=None,
                        mesh=mesh,
                        mesh_idxs=torch.empty(0, dtype=torch.long, device=self.device),
                        mesh_trackers=mesh_trackers,
                        split_counts=split_counts,
                        max_adaptive_splits=max_adaptive_splits,
                        extra_split_idxs=split_positions,
                    )
                    record["mesh_index"] = idxs_transfer[record["mesh_index"]]
                    record = self._remove_failed_record_panels(
                        record,
                        record_keep_mask,
                        chunks,
                        loss_fxn,
                        mesh_init,
                        mesh_final,
                        values_device,
                    )

        if failed_batch is not None:
            # A backward pass ran out of memory: restart from the starting
            # mesh with a smaller batch size, or raise.
            self._raise_after_backward_oom(failed_batch, start_mesh, saved, unsafe)
        if mesh_grad is not None:
            mesh.backward(mesh_grad)

        # An integral that requires grad (from f's parameters, or from bounds
        # that require grad) carries an autograd graph, which holds what f
        # saved for its backward pass on the solver's device until
        # result.integral is differentiated, wherever the values went. Known
        # only once f has run, so given after the loop (no state: the first
        # call of a series).
        if (
            state is None
            and trade_speed_for_memory
            and record["integral"].requires_grad
        ):
            logger.warning(
                "trade_speed_for_memory=True with an integrand output that "
                "requires grad: the autograd graph keeps what f saved for the "
                "backward pass on the solver's device until result.integral is "
                "differentiated, so moving the values may save little. To "
                "integrate a gradient without an autograd graph, let f return it "
                "as extra outputs, computed with torch.func.grad from detached "
                "parameters (see examples/gradient_as_output.py and %s).",
                guide,
            )

        # === Post-convergence: sort results and optimize the mesh ===
        # The fields of chunks go to the result's device, where
        # _get_optimal_mesh also reads y (when build_state_mesh).
        record = self._sort_record(
            record, chunks, mesh, result_device if record_evaluations else self.device
        )

        # Split statistics. A step of the starting mesh has split count 0 and
        # every half of a split step has count >= 1, so the steps still at 0
        # are the starting steps that were never split. The last barrier
        # starts no step. Splits made below by _get_optimal_mesh are not
        # counted, and max_adaptive_splits truncates the depth.
        assert split_counts is not None  # always kept in the loop
        step_split_counts = split_counts[:-1]

        # Final error check: does the returned integral meet the tolerance?
        # Every recorded step is judged again against the final integral (in
        # cumulative mode, the final running integral through the step), the
        # reference _get_optimal_mesh below also uses. A step can fail here
        # although it passed when accepted, because it was judged against the
        # running integral and, without enforce_convergence, nothing
        # re-checked it.
        # A step accepted at max_adaptive_splits usually fails, but can pass.
        # A non-finite error ratio never meets a tolerance; _accept_reject_masks
        # accepts it only because splitting cannot reduce it. In absolute mode
        # mesh_optimal splits every finite failing step; in cumulative mode
        # _prune_excess_mesh can merge one with a right neighbour whose running
        # integral, and so its tolerance, is much larger. Runs before the record
        # is moved to result_device, so all inputs are on the integration device.
        # The record is in mesh order, so in cumulative mode its own running
        # sum (None) is the final running integral.
        final_ratios, final_fails, nonfinite = self._record_error_check(
            record, None, record["integral"].detach()
        )
        # Read only by the checks above; not moved to result_device.
        record.pop("reduced_errors", None)
        final_fails = final_fails | nonfinite
        # The split statistics and the check in one read of the device.
        n_unsplit, max_split_depth, failed = torch.stack(
            [
                (step_split_counts == 0).sum(),
                step_split_counts.max(),
                final_fails.any(),  # int64, by stack's type promotion
            ]
        ).tolist()
        n_start_steps_split = n_start_steps - n_unsplit
        converged = not failed
        if not converged:
            self._warn_not_converged(
                record,
                final_ratios,
                final_fails,
                nonfinite,
                mesh,
                split_counts,
                max_adaptive_splits,
                take_gradient,
                enforce_convergence,
                reference_pinned,
            )

        # Prune over-resolved steps and refine under-resolved ones. The merge
        # prediction applies the method's weights, so it must run with the
        # method's tables and device-bound allocations co-located with its
        # inputs. It runs where the recorded nodes and values are
        # (result_device, or the integration device with
        # record_evaluations=False), with the method converted there for the
        # duration rather than copying the values back. The fields that stay
        # on the integration device (mesh_quadratures and mesh_quadrature_errors,
        # see _RECORD_LOOP_KEYS, and the integral) are moved to join them. Without
        # recorded values (record_evaluations=False) y is None, and
        # _merge_inputs uses merge_rule="power_law".
        # mesh_optimal is the mesh a later call starts from when the caller
        # passes it as mesh=, so it lives on the integration device (its endpoints
        # are mesh_init/mesh_final, which live on self.device). Detached: when a
        # bound requires grad, the mesh points carry its autograd graph, and a
        # later warm call would backpropagate into the earlier call's bound (or
        # through a graph a take_gradient=True call already freed). The bound's
        # gradient reaches it through this call's nodes, not through the mesh.
        # With build_state_mesh=False it is not built, and state.mesh is None.
        mesh_optimal: torch.Tensor | None = None
        if build_state_mesh:
            optimal_device = record["nodes"].device
            with self._integration_on_device(optimal_device):
                mesh_optimal = self._get_optimal_mesh(
                    mesh_quadratures=record["mesh_quadratures"].to(optimal_device),
                    # Absolute values: the merge sums two subintervals' errors,
                    # which signed values could cancel.
                    mesh_quadrature_errors=torch.abs(
                        record["mesh_quadrature_errors"]
                    ).to(optimal_device),
                    integral=record["integral"].to(optimal_device),
                    mesh=mesh.to(optimal_device),
                    y=record.get("y"),
                )
            mesh_optimal = mesh_optimal.to(self.device).detach()

        # Move the record fields still on the integration device (the loop's
        # fields and the loss; every one with record_evaluations=False) to
        # result_device so the rest of the returned IntegrationResult lives on
        # result_device.
        # The integration-output scalars (integral, integral_error) and
        # mesh_optimal stay on the integration device -- the scalars are the
        # primary answer and mesh_optimal is reused as the warm-start mesh.
        for key in record:
            if key not in self._RESULT_INTEGRATION_KEYS:
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
            # The bounds stay on the integration device, with the state's mesh,
            # whose ends they are.
            mesh_init=mesh_init,
            mesh_final=mesh_final,
            nodes=self._flatten_output(record["nodes"]),
            h=record["h"],
            y=self._flatten_output(record["y"]) if "y" in record else None,
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
        ), IntegrationState(mesh_optimal, self._batch, self._batch_converged)

    def _record_error_check(
        self,
        record: dict[str, torch.Tensor],
        cum_mesh_quadratures: torch.Tensor | None,
        integral: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Judge every recorded (accepted) subinterval against a given reference.

        The one definition of "a recorded subinterval fails", used by the
        in-loop re-check of accepted subintervals (Step 4b of ``integrate``)
        and by the final ``converged`` check. It uses the same error ratios and
        accept rule as a freshly evaluated batch (``_compute_error_ratios``,
        ``_accept_reject_masks``), and for the norm family the reduced error
        and rounding floor each subinterval was accepted with (the record's
        ``reduced_errors``), so only the tolerance can change its decision.

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            The record, with ``mesh_quadratures`` and ``mesh_quadrature_errors``
            [N, D] on the integration device, in the order the subintervals
            were accepted (in the loop) or in mesh order (after it). Each
            subinterval's ratio depends only on its own row and on
            ``cum_mesh_quadratures``.
        cum_mesh_quadratures : torch.Tensor or None
            Running integral through each recorded subinterval [N, D], in the
            record's row order (used in cumulative mode only; None there sums
            ``mesh_quadratures``, which must then be in mesh order).
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
            reduced_errors=record.get("reduced_errors"),
        )
        _, reject_mask, nonfinite_mask = self._accept_reject_masks(
            error_ratios, error_ratios_per_dim
        )
        return error_ratios, reject_mask, nonfinite_mask

    def _failed_record_positions(
        self,
        record: dict[str, torch.Tensor],
        cum_mesh_quadratures: torch.Tensor | None,
        integral: torch.Tensor,
        mesh: torch.Tensor,
        split_counts: torch.Tensor | None,
        max_adaptive_splits: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the recorded subintervals to split again, from their stored errors.

        Judges every recorded subinterval with ``_record_error_check`` (no call
        of the integrand) and keeps the failing ones that can be split
        (``_splittable``: below the split cap, and wide enough to halve). Used
        by the in-loop re-check (Step 4b of ``integrate``) and when a fixed
        ``error_integral_reference`` is dropped (Step 7).

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            The running record, in the order the subintervals were accepted,
            with the left mesh index of each in ``mesh_index``.
        cum_mesh_quadratures : torch.Tensor or None
            Running integral through each recorded subinterval [N, D], in the
            record's row order (cumulative mode only; None in absolute mode).
        integral : torch.Tensor
            Reference integral [D] for the absolute-mode tolerance.
        mesh : torch.Tensor
            The current mesh [M, T].
        split_counts : torch.Tensor or None
            Split count of each mesh position [M]; required when
            ``max_adaptive_splits`` is given.
        max_adaptive_splits : int or None
            The split cap, or None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple ``(split_positions, record_keep_mask, fails)``: the mesh
            positions of the failing recorded subintervals that can be split
            (in the record's row order), a mask [N] that keeps every other
            recorded subinterval, and the mask [N] of every failing one
            (including those that cannot be split).
        """
        _, rec_remove, _ = self._record_error_check(
            record, cum_mesh_quadratures, integral
        )
        rec_positions = record["mesh_index"]
        # A failing subinterval that cannot be split (at the split cap, or too
        # narrow to halve) is not split, so it must stay recorded: the record
        # stays consistent with the mesh.
        splittable = rec_remove & self._splittable(
            mesh, rec_positions, split_counts, max_adaptive_splits
        )
        return rec_positions[splittable], ~splittable, rec_remove

    def _warn_not_converged(
        self,
        record: dict[str, torch.Tensor],
        final_ratios: torch.Tensor,
        final_fails: torch.Tensor,
        nonfinite: torch.Tensor,
        mesh: torch.Tensor,
        split_counts: torch.Tensor,
        max_adaptive_splits: int | None,
        take_gradient: bool,
        enforce_convergence: bool,
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

        The remedies depend on why each finite failing subinterval was
        accepted, as ``_splittable`` tells: one too narrow to halve (its ends
        are neighbouring floating-point numbers) is named by its place, and
        only remedies that can help it are offered (another call from the
        state meets it again); one stopped by ``max_adaptive_splits`` and one
        accepted against a tolerance that changed afterwards can be refined by
        a higher cap or by another call from the state.

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
        mesh : torch.Tensor
            The final mesh [N + 1, T]; the record's row i is its subinterval i.
        split_counts : torch.Tensor
            Split count of the subinterval starting at each mesh point [N + 1].
        max_adaptive_splits : int | None
            The split cap used by the call, or None.
        take_gradient : bool
            Whether the call backpropagated per batch (then
            ``enforce_convergence`` is False).
        enforce_convergence : bool
            Whether accepted subintervals were re-checked as the integral
            changed.
        reference_pinned : bool
            Whether a fixed ``error_integral_reference`` still set the
            absolute-mode tolerance at the end (then accepted subintervals were
            not re-checked, and the loop judged them against that reference).
        """
        finite_fails = final_fails & ~nonfinite
        # Why each finite failing subinterval could not be refined:
        # _splittable without the cap tells the ones too narrow to halve, with
        # it also the ones at the cap.
        rows = torch.arange(len(final_fails), device=mesh.device)
        narrow = finite_fails & ~self._splittable(mesh, rows, None, None)
        refinable = finite_fails & ~narrow
        capped = refinable & ~self._splittable(
            mesh, rows, split_counts, max_adaptive_splits
        )
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
        if bool(capped.any()):
            remedies.append("raise max_adaptive_splits")
        if bool(refinable.any()) and not enforce_convergence:
            remedies.append(
                "pass take_gradient=False with enforce_convergence None or True "
                "(accepted steps are then re-checked)"
                if take_gradient
                else "pass enforce_convergence=True"
            )
        if bool(refinable.any()) and reference_pinned:
            remedies.append(
                "use an error_integral_reference no larger than |integral| "
                "(converged is judged against the final integral)"
            )
        if bool(refinable.any()):
            remedies.append("call again with the state it returned (state=state)")
        if bool(narrow.any()):
            places = mesh[:-1][narrow]
            remedies.append(
                f"for the {len(places)} failing subintervals as narrow as "
                f"{str(self.dtype).removeprefix('torch.')} allows (their ends "
                "are neighbouring floating-point numbers; they start at t = "
                f"{', '.join(f'{v:.17g}' for v in places[:5, 0].tolist())}"
                f"{f' and {len(places) - 5} more' if len(places) > 5 else ''}), "
                "which neither halving nor another call from the state can "
                "refine: loosen atol or rtol"
                + (", integrate in float64" if self.dtype == torch.float32 else "")
                + ", or pass a mesh= with a mesh point exactly at the "
                "discontinuity or singularity there; if the integrand is "
                "singular there, the integral may not exist"
            )
        if bool((final_fails & nonfinite).any()):
            remedies.append(
                "find where the integrand or the integral is NaN/Inf "
                "(runtime_checks=True raises at the first such value); "
                "refining cannot remove it"
            )
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

    def _splittable(
        self,
        mesh: torch.Tensor,
        idxs: torch.Tensor,
        split_counts: torch.Tensor | None,
        max_adaptive_splits: int | None,
        halvings: int = 1,
    ) -> torch.Tensor:
        """
        Tell which subintervals can be halved ``halvings`` times, on the device.

        A subinterval cannot be split when its split count plus ``halvings``
        would exceed ``max_adaptive_splits`` (with ``split_counts`` given), or
        when a mesh point the split inserts (``_midpoint`` of its neighbors,
        as in ``_adaptively_increase_mesh``) rounds onto a neighbor: one piece
        would have width zero and another would be the subinterval again.
        Either way it is accepted instead (or, for ``halvings=2``, halved
        once); the final check counts it as failing if its error ratio is 1
        or above. Every split goes through this test: the batch's
        (``_adaptively_increase_mesh``, including the optimal mesh's) and the
        re-check's (``_failed_record_positions``), so the mesh stays strictly
        increasing.

        Parameters
        ----------
        mesh : torch.Tensor
            The mesh. Shape: [M, T].
        idxs : torch.Tensor
            Mesh index of the left end of each subinterval. Shape: [N].
        split_counts : torch.Tensor | None
            Split count of the subinterval starting at each mesh point [M], or
            None (no cap).
        max_adaptive_splits : int | None
            The split cap, or None.
        halvings : int
            1 (default) to split in two, 2 to split in four (each half halved
            again).

        Returns
        -------
        torch.Tensor
            True for the subintervals that can be split. Shape: [N].
        """
        # The subinterval's ends with, in order, the points each halving
        # inserts; a piece has width zero where two neighbors are equal in
        # every coordinate.
        points = [mesh[idxs], mesh[idxs + 1]]
        for _ in range(halvings):
            mids = [_midpoint(a, b) for a, b in itertools.pairwise(points)]
            points = [*itertools.chain(*zip(points, mids, strict=False)), points[-1]]
        splittable = torch.stack(
            [torch.any(a != b, dim=-1) for a, b in itertools.pairwise(points)]
        ).all(dim=0)
        if max_adaptive_splits is not None and split_counts is not None:
            splittable &= split_counts[idxs] + halvings <= max_adaptive_splits
        return splittable

    def _adaptively_increase_mesh(
        self,
        method_output: MethodOutput | None,
        keep_mask: torch.Tensor,
        y_step_eval: torch.Tensor | None,
        nodes: torch.Tensor | None,
        mesh: torch.Tensor,
        mesh_idxs: torch.Tensor,
        mesh_trackers: torch.Tensor,
        tracked_step_eval: tuple[torch.Tensor, ...] | None = None,
        split_counts: torch.Tensor | None = None,
        max_adaptive_splits: int | None = None,
        extra_split_idxs: torch.Tensor | None = None,
        in_place: bool = False,
        error_ratios: torch.Tensor | None = None,
    ) -> tuple[
        MethodOutput | None,
        torch.Tensor | None,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Accept the accurate subintervals of a batch and split the others.

        Each evaluated subinterval is accepted where ``keep_mask`` is True
        (``_accept_reject_masks``: an error ratio below 1, or a non-finite
        one, which splitting cannot be relied on to remove): its tracker is
        set False and its contribution is kept. The others are split: a new
        mesh point is inserted at the midpoint (``_midpoint``), and both
        halves are evaluated in a later pass.

        With ``error_ratios`` (``split_rule="accelerate"``), a failing
        subinterval that is itself a piece of a split one (split count at
        least 1) and whose error ratio is at least 2**k, k the method's
        ``error_exponent``, is split in four instead: both halves are halved
        again in the same update, so the pieces are those of two successive
        halvings. Where the error estimate shrinks as h**k, a half's ratio is
        the parent's divided by 2**k (by 2**(k - 1) when the tolerance is the
        rounding floor, which shrinks with h), so both halves of such a
        subinterval would fail and be split again: no evaluation is spent on a
        four-way split whose halves would have passed.

        A subinterval that cannot be split (``_splittable``: its split count
        has reached ``max_adaptive_splits``, when both are given, or its
        midpoint rounds onto one of its ends) is accepted even if
        ``keep_mask`` is False; one that cannot be halved twice is halved
        once. The pieces of a subinterval with split count n get split count
        n + 1 (two pieces) or n + 2 (four pieces): the split count is the
        number of halvings.

        The update reads the device once, for the number of accepted
        subintervals and of those split in four; every other step indexes
        with integer positions on the device.

        Parameters
        ----------
        method_output : MethodOutput | None
            Integral and error output of the current batch (None in the
            optimal-mesh pass). If given, the split subintervals are removed
            from it.
        keep_mask : torch.Tensor
            True for the subintervals of the batch to accept, from
            ``_accept_reject_masks``. Shape: [N_batch].
        y_step_eval : torch.Tensor | None
            Integrand values of the batch. Shape: [N_batch, C, D].
        nodes : torch.Tensor | None
            Nodes of the batch. Shape: [N_batch, C, T].
        mesh : torch.Tensor
            All mesh points. Shape: [M, T].
        mesh_idxs : torch.Tensor
            Mesh index of the left end of each subinterval of the batch,
            ascending. Shape: [N_batch].
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
        extra_split_idxs : torch.Tensor | None
            Further mesh indices to split in the same update, besides the
            batch's rejected subintervals: recorded subintervals that failed
            the re-check after the running integral changed. In any order
            (they are sorted with the batch's); disjoint from the batch's
            ``mesh_idxs``. Their trackers are set True so both halves are
            evaluated. None splits only the batch's rejected subintervals.
        in_place : bool
            True to keep the accepted subintervals' values and tracked
            variables by moving them to the front of their arrays, which are
            overwritten (``_keep_rows``); only for arrays the solver allocated
            and nothing else reads. False (default) copies them.
        error_ratios : torch.Tensor | None
            Error ratio of each subinterval of the batch (norm family), given
            to split some in four (see above; requires ``split_counts``).
            Shape: [N_batch]. None (default) halves every split subinterval;
            ``extra_split_idxs`` are always halved.

        Returns
        -------
        tuple[MethodOutput | None, torch.Tensor | None, tuple[torch.Tensor, ...] | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
            Tuple ``(method_output, y_step_eval, tracked_step_eval, nodes,
            mesh_new, mesh_trackers_new, idxs_transfer, keep_idx,
            split_counts_new)``:

            - method_output: with the split subintervals removed.
            - y_step_eval: integrand values of the accepted subintervals.
            - tracked_step_eval: their tracked variables (or None).
            - nodes: their nodes.
            - mesh_new: the mesh with the new mesh points inserted.
            - mesh_trackers_new: the trackers, True for the new pieces.
            - idxs_transfer: the position in mesh_new of each point of
              ``mesh`` [M], increasing (``mesh_new[idxs_transfer]`` equals
              ``mesh``).
            - keep_idx: positions in the batch of the accepted subintervals,
              ascending (after the split cap), on the mesh's device.
            - split_counts_new: split counts for mesh_new, with the pieces of
              every split subinterval incremented by its number of halvings
              (or None if ``split_counts`` is None).
        """
        # Subintervals that cannot be split (at the split cap, or too narrow
        # to halve) are accepted as they are.
        keep_mask = keep_mask | ~self._splittable(
            mesh, mesh_idxs, split_counts, max_adaptive_splits
        )
        # How many times each subinterval of the batch is halved: 0 (accepted),
        # 1, or, with error_ratios, 2 (four pieces) for a failing piece of a
        # failed subinterval whose error ratio is at least 2**error_exponent
        # and that can be halved twice. The update's one read of the device:
        # how many are accepted and how many are split in four. Integer
        # positions index every field below without another read (a boolean
        # mask reads the device for the size of its result).
        halvings = (~keep_mask).to(torch.uint8)
        if error_ratios is None:
            n_keep, n_four = int(keep_mask.sum()), 0
        else:
            assert split_counts is not None  # given with error_ratios
            halvings += (
                ~keep_mask
                & (error_ratios >= 2.0**self.method.error_exponent)
                & (split_counts[mesh_idxs] > 0)
                & self._splittable(
                    mesh, mesh_idxs, split_counts, max_adaptive_splits, halvings=2
                )
            )
            n_keep, n_four = torch.stack(
                [keep_mask.sum(), (halvings == 2).sum()]
            ).tolist()
        # The batch's positions, accepted ones first, then those halved once,
        # then those split in four, each group ascending (a stable sort).
        order = torch.sort(halvings, stable=True).indices
        keep_idx = order[:n_keep]
        mesh_trackers.index_fill_(0, mesh_idxs[keep_idx], False)

        # Mesh positions to bisect this update: the batch's rejected steps plus
        # any extra positions (already-accepted panels that failed the re-check
        # against the drifted tolerance). The recorded panels were accepted
        # (tracker False), so re-activate the whole split set to ensure both
        # children of every bisected panel are re-evaluated.
        split_idxs = mesh_idxs[order[n_keep:]]
        four_idxs = split_idxs[len(split_idxs) - n_four :]
        if extra_split_idxs is not None and extra_split_idxs.numel() > 0:
            split_idxs = torch.sort(torch.cat([split_idxs, extra_split_idxs])).values
            # Batch (tracker True) and recorded (tracker False) positions are
            # disjoint, so the combined set must be strictly ascending; a
            # duplicate would corrupt the midpoint-offset math below.
            if self.runtime_checks:
                assert torch.all(split_idxs[1:] > split_idxs[:-1])
            mesh_trackers.index_fill_(0, split_idxs, True)

        # A split in four is two halvings: the first halves every split
        # subinterval, the second both halves of each four-piece one, so its
        # mesh points are those two successive halvings would insert.
        # idxs_transfer composes the transfers of the halvings.
        idxs_transfer: torch.Tensor | None = None
        for _ in range(1 + (n_four > 0)):
            if idxs_transfer is not None:
                # The halves start at the four-piece subinterval's own mesh
                # point and at the midpoint inserted right after it.
                left = idxs_transfer[four_idxs]
                split_idxs = torch.stack([left, left + 1], dim=-1).flatten()
            # Transfer existing barriers to their new positions in the expanded
            # array. Each split causes a +1 offset for all subsequent barriers
            # (because a midpoint is being inserted). idx_offset tracks this
            # shift. All allocations follow ``mesh.device`` so this runs on the
            # integration device in the loop and on result_device in the
            # post-convergence pass.
            N_t_add = len(split_idxs)
            idx_offset = torch.zeros(len(mesh), dtype=torch.long, device=mesh.device)
            idx_offset.index_fill_(0, split_idxs + 1, 1)
            step_transfer = torch.cumsum(idx_offset, dim=0) + torch.arange(
                len(mesh), device=mesh.device
            )
            # The left mesh point of split subinterval j moves to split_left[j]
            # = split_idxs[j] + the number of split subintervals to its left
            # (one midpoint is inserted before it for each), in any order of
            # split_idxs; its midpoint goes right after it, at idxs_new[j], and
            # its right mesh point follows. So
            # step_transfer and idxs_new are disjoint and together cover the
            # new mesh, and every entry of the empty arrays below is written.
            split_left = step_transfer[split_idxs]
            idxs_new = split_left + 1
            mesh_new = mesh.new_empty((N_t_add + len(mesh), mesh.shape[-1]))
            mesh_new[step_transfer] = mesh
            mesh_new[idxs_new] = _midpoint(mesh_new[split_left], mesh_new[idxs_new + 1])
            mesh_trackers_new = torch.ones(
                len(mesh_new), dtype=torch.bool, device=mesh.device
            )
            mesh_trackers_new[step_transfer] = mesh_trackers

            # Expand the per-panel split counts the same way as the barriers:
            # carry existing counts to their new positions, then set both
            # children of each split panel to parent_count + 1.
            if split_counts is not None:
                split_counts_new = split_counts.new_empty(len(mesh_new))
                split_counts_new[step_transfer] = split_counts
                child_counts = split_counts[split_idxs] + 1
                # left child = transferred parent barrier; right child = midpoint
                split_counts_new[split_left] = child_counts
                split_counts_new[idxs_new] = child_counts
            else:
                split_counts_new = None
            idxs_transfer = (
                step_transfer if idxs_transfer is None else step_transfer[idxs_transfer]
            )
            mesh, mesh_trackers, split_counts = (
                mesh_new,
                mesh_trackers_new,
                split_counts_new,
            )
        assert idxs_transfer is not None  # the loop runs at least once

        if method_output is not None:
            method_output.mesh_quadratures = method_output.mesh_quadratures[keep_idx]
            method_output.mesh_quadrature_errors = method_output.mesh_quadrature_errors[
                keep_idx
            ]
            method_output.h = method_output.h[keep_idx]
            method_output.integral = torch.sum(method_output.mesh_quadratures, 0)
            method_output.integral_error = torch.sum(
                method_output.mesh_quadrature_errors, 0
            )
        # The values are on the solver's device, or on result_device with
        # trade_speed_for_memory. Indexing would copy the accepted ones, so the
        # values would be held twice (measured: the peak of a pass of 1,000 gk21
        # subintervals with 1e5 outputs): a batch whose subintervals are all
        # accepted is kept as it is, and otherwise, with in_place, the accepted
        # rows are moved within the arrays. The nodes are small and copied.
        all_kept = n_keep == len(keep_mask)
        if y_step_eval is not None and not all_kept:
            y_step_eval = _keep_rows(y_step_eval, keep_idx, in_place)
        if tracked_step_eval is not None and not all_kept:
            tracked_step_eval = tuple(
                _keep_rows(tv, keep_idx, in_place) for tv in tracked_step_eval
            )
        if nodes is not None and not all_kept:
            nodes = nodes[keep_idx]
        return (
            method_output,
            y_step_eval,
            tracked_step_eval,
            nodes,
            mesh_new,
            mesh_trackers_new,
            idxs_transfer,
            keep_idx,
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
        Runs only when ``self.runtime_checks`` is True (set from the
        ``integrate(runtime_checks=...)`` argument or the constructor).

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
            If ``self.runtime_checks`` is True and any value is NaN or
            Inf. When it is False this check does nothing, and
            ``_accept_reject_masks`` accepts the non-finite subintervals so
            the loop still ends.
        """
        if not self.runtime_checks:
            return
        if not torch.is_tensor(integrand):
            # Bare scalar (e.g. examples.identity returns int 1). math.isfinite
            # handles int/float; there is no node to localize to.
            if not math.isfinite(integrand):
                raise ValueError(
                    f"Integrand f returned a non-finite scalar value "
                    f"({integrand}). Pass runtime_checks=False to return a "
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
            f"runtime_checks=False to accept the affected panels and return "
            f"a NaN-containing result instead of raising."
        )

    def _evaluate_f_on_mesh(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        mesh: torch.Tensor,
        step_idxs: torch.Tensor,
        take_gradient: bool,
        values_device: torch.device | None,
        buffer_bytes: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor,
        MethodOutput,
    ]:
        """
        Evaluate the integrand on the next unevaluated subintervals and integrate them.

        Chooses the size of the next call of f with ``_batch_for`` (the call's
        batch size, or 10 times it while it has not converged, at most the
        evaluations left); then dispatches to ``_evaluate_f_on_full_nodes``
        when ``take_gradient`` is True (whole subintervals, one call), else to
        ``_evaluate_f_on_split_nodes`` (every waiting subinterval).

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
            evaluation, ascending, on the solver's device. Shape: [K], K >= 1.
        take_gradient : bool
            If True, evaluate whole subintervals with one call of ``f`` per batch.
        values_device : torch.device or None
            The device the integrand values and tracked variables are returned
            on (with ``take_gradient`` True, always the solver's device); None
            drops them (``record_evaluations=False``).
        buffer_bytes : float
            With ``take_gradient`` False, see ``_evaluate_f_on_split_nodes``.

        Returns
        -------
        tuple
            ``(nodes, f_evals, tracked, step_idxs, method_output)``: the nodes
            [N, C, T] (on the solver's device) and integrand values [N, C, D] of
            the N evaluated subintervals, their detached tracked variables (each
            [N, C, ...]) or None, the indices of their left mesh points [N],
            ascending, and their integral, error estimate, contributions and
            widths (``MethodOutput``, on the solver's device). ``f_evals`` and
            ``tracked`` are None when ``values_device`` is None.
        """
        # The size of the next call of f: this call's batch size, or 10 times
        # it while it has not converged (_batch_for).
        max_batch = self._batch_for(len(step_idxs) * self.C)
        if take_gradient and not self._batch_converged:
            # Whole subintervals: a small batch size grows to at least one.
            max_batch = max(max_batch, self.C)

        if take_gradient:
            nodes, f_evals, tracked, step_idxs, method_output = (
                self._evaluate_f_on_full_nodes(
                    f, f_args, mesh, step_idxs, max_batch // self.C
                )
            )
            if values_device is None:
                return nodes, None, None, step_idxs, method_output
            return nodes, f_evals, tracked, step_idxs, method_output
        return self._evaluate_f_on_split_nodes(
            f, f_args, mesh, step_idxs, max_batch, values_device, buffer_bytes
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
        MethodOutput,
    ]:
        """
        Evaluate the integrand on the first max_mesh_steps subintervals in one call.

        Used when ``take_gradient`` is True. When f runs out of memory, or, on
        a GPU, when the usable memory after the call is below what the backward
        pass is predicted to need (the call's peak memory times the ratio the
        previous batch's backward pass measured, 1 at first), the outputs are
        dropped (no gradient was written yet) and fewer subintervals are
        evaluated; the call's batch size then stops growing.

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
            Largest number of subintervals to evaluate, at least 1
            (``integrate`` refuses a batch size below C points that cannot
            grow).

        Returns
        -------
        tuple
            ``(nodes, f_evals, tracked, step_idxs, method_output)``: the nodes
            [N, C, T] and integrand values [N, C, D] of the N evaluated
            subintervals, their detached tracked variables (each [N, C, ...]) or
            None, the indices of their left mesh points [N], ascending, and
            their integral, error estimate, contributions and widths.
        """
        cuda = self.device.type == "cuda"
        self._forward_bytes = 0.0  # this batch's, measured below
        # Evaluate the first max_mesh_steps subintervals; while the batch is
        # dropped (f ran out of memory, or the backward pass is predicted not
        # to fit), evaluate again with fewer. No gradient was written yet, so
        # a dropped batch leaves nothing behind. Every retry has fewer
        # subintervals, and a batch of one is always kept (or f's error is
        # raised), so the loop ends.
        output = None
        while output is None:
            # Place C quadrature points within each of the first max_mesh_steps
            # steps; flatten [N, C, T] -> [N*C, T] for one call of f.
            idxs = step_idxs[:max_mesh_steps]
            mesh_left, mesh_right = mesh[idxs], mesh[idxs + 1]
            nodes = self._compute_nodes(mesh_left, mesh_right)
            nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=-2)
            if cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
                before = torch.cuda.memory_allocated(self.device)
            output = self._call_f(f, f_args, nodes_flat, max_mesh_steps > 1)
            if output is None:
                n, why = _shrink_batch(len(nodes_flat)), "f ran out of memory"
            elif cuda and output[0].requires_grad:
                # The backward pass needs about as much memory as this forward
                # pass, times the ratio the previous batch's backward measured.
                self._forward_bytes = (
                    torch.cuda.max_memory_allocated(self.device) - before
                )
                need = self._backward_ratio * self._forward_bytes
                usable = self._get_usable_memory(self._total_mem_usage) * _GB
                if usable < need and max_mesh_steps > 1:
                    output = None
                    n = min(
                        _shrink_batch(len(nodes_flat)),
                        int(len(nodes_flat) * usable / need),
                    )
                    why = "its backward pass is predicted not to fit in memory"
            if output is None:
                max_mesh_steps = max(1, n // self.C)
                self._batch, self._batch_converged = max_mesh_steps * self.C, True
                logger.warning(
                    f"A batch of {len(nodes_flat)} points was dropped because "
                    f"{why}; it is evaluated again in batches of {self._batch} "
                    "points."
                )
        self._batch = max(self._batch, len(nodes_flat))
        step_idxs = idxs
        node_evals, tracked = output
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

        return (
            nodes,
            f_evals,
            tracked_out,
            step_idxs,
            self._calculate_integral(nodes, f_evals, mesh_left, mesh_right),
        )

    def _evaluate_f_in_batches(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        nodes_flat: torch.Tensor,
        max_batch: int,
    ) -> Iterator[tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]]:
        """
        Call f on all of nodes_flat, in order, at most max_batch nodes per call.

        Used by ``_evaluate_f_on_split_nodes``. A call may end inside a
        subinterval. On an out-of-memory error the failed call is retried at
        ``_shrink_batch`` of its size, which becomes the call's converged batch
        size ``_batch``, before anything is yielded; after a call that ran, the
        next one may grow (``_batch_for``).

        Parameters
        ----------
        f : callable
            The integrand.
        f_args : tuple
            Extra arguments passed to ``f`` after the nodes.
        nodes_flat : torch.Tensor
            The nodes to evaluate, in order. Shape: [K, T].
        max_batch : int
            Largest number of nodes per call of ``f``, at least 1.

        Yields
        ------
        tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]
            Each call's integrand values, shape [n, D], and its tracked
            variables (each [n, ...]) or None, in node order; together the
            calls cover all of ``nodes_flat``.

        Raises
        ------
        ValueError
            If f's values are not one row per point (``_call_f``), or (with
            ``runtime_checks``) contain a NaN or Inf.
        torch.OutOfMemoryError
            If one node does not fit in memory (max_batch is already 1).
        """
        # integrate's batch size is always at least 1 (max_batch is validated,
        # and every shrink stops at 1); this guards direct callers.
        assert max_batch > 0
        num_nodes_evaluated = 0
        while num_nodes_evaluated < len(nodes_flat):
            # The size of this call (a retry shrinks it, not a larger plan).
            max_batch = min(max_batch, len(nodes_flat) - num_nodes_evaluated)
            points = nodes_flat[num_nodes_evaluated : num_nodes_evaluated + max_batch]
            f_output = self._call_f(f, f_args, points, max_batch > 1)
            if f_output is None:
                # Shrink the batch that failed; the batch size of the call
                # stops growing (see _batch_for).
                free_mem, total_mem = self._get_memory()
                self._batch, self._batch_converged = _shrink_batch(max_batch), True
                logger.warning(
                    f"Caught OOM in a call of {max_batch} points ({free_mem} GB "
                    f"free of {total_mem} GB after freeing its memory). Reducing "
                    f"max_batch from {max_batch} to {self._batch}. If this warning "
                    "appears often, consider reducing max_batch to avoid deleting "
                    "and rerunning evaluations."
                )
                max_batch = self._batch
                continue

            integrand, tracked = f_output
            self._check_f_output_finite(integrand, points)
            num_nodes_evaluated += len(integrand)
            # A call that ran is known to fit; the next call may be 10 times
            # larger while the batch size has not converged (_batch_for).
            self._batch = max(self._batch, len(integrand))
            max_batch = max(max_batch, self._batch_for(10**18))
            yield integrand, tracked
            # Release this call's output before the next call of f, so the
            # device never holds two calls' values because of this frame.
            del f_output, integrand, tracked

    def _evaluate_f_on_split_nodes(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        mesh: torch.Tensor,
        step_idxs: torch.Tensor,
        max_batch: int,
        values_device: torch.device | None,
        buffer_bytes: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        tuple[torch.Tensor, ...] | None,
        torch.Tensor,
        MethodOutput,
    ]:
        """
        Evaluate and integrate all the given subintervals, max_batch nodes per call.

        Used when ``take_gradient`` is False. The calls of f run through the
        nodes of every given subinterval in order and may end inside one. The
        values of the subintervals whose nodes are all evaluated wait on the
        solver's device until they reach ``buffer_bytes`` (or the last call
        ends); then their contributions, error estimates and widths are
        computed (``_calculate_integral``, on the solver's device), and their
        values and tracked variables go to ``values_device``. Each
        subinterval's numbers depend only on its own nodes and values, so they
        are the same, bit for bit, whatever the buffer; the pass's integral is
        their sum.

        When the values stay on the solver's device for the whole pass (when
        ``values_device`` is that device, or is None with ``buffer_bytes``
        infinite) and f's output does not require grad, the values are written, as each call returns, into one
        array for the whole pass (its size is known before the first call:
        no subinterval is split during a pass), and the waiting values are
        integrated from its rows. The device then holds one copy of the
        pass's values plus one call's output, not a list of the calls' values
        and their joined copy. Otherwise the waiting values are kept in a
        list and joined when they are integrated.

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
        values_device : torch.device or None
            Device of the returned integrand values and tracked variables: the
            solver's device (``trade_speed_for_memory=False``), or
            ``result_device`` (True), so the solver's device holds about one
            call's values plus the buffer instead of the whole pass's. None
            drops them once integrated (``record_evaluations=False``).
        buffer_bytes : float
            Bytes of values of whole subintervals that wait before they are
            integrated: ``math.inf`` integrates the whole pass at once, after
            its last call of f (``trade_speed_for_memory=False``); 0 after
            every call that completes a subinterval.

        Returns
        -------
        tuple
            ``(nodes, f_evals, tracked, step_idxs, method_output)``: the nodes
            [K, C, T] of the K subintervals, on the solver's device; their
            integrand values [K, C, D] and detached tracked variables (each
            [K, C, ...]) or None, on ``values_device`` (both None when it is
            None); ``step_idxs``; and their integral, error estimate,
            contributions and widths (``MethodOutput``, on the solver's
            device).

        Raises
        ------
        torch.OutOfMemoryError
            If the array for the pass's values does not fit in memory, with
            the remedies in its message.
        """
        C = self.C
        mesh_left, mesh_right = mesh[step_idxs], mesh[step_idxs + 1]
        nodes = self._compute_nodes(mesh_left, mesh_right)
        nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=-2)

        # Per integration of waiting values: the contributions and error
        # estimates of their subintervals (solver's device). The values and
        # tracked variables go into arrays for the whole pass on values_device
        # (f_evals, tracked_out): written as each call returns when in_place,
        # else at each integration, unless the first integration covers the
        # whole pass, whose values are then kept as they are (no copy on the
        # same device). in_place is decided at the first whole subinterval.
        quadrature_parts: list[torch.Tensor] = []
        error_parts: list[torch.Tensor] = []
        f_evals: torch.Tensor | None = None
        tracked_out: tuple[torch.Tensor, ...] | None = None
        in_place: bool | None = None
        # Without in_place: values (and tracked variables) of whole
        # subintervals not integrated yet, [n, C, D] each. In both cases the
        # waiting subintervals are n_done to n_filled, of waiting_bytes bytes.
        waiting: list[torch.Tensor] = []
        waiting_tracked: list[tuple[torch.Tensor, ...]] = []
        waiting_bytes = 0
        n_filled = 0  # subintervals whose nodes are all evaluated
        # The values (and tracked variables) of a subinterval the last call
        # ended inside: fewer than C rows, joined to the next call's.
        leftover: torch.Tensor | None = None
        leftover_tracked: tuple[torch.Tensor, ...] | None = None
        n_done = 0  # subintervals integrated so far
        # A final (None, None) after the last call integrates what still waits.
        for integrand, tracked in itertools.chain(
            self._evaluate_f_in_batches(f, f_args, nodes_flat, max_batch),
            [(None, None)],
        ):
            if integrand is not None:
                values = (
                    integrand if leftover is None else torch.cat([leftover, integrand])
                )
                tracked_values = None
                if tracked is not None:
                    tracked_values = tuple(tv.detach() for tv in tracked)
                    if leftover_tracked is not None:
                        tracked_values = tuple(
                            torch.cat([old, new])
                            for old, new in zip(
                                leftover_tracked, tracked_values, strict=True
                            )
                        )
                n_whole = len(values) // C
                if n_whole > 0:
                    whole = values[: n_whole * C].reshape(n_whole, C, -1)
                    whole_tracked = (
                        None
                        if tracked_values is None
                        else tuple(
                            tv[: n_whole * C].reshape(n_whole, C, *tv.shape[1:])
                            for tv in tracked_values
                        )
                    )
                    if in_place is None:
                        # When the pass's values stay on the solver's device
                        # anyway: kept there (values_device), or integrated
                        # once per pass and dropped (no buffer, no record).
                        # An output that requires grad keeps the list: writing
                        # it into an array the integration reads would put
                        # every call in one autograd graph of in-place writes.
                        in_place = not whole.requires_grad and (
                            values_device == self.device
                            or (values_device is None and buffer_bytes == math.inf)
                        )
                    if in_place and f_evals is None and n_whole == len(step_idxs):
                        # One call covers the pass: its values are the pass's
                        # array, without a copy. They are a view of f's
                        # output, which _keep_rows copies from, not overwrites.
                        f_evals, tracked_out = whole, whole_tracked
                    elif in_place:
                        if f_evals is None:
                            try:
                                f_evals = whole.new_empty(
                                    (len(step_idxs), *whole.shape[1:])
                                )
                                if whole_tracked is not None:
                                    tracked_out = tuple(
                                        tv.new_empty((len(step_idxs), *tv.shape[1:]))
                                        for tv in whole_tracked
                                    )
                            except (RuntimeError, MemoryError) as e:
                                if not _is_out_of_memory(e):
                                    raise
                                gb = len(step_idxs) * whole[0].nbytes / _GB
                                raise torch.OutOfMemoryError(
                                    f"{e}\n\nThe integrand values of this pass "
                                    f"({len(step_idxs)} subintervals, {gb:.3g} GB) "
                                    f"do not fit in one array on {self.device}. "
                                    "Pass trade_speed_for_memory=True with "
                                    "record_evaluations=False or result_device="
                                    "'cpu', so that about one call's values are "
                                    "held at a time. On a GPU, when the error above "
                                    "reports much memory reserved but unallocated, "
                                    "setting the environment variable "
                                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                                    "before CUDA is first used can also help. See "
                                    "https://khegazy.github.io/padaquad/user_guide/"
                                    "devices_memory.html#ug-pass-array-oom"
                                ) from e
                        f_evals[n_filled : n_filled + n_whole] = whole
                        if whole_tracked is not None:
                            assert tracked_out is not None
                            for out_tv, tv in zip(
                                tracked_out, whole_tracked, strict=True
                            ):
                                out_tv[n_filled : n_filled + n_whole] = tv
                    else:
                        waiting.append(whole)
                        if whole_tracked is not None:
                            waiting_tracked.append(whole_tracked)
                    n_filled += n_whole
                    waiting_bytes += whole.numel() * whole.element_size()
                    del whole, whole_tracked
                # The rows past the last whole subinterval; a copy, so the
                # slice does not keep this call's values alive.
                has_rest = len(values) > n_whole * C
                leftover = values[n_whole * C :].clone() if has_rest else None
                leftover_tracked = (
                    tuple(tv[n_whole * C :].clone() for tv in tracked_values)
                    if has_rest and tracked_values is not None
                    else None
                )
                del values, tracked_values, integrand, tracked
                if waiting_bytes < buffer_bytes or n_filled == n_done:
                    continue
            elif n_filled == n_done:
                break

            # Integrate the waiting subintervals and send their values on.
            if in_place:
                assert f_evals is not None
                block = f_evals[n_done:n_filled]
                block_tracked = (
                    None
                    if tracked_out is None
                    else tuple(tv[n_done:n_filled] for tv in tracked_out)
                )
            else:
                block = waiting[0] if len(waiting) == 1 else torch.cat(waiting)
                block_tracked = (
                    None
                    if not waiting_tracked
                    else tuple(
                        tv[0] if len(tv) == 1 else torch.cat(tv)
                        for tv in zip(*waiting_tracked, strict=True)
                    )
                )
            waiting, waiting_tracked, waiting_bytes = [], [], 0
            n_block = len(block)
            rows = slice(n_done, n_done + n_block)
            out = self._calculate_integral(
                nodes[rows], block, mesh_left[rows], mesh_right[rows]
            )
            quadrature_parts.append(out.mesh_quadratures)
            error_parts.append(out.mesh_quadrature_errors)
            if in_place:
                pass  # already in f_evals
            elif values_device is not None and n_block == len(step_idxs):
                f_evals = block.to(values_device)
                if block_tracked is not None:
                    tracked_out = tuple(tv.to(values_device) for tv in block_tracked)
            elif values_device is not None:
                if f_evals is None:
                    f_evals = block.new_empty(
                        (len(step_idxs), *block.shape[1:]), device=values_device
                    )
                f_evals[n_done : n_done + n_block] = block
                if block_tracked is not None:
                    if tracked_out is None:
                        tracked_out = tuple(
                            tv.new_empty(
                                (len(step_idxs), *tv.shape[1:]), device=values_device
                            )
                            for tv in block_tracked
                        )
                    for out_tv, tv in zip(tracked_out, block_tracked, strict=True):
                        out_tv[n_done : n_done + n_block] = tv
            n_done += n_block
            del block, block_tracked, out
        assert leftover is None
        assert n_done == len(step_idxs)

        mesh_quadratures = torch.cat(quadrature_parts)
        mesh_quadrature_errors = torch.cat(error_parts)
        del quadrature_parts, error_parts
        # The same sums and widths as _calculate_integral's (y0 = 0) over the
        # whole pass.
        method_output = MethodOutput(
            integral=torch.sum(mesh_quadratures, 0),
            integral_error=torch.sum(mesh_quadrature_errors, 0),
            mesh_quadratures=mesh_quadratures,
            mesh_quadrature_errors=mesh_quadrature_errors,
            h=mesh_right - mesh_left,
        )
        if values_device is None:
            f_evals, tracked_out = None, None  # the in_place arrays are dropped
        return nodes, f_evals, tracked_out, step_idxs, method_output

    #: log2 of a pair's width ratio is rounded to a multiple of 2**-48 before
    #: pairs are grouped by ratio in ``_predict_merged_errors``. Pairs of exactly
    #: equal widths share one set of weight vectors; widths computed from mesh
    #: points differ in their last bits, so a mesh of equal subintervals of
    #: [0, 1] still has a few tens of distinct ratios (29 at 10^5, measured).
    #: Half a step moves the prediction by at most 0.24 of a pair's rounding
    #: floor (measured; PR #69).
    _RATIO_QUANTUM = 2.0**48
    #: Pairs per chunk in ``_predict_merged_errors`` at D = 1 (``_MERGE_CHUNK //
    #: D`` for D outputs); bounds its float64 temporaries for any record.
    _MERGE_CHUNK = 2**16
    #: Largest number of width ratios kept in ``_merge_weight_cache``; a chunk
    #: with more distinct ratios than this builds its vectors without caching.
    _MERGE_CACHE_SIZE = 256
    #: Width ratios per block in ``_merged_pair_weights``. When every pair has
    #: its own ratio, the prediction's temporaries peak at about 0.25 GB for
    #: gk21 and 0.7 GB for cc65 (measured; PR #69).
    _RATIO_BLOCK = 1024

    def _merge_inputs(
        self,
        y: torch.Tensor | None,
        mesh_quadratures: torch.Tensor,
        mesh: torch.Tensor,
    ) -> tuple[torch.Tensor | None, float]:
        """
        Return the merge prediction and the cutoff that decide which pairs merge.

        A pair of neighboring subintervals is merged when its pair ratio (the
        pair error divided by the tolerance the merged subinterval would get,
        from ``_compute_error_ratios``) is below the cutoff. ``merge_rule``
        decides both. With e_i the error estimate of subinterval i:

        - ``"sum"`` (PadaQuad 1.2.2): pair error ``|e_i| + |e_{i+1}|``,
          cutoff ``remove_cut``.
        - ``"power_law"`` (PadaQuad 1.3.0 and 1.4.0): the same pair error, cutoff
          ``remove_cut / 2**(k - 1)`` with ``k = method.error_exponent``. If a
          subinterval's error estimate scaled as ``h**k`` with its width h,
          the merged subinterval of width 2h would have about ``2**(k - 1)``
          times the pair's summed error (PadaQuad's own derivation; the
          ``h**k`` scaling of an embedded pair's estimate is that of its
          lower-degree method, :cite:p:`gonnet2012` Eqs. (4)-(5) and
          footnote 20). It holds only once the subintervals are narrow enough
          for the leading Taylor term to dominate; on wider ones the error
          grows far less, and the rule refuses safe merges.
        - ``"predicted"`` (default): pair error
          ``_predict_merged_errors(...) + |e_i| + |e_{i+1}|``, cutoff
          ``remove_cut``. With more than one coordinate (T > 1) no
          interpolation along a subinterval is defined, so a warning is
          logged and ``"power_law"`` is used; without stored values
          (``integrate(record_evaluations=False)``, which warns), ``"power_law"``
          is used too.

        Parameters
        ----------
        y : torch.Tensor or None
            Integrand values at the nodes, shape [N, C, D], or None when they
            were not kept.
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh : torch.Tensor
            The mesh whose N subintervals these are. Shape: [N + 1, T].

        Returns
        -------
        tuple[torch.Tensor | None, float]
            ``(merge_prediction, merge_cut)``: ``_predict_merged_errors`` for
            ``"predicted"``, shape [N-1, D], which ``_get_optimal_mesh`` adds
            to the summed errors, or None; and the cutoff on the pair ratio.
        """
        if self.merge_rule == "sum":
            return None, self.remove_cut
        power_law_cut = self.remove_cut / 2 ** (self.method.error_exponent - 1)
        if self.merge_rule == "power_law" or y is None or len(mesh) < 3:
            return None, power_law_cut
        if mesh.shape[-1] != 1:
            logger.warning(
                "merge_rule='predicted' needs a one-coordinate integration "
                "variable (T == 1); T = %d here, so the optimal mesh is built "
                "with merge_rule='power_law'.",
                mesh.shape[-1],
            )
            return None, power_law_cut
        return (
            self._predict_merged_errors(y, mesh_quadratures, mesh),
            self.remove_cut,
        )

    def _merged_pair_weights(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Weight vectors that give a merged subinterval's integral and error estimate.

        For a pair whose right neighbor is ``rho`` times as wide as its left
        one, each neighbor's stored values are replaced by the polynomial
        through its distinct nodes, in barycentric form
        (:cite:p:`berrut2004`, weights Eq. (3.2), formula Eq. (4.2)). These
        polynomials are evaluated at the merged subinterval's nodes and the
        method's weights b and error weights b_err are applied. Both results
        are linear in the stored values, so with ``y_L`` and ``y_R`` the
        neighbors' values at their nodes and H the merged width::

            Q_hat = H * (alpha_L . y_L + alpha_R . y_R)
            E_hat = H * (gamma_L . y_L + gamma_R . y_R)

        and the vectors depend on the pair only through ``rho``. A method may
        repeat a node fraction; its stored values there are equal, and the
        repeats get weight 0, so the vectors apply to the values as stored.

        Parameters
        ----------
        rho : torch.Tensor
            Width of the right neighbor divided by that of the left. Shape: [R].

        Returns
        -------
        torch.Tensor
            The four vectors in float64, shape [R, 2, 2, C]: ``[:, 0]`` is
            ``(alpha_L, gamma_L)`` and ``[:, 1]`` is ``(alpha_R, gamma_R)``.
        """
        if len(rho) > self._RATIO_BLOCK:
            return torch.cat(
                [
                    self._merged_pair_weights(rho[i : i + self._RATIO_BLOCK])
                    for i in range(0, len(rho), self._RATIO_BLOCK)
                ]
            )
        c = self.method.node_fractions().to(device=rho.device, dtype=torch.float64)
        # The polynomials go through the U distinct fractions. They are
        # non-decreasing (MethodClass checks it), so repeats are neighbors.
        new = torch.ones_like(c, dtype=torch.bool)
        new[1:] = c[1:] != c[:-1]
        first = torch.nonzero(new)[:, 0]
        u = c[first]
        diff = u[:, None] - u[None, :]
        diff.fill_diagonal_(1.0)
        bary = 1.0 / diff.prod(dim=1)  # Eq. (3.2); a common factor cancels
        bary = bary / bary.abs().max()

        # Merged nodes with the left neighbor at [0, 1] and the right one at
        # [1, 1 + rho]. A merged node inside one neighbor takes that
        # neighbor's polynomial; one between a neighbor's outermost node and
        # its end (Gauss-Kronrod has no node at the ends) is extrapolated
        # slightly: Eq. (4.2) holds at every x that is not a node, inside or
        # outside the nodes, because the identity it divides by, Eq. (4.1),
        # does. A merged node on the shared point 1 (the middle node when
        # rho = 1) takes the mean of both neighbors' polynomials: when 1 is not
        # a node, their extrapolations err in opposite directions where the
        # nodes are rounded (float32 far from 0: the prediction was 100 times
        # larger with the left one alone, measured 2026-11-26).
        x = c[None, :] * (1.0 + rho[:, None])  # [R, C]
        left_share = torch.where(x < 1.0, 1.0, torch.where(x == 1.0, 0.5, 0.0))
        b, b_err = self.method.weights(
            x.unsqueeze(-1).to(self.method.node_fractions().dtype)
        )
        # [R, 2 (b, b_err), C, 1]
        bb = torch.stack([b, b_err], dim=1).to(torch.float64)
        sides = []
        for s, share in ((x, left_share), ((x - 1.0) / rho[:, None], 1.0 - left_share)):
            to_node = s[..., None] - u  # [R, C, U]
            on_node = to_node == 0
            hits = on_node.any(dim=-1, keepdim=True)
            terms = torch.where(
                hits,
                on_node.to(torch.float64),
                bary / torch.where(on_node, 1.0, to_node),
            )
            basis = terms / terms.sum(dim=-1, keepdim=True)  # Eq. (4.2), [R, C, U]
            # [R, 2, U]: summed over the merged nodes, each by its share (a
            # merged node far outside this neighbor, share 0, can make its
            # basis non-finite, so it is masked rather than multiplied by 0)
            shared = torch.where(share[..., None] > 0, basis * share[..., None], 0.0)
            sides.append((bb * shared[:, None]).sum(dim=2))
        # Each distinct fraction's vectors at its first node, 0 at its repeats.
        weights = torch.zeros(
            len(rho), 2, 2, len(c), dtype=torch.float64, device=rho.device
        )
        weights[..., first] = torch.stack(sides, dim=1)
        return weights

    def _predict_merged_errors(
        self,
        y: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        mesh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the errors of merging each pair of neighboring subintervals.

        For pair i (subintervals i and i + 1), the merged subinterval's
        integral ``Q_hat`` and error estimate ``E_hat`` are computed from the
        integrand values already stored for the two neighbors
        (``_merged_pair_weights``); no integrand evaluation is needed. The
        prediction is::

            max(|E_hat|, |Q_hat - (Q_i + Q_{i+1})|)

        - ``|E_hat|`` predicts whether the merged subinterval will pass its
          own error estimate when it is evaluated on the next call.
        - ``|Q_hat - (Q_i + Q_{i+1})|`` compares the merged rule, applied to
          the neighbors' interpolants, with the sum of the two neighbors: the
          halves-versus-whole test of adaptive Simpson quadrature
          (:cite:p:`gander2000`, Sec. 3), used here to decide coarsening. When
          the stored values resolve f, it approximates the merged
          subinterval's true error, and it catches merges whose own error
          estimate would be blind (a kink or a narrow feature that the merged
          nodes skip over).

        A feature that lies between the stored nodes is invisible to every
        term, and so to every rule built from stored values.

        Parameters
        ----------
        y : torch.Tensor
            Integrand values at the nodes of each subinterval, which are at
            the method's node fractions of its subinterval of ``mesh``
            (``_compute_nodes`` places every node there). Shape: [N, C, D],
            N >= 2.
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh : torch.Tensor
            The mesh whose N subintervals these are, in order; the widths come
            from it (a method's nodes need not include the ends). Shape:
            [N + 1, 1].

        Returns
        -------
        torch.Tensor
            The prediction for each pair, in the dtype of ``y``; infinite for
            a pair whose width ratio is not finite. Shape: [N-1, D].
        """
        # The record keeps the autograd graph when take_gradient=False; the
        # prediction is only a decision.
        y, mesh = y.detach(), mesh.detach()
        mesh_quadratures = mesh_quadratures.detach()
        mesh64 = mesh[:, 0].to(torch.float64)
        widths = mesh64[1:] - mesh64[:-1]
        log_ratio = torch.log2(widths[1:] / widths[:-1]) * self._RATIO_QUANTUM
        # A non-finite ratio (zero, overflowing or NaN widths) has no int64 key
        # (the conversion is undefined); such a pair is never merged (below).
        finite = torch.isfinite(log_ratio)
        key = torch.round(torch.where(finite, log_ratio, 0.0)).to(torch.int64)
        n_pairs = len(widths) - 1
        out = torch.empty(n_pairs, y.shape[-1], dtype=y.dtype, device=y.device)
        chunk_size = max(1, self._MERGE_CHUNK // y.shape[-1])
        for lo in range(0, n_pairs, chunk_size):
            hi = min(lo + chunk_size, n_pairs)
            # Vectors only for this chunk's ratios, so memory stays bounded
            # even when every pair has its own ratio.
            keys, inverse = torch.unique(key[lo:hi], return_inverse=True)
            pair_weights = self._cached_merged_pair_weights(keys)[inverse]
            # [P, C, D]: the stored values, views when float64 and contiguous.
            # f's output is recorded as it is, so it can be strided (a
            # transposed output); bmm on strided values rounds differently.
            y_l = y[lo:hi].contiguous().to(torch.float64)
            y_r = y[lo + 1 : hi + 1].contiguous().to(torch.float64)
            width = (widths[lo:hi] + widths[lo + 1 : hi + 1])[:, None, None]
            # [P, 2, D]: row 0 the merged integral Q_hat, row 1 its error E_hat
            q_e_hat = width * (
                torch.bmm(pair_weights[:, 0], y_l) + torch.bmm(pair_weights[:, 1], y_r)
            )
            q_halves = (mesh_quadratures[lo:hi] + mesh_quadratures[lo + 1 : hi + 1]).to(
                torch.float64
            )
            out[lo:hi] = torch.maximum(
                q_e_hat[:, 1].abs(), (q_e_hat[:, 0] - q_halves).abs()
            ).to(y.dtype)
        out[~finite] = torch.inf
        return out

    def _cached_merged_pair_weights(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Return ``_merged_pair_weights`` for quantized width ratios, cached.

        Adaptive meshes repeat a few width ratios (1, 2 and 1/2 from bisection,
        plus those of the starting mesh), and building the vectors costs more
        than applying them on a small mesh, so they are kept on the solver in
        ``_merge_weight_cache``, keyed by the quantized ratio. The cache is
        emptied when the method is rebuilt (``_set_solver_dtype``) or used on
        another device, and holds at most ``_MERGE_CACHE_SIZE`` ratios; a call
        with more distinct ratios builds its vectors without caching.

        Parameters
        ----------
        keys : torch.Tensor
            Distinct quantized ratios, ``round(log2(rho) * _RATIO_QUANTUM)``.
            Shape: [K], int64.

        Returns
        -------
        torch.Tensor
            The vectors of ``_merged_pair_weights`` for each ratio, in the
            order of ``keys``. Shape: [K, 2, 2, C].
        """
        rho = torch.exp2(keys.to(torch.float64) / self._RATIO_QUANTUM)
        if len(keys) > self._MERGE_CACHE_SIZE:
            return self._merged_pair_weights(rho)
        cache = self._merge_weight_cache
        if cache and next(iter(cache.values())).device != keys.device:
            cache.clear()
        key_list = keys.tolist()
        # Positions in keys of the ratios not cached yet
        missing = [i for i, k in enumerate(key_list) if k not in cache]
        if missing:
            if len(cache) + len(missing) > self._MERGE_CACHE_SIZE:
                cache.clear()
                missing = list(range(len(key_list)))
            weights = self._merged_pair_weights(rho[missing])
            cache.update(zip([key_list[i] for i in missing], weights, strict=True))
        return torch.stack([cache[k] for k in key_list])

    def _prune_excess_mesh(
        self,
        mesh: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        mesh_quadrature_errors: torch.Tensor,
        error_ratios_2steps: torch.Tensor,
        merge_cut: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge pairs of neighboring subintervals whose union is predicted to pass.

        Pair i (subintervals i and i + 1) is merged when its pair ratio
        ``error_ratios_2steps[i]`` is below ``merge_cut``; both come from the
        configured ``merge_rule`` (``_merge_inputs``). Of several adjacent
        qualifying pairs, ``_rec_remove`` keeps every other one, so no
        subinterval is merged twice; one call merges at most one level. A
        merge removes the pair's shared mesh point; the merged subinterval is
        not evaluated, and its contribution and error are the pair's sums.

        Parameters
        ----------
        mesh : torch.Tensor
            The mesh of the N subintervals. Shape: [N + 1, T].
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh_quadrature_errors : torch.Tensor
            Error estimate of each subinterval. Shape: [N, D].
        error_ratios_2steps : torch.Tensor
            Pair ratio of each pair of neighboring subintervals, indexed by the
            pair's first subinterval. Shape: [N-1].
        merge_cut : float
            A pair is merged when its pair ratio is below this value.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple ``(mesh, mesh_quadratures, mesh_quadrature_errors)`` with
            each merged pair's shared mesh point removed and the pair replaced
            by one subinterval whose contribution and error are the pair's
            sums. Shapes: [N + 1 - R, T], [N - R, D], [N - R, D] for R merged
            pairs. The inputs are not modified.
        """
        if len(error_ratios_2steps) == 0:
            return mesh, mesh_quadratures, mesh_quadrature_errors
        # Neighboring pairs share a subinterval, which cannot be merged twice;
        # each index is the first subinterval of a merged pair.
        merged = torch.where(self._rec_remove(error_ratios_2steps < merge_cut))[0]
        if self.runtime_checks:
            assert not torch.any(merged[:-1] + 1 == merged[1:])

        if len(merged) == 0:
            return mesh, mesh_quadratures, mesh_quadrature_errors

        # Pair i keeps subinterval i, extended to the end of i + 1: mesh point
        # i + 1 and row i + 1 go, and row i gets the pair's sums.
        keep = torch.ones(len(mesh), dtype=torch.bool, device=mesh.device)
        keep[merged + 1] = False
        rows = keep[:-1].to(mesh_quadratures.device)
        merged = merged.to(mesh_quadratures.device)
        return (
            mesh[keep],
            mesh_quadratures.index_put(
                (merged,), mesh_quadratures[merged] + mesh_quadratures[merged + 1]
            )[rows],
            mesh_quadrature_errors.index_put(
                (merged,),
                mesh_quadrature_errors[merged] + mesh_quadrature_errors[merged + 1],
            )[rows],
        )

    @contextlib.contextmanager
    def _integration_on_device(self, device: torch.device) -> Iterator[None]:
        """
        Temporarily run integration-device-bound operations on ``device``.

        ``_get_optimal_mesh`` (and the helpers it calls) allocate
        on ``self.device`` and read the method's tensors, which live on
        ``self.device``. To run the optimal-mesh computation on
        ``result_device``, so the large per-subinterval record tensors never
        have to be copied back to the integration device, this context manager
        sets ``self.device`` to ``device`` and ``self.method`` to the method
        converted to ``device`` (``MethodClass.to``, built once per process)
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
        original_device, original_method = self.device, self.method
        # Converted before self.device changes: if the conversion raises, the
        # solver is left as it was.
        self.method = original_method.to(device, self.dtype)
        self.device = device
        try:
            yield
        finally:
            self.device, self.method = original_device, original_method

    def _get_optimal_mesh(
        self,
        mesh_quadratures: torch.Tensor,
        mesh_quadrature_errors: torch.Tensor,
        integral: torch.Tensor,
        mesh: torch.Tensor,
        y: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Build the optimal mesh from the results of the integration loop.

        The optimal mesh is the starting mesh of a later call that passes it as
        ``mesh``. It is built in two stages:

        1. Merge: pairs of neighboring subintervals whose merged subinterval is
           predicted to have an error ratio below ``remove_cut`` are merged;
           ``merge_rule`` sets how the prediction is made (``_merge_inputs``,
           ``_prune_excess_mesh``).
        2. Split: subintervals whose error ratio against the final integral is
           still 1 or above get a midpoint. A merged subinterval carries the
           pair's summed error, which is at most the pair error the merge was
           decided on, so its ratio is below the merge cutoff, and so below 1,
           unless the rounding floor sets the tolerance (see
           ``_error_ratios_norm``): this stage does not split a merged
           subinterval, and it does not check the merge prediction.
           Subintervals fail here only when no re-check ran against the final
           integral (``enforce_convergence=False``, the default with
           ``take_gradient=True``) or when ``max_adaptive_splits`` or a
           non-finite value stopped their refinement.

        All inputs must share a device, and the returned mesh is on that
        device. (The caller runs this on ``result_device`` through
        ``_integration_on_device`` and moves the result back to the
        integration device.) The subintervals are those of ``mesh``, in
        order: the record of a finished integration covers its final mesh
        once.

        Parameters
        ----------
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh_quadrature_errors : torch.Tensor
            Absolute value of the error estimate of each subinterval (a merge
            sums two of them, which signed values could cancel). Shape: [N, D].
        integral : torch.Tensor
            The final integral. Shape: [D].
        mesh : torch.Tensor
            The mesh at the end of the loop, whose N subintervals these are.
            Shape: [N + 1, T].
        y : torch.Tensor or None
            Integrand values at the nodes of each subinterval, used by
            ``merge_rule="predicted"``, shape [N, C, D]; None (not kept) uses
            ``"power_law"``.

        Returns
        -------
        torch.Tensor
            The optimal mesh. Shape: [M_opt, T].
        """
        # Merge pairs of over-resolved neighbors: the one place where the pair
        # ratio (error_ratios_2steps) is computed and decides anything. Its
        # numerator is each pair's summed errors, plus the merge prediction
        # (adding a Python 0 is exact, so both cases sum in the same order).
        merge_prediction, merge_cut = self._merge_inputs(y, mesh_quadratures, mesh)
        _, error_ratios_2steps, _ = self._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors,
            mesh_quadratures=mesh_quadratures,
            integral=integral.detach(),
            pair_errors=(0 if merge_prediction is None else merge_prediction)
            + mesh_quadrature_errors[:-1]
            + mesh_quadrature_errors[1:],
        )
        assert error_ratios_2steps is not None  # computed with pair_errors
        mesh_pruned, mesh_quadratures_pruned, mesh_quadrature_errors_pruned = (
            self._prune_excess_mesh(
                mesh,
                mesh_quadratures,
                mesh_quadrature_errors,
                error_ratios_2steps,
                merge_cut,
            )
        )

        # Split the subintervals that fail against the final integral; only
        # the per-subinterval ratios are needed here.
        error_ratios, _, error_ratios_per_dim = self._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors_pruned,
            mesh_quadratures=mesh_quadratures_pruned,
            integral=integral.detach(),
        )
        keep_mask, _, _ = self._accept_reject_masks(error_ratios, error_ratios_per_dim)
        adaptive_step = self._adaptively_increase_mesh(
            method_output=None,
            keep_mask=keep_mask,
            y_step_eval=None,
            nodes=None,
            mesh=mesh_pruned,
            mesh_idxs=torch.arange(len(mesh_pruned) - 1, device=mesh_pruned.device),
            mesh_trackers=torch.zeros(
                len(mesh_pruned), dtype=torch.bool, device=mesh_pruned.device
            ),
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

        In one pass: an entry is kept when it is flagged and its distance from
        the first entry of its run of flags is even. The earlier loop un-flagged
        about two entries per pass over the whole mask, so a mask whose entries
        were mostly True took about N/2 passes (9 s of a 9.5 s call at
        N = 10**5); the output is the same (PR #69).

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask, True for a pair flagged for merging. Shape: [N].

        Returns
        -------
        torch.Tensor
            New mask with no adjacent True values; the input is not modified.
            Shape: [N].
        """
        index = torch.arange(len(mask), device=mask.device)
        previous = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=mask.device), mask[:-1]]
        )
        # Index of the first entry of the run each entry belongs to (0 for an
        # unflagged entry before any run; such entries are False anyway).
        run_start = torch.cummax(
            torch.where(mask & ~previous, index, torch.zeros_like(index)), dim=0
        ).values
        return mask & ((index - run_start) % 2 == 0)

    def _setup_integral_bounds(
        self,
        mesh: torch.Tensor | None,
        mesh_init: torch.Tensor | None,
        mesh_final: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Resolve the interval and the given mesh of a call, in the solver's dtype and device.

        Without a mesh, both bounds must be given. With a mesh, which must be
        [M, T] with M >= 2 increasing points, a missing bound is the mesh's
        end. Every given bound is a tensor of shape [T], and becomes an end
        point of the mesh, so it gets its gradient as any mesh end does: the
        mesh's points between the bounds are kept and the bounds added, which
        cuts the mesh at a bound inside it and extends it to a bound outside it
        (the integration refines the new subinterval). A bound that differs
        from its end also drops the points within rounding of it, a tolerance
        relative to the value compared, ``4 eps max(|p|, |bound|)`` (eps of the
        coarser of the mesh's and that bound's dtypes), including that end when
        the bound is within rounding of it, which avoids subintervals a few
        rounding steps wide, on which variable
        sampling returns NaN. A mesh alone, or bounds equal to its ends, leave
        it unchanged.

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
        tuple[torch.Tensor or None, torch.Tensor, torch.Tensor]
            ``(mesh, mesh_init, mesh_final)``: the mesh cut to the bounds
            ([M', T], or None without a mesh) and the bounds, each of shape [T].

        Raises
        ------
        ValueError
            If the mesh is not [M, T] with M >= 2 increasing points, if there
            is no mesh and a bound is missing, if a bound is not a tensor of
            shape [T] (the mesh's T, or the other bound's shape without a
            mesh), or if the bounds do not satisfy ``mesh_init < mesh_final``.
        """
        # A mesh given without bounds is returned as it is: there is nothing to
        # cut (the loop never writes into it).
        bounds_given = mesh_init is not None or mesh_final is not None
        if mesh is not None:
            # Increasing for T = 1; distinct consecutive points for any T (the
            # order of a path in T > 1 dimensions is its own). T > 1 is not
            # supported end to end yet: the order check below fails for it.
            if mesh.dim() != 2 or len(mesh) < 2:
                valid = False
            elif mesh.shape[-1] == 1:
                valid = bool(torch.all(mesh[1:] > mesh[:-1]))
            else:
                valid = bool(torch.all(torch.any(mesh[1:] != mesh[:-1], dim=-1)))
            if not valid:
                raise ValueError(
                    "mesh must be a tensor of shape [M, T] with M >= 2 increasing "
                    f"points; got shape {list(mesh.shape)}."
                )
        elif mesh_init is None or mesh_final is None:
            raise ValueError(
                "Pass mesh=, or both mesh_init and mesh_final: a call needs "
                "an interval to integrate (a state returned by a call with "
                "build_state_mesh=False has no mesh)."
            )
        # Bounds have shape [T] on every call, like a point of the mesh: the
        # mesh's T, or without a mesh the same shape for both bounds.
        shape = (
            mesh.shape[1:] if mesh is not None else getattr(mesh_init, "shape", None)
        )
        for name, given in (("mesh_init", mesh_init), ("mesh_final", mesh_final)):
            if given is not None and (
                not isinstance(given, torch.Tensor)
                or given.dim() != 1
                or given.shape != shape
            ):
                # The expected shape is known when the mesh or a 1-D mesh_init
                # gives T; else it is [T].
                raise ValueError(
                    f"{name} must be a tensor of shape "
                    f"{list(shape) if shape is not None and len(shape) == 1 else '[T]'}"
                    ", like a point of the mesh; got "
                    f"{type(given).__name__} {list(getattr(given, 'shape', []))}."
                )
        if mesh is None:
            # Narrows the types for mypy: both were checked above.
            assert mesh_init is not None
            assert mesh_final is not None
            mesh_init = mesh_init.to(self.dtype).to(self.device)
            mesh_final = mesh_final.to(self.dtype).to(self.device)
        else:
            # Rounding tolerance relative to the value compared, 4 eps |x|, with
            # eps of the coarser of the mesh's and that bound's dtypes (a float32
            # mesh_init must not widen the tolerance of a float64 mesh_final).
            eps = [
                max(
                    torch.finfo(x.dtype).eps
                    for x in (mesh, given)
                    if x is not None and x.is_floating_point()
                )
                for given in (mesh_init, mesh_final)
            ]
            mesh = mesh.to(self.dtype).to(self.device)
            # A missing bound is the mesh's end; a given one, inside or outside
            # the mesh, becomes the mesh's new end point (the cut below).
            mesh_init, mesh_final = (
                end if given is None else given.to(self.dtype).to(self.device)
                for given, end in ((mesh_init, mesh[0]), (mesh_final, mesh[-1]))
            )

        # Bounds are user input: raise rather than assert, which python -O
        # removes (a reversed call then ran on and stored an out-of-order
        # mesh_optimal with points outside the interval).
        if not mesh_init < mesh_final:
            raise ValueError(
                "The integration bounds must satisfy mesh_init < mesh_final; got "
                f"{mesh_init.tolist()} and {mesh_final.tolist()}. To integrate "
                "from b to a, integrate from a to b and negate the result."
            )

        if mesh is not None and bounds_given:
            # Cut to the bounds: keep the points in [mesh_init, mesh_final] other
            # than the bounds, then add the bounds, so every given bound is a
            # mesh end and gets its gradient. A bound that is not its end also
            # drops the points within rounding of it (the end itself when the
            # bound is within rounding of it): they would leave a subinterval a
            # few rounding steps wide, on which variable sampling returns NaN.
            # Bounds equal to the mesh's ends leave it unchanged.
            keep = torch.all((mesh >= mesh_init) & (mesh <= mesh_final), dim=-1)
            for bound, end, e in (
                (mesh_init, mesh[0], eps[0]),
                (mesh_final, mesh[-1], eps[1]),
            ):
                keep &= ~torch.all(mesh == bound, dim=-1)
                if not torch.equal(bound, end):
                    tol = 4 * e * torch.maximum(mesh.abs(), bound.abs())
                    keep &= ~torch.all(torch.abs(mesh - bound) <= tol, dim=-1)
            mesh = torch.concatenate(
                [mesh_init.unsqueeze(0), mesh[keep], mesh_final.unsqueeze(0)], dim=0
            )
        return mesh, mesh_init, mesh_final

    def _setup_initial_mesh(
        self,
        mesh: torch.Tensor | None,
        mesh_init: torch.Tensor,
        mesh_final: torch.Tensor,
        random_initial_mesh: bool,
        N_init_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the starting mesh of an integration.

        The caller's mesh (already validated and cut to the bounds by
        ``_setup_integral_bounds``, on the solver's dtype and device), or a
        fresh mesh of about N_init_steps points: sqrt(N_init_steps) equal segments, each holding
        sqrt(N_init_steps) + 1 mesh points, placed at random or evenly. A
        point that rounding puts on or below an earlier point, or on or above
        ``mesh_final``, is dropped, so a fresh mesh is strictly increasing
        and may have fewer points.

        Parameters
        ----------
        mesh : torch.Tensor or None
            The caller's mesh. Shape: [M, T].
        mesh_init : torch.Tensor
            Lower bound. Shape: [T].
        mesh_final : torch.Tensor
            Upper bound. Shape: [T].
        random_initial_mesh : bool
            If True, place the mesh points inside each segment at random;
            otherwise evenly.
        N_init_steps : int
            Approximate number of mesh points of a fresh mesh.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The mesh [M, T], and ``mesh_trackers`` [M], True at every mesh
            point but the last (every subinterval needs evaluation).
        """
        if mesh is None:
            # Generate a fresh initial mesh of barriers across [mesh_init, mesh_final].
            # Layout: sqrt(N_init_steps) evenly-spaced top-level segments, each
            # subdivided into sqrt(N_init_steps)+1 sub-barriers. The total
            # barrier count is ~N_init_steps. Sub-barriers are placed
            # randomly (default) or uniformly within each segment.
            # The exact integer square root; int() because math.isqrt refuses
            # a float such as N_init_steps=1e4.
            N_even_t = math.isqrt(int(N_init_steps))
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
            # Rounding can put a point on the point before it, or one unit in
            # the last place below it: offsets closer than the spacing of
            # floating-point numbers at the segment's start round to the same
            # point (float32; float64 far from zero, or with more points than
            # the 24 random bits of torch.rand's float32 draws tell apart),
            # and the last point of a segment can round above the first of
            # the next (float32, N_init_steps=10**6 on [1, 3]). Keep the points
            # above every point before them and below mesh_final, so the mesh
            # is strictly increasing (for T > 1, which integrate does not
            # reach, consecutive points differ) and keeps its exact ends. One
            # read of the device, whether the mesh is strictly increasing (then
            # no point is dropped and it is kept as it is): the rest runs only
            # when a point must be dropped.
            if not bool(torch.all(mesh[1:] > mesh[:-1])):
                inner = mesh[1:-1]
                keep = (
                    (inner > torch.cummax(mesh[:-2], dim=0).values)
                    & (inner < mesh_final)
                ).any(-1)
                mesh = torch.cat([mesh[:1], inner[keep], mesh[-1:]])
        # True at every mesh point but the last, which starts no subinterval;
        # compared on the device, with no copy from the host.
        mesh_trackers = torch.arange(len(mesh), device=self.device) < len(mesh) - 1
        return mesh, mesh_trackers

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
        - a callable: ``error_norm(x)``, which must reduce the last axis,
          each row on its own: the record keeps a subinterval's reduced error
          from its batch (``_reduced_errors``), so a row's value must not
          depend on the other rows. Its output must be a tensor of shape
          [...] on ``x``'s device; that is checked here, the independence of
          the rows is not.

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

        Raises
        ------
        ValueError
            If a callable ``error_norm`` returns anything but a tensor of
            shape [...] on ``x``'s device.
        """
        error_norm = self.error_norm if error_norm is None else error_norm
        if callable(error_norm):
            # Checked from the output's type, shape and device, all known on
            # the host: no read of the device. A shape such as [N, 1]
            # failed later with an indexing error that named nothing of this.
            out = error_norm(x)
            if not (
                isinstance(out, torch.Tensor)
                and out.shape == x.shape[:-1]
                and out.device == x.device
            ):
                got = (
                    f"{out.dtype} tensor of shape {list(out.shape)} on {out.device}"
                    if isinstance(out, torch.Tensor)
                    else type(out).__name__
                )
                raise ValueError(
                    "A callable error_norm must reduce the last axis of its input "
                    "and return a tensor on the input's device: for "
                    f"an input of shape {list(x.shape)} on {x.device}, one of "
                    f"shape {list(x.shape[:-1])}; it returned a {got}."
                )
            return out
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
          ``h * s_k_abs`` the subinterval's quadrature of ``abs(f)``; PadaQuad
          uses the absolute value of the signed contribution, which is smaller
          where f changes sign.
        - scipy raises the error estimate to the floor,
          ``err = max(err, round_err)``, and only when ``round_err`` exceeds the
          smallest positive double; PadaQuad raises the tolerance instead and
          has no such underflow guard.
        - scipy first rescales the error estimate as QUADPACK does,
          ``resasc * min(1, (200 * err / resasc)**1.5)`` (:cite:p:`quadpack_netlib`,
          dqk21.f lines 177-178); PadaQuad does not.

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
        pair_errors: torch.Tensor | None = None,
        reduced_errors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
            floor and, in cumulative mode without ``cum_mesh_quadratures``,
            the running integral (then in mesh order). [N, D]. None gives no
            rounding floor.
        cum_mesh_quadratures : torch.Tensor or None
            Precomputed running integral through each subinterval [N, D]
            (cumulative mode only).
        integral : torch.Tensor or None
            Current total integral [D] (absolute mode only; required there).
        pair_errors : torch.Tensor or None
            Error of each pair of neighbors merged into one [N-1, D], the
            numerator of ``error_ratios_2steps``, built by its one reader,
            ``_get_optimal_mesh``. None (default) skips the pair ratio.
        reduced_errors : torch.Tensor or None
            The norm family's reduced error and rounding floor of each
            subinterval [N, 2] (``_reduced_errors``), as the record keeps them;
            None (default) computes them from the arrays. Ignored by
            ``"failure_fraction"``, whose tolerance is per component.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
            Tuple ``(error_ratios, error_ratios_2steps, error_ratios_per_dim)``:

            - error_ratios: accept-or-split quantity of each subinterval [N]
              (the reduced ratio for the norm family; the failure fraction for
              ``"failure_fraction"``).
            - error_ratios_2steps: merge indicator of each pair of neighbors
              [N-1] when ``pair_errors`` is given, else None.
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
        if not self.use_absolute_error_ratio and cum_mesh_quadratures is None:
            if mesh_quadratures is None:
                raise ValueError("Must give mesh_quadratures or cum_mesh_quadratures")
            cum_mesh_quadratures = torch.cumsum(mesh_quadratures, dim=0)
        if mesh_quadratures is None:
            # No rounding floor: a zero floor leaves every tolerance as it is.
            mesh_quadratures = torch.zeros_like(mesh_quadrature_errors)
        if self._is_failure_fraction:
            return self._error_ratios_failure_fraction(
                torch.abs(mesh_quadrature_errors),
                mesh_quadratures,
                cum_mesh_quadratures,
                integral,
                pair_errors,
            )
        if reduced_errors is None:
            reduced_errors = self._reduced_errors(
                mesh_quadrature_errors, mesh_quadratures
            )
        return self._error_ratios_norm(
            reduced_errors, cum_mesh_quadratures, integral, pair_errors
        )

    def _reduced_errors(
        self, mesh_quadrature_errors: torch.Tensor, mesh_quadratures: torch.Tensor
    ) -> torch.Tensor:
        """
        Reduce each subinterval's error estimate and rounding floor with the norm.

        These are the two numbers of the norm family's error ratio that never
        change once a subinterval is evaluated, ``R(abs(e[k]))`` and
        ``R(floor[k])`` (``_error_ratios_norm``), with R the configured norm
        (``_reduce_norm``) and the rounding floor of ``_round_floor``; only the
        tolerance changes. The integration loop computes them for each batch
        and the record keeps them, so every later judgement of a recorded
        subinterval (the re-check and the final check, ``_record_error_check``)
        uses the very numbers it was accepted with and reduces no [N, D] array
        again.

        Parameters
        ----------
        mesh_quadrature_errors : torch.Tensor
            Error estimate of each subinterval. Shape: [N, D].
        mesh_quadratures : torch.Tensor
            Contribution of each subinterval. Shape: [N, D].

        Returns
        -------
        torch.Tensor
            ``R(abs(e[k]))`` and ``R(floor[k])`` of each subinterval. Shape:
            [N, 2].
        """
        return torch.stack(
            [
                self._reduce_norm(torch.abs(mesh_quadrature_errors)),
                self._reduce_norm(self._round_floor(mesh_quadratures)),
            ],
            dim=-1,
        )

    def _error_ratios_norm(
        self,
        reduced_errors: torch.Tensor,
        cum_mesh_quadratures: torch.Tensor | None,
        integral: torch.Tensor | None,
        pair_errors: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, None]:
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
        (line 410). PadaQuad accepts or splits each subinterval by its own
        error ratio.

        The pair ratio ``error_ratio_2steps[k] = R(p[k]) / tol_2steps[k]``,
        with ``p`` the pair errors (used only to merge subintervals, in
        ``_prune_excess_mesh``), uses the tolerance the merged subinterval
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
        reduced_errors : torch.Tensor
            ``R(abs(e[k]))`` and ``R(floor[k])`` of each subinterval
            (``_reduced_errors``). Shape: [N, 2].
        cum_mesh_quadratures : torch.Tensor | None
            Running integral through each subinterval (cumulative mode;
            required there). Shape: [N, D].
        integral : torch.Tensor | None
            Total integral (absolute mode). Shape: [D].
        pair_errors : torch.Tensor | None
            Error of each pair of neighbors merged into one, the numerator of
            the pair ratio (see ``_compute_error_ratios``), or None to skip the
            pair ratio. Shape: [N-1, D].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, None]
            ``(error_ratio, error_ratio_2steps, None)``, shapes [N] and [N-1]
            (None without ``pair_errors``).
        """
        err, floor = reduced_errors.unbind(-1)
        if self.use_absolute_error_ratio:
            assert integral is not None  # checked in _compute_error_ratios
            error_tol = torch.clamp(
                self.rtol * self._reduce_norm(integral), min=self.atol
            )
            error_tol_2steps = error_tol
        else:
            assert cum_mesh_quadratures is not None  # set in _compute_error_ratios
            error_tol = torch.clamp(
                self.rtol * self._reduce_norm(torch.abs(cum_mesh_quadratures)),
                min=self.atol,
            )
            # A merged pair (i, i+1) ends where step i+1 ends, so it is judged
            # against step i+1's cumulative tolerance (not the larger of the
            # two, which under-predicts the merged ratio when |cum| falls).
            error_tol_2steps = error_tol[1:]

        # Broadcasts a scalar (absolute mode) or per-step (cumulative) tol.
        error_ratio = err / torch.maximum(error_tol, floor)
        if pair_errors is None:
            return error_ratio, None, None
        error_tol_2steps = torch.maximum(error_tol_2steps, floor[:-1] + floor[1:])
        return error_ratio, self._reduce_norm(pair_errors) / error_tol_2steps, None

    def _error_ratios_failure_fraction(
        self,
        abs_err: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        cum_mesh_quadratures: torch.Tensor | None,
        integral: torch.Tensor | None,
        pair_errors: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
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
        of ``p[k, d] / tol_2steps[k, d]``, with the pair error ``p`` and the
        merged subinterval's tolerance as in ``_error_ratios_norm``.

        The rounding floor is a lower bound on each component's tolerance,
        ``effective_tol = max(tol, floor)``, so a component whose error is at
        or below rounding level does not count as failing.

        Parameters
        ----------
        abs_err : torch.Tensor
            Absolute error estimate of each subinterval. Shape: [N, D].
        mesh_quadratures : torch.Tensor
            Contribution of each subinterval (for the rounding floor). Shape:
            [N, D].
        cum_mesh_quadratures : torch.Tensor | None
            Running integral through each subinterval (cumulative mode;
            required there). Shape: [N, D].
        integral : torch.Tensor | None
            Total integral (absolute mode). Shape: [D].
        pair_errors : torch.Tensor | None
            Error of each pair of neighbors merged into one, the numerator of
            the pair ratio (see ``_compute_error_ratios``), or None to skip the
            pair ratio. Shape: [N-1, D].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
            ``(failure_fraction, error_ratio_2steps, ratio)``, shapes [N],
            [N-1] (None without ``pair_errors``) and [N, D].
        """
        if self.use_absolute_error_ratio:
            assert integral is not None  # checked in _compute_error_ratios
            error_tol = self.atol + self.rtol * torch.abs(integral)  # [D]
            error_tol_2steps = error_tol
        else:
            assert cum_mesh_quadratures is not None  # set in _compute_error_ratios
            error_tol = self.atol + self.rtol * torch.abs(cum_mesh_quadratures)
            # Merged pair's tolerance: step i+1's (see _error_ratios_norm).
            error_tol_2steps = error_tol[1:]

        # torch.maximum broadcasts the [D] or [N, D] tolerance with the floor.
        floor = self._round_floor(mesh_quadratures)  # [N, D]
        ratio = abs_err / torch.maximum(error_tol, floor)  # [N, D]
        finite = torch.isfinite(ratio)
        failed = (ratio >= 1.0) & finite
        failure_fraction = failed.to(self.dtype).mean(dim=-1)  # [N]
        if pair_errors is None:
            return failure_fraction, None, ratio

        # Merge/prune indicator: conservative max-norm of the combined 2-step
        # ratio (compared against remove_cut < 1, so a magnitude is needed here
        # rather than a fraction).
        error_tol_2steps = torch.maximum(error_tol_2steps, floor[:-1] + floor[1:])
        error_ratio_2steps = torch.amax(pair_errors / error_tol_2steps, dim=-1)
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

    # Per-subinterval record keys the loop reads, each kept as one tensor on
    # the integration device and appended to every turn: the left mesh index
    # of each recorded subinterval in the current mesh (the order of the
    # cumulative sums, the re-check's mesh positions), the contributions and
    # error estimates (cumulative sums, removals and the re-check of
    # "failure_fraction"), and for the norm family the reduced error and
    # rounding floor (``_reduced_errors``; the re-check and the final check),
    # which ``integrate`` drops after the final check. Every other
    # per-subinterval field is kept as one chunk per batch
    # (``_record_results``).
    _RECORD_LOOP_KEYS = (
        "mesh_index",
        "mesh_quadratures",
        "mesh_quadrature_errors",
        "reduced_errors",
    )

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
        # y is None without record_evaluations; it is then not recorded.
        fields: dict[str, torch.Tensor] = {
            key: getattr(results, key)
            for key in self._RECORD_SCALAR_KEYS + self._RECORD_STEP_KEYS
            if getattr(results, key) is not None
        }
        for i, tracked in enumerate(results.tracked_variables or ()):
            fields[f"{self._TRACKED_KEY_PREFIX}{i}"] = tracked
        return fields

    def _merge_positions(
        self, rec_order: torch.Tensor, new_order: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute merged-array positions from the mesh indices of two row sets.

        Given the mesh index of each existing record row (in any order) and of
        each new row, return where each goes in the combined array in mesh
        order. The mesh indices are distinct integers, so the merge is a rank
        computation, with no comparison of coordinates.

        Parameters
        ----------
        rec_order : torch.Tensor
            Mesh indices of the existing record rows, in the record's row
            order. Shape: [N_record].
        new_order : torch.Tensor
            Mesh indices of the new rows. Shape: [N_result].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (idxs_keep, idxs_input): positions in the merged array for
            the existing and the new rows respectively (consumed by
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
        # Index tensors come from _merge_positions on the integration device;
        # move them to the record's device so a record living on result_device
        # (CPU) merges with CPU indices while a GPU-resident record uses GPU
        # indices. ``result`` is already placed on record.device by the caller.
        # The positions of _merge_positions are a permutation of the merged
        # rows, so every row is written below and need not be zeroed first.
        merged = record.new_empty((len(record) + len(result), *record.shape[1:]))
        merged[record_idxs.to(record.device)] = record
        merged[result_idxs.to(record.device)] = result
        return merged

    def _record_results(
        self,
        record: dict[str, torch.Tensor],
        chunks: dict[str, list[torch.Tensor]],
        take_gradient: bool,
        results: IntegrationResult,
        mesh_index: torch.Tensor,
        result_device: str | torch.device | None = None,
        reduced_errors: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Append the accepted subintervals of a batch to the record.

        The record keeps the subintervals in the order they were accepted;
        ``_sort_record`` puts them in mesh order after the loop. The sums
        (``_RECORD_SCALAR_KEYS``) are added to; the fields the loop reads
        (``_RECORD_LOOP_KEYS``) are appended to one tensor each, on the
        integration device; every other field (``nodes``, ``h``, ``y``,
        ``error_ratios`` and the tracked variables) is appended to its list in
        ``chunks``, one tensor per batch, so the record's values are never
        copied during the loop. The values (``y`` and the tracked variables)
        are moved to ``result_device``, so they do not accumulate on the GPU;
        the other fields, which are small, wait on the integration device and
        ``_sort_record`` moves them once, after the loop. A chunk that is
        a view of a larger array, such as the accepted rows moved to the front
        of a pass's values (``_keep_rows``), is copied, so the rest of that
        array is freed. When ``take_gradient`` is True the per-batch
        ``loss.backward()`` has already run, so tensors are detached first;
        when False the autograd graph is kept so the returned integral stays
        differentiable (the device move is itself differentiable).

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            The sums and the loop's fields. An empty dict on the first call.
        chunks : dict[str, list[torch.Tensor]]
            The other per-subinterval fields, one tensor per batch, in the
            record's row order. An empty dict on the first call.
        take_gradient : bool
            If True, the per-batch backward already ran and tensors are
            detached before recording; if False the graph is kept.
        results : IntegrationResult
            Results of the accepted subintervals of the current batch.
        mesh_index : torch.Tensor
            Left mesh index of each of them in the current mesh [N], on the
            integration device; recorded under the key ``mesh_index``.
        result_device : str | torch.device | None
            Device for the values (``y`` and the tracked variables). None keeps
            them on the integration device.
        reduced_errors : torch.Tensor or None
            The norm family's reduced error and rounding floor of each of them
            [N, 2] (``_reduced_errors``), recorded under the key
            ``reduced_errors``; None (``"failure_fraction"``) records none.

        Returns
        -------
        dict[str, torch.Tensor]
            The record with the batch added. Its keys are the sums, the
            loop's fields and ``mesh_index``; the keys of ``chunks`` are the
            other IntegrationResult field names and one key per tracked
            variable (see ``_record_fields``).

        Raises
        ------
        ValueError
            If the batch's tracked variables are not those of earlier batches.
        """
        if result_device is None:
            result_device = self.device
        fields = self._record_fields(results)
        fields["mesh_index"] = mesh_index
        if reduced_errors is not None:
            fields["reduced_errors"] = reduced_errors
        if record and fields.keys() != record.keys() | chunks.keys():
            raise ValueError(
                "The integrand returned a different set of tracked variables "
                f"than in earlier batches: {sorted(fields)} vs "
                f"{sorted(record.keys() | chunks.keys())}."
            )
        for key, field in fields.items():
            # take_gradient=True does per-batch backward *before* recording,
            # so the graph is no longer needed and is detached.
            # take_gradient=False retains the full graph for a single backward
            # through the returned integral.
            value = field.detach() if take_gradient else field
            if key in self._RECORD_SCALAR_KEYS:
                record[key] = record[key] + value if key in record else value
            elif key in self._RECORD_LOOP_KEYS:
                record[key] = (
                    torch.cat([record[key], value]) if key in record else value
                )
            else:
                if key == "y" or key.startswith(self._TRACKED_KEY_PREFIX):
                    value = value.to(result_device)
                if value.untyped_storage().nbytes() > value.nbytes:
                    value = value.clone()
                chunks.setdefault(key, []).append(value)
        return record

    def _sort_record(
        self,
        record: dict[str, torch.Tensor],
        chunks: dict[str, list[torch.Tensor]],
        mesh: torch.Tensor,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Put the record of a finished integration in mesh order, in one dict.

        The record covers its final mesh once, so the left mesh indices of
        its N rows (``mesh_index``, removed here) are 0 to N - 1 in some order,
        and row i belongs at row ``mesh_index[i]``: each field is written into
        one array at those rows, chunk by chunk, each chunk dropped once
        written. A field held in one tensor whose rows are already in mesh
        order (always, when every subinterval was recorded in one batch) is
        kept as it is, since writing it would copy it, the values included.
        The sums are not reordered. The loop's fields stay on the integration
        device; the fields of ``chunks`` end on ``device``.

        With ``runtime_checks``, it checks in one read of the device that the
        mesh indices are 0 to N - 1 (N the number of subintervals of
        ``mesh``), each once, and that each recorded width ``h`` equals the
        spacing of ``mesh`` at its index, bit for bit: both are the
        difference of the same two mesh points (a batch evaluated on a
        detached copy of the mesh, with ``take_gradient=True`` and a bound
        that requires grad, has the same values), and the mesh points of an
        accepted subinterval never move. The widths catch a wrong but valid
        permutation, such as two rows swapped, which would otherwise put
        every field and the state's mesh out of place silently; they cannot
        catch two rows of exactly equal width swapped.

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            The sums and the loop's fields, in the order the subintervals were
            accepted (``_record_results``).
        chunks : dict[str, list[torch.Tensor]]
            The other per-subinterval fields, in the same row order; emptied.
        mesh : torch.Tensor
            The final mesh [N + 1, T], on the integration device.
        device : str | torch.device
            Device of the fields of ``chunks`` in the returned record.

        Returns
        -------
        dict[str, torch.Tensor]
            The record with every per-subinterval field in mesh order.

        Raises
        ------
        AssertionError
            With ``runtime_checks``, if the record does not cover ``mesh``
            once or a recorded width is not its subinterval's.
        """
        rows = record.pop("mesh_index")
        if self.runtime_checks:
            # Every subinterval of the final mesh is recorded exactly once, and
            # each recorded width is the mesh spacing at its index. One read.
            sorted_rows, order = torch.sort(rows)
            widths = mesh[1:] - mesh[:-1]
            assert len(rows) == len(widths), (
                f"The record has {len(rows)} subintervals, the final mesh "
                f"{len(widths)}."
            )
            assert bool(
                torch.all(sorted_rows == torch.arange(len(rows), device=rows.device))
                & torch.all(torch.cat(chunks["h"])[order] == widths)
            ), (
                "The record does not cover the final mesh once, or a recorded "
                "width is not the width of its subinterval in the mesh."
            )
        # One read of the device. Rows in increasing mesh order are 0 to N - 1.
        in_order = bool(torch.all(rows[1:] > rows[:-1]))
        loop_fields = {
            key: [value]
            for key, value in record.items()
            if key not in self._RECORD_SCALAR_KEYS
        }
        for key, parts in (loop_fields | chunks).items():
            if in_order and len(parts) == 1:
                out = parts.pop()
            else:
                out = parts[0].new_empty((len(rows), *parts[0].shape[1:]))
                at = rows.to(out.device)
                start = 0
                # Each chunk is freed once written (no name keeps the last
                # one while the next field is written).
                while parts:
                    n = len(parts[0])
                    out[at[start : start + n]] = parts.pop(0)
                    start += n
            record[key] = out.to(device) if key in chunks else out
        return record

    def _remove_failed_record_panels(
        self,
        record: dict[str, torch.Tensor],
        record_keep_mask: torch.Tensor,
        chunks: dict[str, list[torch.Tensor]],
        loss_fxn: Callable[[IntegrationResult], torch.Tensor],
        mesh_init: torch.Tensor,
        mesh_final: torch.Tensor,
        values_device: torch.device | None,
    ) -> dict[str, torch.Tensor]:
        """
        Remove from the record the subintervals that failed the re-check.

        They were just split in the mesh (their halves will be evaluated as
        new subintervals), so their entries must leave the record. The
        per-subinterval fields are filtered by ``record_keep_mask`` (each
        field of ``chunks`` becomes one chunk), and each accumulated scalar
        loses what the removed subintervals added to it:

        - ``integral`` and ``integral_error``: the sums of the removed
          ``mesh_quadratures`` and signed ``mesh_quadrature_errors``.
        - ``loss``: ``loss_fxn`` of a result made of the removed subintervals,
          laid out like a per-batch result (in mesh order). For a loss that is
          a sum over subintervals (any loss linear in the integral, including
          the default) the loss and its gradient are then those of the kept
          subintervals; for another loss the per-batch sum is an
          approximation either way. The re-check never runs with
          ``take_gradient=True``, so no backward pass has used the loss yet.

        Parameters
        ----------
        record : dict[str, torch.Tensor]
            The sums and the loop's fields (rows in the order the
            subintervals were accepted).
        record_keep_mask : torch.Tensor
            Boolean mask over the recorded subintervals; True keeps one.
        chunks : dict[str, list[torch.Tensor]]
            The other per-subinterval fields, in the record's row order.
        loss_fxn : Callable[[IntegrationResult], torch.Tensor]
            The active loss function.
        mesh_init : torch.Tensor
            Lower integration bound, for the removed subintervals' result.
        mesh_final : torch.Tensor
            Upper integration bound, for the removed subintervals' result.
        values_device : torch.device or None
            Device of the removed subintervals' integrand values and tracked
            variables in their result: the solver's device, or
            ``result_device`` with ``trade_speed_for_memory``, as in the
            per-batch result ``loss_fxn`` receives (None: the record holds no
            values).

        Returns
        -------
        dict[str, torch.Tensor]
            The record without the failed subintervals.
        """
        # The removed rows in mesh order, as a batch's rows are, so their sums
        # and loss_fxn's input are those the record held in mesh order.
        removed = torch.nonzero(~record_keep_mask).flatten()
        removed = removed[torch.argsort(record["mesh_index"][removed])]
        gone = {
            key: value[removed]
            for key, value in record.items()
            if key not in self._RECORD_SCALAR_KEYS
        }
        for key, parts in chunks.items():
            # Joined, so the removed and the kept rows are indexed once; the
            # kept rows become the field's one chunk. No name keeps the joined
            # rows, so they are freed before the next field is joined.
            parts[:] = [torch.cat(parts)]
            gone[key] = parts[0][removed.to(parts[0].device)]
            parts[0] = parts[0][record_keep_mask.to(parts[0].device)]
        removed_quad_sum = gone["mesh_quadratures"].sum(0)
        removed_err_sum = gone["mesh_quadrature_errors"].sum(0)
        # The result loss_fxn receives has the values (y, tracked variables)
        # on values_device and the rest on the integration device.
        removed_result = IntegrationResult(
            integral=removed_quad_sum,
            integral_error=removed_err_sum,
            nodes=gone["nodes"],
            h=gone["h"],
            y=gone["y"].to(values_device) if "y" in gone else None,
            tracked_variables=tuple(
                value.to(values_device)
                for key, value in gone.items()
                if key.startswith(self._TRACKED_KEY_PREFIX)
            )
            or None,
            mesh_quadratures=gone["mesh_quadratures"],
            mesh_quadrature_errors=torch.abs(gone["mesh_quadrature_errors"]),
            error_ratios=gone["error_ratios"],
            gradient_taken=False,
            mesh_init=mesh_init,
            mesh_final=mesh_final,
            y0=torch.zeros_like(removed_quad_sum),
        )
        record["integral"] = record["integral"] - removed_quad_sum.to(
            record["integral"].device
        )
        record["integral_error"] = record["integral_error"] - removed_err_sum.to(
            record["integral_error"].device
        )
        record["loss"] = record["loss"] - loss_fxn(removed_result).to(
            record["loss"].device
        )
        for key, value in record.items():
            if key not in self._RECORD_SCALAR_KEYS:
                record[key] = value[record_keep_mask]
        return record

    def _flatten_output(self, panel_array: torch.Tensor) -> torch.Tensor:
        """
        Join per-subinterval values into one sequence, each evaluated point once.

        When the method's node fractions start at 0 and end at 1, neighboring
        subintervals share a node, their mesh point: all C entries of the first
        subinterval are kept and the first entry of every later one, which
        repeats the previous subinterval's last entry, is dropped. Otherwise
        (for example Gauss-Kronrod, whose nodes are all interior) no node is
        shared, and every entry is kept.

        Parameters
        ----------
        panel_array : torch.Tensor
            Values per subinterval and node, in mesh order. Shape: [N, C, ...].

        Returns
        -------
        torch.Tensor
            The joined values. Shape: [N * (C - k) + k, ...], with k = 1 when
            the ends are nodes and 0 otherwise.
        """
        n, c, *rest = panel_array.shape
        fractions = self.method.node_fractions()
        # The number of entries a subinterval shares with its left neighbor.
        k = int(float(fractions[0]) == 0.0 and float(fractions[-1]) == 1.0)
        # Written straight into the output: flattening the non-contiguous
        # panel_array[1:, k:] and then concatenating made two full copies, 3
        # times the values on the host at the peak (measured 2026-11-26).
        out = panel_array.new_empty((n * (c - k) + k, *rest))
        out[:c] = panel_array[0]
        out[c:].view(n - 1, c - k, *rest).copy_(panel_array[1:, k:])
        return out
