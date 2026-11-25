"""
Memory management for the adaptive quadrature solvers.

The integrand f is evaluated in batches; how many evaluations fit in one batch
depends on the memory one evaluation uses. This module holds:

- ``_MemoryManagement``: a mixin inherited by ``AdaptiveQuadrature``
  (``base.py``). It measures how the memory of one call of f grows with the
  number of points N, as a fit a + b N + d N^2 (``_setup_memory_checks``),
  queries free memory on the solver's device, and turns both into the batch
  size (``_get_max_f_evals``: the largest N whose fitted
  memory fits in ``_MEMORY_BUDGET`` of the usable memory), recomputed at every
  evaluation from live memory. The benchmark is a heuristic for the common
  case and cannot be right for every f; its known limitations are listed in
  the user guide, ``docs/user_guide/devices_memory.md`` (section "Limitations
  of the memory benchmark").
- ``_fit_up_to_quadratic`` and ``_largest_n``: the fit, and its inverse.
- ``_is_out_of_memory``: recognizes the out-of-memory errors of the GPU, the
  PyTorch CPU allocator, the CUDA libraries and numpy, used by the batch loop
  in ``base.py`` and by the benchmark.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import inspect
import logging
import math
import os
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, ParamSpec, TypeVar

import numpy as np
import psutil
import torch

from padaquad.results import IntegrationResult, IntegrationState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_GB = 1024**3  # bytes per GB; every memory quantity below is in GB

# Multiplies the measured memory cost of one f evaluation when
# take_gradient=True: backpropagation uses roughly twice the forward memory.
# A heuristic, not a measurement.
_GRADIENT_MEMORY_SAFETY_FACTOR = 2.1

# Fraction of the usable memory that one call of f may use: both the largest
# test call of the memory benchmark and every batch of the integration. The
# rest is a margin for errors of the fit.
_MEMORY_BUDGET = 0.75

# After an out-of-memory error, the batch size is multiplied by this factor
# (in the benchmark below, in _shrink_batch and in the batch loop of base.py).
_OOM_BACKOFF = 0.75

# While the batch size of an integrate() call has not converged, a call of f
# that needs more points than it tries this many times as many.
_GROWTH = 10

# The largest test call of integrate() given no max_batch and no state (the
# test calls are 10, 100 and 1,000 points; larger sizes are reached by growth,
# whose calls of f do useful work).
_LARGEST_TEST_CALL = 1000


def _shrink_batch(n: int) -> int:
    """
    Return the batch size to retry with after a batch of n points ran out of memory.

    ``_OOM_BACKOFF * n``, rounded, but always below n (``round(0.75 * 2)`` is 2)
    and at least 1.

    Parameters
    ----------
    n : int
        The batch size that ran out of memory, at least 1.

    Returns
    -------
    int
        The smaller batch size.
    """
    return max(1, int(min(round(_OOM_BACKOFF * n), n - 1)))


# Saved gradients, by id of the tensor: a weak reference to it (so that a
# leaf made for one batch is not kept alive for the whole call) and a copy of
# its .grad before the call (None if it had none).
_SavedGrads = dict[int, tuple["weakref.ref[torch.Tensor]", "torch.Tensor | None"]]


def _save_grad(leaf: torch.Tensor, saved: _SavedGrads) -> None:
    """
    Save leaf's ``.grad`` in ``saved`` unless it is there already.

    A tensor freed during the call leaves its entry; a new tensor that gets
    its ``id`` is told apart by the weak reference and saved again. (A
    callback removing the entry would put ``saved`` in a reference cycle, so
    the copies would outlive the call until Python's cyclic collector runs.)

    Parameters
    ----------
    leaf : torch.Tensor
        A tensor a backward pass may write a gradient to.
    saved : _SavedGrads
        The saved gradients, updated in place.
    """
    entry = saved.get(id(leaf))
    if entry is None or entry[0]() is not leaf:
        grad = None if leaf.grad is None else leaf.grad.clone()
        saved[id(leaf)] = (weakref.ref(leaf), grad)


class _Restart(Exception):
    """Raised by ``integrate`` to be called again with ``args[0]`` as its state."""


# Solver attributes that hold the state of one integrate() call.
_PER_CALL = (
    "_batch",
    "_batch_converged",
    "_total_mem_usage",
    "_forward_bytes",
    "_backward_ratio",
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _restart_after_backward_oom(integrate: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Call ``integrate`` again, with a new state, when it raises ``_Restart``.

    ``integrate`` raises ``_Restart(state)`` after an out-of-memory error in a
    backward pass, once it has put back the gradients the failed attempt
    added. The new call gets the same arguments, except ``mesh`` and
    ``max_batch`` (None) and ``state``. It runs after the failed call has
    returned, so that call's tensors are freed first. ``parameters`` is made
    a list once, so a generator such as ``model.parameters()`` is still there
    for a restart. The solver's per-call attributes are put back when the
    call ends, so an integrand that integrates with the same solver (a double
    integral) does not change the outer call's batch size.

    Parameters
    ----------
    integrate : Callable
        The solver's ``integrate`` method.

    Returns
    -------
    Callable
        ``integrate`` with the restart.
    """
    signature = inspect.signature(integrate)

    @functools.wraps(integrate)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """
        Run ``integrate``, and again after each ``_Restart``.

        Parameters
        ----------
        *args : Any
            The positional arguments of ``integrate``, ``self`` first.
        **kwargs : Any
            Its keyword arguments.

        Returns
        -------
        Any
            What ``integrate`` returns.
        """
        call = signature.bind(*args, **kwargs).arguments
        parameters = call.get("parameters")
        if isinstance(parameters, torch.Tensor):
            call["parameters"] = [parameters]
        elif parameters is not None:
            call["parameters"] = (
                [p for g in parameters.param_groups for p in g["params"]]
                if isinstance(parameters, torch.optim.Optimizer)
                else list(parameters)
            )
        solver = call["self"]
        outer = {name: getattr(solver, name) for name in _PER_CALL}
        while True:
            try:
                return integrate(**call)  # type: ignore[call-arg]
            except _Restart as restart:
                state = restart.args[0]
            finally:
                vars(solver).update(outer)
            solver._release_memory()
            call.update(mesh=None, max_batch=None, state=state)

    return wrapper


class BackwardOutOfMemoryError(torch.OutOfMemoryError):
    """
    Raised when a backward pass of ``integrate(take_gradient=True)`` ran out of memory.

    PadaQuad restarts the integral by itself after such an error when it can
    put back the gradients the failed attempt added. It raises this error
    instead when it cannot: the integrand uses a custom autograd function and
    ``parameters=`` was not given, a parameter has a hook that runs after its
    gradient is added, or ``torch.distributed`` runs more than one process.
    The parameters' ``.grad`` then hold part of the integral's gradient: reset
    them (for example with ``optimizer.zero_grad()``) and call again with
    ``state=error.state`` (see :ref:`ug-memory`).

    Attributes
    ----------
    state : IntegrationState
        The state to call again with: the call's starting mesh and a smaller,
        converged batch size. Set after construction, so the error pickles
        with it (``BaseException`` pickles its ``__dict__``).
    """

    state: Any = None


# Largest number of points in one call: the benchmark's search limit and the
# cap on the batch size.
_MAX_N = 1_000_000_000

# Noise floors of the memory readers (GB): a reading below its reader's floor
# is not used in the fit. The CUDA counter is exact to 512-byte blocks; the CPU
# readers count resident memory, whose readings below a few MiB are mostly
# noise (0 when f reuses memory the process holds, or steps of 2 MiB: the
# kernel's counter batch on a 256-CPU node is 512 pages of 4 KiB).
_CUDA_NOISE_FLOOR_GB = 10_000 / _GB
_CPU_NOISE_FLOOR_GB = 4 * 2**20 / _GB


class _Reader(NamedTuple):
    """
    How the memory of one test call is measured on the solver's device.

    Attributes
    ----------
    reset : Callable[[], None]
        Starts a new peak measurement (a no-op where no resettable peak exists).
    read : Callable[[], int]
        The memory reading in bytes.
    E : float
        The most the reading can be too low because of the kernel's counters
        (GB); added once to every fitted memory.
    floor : float
        The noise floor (GB): readings below it are not fitted.
    counts_reuse : bool
        Whether a reading includes memory that f takes from memory the process
        freed earlier and kept (CUDA always; Linux when glibc's ``malloc_trim``
        returns that memory before each call; also True, wrongly, when an
        allocator such as jemalloc is preloaded in place of glibc's malloc,
        which the trim then cannot reach). Without it a reading can be far
        too low, so the benchmark keeps its steps at 10x (a call 2x larger than
        the previous one would reuse a large share of the previous call's
        memory unseen).
    """

    reset: Callable[[], None]
    read: Callable[[], int]
    E: float
    floor: float
    counts_reuse: bool


def _tensor_bytes(obj: Any, device: torch.device) -> int:
    """
    Return the bytes of the distinct tensor storages in obj on device.

    Used as a lower bound of a test call's memory: the input tile and f's
    outputs are alive when the call is read, so the call used at least their
    storages. A storage shared by several tensors (a view, or an output that
    is the input) is counted once. An output that is a view of a larger tensor
    f did not allocate in this call (a slice of a preallocated buffer) counts
    that tensor's whole storage, so the bound is then too high (safe: the
    batch gets smaller).

    Parameters
    ----------
    obj : Any
        A tensor, or tuples and lists of tensors (f's output may be
        ``(integrand, tracked_variables)``); anything else counts 0.
    device : torch.device
        Only storages on this device type count (the reader measures it; on a
        CUDA solver, storages on another GPU count too).

    Returns
    -------
    int
        Total bytes of the distinct storages found.
    """
    storages: dict[int, int] = {}
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, (tuple, list)):
            stack.extend(item)
        elif torch.is_tensor(item) and item.device.type == device.type:
            try:
                storage = item.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
            except (RuntimeError, NotImplementedError):
                # Sparse and other storage-less layouts: count the elements.
                storages[id(item)] = item.nelement() * item.element_size()
    return sum(storages.values())


def _fit_up_to_quadratic(
    n: Sequence[int], y: Sequence[float]
) -> tuple[float, float, float]:
    """
    Fit y = a + b n + d n^2 with a, b, d >= 0 to readings (n_i, y_i).

    The fit is nonnegative least squares (``scipy.optimize.nnls``): memory and
    time are sums of a fixed cost, a cost per point and a cost per pair of
    points, and nonnegative coefficients also make the fit nondecreasing in n.
    Unweighted least squares is dominated by the largest readings, which are
    the most reliable (small readings can read 0 when f reuses memory the
    process already holds). With k readings, min(k, 3) terms are fitted: b n
    for one reading, a + b n for two, all three from three on.

    Parameters
    ----------
    n : Sequence[int]
        Number of points of each reading (positive).
    y : Sequence[float]
        The quantity read at each n (memory in GB, or time in s).

    Returns
    -------
    tuple[float, float, float]
        The coefficients (a, b, d); those not fitted are 0.
    """
    # Imported here, not at module level: importing scipy takes about 0.25 s,
    # and only the memory benchmark needs it.
    from scipy.optimize import nnls  # noqa: PLC0415

    powers = [(1,), (0, 1), (0, 1, 2)][min(len(n), 3) - 1]
    # Columns in units of the largest n, so n^2 (up to 1e18) stays well scaled.
    n_max = max(n)
    A = np.array([[(ni / n_max) ** p for p in powers] for ni in n], dtype=float)
    scaled, _ = nnls(A, np.array(y, dtype=float))
    coeffs = [0.0, 0.0, 0.0]
    for p, x in zip(powers, scaled, strict=True):
        coeffs[p] = float(x) / n_max**p
    return coeffs[0], coeffs[1], coeffs[2]


def _largest_n(coeffs: tuple[float, float, float], budget: float) -> int:
    """
    Return the largest n with a + b n + d n^2 <= budget, at most ``_MAX_N``.

    It is the positive root of d n^2 + b n - R = 0, R = budget - a, written as

        n = 2 R / (b + sqrt(b^2 + 4 d R)),

    which avoids cancellation when d is tiny and equals R / b when d = 0.

    Parameters
    ----------
    coeffs : tuple[float, float, float]
        (a, b, d), all >= 0, as returned by ``_fit_up_to_quadratic``.
    budget : float
        The bound on a + b n + d n^2.

    Returns
    -------
    int
        0 when even n = 0 exceeds the budget (R <= 0), ``_MAX_N`` when the
        quantity does not grow with n (b = d = 0) or the budget is infinite.
    """
    a, b, d = coeffs
    R = budget - a
    if R <= 0:
        return 0
    if (b <= 0 and d <= 0) or math.isinf(R):
        return _MAX_N
    return int(min(_MAX_N, 2 * R / (b + math.sqrt(b * b + 4 * d * R))))


def _is_out_of_memory(e: BaseException) -> bool:
    """
    Return True if e signals that an allocation did not fit in memory.

    Covers a GPU out-of-memory error, a refused PyTorch CPU allocation (a plain
    RuntimeError naming DefaultCPUAllocator), a failed cudaMalloc or
    cudaHostAlloc outside PyTorch's allocator ("CUDA error: out of memory"), a
    CUDA library's allocation status (CUBLAS_STATUS_ALLOC_FAILED and the
    cuSOLVER, cuSPARSE and cuFFT equivalents; cuDNN 9's
    CUDNN_STATUS_INTERNAL_ERROR_*_ALLOCATION_FAILED), and Python's MemoryError
    (raised by numpy). The strings were checked in torch 2.8's libraries.

    Parameters
    ----------
    e : BaseException
        The exception raised by an allocation or by a call of the integrand.

    Returns
    -------
    bool
        True if ``e`` is one of the out-of-memory errors listed above, and
        not a ``BackwardOutOfMemoryError``.
    """
    if isinstance(e, BackwardOutOfMemoryError):
        # Not a retryable error: an outer call of f must not shrink and retry,
        # which would leave the inner call's partial gradients in place.
        return False
    return isinstance(e, (torch.OutOfMemoryError, MemoryError)) or any(
        s in str(e)
        for s in (
            "DefaultCPUAllocator",
            "out of memory",
            "ALLOC_FAILED",
            "ALLOCATION_FAILED",
        )
    )


class _MemoryManagement:
    """
    Memory benchmark and batch-size budget, mixed into ``AdaptiveQuadrature``.

    Uses these attributes of the solver: ``device`` (set by
    ``DistributedEnvironment``), ``C`` (nodes per subinterval), and
    ``f_mem_coeffs``, ``_oom_max_batch``, ``init_max_batch``,
    ``init_total_mem_usage`` and ``limit_benchmark_time`` (set in
    ``AdaptiveQuadrature.__init__``).
    """

    device: torch.device
    dtype: torch.dtype
    C: int
    _split_f_output: Callable[
        [Any], tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]
    ]
    f_mem_coeffs: tuple[float, float, float] | None
    _batch: int
    _batch_converged: bool
    _total_mem_usage: float
    _forward_bytes: float
    _backward_ratio: float
    _oom_max_batch: int | None
    init_max_batch: int | None
    init_total_mem_usage: float
    limit_benchmark_time: bool

    @staticmethod
    def _check_total_mem_usage(total_mem_usage: float) -> None:
        """
        Validate that ``total_mem_usage`` is a fraction in (0, 1].

        Called by the constructor, ``integrate`` and ``get_max_batch``, so
        every entry point rejects the same values.

        Parameters
        ----------
        total_mem_usage : float
            The value to check.

        Raises
        ------
        ValueError
            If the value is not in (0, 1].
        """
        if not 0 < total_mem_usage <= 1:
            raise ValueError(
                "total_mem_usage is a fraction of the device's memory and must "
                f"satisfy 0 < total_mem_usage <= 1; got {total_mem_usage}."
            )

    def _cap_max_batch(self, max_batch: int) -> int:
        """
        Lower a batch size to the out-of-memory cap, if one is set.

        The one place the cap ``_oom_max_batch`` (set after an out-of-memory
        error, until the next ``integrate`` call starts) is applied: to every
        benchmark's result and to ``get_max_batch``.

        Parameters
        ----------
        max_batch : int
            The batch size to cap.

        Returns
        -------
        int
            ``min(max_batch, _oom_max_batch)``, or ``max_batch`` without a cap.
        """
        if self._oom_max_batch is None:
            return max_batch
        return min(max_batch, self._oom_max_batch)

    def get_max_batch(
        self,
        total_mem_usage: float | None = None,
        *,
        f: Callable[..., Any] | None = None,
        t: torch.Tensor | None = None,
        take_gradient: bool = True,
        f_args: tuple[Any, ...] = (),
    ) -> int | None:
        """
        Return the largest number of integrand evaluations per batch.

        With ``f``, measure it: call ``f`` once on the point ``t`` (an output
        check that also warms ``f`` up), run the memory benchmark for ``f`` and
        return the batch size it gives for the memory free now. Pass that value
        as ``max_batch`` to every later call with the same ``f`` and settings,
        which then skip the benchmark. The constructor's ``max_batch`` is not
        used, and a cap left by an earlier out-of-memory error is cleared
        first. The measurement depends on ``take_gradient``, the dtype of ``t``
        and the solver's device, so measure with the settings of the calls that
        will use it.

        Without ``f``, the benchmark is not run: the ``max_batch`` given at
        construction is returned, else the number of evaluations that fit in
        the memory free now, with the memory per call the last benchmark
        measured. After an out-of-memory error, the result is capped at the
        reduced batch size chosen then, until the next ``integrate`` call
        starts. To reuse the batch size of a particular call, pass its
        ``state`` instead.
        See :ref:`ug-memory`.

        Parameters
        ----------
        total_mem_usage : float or None
            Fraction of the device's total memory that may be in use; the rest
            is kept free. None uses the value given at construction.
        f : Callable or None
            The integrand to measure, called as ``f(t, *f_args)`` on points of
            shape [N, T]. None (default) measures nothing.
        t : torch.Tensor or None
            A point of the integration interval, shape [T], on which ``f`` is
            evaluated, in the dtype of the calls that will use the result,
            float32 or float64 (a float64 point takes twice the memory of a
            float32 one). Moved to the solver's device; the solver's dtype is
            not changed. Required with ``f``.
        take_gradient : bool
            The ``take_gradient`` of the calls that will use the measurement;
            with True (default) the batch leaves memory for backpropagation.
        f_args : tuple
            Extra positional arguments of ``f``; tensors are moved to the
            solver's device. Default ``()``.

        Returns
        -------
        int or None
            The batch size, or None when no source is available (no ``f``, no
            ``max_batch`` given and no benchmark run yet).

        Raises
        ------
        ValueError
            If ``total_mem_usage`` is outside (0, 1], if ``f`` is given without
            a float32 or float64 point ``t`` of shape [T], or if ``f`` does not
            return values of shape [N, D].

        Examples
        --------
        Measure once, then pass the value to every call (the measurement takes
        up to a minute, so this example is not run):

        >>> import torch
        >>> from padaquad import adaptive_quadrature
        >>> solver = adaptive_quadrature("uniform", "gk21", device="cpu")
        >>> t0 = torch.tensor([0.0], dtype=torch.float64)
        >>> m = solver.get_max_batch(f=torch.exp, t=t0)  # doctest: +SKIP
        >>> result, state = solver.integrate(  # doctest: +SKIP
        ...     torch.exp, mesh_init=t0, mesh_final=t0 + 1, max_batch=m
        ... )
        """
        if total_mem_usage is None:
            total_mem_usage = self.init_total_mem_usage
        self._check_total_mem_usage(total_mem_usage)
        if f is not None:
            if (
                not isinstance(t, torch.Tensor)
                or t.dim() != 1
                or t.dtype not in (torch.float32, torch.float64)
            ):
                raise ValueError(
                    "get_max_batch(f=...) needs a point t of shape [T], float32 "
                    "or float64 (the dtypes the solver integrates in), on which "
                    "f can be evaluated."
                )
            # t keeps its dtype: the measurement is for calls in that dtype,
            # whatever dtype an earlier call left the solver in.
            t = t.to(self.device)
            f_args = tuple(
                a.to(self.device) if torch.is_tensor(a) else a for a in f_args
            )
            # A cap from an earlier call or measurement would lower this one.
            self._oom_max_batch = None
            # One point, shaped [1, T] like every batch f receives: checks the
            # output and keeps one-time costs (CUDA start-up, lazy loading) out
            # of the benchmark, as integrate() does.
            integrand, _ = self._split_f_output(f(t.reshape(1, -1), *f_args))
            if integrand.dim() < 2:
                raise ValueError(
                    "f must return values of shape [N, D] for points of shape "
                    f"[N, T]; got shape {list(integrand.shape)}."
                )
            return self._setup_memory_checks(
                f,
                t,
                take_gradient=take_gradient,
                total_mem_usage=total_mem_usage,
                f_args=f_args,
            )
        if self.init_max_batch is not None:
            return self._cap_max_batch(self.init_max_batch)
        if self.f_mem_coeffs is not None:
            return self._cap_max_batch(self._get_max_f_evals(total_mem_usage))
        return None

    def _get_cpu_memory(self) -> tuple[float, float]:
        """
        Query available and total CPU (system) memory in GB.

        Returns
        -------
        tuple[float, float]
            Tuple of (free_gb, total_gb).
        """
        mem = psutil.virtual_memory()
        free = mem.available / _GB
        total = mem.total / _GB
        return free, total

    def _get_cuda_memory(self) -> tuple[float, float]:
        """
        Query available and total CUDA (GPU) memory in GB.

        Accounts for both free system GPU memory and unused PyTorch cache
        memory (reserved but not allocated). This gives a more accurate
        picture of how much memory is truly available for new allocations.

        Returns
        -------
        tuple[float, float]
            Tuple of (free_gb, total_gb).
        """
        mem_info = torch.cuda.mem_get_info(self.device)
        # Total memory on the GPU
        total_gpu = mem_info[1] / _GB
        # Memory that is free outside of the PyTorch cache
        free_gpu = mem_info[0] / _GB
        # Memory reserved for the PyTorch cache
        torch_cache = torch.cuda.memory_reserved(self.device) / _GB
        # Cache memory being used by tensors
        torch_cache_used = torch.cuda.memory_allocated(self.device) / _GB
        # Total free amount of memory that can be used
        total_free = free_gpu + (torch_cache - torch_cache_used)

        return total_free, total_gpu

    def _get_memory(self) -> tuple[float, float]:
        """
        Query available and total memory in GB for the active device type.

        Dispatches to _get_cuda_memory() or _get_cpu_memory() based on
        self.device.type.

        Returns
        -------
        tuple[float, float]
            Tuple of (free_gb, total_gb).
        """
        if self.device.type == "cuda":
            return self._get_cuda_memory()
        else:
            return self._get_cpu_memory()

    def _release_memory(self, empty_cache: bool = False) -> None:
        """
        Free memory after an out-of-memory error or a memory benchmark.

        Runs the garbage collector. With ``empty_cache``, on a CUDA solver,
        also empties PyTorch's caching allocator, which releases the cache of
        every GPU (NativeCachingAllocator::emptyCache in
        c10/cuda/CUDACachingAllocator.cpp loops over all devices). That helps
        only allocations made outside PyTorch's allocator (a CUDA library's,
        for example); after an error of PyTorch's own allocator it only slows
        the next calls down (measured on an A100: the first call after it ran
        about 1.3 s slower, and an out-of-memory episode took 1.4 to 6 s
        longer; the break-and-speed review of the ``change_memory`` branch,
        commit 59b7f11).
        A CPU solver never touches the GPU.

        Parameters
        ----------
        empty_cache : bool
            Whether to empty PyTorch's GPU cache too. Default False.
        """
        gc.collect()
        if empty_cache and self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _call_f(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        points: torch.Tensor,
        can_shrink: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None] | None:
        """
        Call f on points, or return None when it runs out of memory.

        The one place where the integration loop catches an out-of-memory
        error of f (``_is_out_of_memory``). After one, the memory f held is
        released, so the caller can try again with fewer points.

        Parameters
        ----------
        f : Callable
            The integrand.
        f_args : tuple
            Extra arguments passed to ``f`` after the points.
        points : torch.Tensor
            The points to evaluate. Shape: [N, T].
        can_shrink : bool
            Whether the caller can try again with fewer points.

        Returns
        -------
        tuple or None
            ``f``'s values [N, D] and its tracked variables (or None), as
            ``_split_f_output`` returns them; None when ``f`` ran out of
            memory.

        Raises
        ------
        torch.OutOfMemoryError
            If ``f`` ran out of memory and ``can_shrink`` is False.
        """
        outside = False
        try:
            return self._split_f_output(f(points, *f_args))
        except (RuntimeError, MemoryError) as e:
            if not _is_out_of_memory(e):
                raise
            if not can_shrink:
                raise torch.OutOfMemoryError(
                    f"{e}\n\nA call of the integrand f on {len(points)} points ran "
                    "out of memory, and the batch size cannot be reduced."
                ) from e
            # An error from outside PyTorch's allocator (a CUDA library's, or
            # the CPU's) is helped by emptying PyTorch's GPU cache too.
            outside = not isinstance(e, torch.OutOfMemoryError)
        # Released here, after the except block: inside it the error's
        # traceback still holds f's tensors, so they cannot be freed.
        self._release_memory(empty_cache=outside)
        return None

    def _setup_memory_checks(
        self,
        f: Callable[..., Any],
        node_test: torch.Tensor,
        take_gradient: bool,
        total_mem_usage: float,
        f_args: tuple[Any, ...] = (),
    ) -> int:
        """
        Fit the memory of one call of f on N points (``f_mem_coeffs``, GB).

        Returns the batch size the fit gives for the memory free at the end of
        the measurement (lowered to the out-of-memory cap, ``_cap_max_batch``).
        During an integration the batch size is recomputed at every evaluation
        from the memory free at that moment (``_get_max_f_evals``).

        f is called on N copies of ``node_test``, starting at N = 1. For each
        call, M is the memory the call used (``_measure_call``, with the reader
        chosen by ``_peak_memory_reader``; never less than the bytes of the
        input and output tensors) and t its wall time. A reading is *reliable*
        when M is at least the reader's noise floor (10 KB for CUDA, 4 MiB for
        the CPU readers); only reliable readings are fitted, as
        a + b N + d N^2 with a, b, d >= 0 (``_fit_up_to_quadratic``). The fit
        is *ready* when it uses three reliable readings and its memory grows
        by at least the noise floor between the first and the last of them
        (readings that grow less, or not at all, would give a fit that is
        nearly flat and allows far too many points). The next call has

            N_next = min(step N, n_mem, n_time),

        - step: 2 after a call that took more than 0.1 s, if the reader counts
          memory f reuses (CUDA; Linux with ``malloc_trim``), and 10 otherwise.
        - n_mem: the largest N whose predicted memory plus E fits in
          ``_MEMORY_BUDGET`` of the usable memory. The prediction is the fit
          once it is ready; before that, the worst case such a fit allows
          through the reliable reading with the largest N (or, with none,
          through the latest reading), y(k N) <= k^2 y(N) for k >= 1.
        - n_time: the largest N whose predicted time (the fit of all times from
          three calls on, the same worst case before) is at most 1 s, but at
          least 2 N for the first two calls (one or two times say nothing about
          how time grows) and while the fit is not ready (the 1 s limit is
          extended one doubling at a time, only as far as needed). In every
          case the call must be predicted to end within the 60 s total limit.

        So an f whose memory or time grows like N^2 gets smaller steps instead
        of a call that cannot fit. E is the most the kernel's counters can make
        a reading too low (nonzero only on Linux); it is a reading error, so it
        enters as a fixed amount, not one that grows with N.

        The 60 s total limit is checked before each call (no call is started
        that is predicted to end past it) and after it (a call can take longer
        than predicted, and a running call of f cannot be interrupted). The
        solver option ``limit_benchmark_time=False`` removes it: the search
        then goes on until the fit is ready, the memory budget ends it, or N
        reaches 1e9.

        The search stops, keeping the fit, when (in this order):

        1. the measurement has taken longer than 60 s, with the fit ready (a
           warning says so);
        2. c = s (M + E) / N changes by less than 1 % over each of the last
           two calls, the last three readings all reliable and the fit ready
           (the cost per point is constant, so the growth is linear);
        3. a call takes longer than 1 s, with the fit ready;
        4. N reaches 1e9;
        5. N_next < 2 N: the next call would not fit in memory, or would take
           too long, and a call barely larger than this one adds little.

        It raises ``RuntimeError``, asking for ``max_batch``, when:

        - a call uses more than 3 times the memory that growth like N^2 allows
          from an earlier reliable reading, M > 3 (N / N_i)^2 (M_i + E), with
          i the earlier reliable reading that gives the smallest bound (the
          factor 3 absorbs readings off by up to a factor 2). Either the
          memory grows faster than N^2, which the fit cannot follow, or (CPU
          readers) the earlier reading missed memory f reused; the benchmark
          cannot tell which (see the limitations in the Notes);
        - the fit is not ready when the 60 s run out, or the next call needed
          for it would fit in memory but is predicted to take longer than what
          is left of the 60 s (the message names ``limit_benchmark_time``);
        - no reading is reliable when the search ends for another reason (the
          memory budget, an out-of-memory error, or N = 1e9).

        When the memory budget, an out-of-memory error or N = 1e9 ends the
        search before the fit is ready, the stored fit is the worst case N^2
        through the reliable reading with the largest N, and a warning says
        so. That call was already near the budget, so the batch is about its
        size.

        The stored fit is m(B) = s ((a + E) + b B + d B^2), the memory of a
        batch of B points. s is ``_GRADIENT_MEMORY_SAFETY_FACTOR`` (a heuristic
        for backpropagation) when ``take_gradient`` and 1 otherwise. After an
        out-of-memory error at N > 1, the batch size is capped at
        ``_OOM_BACKOFF`` * N for the rest of this ``integrate`` call
        (``_oom_max_batch``, as the batch loop does after such an error).

        On a CPU reader that does not count reused memory (macOS, Windows, a
        C library without ``malloc_trim``), readings can be far too low in a
        process that has already evaluated f, so the batch size is also capped
        at the largest test call (it ran, so a batch that size fit then; at
        least the C nodes of one subinterval), through the same
        ``_oom_max_batch``. The start warning says so and recommends
        ``max_batch``.

        Parameters
        ----------
        f : Callable[..., Any]
            The integrand function to benchmark.
        node_test : torch.Tensor
            A sample node point for benchmarking. Shape: [T] or [1, T].
        take_gradient : bool
            Whether the integration will backpropagate per batch.
        total_mem_usage : float
            Fraction of total memory allowed (0 < value <= 1).
        f_args : tuple[Any, ...]
            Extra arguments passed to f.

        Returns
        -------
        int
            The measured batch size, what ``get_max_batch(f=...)`` returns.

        Raises
        ------
        torch.OutOfMemoryError
            A single evaluation of f does not fit.
        RuntimeError
            The memory grows faster than N^2, the fit was not ready within the
            time limit, no reading was reliable (see above), or the measurement
            failed for another reason; pass ``max_batch`` to skip it.

        Notes
        -----
        The benchmark is a heuristic for the common case: memory that grows
        linearly with N, or at most like N^2, read by a reader that sees every
        allocation (a GPU, or Linux with glibc). It cannot be right for every
        f, and ``max_batch`` skips it. Its known limitations, with the measured
        and simulated numbers behind them, are listed once, in the user guide:
        ``docs/user_guide/devices_memory.md``, section "Limitations of the
        memory benchmark" (label ``ug-memory-limitations``; on the website,
        https://khegazy.github.io/padaquad/user_guide/devices_memory.html#ug-memory-limitations).
        Read them before changing this method, and update them when a change
        fixes or adds one.
        """
        # How the memory of one call is read on this device (see _Reader).
        reader = self._peak_memory_reader()
        E = reader.E
        hidden_note = (
            ""
            if reader.counts_reuse
            else (
                "\nOn this system a memory reading misses memory that f reuses "
                "from memory the process already holds, so in a process that "
                "has already evaluated f the readings can be far too low. "
                "max_batch is then capped at the largest call of f this "
                "measurement made, but that call itself can need more memory "
                "than is available: pass max_batch when the memory of a batch "
                "matters."
            )
        )
        logger.warning(
            "PadaQuad is measuring how much memory one integrand (f) evaluation "
            "uses, to determine max_batch (the maximum number of evaluations per "
            "batch).\n"
            "\t - This measurement can be slow. Run it once, save the max_batch it "
            "returns, and pass that value to integrate().\n"
            "\t - The memory cost measurement may not be optimized for your "
            "machine and may result in an underutilization of your hardware (or, "
            "for some integrands, a batch too large for memory); see "
            "https://khegazy.github.io/padaquad/user_guide/devices_memory.html#ug-memory-limitations "
            "for more details. The only way to guarantee "
            "that max_batch makes full use of your hardware's memory is to measure "
            f"max_batch yourself.{hidden_note}"
        )
        t0 = time.time()

        # node_test is one quadrature node, [T] or [1, T]. Make it a [1, T] row
        # on the solver's device, so each test call can stack N copies of it.
        assert len(node_test.shape) <= 2
        if len(node_test.shape) == 2:
            node_test = node_test[0]
        node_test = node_test.unsqueeze(0).to(self.device)
        self.f_mem_coeffs = None

        # The test calls run f forward only. With take_gradient=True the
        # integration also backpropagates through every batch, which needs
        # roughly as much memory again, so the stored fit is scaled by s.
        s = _GRADIENT_MEMORY_SAFETY_FACTOR if take_gradient else 1.0
        time_limit = 1.0  # seconds per call, once the fit is ready
        # Seconds for the whole measurement; the solver option
        # limit_benchmark_time=False removes the limit.
        total_limit = 60.0 if self.limit_benchmark_time else math.inf
        unlimit = (
            " Pass limit_benchmark_time=False (a solver option, also accepted "
            "by padaquad.integrate) to remove this limit."
        )
        # A call using more than this many times the memory that growth like
        # N^2 allows from the previous reliable reading raises (faster than
        # N^2). Readings above the floor were off by up to a factor 2 (0.5x to
        # 1.4x of the true size on a Mac, in a fresh process; in a process that
        # had evaluated f before, as low as 0.03x), so an earlier reading at
        # 0.5x and this one at 1.5x make N^2 growth look 3 times faster.
        # Growth like N^p is caught from readings k times apart in N when
        # k^(p-2) > 3: N^2.5 across 10x, N^3 across 4x.
        growth_tolerance = 3.0

        # Search state: the points, memory (GB) and wall time (s) of every call
        # so far, and which readings are reliable. c is the latest cost per
        # point, s * (M + E) / N, and c_prev, c_prev2 those of the two calls
        # before; they serve only the convergence stop. reason ends the search
        # and is reported by the end message; error, when set, is raised
        # instead.
        Ns: list[int] = []
        Ms: list[float] = []
        ts: list[float] = []
        reliable: list[bool] = []
        N, c, c_prev, c_prev2 = 1, 0.0, 0.0, 0.0
        reason: str | None = None
        error: str | None = None

        def memory_model() -> tuple[tuple[float, float, float], bool]:
            """
            Return the memory model of one call from the readings so far.

            The fit of the reliable readings only, when it is ready: three or
            more of them, and the fitted memory grows by at least the noise
            floor from the first to the last (less growth is within the noise,
            and a nearly flat fit would allow far too many points). Otherwise
            the worst case a fit a + b N + d N^2 with a, b, d >= 0 allows
            through the reliable reading with the largest N (the latest reading
            if none is reliable): y(k N) <= k^2 y(N). Used both to size the
            next call and for the stored fit, so the two cannot disagree.

            Returns
            -------
            tuple[tuple[float, float, float], bool]
                The coefficients (a, b, d) in GB, without E, and whether the
                fit is ready.
            """
            kept = [i for i, r in enumerate(reliable) if r]
            if len(kept) >= 3:
                a, b, d = _fit_up_to_quadratic(
                    [Ns[i] for i in kept], [Ms[i] for i in kept]
                )
                first, last = Ns[kept[0]], Ns[kept[-1]]
                if b * (last - first) + d * (last**2 - first**2) >= reader.floor:
                    return (a, b, d), True
            # Ns increases, so the last reliable reading has the largest N.
            j = kept[-1] if kept else len(Ns) - 1
            return (0.0, 0.0, Ms[j] / Ns[j] ** 2), False

        try:
            while reason is None:
                # Usable memory (after the total_mem_usage reserve) right before
                # this call: what the next call must fit in.
                usable = self._get_usable_memory(total_mem_usage)
                measured = self._measure_call(
                    f, f_args, node_test, N, reader.reset, reader.read, total_mem_usage
                )

                if measured is None:
                    # N > 1 points did not fit (at N = 1 _measure_call raises).
                    # The peak counters count only the allocations that
                    # succeeded, so the failed call is not a reading. Cap the
                    # batch below N for the rest of this integrate() call, as
                    # the batch loop does after an out-of-memory error.
                    # Can fall below self.C (a known limitation; see the Notes).
                    self._oom_max_batch = max(1, int(_OOM_BACKOFF * N))
                    reason = "an evaluation ran out of memory"
                    continue

                M, seconds = measured
                Ns.append(N)
                Ms.append(M)
                ts.append(seconds)
                reliable.append(reader.floor <= M)
                n_rel = sum(reliable)
                c_prev2, c_prev, c = c_prev, c, s * (M + E) / N
                elapsed = time.time() - t0
                (a, b, d), ready = memory_model()

                # Faster than N^2: with a, b, d >= 0, M(N) <= (N / N_i)^2 M(N_i)
                # for every earlier N_i, so the smallest of these bounds is
                # tested. Unreliable readings are noise and are skipped. A
                # fixed cost enters every bound, so a large one still hides
                # faster growth until it dominates the readings.
                bounds = {
                    i: (N / Ns[i]) ** 2 * (Ms[i] + E)
                    for i in range(len(Ns) - 1)
                    if reliable[i]
                }
                j = min(bounds, key=lambda i: bounds[i], default=None)
                bound = 0.0 if j is None else bounds[j]
                if j is not None and growth_tolerance * bound < M:
                    error = (
                        f"The memory of one call of f grew to {M * _GB:.3g} "
                        f"bytes at {N} points, more than {growth_tolerance:g} "
                        f"times the {bound * _GB:.3g} bytes that growth like N^2 "
                        f"allows from the {Ms[j] * _GB:.3g} bytes at {Ns[j]} "
                        "points. Either the memory of f grows faster than N^2, "
                        "which padaquad's fit cannot follow, or the earlier "
                        "reading was too low because f reused memory the process "
                        "already held (possible on a CPU); padaquad cannot tell "
                        "which, so it cannot choose a batch size that fits."
                    )
                    reason = "a call used more memory than growth like N^2 allows"
                # Stops, checked in order.
                elif elapsed > total_limit:
                    if not ready:
                        error = (
                            f"The memory measurement took longer than its "
                            f"{total_limit:g} s limit before the readings allowed "
                            f"a fit ({n_rel} reached the noise floor of "
                            f"{reader.floor * _GB:.3g} bytes; a fit needs three, "
                            "whose memory grows by at least the floor)."
                            f"{unlimit}"
                        )
                    reason = f"the measurement took longer than {total_limit:g} s"
                elif (
                    ready
                    and all(reliable[-3:])
                    and abs(c - c_prev) < 0.01 * c_prev
                    and abs(c_prev - c_prev2) < 0.01 * c_prev2
                ):
                    # The cost per point changed by < 1 % over each of the last
                    # two calls, all three readings reliable: the fixed per-call
                    # cost is spread out and the growth is linear. Those three
                    # readings grow by about 3 floors or more (proportional to
                    # N, at least the floor, N at least 4 times apart), but the
                    # fit uses every reliable reading, and an earlier one larger
                    # than these (memory f allocated only in that call) can
                    # flatten it, so the fit must also be ready.
                    reason = "the cost per evaluation stopped changing"
                elif ready and seconds > time_limit:
                    # The backstop; before the fit is ready, a call may take
                    # longer (see the time extension below).
                    reason = f"an evaluation took longer than {time_limit} s"
                elif N >= _MAX_N:
                    reason = "the 1e9 search limit was reached"
                else:
                    # Size the next call: min(step N, n_mem, n_time).
                    time_fit = (
                        _fit_up_to_quadratic(Ns, ts)
                        if len(Ns) >= 3
                        else (0.0, 0.0, seconds / N**2)
                    )
                    # The test calls run forward only, so s is not applied. A
                    # call that does not fit would run out of memory on a GPU,
                    # and on a CPU the operating system could kill the process
                    # without an error. E is a fixed reading error.
                    n_mem = _largest_n((a + E, b, d), _MEMORY_BUDGET * usable)
                    n_time = _largest_n(time_fit, time_limit)
                    if len(Ns) < 3 or not ready:
                        # One or two times say nothing about how time grows (the
                        # worst case would end the search after one call of any
                        # f slower than 0.25 s), and before the fit is ready the
                        # 1 s limit must not end the search: allow doubling N,
                        # which takes at most 4 times this call in the worst
                        # case.
                        n_time = max(n_time, 2 * N)
                    # No call is started that is predicted to end past the total
                    # limit (infinite under limit_benchmark_time=False).
                    remaining = total_limit - elapsed
                    n_time = min(n_time, _largest_n(time_fit, remaining))
                    # Small steps once calls get slow, so a sudden jump in time
                    # costs little; only where the reader counts reused memory,
                    # because a 2x larger call would otherwise reuse unseen a
                    # large share of this call's memory.
                    step = 2 if reader.counts_reuse and seconds > 0.1 else 10
                    N_next = min(step * N, n_mem, n_time)
                    if N_next >= 2 * N:
                        N = N_next
                    elif not ready and n_mem >= 2 * N:
                        # Only the total limit stops the search (the time
                        # allowance above is at least 2 N); when the next call
                        # would not fit in memory either, the memory budget ends
                        # it.
                        error = (
                            f"The next test call needed for a fit ({n_rel} "
                            f"readings reached the noise floor of "
                            f"{reader.floor * _GB:.3g} bytes; a fit needs three, "
                            "whose memory grows by at least the floor) is "
                            f"predicted to take longer than the {remaining:.1f} s "
                            f"left of the {total_limit:g} s limit of the memory "
                            f"measurement.{unlimit}"
                        )
                        reason = "the next test would take too long"
                    else:
                        # A call barely larger than this one adds little. Near
                        # 1e9 with 2x steps, the 1e9 limit can be reported as
                        # memory (a known limitation; see the Notes).
                        reason = (
                            "the next test would not fit in memory"
                            if n_mem < 2 * N
                            else "the next test would take too long"
                        )
        except torch.OutOfMemoryError:
            # A single evaluation of f does not fit (raised by _measure_call at
            # N = 1): memory really is too small, so pass the error through.
            raise
        except Exception as e:
            # Anything else (for example a bug in f that appears only for more
            # than one point, or a CUDA error) means c could not be measured.
            raise RuntimeError(
                f"padaquad could not measure the memory cost of f after "
                f"{time.time() - t0:.2f} s ({type(e).__name__}: {e}). Specify "
                "max_batch to skip the measurement."
            ) from e

        self._release_memory(empty_cache=True)
        n_rel = sum(reliable)
        (a, b, d), ready = memory_model()
        if error is None and n_rel == 0:
            error = (
                f"No test call of f used at least the noise floor of "
                f"{reader.floor * _GB:.3g} bytes before the search ended "
                f"({reason})."
            )
        if error is not None:
            # Raised here, outside the try block, so it is not reported as a
            # failed measurement. The readings let the user pick max_batch.
            readings = ", ".join(
                f"N={n}: {m * _GB / 2**20:.2f} MiB in {t:.2f} s"
                + ("" if r else " (below the noise floor)")
                for n, m, t, r in zip(Ns, Ms, ts, reliable, strict=True)
            )
            raise RuntimeError(
                f"{error} Readings after {time.time() - t0:.1f} s: {readings}. "
                "Specify max_batch (to integrate() or the solver): the largest "
                "number of points per call of f that fits in memory."
                f"{hidden_note}"
            )

        # The memory of one call of B points during the integration:
        # m(B) = s * ((a + E) + b B + d B^2). E is added once: it is a reading
        # error, not a cost per point. When the fit is not ready (the memory
        # budget, an out-of-memory error or N = 1e9 ended the search), the
        # model is the cautious worst case.
        self.f_mem_coeffs = (s * (a + E), s * b, s * d)
        if not reader.counts_reuse:
            # The readings can miss memory f reuses, so the fit can allow far
            # more than fits; the largest test call ran, so it fit then. Never
            # below one subinterval's C nodes, the smallest batch the
            # take_gradient=True path can evaluate (the fit still limits it).
            # The cap lasts until the next measurement.
            largest = max(*Ns, self.C)
            self._oom_max_batch = min(largest, self._oom_max_batch or largest)

        # The estimate depends on take_gradient: the fit includes the factor s
        # only when it is True. _cap_max_batch applies the cap an out-of-memory
        # error above set, so the message reports the batch actually used.
        usable = self._get_usable_memory(total_mem_usage)
        max_batch = self._cap_max_batch(self._get_max_f_evals(total_mem_usage))
        a, b, d = self.f_mem_coeffs
        note = ""
        if not ready:
            note = (
                f" The readings did not allow a fit ({n_rel} reached the noise "
                "floor; a fit needs three, whose memory grows by at least the "
                "floor), so max_batch assumes the worst case, memory growing "
                "like N^2 from the largest of them, and can be smaller than "
                "what fits; pass max_batch to choose it yourself."
            )
        elif d * max_batch**2 > 0.1 * (a + b * max_batch + d * max_batch**2):
            # Name the N^2 growth only when it matters at the batch size: every
            # fit of a linear f leaves a tiny d > 0. Strict, so a batch of 0
            # with no fixed cost (a = 0) is not flagged.
            note = (
                " The memory grows like max_batch^2, so a batch larger than "
                "this does not fit; on a CPU, a batch that does not fit can kill "
                "the process without an error."
            )
        logger.warning(
            f"Memory measurement finished in {time.time() - t0:.2f} s (stopped "
            f"because {reason}): one call on N points uses about "
            f"{a * _GB / 2**20:.1f} MiB + {b * _GB:.0f} bytes per evaluation * N "
            f"+ {d * _GB:.3g} bytes * N^2. With the {usable:.1f} GB usable now, "
            f"max_batch is about {max_batch} for take_gradient={take_gradient}; "
            f"pass max_batch={max_batch} to integrate()."
            f"{note}"
        )
        return max_batch

    def _peak_memory_reader(self) -> _Reader:
        """
        Choose how the memory one test call uses (M) is measured on this device.

        M = read() after the call minus read() before it, with reset() called
        right before the "before" reading. Three methods, best first:

        - CUDA: PyTorch's caching allocator tracks the peak of allocated memory
          per device. ``reset`` sets that peak to the memory allocated now, and
          ``read`` returns the peak. Exact to its 512-byte allocation blocks, so
          the noise floor is 10 KB, and it counts every allocation, including
          memory f reuses. Resetting also resets ``max_memory_reserved`` for
          ``self.device``. Memory allocated outside PyTorch's allocator (custom
          ``cudaMalloc``, CuPy, NCCL) is not counted.
        - CPU on Linux: writing 5 to ``/proc/self/clear_refs`` (Linux >= 4.0)
          resets the process's peak resident memory, ``VmHWM`` in
          ``/proc/self/status``, to its current value; ``read`` returns
          ``VmHWM``. The reset also resets what the caller's own ``getrusage``
          or ``/usr/bin/time`` report as the peak from then on. ``ru_maxrss``
          is not read: it also includes the peak of the process this one was
          started from, so in a worker started by a larger process
          (``subprocess``, multiprocessing ``spawn``) it would stay at the
          parent's peak. Before the reset, glibc's ``malloc_trim(0)`` returns
          the heap pages the process freed earlier to the system, so memory f
          reuses raises resident memory again and is counted (glibc trims every
          arena; it has no effect when another allocator such as jemalloc or
          tcmalloc is preloaded, and C libraries without it, such as musl, skip
          it). The trim runs before the call's clock starts; pages it returned
          must be mapped again, so a test call can be slower than the same
          batch during the integration (the time fit errs on the long side).
          The kernel keeps each CPU's share of the resident-memory count
          private until it reaches max(32, 2 * online CPUs) pages
          (``lib/percpu_counter.c``, ``compute_batch_value``; the counters are
          read without those shares, ``get_mm_counter``), and only CPUs the
          process runs on hold a share; a reading is the difference of two
          counts, so it can be low by up to
          E = 2 * (CPUs the process may run on) * max(32, 2 * online CPUs)
          pages (Linux >= 6.2 per-CPU counters; older kernels count per thread
          instead, which this E does not describe). Noise floor 4 MiB.
        - Other CPUs (macOS, Windows, a locked ``/proc``): no resettable peak
          exists. ``read`` returns the current resident memory, so M is the
          memory still held after the call (its result is kept alive for the
          reading). Memory that f frees before returning, such as the
          intermediates of a Jacobian, is missed. Noise floor 4 MiB.

        Both CPU methods count resident memory, so without the trim they miss
        memory f allocates inside pages the process already holds (freed
        earlier and kept by the allocator): M is then too low by up to all of
        it, which E does not cover. On macOS a popcornn-like integrand
        (per-point gradients, about 270 bytes N^2) that uses 255 MiB at
        N = 1000 in a fresh process read 51 MiB there in the third benchmark
        run in one process, and 0 MiB in another run; macOS's
        ``malloc_zone_pressure_relief`` released nothing.

        Returns
        -------
        _Reader
            The reset and read functions, E, the noise floor, and whether the
            reading counts memory f reuses.
        """
        if self.device.type == "cuda":

            def reset_cuda() -> None:
                """Set the device's peak of allocated memory to its current value."""
                torch.cuda.reset_peak_memory_stats(self.device)

            def read_cuda() -> int:
                """
                Read the device's peak of allocated memory.

                Returns
                -------
                int
                    The peak in bytes since the last reset.
                """
                return torch.cuda.max_memory_allocated(self.device)

            return _Reader(reset_cuda, read_cuda, 0.0, _CUDA_NOISE_FLOOR_GB, True)

        clear_refs = Path("/proc/self/clear_refs")
        try:
            # Probe: succeeds only on Linux with a writable /proc.
            clear_refs.write_text("5")
        except OSError:

            def no_reset() -> None:
                """Do nothing: no resettable peak exists on this system."""

            def read_rss() -> int:
                """
                Read the process's current resident memory.

                Returns
                -------
                int
                    The resident memory in bytes.
                """
                return psutil.Process().memory_info().rss

            return _Reader(no_reset, read_rss, 0.0, _CPU_NOISE_FLOOR_GB, False)

        # glibc's malloc_trim, looked up in the process's own C library; None
        # where the library has none (musl) or cannot be opened.
        malloc_trim: Callable[[int], int] | None
        try:
            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        except OSError:
            malloc_trim = None
        if malloc_trim is not None:
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int

        def reset_linux() -> None:
            """Return freed heap memory to the system, then reset VmHWM."""
            if malloc_trim is not None:
                malloc_trim(0)
            clear_refs.write_text("5")

        def read_vmhwm() -> int:
            """
            Read the process's peak resident memory, VmHWM.

            Returns
            -------
            int
                The peak in bytes since the last reset.
            """
            # The line reads "VmHWM:   14336 kB", in KiB.
            status = Path("/proc/self/status").read_text()
            return int(status.split("VmHWM:")[1].split()[0]) * 1024

        online = os.cpu_count() or 1
        used = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else online
        )
        E = 2 * used * max(32, 2 * online) * os.sysconf("SC_PAGE_SIZE") / _GB
        return _Reader(
            reset_linux, read_vmhwm, E, _CPU_NOISE_FLOOR_GB, malloc_trim is not None
        )

    def _measure_call(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        node_test: torch.Tensor,
        N: int,
        reset_peak: Callable[[], None],
        read_memory: Callable[[], int],
        total_mem_usage: float,
    ) -> tuple[float, float] | None:
        """
        Call f once on N copies of node_test; return the memory and time it took.

        The memory is the reader's difference, but never less than the bytes
        of the input tile and of f's outputs on this device (``_tensor_bytes``):
        those are alive when the call is read, so the call used at least that
        much. This removes readings of 0 when f reuses memory the process
        already holds, and is exact for an f that allocates nothing else.

        Parameters
        ----------
        f, f_args : Callable[..., Any], tuple[Any, ...]
            The integrand and its extra arguments.
        node_test : torch.Tensor
            One node as a [1, T] row on ``self.device``.
        N : int
            Number of points in this call.
        reset_peak, read_memory : Callable[[], None], Callable[[], int]
            The measurement method from
            ``_peak_memory_reader``.
        total_mem_usage : float
            Only quoted in the single-evaluation error.

        Returns
        -------
        tuple[float, float] | None
            (M, seconds): the memory the call used (GB) and its wall time, or
            None if the call ran out of memory (N > 1).

        Raises
        ------
        torch.OutOfMemoryError
            N = 1 ran out of memory, so a single
            evaluation of f does not fit.
        Exception
            any other error from f propagates unchanged.
        """
        # Read "before" first, then build the input: the N-point input tile is
        # memory the integration also needs per point, so it is counted in M.
        reset_peak()
        before = read_memory()
        t0 = time.time()
        out_of_memory = False
        t_input: torch.Tensor | None = None
        result = None
        try:
            t_input = torch.tile(node_test, (N, 1))
            result = f(t_input, *f_args)
        except (RuntimeError, MemoryError) as e:
            if not _is_out_of_memory(e):
                raise
            if N == 1:
                raise torch.OutOfMemoryError(
                    f"{e}\n\nA single evaluation of f does not fit in memory on "
                    f"{self.device} with total_mem_usage={total_mem_usage}."
                ) from e
            out_of_memory = True
            outside = not isinstance(e, torch.OutOfMemoryError)
        if out_of_memory:
            # Clean up here, after the except block: inside it, the error's
            # traceback still holds f's tensors, so they could not be freed.
            t_input = result = None
            self._release_memory(empty_cache=outside)
            return None

        if self.device.type == "cuda":
            # GPU calls return once their work is queued, not when it is done;
            # wait for it, so the time below is the real compute time.
            torch.cuda.synchronize(self.device)
        seconds = time.time() - t0
        # Read while the result is still alive, so a before/after difference
        # (the third method) counts the output too.
        M = max(read_memory() - before, _tensor_bytes((t_input, result), self.device))
        del result, t_input
        return M / _GB, seconds

    def _get_usable_memory(self, total_mem_usage: float) -> float:
        """
        Compute how much memory (in GB) can be used for integrand evaluations.

        Reserves a buffer of (1 - total_mem_usage) * total_memory to avoid
        out-of-memory errors from other system/PyTorch allocations.

        Parameters
        ----------
        total_mem_usage : float
            Fraction of total memory allowed (0 < value <= 1).

        Returns
        -------
        float
            Usable memory in GB (non-negative).
        """
        free, total = self._get_memory()
        buffer = (1 - total_mem_usage) * total
        return max(0, free - buffer)

    def _get_max_f_evals(self, total_mem_usage: float) -> int:
        """
        Compute the maximum number of integrand evaluations that fit in memory.

        The largest B whose fitted memory m(B) = a + b B + d B^2
        (``f_mem_coeffs``, measured by ``_setup_memory_checks``) is at most
        ``_MEMORY_BUDGET`` times the usable memory, capped at 1e9
        (``_largest_n``).

        Parameters
        ----------
        total_mem_usage : float
            Fraction of total memory allowed.

        Returns
        -------
        int
            Maximum number of evaluations (integer).

        Raises
        ------
        RuntimeError
            If the memory benchmark has not run (``f_mem_coeffs`` is None).
        """
        if self.f_mem_coeffs is None:
            raise RuntimeError(
                "The memory of f is unknown; run the memory benchmark "
                "(_setup_memory_checks) before sizing a batch."
            )
        usable = self._get_usable_memory(total_mem_usage)
        return _largest_n(self.f_mem_coeffs, _MEMORY_BUDGET * usable)

    def _batch_for(self, n_waiting: int) -> int:
        """
        Return the size of the next call of f when n_waiting points are waiting.

        The batch size ``_batch`` of this ``integrate`` call, or, while it has
        not converged (``_batch_converged``), ``_GROWTH`` times it: so a call
        that needs more points than ``_batch`` tries 10 times as many, and an
        out-of-memory error then shrinks it and sets it converged. Never more
        than n_waiting.

        Parameters
        ----------
        n_waiting : int
            Number of points waiting to be evaluated.

        Returns
        -------
        int
            The number of points of the next call of f.
        """
        growth = 1 if self._batch_converged else _GROWTH
        return min(n_waiting, growth * self._batch)

    def _initialize_max_batch(
        self,
        f: Callable[..., Any],
        f_args: tuple[Any, ...],
        node: torch.Tensor,
        n_first: int,
        total_mem_usage: float,
    ) -> tuple[int, bool]:
        """
        Find the starting batch size of a call given no max_batch and no state.

        Calls f, forward only, on 10, 100 and 1,000 (``_LARGEST_TEST_CALL``)
        copies of ``node``, stopping at the order of the first pass (the
        largest power of 10 not above n_first), only to see whether f runs at
        these sizes; no memory is read. A call
        that runs out of memory is repeated at ``_shrink_batch`` of its size
        until one runs, and no larger call follows.

        Parameters
        ----------
        f, f_args : Callable[..., Any], tuple[Any, ...]
            The integrand and its extra arguments.
        node : torch.Tensor
            One point as a [1, T] row on ``self.device``.
        n_first : int
            Number of points of the first pass.
        total_mem_usage : float
            Only quoted in the error for a single point that does not fit.

        Returns
        -------
        tuple[int, bool]
            The largest test call that ran (1 when n_first < 10), and whether
            a test call ran out of memory (the batch size is then converged).

        Raises
        ------
        torch.OutOfMemoryError
            If f does not fit on a single point.
        """
        batch, converged, n = 1, False, _GROWTH
        while n <= min(n_first, _LARGEST_TEST_CALL) and not converged:
            while (
                self._measure_call(
                    f, f_args, node, n, lambda: None, lambda: 0, total_mem_usage
                )
                is None
            ):
                n, converged = _shrink_batch(n), True
            batch, n = n, n * _GROWTH
        return batch, converged

    def _save_grads(
        self, loss: torch.Tensor, saved: _SavedGrads, listed: bool
    ) -> str | None:
        """
        Save the gradient of every tensor a backward pass of loss will write to.

        Walks the autograd graph of loss: every leaf tensor that receives a
        gradient is the ``variable`` of an accumulate-gradient node. A leaf not
        yet in ``saved`` is added with a copy of its ``.grad`` (or None), so
        that after an out-of-memory error in the backward pass the gradients
        can be put back to what they were before the ``integrate`` call.

        Parameters
        ----------
        loss : torch.Tensor
            The loss about to be backpropagated.
        saved : _SavedGrads
            Saved gradients by ``id`` of the tensor (``_save_grad``); updated
            in place.
        listed : bool
            Whether the user gave ``parameters=`` (already in ``saved``).

        Returns
        -------
        str or None
            Why the saved gradients may not undo the backward pass, or None:
            a custom autograd function can use tensors the walk does not reach
            (unless ``parameters=`` lists them), and a hook run after a
            gradient is added cannot be undone.
        """
        reason = None
        seen, stack = set(), [loss.grad_fn]
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            leaf = getattr(node, "variable", None)
            if leaf is not None:
                _save_grad(leaf, saved)
                if getattr(leaf, "_post_accumulate_grad_hooks", None):
                    reason = "a parameter has a post-accumulate-grad hook"
            elif not listed and isinstance(
                node, torch.autograd.function.BackwardCFunction
            ):
                reason = (
                    "the integrand uses a custom autograd function, which can "
                    "hide parameters; pass parameters= to integrate()"
                )
            stack.extend(n for n, _ in node.next_functions)
        return reason

    def _loss_and_backward(
        self,
        results: IntegrationResult,
        loss_fxn: Callable[[IntegrationResult], torch.Tensor],
        take_gradient: bool,
        saved: _SavedGrads,
        listed: bool,
    ) -> tuple[torch.Tensor | None, str | None]:
        """
        Compute one batch's loss and, with take_gradient, its backward pass.

        Before the backward pass, the gradients it writes to are saved
        (``_save_grads``), so that an out-of-memory error can be undone. On a
        GPU its peak memory is measured: ``_backward_ratio``, the backward
        peak over the forward peak, is what ``_evaluate_f_on_full_nodes``
        checks the next batch against.

        Parameters
        ----------
        results : IntegrationResult
            The batch's accepted subintervals.
        loss_fxn : Callable
            Maps ``results`` to a scalar loss.
        take_gradient : bool
            Whether to run the backward pass.
        saved : _SavedGrads
            Saved gradients by ``id`` of the tensor; updated in place.
        listed : bool
            Whether the user gave ``parameters=`` (already in ``saved``).

        Returns
        -------
        loss : torch.Tensor or None
            The loss, or None when the loss or its backward pass ran out of
            memory (only with ``take_gradient``; the integral must restart).
        unsafe : str or None
            Why the saved gradients may not undo this backward pass, or None
            (``_save_grads``).

        Raises
        ------
        RuntimeError or MemoryError
            Any error that is not out of memory, and every error when
            ``take_gradient`` is False.
        """
        unsafe = None
        try:
            loss = loss_fxn(results)
            if take_gradient and loss.requires_grad:
                unsafe = self._save_grads(loss, saved, listed)
                cuda = self.device.type == "cuda"
                if cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                    before = torch.cuda.memory_allocated(self.device)
                loss.backward()
                if cuda and self._forward_bytes > 0:
                    self._backward_ratio = (
                        torch.cuda.max_memory_allocated(self.device) - before
                    ) / self._forward_bytes
        except (RuntimeError, MemoryError) as e:
            if not (take_gradient and _is_out_of_memory(e)):
                raise
            return None, unsafe
        return loss, unsafe

    def _raise_after_backward_oom(
        self,
        failed_batch: int,
        start_mesh: torch.Tensor,
        saved: _SavedGrads,
        unsafe: str | None,
    ) -> NoReturn:
        """
        Restart the integral after a backward pass ran out of memory, or raise.

        Puts back the gradients the call added and raises ``_Restart`` with
        the call's starting mesh and a smaller, converged batch size, which
        ``_restart_after_backward_oom`` turns into a new call (after this
        call's tensors are freed). Raises ``BackwardOutOfMemoryError`` instead
        when not every gradient can be put back (``unsafe``, or
        ``torch.distributed`` runs more than one process), or when the
        smaller batch cannot hold one subinterval.

        Parameters
        ----------
        failed_batch : int
            Number of points of the batch whose backward pass failed.
        start_mesh : torch.Tensor
            The call's starting mesh. Shape: [M, T].
        saved : _SavedGrads
            The gradients from before the call.
        unsafe : str or None
            Why the saved gradients may not undo the backward passes, or None.

        Raises
        ------
        _Restart
            When the gradients were put back; its argument is the new state.
        BackwardOutOfMemoryError
            When the integral cannot restart; its ``state`` is the state to
            call again with, after resetting the gradients.
        """
        restart = IntegrationState(start_mesh, _shrink_batch(failed_batch), True)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            unsafe = "torch.distributed runs more than one process"
        if unsafe is None:
            for ref, grad in list(saved.values()):
                if (leaf := ref()) is not None:
                    leaf.grad = grad
        if unsafe is not None or restart.max_batch < self.C:
            error = BackwardOutOfMemoryError(
                f"The backward pass of a batch of {failed_batch} points ran out "
                "of memory, and padaquad cannot restart the integral ("
                f"{unsafe or 'a smaller batch cannot hold one subinterval'}). "
                + (
                    "The gradients now hold part of the integral's gradient: "
                    "reset them (for example optimizer.zero_grad()) and call "
                    "again with "
                    if unsafe is not None
                    else "The gradients were put back to their values before "
                    "the call. Call again with "
                )
                + f"state=error.state (batch size {restart.max_batch}).",
            )
            error.state = restart
            raise error
        logger.warning(
            f"The backward pass of a batch of {failed_batch} points ran out of "
            "memory. The gradients it added were removed, and the integral is "
            "restarted from its starting mesh with batches of at most "
            f"{restart.max_batch} points."
        )
        raise _Restart(restart)
