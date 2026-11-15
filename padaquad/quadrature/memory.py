"""
Memory management for the adaptive quadrature solvers.

The integrand f is evaluated in batches; how many evaluations fit in one batch
depends on the memory one evaluation uses. This module holds:

- ``_MemoryManagement``: a mixin inherited by ``AdaptiveQuadrature``
  (``base.py``). It measures how the memory of one call of f grows with the
  number of points N, as a fit a + b N + d N^2 (``_setup_memory_checks``),
  queries free memory on the solver's device, and turns both into the batch
  size (``get_max_batch`` -> ``_get_max_f_evals``: the largest N whose fitted
  memory fits in ``_MEMORY_BUDGET`` of the usable memory), recomputed at every
  evaluation from live memory.
- ``_fit_up_to_quadratic`` and ``_largest_n``: the fit, and its inverse.
- ``_is_out_of_memory``: recognizes the out-of-memory errors of the GPU, the
  PyTorch CPU allocator, the CUDA libraries and numpy, used by the batch loop
  in ``base.py`` and by the benchmark.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
import torch

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
# (in the benchmark below and in the batch loop of base.py).
_OOM_BACKOFF = 0.75

# Largest number of points in one call: the benchmark's search limit and the
# cap on the batch size.
_MAX_N = 1_000_000_000


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
        quantity does not grow with n (b = d = 0).
    """
    a, b, d = coeffs
    R = budget - a
    if R <= 0:
        return 0
    if b <= 0 and d <= 0:
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
        True if ``e`` is one of the out-of-memory errors listed above.
    """
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
    ``DistributedEnvironment``), and ``f_mem_coeffs``, ``_oom_max_batch``,
    ``init_max_batch`` and ``init_total_mem_usage`` (set in
    ``AdaptiveQuadrature.__init__``).
    """

    device: torch.device
    f_mem_coeffs: tuple[float, float, float] | None
    _oom_max_batch: int | None
    init_max_batch: int | None
    init_total_mem_usage: float

    def get_max_batch(
        self,
        total_mem_usage: float | None = None,
        default_max_batch: int | None = None,
    ) -> int | None:
        """
        Return the largest number of integrand evaluations per batch.

        Does not run the memory benchmark. The first available of these is
        used: ``default_max_batch``, the ``max_batch`` given at construction,
        and the number of evaluations that fit in the memory free now, with the
        memory per call the last benchmark measured. After an out-of-memory
        error, the result is capped at the reduced batch size chosen then,
        until the next ``integrate`` call starts.

        Passing the result as ``max_batch`` to a later call or solver skips
        the slow benchmark. See :ref:`ug-memory`.

        Parameters
        ----------
        total_mem_usage : float or None
            Fraction of the device's total memory that may be in use; the rest
            is kept free. None uses the value given at construction.
        default_max_batch : int or None
            A batch size that takes precedence over the other sources. None
            (default) skips it.

        Returns
        -------
        int or None
            The batch size, or None when no source is available (no
            ``max_batch`` given and no benchmark run yet).
        """
        if total_mem_usage is None:
            total_mem_usage = self.init_total_mem_usage
        if default_max_batch is not None:
            max_batch = default_max_batch
        elif self.init_max_batch is not None:
            max_batch = self.init_max_batch
        elif self.f_mem_coeffs is not None:
            max_batch = self._get_max_f_evals(total_mem_usage)
        else:
            max_batch = None
        if (
            self._oom_max_batch is not None
            and max_batch is not None
            and max_batch > self._oom_max_batch
        ):
            max_batch = self._oom_max_batch
        return max_batch

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

    def _release_memory(self) -> None:
        """
        Free memory after an out-of-memory error or a memory benchmark.

        Always runs the garbage collector. On a CUDA solver it also empties
        PyTorch's caching allocator, which releases the cache of every GPU
        (NativeCachingAllocator::emptyCache in c10/cuda/CUDACachingAllocator.cpp
        loops over all devices), so no device context is needed. A CPU solver
        never touches the GPU.
        """
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            # Small grace period for any async CUDA cleanup to land. Avoiding
            # `torch.cuda.synchronize()` here on purpose: after a CUDA OOM the
            # default stream can be in an error state, and synchronize on a
            # faulted stream has been observed to block indefinitely.
            time.sleep(0.05)

    def _setup_memory_checks(
        self,
        f: Callable[..., Any],
        node_test: torch.Tensor,
        take_gradient: bool,
        total_mem_usage: float,
        f_args: tuple[Any, ...] = (),
    ) -> None:
        """
        Fit the memory of one call of f on N points (``f_mem_coeffs``, GB).

        This only measures; it does not choose a batch size. The batch size is
        computed at every evaluation from the memory free at that moment:
        ``get_max_batch`` -> ``_get_max_f_evals``.

        f is called on N copies of ``node_test``, starting at N = 1. For each
        call, M is the memory the call used (``_measure_call``, with the method
        chosen by ``_peak_memory_reader``) and t its wall time. After each call,
        M(N) and t(N) are fitted as a + b N + d N^2 with a, b, d >= 0
        (``_fit_up_to_quadratic``); before three readings exist, the worst case
        such a fit allows through the latest reading is used instead,
        y(k N) <= k^2 y(N) for k >= 1. The next call has

            N_next = min(10 N, n_mem, n_time),

        where n_mem is the largest N whose predicted memory plus E fits in
        ``_MEMORY_BUDGET`` of the usable memory and n_time the largest N whose
        predicted time is at most 1 s (``_largest_n``). So an f whose memory or
        time grows like N^2 gets smaller steps instead of a call that cannot
        fit. E is the most the memory reading can be too low (nonzero only on
        Linux CPUs); it is a reading error, so it enters as a fixed amount, not
        one that grows with N.

        The search stops when:

        Before three readings the time bound always allows doubling N: one
        point of a slow f says nothing about how its time grows, and the worst
        case k^2 would end the search after one call for any f slower than
        0.25 s. Such a doubled call takes at most 4 times the last one, and the
        1 s stop below then ends the search.

        The search stops when:

        1. c = s * (M + E) / N changes by less than 1 % over each of the last
           two calls (the cost per point is constant, so the growth is linear;
           two changes, so noise in one reading does not end the search). An
           N^2 term too small to show in the readings, or hidden by their
           noise, is missed;
        2. a call takes longer than 1 s (a backstop: the step size normally
           prevents it);
        3. N reaches 1e9;
        4. N_next < 2 N: the next call would not fit in memory, or would take
           longer than 1 s, and a call barely larger than this one adds little.

        It raises instead when the memory grows faster than N^2: when a call
        uses more than 3 times the memory that growth like N^2 allows from the
        previous call, M > 3 (N / N_prev)^2 (M_prev + E) (the factor 3 absorbs
        readings off by up to a factor 2), counted only when the
        previous reading is at least 1 MiB (smaller readings are mostly noise:
        they read 0 when f reuses memory the process holds). Such
        growth is caught only once it shows in two readings, so a call that
        first shows it can already exceed the memory budget, and growth that
        a large fixed cost hides, or growth only slightly faster than N^2, is
        not caught.

        The stored fit is m(B) = s * ((a + E) + b B + d B^2), the memory of a
        batch of B points, with a, b, d fitted to every reading. s is
        ``_GRADIENT_MEMORY_SAFETY_FACTOR`` (a heuristic for backpropagation)
        when ``take_gradient`` and 1 otherwise. With fewer than three readings
        the fit is linear, so the memory of a batch can be underestimated when
        the search stopped after one or two calls: for an f whose memory grows
        like N^2 and is heavy, or an f whose first call took longer than 1 s.
        A single reading of 0 gives the largest batch, 1e9 points.

        After an out-of-memory error at N > 1, the batch size is capped at
        ``_OOM_BACKOFF`` * N for the rest of this ``integrate`` call
        (``_oom_max_batch``, as the batch loop does after such an error), and
        the fit from the calls that succeeded is kept.

        Only one input value is used for every point, so an f whose memory
        depends on t can be mismeasured. The start and end messages are logged
        at WARNING level, so they are silent if the application configures
        logging above WARNING.

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

        Raises
        ------
        torch.OutOfMemoryError
            A single evaluation of f does not fit.
        RuntimeError
            The memory grows faster than N^2, or the measurement failed for
            another reason; pass ``max_batch`` to skip it.
        """
        logger.warning(
            "PadaQuad is measuring how much memory one integrand (f) evaluation "
            "uses, to determine max_batch (the maximum number of evaluations per "
            "batch). This is needed because adaptive refinement changes the "
            "evaluation batch at each step during the integral. This measurement "
            "is slow and runs on every integrate() call that has no max_batch. To "
            "skip it, pass max_batch (your own value, or solver.get_max_batch() "
            "after an earlier run), or pass same_f=True to reuse both the "
            "max_batch and the mesh of the previous call."
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
        time_limit = 1.0  # seconds; no test call is planned to take longer
        # A call using more than this many times the memory that growth like
        # N^2 allows from the previous call raises (faster than N^2). Readings
        # of 1 MiB and more were off by up to a factor 2 (0.5x to 1.4x of the
        # true size on a Mac), so a previous reading at 0.5x and this one at
        # 1.5x make N^2 growth look 3 times faster; N^2.5 over a 10x step
        # (316x instead of 100x) is still caught.
        growth_tolerance = 3.0
        # How the memory of one call is read on this device, and E (GB), the
        # most that reading can be too low.
        reset_peak, read_memory, E = self._peak_memory_reader()

        # Search state: the points, memory (GB) and wall time (s) of every call
        # so far. c is the latest cost per point, s * (M + E) / N, and c_prev,
        # c_prev2 those of the two calls before; they serve only the
        # convergence stop. The stop that ends the search sets reason, which
        # the end message reports; too_fast holds the reason to raise instead.
        Ns: list[int] = []
        Ms: list[float] = []
        ts: list[float] = []
        N, c, c_prev, c_prev2 = 1, 0.0, 0.0, 0.0
        reason: str | None = None
        too_fast: str | None = None
        try:
            while reason is None:
                # Usable memory (after the total_mem_usage reserve) right before
                # this call: what the next call must fit in.
                usable = self._get_usable_memory(total_mem_usage)
                measured = self._measure_call(
                    f, f_args, node_test, N, reset_peak, read_memory, total_mem_usage
                )

                if measured is None:
                    # N > 1 points did not fit (at N = 1 _measure_call raises).
                    # The peak counters count only the allocations that
                    # succeeded, so the failed call is not a reading. Cap the
                    # batch below N for the rest of this integrate() call, as
                    # the batch loop does after an out-of-memory error; the fit
                    # from the calls that succeeded is kept.
                    self._oom_max_batch = max(1, int(_OOM_BACKOFF * N))
                    reason = "an evaluation ran out of memory"
                    continue

                M, seconds = measured
                Ns.append(N)
                Ms.append(M)
                ts.append(seconds)
                c_prev2, c_prev, c = c_prev, c, s * (M + E) / N

                # Faster than N^2: with a, b, d >= 0, M(N) <= (N / N_prev)^2
                # M(N_prev). A reading above growth_tolerance times that bound
                # shows growth the fit cannot follow.
                # The previous call is the one to compare with: an earlier one
                # can be dominated by a fixed cost that hides the growth.
                # Readings below 1 MiB are mostly noise (they read 0 when f
                # reuses memory the process holds) and are not compared.
                bound = (
                    (N / Ns[-2]) ** 2 * (Ms[-2] + E)
                    if len(Ns) >= 2 and Ms[-2] >= 2**20 / _GB
                    else 0.0
                )
                if bound > 0 and growth_tolerance * bound < M:
                    too_fast = (
                        f"The memory of one call of f grew to {M * _GB:.3g} "
                        f"bytes at {N} points, more than {growth_tolerance:g} times the "
                        f"{bound * _GB:.3g} bytes that growth like N^2 allows "
                        f"from the {Ms[-2] * _GB:.3g} bytes at {Ns[-2]} points. "
                        "padaquad fits the memory of f up to N^2, so it cannot "
                        "choose a batch size that fits for this integrand. "
                        "Specify max_batch (to integrate() or the solver): the "
                        "largest number of points per call of f that fits in "
                        "memory."
                    )
                    reason = "the memory grows faster than N^2"
                # Stops, checked in order.
                elif (
                    len(Ns) >= 3
                    and abs(c - c_prev) < 0.01 * c_prev
                    and abs(c_prev - c_prev2) < 0.01 * c_prev2
                ):
                    # The cost per point changed by < 1 % over each of the last
                    # two calls: the fixed per-call cost is spread out and the
                    # growth is linear. The three readings make the stored fit a
                    # full quadratic.
                    reason = "the cost per evaluation stopped changing"
                elif seconds > time_limit:
                    reason = f"an evaluation took longer than {time_limit} s"
                elif N >= _MAX_N:
                    reason = "the 1e9 search limit was reached"
                else:
                    # Size the next call from the readings: min(10 N, n_mem, n_time).
                    # Before three readings, use the worst case a fit
                    # a + b N + d N^2 with a, b, d >= 0 allows through this
                    # reading: y(k N) <= k^2 y(N), the coefficients (0, 0, y/N^2).
                    if len(Ns) >= 3:
                        a, b, d = _fit_up_to_quadratic(Ns, Ms)
                        time_fit = _fit_up_to_quadratic(Ns, ts)
                    else:
                        a, b, d = 0.0, 0.0, M / N**2
                        time_fit = (0.0, 0.0, seconds / N**2)
                    # The test calls run forward only, so s is not applied. A
                    # call that does not fit would run out of memory on a GPU,
                    # and on a CPU the operating system could kill the process
                    # without an error. E is a fixed reading error.
                    n_mem = _largest_n((a + E, b, d), _MEMORY_BUDGET * usable)
                    n_time = _largest_n(time_fit, time_limit)
                    if len(Ns) < 3:
                        # One or two times say nothing about how time grows, and
                        # the worst case would end the search after one call of
                        # any f slower than 0.25 s. Allow doubling: that call
                        # takes at most 4 times this one, and the 1 s stop ends
                        # the search after it.
                        n_time = max(n_time, 2 * N)
                    N_next = min(10 * N, n_mem, n_time)
                    if N_next < 2 * N:
                        # A call barely larger than this one adds little.
                        reason = (
                            "the next test would not fit in memory"
                            if n_mem <= n_time
                            else f"the next test would take longer than {time_limit} s"
                        )
                    else:
                        N = N_next
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

        self._release_memory()
        if too_fast is not None:
            # Raised here, outside the try block, so it is not reported as a
            # failed measurement.
            raise RuntimeError(too_fast)

        # The memory of one call of B points during the integration:
        # m(B) = s * ((a + E) + b B + d B^2), fitted to every
        # reading. E is added once: it is a reading error, not a cost per point.
        a, b, d = _fit_up_to_quadratic(Ns, Ms)
        self.f_mem_coeffs = (s * (a + E), s * b, s * d)

        # The estimate depends on take_gradient: the fit includes the factor s
        # only when it is True. get_max_batch applies the cap an out-of-memory
        # error above set, so the message reports the batch actually used.
        usable = self._get_usable_memory(total_mem_usage)
        max_batch = self.get_max_batch(
            total_mem_usage, self._get_max_f_evals(total_mem_usage)
        )
        assert max_batch is not None
        a, b, d = self.f_mem_coeffs
        quadratic = ""
        # Name the N^2 growth only when it matters at the batch size: every
        # fit of a linear f leaves a tiny d > 0. Strict, so a batch of 0 with
        # no fixed cost (a = 0) is not flagged.
        if d * max_batch**2 > 0.1 * (a + b * max_batch + d * max_batch**2):
            quadratic = (
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
            f"pass max_batch={max_batch} to skip this measurement in later runs."
            f"{quadratic}"
        )

    def _peak_memory_reader(
        self,
    ) -> tuple[Callable[[], None], Callable[[], int], float]:
        """
        Choose how the memory one test call uses (M) is measured on this device.

        M = read() after the call minus read() before it, with reset() called
        right before the "before" reading. Three methods, best first:

        - CUDA: PyTorch's caching allocator tracks the peak of allocated memory
          per device. ``reset`` sets that peak to the memory allocated now, and
          ``read`` returns the peak. Exact to its 512-byte allocation blocks.
          Resetting also resets ``max_memory_reserved`` for ``self.device``.
          Memory allocated outside PyTorch's allocator (custom ``cudaMalloc``,
          CuPy, NCCL) is not counted.
        - CPU on Linux: writing 5 to ``/proc/self/clear_refs`` (Linux >= 4.0)
          resets the process's peak resident memory, ``VmHWM`` in
          ``/proc/self/status``, to its current value; ``read`` returns
          ``VmHWM``. The reset also resets what the caller's own ``getrusage``
          or ``/usr/bin/time`` report as the peak from then on. ``ru_maxrss``
          is not read: it also includes the peak of the process this one was
          started from, so in a worker started by a larger process
          (``subprocess``, multiprocessing ``spawn``) it would stay at the
          parent's peak. The kernel keeps each CPU core's share of the count
          private until it reaches max(32, 2 * cores) pages, so the reading can
          be low by up to E = cores * max(32, 2 * cores) pages
          (``lib/percpu_counter.c``, ``compute_batch_value``).
        - Other CPUs (macOS, Windows, a locked ``/proc``): no resettable peak
          exists. ``read`` returns the current resident memory, so M is the
          memory still held after the call (its result is kept alive for the
          reading). Memory that f frees before returning, such as the
          intermediates of a Jacobian, is missed, so c can be too low.

        Returns
        -------
        tuple[Callable[[], None], Callable[[], int], float]
            The functions ``reset`` (starts a new peak measurement; a no-op for
            the last method) and ``read`` (memory in bytes, as described above),
            and E, the GB to add to every M (the most the reading can be low).
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

            return reset_cuda, read_cuda, 0.0

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

            return no_reset, read_rss, 0.0

        def reset_linux() -> None:
            """Reset the process's peak resident memory, VmHWM, to its current value."""
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

        cores = os.cpu_count() or 1
        E = cores * max(32, 2 * cores) * os.sysconf("SC_PAGE_SIZE") / _GB
        return reset_linux, read_vmhwm, E

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
        if out_of_memory:
            # Clean up here, after the except block: inside it, the error's
            # traceback still holds f's tensors, so they could not be freed.
            t_input = result = None
            self._release_memory()
            return None

        if self.device.type == "cuda":
            # GPU calls return once their work is queued, not when it is done;
            # wait for it, so the time below is the real compute time.
            torch.cuda.synchronize(self.device)
        seconds = time.time() - t0
        # Read while the result is still alive, so a before/after difference
        # (the third method) counts the output too.
        M = max(read_memory() - before, 0) / _GB
        del result, t_input
        return M, seconds

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
