"""
Memory management for the adaptive quadrature solvers.

The integrand f is evaluated in batches; how many evaluations fit in one batch
depends on the memory one evaluation uses. This module holds:

- ``_MemoryManagement``: a mixin inherited by ``AdaptiveQuadrature``
  (``base.py``). It measures c, the memory cost of one evaluation of f
  (``_setup_memory_checks``), queries free memory on the solver's device, and
  turns both into the batch size (``get_max_batch`` -> ``_get_max_f_evals`` =
  usable memory / c), recomputed at every evaluation from live memory.
- ``_is_out_of_memory``: recognizes every form an out-of-memory error takes
  (GPU, CPU allocator, CUDA libraries, numpy), used by the batch loop in
  ``base.py`` and by the benchmark.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

_GB = 1024**3  # bytes per GB; every memory quantity below is in GB

# Multiplies the measured memory cost of one f evaluation when
# take_gradient=True: backpropagation uses roughly twice the forward memory.
# A heuristic, not a measurement.
_GRADIENT_MEMORY_SAFETY_FACTOR = 2.1


def _is_out_of_memory(e: BaseException) -> bool:
    """
    True if e signals that an allocation did not fit in memory.

    Covers a GPU out-of-memory error, a refused PyTorch CPU allocation (a plain
    RuntimeError naming DefaultCPUAllocator), a failed cudaMalloc or
    cudaHostAlloc outside PyTorch's allocator ("CUDA error: out of memory"), a
    CUDA library's allocation status (CUBLAS_STATUS_ALLOC_FAILED and the
    cuSOLVER, cuSPARSE and cuFFT equivalents; cuDNN 9's
    CUDNN_STATUS_INTERNAL_ERROR_*_ALLOCATION_FAILED), and Python's MemoryError
    (raised by numpy). The strings were checked in torch 2.8's libraries.
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
    ``DistributedEnvironment``), and ``f_unit_mem_size``, ``_oom_max_batch``,
    ``init_max_batch`` and ``init_total_mem_usage`` (set in
    ``AdaptiveQuadrature.__init__``).
    """

    device: torch.device
    f_unit_mem_size: float | None
    _oom_max_batch: int | None
    init_max_batch: int | None
    init_total_mem_usage: float

    def get_max_batch(
        self,
        total_mem_usage: float | None = None,
        default_max_batch: int | None = None,
    ) -> int | None:
        """Resolve the per-batch evaluation cap without re-benchmarking memory.

        Precedence: an explicit ``default_max_batch`` (e.g. the per-call
        ``max_batch``), then the construction-time ``init_max_batch``, then the
        dynamically computed budget from a prior memory benchmark
        (``f_unit_mem_size``) and the memory free now, using
        ``total_mem_usage`` (default: the solver's own). The result is clamped
        by any ``_oom_max_batch`` discovered after an out-of-memory event; the
        cap is cleared only at the start of the next ``integrate()`` call, so
        it is still applied here when a call returns.

        Tests and example loops use this to capture ``max_batch`` after a first
        run (where the memory benchmark ran once) and pass it back to later
        solvers/calls, or use ``integrate(..., same_f=True)``, so the
        slow benchmark is skipped. See also ``previous_max_batch``.
        """
        if total_mem_usage is None:
            total_mem_usage = self.init_total_mem_usage
        if default_max_batch is not None:
            max_batch = default_max_batch
        elif self.init_max_batch is not None:
            max_batch = self.init_max_batch
        elif self.f_unit_mem_size is not None:
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
        Measure c, the memory cost of one evaluation of f (``f_unit_mem_size``, GB).

        This only measures c; it does not choose a batch size. The batch size
        is computed at every evaluation from the memory free at that moment:
        ``get_max_batch`` -> ``_get_max_f_evals`` = usable memory / c.

        f is called on N = 1, 10, 100, ... copies of ``node_test``. For each
        call, M is the memory the call used (``_measure_call``, with the method
        chosen by ``_peak_memory_reader``) and c = s * (M + E) / N. s is
        ``_GRADIENT_MEMORY_SAFETY_FACTOR`` (a heuristic for backpropagation)
        when ``take_gradient`` and 1 otherwise; E is the most the memory reading
        can be too low (nonzero only on Linux CPUs).

        The search stops, keeping the latest c, when:

        1. c changes by less than 1 % from N/10 to N;
        2. a call takes longer than 1 s;
        3. N reaches 1e9;
        4. the next call is predicted not to fit (10 * M > usable memory).

        After an out-of-memory error at N, the c from the previous N is kept if
        that N was >= 100; otherwise c = max(previous c, s * F / (0.75 * N)) with
        F the free memory before the call, because the peak counters undercount
        a failed call. Finally c is floored at usable / 1e9, so the batch size
        never exceeds 1e9 points.

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
            The measurement failed for another reason; pass
            ``max_batch`` to skip it.
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
        self.f_unit_mem_size = None

        # The test calls run f forward only. With take_gradient=True the
        # integration also backpropagates through every batch, which needs
        # roughly as much memory again, so the measured cost is scaled by s.
        s = _GRADIENT_MEMORY_SAFETY_FACTOR if take_gradient else 1.0
        time_limit = 1.0  # seconds; a slower test call ends the search
        max_test = 1_000_000_000  # largest N tried, and the batch-size cap
        # How the memory of one call is read on this device, and E (GB), the
        # most that reading can be too low.
        reset_peak, read_memory, E = self._peak_memory_reader()

        # Search state. c is the latest cost per evaluation (GB). c_prev and
        # N_prev belong to the previous call, which had 10x fewer points
        # (N_prev = 0 before the first call). The stop that ends the search
        # sets reason, which the end message reports.
        N, c, c_prev, N_prev, reason = 1, 0.0, 0.0, 0, None
        try:
            while reason is None:
                # Memory available right before this call: usable (after the
                # total_mem_usage reserve) is what the next call must fit in;
                # free (raw) bounds the cost if this call runs out of memory.
                usable = self._get_usable_memory(total_mem_usage)
                free = self._get_memory()[0]
                measured = self._measure_call(
                    f, f_args, node_test, N, reset_peak, read_memory, total_mem_usage
                )

                if measured is None:
                    # N points did not fit. If the previous call had at least
                    # 100 points, its c is a trustworthy measurement and is kept
                    # (c still holds it). Otherwise also use what the failure
                    # shows: N points needed more than the free memory F, so one
                    # point costs more than F / N; 0.75 * N makes that bound
                    # stricter. The peak counters cannot be used here, because
                    # they count only the allocations that succeeded.
                    if N_prev < 100:
                        c = max(c_prev, s * free / (0.75 * N))
                    reason = "an evaluation ran out of memory"
                    continue

                # Cost per evaluation: the call's memory, plus the reading's
                # possible shortfall E, spread over its N points. A fixed
                # per-call cost (allocator rounding, workspaces) shrinks as 1/N.
                M, seconds = measured
                c = s * (M + E) / N

                # Stops, checked in order; each keeps this call's c.
                if N_prev and abs(c - c_prev) < 0.01 * c_prev:
                    # 10x more points changed c by < 1 %: the fixed per-call
                    # cost is spread out, and c is the per-point cost.
                    reason = "the cost per evaluation stopped changing"
                elif seconds > time_limit:
                    # The next call has 10x the points, so it would take ~10x
                    # longer; stop to keep the measurement short.
                    reason = f"an evaluation took longer than {time_limit} s"
                elif max_test <= N:
                    reason = "the 1e9 search limit was reached"
                elif usable < 10 * M:
                    # The next call would need about 10 * M. Do not make it: on
                    # a GPU it would run out of memory, and on a CPU the
                    # operating system could kill the process without an error.
                    reason = "the next test would not fit in memory"
                else:
                    c_prev, N_prev, N = c, N, 10 * N
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

        # Floor c at usable / 1e9, so the batch size usable / c never exceeds
        # 1e9 points, the largest N the search could try. This matters for tiny
        # integrands, which use only a few bytes per point.
        usable = self._get_usable_memory(total_mem_usage)
        c = max(c, usable / max_test)
        self.f_unit_mem_size = c

        # The estimate depends on take_gradient: c includes the factor s only
        # when it is True.
        max_batch = self._get_max_f_evals(total_mem_usage)
        logger.warning(
            f"Memory measurement finished in {time.time() - t0:.2f} s (stopped "
            f"because {reason}): {c * _GB:.0f} bytes per evaluation. With the "
            f"{usable:.1f} GB usable now, max_batch is about {max_batch} for "
            f"take_gradient={take_gradient}; pass max_batch={max_batch} to skip "
            "this measurement in later runs."
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
            reset: Starts a new peak measurement (a no-op for the last method).
            read: Memory in bytes, as described above.
            E: GB to add to every M, the most the reading can be too low.
        """
        if self.device.type == "cuda":

            def reset_cuda() -> None:
                torch.cuda.reset_peak_memory_stats(self.device)

            def read_cuda() -> int:
                return torch.cuda.max_memory_allocated(self.device)

            return reset_cuda, read_cuda, 0.0

        clear_refs = Path("/proc/self/clear_refs")
        try:
            # Probe: succeeds only on Linux with a writable /proc.
            clear_refs.write_text("5")
        except OSError:

            def no_reset() -> None:
                pass

            def read_rss() -> int:
                return psutil.Process().memory_info().rss

            return no_reset, read_rss, 0.0

        def reset_linux() -> None:
            clear_refs.write_text("5")

        def read_vmhwm() -> int:
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

        Divides usable memory by the per-evaluation memory cost (measured by
        _setup_memory_checks). A small epsilon (1e-12) prevents division by zero.

        Parameters
        ----------
        total_mem_usage : float
            Fraction of total memory allowed.

        Returns
        -------
        int
            Maximum number of evaluations (integer).
        """
        if self.f_unit_mem_size is None:
            raise RuntimeError(
                "The per-evaluation memory cost is unknown; run the memory "
                "benchmark (_setup_memory_checks) before sizing a batch."
            )
        usable = self._get_usable_memory(total_mem_usage)
        return int(usable // (1e-12 + self.f_unit_mem_size))
