"""Tests for the memory benchmark, ``_setup_memory_checks``.

The benchmark measures one number, c (``f_unit_mem_size``, GB): the memory
cost of a single evaluation of ``f``. It calls ``f`` on N = 1, 10, 100, ...
points, measures M (the memory each call used), and sets c = s * (M + E) / N,
with s the gradient safety factor and E the Linux reading's error allowance.
The batch size is computed later from live free memory (``_get_max_f_evals``)
and is not chosen here.

CPU tests replace the memory readings and the clock with fakes, so every
expected value is exact and independent of the machine:

- ``FakeMemory`` stands in for the process's resident memory. A fake integrand
  adds a known number of bytes per call, so M is known exactly.
- ``FakeClock`` stands in for the ``time`` module of padaquad's quadrature base,
  so the "too slow" stop fires when the test says, without sleeping.

GPU tests use the real CUDA peak counters with buffers of known size.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import types

import pytest
import torch
from _helpers import T_FINAL, T_INIT, make_solver_for_unit_test, needs_cuda

import padaquad.quadrature.memory as qmem
from padaquad import adaptive_quadrature, steps

GB = 1024**3
USABLE_GB = 100.0  # fixed usable memory for the CPU tests
NODE = torch.zeros(1, dtype=torch.float64)


class FakeMemory:
    """Resident memory in bytes; fake integrands add to it on every call."""

    def __init__(self):
        self.rss = 10 * GB


class FakeClock:
    """Replaces ``qmem.time``: ``time()`` advances only when an integrand says."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CostIntegrand:
    """``f(t) = 1`` that "uses" ``fixed + per_point * N`` bytes per call.

    Records every batch size in ``calls``. From ``slow_from`` points on, each
    call takes 2 s on the fake clock; from ``oom_from`` points on, it raises
    ``torch.OutOfMemoryError``; from ``bug_from`` points on, it raises
    ``ValueError`` (an error that is not about memory).
    """

    def __init__(
        self,
        memory,
        clock,
        fixed=0,
        per_point=1024,
        slow_from=None,
        oom_from=None,
        bug_from=None,
    ):
        self.memory, self.clock = memory, clock
        self.fixed, self.per_point = fixed, per_point
        self.slow_from, self.oom_from, self.bug_from = slow_from, oom_from, bug_from
        self.calls: list[int] = []

    def __call__(self, t, *args):
        n = t.shape[0]
        self.calls.append(n)
        if self.oom_from is not None and n >= self.oom_from:
            raise torch.OutOfMemoryError(f"simulated OOM at {n} points")
        if self.bug_from is not None and n >= self.bug_from:
            raise ValueError("f only handles one point")
        if self.slow_from is not None and n >= self.slow_from:
            self.clock.now += 2.0
        self.memory.rss += self.fixed + self.per_point * n
        return torch.ones((n, 1), dtype=t.dtype, device=t.device)


@pytest.fixture
def fake(monkeypatch):
    """CPU solver with fake memory readings, a fake clock, and no Linux reader.

    ``clear_refs`` is made unwritable, so the benchmark uses the before/after
    difference of the (fake) resident memory and E = 0.
    """

    class NoClearRefs:
        def __init__(self, *args):
            pass

        def write_text(self, text):
            raise OSError("no /proc here")

    memory, clock = FakeMemory(), FakeClock()
    monkeypatch.setattr(qmem, "Path", NoClearRefs)
    monkeypatch.setattr(qmem, "time", clock)
    monkeypatch.setattr(
        qmem.psutil,
        "Process",
        lambda: types.SimpleNamespace(
            memory_info=lambda: types.SimpleNamespace(rss=memory.rss)
        ),
    )
    solver = make_solver_for_unit_test()
    monkeypatch.setattr(solver, "_get_usable_memory", lambda total_mem_usage: USABLE_GB)
    monkeypatch.setattr(solver, "_get_memory", lambda: (USABLE_GB, 2 * USABLE_GB))
    return types.SimpleNamespace(solver=solver, memory=memory, clock=clock)


def benchmark(solver, f, take_gradient=False):
    solver._setup_memory_checks(
        f, NODE, take_gradient=take_gradient, total_mem_usage=0.9
    )
    return solver.f_unit_mem_size


def safety(take_gradient):
    return qmem._GRADIENT_MEMORY_SAFETY_FACTOR if take_gradient else 1.0


# ---------------------------------------------------------------------------
# Measuring c and stopping the search (CPU, fake readings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", [False, True])
def test_cost_converges(fake, take_gradient):
    """c(N) = s * (a + b N) / N falls as the fixed cost a is amortized. Going
    from 1e5 to 1e6 points changes it by 0.9 % < 1 %, so the search stops at
    N = 1e6 with exactly that c."""
    a, b = 2**20, 1024
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b)
    c = benchmark(fake.solver, f, take_gradient)
    assert f.calls == [10**k for k in range(7)]
    assert c * GB == pytest.approx(safety(take_gradient) * (a + b * 10**6) / 10**6)


def test_slow_call_stops_search(fake):
    """A call slower than the 1 s limit ends the search at that N. The
    fixed cost keeps c changing by > 1 %, so only the time stop can fire."""
    a, b = 2**20, 4096
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b, slow_from=100)
    c = benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100]
    assert c * GB == pytest.approx((a + b * 100) / 100)


def test_next_call_predicted_not_to_fit(fake):
    """The search stops before a call whose predicted memory, 10 * M, exceeds
    usable memory, so f is never called with that many points."""
    # M(1) = 3 GB (10 * M fits); M(10) = 21 GB, so 10 * M = 210 GB > 100 GB.
    a, b = GB, 2 * GB
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b)
    c = benchmark(fake.solver, f)
    assert f.calls == [1, 10]
    assert c * GB == pytest.approx((a + b * 10) / 10)


def test_tiny_integrand_capped_at_search_limit(fake):
    """For an f that uses almost nothing per point, the batch size is capped
    at the 1e9 search limit: c is floored at usable / 1e9."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=1)
    c = benchmark(fake.solver, f)
    assert c == pytest.approx(USABLE_GB / 10**9)
    # _get_max_f_evals divides by (1e-12 + c), which shifts the result by 1e-5.
    assert fake.solver._get_max_f_evals(0.9) == pytest.approx(10**9, rel=1e-4)


# ---------------------------------------------------------------------------
# Out-of-memory errors during the measurement
# ---------------------------------------------------------------------------


def test_oom_after_n_100_keeps_measured_cost(fake):
    """An out-of-memory error at N = 1000 after a successful N = 100 keeps
    the c measured at N = 100 and ignores the failed call."""
    f = CostIntegrand(fake.memory, fake.clock, fixed=2**20, oom_from=1000)
    c = benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 1000]
    assert c * GB == pytest.approx((2**20 + 1024 * 100) / 100)


def test_oom_at_small_n_uses_free_memory(fake):
    """An out-of-memory error at N = 10 (previous N < 100) gives
    c = max(previous c, s * F / (0.75 * N)), F = free memory before the call:
    the peak counters undercount a failed call, so free memory is the bound."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=1024, oom_from=10)
    c = benchmark(fake.solver, f, take_gradient=True)
    s = safety(True)
    free_gb = USABLE_GB  # fake _get_memory reports (USABLE_GB, ...)
    assert c == pytest.approx(max(s * 1024 / GB, s * free_gb / (0.75 * 10)))


def test_single_evaluation_does_not_fit(fake):
    """An out-of-memory error on one point raises a clear
    ``torch.OutOfMemoryError``, not the "could not measure" ``RuntimeError``."""
    f = CostIntegrand(fake.memory, fake.clock, oom_from=1)
    with pytest.raises(torch.OutOfMemoryError, match="single evaluation of f"):
        benchmark(fake.solver, f)


def test_other_failure_stops_with_clear_message(fake):
    """An error that is not about memory stops the run with a message telling
    the user to pass max_batch, and keeps the original error as the cause."""
    f = CostIntegrand(fake.memory, fake.clock, bug_from=10)
    with pytest.raises(
        RuntimeError, match=r"could not measure.*Specify max_batch"
    ) as info:
        benchmark(fake.solver, f)
    assert isinstance(info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# The Linux peak reader and its error allowance E
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "linux", reason="Linux reader only")
@pytest.mark.parametrize("cores", [12, 256])
def test_linux_reader_adds_error_allowance(fake, monkeypatch, cores):
    """With the Linux reader, M comes from the (reset) peak VmHWM in
    /proc/self/status and E = cores * max(32, 2 * cores) pages is added:
    1.5 MiB on 12 cores and 512 MiB on 256 cores with 4 KiB pages. The call at
    N = 1 is slow, so c = M(1) + E exactly.

    Regression: ``getrusage`` reports a stuck 50 GB peak here, as it does in a
    worker started by a larger process (its ``ru_maxrss`` keeps the parent's
    peak). The old reader used ``ru_maxrss`` and measured M = 0."""

    class FakeProc:
        """/proc/self: clear_refs accepts writes; status reports the fake peak."""

        def __init__(self, path):
            self.path = path

        def write_text(self, text):
            pass

        def read_text(self):
            assert self.path == "/proc/self/status"
            return f"VmRSS:\t1 kB\nVmHWM:\t{fake.memory.rss // 1024} kB\n"

    resource = pytest.importorskip("resource")
    monkeypatch.setattr(
        resource, "getrusage", lambda who: types.SimpleNamespace(ru_maxrss=50 * 2**20)
    )
    monkeypatch.setattr(qmem, "Path", FakeProc)
    monkeypatch.setattr(qmem.os, "cpu_count", lambda: cores)
    f = CostIntegrand(fake.memory, fake.clock, per_point=4096, slow_from=1)
    c = benchmark(fake.solver, f)
    E = cores * max(32, 2 * cores) * os.sysconf("SC_PAGE_SIZE")
    assert f.calls == [1]
    assert c * GB == pytest.approx(4096 + E)


@pytest.mark.skipif(
    not os.access("/proc/self/clear_refs", os.W_OK), reason="needs Linux clear_refs"
)
def test_real_linux_reader_smoke(monkeypatch):
    """With the real reader, a 200 MiB allocation that f frees before returning
    is measured. E is made negligible (one core), so c is the reading itself.
    The allocation is a single-threaded memset (bytearray): its pages are
    counted on one or a few CPUs, so the per-CPU counter error is a few MiB;
    a multithreaded fill such as torch.ones can read low by most of it."""
    monkeypatch.setattr(qmem.os, "cpu_count", lambda: 1)
    clock = FakeClock()
    monkeypatch.setattr(qmem, "time", clock)
    solver = make_solver_for_unit_test()

    def heavy(t, *args):
        bytearray(200 * 2**20)  # 200 MiB, touched, freed before returning
        clock.now += 2.0  # slower than the 1 s limit: one call only
        return torch.ones((t.shape[0], 1), dtype=t.dtype)

    solver._setup_memory_checks(heavy, NODE, take_gradient=False, total_mem_usage=0.9)
    assert solver.f_unit_mem_size * GB >= 150 * 2**20


@pytest.mark.skipif(
    not os.access("/proc/self/clear_refs", os.W_OK), reason="needs Linux clear_refs"
)
def test_real_linux_reader_in_child_of_larger_process():
    """Regression: the reader works in a process started by a larger one.
    ``ru_maxrss`` of such a child starts at the parent's peak, so the old
    reader saw no memory use. This process raises its peak by 768 MiB, then
    runs ``test_real_linux_reader_smoke`` in a child process."""
    bytearray(768 * 2**20)  # raises this process's peak; freed at once
    test = f"{__file__}::test_real_linux_reader_smoke"
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode == 0, child.stdout + child.stderr


# ---------------------------------------------------------------------------
# Through integrate(): messages and the gradient path
# ---------------------------------------------------------------------------


def test_messages_when_measuring(caplog, monkeypatch):
    """Without max_batch the start warning explains the measurement and tells
    the user to pass max_batch; the end warning reports time, c, max_batch and
    the reason. With max_batch, neither appears."""
    clock = FakeClock()
    monkeypatch.setattr(qmem, "time", clock)

    def slow_sin(t, *args):
        clock.now += 2.0  # stops the search after one call; no real waiting
        return torch.sin(t)

    solver = make_solver_for_unit_test(method_name="gk21")
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        solver.integrate(slow_sin, mesh_init=T_INIT, mesh_final=T_FINAL)
    text = caplog.text
    assert "PadaQuad is measuring how much memory" in text
    assert "Pass max_batch to skip this" in text
    assert "Memory measurement finished in" in text
    assert "bytes per evaluation" in text
    assert "stopped because" in text

    caplog.clear()
    solver = make_solver_for_unit_test(method_name="gk21", max_batch=1000)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        solver.integrate(slow_sin, mesh_init=T_INIT, mesh_final=T_FINAL)
    assert "PadaQuad is measuring" not in caplog.text
    assert "Memory measurement finished" not in caplog.text


def test_slow_integrand_gradient_path_completes(monkeypatch):
    """Regression: a slow f used to stop the search at small N, and an
    estimate tied to N gave gk21 (23 points per panel) a batch below one panel
    on the take_gradient=True path, which then failed its assertion. c is now
    measured, so the batch is usable / c and integrate() completes."""
    clock = FakeClock()
    monkeypatch.setattr(qmem, "time", clock)

    def slow(t, *args):
        clock.now += 2.0  # every call is slower than the 1 s limit
        return torch.cos(t)

    solver = make_solver_for_unit_test(method_name="gk21")
    result = solver.integrate(
        slow, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=True
    )
    assert solver.get_max_batch() >= solver.C
    assert float(result.integral.sum()) == pytest.approx(
        torch.sin(T_FINAL).item(), abs=1e-5
    )


def test_release_memory_never_touches_gpu_on_cpu_solver(monkeypatch):
    """A CPU solver must not empty the CUDA cache, even when a GPU is visible."""

    def forbidden():
        raise AssertionError("CPU solver emptied the CUDA cache")

    monkeypatch.setattr(qmem.torch.cuda, "empty_cache", forbidden)
    make_solver_for_unit_test()._release_memory()


# ---------------------------------------------------------------------------
# GPU: the real CUDA peak counters
# ---------------------------------------------------------------------------


def gpu_solver(monkeypatch, usable_gb=10.0):
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="gk21",
        device="cuda:0",
        dtype=torch.float64,
    )
    monkeypatch.setattr(solver, "_get_usable_memory", lambda total_mem_usage: usable_gb)
    return solver


@needs_cuda
def test_gpu_cost_includes_internal_buffers(monkeypatch):
    """f allocates a fixed 1 MiB plus 1 KiB per point and frees both before
    returning. The peak counter sees them, so c is about s * (1 KiB + the input
    and output tensors) per point, and the fixed cost is amortized away."""
    solver = gpu_solver(monkeypatch)
    per_point = 1024

    def f(t, *args):
        fixed = torch.empty(2**20, dtype=torch.uint8, device=t.device)
        scratch = torch.empty(
            per_point * t.shape[0], dtype=torch.uint8, device=t.device
        )
        del fixed, scratch
        return torch.ones((t.shape[0], 1), dtype=t.dtype, device=t.device)

    node = torch.zeros(1, dtype=torch.float64, device="cuda:0")
    solver._setup_memory_checks(f, node, take_gradient=False, total_mem_usage=0.9)
    c_bytes = solver.f_unit_mem_size * GB
    # 1 KiB scratch + 8 B input per point (the output reuses the freed scratch
    # block), plus a residual share of the fixed 1 MiB (< 1 % once converged).
    assert per_point + 8 <= c_bytes <= 1.02 * (per_point + 16)


@needs_cuda
def test_gpu_cost_includes_gradient_inside_f(monkeypatch):
    """popcornn-like: f differentiates a network inside itself with
    take_gradient=False. The hidden activations exist only during the call;
    the peak counter still counts them, so c exceeds one hidden layer."""
    solver = gpu_solver(monkeypatch)
    width = 512
    net = torch.nn.Sequential(
        torch.nn.Linear(1, width), torch.nn.SiLU(), torch.nn.Linear(width, 1)
    ).to("cuda:0", torch.float64)

    def f(t, *args):
        with torch.enable_grad():
            x = t.clone().requires_grad_(True)
            (g,) = torch.autograd.grad(net(x).sum(), x)
        return g.detach()

    node = torch.zeros(1, dtype=torch.float64, device="cuda:0")
    solver._setup_memory_checks(f, node, take_gradient=False, total_mem_usage=0.9)
    assert solver.f_unit_mem_size * GB > width * 8


@needs_cuda
def test_gpu_oom_after_n_100_keeps_measured_cost(monkeypatch):
    """A real CUDA out-of-memory error (an instantly refused 100 TB request)
    at N = 1000 keeps the cost measured at N = 100."""
    solver = gpu_solver(monkeypatch)
    calls = []

    def f(t, *args):
        calls.append(t.shape[0])
        if t.shape[0] >= 1000:
            torch.empty(100 * 1024**4, dtype=torch.uint8, device=t.device)
        # The fixed 1 MiB keeps c changing by > 1 %, so the search reaches 1000.
        fixed = torch.empty(2**20, dtype=torch.uint8, device=t.device)
        scratch = torch.empty(4096 * t.shape[0], dtype=torch.uint8, device=t.device)
        del fixed, scratch
        return torch.ones((t.shape[0], 1), dtype=t.dtype, device=t.device)

    node = torch.zeros(1, dtype=torch.float64, device="cuda:0")
    solver._setup_memory_checks(f, node, take_gradient=False, total_mem_usage=0.9)
    assert calls == [1, 10, 100, 1000]
    # Measured at N = 100: (1 MiB + 4 KiB * 100 + the 800 B input, rounded up to
    # 512-byte blocks) / 100. A free-memory bound would be gigabytes per point.
    measured = (2**20 + 4096 * 100) / 100
    assert measured <= solver.f_unit_mem_size * GB <= measured + 64


@needs_cuda
def test_gpu_next_call_predicted_not_to_fit(monkeypatch, caplog):
    """With 2 MiB usable, an f using a fixed 64 KiB plus 100 KiB per point
    stops after N = 10 because 10 * M(10) (about 10 MiB) > 2 MiB. The fixed
    cost keeps c changing by > 1 %, so the convergence stop (checked first)
    cannot be the one that fires."""
    solver = gpu_solver(monkeypatch, usable_gb=2 * 2**20 / GB)
    calls = []

    def f(t, *args):
        calls.append(t.shape[0])
        fixed = torch.empty(64 * 1024, dtype=torch.uint8, device=t.device)
        scratch = torch.empty(
            100 * 1024 * t.shape[0], dtype=torch.uint8, device=t.device
        )
        del fixed, scratch
        return torch.ones((t.shape[0], 1), dtype=t.dtype, device=t.device)

    node = torch.zeros(1, dtype=torch.float64, device="cuda:0")
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        solver._setup_memory_checks(f, node, take_gradient=False, total_mem_usage=0.9)
    assert calls == [1, 10]
    assert "stopped because the next test would not fit in memory" in caplog.text


@needs_cuda
def test_gpu_benchmark_leaves_no_state(monkeypatch):
    """After a warm-up matmul (which allocates cuBLAS's persistent workspace),
    the benchmark returns allocated memory to its start and leaves the current
    CUDA device unchanged."""
    solver = gpu_solver(monkeypatch)
    node = torch.zeros(1, dtype=torch.float64, device="cuda:0")

    def f(t, *args):
        return torch.sin(t) * t

    warm = node.reshape(1, 1)
    (warm @ warm.T).sum()  # matmul: allocates cuBLAS's persistent workspace
    f(warm)  # warm-up, as integrate() does
    # The benchmark runs gc.collect(); collect first so garbage left by earlier
    # tests is not counted as memory the benchmark freed.
    gc.collect()
    start = torch.cuda.memory_allocated("cuda:0")
    device_before = torch.cuda.current_device()
    solver._setup_memory_checks(f, node, take_gradient=False, total_mem_usage=0.9)
    assert torch.cuda.memory_allocated("cuda:0") == start
    assert torch.cuda.current_device() == device_before
