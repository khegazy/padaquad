"""Tests for the memory benchmark, ``_setup_memory_checks``.

The benchmark fits the memory of one call of ``f`` on N points as
a + b N + d N^2 (``f_mem_coeffs``, GB, scaled by the gradient safety factor s,
with E, the Linux reading's error allowance, added to a). It calls ``f`` on
N = 1 point, then sizes each next call from the fitted memory and time:
N_next = min(10 N, n_mem, n_time), so an ``f`` whose memory grows like N^2
gets smaller steps instead of a call that cannot fit. The batch size is
computed later from live free memory (``_get_max_f_evals``): the largest B
whose fitted memory fits in ``_MEMORY_BUDGET`` of the usable memory.

CPU tests replace the memory readings and the clock with fakes, so every
expected value is exact and independent of the machine:

- ``FakeMemory`` stands in for the process's resident memory. A fake integrand
  adds a known number of bytes per call, so M is known exactly.
- ``FakeClock`` stands in for the ``time`` module of padaquad's memory module,
  so slow calls take the time the test says, without sleeping.

GPU tests use the real CUDA peak counters with buffers of known size.
"""

from __future__ import annotations

import gc
import logging
import math
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
BUDGET = qmem._MEMORY_BUDGET * USABLE_GB * GB  # bytes one call may use
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
    """``f(t) = 1`` that "uses" ``fixed + per_point * N + per_point_sq * N^p``
    bytes per call (p = ``exponent``, 2 by default; larger for memory that
    grows faster than N^2) and takes ``time_fixed + time_per_point_sq * N^2`` s
    on the fake clock.

    Records every batch size in ``calls``, and the bytes and seconds of each
    call that succeeded in ``used`` and ``seconds``. Below ``unread_below``
    points the call uses no memory (as when f reuses memory the process already
    holds, which the reading does not see). ``held`` is the number of freed
    bytes the process still holds and f reuses: a call raises the resident
    memory only by what it uses beyond them, so its reading is low by up to
    ``held`` (``used`` records the true bytes). From ``slow_from`` points on, each
    call takes 2 s more; from ``oom_from`` points on, it raises
    ``torch.OutOfMemoryError``; from ``bug_from`` points on, it raises
    ``ValueError`` (an error that is not about memory).
    """

    def __init__(
        self,
        memory,
        clock,
        fixed=0,
        per_point=1024,
        per_point_sq=0,
        exponent=2,
        time_fixed=0.0,
        time_per_point_sq=0.0,
        unread_below=None,
        held=0,
        slow_from=None,
        oom_from=None,
        bug_from=None,
    ):
        self.memory, self.clock = memory, clock
        self.fixed, self.per_point, self.per_point_sq = fixed, per_point, per_point_sq
        self.exponent, self.time_fixed = exponent, time_fixed
        self.time_per_point_sq, self.unread_below = time_per_point_sq, unread_below
        self.held = held
        self.slow_from, self.oom_from, self.bug_from = slow_from, oom_from, bug_from
        self.calls: list[int] = []
        self.used: list[int] = []
        self.seconds: list[float] = []

    def __call__(self, t, *args):
        n = t.shape[0]
        self.calls.append(n)
        if self.oom_from is not None and n >= self.oom_from:
            raise torch.OutOfMemoryError(f"simulated OOM at {n} points")
        if self.bug_from is not None and n >= self.bug_from:
            raise ValueError("f only handles one point")
        seconds = self.time_fixed + self.time_per_point_sq * n**2
        if self.slow_from is not None and n >= self.slow_from:
            seconds += 2.0
        self.clock.now += seconds
        used = self.fixed + self.per_point * n + self.per_point_sq * n**self.exponent
        if self.unread_below is not None and n < self.unread_below:
            used = 0
        self.memory.rss += max(0, used - self.held)
        self.used.append(used)
        self.seconds.append(seconds)
        return torch.ones((n, 1), dtype=t.dtype, device=t.device)


@pytest.fixture
def fake(monkeypatch):
    """CPU solver with fake memory readings, a fake clock, and no Linux reader.

    ``clear_refs`` is made unwritable, so the benchmark uses the before/after
    difference of the (fake) resident memory and E = 0. Like every CPU reader,
    it is taken to miss memory that f reuses (``set_reader`` changes both).
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


def set_reader(fake, monkeypatch, E=0, misses_reuse=True):
    """Give the fake reader the error allowance ``E`` (bytes) of the Linux
    reader, or (``misses_reuse=False``) make it count every allocation, as the
    GPU reader does."""
    reset, read = fake.solver._peak_memory_reader()[:2]
    monkeypatch.setattr(
        fake.solver, "_peak_memory_reader", lambda: (reset, read, E / GB, misses_reuse)
    )


def benchmark(solver, f, take_gradient=False):
    """Run the benchmark; return the stored fit (a, b, d) in bytes."""
    solver._setup_memory_checks(
        f, NODE, take_gradient=take_gradient, total_mem_usage=0.9
    )
    return tuple(x * GB for x in solver.f_mem_coeffs)


def safety(take_gradient):
    return qmem._GRADIENT_MEMORY_SAFETY_FACTOR if take_gradient else 1.0


# ---------------------------------------------------------------------------
# The fit and its inverse
# ---------------------------------------------------------------------------


def test_fit_recovers_quadratic_and_uses_fewer_terms_for_fewer_readings():
    """Exact readings of a + b n + d n^2 give back (a, b, d). One reading
    fits b n through the origin, two fit a + b n, and the coefficients are
    never negative (decreasing readings fit a constant)."""
    a, b, d = 3.0, 2.0, 0.5
    n = [1, 10, 100, 1000]
    y = [a + b * x + d * x * x for x in n]
    assert qmem._fit_up_to_quadratic(n, y) == pytest.approx((a, b, d), rel=1e-9)
    assert qmem._fit_up_to_quadratic([10], [5.0]) == pytest.approx((0, 0.5, 0))
    assert qmem._fit_up_to_quadratic([1, 10], [3.0, 21.0]) == pytest.approx(
        (1.0, 2.0, 0.0)
    )
    fit = qmem._fit_up_to_quadratic([1, 10, 100], [5.0, 4.0, 3.0])
    assert min(fit) >= 0
    assert fit[1:] == (0.0, 0.0)


@pytest.mark.parametrize(
    ("coeffs", "budget", "expected"),
    [
        ((5.0, 1.0, 0.0), 4.0, 0),  # the fixed cost alone exceeds the budget
        ((1.0, 0.0, 0.0), 4.0, qmem._MAX_N),  # nothing grows with n
        ((1.0, 2.0, 0.0), 8.0, 3),  # linear: (8 - 1) / 2 = 3.5
        ((0.0, 0.0, 2.0), 50.0, 5),  # quadratic: 2 n^2 <= 50
        ((1.0, 1.0, 1.0), 13.0, 3),  # 1 + 3 + 9 = 13 exactly
        ((0.0, 1e-12, 0.0), 1.0, qmem._MAX_N),  # capped at 1e9
    ],
)
def test_largest_n(coeffs, budget, expected):
    """_largest_n returns the largest n with a + b n + d n^2 <= budget."""
    n = qmem._largest_n(coeffs, budget)
    assert n == expected
    a, b, d = coeffs
    if 0 < n < qmem._MAX_N:
        assert a + b * n + d * n * n <= budget
        assert a + b * (n + 1) + d * (n + 1) ** 2 > budget


# ---------------------------------------------------------------------------
# Measuring the fit and sizing the test calls (CPU, fake readings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", [False, True])
def test_cost_converges(fake, caplog, take_gradient):
    """For memory a + b N, c(N) = s * (a + b N) / N falls as the fixed cost a
    is amortized. It changes by 8.4 % from 1e4 to 1e5 points, 0.9 % from 1e5
    to 1e6 and 0.09 % from 1e6 to 1e7: the first two consecutive changes below
    1 % end at N = 1e7, so the search stops there and not at 1e6. The fit is
    s * (a, b, 0), and the batch is the largest B with
    s * (a + b B) <= _MEMORY_BUDGET * usable. The end message gives this stop
    and no N^2 note."""
    a, b = 2**20, 1024
    s = safety(take_gradient)
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f, take_gradient)
    assert f.calls == [10**k for k in range(8)]
    assert "stopped because the cost per evaluation stopped changing" in caplog.text
    assert "grows like max_batch^2" not in caplog.text
    assert fit[:2] == pytest.approx((s * a, s * b), rel=1e-6)
    assert fake.solver._get_max_f_evals(0.9) == pytest.approx(
        (BUDGET - s * a) / (s * b), rel=1e-6
    )


def test_slow_call_stops_search(fake, caplog):
    """A call slower than the 1 s limit ends the search at that N (the
    backstop: the readings before it took no time, so no call was predicted
    to be slow). The fixed cost keeps c changing by > 1 %, so the convergence
    stop cannot fire. The reason is checked: without the backstop, the time
    fit through the three readings would stop the search at the same N."""
    a, b = 2**20, 4096
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b, slow_from=100)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100]
    assert "stopped because an evaluation took longer than 1.0 s" in caplog.text
    assert fit[:2] == pytest.approx((a, b), rel=1e-6)


def test_linear_steps_to_budget(fake, caplog):
    """Memory a + b N with a = 0.25 GB, b = 0.4 GB. Before three readings the
    next call is sized by the worst case a quadratic allows, M(k N) <= k^2 M(N):
    N = 1 -> 10 -> 42 (sqrt(75 / 4.25) = 4.2). From three readings the fit is
    exact, so N = 186, the largest call that fits in 75 GB. No call exceeds
    the budget (the old rule called N = 10 and then stopped)."""
    a, b = GB // 4, 2 * GB // 5
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert f.calls == [1, 10, 42, 186]
    assert max(f.used) <= BUDGET
    assert "the next test would not fit in memory" in caplog.text
    assert fake.solver._get_max_f_evals(0.9) == pytest.approx((BUDGET - a) / b, abs=1)


def test_quadratic_memory_steps_to_budget(fake, monkeypatch, caplog):
    """Regression (popcornn): memory 2148 B * N^2. The old rule predicted the
    next call as 10 * M and called N = 10,000 (200 GB with 100 GB usable).
    Now each call is sized from the fit, so the last one is the largest that
    fits, N = sqrt(75 GB / 2148 B) = 6122, and so is the batch. The batch
    follows the usable memory at the time it is computed: with a quarter of it
    left, it is half as large (3061), as memory growing like N^2 requires."""
    d = 2148
    f = CostIntegrand(fake.memory, fake.clock, per_point=0, per_point_sq=d)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 1000, 6122]
    assert max(f.used) <= BUDGET
    assert fit == pytest.approx((0, 0, d), abs=1e-3)  # bytes
    max_batch = fake.solver._get_max_f_evals(0.9)
    assert max_batch == pytest.approx(math.sqrt(BUDGET / d), abs=1)
    assert d * max_batch**2 <= BUDGET
    assert "grows like max_batch^2" in caplog.text
    monkeypatch.setattr(
        fake.solver, "_get_usable_memory", lambda total_mem_usage: USABLE_GB / 4
    )
    assert fake.solver._get_max_f_evals(0.9) == math.isqrt(int(BUDGET / 4 / d))


def test_fixed_cost_hiding_quadratic_term(fake):
    """Memory 0.4 GB + 2e-4 GB * N^2: the fixed cost hides the N^2 term at
    N <= 10 (the readings grow only 5 %), and a linear extrapolation from
    N = 100 would give a batch needing 3.5 TB. The fit through three readings
    sees the N^2 term, so neither a test call nor the batch exceeds 75 GB."""
    a, d = int(0.4 * GB), 2 * GB // 10**4
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=0, per_point_sq=d)
    benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 610]
    assert max(f.used) <= BUDGET
    max_batch = fake.solver._get_max_f_evals(0.9)
    assert max_batch == 610
    assert a + d * max_batch**2 <= BUDGET


def test_heavy_quadratic_first_steps(fake):
    """Memory 2 GB * N^2: one point uses 2 GB, so 10 points would use 200 GB.
    The worst-case bound M(k N) <= k^2 M(N) sizes the second call to
    k = sqrt(75 / 2) = 6.1 points (72 GB); the call after that would not fit."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=0, per_point_sq=2 * GB)
    benchmark(fake.solver, f)
    assert f.calls == [1, 6]
    assert max(f.used) <= BUDGET


def test_next_call_sized_by_time(fake, caplog):
    """Time 2e-6 s * N^2: N = 1000 would take 2 s. Each call is sized from the
    fitted time, so the last one takes just under the 1 s limit (N = 707)."""
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        fixed=2**20,
        per_point=0,
        per_point_sq=1,
        time_per_point_sq=2e-6,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 707]
    assert max(f.seconds) <= 1.0
    assert "the next test would take longer than 1.0 s" in caplog.text


def test_one_small_change_does_not_stop(fake):
    """Memory 1024 B N + 1 B N^2: c = 1024 + N changes by 0.9 % from N = 1 to
    10, then by 8.7 % from 10 to 100. The stop needs two consecutive changes
    below 1 %, and the latest one is not, so the search goes on past N = 100
    (a stop on the earlier change alone would end it there)."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=1024, per_point_sq=1)
    benchmark(fake.solver, f)
    assert f.calls[:4] == [1, 10, 100, 1000]


def test_next_call_sized_by_usable_memory_read_before_last_call(
    fake, monkeypatch, caplog
):
    """Memory 0.25 GB + 0.4 GB N, as in test_linear_steps_to_budget, but the
    usable memory falls from 100 GB to 40 GB (a 30 GB budget) after the first
    two calls. Each next call is sized from the usable memory read right
    before the call just made: N = 1 -> 10 -> 42 as before (both sized with
    100 GB); after N = 42 the fit allows (30 - 0.25) / 0.4 = 74 points, fewer
    than 2 * 42, so the search stops. The batch, computed from the live usable
    memory, is also 74."""
    f = CostIntegrand(fake.memory, fake.clock, fixed=GB // 4, per_point=2 * GB // 5)
    monkeypatch.setattr(
        fake.solver,
        "_get_usable_memory",
        lambda total_mem_usage: USABLE_GB if len(f.calls) < 2 else 40.0,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert f.calls == [1, 10, 42]
    assert "stopped because the next test would not fit in memory" in caplog.text
    assert fake.solver._get_max_f_evals(0.9) == 74


def test_zero_readings_do_not_converge(fake):
    """popcornn's pattern: calls below 100 points read 0 bytes (the memory is
    reused, not new). A zero cost per point is not "converged", so the search
    goes on and the fit sees the N^2 growth; no call exceeds the budget."""
    d = 2148
    f = CostIntegrand(
        fake.memory, fake.clock, per_point=0, per_point_sq=d, unread_below=100
    )
    fit = benchmark(fake.solver, f)
    assert f.calls[:3] == [1, 10, 100]
    assert len(f.calls) > 3
    assert max(f.used) <= BUDGET
    assert fit[2] > 0


def largest_fitting(memory):
    """The largest N with memory(N) <= BUDGET (memory nondecreasing)."""
    lo, hi = 0, qmem._MAX_N
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if memory(mid) <= BUDGET else (lo, mid - 1)
    return lo


@pytest.mark.parametrize(
    ("fixed", "per_point", "per_point_sq"),
    [
        (50 * 2**20, 50 * 1024, 50),  # all three terms matter near N = 1e3
        (2 * GB, 1024, 20),  # the fixed cost dominates the small calls
        (2**20, 4096, 0.004),  # the N^2 term matters only from N = 1e6 on
    ],
    ids=["comparable", "fixed-dominant", "late-quadratic"],
)
def test_offset_linear_quadratic_memory(fake, fixed, per_point, per_point_sq):
    """Memory a + b N + d N^2 with all three terms: no test call exceeds the
    budget, the fit recovers (a, b, d), and the batch is the largest N whose
    true memory fits in the budget."""
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        fixed=fixed,
        per_point=per_point,
        per_point_sq=per_point_sq,
    )
    fit = benchmark(fake.solver, f)
    assert max(f.used) <= BUDGET
    assert fit == pytest.approx((fixed, per_point, per_point_sq), rel=1e-6)

    def memory(n):
        return fixed + per_point * n + per_point_sq * n * n

    assert fake.solver._get_max_f_evals(0.9) == largest_fitting(memory)


def test_offset_linear_quadratic_memory_and_quadratic_time(fake):
    """Memory 50 MiB + 50 KiB N + 50 B N^2 and time 0.01 s + 1e-7 s N^2: the
    time bound sizes the calls (none takes more than 1 s, so they stop near
    N = 3146), and the memory fit, measured far below the batch size, still
    recovers (a, b, d) and gives the largest batch that fits. A second run on
    the same solver gives the same calls and fit."""
    a, b, d = 50 * 2**20, 50 * 1024, 50
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        fixed=a,
        per_point=b,
        per_point_sq=d,
        time_fixed=0.01,
        time_per_point_sq=1e-7,
    )
    fit = benchmark(fake.solver, f)
    assert f.calls == [1, 9, 89, 890, 3146]
    assert max(f.seconds) <= 1.0
    assert max(f.used) <= BUDGET
    assert fit == pytest.approx((a, b, d), rel=1e-6)
    assert fake.solver._get_max_f_evals(0.9) == largest_fitting(
        lambda n: a + b * n + d * n * n
    )
    # A second run on the same solver repeats the first: no search state is
    # carried over, and the search is deterministic.
    assert benchmark(fake.solver, f) == fit
    assert f.calls == 2 * [1, 9, 89, 890, 3146]


def test_slow_integrand_not_stopped_after_one_call(fake):
    """An f that takes 0.5 s per call whatever N (a fixed cost): the worst
    case k^2 bound alone would predict 2 s for two points and stop after one
    call, leaving a fit through the origin whose b includes the 1 MiB fixed
    cost. Before three readings doubling is always allowed, so the search
    goes on and the batch is the largest that fits."""
    a, b = 2**20, 1024
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b, time_fixed=0.5)
    benchmark(fake.solver, f)
    assert f.calls[:3] == [1, 2, 4]
    assert max(f.seconds) <= 1.0
    assert fake.solver._get_max_f_evals(0.9) == pytest.approx((BUDGET - a) / b, abs=1)


faster_than_n2 = pytest.mark.parametrize(
    ("fixed", "per_point_sq", "exponent", "slow_from"),
    [(0, 50, 2.5, None), (0, 2000, 3, None), (2**20, 0.01, 3, None), (0, 2000, 3, 100)],
    ids=["N^2.5", "N^3", "offset+N^3", "N^3-slow"],
)


@faster_than_n2
def test_memory_faster_than_n2_raises(
    fake, monkeypatch, fixed, per_point_sq, exponent, slow_from
):
    """Memory that grows faster than N^2 cannot be sized by a fit up to N^2.
    With readings that count every allocation (the GPU reader; set on the
    fake reader here), integrate() (which runs the benchmark when no max_batch
    is given) raises a RuntimeError that says why and tells the user to
    specify max_batch, before any test call exceeds the memory budget, and no
    fit is left stored (not even one from an earlier run). The check comes
    before the other stops: for N^3-slow, the call at N = 100 that shows the
    growth also takes longer than 1 s, and the error is still raised."""
    set_reader(fake, monkeypatch, misses_reuse=False)
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        fixed=fixed,
        per_point=0,
        per_point_sq=per_point_sq,
        exponent=exponent,
        slow_from=slow_from,
    )
    fake.solver.f_mem_coeffs = (0.0, 1e-6, 0.0)  # a fit from an earlier run
    with pytest.raises(RuntimeError) as info:
        fake.solver.integrate(
            f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
    message = str(info.value)
    assert "more than 3 times" in message
    assert "growth like N^2 allows" in message
    assert "Specify max_batch" in message
    assert "could not measure" not in message  # not reported as a failure
    assert max(f.used) <= BUDGET
    assert fake.solver.f_mem_coeffs is None


@faster_than_n2
def test_memory_faster_than_n2_caps_batch_on_cpu(
    fake, caplog, fixed, per_point_sq, exponent, slow_from
):
    """On a CPU the same readings do not raise: a CPU reading misses memory
    that f reuses, so growth faster than N^2 cannot be told from an earlier
    reading that was too low (test_low_reading_from_reused_memory_caps_batch).
    The search stops at the call that shows the growth, a warning says so, and
    the batch is capped at that call's N, which ran within the memory budget:
    the integration (44 points, f = 1 on [0, 1]) completes within the cap,
    which is stored for same_f=True. The cap is the largest call of f: no
    later test call or batch exceeds it."""
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        fixed=fixed,
        per_point=0,
        per_point_sq=per_point_sq,
        exponent=exponent,
        slow_from=slow_from,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        result = fake.solver.integrate(
            f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
    cap = max(f.calls)
    assert cap >= 87  # the benchmark's call, not one of the 44-point batches
    assert "growth like N^2 allows" in caplog.text
    assert f"max_batch is capped at {cap}, the largest call that ran" in caplog.text
    assert "Specify max_batch" in caplog.text
    assert f"pass max_batch={cap} " in caplog.text
    assert "grows like max_batch^2" not in caplog.text
    assert max(f.used) <= BUDGET
    assert fake.solver.get_max_batch() == fake.solver.previous_max_batch == cap
    assert fake.solver.f_mem_coeffs is not None
    assert float(result.integral.sum()) == pytest.approx(1.0)


def test_low_reading_from_reused_memory_caps_batch(fake, caplog):
    """Regression: memory 8250 B * N^2 exactly (popcornn's integrand), in a
    process that already holds 95 % of the memory of a 100-point call as freed
    memory, which f reuses. The readings are 0 at N = 1 and 10, 3.9 MiB at
    N = 100 (5 % of the true 78.7 MiB) and nearly the true 7.7 GB at N = 1000:
    2000 times more for 10 times the points. This used to raise "faster than
    N^2" for an integrand that is not (popcornn's Linux run read 4 MiB at
    N = 49 and 1870 MiB at N = 490, which raises with the error allowance of
    a 4-core machine; on a Mac the error was reproduced with a real integrand).
    Now the benchmark warns and caps the batch at 1000, the call that ran;
    the fit alone (d = 8170 B from the last, slightly low reading) would give
    3139 points, which need 1 % more than the budget."""
    d = 8250
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        per_point=0,
        per_point_sq=d,
        held=int(0.95 * d * 100**2),
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 1000]
    assert "or the earlier reading was too low" in caplog.text
    assert "max_batch is capped at 1000" in caplog.text
    assert fit[2] == pytest.approx(d - f.held / 1000**2, rel=1e-3)
    assert fake.solver._get_max_f_evals(0.9) == 3139
    assert d * 3139**2 > BUDGET
    assert fake.solver.get_max_batch() == 1000


def test_only_cuda_reader_counts_reused_memory(fake, monkeypatch):
    """The CPU readers report that they miss memory f reuses, so the benchmark
    warns and caps the batch where it would raise with the CUDA reader, which
    counts every allocation. Building the CUDA reader calls no CUDA function,
    so this runs without a GPU."""
    assert fake.solver._peak_memory_reader()[2:] == (0.0, True)
    monkeypatch.setattr(fake.solver, "device", torch.device("cuda:0"))
    assert fake.solver._peak_memory_reader()[2:] == (0.0, False)


def test_tiny_integrand_capped_at_search_limit(fake):
    """For an f that uses one byte per point, the cost per point is constant,
    so the search stops after three readings, and the batch size is capped at
    the 1e9 search limit."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=1)
    fit = benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100]
    assert fit[1] == pytest.approx(1)
    assert fake.solver._get_max_f_evals(0.9) == qmem._MAX_N


def test_zero_memory_integrand_reaches_search_limit(fake, monkeypatch, caplog):
    """Calls that read no memory and take no time: a cost per point of 0 is
    not "converged" and nothing bounds the next call, so N grows tenfold up to
    1e9 and the search stops at its limit. The fit is 0, the batch is capped
    at 1e9, and the end message has no N^2 note (0 is not more than 10 % of
    0). ``_measure_call`` is replaced, so no input of 1e9 points is built."""
    calls = []

    def measure_call(f, f_args, node_test, N, *rest):
        calls.append(N)
        return 0.0, 0.0  # (M in GB, seconds)

    monkeypatch.setattr(fake.solver, "_measure_call", measure_call)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, None)
    assert calls == [10**k for k in range(10)]
    assert fit == (0.0, 0.0, 0.0)
    assert fake.solver._get_max_f_evals(0.9) == qmem._MAX_N
    assert "stopped because the 1e9 search limit was reached" in caplog.text
    assert "grows like max_batch^2" not in caplog.text


@pytest.mark.parametrize(("share", "noted"), [(0.05, False), (0.2, True)])
def test_quadratic_note_threshold(fake, caplog, share, noted):
    """The end message names the N^2 growth only when the N^2 term is more
    than 10 % of the memory of a batch. Memory b N + d N^2 is built so that
    the batch is B = 1e6 points with the N^2 term a share of 5 % or 20 % of
    its memory."""
    B = 10**6
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        per_point=(1 - share) * BUDGET / B,
        per_point_sq=share * BUDGET / B**2,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert fake.solver._get_max_f_evals(0.9) == pytest.approx(B, abs=1)
    assert ("grows like max_batch^2" in caplog.text) == noted


@pytest.mark.parametrize(
    ("per_point_sq", "exponent", "E", "raises_at"),
    [
        (2**20, 2.47, 0, None),  # 10^2.47 = 295 < 3 * 10^2 per tenfold step
        (2**20, 2.48, 0, (1, 10)),  # 10^2.48 = 302 > 300
        (2**20, 2.48, 2**20, (10, 100)),  # E doubles the first bound
        (2**20 - 1024, 3, 0, (10, 87)),  # 1023 KiB at N = 1: not compared
        (2**20, 3, 0, (1, 10)),  # 1 MiB at N = 1: compared
    ],
    ids=["N^2.47", "N^2.48", "N^2.48+E", "N^3-below-1MiB", "N^3-at-1MiB"],
)
def test_faster_than_n2_tolerance_floor_and_error_allowance(
    fake, monkeypatch, per_point_sq, exponent, E, raises_at
):
    """Memory d N^p with d about 1 MiB, with readings that count every
    allocation (as on a GPU). A call raises when its reading M
    exceeds 3 (N / N_prev)^2 (M_prev + E), compared only with the previous
    reading and only when that reading is at least 1 MiB; ``raises_at`` is
    (N_prev, N) of the call that raises, or None.

    - p = 2.47 grows 295 times over the tenfold step from N = 1 and never
      raises (smaller steps grow less); p = 2.48 grows 302 times and raises
      at N = 10.
    - With E = 1 MiB the bound at N = 10 is 300 * 2 MiB, so p = 2.48 passes
      there; at N = 100, M = 10^4.96 MiB = 91,201 MiB exceeds
      300 * (302 + 1) MiB = 90,900 MiB and raises.
    - For p = 3 the reading at N = 1 decides: 1023 KiB is below the 1 MiB
      floor, so N = 10 is not compared and the search goes on to N = 87 (the
      worst-case step), which is compared and raises; 1 MiB raises at
      N = 10."""
    set_reader(fake, monkeypatch, E=E, misses_reuse=False)
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        per_point=0,
        per_point_sq=per_point_sq,
        exponent=exponent,
    )
    if raises_at is None:
        benchmark(fake.solver, f)
        assert fake.solver.f_mem_coeffs is not None
        return
    with pytest.raises(RuntimeError, match=r"growth like N\^2 allows") as info:
        benchmark(fake.solver, f)
    n_prev, n = raises_at
    assert f.calls[-1] == n
    assert f"bytes at {n} points" in str(info.value)
    assert f"bytes at {n_prev} points" in str(info.value)


# ---------------------------------------------------------------------------
# Out-of-memory errors during the measurement
# ---------------------------------------------------------------------------


def test_oom_after_n_100_keeps_measured_fit(fake, caplog):
    """An out-of-memory error at N = 1000 after a successful N = 100 keeps
    the fit of the calls that succeeded and caps the batch at
    _OOM_BACKOFF * 1000 = 750 for the rest of the integrate() call. The end
    message reports the capped batch, not the ~7.9e7 points the fit allows."""
    a, b = 2**20, 1024
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b, oom_from=1000)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 1000]
    assert fit[:2] == pytest.approx((a, b), rel=1e-6)
    assert fake.solver._oom_max_batch == 750
    assert fake.solver.get_max_batch() == 750
    assert "max_batch is about 750 " in caplog.text
    assert "pass max_batch=750 " in caplog.text


@pytest.mark.parametrize(
    ("time_fixed", "oom_from", "cap"), [(0.0, 10, 7), (0.5, 2, 1)], ids=["10", "2"]
)
def test_oom_at_small_n_caps_batch(fake, time_fixed, oom_from, cap):
    """An out-of-memory error on the second call caps the batch at
    int(0.75 N): 7 points at N = 10, and 1 point at N = 2, reached by doubling
    because the first call takes 0.5 s. The fit of the one call that
    succeeded is kept."""
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        per_point=1024,
        time_fixed=time_fixed,
        oom_from=oom_from,
    )
    fit = benchmark(fake.solver, f, take_gradient=True)
    assert f.calls == [1, oom_from]
    assert fit[1] == pytest.approx(safety(True) * 1024)
    assert fake.solver._oom_max_batch == cap
    assert fake.solver.get_max_batch() == cap


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


@pytest.mark.parametrize("take_gradient", [False, True])
@pytest.mark.parametrize(
    ("per_point", "per_point_sq", "E", "calls"),
    [
        (1024, 0, 2**20, [10**k for k in range(8)]),
        (0, 2148, 25 * GB, [1, 10, 100, 1000, 4999]),
    ],
    ids=["linear", "quadratic"],
)
def test_error_allowance(
    fake, monkeypatch, take_gradient, per_point, per_point_sq, E, calls
):
    """E, the most a memory reading can be too low (nonzero only with the
    Linux reader), is set here on the fake reader. It is a fixed amount, not
    a cost per point:

    - in the cost per point c = s (M + E) / N it acts as a fixed cost
      (linear, E = 1 MiB: c changes by < 1 % twice only at N = 1e7, as in
      test_cost_converges, instead of at N = 100);
    - it takes its share of every test call's budget once (quadratic,
      E = 25 GB: the last call is sqrt(50 GB / 2148 B) = 4999, not 6122);
    - the stored fit is s * (E, b, d), and the batch is the largest B with
      s * (E + b B + d B^2) <= 75 GB.

    The test calls run f forward only, so they do not depend on s."""
    set_reader(fake, monkeypatch, E=E)
    s = safety(take_gradient)
    f = CostIntegrand(
        fake.memory, fake.clock, per_point=per_point, per_point_sq=per_point_sq
    )
    fit = benchmark(fake.solver, f, take_gradient)
    assert f.calls == calls
    assert fit == pytest.approx(
        (s * E, s * per_point, s * per_point_sq), rel=1e-6, abs=1e-3
    )
    assert fake.solver._get_max_f_evals(0.9) == pytest.approx(
        largest_fitting(lambda n: s * (E + per_point * n + per_point_sq * n * n)),
        rel=1e-6,
    )


@pytest.mark.skipif(not hasattr(os, "sysconf"), reason="needs os.sysconf")
@pytest.mark.parametrize("cores", [12, 256])
def test_linux_reader_adds_error_allowance(fake, monkeypatch, cores):
    """With the Linux reader, M comes from the (reset) peak VmHWM in
    /proc/self/status and E = cores * max(32, 2 * cores) pages is added to the
    fixed term of the fit: 1.5 MiB on 12 cores and 512 MiB on 256 cores with
    4 KiB pages. The call at N = 1 is slow, so the fit is the one reading,
    4096 bytes per point, plus E. /proc is faked, so this runs on any system
    with os.sysconf.

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
    fit = benchmark(fake.solver, f)
    E = cores * max(32, 2 * cores) * os.sysconf("SC_PAGE_SIZE")
    assert f.calls == [1]
    assert fit == pytest.approx((E, 4096, 0))
    # A peak of resident memory also misses memory that f reuses.
    assert fake.solver._peak_memory_reader()[2:] == (E / GB, True)


@pytest.mark.skipif(
    not os.access("/proc/self/clear_refs", os.W_OK), reason="needs Linux clear_refs"
)
def test_real_linux_reader_smoke(monkeypatch):
    """With the real reader, a 200 MiB allocation that f frees before returning
    is measured. E is made negligible (one core), so the fit is the reading
    itself.
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
    assert sum(solver.f_mem_coeffs) * GB >= 150 * 2**20  # memory at N = 1


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


@pytest.mark.skipif(
    not os.access("/proc/self/clear_refs", os.W_OK), reason="needs Linux clear_refs"
)
def test_real_linux_reader_quadratic_memory(monkeypatch):
    """Regression (popcornn), with the real reader: f builds an N x N float64
    intermediate (8 N^2 bytes) and frees it before returning. With 2 GiB
    usable, the old code measured 0.75 GiB at N = 10,000 and sized batches as
    usable / (cost per point), about 26,800 points needing 5.4 GiB. Now no
    test call and no batch exceeds the 1.5 GiB budget by more than the fit's
    error: in one run on a Mac (RSS reader, f keeping the N x N buffer alive),
    real readings put about 350 B per point into b and gave d = 7.8 instead
    of 8, so the batch needed 2 % more than the budget, still within the 25 %
    of usable memory the budget keeps free (other runs: b = 150-210 B,
    d = 7.98, 0.1 % over)."""
    usable_gb = 2.0
    solver = make_solver_for_unit_test()
    monkeypatch.setattr(solver, "_get_usable_memory", lambda total_mem_usage: usable_gb)
    calls = []

    def f(t, *args):
        n = t.shape[0]
        calls.append(n)
        a = t.expand(n, n) * t.T  # [N, N] float64: 8 N^2 bytes
        return a.sum(dim=1, keepdim=True)

    solver._setup_memory_checks(f, NODE, take_gradient=False, total_mem_usage=0.9)
    budget = qmem._MEMORY_BUDGET * usable_gb * GB
    # 10 % over the budget is 83 % of usable memory.
    assert all(8 * n**2 <= 1.1 * budget for n in calls)
    max_batch = solver._get_max_f_evals(0.9)
    assert 8 * max_batch**2 <= 1.1 * budget
    if max(calls) >= 1000:  # the N^2 term dominates the readings from here on
        assert 7.0 <= solver.f_mem_coeffs[2] * GB <= 9.0


# ---------------------------------------------------------------------------
# Through integrate(): messages and the gradient path
# ---------------------------------------------------------------------------


def test_messages_when_measuring(caplog, monkeypatch):
    """Without max_batch the start warning explains the measurement and tells
    the user to pass max_batch; the end warning reports time, the reason, c,
    and a max_batch estimate (for this take_gradient) to pass later. With
    max_batch, neither appears."""
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
    assert "To skip it, pass max_batch" in text
    assert "same_f=True" in text
    assert "Memory measurement finished in" in text
    assert "bytes per evaluation" in text
    assert "stopped because" in text
    assert "usable now, max_batch is about" in text
    assert "for take_gradient=True" in text
    assert "pass max_batch=" in text

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
    measured, so the batch comes from the fit and integrate() completes."""
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


@pytest.mark.parametrize(
    ("cost", "test_calls", "max_batch"),
    [
        ({"per_point": 0, "per_point_sq": int(BUDGET / 400)}, [1, 10, 20], 20),
        ({"oom_from": 10}, [1, 10], 7),
    ],
    ids=["N^2", "out-of-memory"],
)
def test_integrate_batches_within_measured_max_batch(
    fake, caplog, cost, test_calls, max_batch
):
    """Through integrate() without max_batch: f is called on one point (the
    warm-up), then by the benchmark, then by the integration in batches of at
    most the measured max_batch, and the integral of f = 1 over [0, 1] is 1.
    For memory BUDGET / 400 * N^2 the batch is the largest that fits,
    sqrt(400) = 20 points; for an out-of-memory error at N = 10 it is the
    cap, 7. Both are below the 44 points (11 subintervals of bosh3's 4 nodes)
    the integration needs, so it takes several batches. The batch is stored
    as previous_max_batch, so a same_f=True call skips the benchmark and
    keeps to it."""
    f = CostIntegrand(fake.memory, fake.clock, **cost)
    bounds = {"mesh_init": T_INIT, "mesh_final": T_FINAL, "take_gradient": False}
    result = fake.solver.integrate(f, **bounds)
    assert f.calls[: 1 + len(test_calls)] == [1, *test_calls]
    batches = f.calls[1 + len(test_calls) :]
    assert len(batches) >= 2
    assert max(batches) <= max_batch
    assert float(result.integral.sum()) == pytest.approx(1.0)
    assert fake.solver.previous_max_batch == max_batch

    n_first = len(f.calls)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        result = fake.solver.integrate(f, same_f=True, **bounds)
    assert "PadaQuad is measuring" not in caplog.text
    assert max(f.calls[n_first:]) <= max_batch
    assert float(result.integral.sum()) == pytest.approx(1.0)


def test_get_max_batch_uses_solver_total_mem_usage(monkeypatch):
    """get_max_batch() without total_mem_usage uses the solver's own value, not
    a second hard-coded default."""
    solver = make_solver_for_unit_test(total_mem_usage=0.5)
    seen = []

    def usable(total_mem_usage):
        seen.append(total_mem_usage)
        return 1.0  # GB

    monkeypatch.setattr(solver, "_get_usable_memory", usable)
    solver.f_mem_coeffs = (0.0, 1e-6, 0.0)  # GB: 1e-6 per evaluation
    assert solver.get_max_batch() == pytest.approx(qmem._MEMORY_BUDGET * 10**6, abs=1)
    assert seen == [0.5]


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
    returning. The peak counter sees them, so the fit's cost per point is
    about 1 KiB + the input and output tensors."""
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
    b_bytes = solver.f_mem_coeffs[1] * GB
    # 1 KiB scratch + 8 B input per point (the output reuses the freed scratch
    # block); the fixed 1 MiB goes to the fit's fixed term.
    assert per_point <= b_bytes <= 1.02 * (per_point + 16)


@needs_cuda
def test_gpu_cost_includes_gradient_inside_f(monkeypatch):
    """popcornn-like: f differentiates a network inside itself with
    take_gradient=False. The hidden activations exist only during the call;
    the peak counter still counts them, so the fitted memory of 1000 points
    exceeds one hidden layer of 1000 points."""
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
    a, b, d = solver.f_mem_coeffs
    assert (a + b * 1000 + d * 1000**2) * GB > width * 8 * 1000


@needs_cuda
def test_gpu_oom_after_n_100_keeps_measured_cost(monkeypatch):
    """A real CUDA out-of-memory error (an instantly refused 100 TB request)
    at N = 1000 keeps the fit of the calls that succeeded and caps the batch
    at _OOM_BACKOFF * 1000 = 750."""
    solver = gpu_solver(monkeypatch)
    calls = []

    def f(t, *args):
        calls.append(t.shape[0])
        if t.shape[0] >= 1000:
            torch.empty(100 * 1024**4, dtype=torch.uint8, device=t.device)
        # The fixed 1 MiB keeps the cost per point changing by > 1 %, so the
        # search reaches 1000.
        fixed = torch.empty(2**20, dtype=torch.uint8, device=t.device)
        scratch = torch.empty(4096 * t.shape[0], dtype=torch.uint8, device=t.device)
        del fixed, scratch
        return torch.ones((t.shape[0], 1), dtype=t.dtype, device=t.device)

    node = torch.zeros(1, dtype=torch.float64, device="cuda:0")
    solver._setup_memory_checks(f, node, take_gradient=False, total_mem_usage=0.9)
    assert calls == [1, 10, 100, 1000]
    assert solver._oom_max_batch == 750
    assert solver.get_max_batch() == 750
    # The fit passes through the reading at N = 100: 1 MiB + 4 KiB * 100 + the
    # 800 B input, rounded up to 512-byte blocks.
    a, b, d = solver.f_mem_coeffs
    measured = 2**20 + 4096 * 100 + 1024
    assert (a + b * 100 + d * 100**2) * GB == pytest.approx(measured, abs=1024)


@needs_cuda
def test_gpu_next_call_predicted_not_to_fit(monkeypatch, caplog):
    """With 32 MiB usable (a 24 MiB budget per call), an f using a fixed
    64 KiB plus 100 KiB per point takes steps 1, 10, then about 48 (the worst
    case k^2 growth before three readings), then the largest call that fits,
    about 245. No call exceeds the budget, and the search stops because the
    next call would not fit."""
    usable_gb = 32 * 2**20 / GB
    solver = gpu_solver(monkeypatch, usable_gb=usable_gb)
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
    assert calls[:2] == [1, 10]
    budget = qmem._MEMORY_BUDGET * usable_gb * GB
    assert 64 * 1024 + 100 * 1024 * max(calls) <= budget
    assert max(calls) > 100  # it got close to the budget
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
