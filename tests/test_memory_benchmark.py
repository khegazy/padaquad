"""Tests for the memory benchmark, ``_setup_memory_checks``.

The benchmark fits the memory of one call of ``f`` on N points as
a + b N + d N^2 (``f_mem_coeffs``, GB, scaled by the gradient safety factor s,
with E, the Linux reading's error allowance, added to a). Only readings at or
above the reader's noise floor (4 MiB on a CPU, 10 KB on a GPU) are fitted,
and the search goes on until the fit is ready: three of them, whose fitted
memory grows by at least the floor. It calls ``f`` on N = 1 point, then
sizes each next call from the fitted memory and time:
N_next = min(step N, n_mem, n_time), so an ``f`` whose memory grows like N^2
gets smaller steps instead of a call that cannot fit. The batch size is
computed later from live free memory (``_get_max_f_evals``): the largest B
whose fitted memory fits in ``_MEMORY_BUDGET`` of the usable memory.

CPU tests replace the memory readings and the clock with fakes, so every
expected value is exact and independent of the machine:

- ``FakeMemory`` stands in for the process's resident memory. A fake integrand
  adds a known number of bytes per call, so M is known exactly (a reading is
  never below the bytes of the call's input and output, 16 bytes per point
  for the fakes here).
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
from padaquad import adaptive_quadrature, integrate, steps

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
    difference of the (fake) resident memory: E = 0, a noise floor of 4 MiB,
    and steps of 10x (it does not count memory f reuses). ``set_reader``
    changes these.
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


def set_reader(solver, monkeypatch, **changes):
    """Change fields of the solver's memory reader (``qmem._Reader``): ``E``
    and ``floor`` in bytes, ``counts_reuse`` (True: steps may be 2x, as on a
    GPU or Linux with ``malloc_trim``)."""
    reader = solver._peak_memory_reader()
    changes = {k: v / GB if k in ("E", "floor") else v for k, v in changes.items()}
    monkeypatch.setattr(
        solver, "_peak_memory_reader", lambda: reader._replace(**changes)
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
        ((0.0, 1.0, 1.0), math.inf, qmem._MAX_N),  # no limit (limit_benchmark_time)
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
    """A call slower than the 1 s limit ends the search at that N once the fit
    is ready (the backstop: the readings before it took no time, so no call
    was predicted to be slow). The 8 MiB fixed cost puts every reading above
    the 4 MiB floor and keeps c changing by > 1 %, so the convergence stop
    cannot fire; 64 KiB per point makes the readings grow by 6.2 MiB, more
    than the floor, from N = 1 to 100. The reason is checked: without the
    backstop, the time fit through the three readings would stop the search
    at the same N."""
    a, b = 8 * 2**20, 64 * 1024
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


def test_heavy_quadratic_first_steps(fake, caplog):
    """Memory 2 GB * N^2: one point uses 2 GB, so 10 points would use 200 GB.
    The worst-case bound M(k N) <= k^2 M(N) sizes the second call to
    k = sqrt(75 / 2) = 6.1 points (72 GB); the call after that would not fit,
    so the memory budget ends the search with only two reliable readings.
    That is not an error: the stored fit is the worst case N^2 through the
    reading at N = 6, so the batch is 6 (it ran), and the end message says
    that max_batch assumes the worst case."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=0, per_point_sq=2 * GB)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f)
    assert f.calls == [1, 6]
    assert max(f.used) <= BUDGET
    assert fit == pytest.approx((0, 0, 2 * GB))
    assert fake.solver._get_max_f_evals(0.9) == 6
    assert "(2 reached the noise floor" in caplog.text


def test_next_call_sized_by_time(fake, caplog):
    """Time 2e-6 s * N^2: N = 1000 would take 2 s. Each call is sized from the
    fitted time, so the last one takes just under the 1 s limit (N = 707)."""
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        fixed=8 * 2**20,  # every reading above the 4 MiB floor
        per_point=0,
        per_point_sq=1024,  # 10 MiB more at N = 100: the fit is ready there
        time_per_point_sq=2e-6,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 707]
    assert max(f.seconds) <= 1.0
    assert "the next test would take too long" in caplog.text


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


@pytest.mark.parametrize(
    ("fixed", "per_point_sq", "exponent", "slow_from"),
    [(0, 50, 2.5, None), (0, 8000, 3, None), (2**20, 0.01, 3, None), (0, 8000, 3, 100)],
    ids=["N^2.5", "N^3", "offset+N^3", "N^3-slow"],
)
@pytest.mark.parametrize("counts_reuse", [False, True], ids=["cpu", "gpu-like"])
def test_memory_faster_than_n2_raises(
    fake, monkeypatch, fixed, per_point_sq, exponent, slow_from, counts_reuse
):
    """Memory that grows faster than N^2 cannot be sized by a fit up to N^2.
    On every reader (the CPU reader of the fake, and one that counts every
    allocation, as on a GPU), get_max_batch(f=...) raises a RuntimeError that says why, names the other
    possible cause on a CPU (an earlier reading too low), and tells the user to
    specify max_batch, before any test call exceeds the memory budget; no fit
    is left stored (not even one from an earlier run). Each case's previous
    reading is above the 4 MiB floor (8000 B N^3 is 8 MB at N = 10). The check
    comes before the other stops: for N^3-slow, the call at N = 100 that shows
    the growth also takes longer than 1 s, and the error is still raised."""
    set_reader(fake.solver, monkeypatch, counts_reuse=counts_reuse)
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
        fake.solver.get_max_batch(f=f, t=T_INIT, take_gradient=False)
    message = str(info.value)
    assert "more than 3 times" in message
    assert "growth like N^2 allows" in message
    assert "the earlier reading was too low" in message
    assert "Specify max_batch" in message
    assert "could not measure" not in message  # not reported as a failure
    assert max(f.used) <= BUDGET
    assert fake.solver.f_mem_coeffs is None


def test_low_reading_from_reused_memory_raises(fake):
    """Memory 8250 B * N^2 exactly (popcornn's integrand), in a process that
    already holds 90 % of the memory of a 100-point call as freed memory, which
    f reuses. The reading at N = 100 is 7.9 MiB (10 % of the true 78.7 MiB,
    above the 4 MiB floor) and nearly the true 7.7 GB at N = 1000: 1000 times
    more for 10 times the points. The benchmark cannot tell this from memory
    growing faster than N^2, so it raises and asks for max_batch, naming both
    causes (Kareem's choice: raise on every reader; on a CPU an N^2 integrand
    can then occasionally raise when the process had evaluated f before)."""
    d = 8250
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        per_point=0,
        per_point_sq=d,
        held=int(0.9 * d * 100**2),
    )
    with pytest.raises(RuntimeError, match="the earlier reading was too low"):
        benchmark(fake.solver, f)
    assert f.calls == [1, 10, 100, 1000]
    assert max(f.used) <= BUDGET


def test_reader_floor_and_reuse(fake, monkeypatch, caplog):
    """The CPU reader without a trim (the fake's before/after resident memory)
    has E = 0, a noise floor of 4 MiB, and does not count memory f reuses, so
    its steps stay at 10x and the start warning tells the user that readings
    can be far too low. The CUDA reader has a floor of 10 KB and counts every
    allocation. Building the CUDA reader calls no CUDA function, so this runs
    without a GPU."""
    reader = fake.solver._peak_memory_reader()
    assert (reader.E, reader.floor * GB, reader.counts_reuse) == (0.0, 4 * 2**20, False)
    f = CostIntegrand(fake.memory, fake.clock, fixed=8 * 2**20)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert "readings can be far too low" in caplog.text
    caplog.clear()
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert "readings can be far too low" not in caplog.text
    cuda = make_solver_for_unit_test()
    cuda.device = torch.device("cuda:0")
    reader = cuda._peak_memory_reader()
    assert (reader.E, reader.floor * GB, reader.counts_reuse) == (0.0, 10_000, True)


def stub_calls(solver, monkeypatch, memory, seconds=lambda n: 0.0):
    """Replace ``_measure_call`` by readings ``memory(N)`` (bytes) and times
    ``seconds(N)``, so no input of up to 1e9 points is built. Returns the list
    of the N called."""
    calls = []

    def measure_call(f, f_args, node_test, N, *rest):
        calls.append(N)
        return memory(N) / GB, seconds(N)

    monkeypatch.setattr(solver, "_measure_call", measure_call)
    return calls


def test_tiny_integrand_capped_at_search_limit(fake, monkeypatch):
    """An f that uses only its input and output, 16 bytes per point: its
    readings reach the 4 MiB floor at N = 1e6, and the cost per point is
    constant, so the search stops at the third reliable reading (N = 1e8), and
    the batch, 75 GB / 16 B, is capped at the 1e9 search limit."""
    calls = stub_calls(fake.solver, monkeypatch, lambda n: 16 * n)
    fit = benchmark(fake.solver, None)
    assert calls == [10**k for k in range(9)]
    assert fit[1] == pytest.approx(16)
    assert fake.solver._get_max_f_evals(0.9) == qmem._MAX_N


def test_no_reading_above_floor_raises(fake, monkeypatch):
    """Calls that read no memory and take no time (memory hidden from the
    reader, or truly none): nothing bounds the next call, so N grows tenfold
    up to the 1e9 search limit without a single reading above the 4 MiB
    floor. Nothing is assumed in that case: the benchmark raises, saying that
    no call reached the floor, listing the readings and asking for
    max_batch."""
    calls = stub_calls(fake.solver, monkeypatch, lambda n: 0)
    with pytest.raises(RuntimeError) as info:
        benchmark(fake.solver, None)
    message = str(info.value)
    assert calls == [10**k for k in range(10)]
    assert "No test call of f used at least the noise floor" in message
    assert "N=1000000000: 0.00 MiB" in message
    assert "below the noise floor" in message
    assert "Specify max_batch" in message
    assert fake.solver.f_mem_coeffs is None


@pytest.mark.parametrize(("share", "noted"), [(0.05, False), (0.2, True)])
def test_quadratic_note_threshold(fake, monkeypatch, caplog, share, noted):
    """The end message names the N^2 growth only when the N^2 term is more
    than 10 % of the memory of a batch. Memory b N + d N^2 is built so that
    the batch is B = 1e6 points with the N^2 term a share of 5 % or 20 % of
    its memory. The reader counts reused memory, so the batch is not capped
    at the largest test call (1e4 points)."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
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
        (2**20, 2.2, 0, None),  # 174^0.2 = 2.8 < 3 over the whole search
        (2**20, 2.47, 0, (1, 100)),  # 10^0.47 = 2.95 < 3, 100^0.47 = 8.7 > 3
        (2**20, 2.48, 0, (1, 10)),  # 10^2.48 = 302 > 300
        (2**20, 2.48, 2**20, (1, 100)),  # E doubles the bound from N = 1
        (2**20 - 1024, 3, 0, (10, 87)),  # 1023 KiB at N = 1: not compared
        (2**20, 3, 0, (1, 10)),  # 1 MiB at N = 1: compared
    ],
    ids=["N^2.2", "N^2.47", "N^2.48", "N^2.48+E", "N^3-below-1MiB", "N^3-at-1MiB"],
)
def test_faster_than_n2_tolerance_floor_and_error_allowance(
    fake, monkeypatch, per_point_sq, exponent, E, raises_at
):
    """Memory d N^p with d about 1 MiB, with the reader's noise floor set to
    1 MiB (the rule is the same for every floor). A call raises when its
    reading M exceeds 3 (N / N_i)^2 (M_i + E) for an earlier reading of at
    least 1 MiB; the message names the one with the smallest bound.
    ``raises_at`` is (N_i, N) of the call that raises, or None. N^p exceeds
    the N^2 bound by (N / N_i)^(p - 2), so it is caught across a range of N
    where this passes 3.

    - p = 2.2 is not caught: across the whole search (N = 1 to 174, the worst
      case step through N = 100) it exceeds N^2 by 2.8 times.
    - p = 2.47 exceeds it by 2.95 times from N = 1 to 10 (passes) and by 8.7
      times from N = 1 to 100 (raises); p = 2.48 by 3.02 times at N = 10.
    - With E = 1 MiB the bound from N = 1 doubles, so p = 2.48 passes at
      N = 10 and raises at N = 100: 10^4.96 MiB = 91,201 MiB exceeds
      3 * 10^4 * 2 MiB.
    - For p = 3 the reading at N = 1 decides: 1023 KiB is below the 1 MiB
      floor, so N = 10 is not compared and the search goes on to N = 87 (the
      worst-case step), which is compared with N = 10 and raises; 1 MiB
      raises at N = 10."""
    set_reader(fake.solver, monkeypatch, E=E, floor=2**20)
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
# Reliable readings, the phase before three of them, and the 60 s limit
# ---------------------------------------------------------------------------


def test_tensor_bytes_counts_distinct_storages_on_device():
    """The lower bound of a reading counts each storage once: a view of a
    tensor shares its storage, and an output that is the input is not counted
    twice. Tensors in nested tuples and lists count; other objects and
    tensors on another device type do not."""
    cpu = torch.device("cpu")
    a = torch.zeros(1000, dtype=torch.float64)  # 8000 bytes
    b = torch.zeros(10, 10, dtype=torch.float32)  # 400 bytes
    assert qmem._tensor_bytes(a, cpu) == 8000
    assert qmem._tensor_bytes((a, a[:10], [a.view(10, 100)]), cpu) == 8000
    assert qmem._tensor_bytes((a, (b[:, :1], None, 3.0)), cpu) == 8400
    assert qmem._tensor_bytes((a, b), torch.device("cuda:0")) == 0


def test_reading_never_below_input_and_output(fake):
    """Calls whose memory is hidden from the reader (it reads 0, as when f
    reuses memory the process holds) still read the bytes of their input and
    output, which are alive when the reading is taken: 16 bytes per point for
    this f. So no reading is 0, and memory that is only input and output is
    measured exactly."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=0, unread_below=10**9)
    reader = fake.solver._peak_memory_reader()
    for n in (1, 1000, 10**5):
        M, _ = fake.solver._measure_call(
            f, (), NODE.unsqueeze(0), n, reader.reset, reader.read, 0.9
        )
        assert 16 * n == M * GB


def test_fit_uses_reliable_readings_only(fake, monkeypatch):
    """popcornn's width-8 integrand, 552 B * N^2, read with noise below the
    floor: every reading under 4 MiB comes out as 2 MiB (an over-read; such
    2 MiB steps were seen on Linux). Fitting those would move memory from the
    N^2 term into a linear term and underestimate large batches; only
    readings at or above the floor are fitted, so d is exactly 552 and the
    batch is the largest that fits."""
    d = 552

    def reading(n):
        return d * n * n if d * n * n >= 4 * 2**20 else 2 * 2**20

    stub_calls(fake.solver, monkeypatch, reading)
    fit = benchmark(fake.solver, None)
    assert fit == pytest.approx((0, 0, d), abs=1e-3)
    assert fake.solver._get_max_f_evals(0.9) == largest_fitting(lambda n: d * n * n)


def test_worst_case_through_largest_n_reliable_reading(fake, monkeypatch):
    """Before three readings are reliable, the next call is sized by the worst
    case N^2 through the reliable reading with the largest N, not the largest
    reading. Readings: 100 MiB at N = 1, then 50 MiB at N = 10 (a
    hidden-memory pattern), with 100 GB usable before the first call and 1 GB
    (a 768 MiB budget) afterwards. Through the reading at N = 10 the next call
    is 10 * sqrt(768 / 50) = 39 points; through the 100 MiB reading it would
    be sqrt(7.68) = 2, under 2 * 10, and the search would stop."""
    readings = {1: 100 * 2**20, 10: 50 * 2**20}
    calls = stub_calls(
        fake.solver,
        monkeypatch,
        lambda n: readings.get(n, 50 * 2**20 * (n / 10) ** 2),
    )
    monkeypatch.setattr(
        fake.solver,
        "_get_usable_memory",
        lambda total_mem_usage: USABLE_GB if not calls else 1.0,
    )
    benchmark(fake.solver, None)
    assert calls[:3] == [1, 10, 39]


def test_convergence_needs_three_reliable_readings(fake):
    """Memory 1024 B * N: the cost per point is constant from N = 1, which
    used to end the search at N = 100 with three readings of 1 KiB to
    100 KiB, all noise on a CPU. The convergence stop now needs the last three
    readings at or above the 4 MiB floor (from N = 1e4 on), so the search ends
    at N = 1e6."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=1024)
    fit = benchmark(fake.solver, f)
    assert f.calls == [10**k for k in range(7)]
    assert fit[1] == pytest.approx(1024)


def test_slow_call_does_not_stop_before_three_reliable_readings(fake, caplog):
    """Memory 1 MiB + 4 KiB * N, calls from N = 100 on taking 2 s: at
    N = 100 the reading (1.4 MiB) is below the 4 MiB floor, so the 1 s limit
    does not end the search there. The 1 s limit is extended one doubling of
    N at a time, within the 60 s total, until three readings are reliable
    (from N = 768 on); then the 2 s call ends the search, and the fit uses
    the reliable readings only."""
    a, b = 2**20, 4096
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b, slow_from=100)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f)
    assert f.calls[:3] == [1, 10, 100]
    reliable = [n for n, m in zip(f.calls, f.used, strict=True) if m >= 4 * 2**20]
    assert len(reliable) == 3
    assert f.calls[-1] == reliable[-1]
    assert f.calls[3] == 200  # one doubling: the time extension
    assert fake.clock.now <= 60
    assert "stopped because an evaluation took longer than 1.0 s" in caplog.text
    assert fit[:2] == pytest.approx((a, b), rel=1e-6)


def test_total_time_limit_raises_before_three_reliable_readings(fake):
    """Calls of 7 s reading 64 B per point, far below the floor: the 1 s
    limit is extended one doubling at a time while each call is predicted to
    finish within the 60 s, up to N = 64 (49 s). The call at N = 128 then
    takes 20 s instead of the predicted 7, so the measurement passes 60 s
    with no reliable reading. It raises, saying what happened, listing every
    reading (with which ones were below the floor) and the elapsed time, and
    asking for max_batch."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=64)

    def timed(t, *args):
        fake.clock.now += 7.0 if t.shape[0] < 128 else 20.0
        return f(t, *args)

    with pytest.raises(RuntimeError) as info:
        benchmark(fake.solver, timed)
    message = str(info.value)
    assert f.calls == [1, 2, 4, 8, 16, 32, 64, 128]
    assert fake.clock.now == 69
    assert "took longer than its 60 s limit before the readings allowed" in message
    assert "limit_benchmark_time=False" in message
    assert "below the noise floor" in message
    assert "N=128: 0.01 MiB in 20.00 s" in message
    assert "Specify max_batch" in message
    assert fake.solver.f_mem_coeffs is None


def test_next_call_predicted_too_long_raises(fake):
    """Time 0.05 s * N^2 and memory 64 B per point (below the floor): before
    three readings are reliable the 1 s limit is extended one doubling at a
    time, but only for a call predicted to finish within what is left of the
    60 s. When the next doubling is predicted to take longer, the benchmark
    raises before making it, so no call ran past the limit."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=64, time_per_point_sq=0.05)
    with pytest.raises(
        RuntimeError, match="is predicted to take longer than the"
    ) as info:
        benchmark(fake.solver, f)
    assert "limit_benchmark_time=False" in str(info.value)
    assert fake.clock.now <= 60
    assert max(f.seconds) <= 60


def test_total_time_limit_after_fit_ready_uses_fit(fake, caplog):
    """Every reading reliable (8 MiB fixed cost) and growing (64 KiB per
    point), so the fit is ready at N = 100. The call at N = 1000, predicted
    to be instant, takes 70 s: the measurement ends past 60 s with the fit
    ready, so the benchmark does not raise; it stops, says why, and uses the
    fit."""
    a, b = 8 * 2**20, 64 * 1024
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b)

    def timed(t, *args):
        fake.clock.now += 70.0 if t.shape[0] == 1000 else 0.0
        return f(t, *args)

    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, timed)
    assert f.calls == [1, 10, 100, 1000]
    assert "stopped because the measurement took longer than 60 s" in caplog.text
    assert fit[:2] == pytest.approx((a, b), rel=1e-6)


@pytest.mark.parametrize(
    ("counts_reuse", "fourth"), [(True, 8), (False, 40)], ids=["gpu-or-trim", "rss"]
)
def test_step_is_2x_after_slow_call_only_where_reuse_counted(
    fake, monkeypatch, counts_reuse, fourth
):
    """Every call takes 0.2 s (more than 0.1 s). The first steps are set by
    the doubling allowance (1, 2, 4). From there the time fit is flat, and the
    step is 2x on a reader that counts memory f reuses (a GPU, or Linux with
    malloc_trim), 10x on one that does not, where a 2x larger call would read
    low because it reuses much of the previous call's memory unseen."""
    set_reader(fake.solver, monkeypatch, counts_reuse=counts_reuse)
    f = CostIntegrand(fake.memory, fake.clock, fixed=8 * 2**20, time_fixed=0.2)
    benchmark(fake.solver, f)
    assert f.calls[:4] == [1, 2, 4, fourth]


def test_fast_call_after_slow_ones_steps_10x(fake, monkeypatch):
    """Only the latest call sets the step: after calls of 0.2 s at N = 1, 2
    and 4, a fast call at N = 8 is followed by a 10x step (to 80), even on a
    reader that counts reused memory."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    f = CostIntegrand(fake.memory, fake.clock, fixed=8 * 2**20)

    def timed(t, *args):
        fake.clock.now += 0.2 if t.shape[0] < 8 else 0.0
        return f(t, *args)

    benchmark(fake.solver, timed)
    assert f.calls[:5] == [1, 2, 4, 8, 80]


def test_small_memory_bounds_calls_before_any_reliable_reading(fake, monkeypatch):
    """300 KiB per point with 32 MiB usable (a 24 MiB budget per call): the
    first readings are below the 4 MiB floor, yet each next call is still
    sized by the worst case N^2 through the latest reading, so no call
    exceeds the budget (10x steps would call 10 points, 2.9 MiB, then 100,
    29 MiB). The budget ends the search with fewer than three reliable
    readings, so the batch is the cautious one and fits."""
    usable = 32 * 2**20 / GB
    monkeypatch.setattr(
        fake.solver, "_get_usable_memory", lambda total_mem_usage: usable
    )
    b = 300 * 1024
    f = CostIntegrand(fake.memory, fake.clock, per_point=b)
    benchmark(fake.solver, f)
    budget = qmem._MEMORY_BUDGET * usable * GB
    assert max(f.used) <= budget
    assert b * fake.solver._get_max_f_evals(0.9) <= budget


@pytest.mark.skipif(
    not os.access("/proc/self/clear_refs", os.W_OK), reason="needs Linux clear_refs"
)
def test_real_linux_malloc_trim_recovers_reused_memory():
    """With the real Linux reader and glibc: f allocates 200 blocks of
    512 KiB, writes them, frees them and returns a small tensor. glibc keeps
    freed heap blocks for reuse, so a repeated call reads almost nothing
    unless malloc_trim(0) returns them before the reading. First, one freed
    16 MiB block raises glibc's mmap threshold (to that block's size, up to
    32 MiB; glibc malloc/malloc.c), so the 512 KiB blocks come from the heap.
    The test first shows the low reading with the trim disabled; if this
    allocator does not retain the memory, there is nothing to test and it is
    skipped."""
    solver = make_solver_for_unit_test()
    reader = solver._peak_memory_reader()
    if not reader.counts_reuse:
        pytest.skip("this C library has no malloc_trim")
    clear_refs = qmem.Path("/proc/self/clear_refs")
    no_trim = reader._replace(reset=lambda: clear_refs.write_text("5"))
    big = torch.ones(2 * 2**20, dtype=torch.float64)  # 16 MiB
    del big  # freed: raises the mmap threshold

    def f(t, *args):
        blocks = [torch.ones(2**16, dtype=torch.float64) for _ in range(200)]
        del blocks  # 100 MiB written, then freed
        return torch.zeros((t.shape[0], 1), dtype=t.dtype)

    def measure(r):
        M, _ = solver._measure_call(f, (), NODE.unsqueeze(0), 1, r.reset, r.read, 0.9)
        return M * GB

    first = measure(reader)
    low = measure(no_trim)
    if low > 0.5 * first:
        pytest.skip("the allocator returned the blocks itself: no reuse to hide")
    assert measure(reader) >= 0.8 * first


# ---------------------------------------------------------------------------
# Out-of-memory errors during the measurement
# ---------------------------------------------------------------------------


def test_oom_after_n_100_keeps_measured_fit(fake, monkeypatch, caplog):
    """An out-of-memory error at N = 1000 after a successful N = 100 keeps
    the fit of the calls that succeeded and caps the batch at
    _OOM_BACKOFF * 1000 = 750 for the rest of the integrate() call. The end
    message reports the capped batch, not the ~1.2e6 points the fit allows.
    The 8 MiB fixed cost makes every reading reliable, and 64 KiB per point
    makes the fit ready at N = 100. The reader counts reused memory, so the
    largest test call (100) does not cap the batch."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    a, b = 8 * 2**20, 64 * 1024
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
def test_oom_at_small_n_caps_batch(
    fake, monkeypatch, caplog, time_fixed, oom_from, cap
):
    """An out-of-memory error on the second call caps the batch at
    int(0.75 N): 7 points at N = 10, and 1 point at N = 2, reached by doubling
    because the first call takes 0.5 s. One reliable reading exists (8 MiB at
    N = 1), so the stored fit is the cautious one, the worst case N^2 through
    it, and the end message says so. The reader counts reused memory, so only
    the out-of-memory error caps the batch."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    a = 8 * 2**20
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        fixed=a,
        per_point=1024,
        time_fixed=time_fixed,
        oom_from=oom_from,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f, take_gradient=True)
    assert f.calls == [1, oom_from]
    assert fit == pytest.approx((0, 0, safety(True) * (a + 1024)))
    assert fake.solver._oom_max_batch == cap
    assert fake.solver.get_max_batch() == cap
    assert "(1 reached the noise floor" in caplog.text


def test_oom_without_reliable_reading_raises(fake):
    """An out-of-memory error at N = 10 when the only reading (1 KiB at N = 1)
    is below the noise floor: there is nothing to size a batch from, so the
    benchmark raises and asks for max_batch."""
    f = CostIntegrand(fake.memory, fake.clock, per_point=1024, oom_from=10)
    with pytest.raises(RuntimeError, match="No test call of f used at least"):
        benchmark(fake.solver, f)


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
    set_reader(fake.solver, monkeypatch, E=E)
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
@pytest.mark.parametrize("trim", [True, False], ids=["glibc", "no-malloc_trim"])
@pytest.mark.parametrize(
    ("online", "allowed"), [(12, 12), (256, 256), (256, 4)], ids=["12", "256", "256@4"]
)
def test_linux_reader(fake, monkeypatch, online, allowed, trim):
    """The Linux reader (``/proc`` faked, so this runs on any system with
    os.sysconf):

    - ``read`` returns the peak VmHWM from /proc/self/status. Regression:
      ``getrusage`` reports a stuck 50 GB peak here, as it does in a worker
      started by a larger process; the old reader used ``ru_maxrss``.
    - ``reset`` calls glibc's ``malloc_trim(0)``, when the C library has it,
      before resetting the peak, so memory f reuses is counted; the reader
      then counts reused memory (steps may be 2x). Without it (musl) it does
      not.
    - E = 2 * (CPUs the process may run on) * max(32, 2 * online CPUs) pages:
      the kernel's counters hold up to max(32, 2 * online) pages on every CPU
      the process ran on, and a reading is the difference of two counts. On a
      256-CPU node pinned to 4 CPUs that is 16 MiB with 4 KiB pages, not
      1 GiB.
    """
    events = []

    class FakeProc:
        """/proc/self: clear_refs records writes; status reports the peak."""

        def __init__(self, path):
            self.path = path

        def write_text(self, text):
            events.append(("clear_refs", text))

        def read_text(self):
            assert self.path == "/proc/self/status"
            return f"VmRSS:\t1 kB\nVmHWM:\t{fake.memory.rss // 1024} kB\n"

    class FakeLibc:
        """A C library with or without malloc_trim."""

        def __getattr__(self, name):
            if name == "malloc_trim" and trim:

                def malloc_trim(pad):
                    events.append(("malloc_trim", pad))
                    return 1

                return malloc_trim
            raise AttributeError(name)

    resource = pytest.importorskip("resource")
    monkeypatch.setattr(
        resource, "getrusage", lambda who: types.SimpleNamespace(ru_maxrss=50 * 2**20)
    )
    monkeypatch.setattr(qmem, "Path", FakeProc)
    monkeypatch.setattr(qmem.ctypes, "CDLL", lambda name: FakeLibc())
    monkeypatch.setattr(qmem.os, "cpu_count", lambda: online)
    monkeypatch.setattr(
        qmem.os, "sched_getaffinity", lambda pid: set(range(allowed)), raising=False
    )
    reader = fake.solver._peak_memory_reader()
    page = os.sysconf("SC_PAGE_SIZE")
    assert pytest.approx(2 * allowed * max(32, 2 * online) * page) == reader.E * GB
    assert reader.floor * GB == 4 * 2**20
    assert reader.counts_reuse == trim
    events.clear()
    reader.reset()
    assert events == ([("malloc_trim", 0)] if trim else []) + [("clear_refs", "5")]
    assert reader.read() == fake.memory.rss // 1024 * 1024


@pytest.mark.skipif(
    not os.access("/proc/self/clear_refs", os.W_OK), reason="needs Linux clear_refs"
)
def test_real_linux_reader_smoke(monkeypatch):
    """With the real reader, an allocation of 200 MiB + 1 MiB per point that f
    frees before returning is measured. E is made negligible (one CPU, 256 KiB), so the fit is the
    reading itself. The allocation is a single-threaded memset (bytearray): its pages are
    counted on one or a few CPUs, so the per-CPU counter error is a few MiB;
    a multithreaded fill such as torch.ones can read low by most of it."""
    monkeypatch.setattr(qmem.os, "cpu_count", lambda: 1)
    monkeypatch.setattr(qmem.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    clock = FakeClock()
    monkeypatch.setattr(qmem, "time", clock)
    solver = make_solver_for_unit_test()

    def heavy(t, *args):
        # 200 MiB plus 1 MiB per point, touched, freed before returning (the
        # 1 MiB per point makes the readings grow: equal readings would raise)
        bytearray((200 + t.shape[0]) * 2**20)
        clock.now += 2.0  # slower than 1 s: stops once 3 readings are reliable
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
    """get_max_batch(f=...) logs the start warning (the measurement can be
    slow: run it once and pass the value; it may not suit every machine) and
    the end warning (time, reason, the fit and a max_batch to pass to
    integrate()). integrate() never runs the benchmark, so it logs neither,
    with or without max_batch."""
    clock = FakeClock()
    monkeypatch.setattr(qmem, "time", clock)

    def slow_sin(t, *args):
        clock.now += 2.0  # stops the search after 3 calls; no real waiting
        return torch.sin(t)

    solver = make_solver_for_unit_test(method_name="gk21")
    # Every reading counts (floor 0), so three calls give a fit. The reader
    # reads nothing, so each reading is the bytes of the input and output,
    # 16 per point: the real resident memory of such small calls is noise,
    # and readings that do not grow would raise.
    set_reader(solver, monkeypatch, floor=0, read=lambda: 0)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        solver.get_max_batch(f=slow_sin, t=T_INIT)
    text = caplog.text
    assert "PadaQuad is measuring how much memory" in text
    assert "This measurement can be slow. Run it once" in text
    assert "may result in an underutilization of your hardware" in text
    assert "devices_memory.html#ug-memory-limitations" in text
    assert "The only way to guarantee" in text
    assert "result.max_batch" not in text
    assert "Memory measurement finished in" in text
    assert "bytes per evaluation" in text
    assert "stopped because" in text
    assert "usable now, max_batch is about" in text
    assert "for take_gradient=True" in text
    assert "to integrate()." in text

    for options in ({}, {"max_batch": 1000}):
        caplog.clear()
        solver = make_solver_for_unit_test(method_name="gk21", **options)
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
    # Every reading counts (floor 0), so three calls give a fit. The reader
    # reads nothing, so each reading is the bytes of the input and output,
    # 16 per point: the real resident memory of such small calls is noise,
    # and readings that do not grow would raise.
    set_reader(solver, monkeypatch, floor=0, read=lambda: 0)
    assert solver.get_max_batch(f=slow, t=T_INIT) >= solver.C
    result = solver.integrate(
        slow,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=True,
        max_batch=solver.get_max_batch(),
    )
    assert float(result.integral.sum()) == pytest.approx(
        torch.sin(T_FINAL).item(), abs=1e-5
    )


@pytest.mark.parametrize(
    ("cost", "test_calls", "max_batch"),
    [
        ({"per_point": 0, "per_point_sq": int(BUDGET / 400)}, [1, 10, 20], 20),
        ({"fixed": 8 * 2**20, "oom_from": 10}, [1, 10], 7),
    ],
    ids=["N^2", "out-of-memory"],
)
def test_integrate_batches_within_measured_max_batch(
    fake, monkeypatch, caplog, cost, test_calls, max_batch
):
    """get_max_batch(f=...): f is called on one point (the warm-up), then by
    the benchmark; integrate() given the measured max_batch then evaluates in
    batches of at most that size, and the integral of f = 1 over [0, 1] is 1.
    For memory BUDGET / 400 * N^2 the batch is the largest that fits,
    sqrt(400) = 20 points; for an out-of-memory error at N = 10 (after one
    reliable reading, 8 MiB at N = 1) it is the cap, 7. Both are below the 44 points (11 subintervals of bosh3's 4 nodes)
    the integration needs, so it takes several batches; the result reports
    the size, converged. The reader counts reused memory, so the largest test
    call does not cap the batch."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    f = CostIntegrand(fake.memory, fake.clock, **cost)
    measured = fake.solver.get_max_batch(f=f, t=T_INIT, take_gradient=False)
    assert f.calls == [1, *test_calls]
    assert measured == max_batch
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        result = fake.solver.integrate(
            f,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=False,
            max_batch=measured,
        )
    assert "PadaQuad is measuring" not in caplog.text
    batches = f.calls[2 + len(test_calls) :]
    assert len(batches) >= 2
    assert max(batches) <= max_batch
    assert float(result.integral.sum()) == pytest.approx(1.0)
    assert (result.max_batch, result.max_batch_converged) == (max_batch, True)


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
    # Warm-up, as integrate() does: a cold first call (kernel loading) can
    # take more than 0.1 s, after which the CUDA reader steps 2x, not 10x.
    f(node.reshape(1, 1))
    calls.clear()
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
    f(node.reshape(1, 1))  # warm-up, as in test_gpu_oom_after_n_100_keeps_measured_cost
    calls.clear()
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


# ---------------------------------------------------------------------------
# Realistic scenarios: heavy and slow integrands, hidden readings, lazy
# loading, flat readings; each docstring names the mutation the test catches
# ---------------------------------------------------------------------------


def test_memory_budget_ends_search_before_time_error(fake, monkeypatch, caplog):
    """A heavy and slow integrand: memory 3 MiB * N^2 and time 1.5 s * N^2,
    with 144 MiB usable (a 108 MiB budget). Calls N = 1 (3 MiB, below the
    floor), 2 and 4 (doubling allowed for the first calls) take 31.5 s. Then
    the next call, 8 points, would need 192 MiB (no fit) and 96 s (more than
    the 28.5 s left). The memory budget ends the search whatever the time, so
    this is the memory stop with two reliable readings: the cautious fit
    (batch 6, the worst case N^2 through 48 MiB at N = 4), not the error that
    the next call would take too long. Catches: raising the time error
    whenever the time bound is below the memory bound (the round-2 condition
    ``n_time < n_mem``), which raised here."""
    usable = 144 * 2**20 / GB
    monkeypatch.setattr(fake.solver, "_get_usable_memory", lambda tmu: usable)
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        per_point=0,
        per_point_sq=3 * 2**20,
        time_per_point_sq=1.5,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, f)
    assert f.calls == [1, 2, 4]
    assert fit == pytest.approx((0, 0, 3 * 2**20))
    assert fake.solver._get_max_f_evals(0.9) == 6
    assert "stopped because the next test would not fit in memory" in caplog.text
    assert "(2 reached the noise floor" in caplog.text


def test_growth_compared_with_previous_reliable_reading(fake, monkeypatch):
    """Readings 8 MiB at N = 1, then 1 MiB at N = 10 (memory hidden from the
    reader, below the floor), then 4 times what growth like N^2 allows from
    the reading at N = 1. The growth check compares with the previous
    reliable reading (N = 1), so the call at N = 97 (the largest N with
    8 MiB N^2 <= 75 GB, the worst case N^2 through N = 1) raises. Comparing
    only with the previous call, when it is reliable, would skip this call (its
    previous reading is unreliable) and end with a cautious batch sized from
    the faster-than-N^2 reading (this happened on a Mac: 7.4 MiB at N = 1000,
    0.0 MiB at N = 3091, then 2,664 MiB at N = 6182 gave a batch needing
    2.7 times the usable memory). Catches: comparing with ``Ms[-2]`` only."""
    readings = {1: 8 * 2**20, 10: 2**20}
    calls = stub_calls(
        fake.solver, monkeypatch, lambda n: readings.get(n, 4 * 8 * 2**20 * n * n)
    )
    with pytest.raises(RuntimeError, match=r"growth like N\^2 allows") as info:
        benchmark(fake.solver, None)
    assert calls == [1, 10, 97]
    assert "bytes at 97 points" in str(info.value)
    assert "8.39e+06 bytes at 1 points" in str(info.value)
    assert fake.solver.f_mem_coeffs is None


@pytest.mark.parametrize(
    ("seconds", "limited", "outcome"),
    [(1.5, True, "fit"), (5.0, True, "error"), (5.0, False, "fit")],
    ids=["1.5s", "5s", "5s-unlimited"],
)
def test_slow_model_doubles_to_floor(fake, monkeypatch, seconds, limited, outcome):
    """A slow model, as an ML potential on a CPU: every call takes the same
    time (longer than the 1 s limit) and uses 1 KiB per point, so readings
    reach the 4 MiB floor only at N = 4096. Before three readings are
    reliable the 1 s limit is extended one doubling of N at a time, on every
    reader (here one that counts reused memory, as Linux with malloc_trim):

    - 1.5 s per call: N = 1, 2, ..., 16384 (15 calls, 22.5 s); the readings
      at 4096, 8192 and 16384 are reliable and the cost per point is
      constant, so the fit is exact.
    - 5 s per call: after N = 2048 (12 calls, 60 s) the next call is
      predicted to end past the 60 s, with no reliable reading, so the
      benchmark raises before making it; the message names
      limit_benchmark_time and max_batch.
    - 5 s per call with ``limit_benchmark_time=False``: the same 15 calls as
      at 1.5 s, 75 s in all, and the exact fit.

    The second case pins a consequence of doubling: a model slower than about
    4 s per call with 1 KiB per point cannot be measured in 60 s. Catches:
    removing the time extension (the 1.5 s case then raises after N = 2);
    ignoring limit_benchmark_time."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    monkeypatch.setattr(fake.solver, "limit_benchmark_time", limited)
    f = CostIntegrand(fake.memory, fake.clock, per_point=1024, time_fixed=seconds)
    if outcome == "fit":
        fit = benchmark(fake.solver, f)
        assert f.calls == [2**k for k in range(15)]
        assert fake.clock.now == pytest.approx(15 * seconds)
        assert fit[1] == pytest.approx(1024)
        return
    with pytest.raises(RuntimeError, match="Specify max_batch") as info:
        benchmark(fake.solver, f)
    assert "limit_benchmark_time=False" in str(info.value)
    assert f.calls == [2**k for k in range(12)]
    assert fake.clock.now <= 60


def test_call_after_failed_benchmark_measures_again(fake):
    """A first get_max_batch(f=...) whose benchmark fails (an out-of-memory
    error at N = 10, the only reading, 1 KiB at N = 1, below the floor) raises,
    but leaves the out-of-memory cap of 7 points on the solver. A later
    measurement runs the benchmark again, and the cap of the failed one does
    not limit its result.
    Catches: not clearing ``_oom_max_batch`` before a measurement."""
    bad = CostIntegrand(fake.memory, fake.clock, per_point=1024, oom_from=10)
    with pytest.raises(RuntimeError, match="No test call of f used at least"):
        fake.solver.get_max_batch(f=bad, t=T_INIT, take_gradient=False)
    assert fake.solver._oom_max_batch == 7
    good = CostIntegrand(fake.memory, fake.clock, fixed=8 * 2**20, per_point=1024)
    measured = fake.solver.get_max_batch(f=good, t=T_INIT, take_gradient=False)
    assert good.calls[1:4] == [1, 10, 100]  # after the one-point check
    assert measured == pytest.approx((BUDGET - 8 * 2**20) / 1024, abs=1)


def test_one_time_cost_of_first_call_absorbed_by_get_max_batch(fake, monkeypatch):
    """A model that loads 48 MiB of weights on its first call and keeps them,
    then uses 1 KiB per point. get_max_batch(f=...) calls f once on one point
    before the benchmark, so the benchmark measures 1 KiB per point and the
    batch is about 75 GB / 1 KiB (the reader counts reused memory, so the
    largest test call does not cap it). Run directly, without that call, the
    48 MiB is the only reliable reading and the cautious batch is 39 points.
    Catches: running the benchmark before the one-point call of f."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)

    def lazy(cost):
        loaded = []

        def f(t, *args):
            if not loaded:
                loaded.append(True)
                fake.memory.rss += 48 * 2**20  # the weights, kept
            return cost(t, *args)

        return f

    cost = CostIntegrand(fake.memory, fake.clock, per_point=1024)
    measured = fake.solver.get_max_batch(f=lazy(cost), t=T_INIT, take_gradient=False)
    assert measured == pytest.approx(BUDGET / 1024, abs=1)
    benchmark(fake.solver, lazy(cost))
    assert fake.solver._get_max_f_evals(0.9) == 39


def test_readings_that_do_not_grow_keep_doubling(fake, monkeypatch, caplog):
    """A slow model with a fixed 8 MiB workspace and 100 bytes per point,
    1.2 s per call, on a reader that counts reused memory (as Linux) and
    reads whole 4 KiB pages: the calls at N = 1, 2 and 4 all read 8 MiB, three
    reliable readings whose fit is flat and would allow 1e9 points per call.
    The fit is not ready until its memory grows by the 4 MiB floor, so the
    1 s limit does not end the search: N doubles up to 65536 (6.25 MiB more
    than at N = 1), and the fit then gives about 100 bytes per point. When an
    out-of-memory error at N = 8 ends the search instead, the stored fit is
    the cautious worst case and the batch is capped below that call (6
    points). Catches: ending the search with a flat fit (round 2 stored it,
    giving max_batch = 1e9; round 3 raised)."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    page = 4096
    calls = stub_calls(
        fake.solver,
        monkeypatch,
        lambda n: (8 * 2**20 + 100 * n) // page * page,
        seconds=lambda n: 1.2,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        fit = benchmark(fake.solver, None)
    assert calls == [2**k for k in range(17)]
    assert "stopped because an evaluation took longer than 1.0 s" in caplog.text
    assert fit[1] == pytest.approx(100, rel=0.05)
    true_batch = (BUDGET - 8 * 2**20) / 100
    assert 0.9 * true_batch <= fake.solver._get_max_f_evals(0.9) <= 1.1 * true_batch

    def out_of_memory_at_8(f, f_args, node_test, N, *rest):
        calls.append(N)
        return None if N >= 8 else ((8 * 2**20 + 100 * N) // page * page / GB, 0.2)

    monkeypatch.setattr(fake.solver, "_measure_call", out_of_memory_at_8)
    benchmark(fake.solver, None)
    assert calls[17:] == [1, 2, 4, 8]
    assert fake.solver.get_max_batch() == 6


def test_first_calls_obey_total_limit(fake, monkeypatch):
    """Time 20 s * N^2: the first call takes 20 s, and doubling N would take
    80 s in the worst case, past the 40 s left of the 60 s. The doubling
    allowed for the first two calls does not override the total limit: the
    benchmark raises after one call, naming limit_benchmark_time. With
    limit_benchmark_time=False it doubles (N = 2 takes 80 s). Catches: the
    first two calls ignoring the 60 s (one review case ran 210 s)."""
    f = CostIntegrand(fake.memory, fake.clock, time_per_point_sq=20.0)
    with pytest.raises(RuntimeError, match="limit_benchmark_time=False"):
        benchmark(fake.solver, f)
    assert f.calls == [1]
    assert fake.clock.now == 20.0

    monkeypatch.setattr(fake.solver, "limit_benchmark_time", False)
    f = CostIntegrand(fake.memory, fake.clock, fixed=8 * 2**20, time_per_point_sq=20.0)
    benchmark(fake.solver, f)
    assert f.calls[:2] == [1, 2]


def test_cubic_growth_caught_after_2x_steps(fake, monkeypatch):
    """Memory 1 MiB * N^3 with 0.2 s per call on a reader that counts reused
    memory, so steps are 2x: N = 1 (1 MiB, below the floor), 2 (8 MiB), 4
    (64 MiB), 8 (512 MiB). Compared only with the previous reading, each
    2x step allows 3 * 4 = 12 times more, and N^3 grows 8 times, so it was
    never caught. Compared with every earlier reliable reading, the call at
    N = 8 exceeds 3 * 16 * 8 MiB = 384 MiB, the bound from N = 2, and raises.
    Catches: comparing only with the previous reliable reading."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    f = CostIntegrand(
        fake.memory,
        fake.clock,
        per_point=0,
        per_point_sq=2**20,
        exponent=3,
        time_fixed=0.2,
    )
    with pytest.raises(RuntimeError, match=r"growth like N\^2 allows") as info:
        benchmark(fake.solver, f)
    assert f.calls == [1, 2, 4, 8]
    assert "bytes at 2 points" in str(info.value)


@pytest.mark.parametrize("counts_reuse", [False, True], ids=["rss", "gpu-or-trim"])
def test_batch_capped_at_largest_test_call_where_reuse_is_missed(
    fake, monkeypatch, caplog, counts_reuse
):
    """Memory 1 MiB + 1 KiB N: the cost per point stops changing at N = 1e7,
    where the search ends, while the fit allows about 7.9e7 points. On a
    reader that misses reused memory (macOS, Windows) the readings can be far
    too low, so the batch is capped at the largest test call, 1e7 (it ran);
    the start warning says so. A reader that counts reused memory keeps the
    batch the fit allows. Catches: no cap; capping on every reader."""
    set_reader(fake.solver, monkeypatch, counts_reuse=counts_reuse)
    a, b = 2**20, 1024
    f = CostIntegrand(fake.memory, fake.clock, fixed=a, per_point=b)
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, f)
    assert f.calls[-1] == 10**7
    fitted = fake.solver._get_max_f_evals(0.9)
    assert fitted == pytest.approx((BUDGET - a) / b, rel=1e-6)
    expected = fitted if counts_reuse else 10**7
    assert fake.solver.get_max_batch() == expected
    assert f"max_batch is about {expected} " in caplog.text
    assert ("capped at the largest call" in caplog.text) == (not counts_reuse)


def test_batch_cap_at_least_one_subinterval(monkeypatch):
    """A slow f whose search ends at N = 4 (three reliable readings, then the
    1 s limit) on a reader that misses reused memory: the cap is not the
    largest test call, 4, but the C = 23 nodes of one gk21 subinterval, the
    smallest batch the take_gradient=True path can evaluate. Catches: a cap
    below one subinterval, which made that path fail its assertion."""
    solver = make_solver_for_unit_test(method_name="gk21")
    monkeypatch.setattr(solver, "_get_usable_memory", lambda total_mem_usage: USABLE_GB)
    set_reader(solver, monkeypatch, floor=0, counts_reuse=False)
    calls = stub_calls(solver, monkeypatch, lambda n: 2**20 * n, seconds=lambda n: 2.0)
    benchmark(solver, None)
    assert calls == [1, 2, 4]
    assert solver.get_max_batch() == solver.C == 23


def test_limit_benchmark_time_is_a_solver_option():
    """limit_benchmark_time is a constructor option (True by default), also
    accepted by padaquad.integrate, which passes it to the constructor."""
    assert adaptive_quadrature().limit_benchmark_time is True
    assert adaptive_quadrature(limit_benchmark_time=False).limit_benchmark_time is False
    result = integrate(
        torch.cos,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        device="cpu",
        max_batch=1000,
        limit_benchmark_time=False,
    )
    assert result.converged


@pytest.mark.parametrize("slow_at_8", [False, True], ids=["predicted", "elapsed"])
def test_time_limit_raises_while_readings_do_not_grow(fake, monkeypatch, slow_at_8):
    """Every reading is reliable (8 MiB fixed cost) but grows only 100 bytes
    per point, so three or more reliable readings do not make the fit ready.
    Calls of 5.5 s double N (the reader counts reused memory).

    - predicted: after 10 calls (55 s) the next one is predicted to end at
      60.5 s, so the benchmark raises before making it.
    - elapsed: the call at N = 8 takes 60 s, so the measurement passes the
      60 s with 4 reliable readings and no fit, and raises.

    Both messages name limit_benchmark_time. Catches: testing the number of
    reliable readings (three) instead of the fit in either time error, which
    stored the cautious fit instead of raising."""
    set_reader(fake.solver, monkeypatch, counts_reuse=True)
    f = CostIntegrand(fake.memory, fake.clock, fixed=8 * 2**20, per_point=100)

    def timed(t, *args):
        fake.clock.now += 60.0 if slow_at_8 and t.shape[0] == 8 else 5.5
        return f(t, *args)

    with pytest.raises(RuntimeError, match="limit_benchmark_time=False") as info:
        benchmark(fake.solver, timed)
    if slow_at_8:
        assert f.calls == [1, 2, 4, 8]
        assert "took longer than its 60 s limit" in str(info.value)
        assert "(4 reached the noise floor" in str(info.value)
    else:
        assert f.calls == [2**k for k in range(10)]
        assert "is predicted to take longer than" in str(info.value)
        assert "(10 readings reached the noise floor" in str(info.value)


def test_cost_convergence_waits_for_fit_ready(fake, monkeypatch, caplog):
    """A GPU-like reader (10 KB floor, counts reused memory) and calls of
    0.2 s, so N doubles. The call at N = 2 reads 1 MiB (a workspace f
    allocates only in that call, as for a kernel first used at two points;
    integrate()'s one-point call absorbs only what N = 1 allocates); every
    other call reads 1 KiB per point. From N = 16 on the readings are
    reliable and the cost per point is constant, so the convergence stop
    alone would end the search at N = 64. The fit uses every reliable
    reading, the 1 MiB one included, and grows by less than the floor there:
    it is not ready, so the stored fit would be the cautious worst case (a
    batch of 70,944 points, the end message saying that four readings reached
    the floor and a fit needs three). The search goes on until the fit is
    ready, at N = 512, and the end message reports that fit. Catches: a
    convergence stop that does not require the fit to be ready."""
    set_reader(fake.solver, monkeypatch, floor=10_000, counts_reuse=True)
    calls = stub_calls(
        fake.solver,
        monkeypatch,
        lambda n: 2**20 if n == 2 else 1024 * n,
        seconds=lambda n: 0.2,
    )
    with caplog.at_level(logging.WARNING, logger=qmem.__name__):
        benchmark(fake.solver, None)
    assert calls == [2**k for k in range(10)]
    assert "stopped because the cost per evaluation stopped changing" in caplog.text
    assert "did not allow a fit" not in caplog.text


def test_out_of_memory_retry_shrinks_the_failed_batch(fake, monkeypatch):
    """integrate() without max_batch: the test call of 10 points (stubbed, so
    f is not called) gives a batch size of 10, not converged, so the first
    pass of 44 points (11 subintervals of bosh3's 4 nodes) runs whole. f runs
    out of memory above 29 points; each retry shrinks the batch that failed,
    44 -> 33 -> 25 (then the 19 points left), and 25 is the converged batch
    size. Catches: a retry that does not shrink the failed batch itself."""
    stub_calls(fake.solver, monkeypatch, lambda n: 2**20 * n)
    f = CostIntegrand(fake.memory, fake.clock, oom_from=30)
    result = fake.solver.integrate(
        f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
    )
    assert f.calls == [1, 44, 33, 25, 19]  # after the one-point check
    assert (result.max_batch, result.max_batch_converged) == (25, True)
    assert float(result.integral.sum()) == pytest.approx(1.0)


def test_reuse_cap_never_raises_out_of_memory_cap(monkeypatch):
    """On a reader that misses reused memory, a heavy f (8 MiB per call) runs
    out of memory at N = 10 after one reading at N = 1. The out-of-memory cap
    is int(0.75 * 10) = 7 points; the cap at the largest test call is raised
    to one gk21 subinterval, 23 points, which is more than the call that
    failed. The smaller of the two holds: the batch is 7. Catches: the cap at
    the largest test call replacing the out-of-memory cap instead of taking
    the minimum."""
    solver = make_solver_for_unit_test(method_name="gk21")
    monkeypatch.setattr(solver, "_get_usable_memory", lambda total_mem_usage: USABLE_GB)
    set_reader(solver, monkeypatch, counts_reuse=False)
    calls = []

    def measure_call(f, f_args, node_test, N, *rest):
        calls.append(N)
        return None if N >= 10 else (8 * 2**20 / GB, 0.0)

    monkeypatch.setattr(solver, "_measure_call", measure_call)
    benchmark(solver, None)
    assert calls == [1, 10]
    assert solver.get_max_batch() == 7 < solver.C


# ---------------------------------------------------------------------------
# get_max_batch(f=..., t=...): the user runs the benchmark and passes the result
# as max_batch. The benchmark is faked: these tests check the plumbing.
# ---------------------------------------------------------------------------

_T = torch.tensor([0.0], dtype=torch.float64)


def _fake_benchmark(solver, monkeypatch, measured=1234):
    """Replace the benchmark by a fake that records its calls and returns
    ``measured``; it also sets a memory fit so integrate() can size batches."""
    calls = []

    def fake(f, node_test, take_gradient, total_mem_usage, f_args=()):
        calls.append({"take_gradient": take_gradient, "f_args": f_args})
        solver.f_mem_coeffs = (0.0, 1e-6, 0.0)
        return measured

    monkeypatch.setattr(solver, "_setup_memory_checks", fake)
    return calls


def test_get_max_batch_measures_when_given_f(monkeypatch):
    """With f and t, get_max_batch runs the benchmark once, with the given
    take_gradient and f_args, clears a stale out-of-memory cap first, and
    returns the measured batch size."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu")
    calls = _fake_benchmark(solver, monkeypatch)
    solver._oom_max_batch = 3
    seen = []

    def f(t, k):
        seen.append(len(t))
        return k * torch.exp(t)

    measured = solver.get_max_batch(f=f, t=_T, take_gradient=False, f_args=(2.0,))
    assert measured == 1234
    assert calls == [{"take_gradient": False, "f_args": (2.0,)}]
    assert seen == [1]  # the one-point output check (warm-up)
    assert solver._oom_max_batch is None


def test_measured_max_batch_skips_the_benchmark_in_integrate(monkeypatch):
    """A batch size measured once and passed as max_batch skips the benchmark
    and is returned as result.max_batch."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu")
    calls = _fake_benchmark(solver, monkeypatch)
    m = solver.get_max_batch(f=torch.exp, t=_T, take_gradient=False)
    out = solver.integrate(
        torch.exp,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        max_batch=m,
        take_gradient=False,
    )
    assert len(calls) == 1
    assert out.max_batch == m


def test_get_max_batch_with_f_ignores_constructor_max_batch(monkeypatch):
    """A constructor max_batch does not stop the measurement the caller asked for."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu", max_batch=7)
    calls = _fake_benchmark(solver, monkeypatch)
    assert solver.get_max_batch(f=torch.exp, t=_T) == 1234
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"f": torch.exp},
        {"f": torch.exp, "t": torch.tensor([[0.0]])},
        {"f": torch.exp, "t": 0.0},
        {"f": torch.exp, "t": torch.tensor([0.0], dtype=torch.float16)},
        {"f": torch.exp, "t": torch.tensor([0])},
        {"f": torch.exp, "t": _T, "total_mem_usage": 1.5},
        {"f": lambda t: torch.exp(t)[:, 0], "t": _T},
    ],
    ids=[
        "no_t",
        "t_not_1d",
        "t_not_a_tensor",
        "t_float16",
        "t_integer",
        "total_mem_usage_above_1",
        "output_not_2d",
    ],
)
def test_get_max_batch_with_f_validates_its_arguments(kwargs, monkeypatch):
    """f needs a float32 or float64 point t of shape [T] (the dtypes the solver
    integrates in), needs
    0 < total_mem_usage <= 1, and must return values of shape [N, D]. The fake
    benchmark keeps a missing check from running the real one."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu")
    calls = _fake_benchmark(solver, monkeypatch)
    with pytest.raises(ValueError, match=r"get_max_batch|total_mem_usage|shape"):
        solver.get_max_batch(**kwargs)
    assert calls == []


def test_get_max_batch_measures_in_the_dtype_of_t(monkeypatch):
    """The benchmark runs in t's dtype, not in whatever dtype an earlier call
    left the solver in, and the solver's dtype is not changed."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu")
    f32 = {"dtype": torch.float32}
    solver.integrate(
        torch.exp,
        mesh_init=torch.tensor([0.0], **f32),
        mesh_final=torch.tensor([1.0], **f32),
        max_batch=10_000,
        take_gradient=False,
    )
    assert solver.dtype == torch.float32
    seen = []

    def fake(f, node_test, take_gradient, total_mem_usage, f_args=()):
        seen.append(node_test.dtype)
        return 1234

    monkeypatch.setattr(solver, "_setup_memory_checks", fake)
    solver.get_max_batch(f=torch.exp, t=_T)
    assert seen == [torch.float64]
    assert solver.dtype == torch.float32


@pytest.mark.parametrize("value", [0.0, -0.5, 1.5], ids=["zero", "negative", "above_1"])
@pytest.mark.parametrize("where", ["constructor", "integrate", "get_max_batch"])
def test_total_mem_usage_outside_0_1_raises(where, value):
    """total_mem_usage is a fraction in (0, 1]: every entry point raises
    ValueError (integrate used an assert, which python -O removes; the
    constructor and get_max_batch() did not check it)."""
    bounds = {"mesh_init": T_INIT, "mesh_final": T_FINAL, "max_batch": 10_000}
    with pytest.raises(ValueError, match="total_mem_usage"):
        if where == "constructor":
            adaptive_quadrature("uniform", "gk21", device="cpu", total_mem_usage=value)
        elif where == "integrate":
            solver = adaptive_quadrature("uniform", "gk21", device="cpu")
            solver.integrate(torch.exp, total_mem_usage=value, **bounds)
        else:
            adaptive_quadrature("uniform", "gk21", device="cpu").get_max_batch(value)


def test_get_max_batch_takes_no_default_max_batch():
    """default_max_batch was removed: it only returned the value passed in
    (lowered to the out-of-memory cap), which callers can pass to integrate
    themselves."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu", max_batch=10_000)
    with pytest.raises(TypeError, match="default_max_batch"):
        solver.get_max_batch(default_max_batch=5)
