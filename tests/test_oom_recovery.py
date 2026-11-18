"""Tests for the out-of-memory (OOM) catch-and-retry path.

When an integrand ``f`` raises ``torch.OutOfMemoryError`` mid-evaluation, the
split evaluation handlers
(``_evaluate_f_on_split_nodes`` / ``_evaluate_f_on_split_residual_nodes``) are
supposed to shrink ``max_batch`` and retry the failed batch, storing the new
size as the call's batch size ``self._batch``, converged, so every later batch
stays clamped. A later call keeps it only when it is passed
``result.state``. The reduction
repeats until the batch fits or a single
evaluation (``max_batch == 1``) still fails, at which point an informative error
is raised.

Every analytic integrand in the suite has a tiny memory footprint, so this path
was never exercised by the existing tests. These tests **simulate** OOM with a
purpose-built integrand (``OOMIntegrand``) that raises ``torch.OutOfMemoryError``
whenever its batch exceeds a configurable ``limit`` and records every batch size
it was asked to evaluate — so we can watch ``max_batch`` shrink and confirm it is
driven by ``_batch``.

Two layers:
  * ``TestSplitHandlersOOM`` — direct calls to both split handlers, parametrized
    over the handler, mirroring the direct-call unit tests in
    ``test_evaluate_f_on_mesh.py``.
  * ``TestIntegrateOOMRecovery`` — full ``.integrate()`` runs (the real user
    path), across ``conserve_memory`` (split vs. split-residual).

The ``take_gradient=True`` path (``_evaluate_f_on_full_nodes``, its backward
pass and restart) is tested in ``test_memory_state.py``.
"""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import (
    T_FINAL,
    T_INIT,
    make_solver_for_unit_test,
    make_uniform_solver,
)

# ---------------------------------------------------------------------------
# Simulated-OOM integrand
# ---------------------------------------------------------------------------


class OOMIntegrand:
    """``f(t) = 1`` that raises ``torch.OutOfMemoryError`` above a batch size.

    Raises when the batch (number of flattened nodes) exceeds ``limit``,
    otherwise returns a finite ``[N, 1]`` tensor of ones. Records the batch size
    of *every* call in ``batch_sizes`` and counts the OOMs in ``n_oom`` so tests
    can inspect how ``max_batch`` was reduced.

    ``max_calls`` is a safety valve: if the handler ever retries more times than
    this it means the reduction is not strictly decreasing (e.g. a naive
    ``round`` that sticks at 2 forever). We raise ``AssertionError`` to turn what
    would otherwise be an infinite loop / hang into a clean test failure.

    ``fail`` replaces the simulated ``torch.OutOfMemoryError`` with another way
    of running out of memory: a callable that raises (see ``FAILURE_MODES``).

    ``value`` replaces ``f(t) = 1`` by another integrand, called on the batch.
    A constant integrand cannot show nodes paired with the wrong values.
    """

    def __init__(self, limit: int, max_calls: int = 2000, fail=None, value=None):
        self.limit = limit
        self.max_calls = max_calls
        self.fail = fail
        self.value = value
        self.batch_sizes: list[int] = []
        self.n_oom = 0

    def __call__(self, t, *args):
        n = t.shape[0]
        self.batch_sizes.append(n)
        if len(self.batch_sizes) > self.max_calls:
            raise AssertionError(
                f"OOMIntegrand called {len(self.batch_sizes)} times "
                f"(> max_calls={self.max_calls}); reduction is not converging."
            )
        if n > self.limit:
            self.n_oom += 1
            if self.fail is not None:
                self.fail()
            raise torch.OutOfMemoryError(f"simulated OOM at batch {n}")
        if self.value is not None:
            return self.value(t)
        return torch.ones((n, 1), dtype=t.dtype, device=t.device)


def _raise_memory_error():
    raise MemoryError("simulated numpy-style MemoryError")


def _raise_cublas_alloc_failed():
    # torch 2.8's message when cuBLAS cannot allocate its handle (TORCH_CUDABLAS_CHECK).
    raise RuntimeError(
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"
    )


# Every out-of-memory form the batch loop must recover from: the GPU error, a
# real refused CPU allocation (a plain RuntimeError naming DefaultCPUAllocator;
# 8 PB is refused instantly and allocates nothing), Python's MemoryError, and a
# CUDA library's allocation status (a plain RuntimeError without "out of memory").
FAILURE_MODES = {
    "torch_oom": None,
    "cpu_refusal": lambda: torch.empty(10**15, dtype=torch.float64),
    "memory_error": _raise_memory_error,
    "cublas_alloc_failed": _raise_cublas_alloc_failed,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UNIT_METHOD = "bosh3"  # C = 4


def _make_mesh(n_panels: int) -> torch.Tensor:
    """Uniform mesh of ``n_panels`` panels on [0, 1] as a [M, 1] barrier array."""
    return torch.linspace(0.0, 1.0, n_panels + 1, dtype=torch.float64).unsqueeze(-1)


def _call_handler(solver, handler: str, f, mesh, step_idxs, max_batch):
    """Invoke the requested split handler with correct args for its signature."""
    if handler == "split":
        return solver._evaluate_f_on_split_nodes(
            f, (), mesh, step_idxs, max_batch=max_batch
        )
    return solver._evaluate_f_on_split_residual_nodes(
        f,
        (),
        mesh,
        step_idxs,
        max_batch=max_batch,
        max_mesh_steps=len(step_idxs),
        split_node_state=(None, None, None, None),
    )


def _expected_reduction_sequence(start: int, limit: int) -> list[int]:
    """Batch sizes a correct handler attempts for the first chunk, in order.

    Mirrors the intended reduction rule
    ``max(1, int(min(round(0.75 * prev), prev - 1)))``: begin at ``start`` and
    keep shrinking while the batch exceeds ``limit``. Stops at the first size
    ``<= limit`` (which succeeds). Assumes ``limit >= 1`` (so it terminates
    without hitting the ``max_batch == 1`` raise).
    """
    seq = [start]
    cur = start
    while cur > limit:
        cur = max(1, int(min(round(0.75 * cur), cur - 1)))
        seq.append(cur)
    return seq


HANDLERS = ["split", "residual"]


def _wave(t):
    """sin(40 t) + t: its integral over [0, 1] is (1 - cos 40) / 40 + 1 / 2."""
    return torch.sin(40 * t) + t


WAVE_INTEGRAL = (1 - math.cos(40.0)) / 40 + 0.5


# ---------------------------------------------------------------------------
# Layer 1 — direct calls to the split handlers
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.parametrize("handler", HANDLERS)
class TestSplitHandlersOOM:
    def _solver(self):
        solver = make_solver_for_unit_test(UNIT_METHOD)
        # Mimic a fresh call whose batch size is fixed (no growth).
        solver._batch, solver._batch_converged = 1, True
        return solver

    def test_first_oom_seeds_from_max_batch(self, handler):
        """Regression for the reported crash: the *first* OOM must shrink the
        live ``max_batch`` (once a ``0.75 * None`` raised ``TypeError``); the
        result is the call's batch size ``_batch``, converged."""
        solver = self._solver()
        n_panels = 8
        mesh = _make_mesh(n_panels)
        step_idxs = torch.arange(n_panels)
        total_nodes = n_panels * solver.C  # 32

        oom = OOMIntegrand(limit=5)
        # No TypeError here is the whole point.
        _call_handler(solver, handler, oom, mesh, step_idxs, max_batch=total_nodes)

        assert oom.n_oom >= 1, "the integrand never OOM'd; test setup is wrong"
        assert isinstance(solver._batch, int)
        assert solver._batch_converged
        assert 1 <= solver._batch <= oom.limit

    def test_reduction_is_strictly_decreasing(self, handler):
        """The retried batch sizes strictly decrease (each ``<= ceil(0.75*prev)``
        and ``<= prev-1``) until one fits, matching the intended rule exactly.

        ``limit=1`` with ``max_batch=3`` forces the sequence through 2, where a
        naive ``round(0.75*2)=2`` would stick forever — locking in the
        strict-decrease guard."""
        solver = self._solver()
        n_panels = 1
        mesh = _make_mesh(n_panels)
        step_idxs = torch.arange(n_panels)

        oom = OOMIntegrand(limit=1)
        _call_handler(solver, handler, oom, mesh, step_idxs, max_batch=3)

        expected = _expected_reduction_sequence(3, oom.limit)  # [3, 2, 1]
        # The first len(expected) calls are the reduction of the first chunk.
        assert oom.batch_sizes[: len(expected)] == expected
        # Strictly decreasing, no repeats.
        oom_attempts = [b for b in oom.batch_sizes if b > oom.limit]
        assert oom_attempts == sorted(oom_attempts, reverse=True)
        assert len(set(oom_attempts)) == len(oom_attempts)
        assert solver._batch <= oom.limit

    def test_oom_max_batch_clamps_rest_of_batch(self, handler):
        """Once shrunk, the call's batch size ``_batch`` clamps every later
        batch: no evaluation after the first successful reduction exceeds it,
        and it does not grow again (converged)."""
        solver = self._solver()
        n_panels = 8
        mesh = _make_mesh(n_panels)
        step_idxs = torch.arange(n_panels)
        total_nodes = n_panels * solver.C

        oom = OOMIntegrand(limit=6)
        _call_handler(solver, handler, oom, mesh, step_idxs, max_batch=total_nodes)

        cap = solver._batch
        assert solver._batch_converged
        # Every batch that actually fit (didn't OOM) is within the cap.
        assert all(b <= cap for b in oom.batch_sizes if b <= oom.limit)
        assert solver._batch_for(10**9) == cap

    def test_unfittable_single_eval_raises(self, handler):
        """If even ``max_batch == 1`` OOMs (``limit=0``), the handler raises an
        informative ``OutOfMemoryError`` rather than shrinking below 1."""
        solver = self._solver()
        mesh = _make_mesh(1)
        step_idxs = torch.arange(1)

        oom = OOMIntegrand(limit=0)
        with pytest.raises(torch.OutOfMemoryError, match="cannot be reduced"):
            _call_handler(solver, handler, oom, mesh, step_idxs, max_batch=2)


@pytest.mark.tier1
@pytest.mark.parametrize("limit_share", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("max_batch", [30, 60, 100, 150, 200])
@pytest.mark.parametrize("method", ["bosh3", "gk15", "gk21"])
def test_residual_handler_pairs_nodes_with_values_after_oom(
    method, max_batch, limit_share
):
    """The ``conserve_memory`` handler lays a pass out for ``max_batch``; an
    out-of-memory retry then changes the batch size inside the pass, so the
    batches end at other nodes than planned. Whatever they evaluated, every
    returned subinterval must hold its own C nodes with the integrand's values
    at them, and the carried part must be the first nodes of the next
    subinterval with their values.

    Regression: for gk21 with ``max_batch=100`` and a retry at 75, the batches
    evaluated 75 + 40 = 115 nodes, one whole subinterval more than the 92
    planned. That subinterval was treated as a partly evaluated one: its nodes
    were dropped (an empty slice) while its 23 values were carried, and the
    next pass paired them with other nodes. ``integrate`` returned 0.38 for an
    integral of 0.54, with ``converged=True``. ``f = 1`` hid this."""
    solver = make_solver_for_unit_test(method)
    solver._oom_max_batch = None
    C = solver.C
    # More nodes than max_batch, as the caller guarantees (_evaluate_f_on_mesh
    # lowers max_batch to the nodes left).
    n_subintervals = 80
    mesh = _make_mesh(n_subintervals)
    oom = OOMIntegrand(limit=int(limit_share * max_batch), value=_wave)
    nodes, f_evals, _, step_idxs, carried = solver._evaluate_f_on_split_residual_nodes(
        oom,
        (),
        mesh,
        torch.arange(n_subintervals),
        max_batch=max_batch,
        max_mesh_steps=max_batch // C,
        split_node_state=(None, None, None, None),
    )
    assert oom.n_oom >= 1, "the integrand never ran out of memory; wrong setup"
    assert len(step_idxs) >= 1
    assert nodes.shape[:2] == f_evals.shape[:2] == (len(step_idxs), C)
    expected_nodes = solver._compute_nodes(mesh[step_idxs], mesh[step_idxs + 1])
    assert torch.allclose(nodes, expected_nodes, rtol=0, atol=1e-14)
    assert torch.allclose(f_evals, _wave(nodes), rtol=0, atol=1e-12)

    carried_nodes, carried_f_evals, _, carried_point = carried
    n_evaluated = sum(b for b in oom.batch_sizes if b <= oom.limit)
    n_carried = 0 if carried_nodes is None else len(carried_nodes)
    assert n_evaluated == len(step_idxs) * C + n_carried
    if carried_point is not None:
        assert 0 < n_carried < C
        k = int(step_idxs.max()) + 1
        assert torch.equal(carried_point, mesh[k])
        first_nodes = solver._compute_nodes(mesh[k : k + 1], mesh[k + 1 : k + 2])
        assert torch.allclose(carried_nodes, first_nodes[0, :n_carried], atol=1e-14)
        assert torch.allclose(carried_f_evals, _wave(carried_nodes), atol=1e-12)


# ---------------------------------------------------------------------------
# Layer 2 — end-to-end via solver.integrate()
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestIntegrateOOMRecovery:
    LIMIT = 10
    BIG_BATCH = 100_000  # >> any panel count, so the first batch OOMs

    @pytest.mark.parametrize("fail", list(FAILURE_MODES))
    @pytest.mark.parametrize("conserve_memory", [False, True])
    def test_integrate_recovers_from_oom(self, conserve_memory, fail):
        """A full ``integrate()`` of ``f=1`` completes despite repeated OOMs,
        returns the correct integral (∫₀¹ 1 dt = 1), and returns a converged
        batch size ``<= limit``. ``conserve_memory`` selects the split vs. split-residual
        handler; ``take_gradient=False`` keeps us off the (unhandled)
        full-nodes path. ``fail`` covers every recognized out-of-memory form;
        before 1.2.3 the CPU refusal and ``MemoryError`` crashed ``integrate``."""
        solver = make_uniform_solver(
            "gk21", atol=1e-6, rtol=1e-6, device="cpu", max_batch=self.BIG_BATCH
        )
        oom = OOMIntegrand(limit=self.LIMIT, fail=FAILURE_MODES[fail])
        result = solver.integrate(
            oom,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=False,
            conserve_memory=conserve_memory,
        )

        assert oom.n_oom >= 1, "the integrand never OOM'd; test setup is wrong"
        assert math.isclose(float(result.integral.sum()), 1.0, abs_tol=1e-5)
        assert result.max_batch_converged
        assert result.max_batch <= oom.limit

    @pytest.mark.parametrize("limit_share", [0.3, 0.5, 0.8])
    @pytest.mark.parametrize("max_batch", [30, 60, 100, 150, 200])
    @pytest.mark.parametrize("method", ["bosh3", "gk15", "gk21"])
    @pytest.mark.parametrize("conserve_memory", [False, True])
    def test_integral_of_non_constant_f_survives_oom(
        self, conserve_memory, method, max_batch, limit_share
    ):
        """``integrate`` of sin(40 t) + t with batches above ``limit_share`` of
        ``max_batch`` running out of memory: the first batch is retried
        smaller inside its pass, and the integral is still right.

        Regression: with ``conserve_memory=True`` the result was off by 0.05
        to 0.29 (gk21 and gk15, ``max_batch`` 60 to 200) and reported
        ``converged=True``; see
        ``test_residual_handler_pairs_nodes_with_values_after_oom``."""
        solver = make_uniform_solver(
            method, atol=1e-6, rtol=1e-6, device="cpu", max_batch=max_batch
        )
        oom = OOMIntegrand(
            limit=int(limit_share * max_batch), value=_wave, max_calls=50_000
        )
        result = solver.integrate(
            oom,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=False,
            conserve_memory=conserve_memory,
        )
        assert oom.n_oom >= 1, "the integrand never ran out of memory; wrong setup"
        assert result.converged
        assert math.isclose(float(result.integral.sum()), WAVE_INTEGRAL, abs_tol=1e-5)

    def test_unrelated_runtime_error_propagates(self):
        """A ``RuntimeError`` that is not an out-of-memory error is a real bug in
        ``f``: the batch loop must re-raise it unchanged, not shrink the batch."""
        solver = make_uniform_solver(
            "gk21", atol=1e-6, rtol=1e-6, device="cpu", max_batch=self.BIG_BATCH
        )

        def broken(t, *args):
            # Only multi-point batches fail, so the one-point warm-up call in
            # integrate() passes and the error comes from the batch loop.
            if t.shape[0] > 1:
                raise RuntimeError("shape mismatch in f")
            return torch.ones((1, 1), dtype=t.dtype, device=t.device)

        with pytest.raises(RuntimeError, match="shape mismatch in f"):
            solver.integrate(broken, mesh_init=T_INIT, mesh_final=T_FINAL)

    def test_shrunk_size_is_not_carried_without_the_state(self):
        """A call that ran out of memory does not change the next call of the
        same solver: without its state, the next call starts from the given
        ``max_batch`` again (the solver keeps nothing between calls)."""
        solver = make_uniform_solver(
            "gk21", atol=1e-6, rtol=1e-6, device="cpu", max_batch=self.BIG_BATCH
        )
        oom = OOMIntegrand(limit=self.LIMIT)
        first = solver.integrate(
            oom, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
        assert first.max_batch_converged
        assert first.max_batch <= self.LIMIT

        n_before = len(oom.batch_sizes)
        oom.limit = self.BIG_BATCH
        solver.integrate(oom, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False)
        assert max(oom.batch_sizes[n_before:]) > first.max_batch

    def test_state_carries_the_shrunk_size(self):
        """The size an out-of-memory error left is the result's state; passed
        to the next call, it replaces the constructor's larger ``max_batch``:
        no out-of-memory error again and no batch above it."""
        solver = make_uniform_solver(
            "gk21", atol=1e-6, rtol=1e-6, device="cpu", max_batch=self.BIG_BATCH
        )
        oom = OOMIntegrand(limit=self.LIMIT)
        first = solver.integrate(
            oom, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
        n_oom, n_before = oom.n_oom, len(oom.batch_sizes)
        assert n_oom >= 1
        result = solver.integrate(oom, state=first.state, take_gradient=False)
        assert math.isclose(float(result.integral.sum()), 1.0, abs_tol=1e-5)
        assert oom.n_oom == n_oom, "the size was forgotten: the call OOM'd again"
        assert all(b <= first.max_batch for b in oom.batch_sizes[n_before:])
        assert (result.max_batch, result.max_batch_converged) == (first.max_batch, True)
