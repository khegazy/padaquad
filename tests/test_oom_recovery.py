"""Tests for the out-of-memory (OOM) catch-and-retry path.

When an integrand ``f`` raises ``torch.OutOfMemoryError`` mid-evaluation, the
split evaluation handlers
(``_evaluate_f_on_split_nodes`` / ``_evaluate_f_on_split_residual_nodes``) are
supposed to shrink ``max_batch`` and retry the failed batch, caching the new cap
in ``self._oom_max_batch`` so every later batch in the same ``integrate()`` call
stays clamped. The reduction repeats until the batch fits or a single evaluation
(``max_batch == 1``) still fails, at which point an informative error is raised.

Every analytic integrand in the suite has a tiny memory footprint, so this path
was never exercised by the existing tests. These tests **simulate** OOM with a
purpose-built integrand (``OOMIntegrand``) that raises ``torch.OutOfMemoryError``
whenever its batch exceeds a configurable ``limit`` and records every batch size
it was asked to evaluate — so we can watch ``max_batch`` shrink and confirm it is
driven by ``_oom_max_batch``.

Two layers:
  * ``TestSplitHandlersOOM`` — direct calls to both split handlers, parametrized
    over the handler, mirroring the direct-call unit tests in
    ``test_evaluate_f_on_mesh.py``.
  * ``TestIntegrateOOMRecovery`` — full ``.integrate()`` runs (the real user
    path), across ``conserve_memory`` (split vs. split-residual).

Note: the ``take_gradient=True`` path (``_evaluate_f_on_full_nodes``) currently
has **no** OOM recovery; it is intentionally out of scope here.
"""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import (
    T_FINAL,
    T_INIT,
    constant_integrand,
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
    """

    def __init__(self, limit: int, max_calls: int = 2000):
        self.limit = limit
        self.max_calls = max_calls
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
            raise torch.OutOfMemoryError(f"simulated OOM at batch {n}")
        return torch.ones((n, 1), dtype=t.dtype, device=t.device)


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


# ---------------------------------------------------------------------------
# Layer 1 — direct calls to the split handlers
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.parametrize("handler", HANDLERS)
class TestSplitHandlersOOM:
    def _solver(self):
        solver = make_solver_for_unit_test(UNIT_METHOD)
        # Mimic integrate()'s per-call reset (base.py:486): no cap carried in.
        solver._oom_max_batch = None
        return solver

    def test_first_oom_seeds_from_max_batch(self, handler):
        """Regression for the reported crash: the *first* OOM must seed
        ``_oom_max_batch`` from the live ``max_batch`` (it starts as ``None``)
        instead of computing ``0.75 * None`` and raising ``TypeError``."""
        solver = self._solver()
        n_panels = 8
        mesh = _make_mesh(n_panels)
        step_idxs = torch.arange(n_panels)
        total_nodes = n_panels * solver.C  # 32

        oom = OOMIntegrand(limit=5)
        # No TypeError here is the whole point.
        _call_handler(solver, handler, oom, mesh, step_idxs, max_batch=total_nodes)

        assert oom.n_oom >= 1, "the integrand never OOM'd; test setup is wrong"
        assert isinstance(solver._oom_max_batch, int)
        assert 1 <= solver._oom_max_batch <= oom.limit

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
        assert solver._oom_max_batch <= oom.limit

    def test_oom_max_batch_clamps_rest_of_batch(self, handler):
        """Once set, ``_oom_max_batch`` clamps every later batch: no evaluation
        after the first successful reduction exceeds it, ``previous_max_batch``
        tracks it, and ``get_max_batch`` refuses to hand back anything larger."""
        solver = self._solver()
        n_panels = 8
        mesh = _make_mesh(n_panels)
        step_idxs = torch.arange(n_panels)
        total_nodes = n_panels * solver.C

        oom = OOMIntegrand(limit=6)
        _call_handler(solver, handler, oom, mesh, step_idxs, max_batch=total_nodes)

        cap = solver._oom_max_batch
        assert cap is not None
        # Every batch that actually fit (didn't OOM) is within the cap.
        assert all(b <= cap for b in oom.batch_sizes if b <= oom.limit)
        assert solver.previous_max_batch == cap
        # The get_max_batch clamp (base.py:914-919) refuses a larger request.
        assert solver.get_max_batch(default_max_batch=10**9) == cap

    def test_unfittable_single_eval_raises(self, handler):
        """If even ``max_batch == 1`` OOMs (``limit=0``), the handler raises an
        informative ``OutOfMemoryError`` rather than shrinking below 1."""
        solver = self._solver()
        mesh = _make_mesh(1)
        step_idxs = torch.arange(1)

        oom = OOMIntegrand(limit=0)
        with pytest.raises(torch.OutOfMemoryError, match="cannot be reduced"):
            _call_handler(solver, handler, oom, mesh, step_idxs, max_batch=2)


# ---------------------------------------------------------------------------
# Layer 2 — end-to-end via solver.integrate()
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestIntegrateOOMRecovery:
    LIMIT = 10
    BIG_BATCH = 100_000  # >> any panel count, so the first batch OOMs

    @pytest.mark.parametrize("conserve_memory", [False, True])
    def test_integrate_recovers_from_oom(self, conserve_memory):
        """A full ``integrate()`` of ``f=1`` completes despite repeated OOMs,
        returns the correct integral (∫₀¹ 1 dt = 1), and leaves a cap
        ``<= limit``. ``conserve_memory`` selects the split vs. split-residual
        handler; ``take_gradient=False`` keeps us off the (unhandled)
        full-nodes path."""
        solver = make_uniform_solver(
            "gk21", atol=1e-6, rtol=1e-6, device="cpu", max_batch=self.BIG_BATCH
        )
        oom = OOMIntegrand(limit=self.LIMIT)
        result = solver.integrate(
            oom,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=False,
            conserve_memory=conserve_memory,
        )

        assert oom.n_oom >= 1, "the integrand never OOM'd; test setup is wrong"
        assert math.isclose(float(result.integral.sum()), 1.0, abs_tol=1e-5)
        assert solver._oom_max_batch is not None
        assert solver._oom_max_batch <= oom.limit

    def test_oom_max_batch_reset_between_calls(self):
        """``_oom_max_batch`` is reset to ``None`` at the start of each
        ``integrate()`` (base.py:486): a clean follow-up run that never OOMs
        must not inherit the previous run's cap."""
        solver = make_uniform_solver(
            "gk21", atol=1e-6, rtol=1e-6, device="cpu", max_batch=self.BIG_BATCH
        )

        oom = OOMIntegrand(limit=self.LIMIT)
        solver.integrate(
            oom, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
        assert solver._oom_max_batch is not None  # cap set by the OOM run

        # A run with a tiny-footprint integrand never OOMs -> cap stays None.
        solver.integrate(
            constant_integrand,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=False,
        )
        assert solver._oom_max_batch is None
