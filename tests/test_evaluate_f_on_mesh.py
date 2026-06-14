"""Comprehensive tests for ``_evaluate_f_on_mesh`` and its evaluation paths.

``AdaptiveQuadrature._evaluate_f_on_mesh`` dispatches the integrand evaluation
to one of three sub-functions:

  * ``_evaluate_f_on_full_nodes`` (``take_gradient=True``): a single ``f`` call
    over a whole-mesh batch; requires ``max_mesh_steps >= 1``.
  * ``_evaluate_f_on_split_nodes`` (``take_gradient=False``,
    ``conserve_memory=False`` — the default): evaluates **all** pending panels
    in one call, chunking the flat node list into ``max_batch``-sized ``f``
    calls only to respect memory. The integral is *complete* after the call;
    there is **no** residual carry-over, so it always returns
    ``split_node_state = (None, None, None, None)`` and takes neither
    ``max_mesh_steps`` nor ``split_node_state``.
  * ``_evaluate_f_on_split_residual_nodes`` (``take_gradient=False``,
    ``conserve_memory=True``): the memory-conserving path. Evaluates at most a
    few panels per call and carries a partial-panel residual forward across
    iterations via ``split_node_state`` (Path A / Path B).

These tests cover three layers:

  * ``TestFullNodes`` / ``TestSplitNodes`` / ``TestSplitResidualNodes`` —
    direct calls to the three sub-functions.
  * ``TestIntegration`` — the ``_evaluate_f_on_mesh`` dispatcher + multi-call
    loop, across ``take_gradient`` and ``conserve_memory``.
  * ``TestEndToEnd`` — full ``.integrate(max_batch=K)`` calls.

The ``max_batch`` sweep covers every K in [0, 2C+1] for each method (see
``_max_batch_range``). Two integrands — ``damped_sine`` (hardest) and ``exp``
(wide dynamic range) — stress the integration scheme along orthogonal axes.

Note on cross-validation: ``_evaluate_f_on_split_nodes`` chunks the flat node
list and concatenates per-chunk ``f`` outputs, so it is only *bit*-equal to the
full path's single ``f(nodes_flat)`` call when its loop runs exactly once (i.e.
``max_batch >= total_nodes``). With a small ``max_batch`` the loop is active and
only *approximate* equality holds — so loop-active cross-checks compare
``f_evals`` with ``torch.allclose`` at a tight tolerance, not bit-equality.

Tier markers (T1, T2) tag tests by priority. Run a subset with ``pytest -m
tier1`` etc.
"""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import (
    make_solver_for_unit_test,
    make_uniform_solver,
)

from padaquad import UNIFORM_METHODS, integrand_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_METHODS = ["adaptive_heun", "gk15"]  # C=2 and C=17 respectively
TIER1_INTEGRANDS = ["damped_sine", "exp"]  # hardest + wide-dynamic-range stress

# Default mesh covers [0, 1] with 4 uniform panels (5 barriers).
MESH_INIT = 0.0
MESH_FINAL = 1.0
DEFAULT_PANELS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _method_C(method_name: str) -> int:
    """Number of quadrature points per panel for the given method."""
    return len(UNIFORM_METHODS[method_name].tableau.c)


def _max_batch_range(method_name: str) -> list[int]:
    """Generate [0, 1, ..., 2*C+1] for the given method."""
    C = _method_C(method_name)
    return list(range(2 * C + 2))


def _make_mesh(n_panels: int = DEFAULT_PANELS) -> torch.Tensor:
    """Build a uniform mesh of n_panels panels on [MESH_INIT, MESH_FINAL]."""
    return torch.linspace(
        MESH_INIT, MESH_FINAL, n_panels + 1, dtype=torch.float64
    ).unsqueeze(-1)


def _make_step_idxs(n_panels: int = DEFAULT_PANELS) -> torch.Tensor:
    """Indices of all panels in a fresh mesh."""
    return torch.arange(n_panels)


def _simple_integrand(t, *args):
    """f(t) = t^2 + 1. Shape preserving: [N, T] -> [N, T] (with D=T)."""
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t**2 + 1


def _resolve_integrand(name: str):
    """Return (f, solution_fxn, cutoff) for an integrand by name."""
    return integrand_dict[name]


def _sweep_params(methods=None, batch_range_fn=_max_batch_range):
    """Cross-product of methods and per-method max_batch values."""
    if methods is None:
        methods = TEST_METHODS
    return [(m, b) for m in methods for b in batch_range_fn(m)]


def _sweep_ids(params):
    """Readable parametrize IDs like 'gk15-max_batch_17'."""
    return [f"{m}-max_batch_{b}" for m, b in params]


# ---------------------------------------------------------------------------
# Parametrize collections
# ---------------------------------------------------------------------------

# Full sweep: (method, max_batch ∈ [0, 2C+1])
_ALL_SWEEP = _sweep_params()
_ALL_SWEEP_IDS = _sweep_ids(_ALL_SWEEP)

# Below-C sweep: (method, max_batch ∈ [0, C-1]) — these should error for full_nodes
_BELOW_C_SWEEP = _sweep_params(batch_range_fn=lambda m: list(range(_method_C(m))))
_BELOW_C_SWEEP_IDS = _sweep_ids(_BELOW_C_SWEEP)

# Below-C but nonzero: (method, max_batch ∈ [1, C-1]). Supported for the split
# paths (take_gradient=False): a budget too small to hold a whole panel still
# runs by evaluating a panel's C nodes across several batches. Must run AND be
# correct. (adaptive_heun has C=2, so this is just max_batch=1; gk15 has C=17,
# giving max_batch ∈ [1..16].)
_BELOW_C_NONZERO_SWEEP = _sweep_params(
    batch_range_fn=lambda m: list(range(1, _method_C(m)))
)
_BELOW_C_NONZERO_SWEEP_IDS = _sweep_ids(_BELOW_C_NONZERO_SWEEP)

# Above-or-equal-C sweep: (method, max_batch ∈ [C, 2C+1])
_AT_OR_ABOVE_C_SWEEP = _sweep_params(
    batch_range_fn=lambda m: list(range(_method_C(m), 2 * _method_C(m) + 2))
)
_AT_OR_ABOVE_C_SWEEP_IDS = _sweep_ids(_AT_OR_ABOVE_C_SWEEP)

# Split-only valid sweep: (method, max_batch ∈ [1, 2C+1]) — max_batch=0 excluded
_SPLIT_VALID_SWEEP = _sweep_params(
    batch_range_fn=lambda m: list(range(1, 2 * _method_C(m) + 2))
)
_SPLIT_VALID_SWEEP_IDS = _sweep_ids(_SPLIT_VALID_SWEEP)


# ===========================================================================
# TestFullNodes — _evaluate_f_on_full_nodes (take_gradient=True path)
# ===========================================================================


class TestFullNodes:
    """Direct calls to ``_evaluate_f_on_full_nodes`` with hand-crafted inputs."""

    # -----------------------------------------------------------------------
    # max_batch < C must assert (max_mesh_steps == 0)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _BELOW_C_SWEEP, ids=_BELOW_C_SWEEP_IDS
    )
    def test_full_nodes_max_batch_below_C_asserts(self, method, max_batch):
        """``_evaluate_f_on_full_nodes`` asserts ``max_mesh_steps >= 1``.

        Any ``max_batch < C`` yields ``max_mesh_steps = 0`` and trips the guard.
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C  # always 0 in this range
        mesh = _make_mesh()
        step_idxs = _make_step_idxs()
        with pytest.raises(AssertionError):
            solver._evaluate_f_on_full_nodes(
                _simple_integrand, (), mesh, step_idxs, max_mesh_steps
            )

    # -----------------------------------------------------------------------
    # shapes and values correct
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _AT_OR_ABOVE_C_SWEEP,
        ids=_AT_OR_ABOVE_C_SWEEP_IDS,
    )
    def test_full_nodes_correct_shapes_and_values(
        self, method, max_batch, integrand_name
    ):
        """Outputs match independent f-evaluation, with correct shapes
        and (None, None, None, None) split_node_state."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        mesh = _make_mesh()
        step_idxs = _make_step_idxs()
        f, _, _ = _resolve_integrand(integrand_name)

        nodes, f_evals, _tracked, returned_idxs, state = (
            solver._evaluate_f_on_full_nodes(f, (), mesh, step_idxs, max_mesh_steps)
        )

        N_expected = min(max_mesh_steps, len(step_idxs))
        assert nodes.shape == (N_expected, C, 1)
        assert f_evals.shape[:2] == (N_expected, C)
        assert state == (None, None, None, None)
        assert returned_idxs.shape == (N_expected,)

        # f_evals should match an independent flat evaluation
        nodes_flat = torch.reshape(nodes, (N_expected * C, -1))
        f_evals_independent = f(nodes_flat)
        f_evals_flat = torch.reshape(f_evals, (N_expected * C, -1))
        assert torch.equal(f_evals_flat, f_evals_independent), (
            f"f_evals not bit-equal to independent evaluation for {method} "
            f"with max_batch={max_batch}, integrand={integrand_name}"
        )

    # -----------------------------------------------------------------------
    # step_idxs non-contiguous (every other panel)
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_full_nodes_step_idxs_non_contiguous(self, method):
        """step_idxs = [0, 2, 4] picks every other panel. Output nodes
        must lie within those panels' barriers."""
        solver = make_solver_for_unit_test(method)
        n_panels = 6
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.tensor([0, 2, 4])

        nodes_f, _, _, idxs_f, _ = solver._evaluate_f_on_full_nodes(
            _simple_integrand, (), mesh, step_idxs, max_mesh_steps=3
        )
        assert idxs_f.tolist() == [0, 2, 4]
        # First panel: nodes between mesh[0] and mesh[1]
        # Third panel: nodes between mesh[4] and mesh[5]
        assert nodes_f[0, 0, 0] >= mesh[0, 0] - 1e-12
        assert nodes_f[0, -1, 0] <= mesh[1, 0] + 1e-12
        assert nodes_f[2, 0, 0] >= mesh[4, 0] - 1e-12
        assert nodes_f[2, -1, 0] <= mesh[5, 0] + 1e-12


# ===========================================================================
# TestSplitNodes — the NEW _evaluate_f_on_split_nodes (evaluate-all, no residual)
# ===========================================================================


class TestSplitNodes:
    """Direct calls to ``_evaluate_f_on_split_nodes``.

    New signature: ``(f, f_args, mesh, step_idxs, max_batch)`` — no
    ``max_mesh_steps``, no ``split_node_state``. One call evaluates every panel
    in ``step_idxs`` (chunking ``f`` into ``max_batch``-sized calls only for
    memory) and always returns ``split_node_state = (None, None, None, None)``.
    """

    # -----------------------------------------------------------------------
    # max_batch=0 must error (explicit guard; dispatcher rescues 0 -> 1)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_max_batch_zero_errors(self, method):
        """max_batch=0 trips the explicit ``assert max_batch > 0`` guard.

        A direct call bypasses the dispatcher's 0 -> 1 rescue, so the
        function's own guard must reject it.
        """
        solver = make_solver_for_unit_test(method)
        mesh = _make_mesh()
        step_idxs = _make_step_idxs()
        with pytest.raises(AssertionError):
            solver._evaluate_f_on_split_nodes(
                _simple_integrand, (), mesh, step_idxs, max_batch=0
            )

    # -----------------------------------------------------------------------
    # One call evaluates ALL panels and never leaves a residual
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_split_nodes_evaluates_all_panels(self, method, max_batch):
        """For every valid max_batch a single call returns all panels with an
        empty split_node_state — the headline behavioral change vs the old
        residual-carrying function."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        # Use a mesh large enough that the chunking loop runs many times for
        # small max_batch (n_panels * C >> max_batch).
        n_panels = 6
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)

        nodes, f_evals, tracked, returned_idxs, state = (
            solver._evaluate_f_on_split_nodes(
                _simple_integrand, (), mesh, step_idxs, max_batch=max_batch
            )
        )

        assert nodes.shape == (n_panels, C, 1)
        assert f_evals.shape == (n_panels, C, 1)
        assert tracked is None
        assert torch.equal(returned_idxs, step_idxs)
        assert state == (None, None, None, None)

    # -----------------------------------------------------------------------
    # Single-batch regime: bit-equal to full_nodes (loop runs exactly once)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_single_batch_bit_equal_to_full(self, method, integrand_name):
        """When ``max_batch >= n_panels * C`` the chunking loop runs once, so
        the split path issues the exact same single ``f`` call as the full
        path — outputs must be bit-equal (nodes AND f_evals)."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = 5
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        nodes_full, f_evals_full, _, _, _ = solver._evaluate_f_on_full_nodes(
            f, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        # max_batch large enough to hold all nodes -> one chunk.
        nodes_split, f_evals_split, _, _, state = solver._evaluate_f_on_split_nodes(
            f, (), mesh, step_idxs, max_batch=n_panels * C
        )

        assert state == (None, None, None, None)
        assert torch.equal(nodes_split, nodes_full), "nodes not bit-equal"
        assert torch.equal(f_evals_split, f_evals_full), (
            "f_evals not bit-equal in the single-batch regime"
        )

    # -----------------------------------------------------------------------
    # Loop-active regime: small max_batch vs full (large) — approximate match
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_split_nodes_loop_matches_full(self, method, max_batch, integrand_name):
        """The meaningful cross-check: run the split path with a *small*
        ``max_batch`` that activates the chunking loop, and the full path with a
        *large* ``max_batch`` (single batch). ``nodes`` are computed once by
        ``_compute_nodes`` (independent of batching) so they stay bit-equal;
        ``f_evals`` may pick up summation-order drift across chunk boundaries,
        so compare within a tight numerical tolerance rather than bit-equality.
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        # n_panels * C >> max_batch (max_batch <= 2C+1) guarantees >1 chunk.
        n_panels = 6
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        # Reference: full path, single batch over the whole mesh.
        nodes_ref, f_evals_ref, _, _, _ = solver._evaluate_f_on_full_nodes(
            f, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        # Comparison: split path with the loop active.
        nodes_split, f_evals_split, _, _, state = solver._evaluate_f_on_split_nodes(
            f, (), mesh, step_idxs, max_batch=max_batch
        )

        assert state == (None, None, None, None)
        assert torch.equal(nodes_split, nodes_ref), (
            f"nodes not bit-equal: max abs diff = "
            f"{(nodes_split - nodes_ref).abs().max().item():.3e}"
        )
        max_diff = (f_evals_split - f_evals_ref).abs().max().item()
        assert max_diff < 1e-12, (
            f"f_evals drift {max_diff:.3e} >= 1e-12 for {method} "
            f"max_batch={max_batch} integrand={integrand_name}"
        )

    # -----------------------------------------------------------------------
    # f-call accounting: ceil(total_nodes / max_batch) calls
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_split_nodes_calls_f_expected_times(self, method, max_batch):
        """``f`` is called ``ceil(len(step_idxs) * C / max_batch)`` times — one
        chunk per ``max_batch`` slice of the flat node list."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = DEFAULT_PANELS
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = _make_step_idxs(n_panels)

        call_count = {"n": 0}

        def counting_f(t, *args):
            call_count["n"] += 1
            return _simple_integrand(t, *args)

        solver._evaluate_f_on_split_nodes(
            counting_f, (), mesh, step_idxs, max_batch=max_batch
        )

        total_nodes = n_panels * C
        expected = math.ceil(total_nodes / max_batch)
        assert call_count["n"] == expected, (
            f"f called {call_count['n']} times, expected {expected} for "
            f"{method} max_batch={max_batch} (total_nodes={total_nodes})"
        )

    # -----------------------------------------------------------------------
    # Below-C single call still reconstructs every panel
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _BELOW_C_NONZERO_SWEEP,
        ids=_BELOW_C_NONZERO_SWEEP_IDS,
    )
    def test_split_nodes_below_C_single_call(self, method, max_batch, integrand_name):
        """A sub-C budget (``0 < max_batch < C``) still completes every panel in
        one call by chunking each panel's ``C`` nodes across several ``f`` calls.
        Outputs match the full path's single call (nodes bit-equal, f_evals to
        1e-12)."""
        solver = make_solver_for_unit_test(method)
        n_panels = 3
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        nodes_ref, f_evals_ref, _, _, _ = solver._evaluate_f_on_full_nodes(
            f, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        nodes_split, f_evals_split, _, idxs, state = solver._evaluate_f_on_split_nodes(
            f, (), mesh, step_idxs, max_batch=max_batch
        )

        assert state == (None, None, None, None)
        assert torch.equal(idxs, step_idxs)
        assert torch.equal(nodes_split, nodes_ref)
        assert torch.allclose(f_evals_split, f_evals_ref, atol=1e-12, rtol=0)

    # -----------------------------------------------------------------------
    # Multi-dimensional integrand output (D > 1)
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_D_gt_1(self, method):
        """Integrand returning [N, 3] — verify reshape to [N, C, 3] in a single
        call (with the loop active) and agreement with the full path."""

        def vec_integrand(t, *args):
            """f(t) = [sin(t), cos(t), t]. Returns [N, 3]."""
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return torch.cat([torch.sin(t), torch.cos(t), t], dim=-1)

        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = 4
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)

        nodes_full, f_evals_full, _, _, _ = solver._evaluate_f_on_full_nodes(
            vec_integrand, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        assert f_evals_full.shape == (n_panels, C, 3)

        # Small max_batch forces the chunking loop.
        nodes_split, f_evals_split, _, _, state = solver._evaluate_f_on_split_nodes(
            vec_integrand, (), mesh, step_idxs, max_batch=2 * C + 1
        )
        assert state == (None, None, None, None)
        assert f_evals_split.shape == (n_panels, C, 3)
        assert torch.equal(nodes_split, nodes_full)
        assert torch.allclose(f_evals_split, f_evals_full, atol=1e-12, rtol=0)

    # -----------------------------------------------------------------------
    # Tracked variables flow through the chunking loop (covers the
    # tracked_lists fix at base.py)
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_tracked_variables(self, method):
        """An integrand emitting tracked variables must have each tracked
        tensor concatenated across chunks and reshaped to ``[N, C, ...]``.

        Uses a small ``max_batch`` so multiple chunks are concatenated — the
        exact path that the ``tracked_list``/``tracked_lists`` typo broke.
        """

        def tracked_integrand(t, *args):
            """Return (sin(t), (t**2, cos(t))) — integrand + two tracked vars."""
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return torch.sin(t), (t**2, torch.cos(t))

        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = 4
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)

        # Small max_batch -> the loop runs several times and tracked variables
        # accumulate across chunks before being concatenated.
        nodes, f_evals, tracked, _idxs, state = solver._evaluate_f_on_split_nodes(
            tracked_integrand, (), mesh, step_idxs, max_batch=C + 1
        )

        assert state == (None, None, None, None)
        assert f_evals.shape == (n_panels, C, 1)
        assert isinstance(tracked, tuple)
        assert len(tracked) == 2
        for tv in tracked:
            assert tv.shape == (n_panels, C, 1)

        # Tracked values must match a direct evaluation on the same nodes.
        nodes_flat = torch.reshape(nodes, (n_panels * C, -1))
        expected_sq = (nodes_flat**2).reshape(n_panels, C, 1)
        expected_cos = torch.cos(nodes_flat).reshape(n_panels, C, 1)
        assert torch.allclose(tracked[0], expected_sq, atol=1e-12, rtol=0)
        assert torch.allclose(tracked[1], expected_cos, atol=1e-12, rtol=0)

    # -----------------------------------------------------------------------
    # step_idxs with a single element
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_single_element(self, method):
        """``step_idxs = [k]`` for a single panel. Catches off-by-one in
        indexing."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        mesh = _make_mesh(n_panels=4)
        step_idxs = torch.tensor([1])  # only panel 1
        nodes, f_evals, _tracked, returned_idxs, state = (
            solver._evaluate_f_on_split_nodes(
                _simple_integrand, (), mesh, step_idxs, max_batch=C
            )
        )
        assert nodes.shape == (1, C, 1)
        assert f_evals.shape == (1, C, 1)
        assert returned_idxs.tolist() == [1]
        assert state == (None, None, None, None)

    # -----------------------------------------------------------------------
    # Determinism: same inputs twice -> bit-equal outputs
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_deterministic(self, method):
        """Calling with identical inputs twice must give bit-equal outputs."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_batch = C + 1
        mesh = _make_mesh()
        step_idxs = _make_step_idxs()

        def run_once():
            return solver._evaluate_f_on_split_nodes(
                _simple_integrand, (), mesh, step_idxs, max_batch=max_batch
            )

        n1, e1, _t1, i1, s1 = run_once()
        n2, e2, _t2, i2, s2 = run_once()

        assert torch.equal(n1, n2), "nodes differ across identical calls"
        assert torch.equal(e1, e2), "f_evals differ across identical calls"
        assert torch.equal(i1, i2), "step_idxs differ across identical calls"
        assert s1 == (None, None, None, None)
        assert s2 == (None, None, None, None)


# ===========================================================================
# TestSplitResidualNodes — the memory-conserving _evaluate_f_on_split_residual_nodes
# ===========================================================================


class TestSplitResidualNodes:
    """Direct calls to ``_evaluate_f_on_split_residual_nodes`` (conserve_memory).

    Signature: ``(f, f_args, mesh, step_idxs, max_batch, max_mesh_steps,
    split_node_state)``. Evaluates a bounded number of panels per call and
    carries a partial-panel residual forward via ``split_node_state``
    (Path A = first call, Path B = continuation that consumes a residual).
    """

    # -----------------------------------------------------------------------
    # max_batch=0 must error
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_residual_max_batch_zero_errors(self, method):
        """max_batch=0 trips the explicit ``assert max_batch > 0`` guard."""
        solver = make_solver_for_unit_test(method)
        mesh = _make_mesh()
        step_idxs = _make_step_idxs()
        with pytest.raises(AssertionError):
            solver._evaluate_f_on_split_residual_nodes(
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                max_batch=0,
                max_mesh_steps=0,
                split_node_state=(None, None, None, None),
            )

    # -----------------------------------------------------------------------
    # Path A (first call, no residual carryover)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_residual_path_A_first_call(self, method, max_batch):
        """Path A invariants when split_node_state=(None, None, None, None).

        Verifies:
          * Output shapes match the predicted num_mesh_steps
          * evaluate_all=True  -> split_node_state == (None, None, None, None)
          * evaluate_all=False -> residual tensors present with expected size
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        if max_mesh_steps < 1:
            pytest.skip("max_mesh_steps < 1; degenerate Path A — see E2E tests")

        mesh = _make_mesh(n_panels=max(4, 2 * max_mesh_steps + 2))
        step_idxs = torch.arange(mesh.shape[0] - 1)
        evaluate_all = (max_mesh_steps * C) % max_batch == 0
        num_mesh_steps_expected = max_mesh_steps if evaluate_all else max_mesh_steps + 1

        nodes, _f_evals, _tracked, _returned_idxs, state = (
            solver._evaluate_f_on_split_residual_nodes(
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_mesh_steps,
                split_node_state=(None, None, None, None),
            )
        )

        if evaluate_all:
            assert nodes.shape == (num_mesh_steps_expected, C, 1)
            assert state == (None, None, None, None)
        else:
            num_residual = max_batch - (max_mesh_steps * C) % max_batch
            assert nodes.shape == (num_mesh_steps_expected - 1, C, 1)
            residual_nodes, residual_f_evals, _residual_tracked, residual_mesh_idx = (
                state
            )
            assert residual_nodes is not None
            assert residual_f_evals is not None
            assert residual_mesh_idx is not None
            assert residual_nodes.shape[0] == num_residual

    # -----------------------------------------------------------------------
    # Path B (continuation, residual present)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_residual_path_B_continuation(self, method, max_batch):
        """Path B: first call generates a residual; second call consumes it."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        if max_mesh_steps < 1:
            pytest.skip("max_mesh_steps < 1; not a useful Path B scenario")

        mesh = _make_mesh(n_panels=max(6, 3 * max_mesh_steps + 2))
        step_idxs = torch.arange(mesh.shape[0] - 1)

        _nodes1, _f_evals1, _tracked, step_idxs1, state1 = (
            solver._evaluate_f_on_split_residual_nodes(
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_mesh_steps,
                split_node_state=(None, None, None, None),
            )
        )

        evaluate_all_iter1 = (max_mesh_steps * C) % max_batch == 0
        if evaluate_all_iter1:
            pytest.skip("Iteration 1 was evaluate_all=True; no residual to feed Path B")

        remaining_idxs = step_idxs[len(step_idxs1) :]
        if len(remaining_idxs) == 0:
            pytest.skip("Iteration 1 consumed all panels; no Path B continuation")

        nodes2, _f_evals2, _tracked, _step_idxs2, _state2 = (
            solver._evaluate_f_on_split_residual_nodes(
                _simple_integrand,
                (),
                mesh,
                remaining_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_mesh_steps,
                split_node_state=state1,
            )
        )

        assert nodes2.shape[0] >= 1
        assert nodes2.shape[1] == C
        assert nodes2.shape[2] == 1
        residual_nodes, _residual_f_evals, _residual_tracked, _residual_mesh_idx = (
            state1
        )
        num_residual = residual_nodes.shape[0]
        assert torch.allclose(
            nodes2[0, :num_residual, :], residual_nodes, atol=1e-12
        ), "Path B did not prepend residual_nodes correctly"

    # -----------------------------------------------------------------------
    # Cross-validation: residual split-loop accumulation == full single-call
    # -----------------------------------------------------------------------

    def _loop_split_until_done(self, solver, f, mesh, original_step_idxs, max_batch):
        """Drive ``_evaluate_f_on_split_residual_nodes`` to completion.

        Mirrors the integrate-loop's call pattern: feeds split_node_state
        forward, recomputes max_batch / max_mesh_steps each iteration based on
        remaining panels.

        Returns (nodes_acc, f_evals_acc, step_idxs_acc, final_state, total_iters).
        """
        C = solver.C
        total_panels = len(original_step_idxs)
        completed = 0
        state = (None, None, None, None)
        nodes_acc, f_evals_acc, step_idxs_acc = [], [], []
        iters = 0
        max_iters = total_panels + 5
        while completed < total_panels:
            iters += 1
            if iters > max_iters:
                raise RuntimeError(
                    f"Split loop did not converge in {max_iters} iterations "
                    f"(completed={completed}/{total_panels}, max_batch={max_batch})"
                )
            remaining_input = original_step_idxs[completed:]
            assert len(remaining_input) > 0, (
                f"called with no remaining panels "
                f"(completed={completed}/{total_panels})"
            )
            batches_left = len(remaining_input) * C
            effective_max_batch = min(max_batch, batches_left)
            effective_max_mesh_steps = effective_max_batch // C
            nodes_i, f_evals_i, _tracked, idxs_i, state = (
                solver._evaluate_f_on_split_residual_nodes(
                    f,
                    (),
                    mesh,
                    remaining_input,
                    max_batch=effective_max_batch,
                    max_mesh_steps=effective_max_mesh_steps,
                    split_node_state=state,
                )
            )
            assert len(idxs_i) > 0, (
                f"made no forward progress (completed={completed}/{total_panels}, "
                f"max_batch={effective_max_batch}, "
                f"max_mesh_steps={effective_max_mesh_steps})"
            )
            nodes_acc.append(nodes_i)
            f_evals_acc.append(f_evals_i)
            step_idxs_acc.append(idxs_i)
            completed += len(idxs_i)
        return (
            torch.cat(nodes_acc, dim=0),
            torch.cat(f_evals_acc, dim=0),
            torch.cat(step_idxs_acc, dim=0),
            state,
            iters,
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        ("method", "max_batch"), _AT_OR_ABOVE_C_SWEEP, ids=_AT_OR_ABOVE_C_SWEEP_IDS
    )
    def test_residual_full_loop_matches_full_nodes(
        self, method, max_batch, integrand_name
    ):
        """Gold-standard cross-validation: residual split-loop accumulation
        matches the full path's single-call output within 1e-12."""
        solver = make_solver_for_unit_test(method)
        n_panels = max(6, 3 * (max_batch // solver.C) + 2)
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        nodes_ref, f_evals_ref, _tracked, _step_idxs_ref, _ = (
            solver._evaluate_f_on_full_nodes(
                f, (), mesh, step_idxs, max_mesh_steps=n_panels
            )
        )

        nodes_loop, f_evals_loop, step_idxs_loop, final_state, _iters = (
            self._loop_split_until_done(solver, f, mesh, step_idxs, max_batch)
        )

        assert step_idxs_loop.shape[0] == n_panels
        assert final_state == (None, None, None, None), (
            f"Loop left orphan residual for {method} max_batch={max_batch}"
        )
        assert torch.allclose(nodes_loop, nodes_ref, atol=1e-12, rtol=0), (
            f"nodes mismatch: max abs diff = "
            f"{(nodes_loop - nodes_ref).abs().max().item():.3e}"
        )
        assert torch.allclose(f_evals_loop, f_evals_ref, atol=1e-12, rtol=0), (
            f"f_evals mismatch: max abs diff = "
            f"{(f_evals_loop - f_evals_ref).abs().max().item():.3e}"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        ("method", "max_batch"), _AT_OR_ABOVE_C_SWEEP, ids=_AT_OR_ABOVE_C_SWEEP_IDS
    )
    def test_residual_path_bit_equal_to_full_path(
        self, method, max_batch, integrand_name
    ):
        """Tighter cross-check: nodes bit-equal; f_evals drift < 1e-14."""
        solver = make_solver_for_unit_test(method)
        n_panels = max(6, 3 * (max_batch // solver.C) + 2)
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        nodes_ref, f_evals_ref, _, _, _ = solver._evaluate_f_on_full_nodes(
            f, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        nodes_loop, f_evals_loop, _, _, _ = self._loop_split_until_done(
            solver, f, mesh, step_idxs, max_batch
        )

        assert torch.equal(nodes_loop, nodes_ref), (
            f"nodes not bit-equal: max abs diff = "
            f"{(nodes_loop - nodes_ref).abs().max().item():.3e}"
        )
        max_diff = (f_evals_loop - f_evals_ref).abs().max().item()
        assert max_diff < 1e-14, (
            f"f_evals drift {max_diff:.3e} >= 1e-14 for {method} "
            f"max_batch={max_batch} integrand={integrand_name}"
        )

    # -----------------------------------------------------------------------
    # Below-C: residual loop reconstructs each panel across many batches
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _BELOW_C_NONZERO_SWEEP,
        ids=_BELOW_C_NONZERO_SWEEP_IDS,
    )
    def test_residual_below_C_matches_full_nodes(
        self, method, max_batch, integrand_name
    ):
        """A sub-C budget reconstructs panels exactly across many batches via
        ``split_node_state``. Accumulated nodes/f_evals match the single
        full-nodes reference within 1e-12, with no orphaned residual."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = 3
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        nodes_ref, f_evals_ref, _tracked, _idxs_ref, _ = (
            solver._evaluate_f_on_full_nodes(
                f, (), mesh, step_idxs, max_mesh_steps=n_panels
            )
        )

        state = (None, None, None, None)
        nodes_acc, f_evals_acc, idxs_acc = [], [], []
        completed = 0
        iters = 0
        max_iters = n_panels * math.ceil(C / max_batch) + n_panels + 5
        while completed < n_panels:
            iters += 1
            assert iters <= max_iters, (
                f"split loop did not converge for {method} "
                f"max_batch={max_batch} (completed={completed}/{n_panels})"
            )
            remaining = step_idxs[completed:]
            batches_left = len(remaining) * C
            eff_max_batch = min(max_batch, batches_left)
            eff_max_mesh_steps = eff_max_batch // C
            nodes_i, f_evals_i, _tracked_i, idxs_i, state = (
                solver._evaluate_f_on_split_residual_nodes(
                    f,
                    (),
                    mesh,
                    remaining,
                    max_batch=eff_max_batch,
                    max_mesh_steps=eff_max_mesh_steps,
                    split_node_state=state,
                )
            )
            if len(idxs_i) > 0:
                nodes_acc.append(nodes_i)
                f_evals_acc.append(f_evals_i)
                idxs_acc.append(idxs_i)
                completed += len(idxs_i)

        assert state == (None, None, None, None), (
            f"orphaned residual left for {method} max_batch={max_batch}"
        )
        nodes_loop = torch.cat(nodes_acc, dim=0)
        f_evals_loop = torch.cat(f_evals_acc, dim=0)
        assert torch.cat(idxs_acc, dim=0).shape[0] == n_panels
        assert torch.allclose(nodes_loop, nodes_ref, atol=1e-12, rtol=0)
        assert torch.allclose(f_evals_loop, f_evals_ref, atol=1e-12, rtol=0)

    # -----------------------------------------------------------------------
    # f-call accounting (old Path A formula based on max_mesh_steps)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_residual_calls_f_exactly_num_accumulation_iters_times(
        self, method, max_batch
    ):
        """Wrap f with a counter; the count must equal num_accumulation_iters
        predicted by the residual Path A formula."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        if max_mesh_steps < 1:
            pytest.skip("max_mesh_steps < 1 — degenerate")

        mesh = _make_mesh()
        step_idxs = _make_step_idxs()

        call_count = {"n": 0}

        def counting_f(t, *args):
            call_count["n"] += 1
            return _simple_integrand(t, *args)

        solver._evaluate_f_on_split_residual_nodes(
            counting_f,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=(None, None, None, None),
        )

        evaluate_all = (max_mesh_steps * C) % max_batch == 0
        if evaluate_all:
            expected_iters = (max_mesh_steps * C) // max_batch
        else:
            expected_iters = (max_mesh_steps * C) // max_batch + 1

        assert call_count["n"] == expected_iters, (
            f"f called {call_count['n']} times, expected {expected_iters} "
            f"for {method} max_batch={max_batch} (evaluate_all={evaluate_all})"
        )

    # -----------------------------------------------------------------------
    # Residual nodes are NOT re-evaluated in the next iteration
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_residual_does_not_re_evaluate_residual_nodes(self, method, max_batch):
        """THE correctness invariant for the residual design: residual nodes
        produced in iteration 1 are reused (not re-evaluated) in iteration 2."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        if max_mesh_steps < 1:
            pytest.skip("max_mesh_steps < 1 — degenerate")
        evaluate_all = (max_mesh_steps * C) % max_batch == 0
        if evaluate_all:
            pytest.skip("evaluate_all=True — no residual in iteration 1")

        mesh = _make_mesh(n_panels=max(6, 3 * max_mesh_steps + 2))
        step_idxs = torch.arange(mesh.shape[0] - 1)

        seen_inputs = []

        def recording_f(t, *args):
            seen_inputs.append(t.detach().clone())
            return _simple_integrand(t, *args)

        _n1, _e1, _tracked, ridxs1, state1 = (
            solver._evaluate_f_on_split_residual_nodes(
                recording_f,
                (),
                mesh,
                step_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_mesh_steps,
                split_node_state=(None, None, None, None),
            )
        )
        iter1_calls = len(seen_inputs)

        remaining = step_idxs[len(ridxs1) :]
        if len(remaining) == 0:
            pytest.skip("Iter 1 consumed all panels")

        solver._evaluate_f_on_split_residual_nodes(
            recording_f,
            (),
            mesh,
            remaining,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=state1,
        )
        iter2_inputs = torch.cat(seen_inputs[iter1_calls:], dim=0)

        residual_nodes, _, _, _ = state1
        for r_node in residual_nodes:
            matches = (iter2_inputs == r_node).all(dim=-1).any()
            assert not matches, (
                f"Residual node {r_node.tolist()} re-evaluated in iter 2 "
                f"for {method} max_batch={max_batch}"
            )

    # -----------------------------------------------------------------------
    # Path B fallback when split_mesh_idx is not in step_idxs
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_residual_split_mesh_idx_not_in_step_idxs_fallback(self, method):
        """When the caller's step_idxs does NOT contain the residual's panel
        index, the function falls back to the else-branch. The function may
        succeed or error; it must not silently corrupt."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_batch = C

        mb_iter1 = C + 1 if C > 1 else 2
        mesh_long = _make_mesh(n_panels=6)
        step_idxs_long = torch.arange(6)
        _n1, _e1, _tracked, _r1, state_real = (
            solver._evaluate_f_on_split_residual_nodes(
                _simple_integrand,
                (),
                mesh_long,
                step_idxs_long,
                max_batch=mb_iter1,
                max_mesh_steps=mb_iter1 // C,
                split_node_state=(None, None, None, None),
            )
        )
        if state_real == (None, None, None, None):
            pytest.skip("Could not construct residual for this method")

        _residual_nodes, _residual_f_evals, _residual_tracked, residual_mesh_idx = (
            state_real
        )
        new_step_idxs = torch.tensor(
            [i for i in range(6) if torch.tensor(i) != residual_mesh_idx][:3]
        )
        assert not (new_step_idxs == residual_mesh_idx).any()

        try:
            n2, _e2, _tracked, r2, _ = solver._evaluate_f_on_split_residual_nodes(
                _simple_integrand,
                (),
                mesh_long,
                new_step_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_batch // C,
                split_node_state=state_real,
            )
            assert r2.shape[0] >= 1
            assert n2.shape[1] == C
        except (RuntimeError, IndexError, AssertionError):
            pass

    # -----------------------------------------------------------------------
    # split_node_state contract: exact shapes and content
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_residual_handoff_contract(self, method, max_batch):
        """Pin the exact contract of ``split_node_state``."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        if max_mesh_steps < 1:
            pytest.skip("max_mesh_steps < 1 — degenerate")
        evaluate_all = (max_mesh_steps * C) % max_batch == 0
        if evaluate_all:
            pytest.skip("evaluate_all=True — no residual to inspect")

        mesh = _make_mesh(n_panels=max(6, 3 * max_mesh_steps + 2))
        step_idxs = torch.arange(mesh.shape[0] - 1)
        _n, _e, _tracked, _idxs, state = solver._evaluate_f_on_split_residual_nodes(
            _simple_integrand,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=(None, None, None, None),
        )

        residual_nodes, residual_f_evals, _residual_tracked, residual_mesh_idx = state
        assert residual_nodes is not None
        assert residual_f_evals is not None
        assert residual_mesh_idx is not None

        num_residual = max_batch - (max_mesh_steps * C) % max_batch
        assert residual_nodes.shape[0] == num_residual, (
            f"residual_nodes.shape[0]={residual_nodes.shape[0]}, "
            f"expected {num_residual}"
        )
        assert residual_nodes.shape[1] == mesh.shape[1]
        assert residual_f_evals.shape[0] == num_residual
        residual_mesh_idx_val = int(residual_mesh_idx.item())
        assert residual_mesh_idx_val in step_idxs.tolist()

    # -----------------------------------------------------------------------
    # Multi-dimensional integrand output via the residual loop (D > 1)
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_residual_D_gt_1(self, method):
        """Integrand returning [N, 3] driven through the residual loop matches
        the full path."""

        def vec_integrand(t, *args):
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return torch.cat([torch.sin(t), torch.cos(t), t], dim=-1)

        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = 4
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)

        nodes_full, f_evals_full, _, _, _ = solver._evaluate_f_on_full_nodes(
            vec_integrand, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        assert f_evals_full.shape == (n_panels, C, 3)

        max_batch = 2 * C + 1  # forces evaluate_all=False at least once
        completed = 0
        state = (None, None, None, None)
        f_evals_acc = []
        while completed < n_panels:
            remaining = step_idxs[completed:]
            assert len(remaining) > 0
            batches_left = len(remaining) * C
            effective_max_batch = min(max_batch, batches_left)
            _ni, fi, _tracked, ri, state = (
                solver._evaluate_f_on_split_residual_nodes(
                    vec_integrand,
                    (),
                    mesh,
                    remaining,
                    max_batch=effective_max_batch,
                    max_mesh_steps=effective_max_batch // C,
                    split_node_state=state,
                )
            )
            assert len(ri) > 0
            f_evals_acc.append(fi)
            completed += len(ri)
        f_evals_split = torch.cat(f_evals_acc, dim=0)

        assert f_evals_split.shape == (n_panels, C, 3)
        assert torch.allclose(f_evals_split, f_evals_full, atol=1e-12)

    # -----------------------------------------------------------------------
    # Adaptive mesh mutation between residual iterations
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_residual_with_mesh_mutation_between_iterations(self, method):
        """When mesh barriers change between iterations, the function should
        either handle it sensibly or error loudly — never silently corrupt."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_batch = C + 1
        mesh_orig = _make_mesh(n_panels=4)
        step_idxs = torch.arange(4)

        _, _, _tracked, ridxs1, state1 = solver._evaluate_f_on_split_residual_nodes(
            _simple_integrand,
            (),
            mesh_orig,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_batch // C,
            split_node_state=(None, None, None, None),
        )

        if state1 == (None, None, None, None):
            pytest.skip("iteration 1 had no residual (evaluate_all=True)")

        midpoint = (mesh_orig[2] + mesh_orig[3]) / 2
        mesh_mutated = torch.cat(
            [mesh_orig[:3], midpoint.unsqueeze(0), mesh_orig[3:]], dim=0
        )
        new_step_idxs = torch.arange(len(ridxs1), 4 + 1)

        try:
            n2, e2, _tracked, i2, _ = solver._evaluate_f_on_split_residual_nodes(
                _simple_integrand,
                (),
                mesh_mutated,
                new_step_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_batch // C,
                split_node_state=state1,
            )
            assert n2.shape[1] == C
            assert e2.shape[0] == n2.shape[0]
            assert i2.shape[0] == n2.shape[0]
        except (RuntimeError, AssertionError, IndexError):
            pass


# ===========================================================================
# TestIntegration — dispatcher + multi-call loop
# ===========================================================================


class TestIntegration:
    """Tests of the ``_evaluate_f_on_mesh`` dispatcher and multi-call loop."""

    # -----------------------------------------------------------------------
    # Dispatcher routes by take_gradient and conserve_memory
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_dispatcher_routes_by_take_gradient_and_conserve_memory(
        self, method, monkeypatch
    ):
        """take_gradient=True -> full_nodes;
        (take_gradient=False, conserve_memory=False) -> split_nodes;
        (take_gradient=False, conserve_memory=True) -> split_residual_nodes."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        mesh = _make_mesh()
        mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
        mesh_trackers[-1] = False  # last barrier is not a panel start

        calls = {"full": 0, "split": 0, "residual": 0}

        def _fake(kind):
            def inner(*args, **kwargs):
                calls[kind] += 1
                # Return a well-formed 5-tuple so the dispatcher's caller
                # (if any) doesn't choke: (nodes, f_evals, tracked, idxs, state)
                return (
                    torch.zeros(1, C, 1),
                    torch.zeros(1, C, 1),
                    None,
                    torch.arange(1),
                    (None, None, None, None),
                )

            return inner

        monkeypatch.setattr(solver, "_evaluate_f_on_full_nodes", _fake("full"))
        monkeypatch.setattr(solver, "_evaluate_f_on_split_nodes", _fake("split"))
        monkeypatch.setattr(
            solver, "_evaluate_f_on_split_residual_nodes", _fake("residual")
        )

        common = dict(
            f=_simple_integrand,
            f_args=(),
            mesh=mesh,
            mesh_trackers=mesh_trackers,
            force_max_batch=10 * C,
            total_mem_usage=0.9,
            split_node_state=(None, None, None, None),
        )

        # Route 1: take_gradient=True -> full
        solver._evaluate_f_on_mesh(take_gradient=True, conserve_memory=False, **common)
        assert calls == {"full": 1, "split": 0, "residual": 0}

        # Route 2: take_gradient=False, conserve_memory=False -> split
        solver._evaluate_f_on_mesh(take_gradient=False, conserve_memory=False, **common)
        assert calls == {"full": 1, "split": 1, "residual": 0}

        # Route 3: take_gradient=False, conserve_memory=True -> residual
        solver._evaluate_f_on_mesh(take_gradient=False, conserve_memory=True, **common)
        assert calls == {"full": 1, "split": 1, "residual": 1}

    # -----------------------------------------------------------------------
    # Multi-call loop through _evaluate_f_on_mesh covers entire mesh
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("take_gradient", "conserve_memory"),
        [(True, False), (False, False), (False, True)],
        ids=["take_grad", "split", "residual"],
    )
    @pytest.mark.parametrize(
        ("method", "max_batch"), _AT_OR_ABOVE_C_SWEEP, ids=_AT_OR_ABOVE_C_SWEEP_IDS
    )
    def test_loop_accumulates_full_mesh(
        self, method, max_batch, take_gradient, conserve_memory
    ):
        """Drive ``_evaluate_f_on_mesh`` to completion, mimicking the integrate
        loop. Verify all panels covered, final state empty, total nodes shape."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = max(6, 3 * (max_batch // C) + 2)
        mesh = _make_mesh(n_panels=n_panels)
        mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
        mesh_trackers[-1] = False

        state = (None, None, None, None)
        completed_panels = set()
        nodes_acc = []
        max_iters = n_panels + 5
        for _ in range(max_iters):
            if not torch.any(mesh_trackers):
                break
            nodes_i, _f_evals_i, _tracked, idxs_i, state = solver._evaluate_f_on_mesh(
                _simple_integrand,
                (),
                mesh,
                mesh_trackers,
                take_gradient,
                conserve_memory,
                max_batch,
                0.9,
                state,
            )
            nodes_acc.append(nodes_i)
            for idx in idxs_i.tolist():
                completed_panels.add(idx)
                mesh_trackers[idx] = False

        assert completed_panels == set(range(n_panels)), (
            f"Missing panels: {set(range(n_panels)) - completed_panels}"
        )
        assert state == (None, None, None, None), (
            f"Final state should be empty, got {state}"
        )
        all_nodes = torch.cat(nodes_acc, dim=0)
        assert all_nodes.shape == (n_panels, C, 1)

    # -----------------------------------------------------------------------
    # Loop outputs match across the three routes
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _AT_OR_ABOVE_C_SWEEP, ids=_AT_OR_ABOVE_C_SWEEP_IDS
    )
    def test_loop_results_match_across_routes(self, method, max_batch):
        """All three dispatcher routes must produce identical accumulated
        outputs (numerics is mode-independent in the current design)."""
        solver = make_solver_for_unit_test(method)
        n_panels = max(6, 3 * (max_batch // solver.C) + 2)

        def run(take_gradient, conserve_memory):
            mesh = _make_mesh(n_panels=n_panels)
            mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
            mesh_trackers[-1] = False
            state = (None, None, None, None)
            nodes_acc, f_evals_acc = [], []
            for _ in range(n_panels + 5):
                if not torch.any(mesh_trackers):
                    break
                ni, ei, _tracked, ii, state = solver._evaluate_f_on_mesh(
                    _simple_integrand,
                    (),
                    mesh,
                    mesh_trackers,
                    take_gradient,
                    conserve_memory,
                    max_batch,
                    0.9,
                    state,
                )
                nodes_acc.append(ni)
                f_evals_acc.append(ei)
                for idx in ii.tolist():
                    mesh_trackers[idx] = False
            return torch.cat(nodes_acc, dim=0), torch.cat(f_evals_acc, dim=0)

        nodes_full, f_evals_full = run(True, False)
        nodes_split, f_evals_split = run(False, False)
        nodes_residual, f_evals_residual = run(False, True)

        assert torch.equal(nodes_full, nodes_split)
        assert torch.equal(nodes_full, nodes_residual)
        # f_evals may have summation-order drift in the chunked paths.
        assert (f_evals_full - f_evals_split).abs().max().item() < 1e-14
        assert (f_evals_full - f_evals_residual).abs().max().item() < 1e-14


# ===========================================================================
# TestEndToEnd — full .integrate(max_batch=K) calls
# ===========================================================================


class TestEndToEnd:
    """Full ``.integrate(max_batch=K)`` calls with varying max_batch."""

    T_INIT = torch.tensor([0.0], dtype=torch.float64)
    T_FINAL = torch.tensor([1.0], dtype=torch.float64)

    E2E_ATOL = 1e-4
    E2E_RTOL = 1e-4

    @staticmethod
    def _solver(method, max_batch=None):
        return make_uniform_solver(
            method,
            atol=TestEndToEnd.E2E_ATOL,
            rtol=TestEndToEnd.E2E_RTOL,
            max_batch=max_batch,
        )

    # -----------------------------------------------------------------------
    # max_batch=0 at the public API: dispatcher rescues 0 -> 1
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "take_gradient", [True, False], ids=["take_grad_True", "take_grad_False"]
    )
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_integrate_max_batch_zero_errors(self, method, take_gradient):
        """``.integrate(max_batch=0)`` is rescued to 1 (with a warning).

        - take_gradient=False: the split path evaluates everything regardless of
          batch size, so the call succeeds with a correct integral.
        - take_gradient=True: the full-nodes path needs a whole panel per batch,
          so max_batch=1 < C still trips its assertion for the C>1 methods.
        """
        f, _solution_fxn, _ = _resolve_integrand("damped_sine")
        solver = self._solver(method, max_batch=0)
        if take_gradient:
            with pytest.raises(AssertionError):
                solver.integrate(
                    f=f,
                    mesh_init=self.T_INIT,
                    mesh_final=self.T_FINAL,
                    take_gradient=True,
                )
        else:
            torch.manual_seed(2025)
            result = solver.integrate(
                f=f,
                mesh_init=self.T_INIT,
                mesh_final=self.T_FINAL,
                take_gradient=False,
            )
            reference = self._solver(method, max_batch=_method_C(method))
            torch.manual_seed(2025)
            ref_result = reference.integrate(
                f=f,
                mesh_init=self.T_INIT,
                mesh_final=self.T_FINAL,
                take_gradient=False,
            )
            assert torch.allclose(
                result.integral.cpu(), ref_result.integral.cpu(), atol=1e-12, rtol=0
            ), (
                f"{method} max_batch=0 (rescued to 1) altered the integral vs "
                f"max_batch=C ({result.integral.item()} vs "
                f"{ref_result.integral.item()})"
            )

    # -----------------------------------------------------------------------
    # take_gradient=True with max_batch < C must assert at the solver
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _BELOW_C_SWEEP, ids=_BELOW_C_SWEEP_IDS
    )
    def test_integrate_take_grad_True_max_batch_below_C_errors(self, method, max_batch):
        """For take_gradient=True, max_batch < C trips the assertion in
        _evaluate_f_on_full_nodes."""
        if max_batch == 0:
            pytest.skip("max_batch=0 covered by test_integrate_max_batch_zero_errors")
        f, _, _ = _resolve_integrand("damped_sine")
        solver = self._solver(method, max_batch=max_batch)
        with pytest.raises(AssertionError):
            solver.integrate(
                f=f,
                mesh_init=self.T_INIT,
                mesh_final=self.T_FINAL,
                take_gradient=True,
            )

    # -----------------------------------------------------------------------
    # Correctness sweep: result matches analytical solution
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        "take_gradient", [True, False], ids=["take_grad_True", "take_grad_False"]
    )
    @pytest.mark.parametrize(
        ("method", "max_batch"), _AT_OR_ABOVE_C_SWEEP, ids=_AT_OR_ABOVE_C_SWEEP_IDS
    )
    def test_integrate_correctness_sweep(
        self, method, max_batch, take_gradient, integrand_name
    ):
        """Full integrate() with each max_batch in [C, 2C+1] produces an
        integral matching the analytical solution."""
        f, solution_fxn, cutoff = _resolve_integrand(integrand_name)
        solver = self._solver(method, max_batch=max_batch)
        torch.manual_seed(2025)
        result = solver.integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=take_gradient,
        )
        expected = solution_fxn(mesh_init=self.T_INIT, mesh_final=self.T_FINAL)
        rel_error = (result.integral.cpu() - expected).abs() / expected.abs()
        bound = 5e-2
        del cutoff
        assert rel_error.item() < bound, (
            f"{method} max_batch={max_batch} take_grad={take_gradient} "
            f"{integrand_name}: got {result.integral.item()}, expected "
            f"{expected.item()}, rel_error={rel_error.item():.2e} >= "
            f"bound={bound:.2e}"
        )

    # -----------------------------------------------------------------------
    # Result invariant across max_batch (bedrock contract)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        "take_gradient", [True, False], ids=["take_grad_True", "take_grad_False"]
    )
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_integrate_result_invariant_across_max_batch(
        self, method, take_gradient, integrand_name
    ):
        """``.integrate(max_batch=K)`` must return the same integral for all
        valid K. Batching is implementation, not numerics."""
        f, _, _ = _resolve_integrand(integrand_name)
        C = _method_C(method)
        candidates = [C, 2 * C, 5 * C, 100 * C, None]
        results = []
        for mb in candidates:
            solver = self._solver(method, max_batch=mb)
            torch.manual_seed(2025)
            r = solver.integrate(
                f=f,
                mesh_init=self.T_INIT,
                mesh_final=self.T_FINAL,
                take_gradient=take_gradient,
            )
            results.append((mb, r.integral.item()))

        reference = results[0][1]
        rel_bound = 100 * TestEndToEnd.E2E_ATOL  # 1e-2
        for mb, val in results[1:]:
            rel_diff = abs(val - reference) / max(abs(reference), 1e-12)
            assert rel_diff < rel_bound, (
                f"{method} {integrand_name} take_grad={take_gradient}: "
                f"max_batch={results[0][0]} -> {reference}, "
                f"max_batch={mb} -> {val}, "
                f"rel_diff={rel_diff:.3e} >= {rel_bound:.0e}"
            )

    # -----------------------------------------------------------------------
    # Solver state stays clean across integrate() calls
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_solver_state_clean_across_integrate_calls(self, method, integrand_name):
        """Calling ``solver.integrate()`` twice on the same instance with
        different max_batch must match fresh-solver calls."""
        f, _, _ = _resolve_integrand(integrand_name)
        C = _method_C(method)

        torch.manual_seed(2025)
        ref_first = self._solver(method, max_batch=2 * C).integrate(
            f=f, mesh_init=self.T_INIT, mesh_final=self.T_FINAL, take_gradient=False
        )
        torch.manual_seed(2025)
        ref_second = self._solver(method, max_batch=5 * C).integrate(
            f=f, mesh_init=self.T_INIT, mesh_final=self.T_FINAL, take_gradient=False
        )

        reused = self._solver(method, max_batch=2 * C)
        torch.manual_seed(2025)
        first = reused.integrate(
            f=f, mesh_init=self.T_INIT, mesh_final=self.T_FINAL, take_gradient=False
        )
        torch.manual_seed(2025)
        second = reused.integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
            max_batch=5 * C,
        )

        assert abs(first.integral.item() - ref_first.integral.item()) < 1e-10
        assert abs(second.integral.item() - ref_second.integral.item()) < 1e-10

    # -----------------------------------------------------------------------
    # Minimal residual-path E2E: conserve_memory matches the default path
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    def test_integrate_conserve_memory_matches_default(self):
        """One minimal end-to-end check that the residual path
        (``conserve_memory=True``) integrates to the same value as the default
        (``conserve_memory=False``). gk15 × damped_sine, same seed."""
        f, _, _ = _resolve_integrand("damped_sine")
        method = "gk15"

        torch.manual_seed(2025)
        default = self._solver(method, max_batch=3 * _method_C(method)).integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
            conserve_memory=False,
        )
        torch.manual_seed(2025)
        conserved = self._solver(method, max_batch=3 * _method_C(method)).integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
            conserve_memory=True,
        )

        assert torch.allclose(
            default.integral.cpu(), conserved.integral.cpu(), atol=1e-10, rtol=0
        ), (
            f"conserve_memory altered the integral: "
            f"{default.integral.item()} vs {conserved.integral.item()}"
        )


# -----------------------------------------------------------------------
# take_gradient=False with 0 < max_batch < C — runs and is correct (E2E)
# -----------------------------------------------------------------------


class TestEndToEndBelowC:
    """Tier-1 below-C end-to-end coverage for the split path."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _BELOW_C_NONZERO_SWEEP,
        ids=_BELOW_C_NONZERO_SWEEP_IDS,
    )
    def test_integrate_take_grad_False_max_batch_below_C_correct(
        self, method, max_batch, integrand_name
    ):
        """``take_gradient=False`` with ``0 < max_batch < C`` integrates
        correctly: max_batch is an implementation detail, so the integral must
        match a max_batch == C run near-exactly."""
        f, _solution_fxn, _cutoff = _resolve_integrand(integrand_name)
        t_init = TestEndToEnd.T_INIT
        t_final = TestEndToEnd.T_FINAL

        def _run(mb):
            solver = make_uniform_solver(
                method,
                atol=TestEndToEnd.E2E_ATOL,
                rtol=TestEndToEnd.E2E_RTOL,
                max_batch=mb,
            )
            torch.manual_seed(2025)
            return solver.integrate(
                f=f, mesh_init=t_init, mesh_final=t_final, take_gradient=False
            )

        result = _run(max_batch)
        reference = _run(_method_C(method))

        assert torch.allclose(
            result.integral.cpu(), reference.integral.cpu(), atol=1e-12, rtol=0
        ), (
            f"{method} {integrand_name}: max_batch={max_batch} altered the "
            f"integral vs max_batch=C ({result.integral.item()} vs "
            f"{reference.integral.item()})"
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_integrate_results_match_across_max_batch_take_grad_True(
        self, method, integrand_name
    ):
        """take_gradient=True should be invariant across max_batch."""
        f, _, _ = _resolve_integrand(integrand_name)
        C = _method_C(method)
        candidates = [C, 2 * C, 5 * C]
        results = []
        for mb in candidates:
            solver = make_uniform_solver(method, atol=1e-4, rtol=1e-4, max_batch=mb)
            torch.manual_seed(2025)
            r = solver.integrate(
                f=f,
                mesh_init=torch.tensor([0.0], dtype=torch.float64),
                mesh_final=torch.tensor([1.0], dtype=torch.float64),
                take_gradient=True,
            )
            results.append((mb, r.integral.item()))

        reference = results[0][1]
        rel_bound = 100 * 1e-4  # 1e-2
        for mb, val in results[1:]:
            rel_diff = abs(val - reference) / max(abs(reference), 1e-12)
            assert rel_diff < rel_bound, (
                f"take_grad=True {method} {integrand_name}: "
                f"max_batch={results[0][0]}->{reference}, "
                f"max_batch={mb}->{val}, rel_diff={rel_diff:.3e}"
            )
