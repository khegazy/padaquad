"""Comprehensive tests for ``_evaluate_f_on_mesh`` and its evaluation paths.

``AdaptiveQuadrature._evaluate_f_on_mesh`` dispatches the integrand evaluation
to one of two functions, each returning ``(nodes, f_evals, tracked,
step_idxs, method_output)``:

  * ``_evaluate_f_on_full_nodes`` (``take_gradient=True``): a single ``f`` call
    over the whole subintervals that fit in one batch; requires
    ``max_mesh_steps >= 1``.
  * ``_evaluate_f_on_split_nodes`` (``take_gradient=False``): evaluates **all**
    waiting subintervals, in ``max_batch``-sized calls of ``f`` that may end
    inside a subinterval. The values of whole subintervals wait until they
    reach ``buffer_bytes`` (``math.inf``: the whole pass; 0: after every call),
    are then integrated and sent to ``values_device`` (the solver's device, or
    ``result_device`` with ``trade_speed_for_memory``; None drops them).

``method_output`` must equal ``_calculate_integral`` on the returned nodes and
values bit for bit, however the calls were cut (``_assert_integrated``).

These tests cover three layers:

  * ``TestFullNodes`` / ``TestSplitNodes`` — direct calls to the two functions.
  * ``TestIntegration`` — the ``_evaluate_f_on_mesh`` dispatcher + multi-call
    loop, across ``take_gradient``.
  * ``TestEndToEnd`` — full ``.integrate(max_batch=K)`` calls.

The ``max_batch`` sweep covers every K in [0, 2C+1] for each method (see
``_max_batch_range``). Two integrands — ``damped_sine`` (hardest) and ``exp``
(wide dynamic range) — stress the integration scheme along orthogonal axes.

Note on cross-validation: ``_evaluate_f_on_split_nodes`` evaluates the flat
node list in chunks, so it is only *bit*-equal to the full path's single
``f(nodes_flat)`` call when one call covers everything (i.e. ``max_batch >=
total_nodes``) or ``f`` is elementwise. With a small ``max_batch``, cross-checks
of ``f_evals`` use ``torch.allclose`` at a tight tolerance.

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

CPU = torch.device("cpu")

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


def _assert_integrated(solver, mesh, idxs, nodes, f_evals, method_output):
    """``method_output`` equals one ``_calculate_integral`` call, bit for bit.

    The split path integrates each call's completed subintervals separately;
    each subinterval's numbers depend only on its own nodes, values and mesh
    points (``idxs``: the evaluated subintervals' left mesh points), so the
    result must not depend on where the calls were cut.
    """
    expected = solver._calculate_integral(nodes, f_evals, mesh[idxs], mesh[idxs + 1])
    for field in (
        "integral",
        "integral_error",
        "mesh_quadratures",
        "mesh_quadrature_errors",
        "h",
    ):
        assert torch.equal(getattr(method_output, field), getattr(expected, field)), (
            f"{field} differs from one _calculate_integral call"
        )


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
        """Outputs match independent f-evaluation, with correct shapes and
        the integral of the returned nodes and values."""
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
        _assert_integrated(solver, mesh, returned_idxs, nodes, f_evals, state)
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
# TestSplitNodes — _evaluate_f_on_split_nodes (take_gradient=False path)
# ===========================================================================


class TestSplitNodes:
    """Direct calls to ``_evaluate_f_on_split_nodes``.

    Signature: ``(f, f_args, mesh, step_idxs, max_batch, values_device,
    buffer_bytes)``.
    One call evaluates every panel in ``step_idxs`` (chunking ``f`` into
    ``max_batch``-sized calls only for memory) and integrates them.
    """

    # -----------------------------------------------------------------------
    # max_batch=0 must error (explicit guard for direct callers)
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
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                values_device=CPU,
                buffer_bytes=0,
                max_batch=0,
            )

    # -----------------------------------------------------------------------
    # One call evaluates and integrates ALL panels
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _SPLIT_VALID_SWEEP, ids=_SPLIT_VALID_SWEEP_IDS
    )
    def test_split_nodes_evaluates_all_panels(self, method, max_batch):
        """For every valid max_batch a single call returns all panels,
        integrated as one ``_calculate_integral`` call would."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        # Use a mesh large enough that the chunking loop runs many times for
        # small max_batch (n_panels * C >> max_batch).
        n_panels = 6
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)

        nodes, f_evals, tracked, returned_idxs, state = (
            solver._evaluate_f_on_split_nodes(
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                values_device=CPU,
                buffer_bytes=0,
                max_batch=max_batch,
            )
        )

        assert nodes.shape == (n_panels, C, 1)
        assert f_evals.shape == (n_panels, C, 1)
        assert tracked is None
        assert torch.equal(returned_idxs, step_idxs)
        _assert_integrated(solver, mesh, step_idxs, nodes, f_evals, state)

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
            f,
            (),
            mesh,
            step_idxs,
            values_device=CPU,
            buffer_bytes=0,
            max_batch=n_panels * C,
        )

        _assert_integrated(solver, mesh, step_idxs, nodes_split, f_evals_split, state)
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
            f,
            (),
            mesh,
            step_idxs,
            values_device=CPU,
            buffer_bytes=0,
            max_batch=max_batch,
        )

        _assert_integrated(solver, mesh, step_idxs, nodes_split, f_evals_split, state)
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
            counting_f,
            (),
            mesh,
            step_idxs,
            values_device=CPU,
            buffer_bytes=0,
            max_batch=max_batch,
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
            f,
            (),
            mesh,
            step_idxs,
            values_device=CPU,
            buffer_bytes=0,
            max_batch=max_batch,
        )

        _assert_integrated(solver, mesh, step_idxs, nodes_split, f_evals_split, state)
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
            vec_integrand,
            (),
            mesh,
            step_idxs,
            values_device=CPU,
            buffer_bytes=0,
            max_batch=2 * C + 1,
        )
        _assert_integrated(solver, mesh, step_idxs, nodes_split, f_evals_split, state)
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
            tracked_integrand,
            (),
            mesh,
            step_idxs,
            values_device=CPU,
            buffer_bytes=0,
            max_batch=C + 1,
        )

        _assert_integrated(solver, mesh, step_idxs, nodes, f_evals, state)
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
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                values_device=CPU,
                buffer_bytes=0,
                max_batch=C,
            )
        )
        assert nodes.shape == (1, C, 1)
        assert f_evals.shape == (1, C, 1)
        assert returned_idxs.tolist() == [1]
        _assert_integrated(solver, mesh, returned_idxs, nodes, f_evals, state)

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
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                values_device=CPU,
                buffer_bytes=0,
                max_batch=max_batch,
            )

        n1, e1, _t1, i1, s1 = run_once()
        n2, e2, _t2, i2, s2 = run_once()

        assert torch.equal(n1, n2), "nodes differ across identical calls"
        assert torch.equal(e1, e2), "f_evals differ across identical calls"
        assert torch.equal(i1, i2), "step_idxs differ across identical calls"
        assert torch.equal(s1.mesh_quadratures, s2.mesh_quadratures)
        assert torch.equal(s1.integral, s2.integral)


# ===========================================================================
# TestIntegration — dispatcher + multi-call loop
# ===========================================================================


class TestIntegration:
    """Tests of the ``_evaluate_f_on_mesh`` dispatcher and multi-call loop."""

    # -----------------------------------------------------------------------
    # Dispatcher routes by take_gradient
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_dispatcher_routes_by_take_gradient(self, method, monkeypatch):
        """take_gradient=True -> full_nodes; take_gradient=False ->
        split_nodes, which receives the values device."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        mesh = _make_mesh()
        mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
        mesh_trackers[-1] = False  # last barrier is not a panel start

        calls = {"full": 0, "split": 0}
        devices = []

        def _fake(kind):
            def inner(*args, **kwargs):
                calls[kind] += 1
                if kind == "split":
                    devices.append(args[-2])
                return (
                    torch.zeros(1, C, 1),
                    torch.zeros(1, C, 1),
                    None,
                    torch.arange(1),
                    None,
                )

            return inner

        monkeypatch.setattr(solver, "_evaluate_f_on_full_nodes", _fake("full"))
        monkeypatch.setattr(solver, "_evaluate_f_on_split_nodes", _fake("split"))

        common = {
            "f": _simple_integrand,
            "f_args": (),
            "mesh": mesh,
            "step_idxs": torch.nonzero(mesh_trackers).flatten(),
            "values_device": CPU,
            "buffer_bytes": 0,
        }
        solver._batch = 10 * C

        solver._evaluate_f_on_mesh(take_gradient=True, **common)
        assert calls == {"full": 1, "split": 0}

        solver._evaluate_f_on_mesh(take_gradient=False, **common)
        assert calls == {"full": 1, "split": 1}
        assert devices == [CPU]

    # -----------------------------------------------------------------------
    # Multi-call loop through _evaluate_f_on_mesh covers entire mesh
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("take_gradient", [True, False], ids=["take_grad", "split"])
    @pytest.mark.parametrize(
        ("method", "max_batch"), _AT_OR_ABOVE_C_SWEEP, ids=_AT_OR_ABOVE_C_SWEEP_IDS
    )
    def test_loop_accumulates_full_mesh(self, method, max_batch, take_gradient):
        """Drive ``_evaluate_f_on_mesh`` to completion, mimicking the integrate
        loop. Verify all panels covered, each call integrated, total nodes shape."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = max(6, 3 * (max_batch // C) + 2)
        mesh = _make_mesh(n_panels=n_panels)
        mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
        mesh_trackers[-1] = False

        completed_panels = set()
        nodes_acc = []
        max_iters = n_panels + 5
        for _ in range(max_iters):
            if not torch.any(mesh_trackers):
                break
            solver._batch = max_batch
            nodes_i, f_evals_i, _tracked, idxs_i, method_output = (
                solver._evaluate_f_on_mesh(
                    _simple_integrand,
                    (),
                    mesh,
                    torch.nonzero(mesh_trackers).flatten(),
                    take_gradient,
                    CPU,
                    0,
                )
            )
            _assert_integrated(solver, mesh, idxs_i, nodes_i, f_evals_i, method_output)
            nodes_acc.append(nodes_i)
            for idx in idxs_i.tolist():
                completed_panels.add(idx)
                mesh_trackers[idx] = False

        assert completed_panels == set(range(n_panels)), (
            f"Missing panels: {set(range(n_panels)) - completed_panels}"
        )
        all_nodes = torch.cat(nodes_acc, dim=0)
        assert all_nodes.shape == (n_panels, C, 1)

    # -----------------------------------------------------------------------
    # Loop outputs match across the two routes
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _AT_OR_ABOVE_C_SWEEP, ids=_AT_OR_ABOVE_C_SWEEP_IDS
    )
    def test_loop_results_match_across_routes(self, method, max_batch):
        """Both dispatcher routes must produce identical accumulated outputs,
        and identical per-subinterval contributions (numerics is
        mode-independent)."""
        solver = make_solver_for_unit_test(method)
        n_panels = max(6, 3 * (max_batch // solver.C) + 2)

        def run(take_gradient):
            mesh = _make_mesh(n_panels=n_panels)
            mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
            mesh_trackers[-1] = False
            nodes_acc, f_evals_acc, quadratures_acc = [], [], []
            for _ in range(n_panels + 5):
                if not torch.any(mesh_trackers):
                    break
                solver._batch = max_batch
                ni, ei, _tracked, ii, mo = solver._evaluate_f_on_mesh(
                    _simple_integrand,
                    (),
                    mesh,
                    torch.nonzero(mesh_trackers).flatten(),
                    take_gradient,
                    CPU,
                    0,
                )
                nodes_acc.append(ni)
                f_evals_acc.append(ei)
                quadratures_acc.append(mo.mesh_quadratures)
                for idx in ii.tolist():
                    mesh_trackers[idx] = False
            return (
                torch.cat(nodes_acc, dim=0),
                torch.cat(f_evals_acc, dim=0),
                torch.cat(quadratures_acc, dim=0),
            )

        nodes_full, f_evals_full, mq_full = run(True)
        nodes_split, f_evals_split, mq_split = run(False)

        assert torch.equal(nodes_full, nodes_split)
        # f_evals may have summation-order drift in the chunked paths.
        assert (f_evals_full - f_evals_split).abs().max().item() < 1e-14
        if torch.equal(f_evals_full, f_evals_split):
            assert torch.equal(mq_full, mq_split)


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
    # max_batch=0 at the public API: ValueError before any work
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "take_gradient", [True, False], ids=["take_grad_True", "take_grad_False"]
    )
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_integrate_max_batch_zero_errors(self, method, take_gradient):
        """``.integrate(max_batch=0)`` raises ``ValueError`` before any work,
        with either ``take_gradient`` (it used to be rescued to 1 with a
        warning, which then failed an assertion with ``take_gradient=True``)."""
        f, _solution_fxn, _ = _resolve_integrand("damped_sine")
        solver = self._solver(method, max_batch=0)
        with pytest.raises(ValueError, match="max_batch must be an int >= 1"):
            solver.integrate(
                f=f,
                mesh_init=self.T_INIT,
                mesh_final=self.T_FINAL,
                take_gradient=take_gradient,
            )

    # -----------------------------------------------------------------------
    # take_gradient=True with max_batch < C raises ValueError before f
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _BELOW_C_SWEEP, ids=_BELOW_C_SWEEP_IDS
    )
    def test_integrate_take_grad_True_max_batch_below_C_errors(self, method, max_batch):
        """For take_gradient=True, a max_batch below one subinterval's C
        points (constructor's) raises ValueError naming it, the method and C,
        before f is called."""
        if max_batch == 0:
            pytest.skip("max_batch=0 covered by test_integrate_max_batch_zero_errors")
        f, _, _ = _resolve_integrand("damped_sine")
        calls = []
        solver = self._solver(method, max_batch=max_batch)
        with pytest.raises(
            ValueError,
            match=rf"max_batch {max_batch} is below the {solver.C} points of one "
            rf"subinterval of {method}: with take_gradient=True",
        ):
            solver.integrate(
                f=lambda t: calls.append(t) or f(t),
                mesh_init=self.T_INIT,
                mesh_final=self.T_FINAL,
                take_gradient=True,
            )
        assert calls == []

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
        result, _ = solver.integrate(
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
            r, _ = solver.integrate(
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
        ref_first, _ = self._solver(method, max_batch=2 * C).integrate(
            f=f, mesh_init=self.T_INIT, mesh_final=self.T_FINAL, take_gradient=False
        )
        torch.manual_seed(2025)
        ref_second, _ = self._solver(method, max_batch=5 * C).integrate(
            f=f, mesh_init=self.T_INIT, mesh_final=self.T_FINAL, take_gradient=False
        )

        reused = self._solver(method, max_batch=2 * C)
        torch.manual_seed(2025)
        first, _ = reused.integrate(
            f=f, mesh_init=self.T_INIT, mesh_final=self.T_FINAL, take_gradient=False
        )
        torch.manual_seed(2025)
        second, _ = reused.integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
            max_batch=5 * C,
        )

        assert abs(first.integral.item() - ref_first.integral.item()) < 1e-10
        assert abs(second.integral.item() - ref_second.integral.item()) < 1e-10


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
            )[0]

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
            r, _ = solver.integrate(
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
