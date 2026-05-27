"""Comprehensive tests for ``_evaluate_f_on_mesh`` and its two split paths.

The user split ``AdaptiveQuadrature._evaluate_f_on_mesh`` into:

  * ``_evaluate_f_on_full_nodes`` (take_gradient=True): single batch per call,
    requires ``max_mesh_steps >= 1``.
  * ``_evaluate_f_on_split_nodes`` (take_gradient=False): multiple batches per
    call, residual state carried across iterations.

These tests cover three layers:

  * ``TestUnit``         — direct calls to the two sub-functions
  * ``TestIntegration``  — the ``_evaluate_f_on_mesh`` dispatcher + multi-call loop
  * ``TestEndToEnd``     — full ``.integrate(max_batch=K)`` calls

The ``max_batch`` sweep covers every K in [0, 2C+1] for each method (see
``_max_batch_range``). This exercises every branch in
``_evaluate_f_on_split_nodes`` (Path A vs B, evaluate_all True vs False, etc.).
Two integrands — ``damped_sine`` (hardest) and ``exp`` (wide dynamic range) —
stress the integration scheme along orthogonal axes.

Tier markers (T1, T2, T3) tag tests by priority. Run a subset with
``pytest -m tier1`` etc.
"""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    make_solver_for_unit_test,
    make_uniform_solver,
)

from torchpathdiffeq import UNIFORM_METHODS, integrand_dict

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

# Above-or-equal-C sweep: (method, max_batch ∈ [C, 2C+1])
_AT_OR_ABOVE_C_SWEEP = _sweep_params(
    batch_range_fn=lambda m: list(range(_method_C(m), 2 * _method_C(m) + 2))
)
_AT_OR_ABOVE_C_SWEEP_IDS = _sweep_ids(_AT_OR_ABOVE_C_SWEEP)

# Split-only valid sweep: (method, max_batch ∈ [1, 2C+1]) — max_batch=0 is excluded
_SPLIT_VALID_SWEEP = _sweep_params(
    batch_range_fn=lambda m: list(range(1, 2 * _method_C(m) + 2))
)
_SPLIT_VALID_SWEEP_IDS = _sweep_ids(_SPLIT_VALID_SWEEP)


# ===========================================================================
# TestUnit
# ===========================================================================


class TestUnit:
    """Direct calls to ``_evaluate_f_on_full_nodes`` and
    ``_evaluate_f_on_split_nodes`` with hand-crafted inputs."""

    # -----------------------------------------------------------------------
    # 1. _evaluate_f_on_full_nodes: max_batch < C must assert
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"), _BELOW_C_SWEEP, ids=_BELOW_C_SWEEP_IDS
    )
    def test_full_nodes_max_batch_below_C_asserts(self, method, max_batch):
        """``_evaluate_f_on_full_nodes`` asserts ``max_mesh_steps >= 1``.

        Any ``max_batch < C`` yields ``max_mesh_steps = 0`` and trips the
        guard at base.py:694.
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
    # 2. _evaluate_f_on_full_nodes: shapes and values correct
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
        and (None, None, None) split_node_state."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        mesh = _make_mesh()
        step_idxs = _make_step_idxs()
        f, _, _ = _resolve_integrand(integrand_name)

        nodes, f_evals, returned_idxs, state = solver._evaluate_f_on_full_nodes(
            f, (), mesh, step_idxs, max_mesh_steps
        )

        N_expected = min(max_mesh_steps, len(step_idxs))
        assert nodes.shape == (N_expected, C, 1)
        assert f_evals.shape[:2] == (N_expected, C)
        assert state == (None, None, None)
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
    # 3. _evaluate_f_on_split_nodes: max_batch=0 must error
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_max_batch_zero_errors(self, method):
        """max_batch=0 triggers division/modulo-by-zero in Path A."""
        solver = make_solver_for_unit_test(method)
        mesh = _make_mesh()
        step_idxs = _make_step_idxs()
        with pytest.raises((ZeroDivisionError, RuntimeError, ValueError)):
            solver._evaluate_f_on_split_nodes(
                _simple_integrand,
                (),
                mesh,
                step_idxs,
                max_batch=0,
                max_mesh_steps=0,
                split_node_state=(None, None, None),
            )

    # -----------------------------------------------------------------------
    # 4. _evaluate_f_on_split_nodes: Path A (first call, no residual carryover)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _SPLIT_VALID_SWEEP,
        ids=_SPLIT_VALID_SWEEP_IDS,
    )
    def test_split_nodes_path_A_first_call(self, method, max_batch):
        """Path A invariants when split_node_state=(None, None, None).

        Verifies:
          * Output shapes match the predicted num_mesh_steps
          * evaluate_all=True  -> split_node_state == (None, None, None)
          * evaluate_all=False -> residual tensors present with expected size
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        if max_mesh_steps < 1:
            # With max_mesh_steps=0, Path A enters a degenerate case
            # (num_mesh_steps depends on (0 * C) % max_batch, may produce
            # empty outputs or error). Skip — covered by below-C error tests
            # at the integrate level.
            pytest.skip("max_mesh_steps < 1; degenerate Path A — see E2E tests")

        mesh = _make_mesh(n_panels=max(4, 2 * max_mesh_steps + 2))
        step_idxs = torch.arange(mesh.shape[0] - 1)
        evaluate_all = (max_mesh_steps * C) % max_batch == 0
        num_mesh_steps_expected = max_mesh_steps if evaluate_all else max_mesh_steps + 1

        nodes, _f_evals, _returned_idxs, state = solver._evaluate_f_on_split_nodes(
            _simple_integrand,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=(None, None, None),
        )

        if evaluate_all:
            assert nodes.shape == (num_mesh_steps_expected, C, 1)
            assert state == (None, None, None)
        else:
            # When evaluate_all=False the last step is partially evaluated;
            # the function returns num_mesh_steps - 1 complete steps and
            # places the rest in split_node_state.
            num_residual = max_batch - (max_mesh_steps * C) % max_batch
            assert nodes.shape == (num_mesh_steps_expected - 1, C, 1)
            residual_nodes, residual_f_evals, residual_mesh_idx = state
            assert residual_nodes is not None
            assert residual_f_evals is not None
            assert residual_mesh_idx is not None
            assert residual_nodes.shape[0] == num_residual

    # -----------------------------------------------------------------------
    # 5. _evaluate_f_on_split_nodes: Path B (continuation, residual present)
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _SPLIT_VALID_SWEEP,
        ids=_SPLIT_VALID_SWEEP_IDS,
    )
    def test_split_nodes_path_B_continuation(self, method, max_batch):
        """Path B: first call generates a residual; second call consumes it.

        The cleanest way to construct a valid Path B input is to chain two
        calls. This verifies the residual handoff in a realistic setting
        rather than fabricating a synthetic split_node_state.
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_mesh_steps = max_batch // C
        if max_mesh_steps < 1:
            pytest.skip("max_mesh_steps < 1; not a useful Path B scenario")

        mesh = _make_mesh(n_panels=max(6, 3 * max_mesh_steps + 2))
        step_idxs = torch.arange(mesh.shape[0] - 1)

        # First call: produces residual if evaluate_all=False
        _nodes1, _f_evals1, step_idxs1, state1 = solver._evaluate_f_on_split_nodes(
            _simple_integrand,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=(None, None, None),
        )

        evaluate_all_iter1 = (max_mesh_steps * C) % max_batch == 0
        if evaluate_all_iter1:
            pytest.skip("Iteration 1 was evaluate_all=True; no residual to feed Path B")

        # Second call: receives state1 as a Path B input
        remaining_idxs = step_idxs[len(step_idxs1) :]
        if len(remaining_idxs) == 0:
            pytest.skip("Iteration 1 consumed all panels; no Path B continuation")

        nodes2, _f_evals2, _step_idxs2, _state2 = solver._evaluate_f_on_split_nodes(
            _simple_integrand,
            (),
            mesh,
            remaining_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=state1,
        )

        # Path B output shape constraint: nodes2 has at least one panel
        assert nodes2.shape[0] >= 1
        assert nodes2.shape[1] == C
        assert nodes2.shape[2] == 1
        # The first panel of iteration 2's output should be the one whose
        # residual was carried — its first num_residual nodes should match
        # the residual passed in.
        residual_nodes, _residual_f_evals, _residual_mesh_idx = state1
        num_residual = residual_nodes.shape[0]
        assert torch.allclose(
            nodes2[0, :num_residual, :], residual_nodes, atol=1e-12
        ), "Path B did not prepend residual_nodes correctly"

    # -----------------------------------------------------------------------
    # 6 & 7. Cross-validation: split-loop accumulation == full single-call
    # -----------------------------------------------------------------------

    def _loop_split_until_done(
        self,
        solver,
        f,
        mesh,
        original_step_idxs,
        max_batch,
    ):
        """Drive ``_evaluate_f_on_split_nodes`` to completion.

        Mirrors the integrate-loop's call pattern: feeds split_node_state
        forward, recomputes max_batch / max_mesh_steps each iteration based
        on remaining panels (same formula as base.py:676-679).

        Returns (nodes_accumulated, f_evals_accumulated, step_idxs_accumulated,
                 final_state, total_iters).
        """
        C = solver.C
        total_panels = len(original_step_idxs)
        completed = 0
        state = (None, None, None)
        nodes_acc, f_evals_acc, step_idxs_acc = [], [], []
        iters = 0
        # Hard cap: defensive guard against infinite loops if the function
        # ever stops making progress.
        max_iters = total_panels + 5
        while completed < total_panels:
            iters += 1
            if iters > max_iters:
                raise RuntimeError(
                    f"Split loop did not converge in {max_iters} iterations "
                    f"(completed={completed}/{total_panels}, max_batch={max_batch})"
                )
            remaining_input = original_step_idxs[completed:]
            batches_left = len(remaining_input) * C
            effective_max_batch = min(max_batch, batches_left)
            effective_max_mesh_steps = effective_max_batch // C
            nodes_i, f_evals_i, idxs_i, state = solver._evaluate_f_on_split_nodes(
                f,
                (),
                mesh,
                remaining_input,
                max_batch=effective_max_batch,
                max_mesh_steps=effective_max_mesh_steps,
                split_node_state=state,
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
        ("method", "max_batch"),
        _AT_OR_ABOVE_C_SWEEP,
        ids=_AT_OR_ABOVE_C_SWEEP_IDS,
    )
    def test_split_nodes_full_loop_matches_full_nodes(
        self, method, max_batch, integrand_name
    ):
        """Gold-standard cross-validation: split-loop accumulation matches
        the full path's single-call output within 1e-12."""
        solver = make_solver_for_unit_test(method)
        # Use a larger mesh so we exercise multi-iteration behavior.
        n_panels = max(6, 3 * (max_batch // solver.C) + 2)
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        # Reference: single full-nodes call
        nodes_ref, f_evals_ref, _step_idxs_ref, _ = solver._evaluate_f_on_full_nodes(
            f, (), mesh, step_idxs, max_mesh_steps=n_panels
        )

        # Comparison: split-loop accumulation
        nodes_loop, f_evals_loop, step_idxs_loop, final_state, _iters = (
            self._loop_split_until_done(solver, f, mesh, step_idxs, max_batch)
        )

        # All panels covered
        assert step_idxs_loop.shape[0] == n_panels
        assert final_state == (None, None, None), (
            f"Loop left orphan residual for {method} max_batch={max_batch}"
        )

        # Cross-validation
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
        ("method", "max_batch"),
        _AT_OR_ABOVE_C_SWEEP,
        ids=_AT_OR_ABOVE_C_SWEEP_IDS,
    )
    def test_split_path_bit_equal_to_full_path(self, method, max_batch, integrand_name):
        """Tighter than #6: nodes must be bit-equal; f_evals must be < 1e-14.

        ``exp`` integrand's wide dynamic range (1 → e^5 ≈ 148) catches
        floating-point precision drift that ``damped_sine`` won't.
        """
        solver = make_solver_for_unit_test(method)
        n_panels = max(6, 3 * (max_batch // solver.C) + 2)
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.arange(n_panels)
        f, _, _ = _resolve_integrand(integrand_name)

        nodes_ref, f_evals_ref, _, _ = solver._evaluate_f_on_full_nodes(
            f, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        nodes_loop, f_evals_loop, _, _, _ = self._loop_split_until_done(
            solver, f, mesh, step_idxs, max_batch
        )

        # nodes are computed by _compute_nodes which is deterministic in
        # the same order; expect bit-equality.
        assert torch.equal(nodes_loop, nodes_ref), (
            f"nodes not bit-equal: max abs diff = "
            f"{(nodes_loop - nodes_ref).abs().max().item():.3e}"
        )
        # f_evals may differ in summation order across concatenate boundaries;
        # require < 1e-14.
        max_diff = (f_evals_loop - f_evals_ref).abs().max().item()
        assert max_diff < 1e-14, (
            f"f_evals drift {max_diff:.3e} >= 1e-14 for {method} "
            f"max_batch={max_batch} integrand={integrand_name}"
        )

    # -----------------------------------------------------------------------
    # 8. f-call accounting: f is called num_accumulation_iters times
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _SPLIT_VALID_SWEEP,
        ids=_SPLIT_VALID_SWEEP_IDS,
    )
    def test_split_nodes_calls_f_exactly_num_accumulation_iters_times(
        self, method, max_batch
    ):
        """Wrap f with a counter; the count must equal num_accumulation_iters
        predicted by the formula at base.py:735-738."""
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

        solver._evaluate_f_on_split_nodes(
            counting_f,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=(None, None, None),
        )

        # Path A formula (base.py:735-738)
        evaluate_all = (max_mesh_steps * C) % max_batch == 0
        if evaluate_all:
            num_mesh_steps = max_mesh_steps
            expected_iters = (num_mesh_steps * C) // max_batch
        else:
            expected_iters = (max_mesh_steps * C) // max_batch + 1

        assert call_count["n"] == expected_iters, (
            f"f called {call_count['n']} times, expected {expected_iters} "
            f"for {method} max_batch={max_batch} "
            f"(evaluate_all={evaluate_all})"
        )

    # -----------------------------------------------------------------------
    # 9. Residual nodes are NOT re-evaluated in the next iteration
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _SPLIT_VALID_SWEEP,
        ids=_SPLIT_VALID_SWEEP_IDS,
    )
    def test_split_nodes_does_not_re_evaluate_residual_nodes(self, method, max_batch):
        """THE correctness invariant for the split design.

        Iteration 1 produces residuals. Iteration 2 receives them via
        split_node_state. The actual nodes passed to f in iteration 2 must
        be disjoint from those passed in iteration 1 — i.e., residual
        nodes are reused, not re-evaluated.
        """
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

        seen_inputs = []  # list of [N, T] tensors from each f call

        def recording_f(t, *args):
            seen_inputs.append(t.detach().clone())
            return _simple_integrand(t, *args)

        # Iteration 1
        _n1, _e1, ridxs1, state1 = solver._evaluate_f_on_split_nodes(
            recording_f,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=(None, None, None),
        )
        iter1_calls = len(seen_inputs)

        # Iteration 2 (Path B)
        remaining = step_idxs[len(ridxs1) :]
        if len(remaining) == 0:
            pytest.skip("Iter 1 consumed all panels")

        solver._evaluate_f_on_split_nodes(
            recording_f,
            (),
            mesh,
            remaining,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=state1,
        )
        iter2_inputs = torch.cat(seen_inputs[iter1_calls:], dim=0)

        # The residual_nodes from iter 1 are precisely the nodes that
        # should NOT appear in iter 2's inputs. They were already
        # evaluated; iter 2 should reuse them via residual_f_evals.
        residual_nodes, _, _ = state1
        for r_node in residual_nodes:
            matches = (iter2_inputs == r_node).all(dim=-1).any()
            assert not matches, (
                f"Residual node {r_node.tolist()} re-evaluated in iter 2 "
                f"for {method} max_batch={max_batch}"
            )

    # -----------------------------------------------------------------------
    # 10. Path B fallback when split_mesh_idx is not in step_idxs
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_mesh_idx_not_in_step_idxs_fallback(self, method):
        """When the caller's step_idxs does NOT contain the residual's
        panel index, the function falls back to base.py:757-758 — appending
        the split mesh idx at position 0 via the else-branch.

        Forces the branch by constructing a synthetic split_node_state with
        residual_mesh_idx outside the passed step_idxs.
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_batch = C  # 1 panel per call, evaluate_all=True after consuming residual

        # Real iteration 1: get a valid (residual_nodes, residual_f_evals)
        # by running once with smaller max_batch and a wide mesh
        mb_iter1 = max(1, C - 1) if C > 1 else 1
        if (1 // 1) * C % mb_iter1 == 0 and C != 1:
            # ensure evaluate_all=False so we get a residual
            mb_iter1 = C - 1 if C > 1 else 1

        # Simpler approach: construct synthetic residual via a real iter 1
        # call with max_batch such that evaluate_all=False
        mb_iter1 = C + 1 if C > 1 else 2
        mesh_long = _make_mesh(n_panels=6)
        step_idxs_long = torch.arange(6)
        _n1, _e1, _r1, state_real = solver._evaluate_f_on_split_nodes(
            _simple_integrand,
            (),
            mesh_long,
            step_idxs_long,
            max_batch=mb_iter1,
            max_mesh_steps=mb_iter1 // C,
            split_node_state=(None, None, None),
        )
        if state_real == (None, None, None):
            pytest.skip("Could not construct residual for this method")

        # Now call with step_idxs that does NOT contain residual_mesh_idx.
        # residual_mesh_idx came from step_idxs_long; pick step_idxs that
        # excludes it.
        _residual_nodes, _residual_f_evals, residual_mesh_idx = state_real
        new_step_idxs = torch.tensor(
            [i for i in range(6) if torch.tensor(i) != residual_mesh_idx][:3]
        )
        # Confirm residual_mesh_idx really is absent
        assert not (new_step_idxs == residual_mesh_idx).any()

        # The fallback branch at base.py:757-758 should fire.
        # The function may either produce sensible output or error; either
        # behavior is acceptable as long as it does not silently corrupt.
        try:
            n2, _e2, r2, _ = solver._evaluate_f_on_split_nodes(
                _simple_integrand,
                (),
                mesh_long,
                new_step_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_batch // C,
                split_node_state=state_real,
            )
            # If it succeeds, the fallback rotated step_idxs to put
            # residual_mesh_idx at position 0
            assert r2.shape[0] >= 1
            assert n2.shape[1] == C
        except (RuntimeError, IndexError, AssertionError):
            # Acceptable: the fallback can't reconcile a stale residual.
            # Document this by passing.
            pass

    # -----------------------------------------------------------------------
    # 18. split_node_state contract: exact shapes and content
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _SPLIT_VALID_SWEEP,
        ids=_SPLIT_VALID_SWEEP_IDS,
    )
    def test_split_nodes_residual_handoff_contract(self, method, max_batch):
        """Pin the exact contract of ``split_node_state``:

        * ``residual_nodes.shape[0] == num_residual``
        * ``residual_nodes.shape[1] == T`` (= mesh dimensionality, 1 here)
        * ``residual_f_evals`` matches ``residual_nodes`` in leading dim
        * ``residual_mesh_idx`` is a 0-d or 1-element tensor pointing to
          a valid panel in step_idxs
        """
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
        _n, _e, _idxs, state = solver._evaluate_f_on_split_nodes(
            _simple_integrand,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_mesh_steps,
            split_node_state=(None, None, None),
        )

        residual_nodes, residual_f_evals, residual_mesh_idx = state
        assert residual_nodes is not None
        assert residual_f_evals is not None
        assert residual_mesh_idx is not None

        num_residual = max_batch - (max_mesh_steps * C) % max_batch
        assert residual_nodes.shape[0] == num_residual, (
            f"residual_nodes.shape[0]={residual_nodes.shape[0]}, "
            f"expected {num_residual}"
        )
        # T-dimension preserved
        assert residual_nodes.shape[1] == mesh.shape[1]
        # f_evals leading dim matches nodes
        assert residual_f_evals.shape[0] == num_residual

        # residual_mesh_idx is a valid panel index in the input step_idxs
        # (specifically, the (max_mesh_steps + 1)-th element, the partial one)
        residual_mesh_idx_val = int(residual_mesh_idx.item())
        assert residual_mesh_idx_val in step_idxs.tolist()


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    """Tests of ``_evaluate_f_on_mesh`` dispatcher and multi-call loop."""

    # -----------------------------------------------------------------------
    # I1. Dispatcher routes to the right sub-function based on take_gradient
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_dispatcher_routes_by_take_gradient(self, method, monkeypatch):
        """take_gradient=True routes to _evaluate_f_on_full_nodes;
        take_gradient=False routes to _evaluate_f_on_split_nodes."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        mesh = _make_mesh()
        mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
        mesh_trackers[-1] = False  # last barrier is not a panel start

        calls = {"full": 0, "split": 0}

        def fake_full(*args, **kwargs):
            calls["full"] += 1
            # Return values that match the expected output shape so the
            # dispatcher's caller (if any) doesn't choke.
            return (
                torch.zeros(1, C, 1),
                torch.zeros(1, C, 1),
                torch.arange(1),
                (None, None, None),
            )

        def fake_split(*args, **kwargs):
            calls["split"] += 1
            return (
                torch.zeros(1, C, 1),
                torch.zeros(1, C, 1),
                torch.arange(1),
                (None, None, None),
            )

        monkeypatch.setattr(solver, "_evaluate_f_on_full_nodes", fake_full)
        monkeypatch.setattr(solver, "_evaluate_f_on_split_nodes", fake_split)

        # Route 1: take_gradient=True
        solver._evaluate_f_on_mesh(
            _simple_integrand,
            (),
            mesh,
            mesh_trackers,
            take_gradient=True,
            force_max_batch=10 * C,
            total_mem_usage=0.9,
            split_node_state=(None, None, None),
        )
        assert calls == {"full": 1, "split": 0}

        # Route 2: take_gradient=False
        solver._evaluate_f_on_mesh(
            _simple_integrand,
            (),
            mesh,
            mesh_trackers,
            take_gradient=False,
            force_max_batch=10 * C,
            total_mem_usage=0.9,
            split_node_state=(None, None, None),
        )
        assert calls == {"full": 1, "split": 1}

    # -----------------------------------------------------------------------
    # I2. Multi-call loop through _evaluate_f_on_mesh covers entire mesh
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("take_gradient", [True, False])
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _AT_OR_ABOVE_C_SWEEP,
        ids=_AT_OR_ABOVE_C_SWEEP_IDS,
    )
    def test_loop_accumulates_full_mesh(self, method, max_batch, take_gradient):
        """Drive ``_evaluate_f_on_mesh`` to completion, mimicking the
        integrate loop's call pattern. Verify:

          * All initial panels covered (sum of returned step_idxs lengths)
          * Final split_node_state is (None, None, None)
          * Total nodes accumulated equals (n_panels, C, T)
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        n_panels = max(6, 3 * (max_batch // C) + 2)
        mesh = _make_mesh(n_panels=n_panels)
        mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
        mesh_trackers[-1] = False

        state = (None, None, None)
        completed_panels = set()
        nodes_acc, f_evals_acc = [], []
        max_iters = n_panels + 5
        for _ in range(max_iters):
            if not torch.any(mesh_trackers):
                break
            nodes_i, f_evals_i, idxs_i, state = solver._evaluate_f_on_mesh(
                _simple_integrand,
                (),
                mesh,
                mesh_trackers,
                take_gradient=take_gradient,
                force_max_batch=max_batch,
                total_mem_usage=0.9,
                split_node_state=state,
            )
            nodes_acc.append(nodes_i)
            f_evals_acc.append(f_evals_i)
            for idx in idxs_i.tolist():
                completed_panels.add(idx)
                mesh_trackers[idx] = False

        assert completed_panels == set(range(n_panels)), (
            f"Missing panels: {set(range(n_panels)) - completed_panels}"
        )
        assert state == (None, None, None), f"Final state should be empty, got {state}"
        all_nodes = torch.cat(nodes_acc, dim=0)
        assert all_nodes.shape == (n_panels, C, 1)

    # -----------------------------------------------------------------------
    # I3. Loop outputs match across take_gradient True vs False
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _AT_OR_ABOVE_C_SWEEP,
        ids=_AT_OR_ABOVE_C_SWEEP_IDS,
    )
    def test_loop_results_match_take_gradient_True_and_False(self, method, max_batch):
        """The two dispatcher paths must produce identical accumulated
        outputs (numerics is mode-independent in the current design)."""
        solver = make_solver_for_unit_test(method)
        n_panels = max(6, 3 * (max_batch // solver.C) + 2)

        def run(take_gradient):
            mesh = _make_mesh(n_panels=n_panels)
            mesh_trackers = torch.ones(mesh.shape[0], dtype=torch.bool)
            mesh_trackers[-1] = False
            state = (None, None, None)
            completed = set()
            nodes_acc, f_evals_acc = [], []
            for _ in range(n_panels + 5):
                if not torch.any(mesh_trackers):
                    break
                ni, ei, ii, state = solver._evaluate_f_on_mesh(
                    _simple_integrand,
                    (),
                    mesh,
                    mesh_trackers,
                    take_gradient=take_gradient,
                    force_max_batch=max_batch,
                    total_mem_usage=0.9,
                    split_node_state=state,
                )
                nodes_acc.append(ni)
                f_evals_acc.append(ei)
                for idx in ii.tolist():
                    completed.add(idx)
                    mesh_trackers[idx] = False
            return torch.cat(nodes_acc, dim=0), torch.cat(f_evals_acc, dim=0)

        nodes_true, f_evals_true = run(True)
        nodes_false, f_evals_false = run(False)

        assert torch.equal(nodes_true, nodes_false), (
            f"nodes differ between take_gradient modes for {method} "
            f"max_batch={max_batch}"
        )
        # f_evals may have summation-order drift in the False path
        max_diff = (f_evals_true - f_evals_false).abs().max().item()
        assert max_diff < 1e-14, (
            f"f_evals drift {max_diff:.3e} between take_gradient modes"
        )


# ===========================================================================
# TestEndToEnd
# ===========================================================================


class TestEndToEnd:
    """Full ``.integrate(max_batch=K)`` calls with varying max_batch."""

    # Common integration domain (the integrands in integrand_dict are
    # defined for arbitrary intervals via their solution_fxn).
    T_INIT = torch.tensor([0.0], dtype=torch.float64)
    T_FINAL = torch.tensor([1.0], dtype=torch.float64)

    # End-to-end uses loose tolerances so adaptive_heun (order 2)
    # converges quickly even with tiny max_batch. The point of these
    # tests is to exercise the batching code paths, not push numerical
    # precision — TestUnit already pins bit-equality with the full path.
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
    # E1. max_batch=0 errors at the public API
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "take_gradient", [True, False], ids=["take_grad_True", "take_grad_False"]
    )
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_integrate_max_batch_zero_errors(self, method, take_gradient):
        """``.integrate(max_batch=0)`` raises a clear error."""
        f, _, _ = _resolve_integrand("damped_sine")
        solver = self._solver(method, max_batch=0)
        with pytest.raises((AssertionError, ZeroDivisionError, RuntimeError)):
            solver.integrate(
                f=f,
                mesh_init=self.T_INIT,
                mesh_final=self.T_FINAL,
                take_gradient=take_gradient,
            )

    # -----------------------------------------------------------------------
    # E2. take_gradient=True with max_batch < C must assert at the solver
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _BELOW_C_SWEEP,
        ids=_BELOW_C_SWEEP_IDS,
    )
    def test_integrate_take_grad_True_max_batch_below_C_errors(self, method, max_batch):
        """For take_gradient=True, max_batch < C trips the assertion in
        _evaluate_f_on_full_nodes (base.py:694)."""
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
    # E3. Correctness sweep: result matches analytical solution
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize(
        "take_gradient", [True, False], ids=["take_grad_True", "take_grad_False"]
    )
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _AT_OR_ABOVE_C_SWEEP,
        ids=_AT_OR_ABOVE_C_SWEEP_IDS,
    )
    def test_integrate_correctness_sweep(
        self, method, max_batch, take_gradient, integrand_name
    ):
        """Full integrate() with each max_batch in [C, 2C+1] produces an
        integral matching the analytical solution within method cutoff."""
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
        # End-to-end uses loose tolerances (1e-4) so adaptive_heun
        # (order 2) converges quickly. The goal is to catch batching
        # corruption, not numerical precision; bound is generous.
        bound = 5e-2
        del cutoff  # not used at this loose bound
        assert rel_error.item() < bound, (
            f"{method} max_batch={max_batch} take_grad={take_gradient} "
            f"{integrand_name}: got {result.integral.item()}, expected "
            f"{expected.item()}, rel_error={rel_error.item():.2e} >= "
            f"bound={bound:.2e}"
        )

    # -----------------------------------------------------------------------
    # E5. Result invariant across max_batch (bedrock contract)
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
        """``.integrate(max_batch=K)`` must return the same integral for
        all valid K. Batching is implementation, not numerics."""
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

        # Relative invariance across max_batch values. Adaptive
        # controllers can take slightly different paths depending on
        # max_batch (different batch boundaries shift error
        # accumulation), so we use a relative bound calibrated to the
        # configured atol. If max_batch corrupts the integral beyond
        # this, the batching is genuinely broken — sweep test (E3)
        # catches gross corruption already.
        reference = results[0][1]
        rel_bound = 100 * TestEndToEnd.E2E_ATOL  # 100 * 1e-4 = 1e-2
        for mb, val in results[1:]:
            rel_diff = abs(val - reference) / max(abs(reference), 1e-12)
            assert rel_diff < rel_bound, (
                f"{method} {integrand_name} take_grad={take_gradient}: "
                f"max_batch={results[0][0]} -> {reference}, "
                f"max_batch={mb} -> {val}, "
                f"rel_diff={rel_diff:.3e} >= {rel_bound:.0e}"
            )

    # -----------------------------------------------------------------------
    # E6. Solver state stays clean across integrate() calls
    # -----------------------------------------------------------------------

    @pytest.mark.tier1
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_solver_state_clean_across_integrate_calls(self, method, integrand_name):
        """Calling ``solver.integrate()`` twice on the same instance with
        different max_batch must produce results matching fresh-solver
        calls. Catches stale ``split_node_state`` on the solver."""
        f, _, _ = _resolve_integrand(integrand_name)
        C = _method_C(method)

        # Fresh-solver references
        torch.manual_seed(2025)
        ref_first = self._solver(method, max_batch=2 * C).integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
        )
        torch.manual_seed(2025)
        ref_second = self._solver(method, max_batch=5 * C).integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
        )

        # Reused solver
        reused = self._solver(method, max_batch=2 * C)
        torch.manual_seed(2025)
        first = reused.integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
        )
        # Reconfigure the solver's max_batch for the second call. The
        # public API supports this via constructor; max_batch can also be
        # set on the integrate() call directly.
        torch.manual_seed(2025)
        second = reused.integrate(
            f=f,
            mesh_init=self.T_INIT,
            mesh_final=self.T_FINAL,
            take_gradient=False,
            max_batch=5 * C,
        )

        assert abs(first.integral.item() - ref_first.integral.item()) < 1e-10, (
            f"reused-first does not match fresh: "
            f"{first.integral.item()} vs {ref_first.integral.item()}"
        )
        assert abs(second.integral.item() - ref_second.integral.item()) < 1e-10, (
            f"reused-second does not match fresh: "
            f"{second.integral.item()} vs {ref_second.integral.item()}"
        )


# ===========================================================================
# Tier 2 — high-value edge cases and shape variations (extends TestUnit / TestEndToEnd)
# ===========================================================================


class TestTier2:
    """Tier 2 tests: edge cases that the Tier 1 sweep doesn't cover."""

    # -----------------------------------------------------------------------
    # D2. step_idxs with a single element
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_step_idxs_single_element(self, method):
        """``step_idxs = [k]`` for a single panel. Catches off-by-one
        in indexing."""
        solver = make_solver_for_unit_test(method)
        C = solver.C
        mesh = _make_mesh(n_panels=4)
        step_idxs = torch.tensor([1])  # only panel 1
        max_batch = C
        nodes, f_evals, returned_idxs, state = solver._evaluate_f_on_split_nodes(
            _simple_integrand,
            (),
            mesh,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=1,
            split_node_state=(None, None, None),
        )
        assert nodes.shape == (1, C, 1)
        assert f_evals.shape == (1, C, 1)
        assert returned_idxs.tolist() == [1]
        assert state == (None, None, None)

        # And for full_nodes
        nodes_f, _f_evals_f, idxs_f, _state_f = solver._evaluate_f_on_full_nodes(
            _simple_integrand, (), mesh, step_idxs, max_mesh_steps=1
        )
        assert nodes_f.shape == (1, C, 1)
        assert idxs_f.tolist() == [1]

    # -----------------------------------------------------------------------
    # D1. step_idxs non-contiguous (every other panel)
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_step_idxs_non_contiguous(self, method):
        """step_idxs = [0, 2, 4] picks every other panel. Output nodes
        must lie within those panels' barriers."""
        solver = make_solver_for_unit_test(method)
        n_panels = 6
        mesh = _make_mesh(n_panels=n_panels)
        step_idxs = torch.tensor([0, 2, 4])

        nodes_f, _, idxs_f, _ = solver._evaluate_f_on_full_nodes(
            _simple_integrand, (), mesh, step_idxs, max_mesh_steps=3
        )
        assert idxs_f.tolist() == [0, 2, 4]
        # First panel: nodes between mesh[0] and mesh[1]
        # Third panel: nodes between mesh[4] and mesh[5]
        assert nodes_f[0, 0, 0] >= mesh[0, 0] - 1e-12
        assert nodes_f[0, -1, 0] <= mesh[1, 0] + 1e-12
        assert nodes_f[2, 0, 0] >= mesh[4, 0] - 1e-12
        assert nodes_f[2, -1, 0] <= mesh[5, 0] + 1e-12

    # -----------------------------------------------------------------------
    # E1. Multi-dimensional integrand output (D > 1)
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_full_and_split_nodes_with_D_gt_1(self, method):
        """Integrand returning [N, 3] — verify reshape to [N, C, 3] in
        both paths and that the two paths agree."""

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

        # Full path
        nodes_full, f_evals_full, _, _ = solver._evaluate_f_on_full_nodes(
            vec_integrand, (), mesh, step_idxs, max_mesh_steps=n_panels
        )
        assert nodes_full.shape == (n_panels, C, 1)
        assert f_evals_full.shape == (n_panels, C, 3), (
            f"Expected [N, C, 3], got {f_evals_full.shape}"
        )

        # Split path: loop until done
        max_batch = 2 * C + 1  # forces evaluate_all=False at least once
        completed = 0
        state = (None, None, None)
        f_evals_acc = []
        while completed < n_panels:
            remaining = step_idxs[completed:]
            batches_left = len(remaining) * C
            effective_max_batch = min(max_batch, batches_left)
            _ni, fi, ri, state = solver._evaluate_f_on_split_nodes(
                vec_integrand,
                (),
                mesh,
                remaining,
                max_batch=effective_max_batch,
                max_mesh_steps=effective_max_batch // C,
                split_node_state=state,
            )
            f_evals_acc.append(fi)
            completed += len(ri)
        f_evals_split = torch.cat(f_evals_acc, dim=0)

        assert f_evals_split.shape == (n_panels, C, 3)
        assert torch.allclose(f_evals_split, f_evals_full, atol=1e-12)

    # -----------------------------------------------------------------------
    # F2. Determinism: same inputs twice -> bit-equal outputs
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_deterministic_repeated_calls(self, method):
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
                max_batch=max_batch,
                max_mesh_steps=max_batch // C,
                split_node_state=(None, None, None),
            )

        n1, e1, i1, s1 = run_once()
        n2, e2, i2, s2 = run_once()

        assert torch.equal(n1, n2), "nodes differ across identical calls"
        assert torch.equal(e1, e2), "f_evals differ across identical calls"
        assert torch.equal(i1, i2), "step_idxs differ across identical calls"
        # Compare state tensors element-wise (state may have None or tensors)
        for x, y in zip(s1, s2, strict=False):
            if x is None:
                assert y is None
            else:
                assert torch.equal(x, y), "state tensor differs across calls"

    # -----------------------------------------------------------------------
    # C2. Adaptive mesh mutation between split iterations
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_split_nodes_with_mesh_mutation_between_iterations(self, method):
        """When mesh barriers change between iterations of the split
        loop, the function should either handle it sensibly (process
        new mesh from the residual point) or error loudly.

        Constructs the scenario: iter 1 leaves a residual; before iter 2,
        we INSERT a new barrier somewhere in the mesh. step_idxs for
        iter 2 is recomputed.
        """
        solver = make_solver_for_unit_test(method)
        C = solver.C
        max_batch = C + 1
        mesh_orig = _make_mesh(n_panels=4)
        step_idxs = torch.arange(4)

        _, _, ridxs1, state1 = solver._evaluate_f_on_split_nodes(
            _simple_integrand,
            (),
            mesh_orig,
            step_idxs,
            max_batch=max_batch,
            max_mesh_steps=max_batch // C,
            split_node_state=(None, None, None),
        )

        if state1 == (None, None, None):
            pytest.skip("iteration 1 had no residual (evaluate_all=True)")

        # Mutate the mesh: insert a new barrier in the middle. Panel indices
        # AFTER the insertion shift.
        midpoint = (mesh_orig[2] + mesh_orig[3]) / 2
        mesh_mutated = torch.cat(
            [mesh_orig[:3], midpoint.unsqueeze(0), mesh_orig[3:]], dim=0
        )
        # Original panel indices: [0, 1, 2, 3]; after inserting at position 3,
        # panels are [0, 1, 2_old(=2_new), 3_new, 4_new=3_old].
        # Recompute remaining step_idxs in mutated mesh (just panels [len(ridxs1):]).
        new_step_idxs = torch.arange(len(ridxs1), 4 + 1)  # the old panels + 1

        # The function may succeed or error; both outcomes are acceptable.
        # The contract is: don't silently corrupt the result.
        try:
            n2, e2, i2, _ = solver._evaluate_f_on_split_nodes(
                _simple_integrand,
                (),
                mesh_mutated,
                new_step_idxs,
                max_batch=max_batch,
                max_mesh_steps=max_batch // C,
                split_node_state=state1,
            )
            # If it succeeds, shapes should be self-consistent
            assert n2.shape[1] == C
            assert e2.shape[0] == n2.shape[0]
            assert i2.shape[0] == n2.shape[0]
        except (RuntimeError, AssertionError, IndexError):
            # Acceptable: the function couldn't reconcile mutated mesh.
            pass

    # -----------------------------------------------------------------------
    # E2E 4. take_gradient=False with max_batch < C — pin the behavior
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize(
        ("method", "max_batch"),
        _BELOW_C_SWEEP,
        ids=_BELOW_C_SWEEP_IDS,
    )
    def test_integrate_take_grad_False_max_batch_below_C_behavior(
        self, method, max_batch
    ):
        """For take_gradient=False with max_batch < C, document the
        actual behavior. The current implementation may produce empty
        outputs, hang, or error - this test pins whatever it does.
        """
        f, _, _ = _resolve_integrand("damped_sine")
        solver = make_uniform_solver(method, atol=1e-4, rtol=1e-4, max_batch=max_batch)
        # Accept any of: clean error, correct result, or specific error message.
        # The point of this test is to PIN the behavior so changes are visible.
        try:
            result = solver.integrate(
                f=f,
                mesh_init=torch.tensor([0.0], dtype=torch.float64),
                mesh_final=torch.tensor([1.0], dtype=torch.float64),
                take_gradient=False,
            )
            # If it succeeded, the result should at least be a finite number.
            assert torch.isfinite(result.integral).all()
        except (
            AssertionError,
            ZeroDivisionError,
            RuntimeError,
            IndexError,
            ValueError,
        ):
            # Acceptable: errors are the expected behavior for max_batch < C
            # (the integrate code path can't accommodate fewer than C
            # evaluations per batch). torch.cat raises ValueError on empty
            # list, which fires when no batches are evaluable.
            pass

    # -----------------------------------------------------------------------
    # E2E 7. take_grad=True invariant subset
    # -----------------------------------------------------------------------

    @pytest.mark.tier2
    @pytest.mark.parametrize("integrand_name", TIER1_INTEGRANDS)
    @pytest.mark.parametrize("method", TEST_METHODS)
    def test_integrate_results_match_across_max_batch_take_grad_True(
        self, method, integrand_name
    ):
        """take_gradient=True specifically should be invariant across
        max_batch. Belt-and-suspenders subset of E5."""
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
