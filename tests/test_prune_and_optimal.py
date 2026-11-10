"""Unit tests for _prune_excess_mesh and _get_optimal_mesh."""

from __future__ import annotations

import pytest
import torch
from _helpers import make_solver_for_unit_test, make_uniform_solver

from padaquad import integrand_dict

# ---------------------------------------------------------------------------
# _prune_excess_mesh
# ---------------------------------------------------------------------------


class TestPruneExcessT:
    """Tests for AdaptiveQuadrature._prune_excess_mesh."""

    def _make_t(self, solver, t_start, t_end, N):
        """Create [N, C, 1] time tensor with uniform steps."""
        boundaries = torch.linspace(t_start, t_end, N + 1, dtype=torch.float64)
        return solver._compute_nodes(
            boundaries[:-1].unsqueeze(-1), boundaries[1:].unsqueeze(-1)
        )

    def test_no_pairs_below_cut(self):
        """All error_ratios_2steps above remove_cut: no pruning."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 4)
        ss = torch.ones(4, 1, dtype=torch.float64)
        se = torch.ones(4, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.5, 0.8, 0.3])  # All > 0.1

        t_p, _ss_p, _se_p = solver._prune_excess_mesh(t, ss, se, er2)

        assert t_p.shape[0] == 4
        assert torch.equal(t_p, t)

    def test_one_pair_pruned(self):
        """First pair below cut: output has N-1 steps."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 4)
        ss = torch.ones(4, 1, dtype=torch.float64)
        se = torch.ones(4, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.8, 0.9])  # Pair 0 below 0.1 / 2**(3 - 1)

        t_p, _, _ = solver._prune_excess_mesh(t, ss, se, er2)

        assert t_p.shape[0] == 3

    def test_empty_error_ratios(self):
        """Single step (empty 2steps array): returns unchanged."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 1)
        ss = torch.ones(1, 1, dtype=torch.float64)
        se = torch.ones(1, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([])

        t_p, _ss_p, _se_p = solver._prune_excess_mesh(t, ss, se, er2)

        assert t_p.shape[0] == 1
        assert torch.equal(t_p, t)

    def test_no_adjacent_merges(self):
        """Adjacent pairs both below cut: _rec_remove prevents double merge."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 5)
        ss = torch.ones(5, 1, dtype=torch.float64)
        se = torch.ones(5, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.01, 0.8, 0.01])  # Pairs 0,1 adjacent

        t_p, _, _ = solver._prune_excess_mesh(t, ss, se, er2)

        # Only non-adjacent pairs merged: pair 0 and pair 3 (indices 0 and 3)
        # Pair 1 blocked by adjacency. Result: 5 - 2 = 3 steps
        assert t_p.shape[0] == 3

    def test_mesh_quadratures_accumulated(self):
        """After merge, sum of pair equals sum of originals."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        se = torch.ones(3, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.8])  # Pair 0 below cut

        _, ss_p, _ = solver._prune_excess_mesh(t, ss, se, er2)

        assert ss_p.shape[0] == 2
        # Merged: 1+2=3, remaining: 3
        assert torch.allclose(ss_p[0], torch.tensor([3.0], dtype=torch.float64))

    @pytest.mark.parametrize("method", ["bosh3", "dopri5", "gk21", "cc17"])
    def test_merge_cut_accounts_for_merged_step_error(self, method):
        """A pair merges only when its summed ratio is below
        remove_cut / 2**(k - 1): the merged step's error is ~2**(k - 1) times
        the sum, k = error_exponent. A pair between that cut and remove_cut
        (merged before) is kept."""
        solver = make_solver_for_unit_test(method)
        merge_cut = solver.remove_cut / 2 ** (solver.method.error_exponent - 1)
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.ones(3, 1, dtype=torch.float64)
        se = torch.ones(3, 1, dtype=torch.float64) * 0.01

        kept, _, _ = solver._prune_excess_mesh(
            t, ss, se, torch.tensor([2 * merge_cut, 0.8])
        )
        merged, _, _ = solver._prune_excess_mesh(
            t, ss, se, torch.tensor([0.5 * merge_cut, 0.8])
        )

        assert 2 * merge_cut < solver.remove_cut
        assert kept.shape[0] == 3
        assert merged.shape[0] == 2


# ---------------------------------------------------------------------------
# _get_optimal_mesh
# ---------------------------------------------------------------------------


class TestGetOptimalTStepBarriers:
    """Tests for _get_optimal_mesh: post-convergence mesh optimization."""

    def _make_record(self, solver, N, integral_val=1.0, error_scale=0.001):
        """Build a record dict with N uniform steps in [0, 1]."""
        boundaries = torch.linspace(0, 1, N + 1, dtype=torch.float64)
        t = solver._compute_nodes(
            boundaries[:-1].unsqueeze(-1), boundaries[1:].unsqueeze(-1)
        )
        return {
            "nodes": t,
            "mesh_quadratures": torch.ones(N, 1, dtype=torch.float64)
            * (integral_val / N),
            "mesh_quadrature_errors": torch.ones(N, 1, dtype=torch.float64)
            * error_scale,
            "integral": torch.tensor([integral_val], dtype=torch.float64),
        }

    def test_over_resolved_pruned(self):
        """Record with tiny errors → fewer barriers."""
        solver = make_solver_for_unit_test(atol=1e-3, rtol=1e-3)
        N = 20
        record = self._make_record(solver, N, error_scale=1e-10)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        assert len(barriers_opt) < len(barriers)

    def test_under_resolved_refined(self):
        """Record with large errors → more barriers."""
        solver = make_solver_for_unit_test(atol=1e-12, rtol=1e-12)
        N = 3
        record = self._make_record(solver, N, error_scale=10.0)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        assert len(barriers_opt) > len(barriers)

    def test_output_sorted(self):
        """Output barriers are strictly increasing."""
        solver = make_solver_for_unit_test(atol=1e-6, rtol=1e-6)
        N = 10
        record = self._make_record(solver, N, error_scale=0.001)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        diffs = barriers_opt[1:, 0] - barriers_opt[:-1, 0]
        assert torch.all(diffs > 0), f"Barriers not sorted: {barriers_opt[:, 0]}"

    def test_endpoints_preserved(self):
        """First and last barriers match 0 and 1."""
        solver = make_solver_for_unit_test(atol=1e-6, rtol=1e-6)
        N = 10
        record = self._make_record(solver, N, error_scale=0.001)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        assert torch.allclose(barriers_opt[0], torch.tensor([0.0], dtype=torch.float64))
        assert torch.allclose(
            barriers_opt[-1], torch.tensor([1.0], dtype=torch.float64)
        )


@pytest.mark.parametrize("method", ["dopri5", "gk7", "cc17"])
def test_optimal_mesh_passes_when_reevaluated(method):
    """Re-integrating the same integrand from its own mesh_optimal fails no
    step: merged steps are predicted to stay below tolerance. With the old cut
    (summed ratio < remove_cut) 7-22 % of these steps failed. max_path_change
    (tiny) turns any failing starting step into converged=False."""
    torch.manual_seed(0)
    f = integrand_dict["damped_sine"][0]
    solver = make_uniform_solver(
        method, atol=1e-9, rtol=1e-9, device="cpu", max_path_change=1e-12
    )
    solver.integrate(
        f=f,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([1.0], dtype=torch.float64),
        take_gradient=False,
    )
    out = solver.integrate(f=f, same_integrand=True, take_gradient=False)
    assert out.converged is True
