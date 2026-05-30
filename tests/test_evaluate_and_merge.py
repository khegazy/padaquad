"""Unit tests for _evaluate_adaptive_nodes and _merge_excess_nodes (uniform + variable)."""

from __future__ import annotations

import torch
from _helpers import (
    constant_integrand,
    make_solver_for_unit_test,
    make_variable_solver_for_unit_test,
)

from padaquad import UNIFORM_METHODS

# ---------------------------------------------------------------------------
# Uniform _evaluate_adaptive_nodes
# ---------------------------------------------------------------------------


class TestUniformEvaluateAdaptiveY:
    """Tests for _UniformAdaptiveQuadratureBase._evaluate_adaptive_nodes."""

    def _make_t(self, solver, t_start, t_end, N):
        """Create [N, C, 1] time tensor with uniform steps."""
        boundaries = torch.linspace(t_start, t_end, N + 1, dtype=torch.float64)
        return solver._compute_nodes(
            boundaries[:-1].unsqueeze(-1), boundaries[1:].unsqueeze(-1)
        )

    def test_single_step_split_shapes(self):
        """Splitting 1 step produces [2, C, 1] for y and t."""
        solver = make_solver_for_unit_test("bosh3")
        C = solver.C  # 4
        t = self._make_t(solver, 0.0, 1.0, 1)
        y = torch.ones(1, C, 1, dtype=torch.float64)

        y_add, t_add = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0]), y, t
        )

        assert y_add.shape == (2, C, 1)
        assert t_add.shape == (2, C, 1)

    def test_multiple_steps_split(self):
        """Splitting 2 of 3 steps produces [4, C, 1]."""
        solver = make_solver_for_unit_test("bosh3")
        C = solver.C
        t = self._make_t(solver, 0.0, 1.0, 3)
        y = torch.ones(3, C, 1, dtype=torch.float64)

        y_add, t_add = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0, 2]), y, t
        )

        assert y_add.shape == (4, C, 1)
        assert t_add.shape == (4, C, 1)

    def test_midpoint_exact(self):
        """Step [0, 1] split: sub-step boundary at 0.5."""
        solver = make_solver_for_unit_test("bosh3")
        C = solver.C
        t = self._make_t(solver, 0.0, 1.0, 1)
        y = torch.ones(1, C, 1, dtype=torch.float64)

        _, t_add = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0]), y, t
        )

        # First sub-step ends at 0.5, second starts at 0.5
        assert torch.allclose(t_add[0, -1, 0], torch.tensor(0.5, dtype=torch.float64))
        assert torch.allclose(t_add[1, 0, 0], torch.tensor(0.5, dtype=torch.float64))

    def test_c_positions_in_substeps(self):
        """Quadrature positions in sub-steps match tableau c scaled to sub-interval."""
        solver = make_solver_for_unit_test("bosh3")
        c = UNIFORM_METHODS["bosh3"].tableau.c.to(torch.float64)
        t = self._make_t(solver, 0.0, 1.0, 1)
        y = torch.ones(1, solver.C, 1, dtype=torch.float64)

        _, t_add = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0]), y, t
        )

        # First sub-step [0, 0.5]: points = 0 + c * 0.5
        expected_first = c * 0.5
        assert torch.allclose(t_add[0, :, 0], expected_first)
        # Second sub-step [0.5, 1.0]: points = 0.5 + c * 0.5
        expected_second = 0.5 + c * 0.5
        assert torch.allclose(t_add[1, :, 0], expected_second)

    def test_fxn_values(self):
        """constant_integrand returns all ones."""
        solver = make_solver_for_unit_test("bosh3")
        t = self._make_t(solver, 0.0, 1.0, 1)
        y = torch.ones(1, solver.C, 1, dtype=torch.float64)

        y_add, _ = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0]), y, t
        )

        assert torch.allclose(y_add, torch.ones_like(y_add))


# ---------------------------------------------------------------------------
# Uniform _merge_excess_nodes
# ---------------------------------------------------------------------------


class TestUniformMergeExcessT:
    """Tests for _UniformAdaptiveQuadratureBase._merge_excess_nodes."""

    def _make_t(self, solver, t_start, t_end, N):
        """Create [N, C, 1] time tensor with uniform steps."""
        boundaries = torch.linspace(t_start, t_end, N + 1, dtype=torch.float64)
        return solver._compute_nodes(
            boundaries[:-1].unsqueeze(-1), boundaries[1:].unsqueeze(-1)
        )

    def test_merge_first_pair(self):
        """3 steps, merge pair 0: output has 2 steps."""
        solver = make_solver_for_unit_test("bosh3")
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.ones(3, 1, dtype=torch.float64) * 0.1
        se = torch.ones(3, 1, dtype=torch.float64) * 0.01

        t_p, _ss_p, _se_p = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))

        assert t_p.shape[0] == 2
        # First merged step spans from 0 to 2/3
        assert torch.allclose(t_p[0, 0, 0], torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(
            t_p[0, -1, 0], torch.tensor(2.0 / 3, dtype=torch.float64), atol=1e-10
        )

    def test_merge_last_pair(self):
        """3 steps, merge pair 1: output has 2 steps."""
        solver = make_solver_for_unit_test("bosh3")
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.ones(3, 1, dtype=torch.float64) * 0.1
        se = torch.ones(3, 1, dtype=torch.float64) * 0.01

        t_p, _, _ = solver._merge_excess_nodes(t, ss, se, torch.tensor([1]))

        assert t_p.shape[0] == 2

    def test_mesh_quadratures_accumulated(self):
        """Merged step sum = sum of original pair."""
        solver = make_solver_for_unit_test("bosh3")
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        se = torch.tensor([[0.01], [0.02], [0.03]], dtype=torch.float64)

        _, ss_p, _ = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))

        # Merged pair 0: 1+2=3, remaining step 2: 3
        assert torch.allclose(ss_p[0], torch.tensor([3.0], dtype=torch.float64))
        assert torch.allclose(ss_p[1], torch.tensor([3.0], dtype=torch.float64))

    def test_sum_errors_accumulated(self):
        """Merged step error = sum of original pair."""
        solver = make_solver_for_unit_test("bosh3")
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.ones(3, 1, dtype=torch.float64)
        se = torch.tensor([[0.01], [0.02], [0.03]], dtype=torch.float64)

        _, _, se_p = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))

        assert torch.allclose(se_p[0], torch.tensor([0.03], dtype=torch.float64))

    def test_empty_remove_idxs(self):
        """Empty remove_idxs returns inputs unchanged."""
        solver = make_solver_for_unit_test("bosh3")
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.ones(3, 1, dtype=torch.float64)
        se = torch.ones(3, 1, dtype=torch.float64)

        t_p, _ss_p, _se_p = solver._merge_excess_nodes(
            t, ss, se, torch.tensor([], dtype=torch.long)
        )

        assert torch.equal(t_p, t)
        assert torch.equal(_ss_p, ss)

    def test_time_ordering(self):
        """After merge, flattened t is non-decreasing."""
        solver = make_solver_for_unit_test("bosh3")
        t = self._make_t(solver, 0.0, 1.0, 4)
        ss = torch.ones(4, 1, dtype=torch.float64)
        se = torch.ones(4, 1, dtype=torch.float64) * 0.01

        t_p, _, _ = solver._merge_excess_nodes(t, ss, se, torch.tensor([1]))

        t_flat = torch.flatten(t_p, 0, 1)
        assert torch.all(t_flat[1:] - t_flat[:-1] + 1e-15 >= 0)

    def test_new_c_positions(self):
        """Merged step has c-interpolated points spanning wider interval."""
        solver = make_solver_for_unit_test("bosh3")
        c = UNIFORM_METHODS["bosh3"].tableau.c.to(torch.float64)
        t = self._make_t(solver, 0.0, 1.0, 2)
        ss = torch.ones(2, 1, dtype=torch.float64)
        se = torch.ones(2, 1, dtype=torch.float64)

        t_p, _, _ = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))

        # Merged step spans [0, 1]: points = 0 + c * 1
        assert torch.allclose(t_p[0, :, 0], c)


# ---------------------------------------------------------------------------
# Variable _evaluate_adaptive_nodes
# ---------------------------------------------------------------------------


class TestVariableEvaluateAdaptiveY:
    """Tests for _VariableAdaptiveQuadratureBase._evaluate_adaptive_nodes."""

    def test_reuses_old_evals(self):
        """Split step: old y values appear in y_add."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor([[[0.0], [1.0]]], dtype=torch.float64)
        y = torch.tensor([[[10.0], [20.0]]], dtype=torch.float64)

        y_add, _ = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0]), y, t
        )

        # Old values 10.0 and 20.0 should appear in the output
        y_flat = y_add.flatten()
        assert 10.0 in y_flat
        assert 20.0 in y_flat

    def test_midpoint_computed(self):
        """Step [0, 1] with C=2: midpoint at 0.5 evaluated."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor([[[0.0], [1.0]]], dtype=torch.float64)
        y = torch.ones(1, 2, 1, dtype=torch.float64)

        _, t_add = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0]), y, t
        )

        # Midpoint 0.5 should appear in t_add
        t_flat = t_add.flatten()
        assert torch.any(torch.isclose(t_flat, torch.tensor(0.5, dtype=torch.float64)))

    def test_output_shapes(self):
        """R=2 splits produce [4, C, D] and [4, C, T]."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        C = solver.C  # 2
        t = torch.tensor(
            [
                [[0.0], [0.5]],
                [[0.5], [0.8]],
                [[0.8], [1.0]],
            ],
            dtype=torch.float64,
        )
        y = torch.ones(3, C, 1, dtype=torch.float64)

        y_add, t_add = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0, 2]), y, t
        )

        assert y_add.shape == (4, C, 1)
        assert t_add.shape == (4, C, 1)

    def test_boundary_shared(self):
        """The boundary point between sub-steps is shared."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor([[[0.0], [1.0]]], dtype=torch.float64)
        y = torch.ones(1, 2, 1, dtype=torch.float64)

        _, t_add = solver._evaluate_adaptive_nodes(
            constant_integrand, torch.tensor([0]), y, t
        )

        # Last point of first sub-step == first point of second sub-step
        assert torch.allclose(t_add[0, -1], t_add[1, 0])


# ---------------------------------------------------------------------------
# Variable _merge_excess_nodes
# ---------------------------------------------------------------------------


class TestVariableMergeExcessT:
    """Tests for _VariableAdaptiveQuadratureBase._merge_excess_nodes."""

    def test_concatenate_and_subsample(self):
        """C=2: two steps merged → 3 combined → subsample to 2."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor(
            [
                [[0.0], [0.5]],
                [[0.5], [1.0]],
            ],
            dtype=torch.float64,
        )
        ss = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        se = torch.tensor([[0.01], [0.02]], dtype=torch.float64)

        t_p, _ss_p, _se_p = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))

        assert t_p.shape == (1, 2, 1)
        # Subsampled from [0, 0.5, 1.0] at indices [0, 2] → [0, 1.0]
        assert torch.allclose(t_p[0, 0, 0], torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(t_p[0, -1, 0], torch.tensor(1.0, dtype=torch.float64))

    def test_shared_boundary_not_duplicated(self):
        """Combined array has 2C-1=3 points, not 2C=4."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        C = solver.C  # 2
        t = torch.tensor(
            [
                [[0.0], [0.5]],
                [[0.5], [1.0]],
            ],
            dtype=torch.float64,
        )
        ss = torch.ones(2, 1, dtype=torch.float64)
        se = torch.ones(2, 1, dtype=torch.float64) * 0.01

        # The concatenation is t[0,:] + t[1,1:] = [0, 0.5] + [1.0] = 3 points = 2C-1
        t_p, _, _ = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))
        # After subsampling, back to C=2 points
        assert t_p.shape[1] == C

    def test_mesh_quadratures_accumulated(self):
        """After merge, sum is correct."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor(
            [
                [[0.0], [0.5]],
                [[0.5], [1.0]],
            ],
            dtype=torch.float64,
        )
        ss = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        se = torch.tensor([[0.01], [0.02]], dtype=torch.float64)

        _, ss_p, _ = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))

        assert torch.allclose(ss_p[0], torch.tensor([3.0], dtype=torch.float64))

    def test_time_ordering(self):
        """After merge, times are non-decreasing."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor(
            [
                [[0.0], [0.3]],
                [[0.3], [0.7]],
                [[0.7], [1.0]],
            ],
            dtype=torch.float64,
        )
        ss = torch.ones(3, 1, dtype=torch.float64)
        se = torch.ones(3, 1, dtype=torch.float64) * 0.01

        t_p, _, _ = solver._merge_excess_nodes(t, ss, se, torch.tensor([0]))

        t_flat = torch.flatten(t_p, 0, 1)
        assert torch.all(t_flat[1:] - t_flat[:-1] + 1e-15 >= 0)
