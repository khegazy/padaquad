"""Unit tests for VariableAdaptiveQuadrature._evaluate_adaptive_nodes.

The method that splits a subinterval and reuses its integrand values. The
integration loop does not call it (see ``docs/issues/variable-sampling-reuse.md``).
The uniform solver's version and the merge methods of both solvers were
removed: a merge drops the pair's shared mesh point (tested in
``test_prune_and_optimal.py``), and a uniform split is the loop's own.
"""

from __future__ import annotations

import torch
from _helpers import constant_integrand, make_variable_solver_for_unit_test

# ---------------------------------------------------------------------------
# Variable _evaluate_adaptive_nodes
# ---------------------------------------------------------------------------


class TestVariableEvaluateAdaptiveY:
    """Tests for VariableAdaptiveQuadrature._evaluate_adaptive_nodes."""

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
