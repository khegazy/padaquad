"""Unit tests for adaptive_quadrature factory, _get_tableau_b, and _get_num_tableau_c."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    REMOVE_CUT,
    make_solver_for_unit_test,
    make_variable_solver_for_unit_test,
)

from padaquad.base import steps
from padaquad.runge_kutta import (
    UniformAdaptiveQuadrature,
    VariableAdaptiveQuadrature,
    adaptive_quadrature,
)

# ---------------------------------------------------------------------------
# adaptive_quadrature factory
# ---------------------------------------------------------------------------


class TestGetParallelRKSolver:
    """Tests for the factory function that creates uniform or variable solvers."""

    def test_uniform_string(self):
        """String 'uniform' returns a uniform solver."""
        solver = adaptive_quadrature(
            sampling_type="uniform",
            method="bosh3",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, UniformAdaptiveQuadrature)

    def test_variable_string(self):
        """String 'variable' returns a variable solver."""
        solver = adaptive_quadrature(
            sampling_type="variable",
            method="adaptive_heun",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, VariableAdaptiveQuadrature)

    def test_uniform_enum(self):
        """steps.ADAPTIVE_UNIFORM enum returns a uniform solver."""
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_UNIFORM,
            method="bosh3",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, UniformAdaptiveQuadrature)

    def test_variable_enum(self):
        """steps.ADAPTIVE_VARIABLE enum returns a variable solver."""
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_VARIABLE,
            method="adaptive_heun",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, VariableAdaptiveQuadrature)

    def test_fixed_raises(self):
        """steps.FIXED raises ValueError (not supported for parallel RK)."""
        with pytest.raises(ValueError):
            adaptive_quadrature(
                sampling_type=steps.FIXED,
                method="bosh3",
                atol=1e-6,
                rtol=1e-6,
                remove_cut=REMOVE_CUT,
            )


# ---------------------------------------------------------------------------
# Uniform _get_tableau_b
# ---------------------------------------------------------------------------


class TestUniformGetTableauB:
    """Tests for UniformAdaptiveQuadrature._get_tableau_b."""

    def test_shape(self):
        """b and b_error have shape [C, 1] for bosh3 (C=4)."""
        solver = make_solver_for_unit_test("bosh3")
        t = torch.rand(5, 4, 1, dtype=torch.float64)
        b, b_error = solver._get_tableau_b(t)
        assert b.shape == (4, 1)
        assert b_error.shape == (4, 1)

    def test_t_independent(self):
        """Different t values give identical weights."""
        solver = make_solver_for_unit_test("bosh3")
        t1 = torch.rand(3, 4, 1, dtype=torch.float64)
        t2 = torch.rand(7, 4, 1, dtype=torch.float64)
        b1, _ = solver._get_tableau_b(t1)
        b2, _ = solver._get_tableau_b(t2)
        assert torch.equal(b1, b2)

    def test_values_match_tableau(self):
        """Returned b matches method.tableau.b unsqueezed."""
        solver = make_solver_for_unit_test("bosh3")
        t = torch.rand(1, 4, 1, dtype=torch.float64)
        b, _ = solver._get_tableau_b(t)
        expected = solver.method.tableau.b.unsqueeze(-1)
        assert torch.equal(b, expected)


# ---------------------------------------------------------------------------
# Variable _get_tableau_b
# ---------------------------------------------------------------------------


class TestVariableGetTableauB:
    """Tests for VariableAdaptiveQuadrature._get_tableau_b."""

    def test_shape_adaptive_heun(self):
        """adaptive_heun variable: b shape [1, 2, 1] (constant, broadcast over N)."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor(
            [
                [[0.0], [1.0]],
                [[1.0], [2.0]],
                [[2.0], [3.0]],
                [[3.0], [4.0]],
                [[4.0], [5.0]],
            ],
            dtype=torch.float64,
        )
        b, b_error = solver._get_tableau_b(t)
        # 2nd-order: weights are constant, returned as [1, C, 1]
        assert b.shape == (1, 2, 1)
        assert b_error.shape == (1, 2, 1)

    def test_normalizes_to_unit(self):
        """generic3: mesh=[2,3,4] normalizes to c=[0, 0.5, 1] → Simpson weights."""
        solver = make_variable_solver_for_unit_test("interpolatory3_variable")
        t = torch.tensor([[[2.0], [3.0], [4.0]]], dtype=torch.float64)
        b, _ = solver._get_tableau_b(t)
        # At a=0.5: b0=1/6, ba=2/3, b1=1/6
        expected = torch.tensor(
            [[[1.0 / 6], [2.0 / 3], [1.0 / 6]]], dtype=torch.float64
        )
        assert torch.allclose(b, expected, atol=1e-12)

    def test_different_positions_different_weights(self):
        """generic3: two steps with different midpoints produce different weights."""
        solver = make_variable_solver_for_unit_test("interpolatory3_variable")
        t = torch.tensor(
            [
                [[0.0], [0.3], [1.0]],  # a=0.3
                [[0.0], [0.7], [1.0]],  # a=0.7
            ],
            dtype=torch.float64,
        )
        b, _ = solver._get_tableau_b(t)
        assert not torch.allclose(b[0], b[1])


# ---------------------------------------------------------------------------
# _get_num_tableau_c
# ---------------------------------------------------------------------------


class TestGetNumTableauC:
    """Tests for _get_num_tableau_c on both solver types."""

    def test_uniform_bosh3(self):
        assert make_solver_for_unit_test("bosh3")._get_num_tableau_c() == 4

    def test_uniform_dopri5(self):
        assert make_solver_for_unit_test("dopri5")._get_num_tableau_c() == 7

    def test_uniform_adaptive_heun(self):
        assert make_solver_for_unit_test("adaptive_heun")._get_num_tableau_c() == 2

    def test_variable_adaptive_heun(self):
        assert (
            make_variable_solver_for_unit_test("adaptive_heun")._get_num_tableau_c()
            == 2
        )

    def test_variable_generic3(self):
        assert (
            make_variable_solver_for_unit_test(
                "interpolatory3_variable"
            )._get_num_tableau_c()
            == 3
        )
