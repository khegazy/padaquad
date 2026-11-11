"""Unit tests for the adaptive_quadrature factory, MethodClass.weights through a
solver, and the per-step node count C."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    REMOVE_CUT,
    make_solver_for_unit_test,
    make_variable_solver_for_unit_test,
)

from padaquad import (
    UniformAdaptiveQuadrature,
    VariableAdaptiveQuadrature,
    adaptive_quadrature,
)
from padaquad.base import steps

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

    def test_default_is_uniform(self):
        """No sampling_type gives a uniform solver (the README's warm-start example)."""
        solver = adaptive_quadrature(method="gk21", atol=1e-8, rtol=1e-8)
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
        """steps.FIXED raises ValueError naming it and the supported types."""
        with pytest.raises(ValueError, match=r"ADAPTIVE_UNIFORM.*got steps\.FIXED"):
            adaptive_quadrature(
                sampling_type=steps.FIXED,
                method="bosh3",
                atol=1e-6,
                rtol=1e-6,
                remove_cut=REMOVE_CUT,
            )

    def test_unknown_string_raises(self):
        """An unknown string raises ValueError (not KeyError) naming it."""
        with pytest.raises(ValueError, match=r"'bogus'.*'uniform'"):
            adaptive_quadrature(sampling_type="bogus", method="bosh3")


# ---------------------------------------------------------------------------
# Uniform rule weights
# ---------------------------------------------------------------------------


class TestUniformWeights:
    """Tests for UniformMethodClass.weights through a uniform solver."""

    def test_shape(self):
        """b and b_error have shape [N, C, 1] for bosh3 (C=4)."""
        solver = make_solver_for_unit_test("bosh3")
        t = torch.rand(5, 4, 1, dtype=torch.float64)
        b, b_error = solver.method.weights(t)
        assert b.shape == (5, 4, 1)
        assert b_error.shape == (5, 4, 1)

    def test_t_independent(self):
        """Different t values give identical weights for every panel."""
        solver = make_solver_for_unit_test("bosh3")
        t1 = torch.rand(3, 4, 1, dtype=torch.float64)
        t2 = torch.rand(7, 4, 1, dtype=torch.float64)
        b1, _ = solver.method.weights(t1)
        b2, _ = solver.method.weights(t2)
        assert torch.equal(b1[0], b2[0])
        assert torch.equal(b1, b1[:1].expand_as(b1))
        assert torch.equal(b2, b2[:1].expand_as(b2))

    def test_values_match_tableau(self):
        """Returned b matches method.tableau.b unsqueezed."""
        solver = make_solver_for_unit_test("bosh3")
        t = torch.rand(1, 4, 1, dtype=torch.float64)
        b, _ = solver.method.weights(t)
        expected = solver.method.tableau.b.unsqueeze(-1)
        assert torch.equal(b[0], expected)


# ---------------------------------------------------------------------------
# Variable-solver rule weights
# ---------------------------------------------------------------------------


class TestVariableWeights:
    """Tests for MethodClass.weights through a variable solver."""

    def test_shape_adaptive_heun(self):
        """adaptive_heun variable: b shape [N, 2, 1] (constant over panels)."""
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
        b, b_error = solver.method.weights(t)
        # 2nd-order: weights are constant, expanded to [N, C, 1]
        assert b.shape == (5, 2, 1)
        assert b_error.shape == (5, 2, 1)

    def test_normalizes_to_unit(self):
        """generic3: mesh=[2,3,4] normalizes to c=[0, 0.5, 1] → Simpson weights."""
        solver = make_variable_solver_for_unit_test("interpolatory3_variable")
        t = torch.tensor([[[2.0], [3.0], [4.0]]], dtype=torch.float64)
        b, _ = solver.method.weights(t)
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
        b, _ = solver.method.weights(t)
        assert not torch.allclose(b[0], b[1])


# ---------------------------------------------------------------------------
# Number of quadrature points per step (solver.C)
# ---------------------------------------------------------------------------


class TestNumQuadraturePoints:
    """Tests that solver.C matches the method's node count on both solver types."""

    def test_uniform_bosh3(self):
        assert make_solver_for_unit_test("bosh3").C == 4

    def test_uniform_dopri5(self):
        assert make_solver_for_unit_test("dopri5").C == 7

    def test_uniform_adaptive_heun(self):
        assert make_solver_for_unit_test("adaptive_heun").C == 2

    def test_variable_adaptive_heun(self):
        assert make_variable_solver_for_unit_test("adaptive_heun").C == 2

    def test_variable_generic3(self):
        assert make_variable_solver_for_unit_test("interpolatory3_variable").C == 3
