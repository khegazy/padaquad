"""Unit tests for base utilities, method retrieval, tableaux, and example integrands."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    INTEGRAND_NAMES,
    T_FINAL,
    T_INIT,
    UNIFORM_METHOD_NAMES,
    VARIABLE_METHOD_NAMES,
    make_solver_for_unit_test,
    make_variable_solver_for_unit_test,
)

from padaquad import UNIFORM_METHODS, integrand_dict
from padaquad.base import get_sampling_type, steps
from padaquad.examples import (
    damped_sine,
    exp_solution,
    identity,
    identity_solution,
    sine_squared,
    t_solution,
    t_squared,
    t_squared_solution,
)
from padaquad.examples import (
    exp as exp_fn,
)
from padaquad.examples import (
    t as t_fn,
)
from padaquad.methods import MethodClass, _get_method
from padaquad.methods.interpolatory import _interpolatory3_weights

# ---------------------------------------------------------------------------
# get_sampling_type
# ---------------------------------------------------------------------------


class TestGetSamplingType:
    """Tests for the string-to-enum conversion utility."""

    def test_uniform_short(self):
        assert get_sampling_type("uniform") == steps.ADAPTIVE_UNIFORM

    def test_uniform_full(self):
        assert get_sampling_type("adaptive_uniform") == steps.ADAPTIVE_UNIFORM

    def test_variable_short(self):
        assert get_sampling_type("variable") == steps.ADAPTIVE_VARIABLE

    def test_variable_full(self):
        assert get_sampling_type("adaptive_variable") == steps.ADAPTIVE_VARIABLE

    def test_fixed(self):
        assert get_sampling_type("fixed") == steps.FIXED

    def test_invalid_raises(self):
        """An unknown name raises ValueError naming it and the valid names."""
        with pytest.raises(ValueError, match=r"'nonexistent'.*'uniform'"):
            get_sampling_type("nonexistent")


# ---------------------------------------------------------------------------
# _Tableau dtype/device conversion
# ---------------------------------------------------------------------------


class TestTableau:
    """Tests for _Tableau dtype and device conversion."""

    def test_to_dtype_converts(self):
        """Converting float64 -> float32 changes all tensor dtypes."""
        # Clone the singleton so this test cannot pollute it for
        # downstream tests (float32 round-trip is precision-lossy).
        method = UNIFORM_METHODS["bosh3"].clone()
        original_b = method.tableau.b.clone()
        method.to_dtype(torch.float32)
        assert method.tableau.b.dtype == torch.float32
        assert method.tableau.c.dtype == torch.float32
        assert method.tableau.b_error.dtype == torch.float32
        method.to_dtype(torch.float64)
        assert torch.allclose(method.tableau.b, original_b)

    def test_to_dtype_preserves_values(self):
        """Round-tripping float64 -> float32 -> float64 preserves values within float32 precision."""
        method = UNIFORM_METHODS["dopri5"].clone()
        original_b = method.tableau.b.clone()
        method.to_dtype(torch.float32)
        method.to_dtype(torch.float64)
        assert torch.allclose(method.tableau.b, original_b, atol=1e-7)


# ---------------------------------------------------------------------------
# _get_method factory
# ---------------------------------------------------------------------------


class TestGetMethod:
    """Tests for the method retrieval factory."""

    @pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
    def test_uniform_has_tableau(self, method_name):
        """Uniform methods have a tableau with c, b, b_error attributes."""
        method = _get_method(steps.ADAPTIVE_UNIFORM, method_name, "cpu", torch.float64)
        assert hasattr(method, "tableau")
        assert hasattr(method.tableau, "c")
        assert hasattr(method.tableau, "b")
        assert hasattr(method.tableau, "b_error")

    @pytest.mark.parametrize("method_name", VARIABLE_METHOD_NAMES)
    def test_variable_has_method_interface(self, method_name):
        """Variable methods implement the MethodClass interface."""
        method = _get_method(steps.ADAPTIVE_VARIABLE, method_name, "cpu", torch.float64)
        assert isinstance(method, MethodClass)
        assert method.n_nodes == len(method.node_fractions())
        assert callable(method.weights)

    @pytest.mark.parametrize(
        "sampling_type", [steps.ADAPTIVE_UNIFORM, steps.ADAPTIVE_VARIABLE]
    )
    def test_invalid_name_raises(self, sampling_type):
        """Unknown method name raises ValueError naming the valid methods."""
        with pytest.raises(ValueError, match=r"nonexistent.*choose one of"):
            _get_method(sampling_type, "nonexistent", "cpu", torch.float64)

    def test_solver_with_invalid_name_raises(self):
        """A solver built with an unknown method name raises ValueError."""
        with pytest.raises(ValueError, match="nonexistent"):
            make_solver_for_unit_test("nonexistent")

    @pytest.mark.parametrize(
        ("kind", "method_name"),
        [("uniform", m) for m in UNIFORM_METHOD_NAMES]
        + [("variable", m) for m in VARIABLE_METHOD_NAMES],
    )
    def test_error_exponent_matches_weights(self, kind, method_name):
        """``error_exponent`` k: the per-step error estimate scales as h^k.

        On a unit step with nodes x = 2c - 1, the estimate applied to x^j is the
        moment m_j = sum_i b_error_i x_i^j. It is zero (round-off) for every j
        the estimate's less accurate rule integrates exactly, and the first
        nonzero moment is at j = k - 1 (the error of a degree-j term on a step
        of width h scales as h^(j+1)). The 1e-13 relative threshold sits between
        the round-off of the zero moments (< 1e-15) and the smallest first
        nonzero moment of any method (cc65, ~1e-12).
        """
        solver = (
            make_solver_for_unit_test(method_name)
            if kind == "uniform"
            else make_variable_solver_for_unit_test(method_name)
        )
        nodes = solver._compute_nodes(
            torch.zeros(1, 1, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.float64),
        )
        _, b_error = solver.method.weights(nodes)
        b_error = b_error.reshape(-1)
        x = 2 * nodes.reshape(-1) - 1

        def relative_moment(j):
            return float(
                (b_error * x**j).sum().abs() / (b_error.abs() * x.abs() ** j).sum()
            )

        k = solver.method.error_exponent
        assert all(relative_moment(j) < 1e-13 for j in range(k - 1))
        assert relative_moment(k - 1) > 1e-13


# ---------------------------------------------------------------------------
# Sanderse-Veldman 3-node weight function (interpolatory3_variable)
# ---------------------------------------------------------------------------


def _nodes_at(a: torch.Tensor) -> torch.Tensor:
    """Normalized 3-node panels [0, a, 1]. Shape: [len(a), 3, 1]."""
    return torch.stack([torch.zeros_like(a), a, torch.ones_like(a)], dim=1).unsqueeze(
        -1
    )


class TestVariableThirdOrder:
    """Tests for ``_interpolatory3_weights`` (Sanderse-Veldman 3-node weights)."""

    def test_weights_at_half(self):
        """At a = 0.5: b0 = 0.5 - 1/3 = 1/6, ba = 1/(6/4) = 2/3, b1 = 0.5/3 = 1/6."""
        b, _ = _interpolatory3_weights(_nodes_at(torch.tensor([0.5])))
        assert torch.allclose(b, torch.tensor([[1.0 / 6, 2.0 / 3, 1.0 / 6]]))

    def test_weights_sum_to_one_sweep(self):
        """b0 + ba + b1 = 1 for many values of a in (0, 1)."""
        a_vals = torch.linspace(0.01, 0.99, 50, dtype=torch.float64)
        b, _ = _interpolatory3_weights(_nodes_at(a_vals))
        sums = b.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12), (
            f"Max deviation from 1: {torch.max(torch.abs(sums - 1)).item():.2e}"
        )

    def test_error_weights_are_b_minus_trapezoid(self):
        """b_error = b - [1/2, 0, 1/2] (error estimate against the trapezoidal rule)."""
        a_vals = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
        b, b_error = _interpolatory3_weights(_nodes_at(a_vals))
        trapezoid = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64)
        assert torch.equal(b_error, b - trapezoid)

    def test_weight_shape(self):
        """The weight function returns [N, C] for N=5, C=3."""
        b, b_error = _interpolatory3_weights(_nodes_at(torch.rand(5) * 0.8 + 0.1))
        assert b.shape == (5, 3)
        assert b_error.shape == (5, 3)


# ---------------------------------------------------------------------------
# Example integrands — known values
# ---------------------------------------------------------------------------


class TestExampleIntegrands:
    """Tests for the example integrand functions at known evaluation points."""

    def test_identity_returns_one(self):
        result = identity(torch.tensor([[0.5]]))
        assert result == 1

    def test_t_returns_input(self):
        inp = torch.tensor([[0.3]])
        result = t_fn(inp)
        assert torch.allclose(result, inp)

    def test_t_squared_returns_square(self):
        inp = torch.tensor([[3.0]])
        result = t_squared(inp)
        assert torch.allclose(result, torch.tensor([[9.0]]))

    def test_sine_squared_at_zero(self):
        result = sine_squared(torch.tensor([[0.0]]))
        assert torch.allclose(result, torch.tensor([[0.0]]))

    def test_exp_at_zero(self):
        result = exp_fn(torch.tensor([[0.0]]))
        assert torch.allclose(result, torch.tensor([[1.0]]))

    def test_damped_sine_at_zero(self):
        """sin(0) = 0, so damped_sine(0) = exp(0)*sin(0) = 0."""
        result = damped_sine(torch.tensor([[0.0]]))
        assert torch.allclose(result, torch.tensor([[0.0]]))


class TestExampleSolutions:
    """Tests for the analytical solution functions at known points."""

    def test_identity_solution(self):
        result = identity_solution(torch.tensor([0.0]), torch.tensor([1.0]))
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_t_solution(self):
        result = t_solution(torch.tensor([0.0]), torch.tensor([1.0]))
        assert torch.allclose(result, torch.tensor([0.5]))

    def test_t_squared_solution(self):
        result = t_squared_solution(torch.tensor([0.0]), torch.tensor([1.0]))
        assert torch.allclose(result, torch.tensor([1.0 / 3]))

    def test_exp_solution_unit_interval(self):
        """∫₀¹ exp(5t) dt = (e⁵ - 1)/5."""
        result = exp_solution(torch.tensor([0.0]), torch.tensor([1.0]))
        expected = (torch.exp(torch.tensor([5.0])) - 1.0) / 5.0
        assert torch.allclose(result, expected)

    def test_identity_solution_nonunit_interval(self):
        """∫₂⁵ 1 dt = 3."""
        result = identity_solution(torch.tensor([2.0]), torch.tensor([5.0]))
        assert torch.allclose(result, torch.tensor([3.0]))

    def test_t_solution_nonunit_interval(self):
        """∫₁³ t dt = (9-1)/2 = 4."""
        result = t_solution(torch.tensor([1.0]), torch.tensor([3.0]))
        assert torch.allclose(result, torch.tensor([4.0]))

    @pytest.mark.parametrize("integrand_name", INTEGRAND_NAMES)
    def test_solution_consistency(self, integrand_name):
        """Each integrand_dict solution function gives a finite result on [0, 1]."""
        _, solution_fxn, _ = integrand_dict[integrand_name]
        result = solution_fxn(mesh_init=T_INIT, mesh_final=T_FINAL)
        assert torch.isfinite(result).all(), f"{integrand_name} solution is not finite"
        assert result.numel() > 0, f"{integrand_name} solution is empty"
