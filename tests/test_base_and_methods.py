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
from padaquad.methods import _get_method, _Interpolatory3Variable

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
        with pytest.raises(KeyError):
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
    def test_variable_has_tableau_b_method(self, method_name):
        """Variable methods have a callable tableau_b method and an order."""
        method = _get_method(steps.ADAPTIVE_VARIABLE, method_name, "cpu", torch.float64)
        assert hasattr(method, "order")
        assert callable(getattr(method, "tableau_b", None))

    def test_invalid_uniform_name_raises(self):
        """Unknown method name raises KeyError."""
        with pytest.raises(KeyError):
            _get_method(steps.ADAPTIVE_UNIFORM, "nonexistent", "cpu", torch.float64)


# ---------------------------------------------------------------------------
# _Interpolatory3Variable weight computation
# ---------------------------------------------------------------------------


class TestVariableThirdOrder:
    """Tests for the Sanderse-Veldman 3rd-order variable weight formulas."""

    def test_b0_at_half(self):
        """b0(0.5) = 0.5 - 1/(6*0.5) = 1/6."""
        method = _Interpolatory3Variable()
        a = torch.tensor([0.5])
        assert torch.allclose(method._b0(a), torch.tensor([1.0 / 6]))

    def test_ba_at_half(self):
        """ba(0.5) = 1/(6*0.5*0.5) = 2/3."""
        method = _Interpolatory3Variable()
        a = torch.tensor([0.5])
        assert torch.allclose(method._ba(a), torch.tensor([2.0 / 3]))

    def test_b1_at_half(self):
        """b1(0.5) = (2-1.5)/(6*0.5) = 1/6."""
        method = _Interpolatory3Variable()
        a = torch.tensor([0.5])
        assert torch.allclose(method._b1(a), torch.tensor([1.0 / 6]))

    def test_weights_sum_to_one_sweep(self):
        """b0 + ba + b1 = 1 for many values of a in (0, 1)."""
        method = _Interpolatory3Variable()
        a_vals = torch.linspace(0.01, 0.99, 50)
        b0 = method._b0(a_vals)
        ba = method._ba(a_vals)
        b1 = method._b1(a_vals)
        sums = b0 + ba + b1
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12), (
            f"Max deviation from 1: {torch.max(torch.abs(sums - 1)).item():.2e}"
        )

    def test_tableau_b_shape(self):
        """tableau_b returns correct shapes for N=5, C=3."""
        method = _Interpolatory3Variable()
        c = torch.rand(5, 3, 1)
        # Ensure c[:,0,:] = 0 and c[:,2,:] = 1 (endpoints)
        c[:, 0, :] = 0.0
        c[:, 2, :] = 1.0
        # Middle values in (0, 1)
        c[:, 1, :] = torch.rand(5, 1) * 0.8 + 0.1

        b, b_error = method.tableau_b(c)

        assert b.shape == (5, 3)
        assert b_error.shape == (5, 3)

    def test_tableau_b_midpoint_weights(self):
        """With a=0.5 (midpoint), weights should be [1/6, 2/3, 1/6] (Simpson's rule)."""
        method = _Interpolatory3Variable()
        c = torch.tensor([[[0.0], [0.5], [1.0]]])  # [1, 3, 1]
        b, _ = method.tableau_b(c)
        expected = torch.tensor([[1.0 / 6, 2.0 / 3, 1.0 / 6]])
        assert torch.allclose(b, expected, atol=1e-12)


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
