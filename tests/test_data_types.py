"""Tests for correct handling of float32 and float64 dtypes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _helpers import (
    ATOL_MED,
    RTOL_MED,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    UNIFORM_METHOD_NAMES,
    assert_optimal_mesh_ordering,
    assert_step_continuity,
    assert_time_ordering,
    make_uniform_solver,
)

from padaquad import integrand_dict

INTEGRAND_NAME = "damped_sine"


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
class TestDtypeHandling:
    """Verify integration works correctly across floating-point precisions."""

    def _integrate(self, method_name, dtype, take_gradient):
        """Run damped_sine integration at the given dtype."""
        f, solution_fxn, _ = integrand_dict[INTEGRAND_NAME]
        correct = solution_fxn(
            mesh_init=torch.tensor([0], dtype=torch.float64),
            mesh_final=torch.tensor([1], dtype=torch.float64),
        )
        # Use a cutoff appropriate for medium tolerances (1e-9/1e-7),
        # which are looser than the tight tolerances used in test_integrals.
        cutoff = 1e-2
        # Relax further for float32 (less precision available)
        if dtype == torch.float32:
            cutoff = np.sqrt(cutoff)

        solver = make_uniform_solver(method_name, atol=ATOL_MED, rtol=RTOL_MED)
        output = solver.integrate(
            f,
            mesh_init=torch.tensor([0], dtype=dtype),
            mesh_final=torch.tensor([1], dtype=dtype),
            take_gradient=take_gradient,
        )
        return output, correct, cutoff

    def test_integral_accuracy(self, method_name, dtype, take_gradient):
        """Integral is accurate at both float32 and float64 precision."""
        output, correct, cutoff = self._integrate(method_name, dtype, take_gradient)
        rel_error = torch.abs((output.integral.cpu() - correct) / correct)
        assert rel_error < cutoff, (
            f"{method_name} ({dtype}) failed on {INTEGRAND_NAME}: "
            f"got {output.integral.item()}, expected {correct.item()}, "
            f"rel_error={rel_error.item():.2e} >= cutoff={cutoff:.2e}"
        )

    def test_time_ordering(self, method_name, dtype, take_gradient):
        """Time points are non-decreasing regardless of dtype."""
        output, _, _ = self._integrate(method_name, dtype, take_gradient)
        assert_time_ordering(output)

    def test_optimal_mesh_ordering(self, method_name, dtype, take_gradient):
        """Optimal mesh is non-decreasing regardless of dtype."""
        output, _, _ = self._integrate(method_name, dtype, take_gradient)
        assert_optimal_mesh_ordering(output)

    def test_step_continuity(self, method_name, dtype, take_gradient):
        """Consecutive steps share boundary points regardless of dtype."""
        output, _, _ = self._integrate(method_name, dtype, take_gradient)
        assert_step_continuity(output)
