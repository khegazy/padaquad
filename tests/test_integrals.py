"""Tests for numerical integral accuracy against known analytical solutions."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    ATOL_TIGHT,
    INTEGRAND_NAMES,
    RTOL_TIGHT,
    SEED,
    T_FINAL,
    T_INIT,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    UNIFORM_METHOD_NAMES,
    assert_optimal_mesh_ordering,
    assert_step_continuity,
    assert_time_ordering,
    make_uniform_solver,
)

from padaquad import integrand_dict


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
@pytest.mark.parametrize("integrand_name", INTEGRAND_NAMES)
class TestUniformIntegralAccuracy:
    """Verify that each uniform RK method correctly integrates each test function."""

    def _integrate(self, method_name, integrand_name, take_gradient):
        """Run integration for the given method and integrand, return (output, correct, cutoff)."""
        f, solution_fxn, cutoff = integrand_dict[integrand_name]
        correct = solution_fxn(mesh_init=T_INIT, mesh_final=T_FINAL)
        torch.manual_seed(SEED)
        solver = make_uniform_solver(method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT)
        output = solver.integrate(
            f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=take_gradient
        )
        return output, correct, cutoff

    def test_integral_value(self, method_name, integrand_name, take_gradient):
        """Computed integral matches the analytical solution within the error cutoff."""
        output, correct, cutoff = self._integrate(
            method_name, integrand_name, take_gradient
        )
        rel_error = torch.abs((output.integral.cpu() - correct) / correct)
        assert rel_error < cutoff, (
            f"{method_name} failed on {integrand_name}: "
            f"got {output.integral.item()}, expected {correct.item()}, "
            f"rel_error={rel_error.item():.2e} >= cutoff={cutoff:.2e}"
        )

    def test_time_ordering(self, method_name, integrand_name, take_gradient):
        """All time points in the integration output are non-decreasing."""
        output, _, _ = self._integrate(method_name, integrand_name, take_gradient)
        assert_time_ordering(output)

    def test_optimal_mesh_ordering(self, method_name, integrand_name, take_gradient):
        """Optimal mesh time points are non-decreasing."""
        output, _, _ = self._integrate(method_name, integrand_name, take_gradient)
        assert_optimal_mesh_ordering(output)

    def test_step_continuity(self, method_name, integrand_name, take_gradient):
        """Consecutive integration steps share boundary points."""
        output, _, _ = self._integrate(method_name, integrand_name, take_gradient)
        assert_step_continuity(output)
