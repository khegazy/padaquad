"""Integration accuracy tests for variable-sampling methods.

Phase 4 restored the variable solver by implementing
``_compute_nodes`` with uniformly-spaced initial node
placement. The existing ``_evaluate_adaptive_nodes`` (which interleaves
old and new nodes for split reuse) and ``_merge_excess_nodes`` (which
subsamples merged panels back to C nodes) now run end-to-end.
"""

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
    VARIABLE_METHOD_NAMES,
    assert_optimal_mesh_ordering,
    assert_step_continuity,
    assert_time_ordering,
)

from torchpathdiffeq import adaptive_quadrature, integrand_dict, steps


def _make_variable_solver(method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT):
    """Construct a parallel variable-sampling solver."""
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_VARIABLE,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=0.1,
    )


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("method_name", VARIABLE_METHOD_NAMES)
@pytest.mark.parametrize("integrand_name", INTEGRAND_NAMES)
class TestVariableIntegralAccuracy:
    """Each variable method correctly integrates each integrand_dict integrand."""

    def _integrate(self, method_name, integrand_name, take_gradient):
        """Run integration; return (output, correct, cutoff)."""
        f, solution_fxn, cutoff = integrand_dict[integrand_name]
        correct = solution_fxn(mesh_init=T_INIT, mesh_final=T_FINAL)
        torch.manual_seed(SEED)
        solver = _make_variable_solver(method_name)
        output = solver.integrate(
            f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=take_gradient
        )
        return output, correct, cutoff

    def test_integral_value(self, method_name, integrand_name, take_gradient):
        """Computed integral matches the analytical solution within cutoff."""
        output, correct, cutoff = self._integrate(
            method_name, integrand_name, take_gradient
        )
        rel_error = torch.abs((output.integral.cpu() - correct) / correct)
        # Variable methods are 2nd or 3rd order so we relax the cutoff
        # by 100x relative to the uniform-method cutoff in integrand_dict.
        # The uniform cutoff was tuned for dopri5 (5th order); variable
        # methods cannot match that on the same atol/rtol setting.
        adjusted_cutoff = cutoff * 100.0
        assert rel_error < adjusted_cutoff, (
            f"{method_name} (variable) on {integrand_name}: "
            f"got {output.integral.item()}, expected {correct.item()}, "
            f"rel_error={rel_error.item():.2e} >= adjusted_cutoff={adjusted_cutoff:.2e}"
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
