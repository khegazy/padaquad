"""Tests that the integrate() wrapper produces identical results to direct solver usage."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    ATOL_MED,
    RTOL_MED,
    SEED,
    T_FINAL,
    T_INIT,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    UNIFORM_METHOD_NAMES,
)

from padaquad import (
    UniformAdaptiveQuadrature,
    integrate,
)


def _integrand(t, y=0):
    """Test integrand: modulated Gaussian * cosine chirp."""
    return torch.exp(-5 * (t - 0.5) ** 2) * 4 * torch.cos(3 * t**2)


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
def test_wrapper_matches_direct_solver_uniform(method_name, take_gradient):
    """integrate(sampling='uniform') matches direct RK solver."""
    torch.manual_seed(SEED)

    wrapper_output = integrate(
        f=_integrand,
        method=method_name,
        sampling="uniform",
        atol=ATOL_MED,
        rtol=RTOL_MED,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        y0=torch.tensor([0], dtype=torch.float64),
        mesh=None,
        take_gradient=take_gradient,
    )

    torch.manual_seed(SEED)

    direct_solver = UniformAdaptiveQuadrature(
        method=method_name,
        atol=ATOL_MED,
        rtol=RTOL_MED,
        f=_integrand,
    )
    direct_output = direct_solver.integrate(
        mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=take_gradient
    )

    assert torch.allclose(wrapper_output.integral, direct_output.integral), (
        f"Integral mismatch for {method_name}: "
        f"wrapper={wrapper_output.integral.item()}, direct={direct_output.integral.item()}"
    )
    assert torch.allclose(
        wrapper_output.integral_error, direct_output.integral_error
    ), f"Integral error mismatch for {method_name}"
    assert torch.allclose(wrapper_output.mesh_optimal, direct_output.mesh_optimal), (
        f"Optimal mesh mismatch for {method_name}"
    )
    assert torch.allclose(wrapper_output.y, direct_output.y), (
        f"y values mismatch for {method_name}"
    )
    assert torch.allclose(wrapper_output.nodes, direct_output.nodes), (
        f"t values mismatch for {method_name}"
    )
    assert torch.allclose(wrapper_output.h, direct_output.h), (
        f"Step sizes mismatch for {method_name}"
    )
    assert torch.allclose(
        wrapper_output.mesh_quadratures, direct_output.mesh_quadratures
    ), f"mesh_quadratures mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.mesh_quadrature_errors, direct_output.mesh_quadrature_errors
    ), f"mesh_quadrature_errors mismatch for {method_name}"
    assert torch.allclose(wrapper_output.error_ratios, direct_output.error_ratios), (
        f"error_ratios mismatch for {method_name}"
    )
