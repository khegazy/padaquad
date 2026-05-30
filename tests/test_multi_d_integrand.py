"""Multi-dimensional integrand output (D > 1) tests.

Most existing test integrands return shape ``[N, 1]``: scalar output
per time point. The library does support vector-valued integrands
``f(t) -> [N, D]`` for arbitrary D, but the existing test suite barely
exercises this path. These tests pin D > 1 behavior end-to-end to
catch any shape-handling regression introduced by the refactor.

Coverage notes:

  - For each method we integrate a 3-component vector
    ``[sin(t), cos(t), t]`` over [0, π/2]. Expected:
    ``[1.0, 1.0, π²/8]``.
  - Verifies result shapes: ``integral`` has shape ``[D]``,
    ``integral_error`` has shape ``[D]``, ``y`` has shape
    ``[N, C, D]``.
  - Tests both uniform and variable solvers.
  - Tests dopri5 (existing RK), gk21 (new GK), cc33 (new CC) to
    exercise all three method families.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests._helpers import TAKE_GRADIENT_IDS, TAKE_GRADIENT_VALUES

from padaquad import VARIABLE_METHODS, integrate
from padaquad.methods import UNIFORM_METHODS

D = 3
ATOL = 1e-8
RTOL = 1e-8


def _vector_integrand(t: torch.Tensor) -> torch.Tensor:
    """f(t) = [sin(t), cos(t), t]. Shape: [N, 1] -> [N, 3]."""
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return torch.cat([torch.sin(t), torch.cos(t), t], dim=-1)


def _truth(a: float, b: float) -> torch.Tensor:
    """Closed-form ∫ [sin, cos, t] dt over [a, b]."""
    return torch.tensor(
        [
            -math.cos(b) - -math.cos(a),
            math.sin(b) - math.sin(a),
            (b**2 - a**2) / 2,
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("method", ["dopri5", "gk21", "cc33"])
def test_uniform_methods_integrate_vector_valued_integrand(method, take_gradient):
    """3-component vector integrand integrated correctly element-wise."""
    a, b = 0.0, math.pi / 2

    result = integrate(
        f=_vector_integrand,
        method=method,
        sampling="uniform",
        atol=ATOL,
        rtol=RTOL,
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        take_gradient=take_gradient,
    )

    truth = _truth(a, b)
    assert result.integral.shape == (D,), (
        f"integral has shape {result.integral.shape}, expected ({D},)"
    )
    assert torch.allclose(result.integral, truth, atol=1e-5), (
        f"{method}: got {result.integral.tolist()}, "
        f"expected {truth.tolist()}, diff "
        f"{(result.integral - truth).abs().tolist()}"
    )


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_integration_result_shapes_for_multi_d(take_gradient):
    """All result fields that depend on D have the right last dim."""
    a, b = 0.0, math.pi / 2

    result = integrate(
        f=_vector_integrand,
        method="gk21",
        atol=ATOL,
        rtol=RTOL,
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        take_gradient=take_gradient,
    )

    assert result.integral.shape == (D,)
    assert result.integral_error.shape == (D,)
    # y is [N, C, D] — number of panels times nodes per panel times output dim.
    assert result.y.shape[-1] == D, f"y last dim is {result.y.shape[-1]}, expected {D}"
    # mesh_quadratures is [N, D].
    assert result.mesh_quadratures.shape[-1] == D
    assert result.mesh_quadrature_errors.shape[-1] == D


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("method", list(VARIABLE_METHODS.keys()))
def test_variable_methods_integrate_vector_valued_integrand(method, take_gradient):
    """Variable solvers also handle D > 1."""
    a, b = 0.0, math.pi / 2

    result = integrate(
        f=_vector_integrand,
        method=method,
        sampling="variable",
        atol=ATOL,
        rtol=ATOL,  # generous; variable is order-2/3
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        take_gradient=take_gradient,
    )

    truth = _truth(a, b)
    assert result.integral.shape == (D,)
    # Looser tolerance for low-order variable methods; this test
    # checks shape correctness, not max accuracy.
    assert torch.allclose(result.integral, truth, atol=1e-3), (
        f"{method}: got {result.integral.tolist()}, expected {truth.tolist()}"
    )


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_per_method_independence_across_output_dimensions(take_gradient):
    """Each output dim should integrate to the same value as it would
    if integrated alone — i.e., the multi-D path is just D parallel
    scalar integrations, no cross-dimension contamination.
    """
    a, b = 0.0, math.pi / 2

    # Vector integration.
    vec_result = integrate(
        f=_vector_integrand,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        take_gradient=take_gradient,
    )

    # Scalar integration of each component.
    scalar_results = []
    for i in range(D):
        scalar_result = integrate(
            f=lambda t, idx=i: _vector_integrand(t)[..., idx : idx + 1],
            method="gk21",
            atol=1e-10,
            rtol=1e-10,
            mesh_init=torch.tensor([a], dtype=torch.float64),
            mesh_final=torch.tensor([b], dtype=torch.float64),
            take_gradient=take_gradient,
        )
        scalar_results.append(scalar_result.integral.item())

    # The vector integration's components should match the scalar
    # integrations of each component.
    for i, scalar_val in enumerate(scalar_results):
        assert abs(vec_result.integral[i].item() - scalar_val) < 1e-7, (
            f"dim {i}: vector got {vec_result.integral[i].item()}, "
            f"scalar got {scalar_val}"
        )


def test_uniform_methods_registry_complete():
    """Quick smoke check that UNIFORM_METHODS still has the expected
    families after the Phase 5 file split. (Acts as a registry-shape
    pin alongside the dedicated test_methods_registry.py file.)
    """
    assert len(UNIFORM_METHODS) >= 10
    for required in ("dopri5", "gk21", "cc33"):
        assert required in UNIFORM_METHODS
