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

from padaquad import VARIABLE_METHODS, adaptive_quadrature, integrate
from padaquad.methods import UNIFORM_METHODS

D = 3
ATOL = 1e-8
RTOL = 1e-8


def _vector_integrand(t: torch.Tensor) -> torch.Tensor:
    """f(t) = [sin(t), cos(t), t]. Shape: [N, 1] -> [N, 3]."""
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return torch.cat([torch.sin(t), torch.cos(t), t], dim=-1)


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Callable error_norm reducing the last (D) axis (L2)."""
    return torch.sqrt(torch.sum(x**2, dim=-1))


def _heterogeneous_integrand(t: torch.Tensor) -> torch.Tensor:
    """Two smooth components and one oscillatory (hard) component.

    f(t) = [t, t, sin(2*pi*5*t)]. The third component needs much finer
    resolution than the first two, so the choice of ``error_norm`` (which
    decides whether that one hard component can force a split) visibly
    changes the refinement.
    """
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return torch.cat([t, t, torch.sin(2 * math.pi * 5 * t)], dim=-1)


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
    integral_cpu = result.integral.cpu()
    assert torch.allclose(integral_cpu, truth, atol=1e-5), (
        f"{method}: got {integral_cpu.tolist()}, "
        f"expected {truth.tolist()}, diff "
        f"{(integral_cpu - truth).abs().tolist()}"
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
    assert torch.allclose(result.integral.cpu(), truth, atol=1e-3), (
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


@pytest.mark.parametrize(
    "error_norm",
    [
        "2",
        "max",
        "rms",
        "failure_fraction",
        pytest.param(_l2_norm, id="callable_l2"),
    ],
)
def test_error_norm_schemes_integrate_accurately(error_norm):
    """Every error_norm scheme integrates the vector integrand accurately."""
    a, b = 0.0, math.pi / 2
    result = integrate(
        f=_vector_integrand,
        method="gk21",
        atol=ATOL,
        rtol=RTOL,
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        error_norm=error_norm,
        take_gradient=False,
    )
    truth = _truth(a, b)
    assert result.integral.shape == (D,)
    assert torch.allclose(result.integral.cpu(), truth, atol=1e-5), (
        f"error_norm={error_norm}: got {result.integral.cpu().tolist()}, "
        f"expected {truth.tolist()}"
    )


def test_failure_fraction_tol_zero_bounds_every_component():
    """With mesh_failure_tolerance=0 every output element must meet tolerance,
    including the hard oscillatory component."""
    a, b = 0.0, 1.0
    result = integrate(
        f=_heterogeneous_integrand,
        method="gk21",
        atol=1e-7,
        rtol=1e-7,
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        error_norm="failure_fraction",
        mesh_failure_tolerance=0.0,
        take_gradient=False,
    )
    # truth: [0.5, 0.5, (1 - cos(2*pi*5))/(2*pi*5)] = [0.5, 0.5, 0.0]
    truth = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    assert torch.allclose(result.integral.cpu(), truth, atol=1e-5)


def test_failure_fraction_permissive_uses_no_more_panels():
    """A permissive mesh_failure_tolerance accepts panels the strict setting
    would split, so (from the same initial mesh) it never produces a finer
    optimal mesh."""
    a, b = 0.0, 1.0
    init_mesh = torch.linspace(a, b, 9, dtype=torch.float64).unsqueeze(-1)
    common = {
        "f": _heterogeneous_integrand,
        "method": "gk21",
        "atol": 1e-5,
        "rtol": 1e-5,
        "mesh": init_mesh,
        "error_norm": "failure_fraction",
        "take_gradient": False,
    }
    strict = integrate(mesh_failure_tolerance=0.0, **common)
    # 0.67 allows up to 2 of 3 components to fail (only an all-fail panel splits).
    permissive = integrate(mesh_failure_tolerance=0.67, **common)
    assert len(permissive.mesh_optimal) <= len(strict.mesh_optimal)


def test_error_norm_init_and_per_call_override():
    """Constructor default is used when integrate() args are None; explicit
    per-call args take priority."""
    solver = adaptive_quadrature(
        "uniform",
        method="gk21",
        atol=1e-5,
        rtol=1e-5,
        error_norm="max",
        mesh_failure_tolerance=0.3,
    )
    kwargs = {
        "f": _vector_integrand,
        "mesh_init": torch.tensor([0.0], dtype=torch.float64),
        "mesh_final": torch.tensor([1.0], dtype=torch.float64),
        "take_gradient": False,
    }

    solver.integrate(**kwargs)
    assert solver.error_norm == "max"
    assert solver.mesh_failure_tolerance == 0.3

    solver.integrate(error_norm="2", mesh_failure_tolerance=0.5, **kwargs)
    assert solver.error_norm == "2"
    assert solver.mesh_failure_tolerance == 0.5

    # None falls back to the construction-time values.
    solver.integrate(**kwargs)
    assert solver.error_norm == "max"
    assert solver.mesh_failure_tolerance == 0.3


def test_invalid_error_norm_rejected():
    """An unknown error_norm string is rejected at construction."""
    with pytest.raises(ValueError, match="error_norm"):
        adaptive_quadrature("uniform", method="gk21", error_norm="bogus")


def test_invalid_mesh_failure_tolerance_rejected():
    """mesh_failure_tolerance outside [0, 1] is rejected at construction."""
    with pytest.raises(ValueError, match="mesh_failure_tolerance"):
        adaptive_quadrature("uniform", method="gk21", mesh_failure_tolerance=1.5)


def test_uniform_methods_registry_complete():
    """Quick smoke check that UNIFORM_METHODS still has the expected
    families after the Phase 5 file split. (Acts as a registry-shape
    pin alongside the dedicated test_methods_registry.py file.)
    """
    assert len(UNIFORM_METHODS) >= 10
    for required in ("dopri5", "gk21", "cc33"):
        assert required in UNIFORM_METHODS
