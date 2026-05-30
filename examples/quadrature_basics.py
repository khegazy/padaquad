"""
quadrature_basics.py — minimal one-shot quadrature examples.

Run with::

    python examples/quadrature_basics.py

Each example computes a definite integral via ``integrate(f, ...)`` and
checks the result against an analytical reference. Nothing fancy: no
neural networks, no autograd, no training loops. The point is to show
the simplest possible use of the library.
"""

from __future__ import annotations

import math

import torch

from padaquad import adaptive_quadrature, integrate, steps


def _check(label: str, got: float, expected: float, tol: float = 1e-6) -> None:
    err = abs(got - expected)
    status = "OK " if err < tol else "FAIL"
    print(
        f"  [{status}] {label}: got {got:.10f}, expected {expected:.10f}, err {err:.2e}"
    )


def example_sin_over_pi() -> None:
    """∫ sin(t) dt from 0 to π  =  2."""
    result = integrate(
        f=torch.sin,
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        mesh_init=torch.tensor([0.0]),
        mesh_final=torch.tensor([math.pi]),
    )
    print("sin over [0, pi]:")
    _check("integral", result.integral.item(), 2.0)
    print(f"    estimated error : {result.integral_error.item():.2e}")
    print(f"    optimal mesh    : {result.mesh_optimal.shape[0]} barriers")
    print(f"    converged       : {result.converged}")


def example_polynomial() -> None:
    """∫ t^4 dt from 0 to 1  =  1/5.

    gk21 is exact for polynomials of degree <= 31, so a single-panel
    application of the rule gives bit-exact answer in float64. The
    adaptive controller in this example splits the panel anyway (its
    per-step error estimator uses a *different* polynomial estimate
    than the rule itself), so the result accumulates a few ULPs of
    summation error across panels.
    """
    result = integrate(
        f=lambda t: t**4,
        method="gk21",
        atol=1e-12,
        rtol=1e-12,
        mesh_init=torch.tensor([0.0]),
        mesh_final=torch.tensor([1.0]),
    )
    print("t^4 over [0, 1]:")
    _check("integral", result.integral.item(), 1.0 / 5.0, tol=1e-7)


def example_gaussian_bump() -> None:
    """∫ exp(-t^2) dt from -2 to 2  =  sqrt(π) * erf(2)."""
    result = integrate(
        f=lambda t: torch.exp(-(t**2)),
        method="gk21",
        atol=1e-10,
        rtol=1e-10,
        mesh_init=torch.tensor([-2.0]),
        mesh_final=torch.tensor([2.0]),
    )
    expected = math.sqrt(math.pi) * math.erf(2.0)
    print("exp(-t^2) over [-2, 2]:")
    _check("integral", result.integral.item(), expected)


def example_method_comparison() -> None:
    """Compare several methods on a hard wiggly integrand.

    f(t) = sin(10*t) * exp(-0.1*t) on [0, 4]. The analytical solution
    requires complex arithmetic so we just compare the methods to
    each other.
    """
    f = lambda t: torch.sin(10 * t) * torch.exp(-0.1 * t)  # noqa: E731
    print("damped sinusoid over [0, 4] — comparison across methods:")
    for method in ("adaptive_heun", "bosh3", "dopri5", "gk21", "cc33"):
        result = integrate(
            f=f,
            method=method,
            atol=1e-8,
            rtol=1e-8,
            mesh_init=torch.tensor([0.0]),
            mesh_final=torch.tensor([4.0]),
        )
        n_evals = result.nodes.shape[0] * result.nodes.shape[1]
        print(
            f"  {method:>14}: integral={result.integral.item():+.10f}  "
            f"err≈{result.integral_error.item():.2e}  evals={n_evals:>5}"
        )


def example_warm_start_loop() -> None:
    """Use the class API to warm-start across a series of similar calls.

    This pattern is what training loops use: the integrand changes only
    slightly each iteration, so the previous run's optimal mesh is a
    near-perfect starting point for the next run.
    """
    print("warm-start loop over scaled sin integrand:")
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
    )
    mesh_init = torch.tensor([0.0])
    mesh_final = torch.tensor([math.pi])

    # Use a stateful callable so id(f) is stable across iterations —
    # otherwise reuse_mesh=True warns that the integrand changed.
    class ScaledSin:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def __call__(self, t: torch.Tensor) -> torch.Tensor:
            return self.scale * torch.sin(t)

    f = ScaledSin(scale=1.0)
    for k, scale in enumerate([1.0, 1.05, 1.1, 1.15]):
        f.scale = scale
        result = solver.integrate(
            f=f,
            mesh_init=mesh_init,
            mesh_final=mesh_final,
            reuse_mesh=(k > 0),
        )
        print(
            f"  iter {k} (scale={scale:.2f}): integral={result.integral.item():.6f}  "
            f"optimal mesh={result.mesh_optimal.shape[0]}"
        )


if __name__ == "__main__":
    # Seed for reproducibility. The default initial-mesh path uses
    # torch.rand to break alignment with uniformly-spaced features in
    # the integrand; a fixed seed makes the example print stable
    # numbers run-to-run (and keeps test_examples_run.py deterministic).
    torch.manual_seed(0)
    print("=" * 64)
    example_sin_over_pi()
    print()
    example_polynomial()
    print()
    example_gaussian_bump()
    print()
    example_method_comparison()
    print()
    example_warm_start_loop()
    print("=" * 64)
