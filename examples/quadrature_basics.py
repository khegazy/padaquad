"""
Basic integrals
===============

Run with::

    python examples/quadrature_basics.py

Each example computes a definite integral with ``integrate(f, ...)`` and
compares it with the exact value. Every call passes float64 bounds (padaquad
computes in the dtype of the bounds), ``device="cpu"`` and a fixed
``max_batch``, which skips the memory benchmark that otherwise runs on every
call without one.
"""

from __future__ import annotations

import math

import torch

from padaquad import adaptive_quadrature, integrate

F64 = torch.float64
# Keyword options shared by every call: CPU, and a fixed batch size.
OPTIONS = {"device": "cpu", "max_batch": 10_000}


def _check(label: str, got: float, expected: float, tol: float = 1e-6) -> None:
    err = abs(got - expected)
    status = "OK " if err < tol else "FAIL"
    print(
        f"  [{status}] {label}: got {got:.10f}, expected {expected:.10f}, err {err:.2e}"
    )


# %%
# The starting mesh is random (see the user guide); a fixed seed makes the
# printed numbers the same on every run.
torch.manual_seed(0)


def example_sin_over_pi() -> None:
    """Integrate sin(t) from 0 to pi; the exact value is 2."""
    result = integrate(
        torch.sin,
        "gk21",
        atol=1e-10,
        rtol=1e-10,
        mesh_init=torch.tensor([0.0], dtype=F64),
        mesh_final=torch.tensor([math.pi], dtype=F64),
        **OPTIONS,
    )
    print("sin over [0, pi]:")
    _check("integral", float(result.integral), 2.0)
    print(f"    estimated error : {float(result.integral_error):.2e}")
    print(f"    converged       : {result.converged}")


# %%
example_sin_over_pi()
print()


def example_polynomial() -> None:
    """Integrate t^4 from 0 to 1; the exact value is 1/5.

    gk21 integrates every polynomial of degree at most 31 exactly, and its error
    estimate subtracts the 10-point Gauss method, which is exact through degree
    19. Both are exact for t^4, so every subinterval of the starting mesh has an
    error estimate at rounding level and none is split; the result is exact to
    float64 rounding.
    """
    result = integrate(
        lambda t: t**4,
        "gk21",
        atol=1e-12,
        rtol=1e-12,
        mesh_init=torch.tensor([0.0], dtype=F64),
        mesh_final=torch.tensor([1.0], dtype=F64),
        **OPTIONS,
    )
    print("t^4 over [0, 1]:")
    _check("integral", float(result.integral), 1.0 / 5.0, tol=1e-14)
    print(f"    subintervals split: {result.n_start_steps_split}")


# %%
example_polynomial()
print()


def example_gaussian_bump() -> None:
    """Integrate exp(-t^2) from -2 to 2; the exact value is sqrt(pi) erf(2)."""
    result = integrate(
        lambda t: torch.exp(-(t**2)),
        "gk21",
        atol=1e-10,
        rtol=1e-10,
        mesh_init=torch.tensor([-2.0], dtype=F64),
        mesh_final=torch.tensor([2.0], dtype=F64),
        **OPTIONS,
    )
    expected = math.sqrt(math.pi) * math.erf(2.0)
    print("exp(-t^2) over [-2, 2]:")
    _check("integral", float(result.integral), expected)


# %%
example_gaussian_bump()
print()


def example_method_comparison() -> None:
    """Compare several methods on an oscillating integrand.

    f(t) = sin(10 t) exp(-0.1 t) on [0, 4]. With a = -0.1 and b = 10, an
    antiderivative of exp(a t) sin(b t) is
    exp(a t) (a sin(b t) - b cos(b t)) / (a^2 + b^2).
    """

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.sin(10 * t) * torch.exp(-0.1 * t)

    a, b = -0.1, 10.0

    def antiderivative(t: float) -> float:
        return (
            math.exp(a * t)
            * (a * math.sin(b * t) - b * math.cos(b * t))
            / (a * a + b * b)
        )

    expected = antiderivative(4.0) - antiderivative(0.0)
    print("damped sinusoid over [0, 4], atol = rtol = 1e-8, by method:")
    for method in ("adaptive_heun", "bosh3", "dopri5", "gk21", "cc33"):
        result = integrate(
            f,
            method,
            atol=1e-8,
            rtol=1e-8,
            mesh_init=torch.tensor([0.0], dtype=F64),
            mesh_final=torch.tensor([4.0], dtype=F64),
            **OPTIONS,
        )
        error = abs(float(result.integral) - expected)
        # result.y holds one integrand value per evaluated node.
        print(
            f"  {method:>14}: error {error:.1e}   "
            f"integrand evaluations {result.y.shape[0]:>5}"
        )


# %%
example_method_comparison()
print()


def example_warm_start_loop() -> None:
    """Warm-start a series of similar integrals with one solver.

    This is the pattern of a training loop: the integrand changes a little at
    each iteration, so the mesh the previous call ended with is a good starting
    mesh for the next one (``same_f=True``). The first call starts from a fresh
    mesh and splits some of its subintervals; the later calls start from the
    refined mesh and need fewer splits. The low-degree method ``bosh3`` is used so
    that the first call has to refine; ``gk21`` would need no split here.
    """
    print("warm-start loop over a scaled sin integrand:")
    solver = adaptive_quadrature("uniform", "bosh3", atol=1e-8, rtol=1e-8, **OPTIONS)
    for k, scale in enumerate([1.0, 1.05, 1.1, 1.15]):
        result = solver.integrate(
            lambda t, s=scale: torch.sin(s * t),
            mesh_init=torch.tensor([0.0], dtype=F64),
            mesh_final=torch.tensor([3 * math.pi], dtype=F64),
            same_f=True,
        )
        expected = (1 - math.cos(3 * math.pi * scale)) / scale
        print(
            f"  iter {k} (scale={scale:.2f}): error "
            f"{abs(float(result.integral) - expected):.1e}, starting subintervals "
            f"split: {result.percent_start_steps_split:.0f}%"
        )


# %%
example_warm_start_loop()
