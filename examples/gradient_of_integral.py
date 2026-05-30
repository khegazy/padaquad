"""
gradient_of_integral.py — two equivalent ways to compute
``d/dtheta integrate(f_theta(t)) dt``.

The integration operator is linear in the integrand, so it commutes
with differentiation by a parameter that ``f`` depends on linearly.
This means there are two paths to the same answer:

  (A) Backprop through the integral.
      Just write ``f_theta(t)`` as a normal PyTorch callable that uses
      ``theta`` somewhere, run ``integrate(f_theta, ...)``, and call
      ``loss.backward()`` (or use the solver's ``take_gradient=True``).

  (B) Integrate ``df_theta/dtheta`` directly.
      You provide the closed-form derivative of the integrand with
      respect to ``theta`` and integrate that. Useful when ``f`` is
      expensive but its analytic gradient is cheap.

Both paths produce the same scalar gradient. This file demonstrates
agreement on a simple integrand ``f_theta(t) = theta * sin(t)`` over
``[0, pi]`` whose gradient is exactly 2.

Run with::

    python examples/gradient_of_integral.py
"""

from __future__ import annotations

import math

import torch

from padaquad import adaptive_quadrature, integrate, steps

# Common settings.
MESH_INIT = torch.tensor([0.0])
MESH_FINAL = torch.tensor([math.pi])
ATOL = 1e-8
RTOL = 1e-8


def path_a_backprop_through_integral(theta: torch.Tensor) -> float:
    """Path A: backprop through the integration loop.

    The recommended pattern for differentiable integration: pass
    ``take_gradient=True`` to ``solver.integrate(...)``. This makes
    the solver call ``loss.backward()`` after every accepted batch
    of panels, so the autograd graph for that batch can be released
    immediately and the next batch's graph fits in memory.
    """
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="gk21",
        atol=ATOL,
        rtol=RTOL,
    )

    def f_theta(t: torch.Tensor) -> torch.Tensor:
        return theta * torch.sin(t)

    # Reset any prior gradient. take_gradient=True will accumulate
    # into theta.grad.
    if theta.grad is not None:
        theta.grad.zero_()

    solver.integrate(
        f=f_theta,
        mesh_init=MESH_INIT,
        mesh_final=MESH_FINAL,
        take_gradient=True,
    )
    return theta.grad.item()


def path_b_integrate_the_gradient() -> float:
    """Path B: integrate ``df_theta / dtheta = sin(t)`` directly.

    No autograd needed: just construct the closed-form derivative as
    its own integrand callable and integrate that.
    """

    def df_dtheta(t: torch.Tensor) -> torch.Tensor:
        # f_theta(t) = theta * sin(t)  =>  df/dtheta = sin(t)
        return torch.sin(t)

    out = integrate(
        f=df_dtheta,
        method="gk21",
        atol=ATOL,
        rtol=RTOL,
        mesh_init=MESH_INIT,
        mesh_final=MESH_FINAL,
    )
    return out.integral.item()


def main() -> None:
    theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)

    grad_a = path_a_backprop_through_integral(theta)
    grad_b = path_b_integrate_the_gradient()
    expected = 2.0  # int sin(t) dt over [0, pi]

    print("d/dtheta integrate(theta * sin(t)) over [0, pi]:")
    print(f"  expected (== int sin dt over [0, pi]) : {expected:.10f}")
    print(f"  path A (autograd through integration): {grad_a:.10f}")
    print(f"  path B (integrate the derivative)    : {grad_b:.10f}")
    print(f"  |A - B| = {abs(grad_a - grad_b):.2e}")
    assert abs(grad_a - expected) < 1e-6, "path A disagrees with closed form"
    assert abs(grad_b - expected) < 1e-6, "path B disagrees with closed form"
    assert abs(grad_a - grad_b) < 1e-6, (
        "path A and path B disagree (autodiff consistency violation)"
    )
    print("  All three agree to within 1e-6.")


if __name__ == "__main__":
    # Seed for reproducibility — see quadrature_basics.py for the why.
    torch.manual_seed(0)
    main()
