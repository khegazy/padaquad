"""Autodiff consistency: gradient-of-integral vs integral-of-gradient.

For a learnable scalar parameter ``theta`` and integrand
``f_theta(t) = theta * g(t)``, two paths give the same gradient:

  A) Solve ``int f_theta dt`` with ``take_gradient=True``. The solver
     calls ``loss.backward()`` after each accepted batch and
     accumulates the per-batch gradient into ``theta.grad``. This is
     the supported path for autodiff through the integrator.

  B) Differentiate the integrand symbolically: ``df_theta/dtheta = g(t)``.
     Integrate ``g(t)``. The result is ``d/dtheta int f_theta dt``.

These must match — the integration operator is linear in ``f``, so it
commutes with differentiation by a parameter that ``f`` depends on
linearly. Disagreement would indicate a bug in how the integrator
accumulates per-batch gradients.

Design note (per @khegazy 2026-05-09): per-batch ``.backward()`` is
the intentional path for autodiff because evaluation counts can grow
to where the full autograd graph does not fit in GPU memory. Once a
batch is backprop'd, its contribution to the running record is
``.detach()``ed (see ``_record_results``); only the running gradient
in ``theta.grad`` is meaningful, not ``out.integral.grad_fn``.

Single-batch integrals (small problems) work via
``torch.autograd.grad(out.integral, theta)`` because no detachment
fires; that case is also tested below as
``test_single_batch_autograd_grad_works``. Multi-batch integrals via
that path silently lose precision and should be avoided — users who
need autodiff through multi-batch integration must pass
``take_gradient=True`` and read from ``theta.grad``.

Phase 0 of the quadrature alignment plan.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests._helpers import cached_max_batch, make_uniform_solver

from padaquad import integrate

ATOL = 1e-8
RTOL = 1e-8
# Path-A and Path-B come from independent computational paths and the
# adaptive controller's mesh selection differs slightly between them,
# so bit-equality is not the bar — five decimal places of agreement is.
CONSISTENCY_BOUND = 1e-5


# (name, g_callable, interval) — each ``g`` is the linear-in-theta
# factor of f_theta and equals df_theta/dtheta.
_INTEGRANDS = [
    ("sin", torch.sin, (0.0, math.pi)),
    ("t_squared", lambda t: t**2, (0.0, 1.0)),
    ("damped_sine", lambda t: torch.exp(-0.5 * t) * torch.sin(t), (0.0, 4.0)),
    ("gaussian", lambda t: torch.exp(-(t**2)), (-2.0, 2.0)),
]


@pytest.mark.parametrize(
    ("name", "g", "interval"), _INTEGRANDS, ids=[t[0] for t in _INTEGRANDS]
)
def test_grad_of_integral_matches_integral_of_grad(name, g, interval):
    """Path A (take_gradient=True + theta.grad) and Path B (integrate g) match."""
    a, b = interval
    mesh_init = torch.tensor([a], dtype=torch.float64)
    mesh_final = torch.tensor([b], dtype=torch.float64)
    method = "dopri5"

    # --- Path A: solver class directly so we can pass take_gradient=True. ---
    theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)

    def f_theta(t):
        return theta * g(t)

    solver = make_uniform_solver(method, atol=ATOL, rtol=RTOL)
    solver.integrate(
        f=f_theta,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        take_gradient=True,
    )
    assert theta.grad is not None, "take_gradient=True did not populate theta.grad"
    grad_a = theta.grad.item()

    # --- Path B: directly integrate g(t) = df_theta / dtheta. ---
    out_b = integrate(
        f=g,
        method=method,
        atol=ATOL,
        rtol=RTOL,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        max_batch=cached_max_batch(),
    )
    grad_b = out_b.integral.item()

    assert abs(grad_a - grad_b) < CONSISTENCY_BOUND, (
        f"{name}: path-A grad={grad_a}, path-B grad={grad_b}, "
        f"diff={abs(grad_a - grad_b)}, bound={CONSISTENCY_BOUND}. "
        f"This indicates a per-batch backward / autodiff inconsistency."
    )


def test_single_batch_autograd_grad_works():
    """With ``take_gradient=False``, the full computation graph is preserved
    and ``torch.autograd.grad(out.integral, theta)`` works correctly.

    Use ``take_gradient=True`` instead when GPU memory prevents holding the
    full graph (see ``test_grad_of_integral_matches_integral_of_grad``).
    """
    theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    out = integrate(
        f=lambda t: theta * (t**2),  # very simple, fits in one batch
        method="dopri5",
        atol=ATOL,
        rtol=RTOL,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([1.0], dtype=torch.float64),
        take_gradient=False,
        max_batch=cached_max_batch(),
    )
    grad_a = torch.autograd.grad(out.integral.sum(), theta)[0].item()
    # int_0^1 t^2 dt = 1/3
    assert abs(grad_a - 1.0 / 3.0) < CONSISTENCY_BOUND


def test_take_gradient_does_not_corrupt_integral_value():
    """When ``take_gradient=True`` is used, the returned ``out.integral``
    value must still be correct; the backward calls per batch should
    not perturb the forward computation.
    """
    a, b = 0.0, math.pi
    mesh_init = torch.tensor([a], dtype=torch.float64)
    mesh_final = torch.tensor([b], dtype=torch.float64)
    method = "dopri5"

    out_no_grad = integrate(
        f=lambda t: 1.7 * torch.sin(t),
        method=method,
        atol=ATOL,
        rtol=RTOL,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        max_batch=cached_max_batch(),
    )

    theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    solver = make_uniform_solver(method, atol=ATOL, rtol=RTOL)
    out_with_grad = solver.integrate(
        f=lambda t: theta * torch.sin(t),
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        take_gradient=True,
    )

    assert (
        abs(out_no_grad.integral.item() - out_with_grad.integral.detach().item())
        < CONSISTENCY_BOUND
    ), "take_gradient=True changed the forward integral value"
