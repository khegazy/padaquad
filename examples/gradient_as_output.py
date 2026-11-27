"""
The gradient as extra outputs
=============================

Integrate a function and its gradient with respect to parameters in one call,
without keeping an autograd graph, and choose which of them decides where the
mesh is refined.

``integrate`` can differentiate an integral by autograd (see
:doc:`gradient_of_integral`): with ``take_gradient=False`` the graph of every
integrand value is kept until ``result.integral`` is differentiated, and with
``take_gradient=True`` each batch's graph is backpropagated and freed. When the
gradient has many components, or the graph is large, a third way is often
better: let ``f`` return the gradient of the integrand as extra outputs. The
integral of those outputs is the gradient of the integral (the parameters do
not change the bounds here, so no boundary terms appear), and no graph is kept
after a call of ``f``. It is also what makes ``trade_speed_for_memory=True``
worthwhile for a gradient: that option moves the integrand values off the
solver's device as soon as they are integrated, which saves little when an
autograd graph keeps what ``f`` saved for its backward pass on the device.

The integrand here is ``f(t; a, b) = a sin(b t^2)`` on ``[0, 3]``, with
parameters ``theta = (a, b)``. Its gradient,
``(sin(b t^2), a t^2 cos(b t^2))``, grows with ``t`` where the integrand does
not, so a mesh refined for the integrand is not the one the gradient needs.
The method is ``dopri5``, of low degree, so the difference shows at moderate
tolerances.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
from scipy.integrate import quad

from padaquad import integrate

theta = torch.tensor([1.3, 2.5], dtype=torch.float64)  # (a, b), no grad needed
BOUNDS = {
    "mesh_init": torch.tensor([0.0], dtype=torch.float64),
    "mesh_final": torch.tensor([3.0], dtype=torch.float64),
}


def g(theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate the integrand at one point t (a scalar)."""
    return theta[0] * torch.sin(theta[1] * t**2)


# %%
# The integrand and its gradient in one output
# --------------------------------------------
# ``torch.func.grad`` differentiates ``g`` with respect to ``theta`` at one
# point, and ``torch.func.vmap`` does it for every point of the call. ``theta``
# does not require grad, so the outputs carry no autograd graph. (With an
# ``nn.Parameter``, pass ``theta.detach()``, or call ``f`` under
# ``torch.no_grad()``; ``torch.func.grad`` still differentiates there.) The
# output has D = 3 components: the integrand, then its two derivatives.

gradient_of_g = torch.func.vmap(torch.func.grad(g), in_dims=(None, 0))


def f(t: torch.Tensor) -> torch.Tensor:
    """Integrand and gradient at the points t (shape [N, 1]): shape [N, 3]."""
    value = g(theta, t[:, 0])
    return torch.cat([value[:, None], gradient_of_g(theta, t[:, 0])], dim=-1)


# %%
# Reference values from ``scipy.integrate.quad`` at a tight tolerance.

a, b = theta.tolist()
references = [
    lambda t: a * math.sin(b * t * t),  # the integrand
    lambda t: math.sin(b * t * t),  # its derivative with respect to a
    lambda t: a * t * t * math.cos(b * t * t),  # with respect to b
]
exact = torch.tensor(
    [quad(h, 0.0, 3.0, epsabs=0.0, epsrel=1e-13, limit=500)[0] for h in references],
    dtype=torch.float64,
)
print("integral and gradient (scipy):", exact.tolist())

# %%
# Which outputs decide a split
# ----------------------------
# A subinterval is split when its error is too large. With D outputs,
# ``error_norm`` turns the error of the D components into one number: the
# default ``"2"`` takes the Euclidean norm of all of them. A callable chooses
# the components: it receives tensors whose last axis has the D outputs
# (``[N, D]`` for the subintervals' errors, ``[N - 1, D]`` for pairs of
# neighbouring subintervals, ``[D]`` for the integral the tolerance is relative
# to) and must remove that axis. Below, the integrand alone, or the gradient
# alone, decides. (``"failure_fraction"`` instead judges each output against
# its own tolerance, ``atol + rtol |I_d|``, which suits outputs of very
# different sizes.)

choices = {
    "all outputs": "2",
    "integrand only": lambda x: x[..., 0].abs(),
    "gradient only": lambda x: torch.linalg.vector_norm(x[..., 1:], dim=-1),
}


def run(error_norm, rtol):
    """Integrate f; return the subintervals' count and the relative errors.

    The errors are the integral's and the largest of the gradient's components.
    """
    torch.manual_seed(0)
    result, _ = integrate(
        f,
        "dopri5",
        device="cpu",
        max_batch=1000,
        atol=0.0,
        rtol=rtol,
        error_norm=error_norm,
        take_gradient=False,
        **BOUNDS,
    )
    error = ((result.integral - exact).abs() / exact.abs()).tolist()
    return len(result.h), error[0], max(error[1:])


rtols = [1e-4, 1e-5, 1e-6]
runs = {name: [run(norm, rtol) for rtol in rtols] for name, norm in choices.items()}
for name, rows in runs.items():
    for rtol, (n, integral_error, gradient_error) in zip(rtols, rows, strict=True):
        print(
            f"{name:15s} rtol {rtol:.0e}: {n:3d} subintervals, relative error "
            f"of the integral {integral_error:.1e}, of the gradient {gradient_error:.1e}"
        )

# At rtol = 1e-6, judging on the gradient refines more and gives the gradient
# the accuracy asked for; judging on the integrand alone stops earlier and
# leaves the gradient's error more than 10 times larger. Every choice meets the
# integral's tolerance.
n_gradient, _, gradient_error = runs["gradient only"][-1]
n_integrand, _, gradient_error_integrand_only = runs["integrand only"][-1]
assert gradient_error < 1e-6
assert n_gradient > n_integrand
assert gradient_error_integrand_only > 10 * gradient_error
for rows in runs.values():
    assert rows[-1][1] < 1e-6

# %%
# Accuracy against the number of subintervals
# -------------------------------------------
# One point per tolerance (1e-4, 1e-5, 1e-6). For an optimization driven by
# this gradient, judge on the gradient (or on all outputs): the integrand-only
# mesh is smaller, but its gradient is less accurate at the same tolerance.

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
for name, rows in runs.items():
    n = [row[0] for row in rows]
    axes[0].loglog(n, [row[2] for row in rows], "o-", label=name)
    axes[1].loglog(n, [row[1] for row in rows], "o-", label=name)
axes[0].set_title("gradient")
axes[1].set_title("integral")
for ax in axes:
    ax.set_xlabel("subintervals")
    ax.set_ylabel("largest relative error")
axes[0].legend()
fig.tight_layout()
plt.show()
