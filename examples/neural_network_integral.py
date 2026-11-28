"""
Integrating a neural network
============================

A neural network that maps the integration variable t to several outputs is
an integrand like any other: PadaQuad calls it on a batch of points of shape
[N, 1] and it returns values of shape [N, D]. A ``torch.nn.Module`` does
exactly that, so it can be passed to ``integrate`` as it is.

This example integrates a small network with three outputs over [0, 1],
checks the result against ``scipy.integrate.quad``, then computes the gradient
of the integral with respect to the network's weights and checks one entry of
it against a finite difference.

Run with::

    python examples/neural_network_integral.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import scipy.integrate
import torch

from padaquad import integrate

torch.manual_seed(0)  # the network's weights and PadaQuad's starting mesh
net = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 3),
).double()
a = torch.tensor([0.0], dtype=torch.float64)
b = torch.tensor([1.0], dtype=torch.float64)
# Keyword options shared by every call: the CPU, a fixed batch size (which
# skips the search for one), and one tolerance for all three outputs.
OPTIONS = {"device": "cpu", "max_batch": 10_000, "atol": 1e-10, "rtol": 0.0}

# %%
# The integral of each output
# ---------------------------
# ``take_gradient=False``: only the value is needed here. ``integrate``
# returns the result and a state; the state's mesh is reused below.
result, state = integrate(
    net, "gk21", mesh_init=a, mesh_final=b, take_gradient=False, **OPTIONS
)
print("integral of each output:", result.integral.tolist())
print(f"converged {result.converged}, {len(result.h)} subintervals")

# %%
# An independent check: ``scipy.integrate.quad``, one output at a time, at a
# much tighter tolerance.
with torch.no_grad():
    for d in range(3):
        reference, _ = scipy.integrate.quad(
            lambda s, d=d: float(net(torch.tensor([[s]], dtype=torch.float64))[0, d]),
            0.0,
            1.0,
            epsabs=1e-13,
            epsrel=0.0,
        )
        error = abs(float(result.integral[d]) - reference)
        print(
            f"output {d}: PadaQuad {float(result.integral[d]):.12f}, scipy {reference:.12f}, difference {error:.1e}"
        )
        assert error < 1e-9

# %%
# The gradient of the integral
# ----------------------------
# With ``take_gradient=True`` PadaQuad backpropagates each batch's integral
# as soon as it is computed, so the gradient of the integral with respect to
# every weight accumulates in ``.grad`` without the whole computation graph
# being held at once. The loss here is the sum of the three integrals (a
# ``loss_fxn`` can define another).
#
# The check below compares one entry of that gradient with a central finite
# difference of two integrals. All three computations use the same fixed mesh
# (``mesh=state.mesh`` with ``max_adaptive_splits=0``, so no subinterval is
# split): two adaptive meshes could differ, and the difference of two
# integrals over 2e-6 would then measure the meshes, not the gradient.
fixed = {"mesh": state.mesh, "max_adaptive_splits": 0, **OPTIONS}


def total_integral() -> float:
    """Return the sum of the three integrals on the fixed mesh."""
    with torch.no_grad():
        r, _ = integrate(net, "gk21", take_gradient=False, **fixed)
    return float(r.integral.sum())


net.zero_grad()
integrate(net, "gk21", take_gradient=True, loss_fxn=lambda r: r.integral.sum(), **fixed)
weight = net[2].weight  # the second hidden layer's weights
gradient = float(weight.grad[0, 0])

step = 1e-6
with torch.no_grad():
    weight[0, 0] += step
    plus = total_integral()
    weight[0, 0] -= 2 * step
    minus = total_integral()
    weight[0, 0] += step
finite_difference = (plus - minus) / (2 * step)
print(f"gradient {gradient:.10f}, finite difference {finite_difference:.10f}")
assert abs(gradient - finite_difference) <= 1e-5 * abs(finite_difference)

# %%
# The three outputs over [0, 1], with the left end of every subinterval of the
# first integration marked below them.
t = torch.linspace(0.0, 1.0, 501, dtype=torch.float64).reshape(-1, 1)
left = a + torch.cumsum(result.h[:, 0], 0) - result.h[:, 0]
with torch.no_grad():
    values = net(t)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(t[:, 0], values, label=[f"output {d}" for d in range(3)])
ax.plot(
    left,
    torch.full_like(left, float(values.min()) - 0.05),
    "|",
    color="gray",
    label="subinterval starts",
)
ax.set(xlabel="t", ylabel="network output")
ax.legend()
fig.tight_layout()
plt.show()
