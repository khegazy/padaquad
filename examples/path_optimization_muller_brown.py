"""
Optimizing a reaction path
==========================

A chemical reaction moves a molecule from one minimum of its potential energy
E to another. Along the way it has to climb over a ridge, and the lowest
pass over the ridge, a saddle point of E, is the transition state that sets
the reaction rate. This example finds that pass by optimizing a path whose
loss is an integral along it, computed with padaquad.

The potential is the two-dimensional Müller-Brown surface (Müller and Brown,
1979). The path, the loss and the optimizer follow the Müller-Brown example of
popcornn (``examples/configs/muller_brown.yaml``, commit ``a2581fc`` of
https://github.com/khegazy/popcornn), written here in plain PyTorch:

- the path x(t), t from 0 to 1, runs between two minima A and B:
  ``x(t) = x_A + t (x_B - x_A) + t (1 - t) MLP(2t - 1)``, where the network
  MLP (one hidden layer of 128 GELU units) bends it; the factor t (1 - t)
  keeps its ends fixed;
- the loss is popcornn's PVRE, the integral of |v . F| along the path, with
  the velocity v = dx/dt and the force F = -grad E;
- Adam with learning rate 0.01, at most 1,000 iterations, stopping when the
  norm of the gradient falls below 0.1.

Why this loss finds the pass: by the chain rule, v . F = -dE/dt, the rate at
which the energy changes along the path. So the loss is the total variation
of E along the path: everything it climbs plus everything it descends. A path
that climbs only once, to its highest energy E_max, and then only descends has
a loss of exactly 2 E_max - E_A - E_B; any other path has more. Minimizing the
loss therefore lowers the highest point of the path, until it crosses the
ridge at its lowest pass. (This derivation is ours, not popcornn's.)

Two settings differ from popcornn's configuration: everything runs in float64
on the CPU, and padaquad integrates the loss (``take_gradient=True``) to the
tolerance popcornn uses when it integrates the loss itself, ``atol = 1e-4``,
``rtol = 0`` (popcornn integrates the gradient directly, to its own
tolerance).

Run with::

    python examples/path_optimization_muller_brown.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
from matplotlib import cm, colors

from padaquad import adaptive_quadrature

F64 = torch.float64
torch.manual_seed(0)  # popcornn's seed

# %%
# The potential
# -------------
# E(x, y) = sum over k of A_k exp(a_k (x - x0_k)^2 + b_k (x - x0_k)(y - y0_k)
# + c_k (y - y0_k)^2), with the parameters of Müller and Brown (1979), as in
# popcornn's ``potentials/muller_brown.py``.
A = torch.tensor([-200.0, -100.0, -170.0, 15.0], dtype=F64)
a = torch.tensor([-1.0, -1.0, -6.5, 0.7], dtype=F64)
b = torch.tensor([0.0, 0.0, 11.0, 0.6], dtype=F64)
c = torch.tensor([-10.0, -10.0, -6.5, 0.7], dtype=F64)
x0 = torch.tensor([1.0, 0.0, -0.5, -1.0], dtype=F64)
y0 = torch.tensor([0.0, 0.5, 1.5, 1.0], dtype=F64)


def energy(xy: torch.Tensor) -> torch.Tensor:
    """Return E at points xy of shape [N, 2], as shape [N, 1]."""
    dx, dy = xy[:, :1] - x0, xy[:, 1:] - y0
    return (A * torch.exp(a * dx**2 + b * dx * dy + c * dy**2)).sum(-1, keepdim=True)


def gradient(p: np.ndarray) -> np.ndarray:
    """Return grad E at one point, for scipy."""
    xy = torch.tensor(p, dtype=F64).reshape(1, 2).requires_grad_()
    (g,) = torch.autograd.grad(energy(xy).sum(), xy)
    return g.reshape(2).numpy()


# %%
# The minima and the two saddle points, found by solving grad E = 0 from rough
# positions. The minimum-energy path from A to B runs over S1, down into the
# intermediate minimum C and over S2. S1 is the higher saddle, so every path
# from A to B climbs at least to E(S1): S1 is the lowest pass.
points = {}
for name, guess in {
    "A": (-0.558, 1.442),
    "B": (0.623, 0.028),
    "C": (-0.05, 0.467),
    "S1": (-0.82, 0.62),
    "S2": (0.21, 0.29),
}.items():
    solution = scipy.optimize.root(gradient, guess, tol=1e-12)
    assert solution.success, name
    points[name] = solution.x
    xy = torch.tensor(solution.x, dtype=F64).reshape(1, 2)
    print(
        f"{name:2s} at ({solution.x[0]:+.4f}, {solution.x[1]:+.4f}), E = {float(energy(xy)):.4f}"
    )
E_S1 = float(energy(torch.tensor(points["S1"], dtype=F64).reshape(1, 2)))

# %%
# The path and the loss
# ---------------------
# The ends are popcornn's (A and B rounded to three decimals; their energies
# differ from the minima's by less than 1e-4).
x_A = torch.tensor([[-0.558, 1.442]], dtype=F64)
x_B = torch.tensor([[0.623, 0.028]], dtype=F64)
mlp = torch.nn.Sequential(
    torch.nn.Linear(1, 128), torch.nn.GELU(), torch.nn.Linear(128, 2)
).double()


def path(t: torch.Tensor) -> torch.Tensor:
    """Return the path's points at t of shape [N, 1], as shape [N, 2]."""
    return x_A + t * (x_B - x_A) + t * (1 - t) * mlp(2 * t - 1)


evaluations = 0  # points at which the loss's integrand was evaluated


def pvre(t: torch.Tensor) -> torch.Tensor:
    """
    Return |v . F| at the path's points t (shape [N, 1]), as shape [N, 1].

    Both derivatives come from autograd, and both keep their graph
    (``create_graph=True``) so that the loss can be differentiated with
    respect to the network's weights. Summing over the batch before
    differentiating is exact here, because each point of the path depends
    only on its own t.
    """
    global evaluations  # noqa: PLW0603
    evaluations += len(t)
    t = t.detach().requires_grad_()  # keeps padaquad's node placement out of the graph
    x = path(t)
    v = torch.cat(
        [torch.autograd.grad(x[:, i].sum(), t, create_graph=True)[0] for i in range(2)],
        dim=-1,
    )
    (grad_E,) = torch.autograd.grad(energy(x).sum(), x, create_graph=True)
    return (v * -grad_E).sum(-1, keepdim=True).abs()


# %%
# The optimization
# ----------------
# One solver for every iteration. gk7 is popcornn's default method (its
# documentation reports that it was 30% faster than gk21 for the same
# transition states). ``take_gradient=True`` backpropagates the loss batch by
# batch into the network's ``.grad``; ``state=state`` starts every iteration
# from the previous one's mesh, which the small change of the path keeps
# nearly right.
solver = adaptive_quadrature(
    method="gk7", atol=1e-4, rtol=0.0, device="cpu", max_batch=10_000
)
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
lo = torch.tensor([0.0], dtype=F64)
hi = torch.tensor([1.0], dtype=F64)
t_plot = torch.linspace(0.0, 1.0, 101, dtype=F64).reshape(-1, 1)
history = {"loss": [], "evaluations": [], "paths": [], "energies": []}
state = None
for iteration in range(1000):
    optimizer.zero_grad()
    evaluations = 0
    result, state = solver.integrate(
        pvre, mesh_init=lo, mesh_final=hi, state=state, take_gradient=True
    )
    grad_norm = float(torch.cat([p.grad.reshape(-1) for p in mlp.parameters()]).norm())
    with torch.no_grad():
        points_now = path(t_plot)
        history["paths"].append(points_now.numpy())
        history["energies"].append(energy(points_now)[:, 0].numpy())
    history["loss"].append(float(result.integral))
    history["evaluations"].append(evaluations)
    if iteration % 25 == 0:
        print(
            f"iteration {iteration:4d}: loss {float(result.integral):.4f}, gradient norm {grad_norm:.3f}, {evaluations} evaluations"
        )
    if grad_norm < 0.1:  # popcornn's threshold
        break
    optimizer.step()
print(
    f"stopped at iteration {iteration}: loss {history['loss'][-1]:.4f}, gradient norm {grad_norm:.3f}"
)

# %%
# The checks: the highest point of the final path is the pass S1, and the loss
# reached its lower bound 2 E(S1) - E(x_A) - E(x_B), up to the integration
# error.
with torch.no_grad():
    dense = path(torch.linspace(0.0, 1.0, 20001, dtype=F64).reshape(-1, 1))
    energies = energy(dense)[:, 0]
    highest = dense[int(energies.argmax())].numpy()
    bound = 2 * E_S1 - float(energy(x_A)) - float(energy(x_B))
print(
    f"highest point of the path ({highest[0]:+.4f}, {highest[1]:+.4f}), E = {float(energies.max()):.4f}"
)
print(
    f"S1                        ({points['S1'][0]:+.4f}, {points['S1'][1]:+.4f}), E = {E_S1:.4f}"
)
print(f"final loss {history['loss'][-1]:.4f}, lower bound {bound:.4f}")
assert np.linalg.norm(highest - points["S1"]) < 0.01
assert abs(history["loss"][-1] - bound) < 0.05

# %%
# The path during the optimization
# --------------------------------
# The path starts as the straight line from A to B (lightest) and bends until
# it crosses the ridge at S1 (darkest), without descending into C.
n = len(history["loss"])
shown = sorted({0, n - 1, *np.unique(np.geomspace(1, n - 1, 4).astype(int))})
shade = colors.Normalize(vmin=-0.3 * (n - 1), vmax=n - 1)  # skip the palest blues
grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.2, 300), np.linspace(-0.4, 2.0, 300))
grid = torch.tensor(np.stack([grid_x.ravel(), grid_y.ravel()], -1), dtype=F64)
with torch.no_grad():
    surface = energy(grid)[:, 0].reshape(grid_x.shape).numpy()
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.contour(
    grid_x,
    grid_y,
    surface,
    levels=np.linspace(-150, 60, 22),
    colors="0.7",
    linewidths=0.6,
)
for i in shown:
    ax.plot(*history["paths"][i].T, color=cm.Blues(shade(i)), lw=2)
for name, (px, py) in points.items():
    ax.plot(px, py, "o" if name in ("A", "B", "C") else "X", color="black", ms=8)
    ax.annotate(name, (px, py), xytext=(6, 6), textcoords="offset points")
fig.colorbar(cm.ScalarMappable(norm=shade, cmap="Blues"), ax=ax, label="iteration")
ax.set(
    xlabel="x", ylabel="y", title="Paths at six iterations on the Müller-Brown energy"
)
fig.tight_layout()

# %%
# What changed with the iterations
# --------------------------------
# Left: the loss falls to its lower bound (dashed); the computed loss carries
# the integration error, so it can end a little below it. Middle: the energy
# along the path at the same six iterations: the barrier drops to E(S1)
# (dashed). Right: the integrand evaluations of each iteration.
fig, (left, middle, right) = plt.subplots(1, 3, figsize=(12, 3.6))
left.plot(history["loss"], lw=1.5)
left.axhline(bound, ls="--", color="0.4", lw=1, label="2 E(S1) - E(A) - E(B)")
left.set(xlabel="iteration", ylabel="loss (integral of |v . F|)", yscale="log")
left.legend()
for i in shown:
    middle.plot(
        t_plot[:, 0],
        history["energies"][i],
        color=cm.Blues(shade(i)),
        lw=1.5,
        label=f"iteration {i}",
    )
middle.axhline(E_S1, ls="--", color="0.4", lw=1)
middle.set(xlabel="t (position along the path)", ylabel="E")
middle.legend(fontsize=7)
right.plot(history["evaluations"], lw=1.5)
right.set(xlabel="iteration", ylabel="integrand evaluations", yscale="log")
fig.tight_layout()
plt.show()
