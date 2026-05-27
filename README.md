# torchpathdiffeq

**A PyTorch library for adaptive numerical quadrature — computing
$\int_{a}^{b} f(t)\, dt$ for a known integrand $f$, in parallel batches,
with full autograd support.**

Adaptive quadrature is the workhorse behind every problem that reduces to
"integrate this function and we already know how to evaluate it":
computing the action along a known trajectory, the loss along a learned
path, an expectation under a base measure, an ODE residual along a
candidate solution, or simply $\int_{0}^{\pi}\sin(t)\,dt$. Classical
adaptive-quadrature libraries (QUADPACK, `scipy.integrate.quad`) handle
this well, but they are sequential and not differentiable. PyTorch's
`torchdiffeq` is differentiable but solves a different problem (true ODE
integration where the next state depends on the previous), so it must
evaluate steps sequentially even when the integrand has no
state-coupling.

torchpathdiffeq fills the gap: it is **adaptive quadrature**, not ODE
solving, and it exploits the lack of state coupling to **evaluate many
panels in parallel** on GPU. With full autograd through the integration
loop, it is suitable for:

- training a learnable function $\phi_\theta(t)$ whose loss is a path
  integral, and back-propagating through that integral;
- computing $\nabla_\theta \int f_\theta(t)\,dt$ either by autograd
  through the integral (option A) or by integrating $\nabla_\theta f$
  directly (option B);
- one-shot definite integrals where you want batched parallel evaluation
  of a smooth integrand — often two orders of magnitude faster than
  sequential ODE-style integrators on the same problem.

## What this is for

- ✅ Compute $\int_{a}^{b} f(t)\,dt$ where $f$ is given as a callable.
- ✅ Compute $\int f(t, \phi_\theta(t))\,dt$ for a learnable
  $\phi_\theta$, and back-propagate through it.
- ✅ Compute integrals of gradients, expectations, residuals — anything
  the user constructs as a $t\mapsto \mathbb{R}^D$ callable.
- ✅ Run on GPU or CPU; the parallel evaluation pays off most when
  evaluating $f$ is itself non-trivial (a neural net, a PDE solver,
  etc.).

## What this is **not** for

- ❌ True ODE integration $\dot y = f(t, y)$ with state coupling — the
  parallel trick relies on independence between panels. For state-
  coupled problems use [torchdiffeq](https://github.com/rtqichen/torchdiffeq),
  [torchode](https://github.com/martenlienen/torchode), or
  [diffrax](https://github.com/patrick-kidger/diffrax).
- ❌ Multi-dimensional adaptive integration (cubature). Use a sparse-grid
  or Monte-Carlo library.
- ❌ Long-time symplectic / Hamiltonian integration. Use a dedicated
  geometric integrator.

## Installation

```bash
pip install torchpathdiffeq
```

Or from source:

```bash
git clone https://github.com/khegazy/torchpathdiffeq.git
cd torchpathdiffeq
pip install -e .
```

Runtime dependencies: Python 3.10+, PyTorch, NumPy, SciPy, einops, psutil.

### For developers

```bash
pip install -e ".[dev]"
pre-commit install
```

The dev extras add pytest, ruff, mypy, pre-commit, typos, and torchdiffeq
(used by the speed-test benchmark only).

## Quick start

```python
import math
import torch
from torchpathdiffeq import integrate

result = integrate(
    f=lambda t: torch.sin(t),
    method="gk21",  # default; Gauss-Kronrod 21-point pair (G10-K21)
    mesh_init=torch.tensor([0.0]),
    mesh_final=torch.tensor([math.pi]),
)
print(result.integral)  # tensor([2.0000])
print(result.integral_error)  # estimated absolute error
print(result.converged)  # True
```

The integrand `f` takes a tensor `t` of shape `[N, T]` (batched time
points) and returns a tensor of shape `[N, D]` (batched output values).
Because `f` depends only on `t`, the solver can evaluate many panels'
quadrature points simultaneously on GPU.

## Differentiable integration

Every step of the integration is differentiable, so you can take
gradients of the result with respect to anything the integrand depends
on. There are two equivalent ways to compute
$\nabla_\theta \int f_\theta(t)\,dt$:

```python
import torch
from torchpathdiffeq import adaptive_quadrature, steps

theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
mesh_init = torch.tensor([0.0])
mesh_final = torch.tensor([torch.pi.item()])


# (A) Backprop through the integral.
solver = adaptive_quadrature(
    sampling_type=steps.ADAPTIVE_UNIFORM,
    method="gk21",
    atol=1e-8,
    rtol=1e-8,
)
solver.integrate(
    f=lambda t: theta * torch.sin(t),
    mesh_init=mesh_init,
    mesh_final=mesh_final,
    take_gradient=True,  # per-batch backward; accumulates into theta.grad
)
print(theta.grad)  # 2.0


# (B) Integrate the gradient of the integrand directly.
df_dtheta = lambda t: torch.sin(t)  # closed-form derivative
out_b = solver.integrate(f=df_dtheta, mesh_init=mesh_init, mesh_final=mesh_final)
print(out_b.integral)  # also 2.0
```

The two paths agree to machine precision on smooth integrands; this
consistency is verified in `tests/test_autodiff_consistency.py`.

**Memory-bounded gradients.** Setting `take_gradient=True` makes the
solver call `loss.backward()` after each accepted batch instead of
holding the full autograd graph until the end. This is essential when
the number of panel evaluations is large enough that the graph would
otherwise exceed GPU memory.

## Adaptive mesh and warm-starting

Each call returns a `result.mesh_optimal` — the post-refinement,
post-pruning mesh that the integration converged to. This mesh is the
ideal starting point for a *similar* integrand on the next call: when
training a model, the integrand changes only slightly between iterations,
so re-running adaptive refinement from a coarse random mesh wastes
work.

```python
solver = adaptive_quadrature(method="gk21", atol=1e-8, rtol=1e-8)

for epoch in range(N_epochs):
    # On the second and later iterations, reuse the previous run's
    # optimal mesh as the warm-start. Default is reuse_mesh=False so
    # that integrating a *different* integrand never silently shares
    # a stale mesh.
    result = solver.integrate(
        f=f_theta,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        reuse_mesh=(epoch > 0),
        take_gradient=True,
    )
    optimizer.step()
```

## Available methods

The library ships thirteen quadrature rules across three families:

| Family | Methods | Polynomial exactness | Notes |
|---|---|---|---|
| Gauss-Kronrod | `gk7`, `gk15`, `gk21`, `gk31` | 10 / 22 / 31 / 46 | Embedded G_n / K_(2n+1) pair, the canonical adaptive-quadrature workhorse since 1965 (Laurie 1997). `gk21` is the default. `gk7` (G3-K7, 9 evaluation slots) fills the low-node end. |
| Clenshaw-Curtis | `cc5`, `cc9`, `cc17`, `cc33`, `cc65` | 4 / 8 / 16 / 32 / 64 | Chebyshev nodes, **nested** by doubling. Excellent on analytic integrands (Trefethen 2008). |
| Runge-Kutta | `adaptive_heun`, `fehlberg2`, `bosh3`, `dopri5` | 1 / 1 / 2 / 4 | Embedded RK pairs from the ODE-solver literature. |

Plus two variable-node methods (`adaptive_heun` and
`interpolatory3_variable`) that re-weight existing evaluations on mesh
splits — useful when integrand evaluations are expensive.

For smooth integrands at moderate-to-high accuracy, prefer `gk21` or
`cc33`. The low-node methods (`gk7`, `cc5`, `cc9`) fill the gap between
the RK rules and the headline `gk15`/`cc17`: useful when the adaptive
controller refines aggressively and a high-node rule would waste
evaluations on smooth regions. The RK methods are kept for
backwards-compatibility and as baselines.

## Use cases

### Path integrals over learned functions

Compute $\int f(t, \phi_\theta(t))\,dt$ where $\phi_\theta$ is a neural
network parameterizing a path. Backprop through the integral updates
$\theta$ to optimize the integrand against any objective.

### PINN-style residual minimization

Solve a differential equation by parameterizing the solution as
$y_\theta(t)$, then minimizing
$\int |\mathcal{L}\,y_\theta(t)|^2\,dt$ where $\mathcal{L}$ is the
differential operator. The collocation residual at every $t$ is just a
function of $t$, so it fits the quadrature framework exactly.

### Expectation under a base measure

Compute $\int f(t)\,p(t)\,dt$ where $p$ is a known density and $f$ is a
quantity of interest. The integrand is the product `f(t) * p(t)`.

The library does not bundle these as application APIs — they are simply
uses of `integrate(f, ...)` with the right `f`.

## API reference

### Free function

```text
torchpathdiffeq.integrate(
    f, method="gk21", sampling="uniform",
    atol=1e-5, rtol=1e-5,
    mesh=None, mesh_init=None, mesh_final=None,
    y0=None, remove_cut=0.1, total_mem_usage=0.9,
    use_absolute_error_ratio=True, device=None,
    **kwargs,
) -> IntegrationResult
```

One-shot integration. Constructs an `AdaptiveQuadrature` and calls
`integrate()` on it. For repeated calls (e.g. training loops),
instantiate the class directly via `adaptive_quadrature(...)` so
warm-start cache state persists across iterations.

### Class API

```text
torchpathdiffeq.adaptive_quadrature(
    sampling_type, method="gk21", atol=1e-5, rtol=1e-5,
    mesh_init=None, mesh_final=None, f=None,
    remove_cut=0.1, max_batch=None, total_mem_usage=0.9,
    max_path_change=None, use_absolute_error_ratio=True,
    device=None, **kwargs,
) -> UniformAdaptiveQuadrature | VariableAdaptiveQuadrature

solver.integrate(
    f=None, y0=None,
    mesh=None, mesh_init=None, mesh_final=None,
    take_gradient=False, is_training=None,
    reuse_mesh=False, random_initial_mesh=True,
    loss_fxn=None, total_mem_usage=None, max_batch=None,
    N_init_steps=13, f_args=(),
) -> IntegrationResult
```

### `IntegrationResult` fields

- `integral` — the computed integral, shape `[D]`.
- `integral_error` — estimated total error, shape `[D]`.
- `mesh_optimal` — refined-and-pruned barriers from this run, shape
  `[M, T]`. Pass back via `mesh=...` for explicit warm-start.
- `mesh_init`, `mesh_final` — bounds used.
- `nodes` — per-step quadrature points, shape `[N, C, T]`.
- `h` — per-step widths, shape `[N, T]`.
- `y` — integrand evaluations at `nodes`, shape `[N, C, D]`.
- `mesh_quadratures`, `mesh_quadrature_errors`, `error_ratios` — per-step diagnostics.
- `loss`, `gradient_taken`, `y0` — training-loop diagnostics.
- `converged: bool` — `True` for normal completion; `False` only when
  `max_path_change` triggers an early exit on a user-provided mesh.

## How it works

1. **Initial mesh.** When `mesh` is not given, the integration domain
   $[a, b]$ is divided into ~$\sqrt{N_\text{init}}$ top-level segments
   each subdivided into ~$\sqrt{N_\text{init}}+1$ sub-barriers; the
   total mesh has ~$N_\text{init}$ barriers (default 13). Sub-barrier
   placement is randomized by default — uniform spacing accidentally
   aliases with periodic / polynomial-extremum integrands and the
   adaptive controller cannot recover from that. (For deterministic
   reproducibility, call `torch.manual_seed` before `integrate()`.)

2. **Parallel batched evaluation.** All quadrature points across a
   batch of panels are flattened into one tensor and evaluated in a
   single forward pass of the integrand. Batch size is chosen to fit
   a fraction `total_mem_usage` of GPU memory after a small
   benchmarking run that measures the integrand's memory footprint.

3. **Per-step error estimate.** Each method computes both a primary
   integral and an embedded lower-order estimate; their difference is
   the per-step error. The error ratio
   $\text{step\\_error} / (\text{atol} + \text{rtol} \cdot |I|)$
   controls acceptance: ratios $< 1$ accept; ratios $\geq 1$ reject and
   split the step at its midpoint.

4. **Adaptive refinement.** The solver alternates batched evaluation
   with split (high-error steps subdivide) and merge (consecutive
   low-error pairs combine) until every step's ratio is below 1.

5. **Optimal mesh.** After convergence, a final pruning + refinement
   pass produces `mesh_optimal` — the smallest mesh that still meets
   tolerance. This is what `reuse_mesh=True` consumes on the next call.

## Comparisons

| Library | Adaptive | Differentiable | Parallel | Best for |
|---|---|---|---|---|
| `scipy.integrate.quad` | ✓ | ✗ | ✗ | classical one-shot smooth quadrature |
| `torchquad` | ✗ | ✓ | ✓ | uniform-grid Monte-Carlo / Trapezoidal |
| `torchdiffeq` | ✓ | ✓ | ✗ | true state-coupled ODEs $\dot y = f(t, y)$ |
| **torchpathdiffeq** | ✓ | ✓ | ✓ | path integrals, learned-integrand integration, PINN-style residuals |

For one-shot scalar quadrature with no autograd, `scipy.integrate.quad`
is fine. The library's value is when the integrand is a learnable
PyTorch function and many evaluations are needed.

## References

- Piessens, de Doncker-Kapenga, Überhuber, Kahaner. **QUADPACK: A
  subroutine package for automatic integration**. Springer, 1983.
- Laurie. **Calculation of Gauss-Kronrod quadrature rules**. *Math.
  Comp.* 66 (1997), 1133-1145.
- Trefethen. **Is Gauss quadrature better than Clenshaw-Curtis?**
  *SIAM Review* 50:1 (2008), 67-87.
- Sanderse and Veldman. **Constraint-consistent Runge-Kutta methods
  for one-dimensional incompressible multiphase flow**. *J. Comput.
  Phys.* 384 (2019).
- Chen, Rubanova, Bettencourt, Duvenaud. **Neural Ordinary Differential
  Equations**. *NeurIPS* 2018. (For comparison; this library is not an
  ODE solver but is related.)

## License

CC-BY-4.0. See `LICENSE` for details.
