# padaquad

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

padaquad fills the gap: it is **adaptive quadrature**, not ODE
solving, and it exploits the lack of state coupling to **evaluate many
subintervals in parallel** on GPU. With full autograd through the integration
loop, it is suitable for:

- training a learnable function $\phi_\theta(t)$ whose loss is a path
  integral, and back-propagating through that integral;
- computing $\nabla_\theta \int f_\theta(t)\,dt$ either by autograd
  through the integral (option A) or by integrating $\nabla_\theta f$
  directly (option B);
- one-shot definite integrals where you want batched parallel evaluation
  of a smooth integrand — often two orders of magnitude faster than
  sequential ODE-style integrators on the same problem.

<!-- include-start: what-for -->
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
  parallel trick relies on independence between subintervals. For state-
  coupled problems use [torchdiffeq](https://github.com/rtqichen/torchdiffeq),
  [torchode](https://github.com/martenlienen/torchode), or
  [diffrax](https://github.com/patrick-kidger/diffrax).
- ❌ Multi-dimensional adaptive integration (cubature). Use a sparse-grid
  or Monte-Carlo library.
- ❌ Long-time symplectic / Hamiltonian integration. Use a dedicated
  geometric integrator.

<!-- include-end: what-for -->
## Installation
<!-- include-start: install -->

```bash
pip install padaquad
```

Or from source:

```bash
git clone https://github.com/khegazy/padaquad.git
cd padaquad
pip install -e .
```

Runtime dependencies: Python 3.11+, PyTorch, NumPy, SciPy, einops, psutil.

**For developers:**

```bash
pip install -e ".[dev]"
pre-commit install
```

The dev extras add pytest, ruff, mypy, pre-commit, typos, and torchdiffeq
(used by the speed-test benchmark only).

<!-- include-end: install -->
## Quick start
<!-- include-start: quickstart -->

```python
import math
import torch
from padaquad import integrate

result = integrate(
    f=lambda t: torch.sin(t),
    method="gk21",  # default; Gauss-Kronrod 21-point pair (G10-K21)
    mesh_init=torch.tensor([0.0]),
    mesh_final=torch.tensor([math.pi]),
)
print(result.integral)  # tensor([2.0000])
print(result.integral_error)  # estimated error: signed sum over subintervals
print(result.converged)  # True
```

The integrand `f` takes a tensor `t` of shape `[N, T]` (a batch of points
of the integration variable) and returns a tensor of shape `[N, D]` (its
values there). Because `f` depends only on `t`, the solver can evaluate
the nodes of many subintervals in one call, on a CPU or a GPU.

<!-- include-end: quickstart -->
## Differentiable integration
<!-- include-start: autodiff -->

Every step of the integration is differentiable, so you can take
gradients of the result with respect to anything the integrand depends
on. There are two equivalent ways to compute
$\nabla_\theta \int f_\theta(t)\,dt$:

```python
import torch
from padaquad import adaptive_quadrature, steps

theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
mesh_init = torch.tensor([0.0])
mesh_final = torch.tensor([torch.pi])


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
the number of integrand evaluations is large enough that the graph would
otherwise exceed GPU memory.

<!-- include-end: autodiff -->
## Adaptive mesh and warm-starting

Each call returns a `result.mesh_optimal` — the post-refinement,
post-pruning mesh that the integration converged to. This mesh is the
ideal starting point for a *similar* integrand on the next call: when
training a model, the integrand changes only slightly between iterations,
so re-running adaptive refinement from a coarse random mesh wastes
work. The sketch below assumes your own integrand `f_theta`, bounds and
optimizer.

<!-- not-tested -->
```python
solver = adaptive_quadrature(method="gk21", atol=1e-8, rtol=1e-8)

for epoch in range(N_epochs):
    # same_f=True: from the second iteration on, start from the
    # previous run's optimal mesh and max_batch (they replace any mesh or
    # max_batch passed here) and skip the memory benchmark. The first
    # iteration has nothing stored, so it uses a fresh mesh. The default,
    # same_f=False, never reuses a stored mesh for a different
    # integrand.
    result = solver.integrate(
        f=f_theta,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        same_f=True,
        take_gradient=True,
    )
    optimizer.step()
```

## Available methods

The library ships thirteen quadrature rules across three families:

| Family | Methods | Degree | Notes |
|---|---|---|---|
| Gauss-Kronrod | `gk7`, `gk15`, `gk21`, `gk31` | 11 / 23 / 31 / 47 | Embedded G_n / K_(2n+1) pair: the error estimate is the difference between the Kronrod and the Gauss rule, as in QUADPACK (Gonnet 2012, Sec. 2.4). `gk21` is the default. `gk7` (G3-K7, 9 evaluation slots) fills the low-node end. |
| Clenshaw-Curtis | `cc5`, `cc9`, `cc17`, `cc33`, `cc65` | 5 / 9 / 17 / 33 / 65 | Chebyshev nodes, **nested** by doubling. Excellent on analytic integrands (Trefethen 2008). |
| Runge-Kutta | `adaptive_heun`, `fehlberg2`, `bosh3`, `dopri5` | 1 / 1 / 2 / 4 | Embedded RK pairs from the ODE-solver literature. `fehlberg2`'s error estimate ignores curvature and can be far too small (see `examples/verification/integral_accuracy.py`). |

Plus two methods for the variable-sampling solver (`adaptive_heun` and
`interpolatory3_variable`), whose weights are computed from the actual node
positions. The integration loop currently places every subinterval's nodes at
the method's standard positions and evaluates fresh nodes in both halves of a
split, as the uniform solver does, so `interpolatory3_variable` always has
its middle node at the center, where it is Simpson's rule.

For smooth integrands at moderate-to-high accuracy, prefer `gk21` or
`cc33`. The low-node methods (`gk7`, `cc5`, `cc9`) fill the gap between
the RK rules and the headline `gk15`/`cc17`: useful when the adaptive
controller refines aggressively and a high-node rule would waste
evaluations on smooth regions. The RK methods are kept for
backwards-compatibility and as baselines.

A method's *degree* is the highest polynomial degree it integrates exactly;
`UNIFORM_METHODS["gk21"].degree` gives it, and `order` is `degree + 1`.

### Check the accuracy yourself

Two scripts in [`examples/verification/`](examples/verification/) check the
claims above on your machine, and the test suite runs them:

- `polynomial_degree.py` measures every method's degree from its stored nodes
  and weights, and checks that its error estimate has the strength the solver
  assumes. You can do the same for one method with
  `UNIFORM_METHODS["gk21"].measure_degree()`.
- `integral_accuracy.py` integrates functions with known integrals using every
  method at several tolerances, and checks each actual error against what the
  tolerance allows.

## Use cases
<!-- include-start: use-cases -->

**Path integrals over learned functions.** Compute $\int f(t, \phi_\theta(t))\,dt$ where $\phi_\theta$ is a neural
network parameterizing a path. Backprop through the integral updates
$\theta$ to optimize the integrand against any objective.

**Physics-informed residual minimization.** Solve a differential equation by parameterizing the solution as
$y_\theta(t)$, then minimizing
$\int |\mathcal{L}\,y_\theta(t)|^2\,dt$ where $\mathcal{L}$ is the
differential operator. The collocation residual at every $t$ is just a
function of $t$, so it fits the quadrature framework exactly.

**Expectation under a base measure.** Compute $\int f(t)\,p(t)\,dt$ where $p$ is a known density and $f$ is a
quantity of interest. The integrand is the product `f(t) * p(t)`.

The library does not bundle these as application APIs — they are simply
uses of `integrate(f, ...)` with the right `f`.

<!-- include-end: use-cases -->
## API reference

The full reference, generated from the docstrings, is on the documentation
site, <https://khegazy.github.io/padaquad/>; `help(padaquad.integrate)` shows
the same text. In short:

- `integrate(f, method="gk21", sampling="uniform", **options)` computes one
  integral. Each keyword option goes to the solver's `integrate` call if it is
  one of that call's arguments (`mesh_init`, `mesh_final`, `mesh`, `y0`,
  `take_gradient`, `f_args`, `same_f`, `max_batch`, ...), and to the solver
  constructor otherwise (`atol`, `rtol`, `remove_cut`, `device`, ...).
- `adaptive_quadrature(sampling_type="uniform", method="gk21", **options)`
  builds a solver to reuse across calls, for example to warm-start with
  `same_f=True`. Every constructor option is keyword-only.
- Both return an `IntegrationResult`: the integral, its estimated error (a
  signed sum over subintervals), the mesh, and diagnostics such as
  `converged`, `mesh_optimal` and the split statistics.

## How it works
<!-- include-start: how-it-works -->

1. **Starting mesh.** When no `mesh` is given, $[a, b]$ is cut into about
   $\sqrt{N}$ equal pieces and each piece is cut again at random points, for
   about $N$ = `N_init_steps` mesh points (default 13). The points are random
   because evenly spaced subintervals can line up with the features of a
   periodic integrand, so that every error estimate misses them. Call
   `torch.manual_seed` first for results that repeat exactly.

2. **Batched evaluation.** The nodes of a batch of subintervals are stacked
   into one tensor and the integrand is called once on all of them. The batch
   size fills a fraction `total_mem_usage` of the free device memory, using a
   short measurement of how much memory one evaluation of the integrand takes,
   unless `max_batch` is given.

3. **Error estimate per subinterval.** Each method computes the integral of a
   subinterval twice: with its own weights and with a lower-degree method on
   the same nodes. The difference $e$ is the error estimate. The error ratio
   $|e| / \max(\text{atol}, \text{rtol} \cdot |I|)$, where $I$ is the
   integral (or $|e_d| / (\text{atol} + \text{rtol} \cdot |I_d|)$ per output
   element with `error_norm="failure_fraction"`), decides: below 1 the
   subinterval is accepted, otherwise it is split at its midpoint.

4. **Refinement.** Batches of evaluation and splitting alternate until every
   subinterval is accepted, or until `max_adaptive_splits` stops the
   splitting (then `converged` is usually `False`).

5. **Optimal mesh.** After convergence, pairs of neighbouring subintervals with
   a small combined error are merged, and the result is returned as
   `mesh_optimal`, which `same_f=True` starts from on the next call. A pair is
   merged only when the merged subinterval's predicted error ratio is below
   `remove_cut`. If the error estimate of a subinterval of width $h$ scales as
   $h^k$, merging two of them multiplies the summed error by about
   $2^{k-1}$ (padaquad's own derivation; $k$ is the method's
   `error_exponent`). Each call merges at most one level, so a mesh that is
   finer than needed shrinks over repeated calls. A call that did not
   converge stores a `mesh_optimal` in which the failing subintervals are
   split, so repeated `same_f=True` calls refine further.

<!-- include-end: how-it-works -->
## Comparisons
<!-- include-start: comparisons -->

| Library | Adaptive subdivision | Differentiable | Parallel | Best for |
|---|---|---|---|---|
| `scipy.integrate.quad` | ✓ | ✗ | ✗ | classical one-shot smooth quadrature |
| `torchquad` | ✗ | ✓ | ✓ | fixed-grid rules and Monte Carlo (including VEGAS), also in several dimensions |
| `torchdiffeq` | adaptive step size | ✓ | ✗ | true state-coupled ODEs $\dot y = f(t, y)$ |
| **padaquad** | ✓ | ✓ | ✓ | path integrals, learned-integrand integration, PINN-style residuals |

torchquad's README (commit fc6f9cd, 2026-10-28) states that it does not
implement adaptive subdivision. For one-shot scalar quadrature with no
autograd, `scipy.integrate.quad` is fine. The library's value is when the integrand is a learnable
PyTorch function and many evaluations are needed.

<!-- include-end: comparisons -->
## References

- Piessens, de Doncker-Kapenga, Überhuber, Kahaner. **QUADPACK: A
  subroutine package for automatic integration**. Springer, 1983.
- Netlib QUADPACK routines `dqk15.f`, `dqk21.f`, `dqk31.f`
  (<https://www.netlib.org/quadpack/>): source of the Gauss-Kronrod digits.
- Gonnet. **A review of error estimation in adaptive quadrature**. *ACM
  Comput. Surv.* 44(4), Article 22 (2012).
  [doi:10.1145/2333112.2333117](https://doi.org/10.1145/2333112.2333117)
- Fehlberg. **Low-order classical Runge-Kutta formulas with stepsize
  control and their application to some heat transfer problems**. NASA
  TR R-315 (1969), Table XIV (`fehlberg2`).
- Trefethen. **Is Gauss quadrature better than Clenshaw-Curtis?**
  *SIAM Review* 50:1 (2008), 67-87.
- Sanderse and Veldman. **Constraint-consistent Runge-Kutta methods
  for one-dimensional incompressible multiphase flow**. *J. Comput.
  Phys.* 384 (2019), 170-199, Eq. (119) (`interpolatory3_variable`
  weights).
- Chen, Rubanova, Bettencourt, Duvenaud. **Neural Ordinary Differential
  Equations**. *NeurIPS* 2018. (For comparison; this library is not an
  ODE solver but is related.)

## License

CC-BY-4.0. See [`LICENSE.md`](LICENSE.md) for details.
