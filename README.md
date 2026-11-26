# PadaQuad

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

PadaQuad fills the gap: it is **adaptive quadrature**, not ODE
solving, and it exploits the lack of state coupling to **evaluate many
subintervals in parallel** on GPU. With full autograd through the integration
loop, it is suitable for:

- training a learnable function $\phi_\theta(t)$ whose loss is a path
  integral, and back-propagating through that integral;
- computing $\nabla_\theta \int f_\theta(t)\,dt$ either by autograd
  through the integral (option A) or by integrating $\nabla_\theta f$
  directly (option B);
- one-shot definite integrals where you want batched parallel evaluation
  of a smooth integrand; the
  [Performance page](https://khegazy.github.io/padaquad/performance.html)
  compares it with torchdiffeq, with the measurements and how to reproduce
  them.

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

Runtime dependencies: Python 3.11+, PyTorch, NumPy, SciPy, psutil.

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

result, state = integrate(
    torch.sin,
    mesh_init=torch.tensor([0.0], dtype=torch.float64),
    mesh_final=torch.tensor([math.pi], dtype=torch.float64),
)
print(round(float(result.integral), 6))  # 2.0
print(float(result.integral_error))  # estimated error: signed sum over subintervals
print(result.converged)  # True
```

The integrand `f` takes a tensor `t` of shape `[N, T]` (a batch of points
of the integration variable) and returns a tensor of shape `[N, D]` (its
values there). Because `f` depends only on `t`, the solver can evaluate
the nodes of many subintervals in one call, on a CPU or a GPU.

<!-- include-end: quickstart -->
## Reusing the mesh and batch size
<!-- include-start: reuse -->

When you integrate a similar integrand many times, for example once per
optimizer step of a model, each call can start where the last one ended.
Every call returns, beside its result, a `state` with what is worth reusing:
`result, state = solver.integrate(...)`.

- `state.mesh`, the optimal mesh (the final mesh with low-error subintervals
  merged), so the next call needs few splits;
- `state.max_batch`, the batch size (the most points per call of the
  integrand known to fit), and `state.max_batch_converged`, whether it may
  still grow.

Pass it to the next call as `state=state`. The solver keeps nothing
between calls: a call reuses exactly what you pass, and no call can change what
another one reuses.

```python
import torch
from padaquad import adaptive_quadrature

solver = adaptive_quadrature("uniform", "gk21", device="cpu")
a = torch.tensor([0.0], dtype=torch.float64)
b = torch.tensor([1.0], dtype=torch.float64)

# Training: the first iteration builds a fresh mesh on [a, b]; every later
# one starts from the state of the one before.
result, state = solver.integrate(lambda t: torch.cos(t), mesh_init=a, mesh_final=b)
for scale in (1.1, 1.2):
    result, state = solver.integrate(lambda t, s=scale: torch.cos(s * t), state=state)

# Validation on [0, 0.5]: a bound inside the state's mesh cuts it. state is
# not reassigned, so the training loop continues unchanged.
half, _ = solver.integrate(lambda t: torch.cos(1.2 * t), state=state, mesh_final=b / 2)
print(round(float(result.integral), 4), round(float(half.integral), 4))  # 0.7767 0.4705
```

| To reuse | Pass |
|---|---|
| the mesh and the batch size (same or similar integrand) | `state=state`, and bounds inside its mesh to integrate part of it |
| only the batch size (another interval or your own mesh) | the bounds or your mesh, and `max_batch=state.max_batch` (then fixed), or `state=dataclasses.replace(state, mesh=None)` with the bounds |
| nothing | the bounds, or a mesh |

A call needs a `mesh`, a `state` with a mesh, or both bounds. Bounds inside the
mesh cut it; a bound outside it extends it, and the integration refines the new
subinterval. Without `max_batch` or a state, the first call evaluates its
first pass in one call of the integrand (at most 10,000 points) and then lets
the batch size grow by factors of 10 until the integrand runs out of memory; a
state carries what it found to the next call.

<!-- include-end: reuse -->
More, including several integrals per step, checkpoints and when not to reuse
`max_batch`:
[Reusing the mesh and batch size](https://khegazy.github.io/padaquad/user_guide/repeated_calls.html).

## Differentiable integration
<!-- include-start: autodiff -->

Every operation of the integration is differentiable, so you can take
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
out_b, _ = solver.integrate(f=df_dtheta, mesh_init=mesh_init, mesh_final=mesh_final)
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
## Available methods

The library ships thirteen quadrature methods in three families:
Gauss-Kronrod (`gk7`, `gk15`, `gk21`, `gk31`; `gk21` is the default),
Clenshaw-Curtis (`cc5`, `cc9`, `cc17`, `cc33`, `cc65`) and Runge-Kutta
(`adaptive_heun`, `fehlberg2`, `bosh3`, `dopri5`). The
[methods page](https://khegazy.github.io/padaquad/user_guide/methods.html)
gives each one's degree, order and sources, generated from the code.
`fehlberg2`'s error estimate ignores curvature and can be far too small (see
`examples/verification/integral_accuracy.py`).

Plus two methods for the variable-sampling solver (`adaptive_heun` and
`interpolatory3_variable`), whose weights are computed from the actual node
positions. The integration loop currently places every subinterval's nodes at
the method's standard positions and evaluates fresh nodes in both halves of a
split, as the uniform solver does, so `interpolatory3_variable` always has
its middle node at the center, where it is Simpson's rule.

For smooth integrands at moderate-to-high accuracy, prefer `gk21` or
`cc33`. The low-node methods (`gk7`, `cc5`, `cc9`) waste fewer evaluations on
smooth regions while the mesh is refined around a rough feature. The
Runge-Kutta methods are kept as baselines.

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

- `integrate(f, method, sampling, **options)` computes one integral. Each
  keyword option goes to the solver's `integrate` call if it is one of that
  call's arguments (`mesh_init`, `mesh_final`, `mesh`, `y0`, `take_gradient`,
  `f_args`, `max_batch`, ...), and to the solver constructor otherwise
  (`atol`, `rtol`, `remove_cut`, `merge_rule`, `device`, ...).
- `adaptive_quadrature(sampling_type, method, **options)` builds a solver to
  call many times; pass `state=state` to start a call from an earlier
  one (see
  "Reusing the mesh and batch size" above). Every constructor option is
  keyword-only.
- Both return a pair `result, state`. The `IntegrationResult` holds the
  integral, its estimated error (a signed sum over subintervals), the mesh,
  and diagnostics such as `converged` and the split statistics; the
  `IntegrationState` holds the optimal mesh and the batch size for the next
  call.

## How it works
<!-- include-start: how-it-works -->

1. **Starting mesh.** When no `mesh` is given, $[a, b]$ is cut into a few
   equal pieces, each split again at random points (`N_init_steps` sets the
   size). The points are random because evenly spaced subintervals can line up
   with the features of a periodic integrand, so that every error estimate
   misses them. Call `torch.manual_seed` first for results that repeat
   exactly.

2. **Batched evaluation.** The nodes of a batch of subintervals are stacked
   into one tensor and the integrand is called once on all of them. Unless
   `max_batch` is given, the first call is the first pass (at most 10,000
   points), and the batch size grows by factors of 10 whenever more points
   are waiting, until the integrand runs out of memory; then it shrinks and
   stops growing. Every point the integrand is called at belongs to the
   integral.

3. **Error estimate per subinterval.** Each method computes the integral of a
   subinterval twice: with its own weights and with a lower-degree method on
   the same nodes. The difference $e$ is the error estimate. The error ratio
   $|e| / \max(\text{atol}, \text{rtol} \cdot |I|)$, where $I$ is the
   integral (per output element, $|e_d| / (\text{atol} + \text{rtol} \cdot
   |I_d|)$ with `error_norm="failure_fraction"`), decides: below 1 the
   subinterval is accepted, otherwise it is split at its midpoint (or into
   four equal pieces, with `split_rule="accelerate"`).

4. **Refinement.** Batches of evaluation and splitting alternate until every
   subinterval is accepted, or until `max_adaptive_splits` stops the
   splitting (then `converged` is usually `False`). With
   `enforce_convergence` (the default when `take_gradient=False`), accepted
   subintervals are judged again in every batch from their stored error
   estimates, and split if the changed integral makes them fail; a fixed
   `error_integral_reference` is used until every subinterval is accepted,
   then replaced by the integral.

5. **Optimal mesh.** After convergence, pairs of neighbouring subintervals are
   merged into the state's mesh, which a later call starts from when it is
   passed as `state=`, when the merged subinterval is predicted to pass with a
   margin. By
   default the prediction interpolates the integrand values already stored for
   the two neighbours onto the merged subinterval's nodes, so no new
   evaluation is needed; `remove_cut` sets the margin and `merge_rule` the
   prediction.

<!-- include-end: how-it-works -->
## Comparisons
<!-- include-start: comparisons -->

| Library | Adaptive subdivision | Differentiable | Parallel | Best for |
|---|---|---|---|---|
| `scipy.integrate.quad` | ✓ | ✗ | ✗ | classical one-shot smooth quadrature |
| `torchquad` | ✗ | ✓ | ✓ | fixed-grid methods and Monte Carlo (including VEGAS), also in several dimensions |
| `torchdiffeq` | adaptive step size | ✓ | ✗ | true state-coupled ODEs $\dot y = f(t, y)$ |
| **PadaQuad** | ✓ | ✓ | ✓ | path integrals, learned-integrand integration, physics-informed residual losses |

torchquad's README (commit fc6f9cd, 2026-10-28) states that it does not
implement adaptive subdivision. For one-shot scalar quadrature with no
autograd, `scipy.integrate.quad` is fine. The library's value is when the integrand is a learnable
PyTorch function and many evaluations are needed.

<!-- include-end: comparisons -->
## References

Generated from `docs/references.bib`, which lists only sources that were
opened and checked (`python docs/_ext/readme_references.py` rewrites this
list).

<!-- references-start -->
- Berrut, Jean-Paul, Trefethen, Lloyd N. (2004). **Barycentric Lagrange Interpolation.** *SIAM Review* 46(3), 501-517. [doi:10.1137/S0036144502417715](https://doi.org/10.1137/S0036144502417715)
- Chen, Ricky T. Q., Rubanova, Yulia, Bettencourt, Jesse, Duvenaud, David K. (2018). **Neural Ordinary Differential Equations.** *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*. <https://papers.nips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html>
- NIST (2026). **NIST Digital Library of Mathematical Functions.** *Online, accessed 2026-11-12*. <https://dlmf.nist.gov/18.9>
- Fehlberg, Erwin (1969). **Low-Order Classical Runge-Kutta Formulas with Stepsize Control and Their Application to Some Heat Transfer Problems.** *NASA Marshall Space Flight Center, TR R-315*. <https://ntrs.nasa.gov/citations/19690021375>
- Gander, Walter, Gautschi, Walter (2000). **Adaptive Quadrature--Revisited.** *BIT Numerical Mathematics* 40(1), 84-101. [doi:10.1023/A:1022318402393](https://doi.org/10.1023/A:1022318402393)
- Gonnet, Pedro (2012). **A Review of Error Estimation in Adaptive Quadrature.** *ACM Computing Surveys* 44(4), 22:1-22:36. arXiv:1003.4629. [doi:10.1145/2333112.2333117](https://doi.org/10.1145/2333112.2333117)
- Piessens, Robert, de Doncker, Elise (1983). **QUADPACK routines dqk15, dqk21, dqk31 (double-precision Gauss-Kronrod quadrature).** *Netlib*. <https://netlib.org/quadpack/>
- Sanderse, B., Veldman, A. E. P. (2019). **Constraint-Consistent Runge-Kutta Methods for One-Dimensional Incompressible Multiphase Flow.** *Journal of Computational Physics* 384, 170-199. arXiv:1809.06114. [doi:10.1016/j.jcp.2019.02.001](https://doi.org/10.1016/j.jcp.2019.02.001)
- The SciPy community (2026). **SciPy source file scipy/integrate/_quad_vec.py and scipy/integrate/_ivp/rk.py.** *GitHub, tags v1.18.1 (tag object c2df8ace) and v1.17.0 (tag object e2f9e521)*. <https://github.com/scipy/scipy/blob/v1.18.1/scipy/integrate/_quad_vec.py>
- Chen, Ricky T. Q. (2018). **torchdiffeq.** <https://github.com/rtqichen/torchdiffeq>
- esa/torchquad contributors (2026). **torchquad.** <https://github.com/esa/torchquad>
- Trefethen, Lloyd N. (2008). **Is Gauss Quadrature Better than Clenshaw-Curtis?** *SIAM Review* 50(1), 67-87. [doi:10.1137/060659831](https://doi.org/10.1137/060659831)
<!-- references-end -->

## Citing

To cite PadaQuad, use [`CITATION.cff`](CITATION.cff) (GitHub shows it under
"Cite this repository").

## License

CC-BY-4.0. See [`LICENSE.md`](LICENSE.md) for details.
