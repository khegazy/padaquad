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

## Where to go next

- **[Quickstart](quickstart.md)** — install padaquad and compute your first
  integral in a few lines.
- **User Guide** — [core concepts](user_guide/concepts.md),
  [the method portfolio](user_guide/methods.md),
  [error control](user_guide/error_control.md),
  [differentiable integration](user_guide/autodiff.md), and
  [advanced features](user_guide/advanced.md).
- **[API Reference](reference/padaquad/index.md)** — auto-generated
  documentation for every public module, class, and function.

## Comparisons

| Library | Adaptive | Differentiable | Parallel | Best for |
|---|---|---|---|---|
| `scipy.integrate.quad` | ✓ | ✗ | ✗ | classical one-shot smooth quadrature |
| `torchquad` | ✗ | ✓ | ✓ | uniform-grid Monte-Carlo / Trapezoidal |
| `torchdiffeq` | ✓ | ✓ | ✗ | true state-coupled ODEs $\dot y = f(t, y)$ |
| **padaquad** | ✓ | ✓ | ✓ | path integrals, learned-integrand integration, PINN-style residuals |

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
  Equations**. *NeurIPS* 2018.

## License

CC-BY-4.0. See the [License](about/license.md) page for details.
