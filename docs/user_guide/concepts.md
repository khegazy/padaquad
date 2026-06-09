# Core concepts

## Quadrature, not ODE solving

padaquad computes definite integrals $\int_a^b f(t)\,dt$ for a **known**
integrand $f$. The critical distinction from an ODE solver is that the
integrand `f` depends only on `t` (and optional extra args), **not** on the
accumulated state `y`. This independence between panels is what enables
parallel evaluation.

This is numerical quadrature, not ODE solving — for state-coupled
$\dot y = f(t, y)$, use
[torchdiffeq](https://github.com/rtqichen/torchdiffeq),
[torchode](https://github.com/martenlienen/torchode), or
[diffrax](https://github.com/patrick-kidger/diffrax) directly.

## How it works

1. **Initial mesh.** When `mesh` is not given, the integration domain
   $[a, b]$ is divided into ~$\sqrt{N_\text{init}}$ top-level segments
   each subdivided into ~$\sqrt{N_\text{init}}+1$ sub-barriers; the total
   mesh has ~$N_\text{init}$ barriers (default 13). Sub-barrier placement
   is randomized by default — uniform spacing accidentally aliases with
   periodic / polynomial-extremum integrands and the adaptive controller
   cannot recover from that. (For deterministic reproducibility, call
   `torch.manual_seed` before `integrate()`.)

2. **Parallel batched evaluation.** All quadrature points across a batch of
   panels are flattened into one tensor and evaluated in a single forward
   pass of the integrand. Batch size is chosen to fit a fraction
   `total_mem_usage` of GPU memory after a small benchmarking run that
   measures the integrand's memory footprint.

3. **Per-step error estimate.** Each method computes both a primary integral
   and an embedded lower-order estimate; their difference is the per-step
   error. The error ratio controls acceptance: ratios $< 1$ accept; ratios
   $\geq 1$ reject and split the step at its midpoint. (See
   [Error control](error_control.md).)

4. **Adaptive refinement.** The solver alternates batched evaluation with
   split (high-error steps subdivide) and merge (consecutive low-error pairs
   combine) until every step's ratio is below 1.

5. **Optimal mesh.** After convergence, a final pruning + refinement pass
   produces `mesh_optimal` — the smallest mesh that still meets tolerance.
   This is what `reuse_mesh=True` consumes on the next call (see
   [Advanced features](advanced.md)).

## The mesh

`mesh` is the boundary array dividing $[\text{mesh\_init}, \text{mesh\_final}]$
into integration steps (panels). Between consecutive barriers, `C` quadrature
points (`nodes`) are placed per the rule's tableau. `mesh_trackers` is a
boolean array where `True` means the panel still needs evaluation.

## Tensor shape conventions

| Symbol | Meaning |
|---|---|
| **N** | number of integration steps in a batch |
| **C** | number of quadrature points per step (from the rule's tableau, e.g. 4 for bosh3, 7 for dopri5, 23 for gk21 with endpoint padding) |
| **T** | dimensionality of time (usually 1, but multi-D is supported) |
| **D** | dimensionality of `f`'s output |

Key tensors: `nodes: [N, C, T]`, `y: [N, C, D]`, `tableau_b: [1, C, 1]` or
`[N, C, 1]`, `y0: [D]`, mesh barriers: `[M, T]`.

## Uniform vs. variable sampling

- **Uniform** sampling places nodes at fixed fractional positions within each
  panel (the tableau `c` values), and the tableau `b` weights are constants.
  On a split, old evaluations are discarded and fresh quadrature points are
  placed at the standard positions in the new sub-panels.

- **Variable** sampling places nodes at arbitrary positions; tableau `b`
  weights are computed dynamically. On a split it **reuses** existing
  evaluations by inserting new midpoints between consecutive points — useful
  when integrand evaluations are expensive.

The two concrete solver classes, `UniformAdaptiveQuadrature` and
`VariableAdaptiveQuadrature`, are selected via the `sampling_type` argument to
[`adaptive_quadrature()`](../reference/padaquad/runge_kutta.md) using the
[`steps`](../reference/padaquad/base.md) enum.

## Use cases

### Path integrals over learned functions

Compute $\int f(t, \phi_\theta(t))\,dt$ where $\phi_\theta$ is a neural
network parameterizing a path. Backprop through the integral updates $\theta$
to optimize the integrand against any objective.

### PINN-style residual minimization

Solve a differential equation by parameterizing the solution as $y_\theta(t)$,
then minimizing $\int |\mathcal{L}\,y_\theta(t)|^2\,dt$ where $\mathcal{L}$ is
the differential operator. The collocation residual at every $t$ is just a
function of $t$, so it fits the quadrature framework exactly.

### Expectation under a base measure

Compute $\int f(t)\,p(t)\,dt$ where $p$ is a known density and $f$ is a
quantity of interest. The integrand is the product `f(t) * p(t)`.

The library does not bundle these as application APIs — they are simply uses
of `integrate(f, ...)` with the right `f`.
