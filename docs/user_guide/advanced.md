# Advanced features

## Warm-starting (`reuse_mesh=True`)

Each call returns a `result.mesh_optimal` — the post-refinement, post-pruning
mesh that the integration converged to. This mesh is the ideal starting point
for a *similar* integrand on the next call: when training a model, the
integrand changes only slightly between iterations, so re-running adaptive
refinement from a coarse random mesh wastes work.

```python
solver = adaptive_quadrature(method="gk21", atol=1e-8, rtol=1e-8)

for epoch in range(N_epochs):
    result = solver.integrate(
        f=f_theta,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        reuse_mesh=(epoch > 0),
        take_gradient=True,
    )
    optimizer.step()
```

How it works:

- The solver caches `mesh_optimal` from the previous successful call.
- On the next call with `reuse_mesh=True`, the cached mesh is filtered to
  `[mesh_init, mesh_final]`, padded if needed, and used as the initial mesh.
- Integrand identity is sanity-checked via `id(f)`. On id mismatch the solver
  warns but proceeds.
- The default is `reuse_mesh=False`, so calling the same solver on a *new*
  integrand never silently reuses a stale mesh.

You can also pass a mesh explicitly via `mesh=result.mesh_optimal` for full
control.

## The `y0` additive offset

`y0` is an optional initial value of the integral accumulator. The returned
`result.integral` is `y0 + ∫f(t)dt`. The default is zeros.

- Per-batch integral computations inside the loop use `y0=zeros`, so each
  batch returns only its step contributions. The user-supplied `y0` is added
  once at the final result.
- `result.y0` echoes the value that was used (after dtype/device coercion) so
  callers can recover what offset was applied.

## Extra integrand arguments (`f_args`)

`f_args` is a tuple forwarded positionally to the integrand: `f(t, *f_args)`.
Use it to pass path parameters or other fixed arguments without closing over
them in a lambda.

## Memory management

The solver auto-sizes its batches to the available device memory:

- `_setup_memory_checks` benchmarks `f` with increasing `N`, measuring the
  per-evaluation memory cost (with a 2.1× safety factor).
- The maximum number of `f` evaluations per batch is
  `usable_memory / per_eval_size`.
- `usable_memory = free - buffer`, where
  `buffer = (1 - total_mem_usage) * total_memory`.
- Both CUDA (`torch.cuda.mem_get_info`) and CPU (`psutil.virtual_memory`) are
  supported.

You can override the auto-sizing with `max_batch`, and tune the memory
headroom with `total_mem_usage` (default `0.9`). Out-of-memory errors during
integration and benchmarking are caught: `max_batch` is lowered and the
evaluation retried. `max_batch` is cached between calls when `f` is the same
function, avoiding a full re-benchmark on every call.

## Data types

- `float64` and `float32` are the supported runtime dtypes. `float16` is
  refused at construction: its ~`1e-3` precision floor exceeds typical
  adaptive tolerances.
- The solver maintains two tolerance levels: `atol`/`rtol` (integration error
  control, may be float32) and `atol_assert`/`rtol_assert` (geometric
  assertions like time ordering, always looser).
- Methods are cloned per solver instance, so dtype/device mutations stay
  isolated to one solver and never propagate through shared singletons.

See the [API Reference](../reference/padaquad/index.md) for the full signatures of
[`adaptive_quadrature()`](../reference/padaquad/runge_kutta.md) and
[`integrate()`](../reference/padaquad/integrate.md).
