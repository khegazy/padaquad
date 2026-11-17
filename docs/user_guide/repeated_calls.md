(ug-reuse)=
# Reusing the mesh and batch size

```{include} ../../README.md
:start-after: "<!-- include-start: reuse -->"
:end-before: "<!-- include-end: reuse -->"
```

## How a call chooses its interval

| The call passes | Interval integrated | Starting mesh |
|---|---|---|
| `mesh` only | `mesh[0]` to `mesh[-1]` | `mesh`, as given |
| `mesh` and bounds inside it | the bounds; a missing one is the mesh's end | `mesh` cut to the bounds: its points between them, plus the bounds |
| `mesh` and a bound outside it | — | `ValueError` |
| both bounds, no `mesh` | the bounds | a fresh mesh (`N_init_steps`, `random_initial_mesh`) |
| one bound or none, no `mesh` | — | `ValueError` |

- Every given bound becomes an end point of the starting mesh, so its
  gradient is computed as for any mesh end, also when it equals an end of a
  `mesh_optimal` (which is detached from autograd).
- A bound within a few rounding steps of its mesh end (4 machine epsilons of
  the coarser of the bound's and the mesh's dtypes, relative to that end)
  replaces that end, so a bound that differs from `mesh[-1]` only by float32
  rounding neither raises nor adds a near-zero-width subinterval. For the same
  reason, when a bound cuts the mesh, the mesh points within rounding of it
  are dropped.
- A given mesh must have shape `[M, T]` with at least two increasing points.
  Bounds are tensors of shape `[T]` on every call, with or without a mesh
  (`torch.tensor([0.0])`, not `torch.tensor(0.0)`).
- The given mesh's dtype is the call's dtype (see {doc}`devices_memory`), so bounds
  of another dtype are converted to it.

## Extending a mesh

A bound outside the given mesh raises an error, because the call would
otherwise integrate an interval you did not give. To integrate further, for
example when the training interval grows, add the new end point to the mesh
yourself, on the mesh's device and in its dtype:

<!-- not-tested -->
```python
mesh = result.mesh_optimal
new_end = torch.tensor([2.0]).to(mesh)  # same device and dtype as the mesh
result = solver.integrate(
    f, mesh=torch.cat([mesh, new_end.reshape(1, -1)]), max_batch=result.max_batch
)
```

## Several integrals per step

Keep one result per integral, for example one per loss term or per interval,
and pass each its own `mesh_optimal`. A batch size can be shared between
integrals of the same integrand.

## Checkpoints

The two values are ordinary tensors and integers, so they can be saved with a
model and loaded to resume training:

<!-- not-tested -->
```python
torch.save({"mesh": result.mesh_optimal, "max_batch": result.max_batch}, path)
state = torch.load(path, map_location=device)
result = solver.integrate(f, mesh=state["mesh"], max_batch=state["max_batch"])
```

## When not to reuse `max_batch`

A batch size depends on the integrand and on how the call is run. Measure a
new one (pass no `max_batch`, or call `solver.get_max_batch(f=f, t=t,
take_gradient=...)`) when any of these changes:

- `take_gradient`: with `take_gradient=True` a batch keeps its autograd graph
  until it is backpropagated, so the measurement leaves room for it (a factor
  2.1, `_GRADIENT_MEMORY_SAFETY_FACTOR`). A batch size measured with
  `take_gradient=False` is too large for `take_gradient=True`; the other way
  round it is only smaller than needed.
- the dtype: a float64 point takes twice the memory of a float32 one;
- the device.

## `result.max_batch` and `solver.get_max_batch()`

- `result.max_batch` is the `max_batch` a call was given, or the one its
  memory measurement found (the integration itself sizes each batch from the
  memory free at that moment), lowered if the call ran out of memory and
  retried with smaller batches. Passing it on carries that
  reduction to the next call.
- `solver.get_max_batch(f=f, t=t, ...)` runs the memory measurement for `f`
  now and returns the batch size, so it can be chosen before any integral.
  Pass `t` in the dtype of the calls that will use the result.
- `solver.get_max_batch()` without `f` estimates a batch size from the last
  measurement this solver ran and the memory free now. Any call that measures
  changes it, so prefer `result.max_batch` of the call you want to repeat.

A call that raises an error returns nothing to reuse. It can still have
switched the solver's dtype, which follows the dtype of the last `mesh` or
bounds passed (see {doc}`devices_memory`).

## Offsets and extra arguments

- `y0` is added to the returned integral: `result.integral` is
  `y0` + $\int f\,dt$, and `result.y0` echoes the value used.
- `f_args` is a tuple passed to the integrand after `t`: the solver calls
  `f(t, *f_args)`.

(ug-tracked-variables)=
## Tracked variables

The integrand may return a pair `(values, tracked_variables)`, where
`tracked_variables` is a tuple of tensors with one row per point. They are
evaluated at every node but not integrated, and come back in
`result.tracked_variables`, detached and aligned row by row with `result.nodes`
and `result.y`. In the per-batch result passed to `loss_fxn` they keep one row
per subinterval (shape `[N, C, ...]`).
