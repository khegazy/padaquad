(ug-reuse)=
# Reusing the mesh and batch size

```{include} ../../README.md
:start-after: "<!-- include-start: reuse -->"
:end-before: "<!-- include-end: reuse -->"
```

## How a call chooses its interval

| The call passes | Interval integrated | Starting mesh |
|---|---|---|
| `mesh` (or a `state` with a mesh) only | `mesh[0]` to `mesh[-1]` | `mesh`, as given |
| `mesh` and bounds inside it | the bounds; a missing one is the mesh's end | `mesh` cut to the bounds: its points between them, plus the bounds |
| `mesh` and a bound outside it | the bounds; a missing one is the mesh's end | `mesh` extended: all its points between the bounds, plus the bounds; the integration refines the new subinterval |
| both bounds, no `mesh` | the bounds | a fresh mesh (`N_init_steps`, `random_initial_mesh`) |
| one bound or none, no `mesh` (also with a `state` whose mesh is `None`) | — | `ValueError` |
| both `mesh` and `state` | — | `ValueError`; use `dataclasses.replace(state, mesh=m)` |

- Every given bound becomes an end point of the starting mesh, so its
  gradient is computed as for any mesh end, also when it equals an end of a
  state's mesh (which is detached from autograd).
- A bound within a few rounding steps of its mesh end (4 machine epsilons of
  the coarser of the bound's and the mesh's dtypes, relative to that end)
  replaces that end, so a bound that differs from `mesh[-1]` only by float32
  rounding does not add a near-zero-width subinterval. For the same reason,
  the mesh points within rounding of a bound that cuts or extends the mesh
  are dropped.
- A given mesh must have shape `[M, T]` with at least two increasing points.
  Bounds are tensors of shape `[T]` on every call, with or without a mesh
  (`torch.tensor([0.0])`, not `torch.tensor(0.0)`).
- The given mesh's dtype is the call's dtype (see {doc}`devices_memory`), so bounds
  of another dtype are converted to it.

## Extending a mesh

To integrate further, for example when the training interval grows, pass the
new bound with the mesh: a bound outside the mesh becomes its new end point,
and the integration refines the new subinterval like any other.

<!-- not-tested -->
```python
result, state = solver.integrate(
    f, state=state, mesh_final=torch.tensor([2.0], dtype=torch.float64)
)
```

## Several integrals per step

Keep one result per integral, for example one per loss term or per interval,
and pass each its own `state`.

## Checkpoints

`dataclasses.asdict(state)` is a dictionary of a tensor (or `None`), an int
and a bool, so it can be saved with a model and loaded (also with `torch.load`'s
default `weights_only=True`) to resume training:

<!-- not-tested -->
```python
import dataclasses
from padaquad import IntegrationState

torch.save(dataclasses.asdict(state), path)
state = IntegrationState(**torch.load(path, map_location=device))
result, state = solver.integrate(f, state=state)
```

## The batch size of a state

- `state.max_batch` is the largest number of points per call of the
  integrand known to fit: the `max_batch` the call was given, else the
  largest batch that ran.
- `state.max_batch_converged` is True when it will not grow again: the size
  was given (`max_batch=`), or a call ran out of memory and shrank it. While
  it is False, a call that needs larger batches tries 10 times the batch
  size.
- A size that shrank never grows back, also when the memory shortage was
  temporary. To let it grow again, pass
  `state=dataclasses.replace(state, max_batch_converged=False)`.
- A batch size depends on the integrand, `take_gradient`, the dtype and the
  device. A state reused with another of these still works: an out-of-memory
  error shrinks the batch until it fits, at the cost of the failed calls.
- `max_batch=` given with a state replaces its batch size, which is then
  fixed. `solver.get_max_batch(f=f, t=t, ...)` measures a batch size before
  any integral (it can be slow; measure once and pass the value as
  `max_batch`). `solver.get_max_batch()` without `f` returns the
  constructor's `max_batch`, else the value of the last measurement with `f`.

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
