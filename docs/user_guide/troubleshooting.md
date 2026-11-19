# Troubleshooting

(ug-convergence)=
## `converged` is False

`result.converged` is `True` when every accepted subinterval meets its
tolerance judged against the final integral (in cumulative mode, against the
final running integral through it). When it is `False`, a warning with the size
of the error is logged. Common causes:

- **`max_adaptive_splits` stopped the refinement.** It caps how many times a
  subinterval may be halved since the starting mesh; at the cap a failing
  subinterval is accepted. Raise the cap, or use a higher-degree method.
- **The tolerance changed after acceptance.** In absolute mode a subinterval
  is accepted against the running integral; with `take_gradient=True` it is
  not checked again when the total grows ({ref}`ug-error-reference`). Pass
  `error_integral_reference`, or use `take_gradient=False`, which re-checks.
- **A non-finite value** with `error_on_nonfinite=False`.

`result.error_ratios` holds each subinterval's ratio when it was accepted, so
every entry can be below 1 while `converged` is `False`. With
`error_norm="failure_fraction"` it holds failure fractions; compare them with
`mesh_failure_tolerance`.

## Split statistics

- `n_start_steps` is the number of subintervals in the starting mesh.
- `n_start_steps_split` counts those split at least once (splits made to build
  the state's mesh are not counted), and `percent_start_steps_split` gives it as
  a percentage. A small value means the starting mesh was nearly fine enough,
  which is what a warm start should achieve.
- `max_split_depth` is the largest number of times any subinterval was halved;
  it is at most `max_adaptive_splits`.

With `max_adaptive_splits=0` nothing is split, so `n_start_steps_split` is 0
even when subintervals fail; read `converged` instead.

## `get_max_batch(f=...)` is slow and logs warnings

That is the memory benchmark ({ref}`ug-memory`); `integrate` never runs it.
Measure once and pass the value as `max_batch`.

## A call logs "Caught OOM" or "was dropped"

The batch size grew until the integrand ran out of memory, then shrank
({ref}`ug-memory`). This happens once per state: pass the state the call
returned to the next call, which starts from the size that fit.

## Out of memory

padaquad shrinks the batch and retries when an evaluation runs out of memory,
and with `take_gradient=True` restarts the integral when a backward pass does
({ref}`ug-memory`). If a single evaluation of the integrand does not fit, it
raises `torch.OutOfMemoryError`. It raises `BackwardOutOfMemoryError` when a
backward pass ran out of memory and the integrand uses a custom autograd
function: pass `parameters=` (the tensors, or their optimizer) so padaquad can
restart, or reset the gradients and call again with `state=error.state`. Else
pass a smaller `max_batch`, use `take_gradient=True` (which does not keep the
whole graph), or set `conserve_memory=True`.

## The error estimate seems unreliable

`integral_error` is the sum of the subintervals' signed error estimates, an
estimate of the embedded method's error, usually much larger than the true
error ({ref}`ug-error-control`). With `take_gradient=False`, when the re-check
removes an accepted subinterval, the absolute value of its estimate is
subtracted from the signed sum, so `integral_error` is then approximate.
`fehlberg2`'s estimate ignores curvature and can be far too small
({doc}`methods`).

## Renamed and removed arguments

- `reuse_mesh` was renamed `same_f` (1.3.0), and `same_f` was removed
  (2.0.0): pass `state=state`, the state the call to start from returned
  ({ref}`ug-reuse`).
- The constructor's `mesh_init` and `mesh_final` were removed (2.0.0):
  pass the bounds, or a mesh, to every `integrate` call.
- `max_path_change` was removed; use `max_adaptive_splits` to limit refinement
  and read the split statistics above.
- Solver options are keyword-only, and `integrate()` accepts every option of
  the solver constructor and of its `integrate` method.
