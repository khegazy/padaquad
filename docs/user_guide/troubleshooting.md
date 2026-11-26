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
  is accepted against the running integral; without `enforce_convergence`
  (the default with `take_gradient=True`) it is not checked again when the
  total grows, and a fixed `error_integral_reference` is never replaced by the
  final integral ({ref}`ug-error-reference`). Use `take_gradient=False`, with
  which `enforce_convergence` is on by default, or pass an
  `error_integral_reference` no larger than |integral|.
- **A non-finite value** with `runtime_checks=False`. Refining cannot
  remove it; pass `runtime_checks=True` to raise where it first appears.
- **A subinterval too narrow to halve.** When the ends of a failing
  subinterval are neighboring floating-point numbers, its midpoint rounds onto
  one of them, so it cannot be split; it is accepted as it is, like a
  subinterval at `max_adaptive_splits`. This happens at a jump of the
  integrand when the tolerance is at rounding level, and near a singularity
  inside the interval, such as that of $1/(t - 0.5)^2$ on $[0, 1]$. Near
  such a point $t - 0.5$ keeps only a few correct digits, so the integrand's
  own rounding error can fail the tolerance on every subinterval close to it,
  and the refinement can go on for a long time and use much memory before
  each of them is that narrow: pass `max_adaptive_splits` to stop it sooner.

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
  it is at most `max_adaptive_splits`. A split into four pieces
  (`split_rule="accelerate"`, {ref}`ug-four-pieces`) counts as two halvings.

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

PadaQuad shrinks the batch and retries when an evaluation runs out of memory,
and with `take_gradient=True` restarts the integral when a backward pass does
({ref}`ug-memory`). If a single evaluation of the integrand does not fit, it
raises `torch.OutOfMemoryError`. It raises `BackwardOutOfMemoryError` when a
backward pass ran out of memory and the integrand uses a custom autograd
function: pass `parameters=` (the tensors, or their optimizer) so PadaQuad can
restart, or reset the gradients and call again with `state=error.state`. It
raises it also when the backward pass of a single subinterval does not fit (a
batch holds at least one subinterval with `take_gradient=True`): free memory on
the device or choose a method with fewer points per subinterval. Else
pass a smaller `max_batch`, use `take_gradient=True` (which does not keep the
whole graph), or, with `take_gradient=False` and an integrand with many outputs,
set `trade_speed_for_memory=True` or `record_evaluations=False`
({ref}`ug-trade-speed-for-memory`).

## A call is slower than expected, or its time varies

On the CPU, and for the host's share of a GPU solver's work, the number of
threads PyTorch uses changes the time of a call: on one 64-core node, PyTorch's
default of 64 threads took up to 34 % longer than 4 to 8 threads on as many
cores, and its slowest call up to 2.5 times its median. Set the thread count
and the CPUs the process gets ({ref}`ug-cpu-threads`).

## The error estimate seems unreliable

`integral_error` is the sum of the subintervals' signed error estimates, an
estimate of the embedded method's error, usually much larger than the true
error ({ref}`ug-error-control`). When the re-check (`enforce_convergence`)
removes an accepted subinterval, its signed estimate is subtracted, so
`integral_error` stays the sum over the returned subintervals. `fehlberg2`'s estimate ignores curvature and can be far too small
({doc}`methods`).

## Renamed and removed arguments

- `reuse_mesh` was renamed `same_f` (1.3.0), and `same_f` was removed
  (2.0.0): pass `state=state`, the state the call to start from returned
  ({ref}`ug-reuse`).
- The constructor's `mesh_init` and `mesh_final` were removed (2.0.0):
  pass the bounds, or a mesh, to every `integrate` call.
- `conserve_memory` was removed (2.0.0); `trade_speed_for_memory=True` takes
  its place. It does not change the mesh: every pass is judged whole, and the
  option only changes when the integrand values are integrated and moved off
  the solver's device ({ref}`ug-trade-speed-for-memory`).
- `max_path_change` was removed; use `max_adaptive_splits` to limit refinement
  and read the split statistics above.
- Solver options are keyword-only, and `integrate()` accepts every option of
  the solver constructor and of its `integrate` method.
