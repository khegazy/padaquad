# Bug hunt: memory state, growing batch size and backward restart (branch `change_memory`)

Date: 2026-11-18. Reviewer: an adversarial bug-hunt session. The tests are in
`tests/test_memory_state_adversarial.py`. Tests marked
`xfail(strict=True)` (expected to fail) fail because of the bug named in their
`reason`. The other tests pin behavior that was checked and found correct.
Line numbers are from the working tree on 2026-11-18, after the change that makes
`integrate` return `(result, state)`. Every test calls `integrate` through one
helper, `run`, which accepts both the old and the new return type.

Terms used below:

- A **batch** is one call of the integrand `f` on many points.
- A **restart** happens when the backward pass of a batch runs out of memory.
  `integrate` puts the gradients back and calls itself again with a smaller
  batch size.
- A **leaf** is a tensor created directly by the user, not computed from other
  tensors, that receives a gradient (`.grad`) in a backward pass.
- A **frame** is the set of local variables of a running Python function call.

## Summary

1. **A restart keeps the failed batch in memory.** The restart runs inside the failed call's frame, which still holds that batch's autograd graph. Each nested restart adds one more. This works against the purpose of a restart after an out-of-memory error.
2. **Every batch's leaf tensors stay alive until the call ends.** This covers padaquad's own mesh copy when a bound requires grad, and any leaf the integrand creates. The result is a memory leak on the device that grows with the number of batches. It is a regression: on `main`, only one batch's leaves are alive at a time.
3. **A restart silently drops the gradient of a mesh that requires grad** (passed as `mesh=` or `state.mesh`, without bounds). `mesh.grad` ends as None instead of the gradient.
4. **The memory check before a backward pass (GPU only) can be switched off by a stale value.** A backward pass after a batch whose output needs no gradient computes the ratio from an earlier batch's forward memory. The next batch then passes the check unchecked.
5. **A `max_batch` that is a NumPy integer or a float now makes `integrate` raise** `ValueError` at its very end, after all the work. On `main` the same call worked.
6. **An out-of-memory retry shrinks the planned size, not the size that failed.** The same failing call is repeated several times; in the test, 3 failed calls instead of 1.
7. **A small batch size that has not converged fails with `take_gradient=True`.** It raises an `AssertionError` ("Not enough free memory") instead of growing.
8. **(Low severity, documented behavior.)** When a restart is refused because a batch cannot hold one subinterval, the gradients are not put back, although all of them were saved.

## Bug 1: the restart keeps the failed batch alive

**What goes wrong.** After a backward out-of-memory error, `integrate` calls
`return self.integrate(...)` from inside the loop's frame
(`padaquad/quadrature/base.py:1297`). Before that call it deletes `loss`,
`intermediate_results`, `method_output`, `nodes` and `y_step_eval` (line 1269).
Other locals still reference the same objects:

- `kept_output`, `kept_y` and `kept_nodes` are the same objects as `method_output`, `y_step_eval` and `nodes` (line 1164).
- `all_mesh_quadratures` and `cum_mesh_quadratures` are not detached.
- `tracked_step_eval`, `mesh_eval` and `saved` are also still alive. `saved` holds a copy of every gradient the walk found.

The failed batch's autograd graph, including the activations it saved for its
backward pass, therefore stays alive during the whole restarted integral. Each
nested restart keeps one more frame. In the test, after three nested restarts,
1, then 2, then 3 failed batches were still alive. The restart exists because
memory ran out, so holding this memory makes the next failure more likely. It
can turn a recoverable error into a `BackwardOutOfMemoryError`.

**Test.** `test_restart_releases_the_failed_batch` (xfail). It keeps a weak
reference to the tensor that the failed batch saved for backward. It then
checks that this tensor is freed when the restarted call first evaluates `f`.

**Proposed fix.** Do not recurse from inside the attempt. Move the body of
`integrate` into a private method that returns either the result or the restart
state. `integrate` then loops: `while isinstance(out := self._attempt(...),
IntegrationState): ...`. When the attempt returns, its frame is released
completely. The alternative, a longer `del` list, is easy to get wrong when
locals are added later.

## Bug 2: the saved-gradient dictionary keeps every batch's leaves alive

**What goes wrong.** `_save_grads` (`padaquad/quadrature/memory.py:1416`,
store at line 1459) puts `(leaf, grad)` into `saved` for every leaf it meets,
and `saved` lives for the whole call. Two kinds of leaf are new in every batch:

- `mesh_eval` (`base.py:1003`) is a detached copy of the whole mesh, shape [M, T]. It is made per batch whenever a bound or the mesh requires grad, and its `.grad` is filled by the backward pass.
- Any leaf the integrand creates per batch is also kept, for example `x = t.detach().requires_grad_()` to differentiate with respect to the input, as physics-informed losses do.

All of them, with their gradients, stay on the device until `integrate`
returns. Measured on the CPU with 21 batches: on `main` at most 1 earlier leaf
is alive at each call of `f`; on the branch all earlier ones are (15 to 19 at
the last calls). The memory grows with the number of batches. For the mesh
copies it is 2 x 8 bytes x M x T per batch. As an estimate (reasoning, not
measured): 100 batches on a mesh of 10^6 points hold about 1.6 GB. With
`result_device="cpu"` the record is moved off the GPU, but these leaves are not.

**Test.** `test_saved_gradients_do_not_keep_every_batch_alive[made_in_f]` and
`[mesh_copy]` (both xfail).

**Proposed fix.** Store only what a restore needs, and hold it weakly. Use
`saved[id(leaf)] = (weakref.ref(leaf), grad_clone)`, or use
`torch.utils.weak.WeakIdKeyDictionary`. When restoring, skip leaves that have
died: nothing outside the call can see their gradient. Leaves given through
`parameters=` are held by the caller anyway.

## Bug 3: a restart drops the gradient of a mesh that requires grad

**What goes wrong.** Without bounds, a mesh that requires grad (passed as `mesh=`
or as `state.mesh`) gets the integral's gradient at its end points, by the
same route as the bounds (`collect_mesh_grad`, `base.py:991`). The restart
uses `start_mesh = mesh.detach().clone()` (line 954) and passes it as
`state.mesh` with no bounds (line 1271). The first attempt's `mesh_grad` is
thrown away. As a result, the caller's mesh never receives a gradient:
`mesh.grad` is None. An uninterrupted run gives `f(1) = sin(40) + 1` at the
last point. Bounds that require grad are not affected, because they are passed
again and added to the mesh.

**Test.** `test_restart_keeps_the_gradient_of_a_mesh_that_requires_grad[mesh]`
and `[state]` (both xfail).

**Proposed fix.** For the restart, pass the mesh as the caller gave it, cut to
the bounds but not detached. For example, keep `start_mesh = mesh` (the output
of `_setup_integral_bounds`, before any split) instead of a detached clone. A
second option is to refuse the restart (raise `BackwardOutOfMemoryError`) when
the starting mesh requires grad. Either way, document whether a mesh that
requires grad is supported. The docstring of `mesh=` does not say.

## Bug 4: a stale forward-memory value disables the check before a backward pass (GPU)

**What goes wrong.** The check in `_evaluate_f_on_full_nodes` predicts the
backward memory as `_backward_ratio * _forward_bytes`, where:

- `_forward_bytes` is the forward peak memory of the batch, written only when `f`'s output requires grad (`base.py:2091`).
- `_backward_ratio` is set after each backward pass from the current `_forward_bytes`, whenever that is above 0 (`base.py:1238`).

`_forward_bytes` is never reset, neither per batch nor per call (only
`_backward_ratio` is reset, at line 927). A backward pass after a batch whose
output needs no gradient therefore divides by the forward peak of an earlier
batch, possibly from an earlier call. This happens, for example, when the loss
has a learnable term while `f` does not need a gradient on that part of the
interval.

Measured on an A100: an earlier call left `_forward_bytes` at 18.9 MB. Two
left batches whose outputs needed no gradient set the ratio to 5.4e-5. The
first batch with t > 0.5 then needed 4 bytes, passed the check with 1 kB
usable, and its backward pass ran unchecked. On a fresh solver the same batch
is dropped and evaluated again, smaller. The opposite case can also happen: a
stale small `_forward_bytes` gives a large ratio, which shrinks batches that
did not need to shrink.

**Tests.**
- `test_backward_ratio_is_not_computed_from_a_stale_forward_peak` (xfail, `needs_cuda`).
- Its control, `test_backward_check_control_on_a_fresh_solver`, which passes.

**Proposed fix.** Set `self._forward_bytes = 0.0` at the top of
`_evaluate_f_on_full_nodes`, before the loop. A batch whose output needs no
gradient then leaves 0, and the ratio is not updated. This is one line.

## Bug 5: `max_batch` given as a NumPy integer or a float makes `integrate` raise

**What goes wrong.** `integrate` stores `max_batch` as given (`base.py:866-870`)
and never checks or converts it. `IntegrationState.__post_init__`
(`padaquad/results.py:64`) accepts only a Python `int`. With the new return
type, `integrate` builds the state at its end (`base.py:1419`), so
`integrate(..., max_batch=np.int64(5000))` does all the work and then raises
`ValueError: IntegrationState.max_batch must be an int.` On `main` the same call
returned normally (checked). NumPy integers are common, for example a size
read from a config file through NumPy or computed with `np.floor`. A float
`5000.0` fails the same way.

**Test.** `test_result_state_after_a_numpy_or_float_max_batch[...]` (xfail, two
cases).

**Proposed fix.** Validate at the boundary. In `integrate`, and in the
constructor for its `max_batch`, use `max_batch = operator.index(max_batch)`,
which accepts NumPy integers and rejects floats with a clear `TypeError`, and
raise `ValueError` for a value below 1. The same check also catches 0 and
negative values. Today 0 is "rescued" to 1 with a warning, and a negative value
ends in an `AssertionError`.

## Bug 6: an out-of-memory retry shrinks the planned size instead of the failed call

**What goes wrong.** In `_evaluate_f_in_batches`, after a call runs out of
memory, the new size is `_shrink_batch(max_batch)` (`base.py:2198`). Here
`max_batch` is the planned size, which grows to 10 times the batch size inside
one pass. The call that failed can be smaller, because only the remaining
points are evaluated. The retry then evaluates the same points again, fails
again, and keeps doing so until 0.75^k x `max_batch` drops below the failed
size.

In the test, a state of 100 points that has not converged meets a pass of
5,750 points with a limit of 4,500. The calls are 1,000, then 4,750 (fails),
4,750 (fails), 4,750 (fails), 4,219 and 531. Three failed calls where one would
do. The warning also says "Reducing max_batch from 10000 to 7500" for a call of
4,750 points. On a GPU each failed call is a full forward pass plus an
`empty_cache()`. With a large ratio between the planned and the remaining size,
there can be tens of failed calls (reasoning: log(ratio) / log(4/3)). The
integral is still right.

**Test.** `test_out_of_memory_retry_shrinks_the_call_that_failed` (xfail).

**Proposed fix.** Shrink the size of the call that failed:
`n_failed = min(max_batch, num_nodes_to_eval_left)`, which is the length of
the slice, and set `self._batch = _shrink_batch(n_failed)`.

## Bug 7: a small unconverged batch size fails with `take_gradient=True`

**What goes wrong.** With `take_gradient=True` each call needs whole
subintervals (23 points for gk21). The dispatcher computes `max_mesh_steps =
_batch_for(n) // C` (`base.py:1984`). For a state with batch size 2 that has
not converged, this is `min(n, 20) // 23 = 0`. `_evaluate_f_on_full_nodes`
then fails its assertion (line 2060), "Not enough free memory ...", although
nothing ran out of memory and the size was allowed to grow. The same state
works with `take_gradient=False`.

Such states come from users who build an `IntegrationState` by hand. One case
could also come from a `take_gradient=False` call with fewer than 10 points in
its first pass (reasoning; not reproduced end to end).

**Test.** `test_small_unconverged_state_grows_to_one_subinterval_with_gradients`
(xfail).

**Proposed fix.** When the size has not converged, let the dispatcher take at
least one subinterval:
`max_mesh_steps = max(max_batch // C, 0 if self._batch_converged else 1)`.
Also replace the assertion by a `ValueError` that names `max_batch` and `C`.

## Bug 8 (low severity): a refused restart does not put back the gradients

**What goes wrong.** When the restart is refused because a smaller batch cannot
hold one subinterval (`restart.max_batch < C`, `base.py:1278`), or because
`torch.distributed` runs more than one process, the error is raised
(line 1288) before the gradients are restored (line 1289). The error message
asks the user to reset them, for example with `optimizer.zero_grad()`. That
also erases gradients the user had accumulated before the call, such as an
initial-condition loss backpropagated earlier, or gradient accumulation over
several integrals. In these two cases the walk saved every gradient, so
padaquad could put them back. Only the "custom autograd function" and the
"post-accumulate hook" cases really cannot be undone.

This matches the current documentation. It is reported as a gap, not as a
contradiction of the docs.

**Test.** `test_refused_restart_puts_back_gradients_it_can` (xfail).

**Proposed fix.** Restore the saved gradients before raising whenever `unsafe`
is None, and say in the message that they were restored, so no reset is needed.

## Checked and found correct

- **Restart arguments.** The restart repeats the call with every argument: the constructor's integrand (`f=None`) and `y0`, `f_args`, `loss_fxn`, `error_integral_reference`, and a bound that requires grad. Integral, loss and both gradients equal an uninterrupted run's to 1e-10 (`test_restart_keeps_every_other_argument`).
- **`conserve_memory=True` and the default path.** Growth inside one pass combined with out-of-memory retries keeps the values aligned: the integral is right, and every value and tracked variable belongs to its node. Five fixed cases are in the test file (`test_conserve_memory_growth_and_retries_keep_values_aligned`). 300 random cases (limit, starting size, converged or not, mesh, method gk21/dopri5/cc9, both paths) were run on the compute node, with 0 failures.
- **Test calls.** The test calls work with the variable-sampling solver (`interpolatory3_variable`), in float32 and float64, with `f_args`, and with an integrand that returns tracked variables (`test_test_calls_with_both_solvers_and_dtypes`).
- **The graph walk.** It deduplicates graph nodes, so a graph made of 40 nested "diamonds" (each value used twice) is walked in under 1 ms. PyTorch keeps one Python object per graph node, so the `seen` set works.
- **Loop termination.** The `take_gradient=True` forward retry always ends: each retry lowers the number of subintervals by at least one, and with one subinterval left it raises the original error. The check cannot divide by zero: when the predicted need is 0 it passes.
- **Restart of a `state=` call.** The restart replaces `state`, `mesh` and `max_batch` correctly, and the constructor's `max_batch` does not override the restart's size. The restart reuses the same starting mesh, so the random number generator is not consumed again.
- **The existing tests.** The author's `tests/test_memory_state.py` GPU test passes on an A100. That file had one failure on the node, an `AttributeError` on `.state`, because it was being ported to the new return type at the time.

## What I could not check

- **Real GPU out-of-memory errors.** All out-of-memory errors were simulated. I did not check how the caching allocator behaves after a real out-of-memory error in a backward pass, in particular whether memory freed after Bug 1 is fixed is really returned in time.
- **Running out of memory on the CPU.** I did not check that the growing batch size gets the process killed on a CPU when memory runs out. That would take down the node. It is documented as a limitation (`docs/user_guide/devices_memory.md`, "Growth needs a clean failure").
- **Multiple processes.** I did not check runs with more than one `torch.distributed` process.
- **Hooks with side effects.** Hooks registered with `tensor.register_hook` on parameters run again on a restart, and the walk does not detect them. Only post-accumulate hooks are detected. A hook that has side effects (counters, moving averages) therefore repeats them. This is not tested. The documentation mentions repeated side effects of `f` and `loss_fxn`, but not of hooks.
- **Nested calls on one solver.** An integrand that calls `integrate` on the same solver object overwrites the outer call's `_batch`, `_batch_converged` and `_backward_ratio`. `main` has the same class of problem through `f_mem_coeffs`, so this is not new; not tested.
- **Cost of the graph walk.** I did not measure the walk on large graphs, such as an integrand with a time-stepping loop of 10^5 operations. Each batch walks the whole graph in Python.
- **Stale docstring.** The docstring of `get_max_batch` (`memory.py:407`) and `_cap_max_batch` (`memory.py:364`) still say the out-of-memory cap lasts "until the next `integrate` call starts". `integrate` no longer touches `_oom_max_batch`.
