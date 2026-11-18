# Break-and-speed review: memory state, growing batch size, backward restart (branch `change_memory`)

Date: 2026-11-18. Reviewer: an adversarial break-and-speed session. Code at
commit `d0bcbdc`; line numbers refer to it. Tests:
`tests/test_memory_state_break.py` (tests marked `xfail(strict=True)` fail
because of the bug in their `reason`; the others pin behavior checked and found
correct). Scripts and raw output: `/pscratch/sd/k/khegazy/padaquad_speed/`.
Hardware: one Perlmutter GPU node (NVIDIA A100-SXM4-80GB, AMD EPYC 7763), torch
2.8.0+cu128. The eight bugs of the first hunt are not repeated.

Terms: a *batch* is one call of the integrand `f`; an *out-of-memory episode*
is everything from the first failed call of `f` to the first call that fits; a
*restart* is `integrate` starting over after an out-of-memory error in a
backward pass; the *cache* is the GPU memory PyTorch's caching allocator keeps
after tensors are freed.

## Summary

Bugs, most severe first:

1. An out-of-memory error in `loss_fxn` (after `f`, before the backward pass) is not handled: a plain error, and the earlier batches' gradients stay in `.grad` (reproduced with a real CUDA error).
2. The saved copies of `.grad` sit in a reference cycle and outlive the call until Python's cyclic garbage collector runs: 1 to 3 extra gradient copies on the GPU between calls (measured).
3. An integrand that calls `integrate` on the same solver (a double integral) overwrites the outer call's batch size: `max_batch=50` was ignored (calls of 2,530 points). A regression from `main`.
4. `max_batch` is converted with `int()`: `True`, `2.7` and `"100"` are accepted silently; `-5` ends in a bare `AssertionError` after `f` ran.
5. A `torch.compile`d integrand counts as a "custom autograd function", so a backward out-of-memory error raises instead of restarting, although the walk reaches every parameter.

Speed-ups, largest first (A100 80 GB, medians):

1. Do not empty the cache, and do not sleep, after an out-of-memory error in `f`: an episode costs 1.4 to 6.0 s less (6.05 to 4.64 s, 7.96 to 5.47 s, 16.04 to 10.07 s); a chain of 5 restarts 0.77 to 0.49 s.
2. Drop the test calls for large starting meshes: cold calls 10 to 32 % faster (0.81 to 0.55 s); with small meshes, the first pass should start at its own size instead.
3. Shrink by a geometric bisection after a failure: 3 failed calls become 1, a further 0.1 to 0.8 s.
4. Read the GPU memory counters less often: about 0.2 to 0.4 ms per gradient batch (3 to 7 % of small warm gradient calls).

No change needed: the graph walk (4 to 7 % of the backward pass), host-device synchronizations, resuming instead of restarting.

## Bugs

### Bug 1: an out-of-memory error in `loss_fxn` leaves partial gradients

**What goes wrong.** The batch size grows until `f` fails. Whatever runs after
`f` gets only the memory that `f` left. `loss = loss_fxn(intermediate_results)`
(`padaquad/quadrature/base.py:1215`) and the quadrature bookkeeping before it
(`_calculate_integral` at line 1022, the error ratios, the mesh update) are
outside every `try`. An out-of-memory error there propagates as a plain
`torch.OutOfMemoryError`. Nothing is retried, nothing is restarted, and the
gradients of the batches that already ran their backward pass stay in `.grad`.
A caller that catches the error and calls again, for example with a smaller
`max_batch`, counts those batches twice. For a backward-pass error the same
situation is handled: the gradients are put back and the integral restarts, or
`BackwardOutOfMemoryError` says to reset them. On `main` the batch was sized
at 0.75 of the usable memory divided by 2.1 times `f`'s memory, which left room
for `loss_fxn`. The backward check (line 2095 on) runs before `loss_fxn`, so it
does not see this memory.

**Reproduced on an A100.** `f = w sin(40 t)` is cheap, and `loss_fxn` penalizes
every integrand value through a [points, 4096] temporary. Under a 1 GiB cap the
batch grows from 2,300 to 23,000 points, and the loss of the 23,000-point batch
does not fit: "CUDA out of memory. Tried to allocate 720.00 MiB" (23,000 x
4,096 x 8 bytes), with `w.grad = 0.00197` left from the first batch.

**Tests.** `test_out_of_memory_in_loss_fxn_puts_back_the_gradients` (CPU,
simulated) and `test_real_out_of_memory_in_loss_fxn_puts_back_the_gradients`
(`needs_cuda`, real error). Both xfail.

**Smallest fix.** Move the loss into the `try` that already catches the
backward error, so both take the restart path (`base.py:1215-1236`):

```diff
-                loss = loss_fxn(intermediate_results)
-                if take_gradient and loss.requires_grad:
-                    unsafe = ...
-                    ...
-                    try:
-                        loss.backward()
-                    except (RuntimeError, MemoryError) as e:
-                        if not _is_out_of_memory(e):
-                            raise
-                        failed_batch = batch_size
+                try:
+                    loss = loss_fxn(intermediate_results)
+                    if take_gradient and loss.requires_grad:
+                        unsafe = ...            # unchanged
+                        ...                     # unchanged peak-memory reset
+                        loss.backward()
+                except (RuntimeError, MemoryError) as e:
+                    if not (take_gradient and _is_out_of_memory(e)):
+                        raise
+                    failed_batch = batch_size
                 if failed_batch is not None:
                     break
```

The bookkeeping between `f` and the loss (`_calculate_integral`, error ratios,
mesh update) uses a few times the batch's output. That is usually small next to
`f`'s own memory, but the same `try` could cover it. A restart starts from the
starting mesh, so the loop's state at the failure does not matter. Document it
in the user guide's "With `take_gradient=True`" paragraph.

### Bug 2: copies of `.grad` outlive the call (reference cycle)

**What goes wrong.** `_save_grad` (`padaquad/quadrature/memory.py:107-110`)
stores `(weakref.ref(leaf, lambda _: saved.pop(key, None)), grad_copy)` in
`saved`. A weak reference holds its callback, and the callback holds `saved`.
So `saved` → tuple → weak reference → callback → `saved` is a cycle. When
`integrate` returns, reference counting cannot free `saved`, and every copy of
`.grad` it holds stays alive until a run of the cyclic garbage collector. A
copy exists whenever a parameter has a gradient before the call: gradient
accumulation, `zero_grad(set_to_none=False)`, or a loss term backpropagated
first. Each copy is as large as the parameter.

**Measured on an A100.** The test model was `Linear(4096, 4096)`, 64 MiB of
gradient, with gradients preset and 12 calls in a loop. The allocated memory
after each call was 208, 272, 208, 272, 208, 272, 336, 208, 272, 208, 272 and
336 MiB, against 144 MiB with the fix. That is 1 to 3 extra copies held between
calls. A model of 1 billion float32 parameters would hold 4 to 12 GB this way,
which the next optimizer step can need.

**Test.** `test_saved_gradient_copies_are_freed_when_integrate_returns` (xfail).
It turns off the cyclic collector during the call and checks, through a weak
reference, that the copy is freed when the call returns.

**Smallest fix.** Remove the callback, and detect a reused `id` on lookup:

```diff
 def _save_grad(leaf: torch.Tensor, saved: _SavedGrads) -> None:
-    key = id(leaf)
-    if key not in saved:
-        grad = None if leaf.grad is None else leaf.grad.clone()
-        saved[key] = (weakref.ref(leaf, lambda _: saved.pop(key, None)), grad)
+    entry = saved.get(id(leaf))
+    # New, or the id of a freed tensor reused by a new one.
+    if entry is None or entry[0]() is not leaf:
+        grad = None if leaf.grad is None else leaf.grad.clone()
+        saved[id(leaf)] = (weakref.ref(leaf), grad)
```

Entries of freed tensors then stay in the dictionary until the call ends. They
hold a copy only if the tensor had a gradient before the call. Tensors made per
batch (the mesh copy, leaves made in `f`) have none, so they hold `None`.

### Bug 3: a nested integral on the same solver replaces the outer batch size

**What goes wrong.** The batch size of a call lives in solver attributes
(`_batch`, `_batch_converged`; also `_total_mem_usage`, `_forward_bytes` and
`_backward_ratio`), set at the start of every call (`base.py:864-873`, `924-925`,
`952-959`). An integrand that calls `integrate` on the same solver (a double
integral) resets them in the middle of the outer call's loop. In the test, the
outer call has `max_batch=50`. After the first inner call, the outer call
evaluates `f` on 2,530 points and returns the inner call's state
(`max_batch=2070`, not converged). On `main` a given `max_batch` was a local
variable of the call (`force_max_batch`) and was respected. Two threads that
share one solver hit the same problem. The per-call `error_norm` and
`error_on_nonfinite` attributes have had this problem since before this branch.

**Test.** `test_nested_integral_on_the_same_solver_keeps_the_outer_max_batch`
(xfail).

**Smallest fix.** The restart wrapper already surrounds every call. Let it save
these attributes on entry and restore them on exit
(`memory.py:162-176`):

```diff
         call = signature.bind(*args, **kwargs).arguments
+        solver = call["self"]
+        outer = {k: getattr(solver, k) for k in ("_batch", "_batch_converged",
+                 "_total_mem_usage", "_forward_bytes", "_backward_ratio")}
         ...
         while True:
             try:
                 return integrate(**call)
             except _Restart as restart:
                 state = restart.args[0]
+            finally:
+                vars(solver).update(outer)
```

The returned state is built before the `finally` runs, and a restart sets the
attributes again from its state, so neither is affected. Threads stay unsafe;
document "one solver per thread".

### Bug 4: `max_batch` is not validated

**What goes wrong.** `self._batch = int(given)` (`base.py:868`) accepts `True`
(batch size 1), truncates `2.7` to 2 and parses `"100"`. A negative value is
stored, and `_evaluate_f_on_full_nodes` or `_evaluate_f_on_split_nodes` then
fails a bare `assert` after the output check has called `f`.
`IntegrationState` rejects a bool and a non-int (`results.py:64`), so the two
entry points disagree.

**Test.** `test_invalid_max_batch_is_rejected_before_any_work[-5, True, 2.7, 100]`
(xfail, four cases).

**Smallest fix.** At `base.py:867` and in the constructor:

```diff
-        if given is not None:
-            self._batch, self._batch_converged = int(given), True
+        if given is not None:
+            if isinstance(given, bool) or operator.index(given) < 1:
+                raise ValueError(f"max_batch must be an int >= 1; got {given!r}.")
+            self._batch, self._batch_converged = operator.index(given), True
```

`operator.index` accepts NumPy integers (the first hunt's bug 5) and raises
`TypeError` for floats and strings.

### Bug 5: a `torch.compile`d integrand never restarts

**What goes wrong.** `_save_grads` flags any `BackwardCFunction` node as a
custom autograd function that can hide parameters
(`memory.py:1555-1561`). A graph through `torch.compile` contains a
`CompiledFunctionBackward` node, which is a `BackwardCFunction` (checked, torch
2.8, backend `aot_eager`). The parameters are inputs of that node, and the walk
found their accumulate-gradient nodes under it. Still, without `parameters=` a
backward out-of-memory error raises `BackwardOutOfMemoryError` ("the integrand
uses a custom autograd function"). Nothing in the documentation tells a
`torch.compile` user that this applies to them: the docstring and the user
guide name activation checkpointing as the example.

**Tests.** `test_compiled_integrand_restarts_after_a_backward_out_of_memory_error`
(xfail). `test_restart_of_a_compiled_integrand_with_parameters` passes: with
`parameters=` the restart works and matches an uninterrupted run.

**Smallest fix.** Either document `torch.compile` under `parameters=`, or treat
compiled functions as transparent:

```diff
             elif not listed and isinstance(
                 node, torch.autograd.function.BackwardCFunction
-            ):
+            ) and type(node).__name__ != "CompiledFunctionBackward":
```

The second option rests on my reading of AOTAutograd, PyTorch's compiler front
end, which passes a module's parameters to the compiled function as inputs. I
checked this for one `nn.Linear` only. Whether a compiled region that contains
a reentrant checkpoint could hide parameters, I did not check. Until that is
checked, the documentation fix is the safe one.

### Minor findings (no test)

- **`parameters=` given one tensor.** A 1-D tensor is iterated into its rows: the call works, but it saves views and warns "The .grad attribute of a Tensor that is not a leaf". A 0-d tensor raises `TypeError: iteration over a 0-d tensor` (`memory.py:168`). Fix: `if isinstance(parameters, torch.Tensor): parameters = [parameters]`.
- **A converged batch size below one subinterval with gradients.** This happens with `max_batch=10`, or with a state from a `take_gradient=False` call that converged below 23 points. It fails the `assert` at `base.py:2068`, and under `python -O` it fails with "cannot reshape tensor of 0 elements". The same happens on `main`. A `ValueError` that names `max_batch` and the subinterval's node count would be clearer.
- **The ratio of the first backward pass is inflated (reasoning, not reproduced).** The first backward pass of a call with `.grad = None` allocates every parameter's gradient. That fixed cost enters the backward/forward ratio of the next batch's check. The next batch is at most about twice as large, because passes at most double, so the check shrinks it needlessly only when twice the gradient memory exceeds the usable memory. The shrink would be permanent, because a converged size never grows. I judge this narrow and did not reproduce it.

## Speed-ups

### Speed-up 1: keep PyTorch's cache, and drop the sleep, after an out-of-memory error

**What costs time.** `_release_memory` (`memory.py:659-676`) runs after every
failed call of `f` and before every restart. It runs `gc.collect()`,
`torch.cuda.empty_cache()` and `time.sleep(0.05)`. The sleep and the emptying
themselves are cheap: 0.05 s per failed call, and 0.03 to 0.38 s per episode. The expensive
part is their effect on the following calls of `f`:

- The first call after the cache was emptied runs about 1.26 s slower. A call of 2,070,000 points took 2.25 s after emptying and 0.99 s without it (3 runs each). The same call with a fixed batch, never failing, took 1.00 s.
- With the cache emptied, a call slightly too large runs for 1.5 to 2.1 s before it fails. With the cache kept, it fails within 3 to 21 ms. I did not establish why. A likely cause, not verified, is that taking tens of GB back from the driver is slow.

PyTorch's documentation of `empty_cache` (torch 2.8 docstring) says it "doesn't
increase the amount of GPU memory available for PyTorch". It frees memory only
for other programs and for allocations outside PyTorch's allocator. A retry
therefore gains nothing from it for an integrand that allocates through
PyTorch. The critique calls the sleep "added by accident" (its section 3, citing the brainstorm's section 2.3).

**Measurement** (`oom.py`). The test integrand is a 3-layer tanh network of
width 2048 without gradients, gk21 (23 nodes per subinterval), and one pass. The
whole call is timed, including the failed calls.

| Start | Pass (points) | Current | No sleep | Cache kept (gc only) | Cache kept + bisection |
|---|---|---|---|---|---|
| State of 69,000 points, not converged | 6.9 M | 6.05 s (1 failed call) | 5.98 s | 4.64 s (2 failed, the second in 8 ms) | 4.55 s (1 failed) |
| No state (test calls) | 6.9 M | 7.96 s (2 failed) | | 5.47 s (3 failed, two in under 21 ms) | 5.24 s (1 failed) |
| No state (test calls) | 11.5 M | 16.04 s (4 failed, 4.24 s in failed calls) | | 10.07 s (3 failed, 0.52 s) | 9.31 s (1 failed) |

The same 6.9 M-point integral with a fixed batch size and no failure takes
4.02 to 4.08 s (four batch sizes from 2.07 to 3.88 million points). Keeping the
cache therefore cuts the time over that fixed-batch run from about 2.0 s to
about 0.6 s (state start), and from about 3.9 s to about 1.4 s (cold start,
which includes about 0.55 s of test calls). For a chain of
5 restarts (the real backward error of the GPU test, 1 GiB cap), the median of
3 runs went from 0.77 s to 0.49 s, with the gradients bit-identical either way.

In the 11.5 M row, the current code shows one more cost. The test call of
4,218,750 points ran, but the first real call of the same size then failed
(after the cache had been emptied), which cost one more shrink.

**Minimal diff.**

```diff
-    def _release_memory(self) -> None:
+    def _release_memory(self, empty_cache: bool = False) -> None:
         gc.collect()
-        if self.device.type == "cuda":
+        if empty_cache and self.device.type == "cuda":
             torch.cuda.empty_cache()
-            time.sleep(0.05)
```

Pass `empty_cache=True` at the end of the benchmark (`memory.py:1087`, where
memory is measured). In the retry paths (`memory.py:1372`, `base.py:2093`,
`base.py:2206`, and the restart at `memory.py:175`), pass it only when the error
did not come from PyTorch's allocator (`not isinstance(e, torch.OutOfMemoryError)`,
for example a cuBLAS `ALLOC_FAILED` or a raw "CUDA error: out of memory"). Only
those allocations, made outside PyTorch, can use what the cache gives back.

**Risk.** Memory in PyTorch's cache is not available to other processes on the
GPU, or to libraries in `f` that allocate on their own (CuPy, custom CUDA). The
user guide already says the forward calls fill the memory. The second case is
covered by emptying the cache for errors from outside the allocator, as
described above. Correctness is unaffected: every run above returned the same
integral.

### Speed-up 2: test calls cost up to a third of a cold call with a large starting mesh

**What costs time.** `_first_batch` (`memory.py:1462-1511`) calls `f` on 10,
100, ... points, up to the largest power of 10 not above the first pass. Their
results are thrown away. These are the same sizes that growth inside the first
pass would reach anyway, and the growth calls do useful work. The total is up
to 1.11 times the largest test call, so up to about half of the first pass is
wasted.

**Measurement** (`compare.py`, A100, 7 runs, cold calls of a fresh solver with
`mesh=` given and no state). "No test calls" is the prototype
`_first_batch = (1, False)`, so the first pass grows from 10 points.

| Integrand, starting mesh | Current | Of which test calls | No test calls | `main` (3 runs) |
|---|---|---|---|---|
| 3x512 net, `take_gradient=False`, 100,000 subintervals | 0.520 s | 0.077 s | 0.409 s (−21 %) | 0.600 s |
| 6x2048 net, `take_gradient=False`, 10,000 subintervals | 0.812 s | 0.255 s | 0.553 s (−32 %) | 1.142 s |
| 6x2048 net, `take_gradient=True`, 10,000 subintervals | 1.862 s | 0.255 s | 1.669 s (−10 %) | 2.188 s |

With the default starting mesh (253 points), the test calls cost 0.3 to 1.4 ms
and nothing is gained. Starting from 10 points then makes small gradient calls
slower, because the first pass is split into batches of 23 and 230 points, each
with its own backward pass. Cold times (5 runs) went from 9.6 to 14.3 ms
(3x512) and from 12.1 to 16.5 ms (6x2048) on the GPU, and from 6.2 to 8.0 ms on
the CPU.

**Proposal.** Start the batch size at the first pass itself:
`self._batch, self._batch_converged = max(1, -(-n_first // _GROWTH)), False`
in place of the `_first_batch` call (`base.py:952-959`), so that the first call
covers the whole first pass. If that call runs out of memory, the existing
retry handles it. This variant was **not measured**. Subtracting the measured
test-call time gives an estimate of 0.44, 0.56 and 1.61 s for the three rows,
and no change for the default mesh. On a CPU, where a call too large can get the
process killed instead of raising, keep the test calls (the critique's F9
reason). This is the owner's decision (plan decision 11). The measurements say
that on a GPU the test calls protect nothing that the retry does not.

### Speed-up 3: a geometric bisection after a failed call

**What costs time.** After a failure `_shrink_batch` (`memory.py:67-84`)
multiplies by 0.75. When the failed call was 10 times a call that ran, it can
take up to 8 failed calls to come back under the limit. A call slightly too
large fails late (1.5 to 2.1 s each with the cache emptied).

**Change.** When a size `good` is known to have run in this call and the failed
size `n` is more than 1.5 times it, try `sqrt(good * n)`; otherwise 0.75 n.

```diff
-def _shrink_batch(n: int) -> int:
-    return max(1, int(min(round(_OOM_BACKOFF * n), n - 1)))
+def _shrink_batch(n: int, good: int = 0) -> int:
+    m = math.isqrt(good * n) if 0 < good and 1.5 * good < n else round(_OOM_BACKOFF * n)
+    return max(1, int(min(m, n - 1)))
```

`_evaluate_f_in_batches` keeps a local `good` (`self._batch` at entry while not
converged, raised after each call that ran) and passes it at `base.py:2209`.
`_first_batch` passes its last test call that ran.

**Measurement** (same table as speed-up 1). Failed calls went from 2 or 3 to 1. With
the cache kept, the time went from 5.47 to 5.24 s (cold, 6.9 M), from 10.07 to
9.31 s (cold, 11.5 M) and from 4.64 to 4.55 s (state). With the current
clean-up, it went from 7.96 to 6.77 s (cold). For a restart, the same rule
would carry the largest batch whose backward pass ran. I estimate that this
cuts the 5 restarts of the GPU test to 2; not measured.

**Risk.** It converges lower: for example 2.07 M points where 0.75 steps reach
3.49 M. For this integrand that cost nothing, because the fixed-batch runs at
2.07 M to 3.88 M points take 4.02 to 4.08 s: the GPU is saturated above about
1 M points. On a smaller GPU the converged size could fall below saturation.

### Speed-up 4: fewer reads of the GPU memory counters per gradient batch

**What costs time.** Each of `torch.cuda.memory_allocated`,
`max_memory_allocated` and `memory_reserved` builds the allocator's whole
statistics dictionary in Python: 62 to 74 µs each, idle, on the A100.
`_get_usable_memory` makes two of them plus `mem_get_info`, 165 µs in total.
Per gradient batch on a GPU the code makes six such reads
(`base.py:2081-2104`, `1227-1240`), about 0.46 ms. No read waits for queued
GPU work: with about 300 ms of matrix products queued, each returned in under
0.35 ms. A small warm gradient call (3x512 network, 243 points, 15 runs) takes
8.5 ms on the branch and 7.2 ms on `main`.

**Measurement** (`check_cost.py`, 15 runs). As an upper bound, the three reads
were served from the private `torch._C._cuda_memoryStats` (9.5 µs a read,
against 60.7 µs for `torch.cuda.memory_stats`). A warm gradient call on the
default mesh then took 7.25 ms instead of 7.48 ms (−3 %). On 22,001 points in
batches of 2,300 it took 70.3 ms instead of 75.9 ms (−7 %).

**Minimal change with the public API.** After `f`, make one
`stats = torch.cuda.memory_stats(self.device)` call. It gives the peak, the
allocated and the reserved bytes together. Pass its two values into
`_get_cuda_memory` instead of reading them again. That makes 4 reads per batch
instead of 6, about 0.13 ms less per batch (estimate). The gain is small; it
matters only for many small gradient calls.

### Checked and not worth changing

- **The graph walk** (`_save_grads`, `memory.py:1513-1563`; `walk.py`). The test integrand was a residual tanh network of width 16 applied in a Python loop.
  - GPU: with 1,006 graph nodes the walk took 0.93 ms against a backward pass of 25.2 ms (3.7 %). With 10,006 nodes it took 9.5 ms against 220.5 ms (4.3 %).
  - CPU: 256 nodes, 0.31 ms against 4.5 ms (7 %).
  - Whole calls with and without the walk differ by less than their noise (52.3 against 53.1 ms; 432 against 469 ms).
  - A walk that checks the node type before `getattr` is 30 % faster, about 1 % of the backward pass.
  
  The walk costs about 1 µs of Python per node, and the backward pass about 20 µs per node, so the walk cannot dominate.
- **Host-device synchronizations.** The memory queries do not synchronize (above). The per-batch synchronizations (`torch.any(mesh_trackers)`, `int(batches_left)`, the order assertion, the finite check) were all there before this branch. Each costs about 30 µs when the GPU is idle, and the host needs those sizes to go on.
- **Resuming instead of restarting.** In the GPU test, 5 restarts cost 0.77 s against 0.04 s for the integral; with the cache kept, 0.49 s. Most of that is the per-restart clean-up and the failing batches, not rerunning work. While the size grows, the batches before a failing batch hold at most a ninth of its points (sum of 10^k / 9), so in a cold call a resume saves little. Resuming would need to put the failed batch's accepted subintervals back in the queue, and to keep a copy of the gradients before every batch, not once per call. Speed-ups 1 and 3 reduce the cost of restarts with no new state.

## Branch against `main`

Cold calls on the branch avoid `main`'s memory benchmark (0.1 to 2.0 s per
call). Warm calls are equal, except small gradient calls (speed-up 4).
GPU: A100, medians of 5 runs; CPU: 32 cores of the EPYC 7763, medians of 5
runs. The default mesh has 253 points (gk21).

| Device | Integrand | `take_gradient` | Branch cold | `main` cold (benchmark) | Branch warm | `main` warm |
|---|---|---|---|---|---|---|
| GPU | sin(100 t) e^(−t), float64 | False | 0.025 s | 0.163 s (0.137) | 6.8 ms | 7.1 ms |
| GPU | 3x512 net | False | 0.008 s | 1.877 s (1.865) | 6.7 ms | 6.8 ms |
| GPU | 3x512 net | True | 0.010 s | 2.034 s (2.019) | 8.4 ms | 7.7 ms |
| GPU | 3x512 net, 15 runs | True | 0.010 s | 1.836 s | 8.5 ms | 7.2 ms |
| GPU | 3x512 net, 2,000 subintervals, 15 runs | True | 0.022 s | 0.129 s | 14.1 ms | 15.3 ms |
| GPU | 6x2048 net | False | 0.009 s | 0.374 s (0.365) | 6.6 ms | 7.1 ms |
| GPU | 6x2048 net | True | 0.012 s | 0.838 s (0.825) | 16.6 ms (noisy: 9.9 to 30.9) | 9.0 ms |
| CPU | sin(100 t) e^(−t) | False | 0.011 s | 1.669 s (1.656) | 16.0 ms (noisy: 2.7 to 38.1) | 2.9 ms |
| CPU | 3x512 net | False | 0.006 s | 1.361 s (1.355) | 4.0 ms | 4.1 ms |
| CPU | 3x512 net | True | 0.006 s | 1.346 s (1.339) | 5.3 ms | 5.0 ms |

The large-mesh rows are in the table of speed-up 2.

## All measurements

All on the node `nid008209` (A100-SXM4-80GB, EPYC 7763) unless marked "login",
torch 2.8.0+cu128. Every number below was checked against the raw output file
named.

| # | What | Result | Script, output |
|---|---|---|---|
| 1 | Memory queries, idle, median of 200 (µs) | reset_peak_memory_stats 1.6; memory_allocated 70.8; max_memory_allocated 74.0; memory_reserved 62.1; mem_get_info 12.8; _get_usable_memory 165.1; int(tensor.sum()) 28.9 (287,213 with queued work); none of the memory queries waited for queued work | `gpu_misc.py calls`, `g0.out` |
| 2 | Statistics reads (µs, median of 500) | memory_stats 60.7; torch._C._cuda_memoryStats 9.5; mem_get_info 11.9 | `check_cost.py`, `h2.out` |
| 3 | Warm gradient calls with cheaper reads (ms, 15 runs) | 133 points: 7.48 → 7.25 (no usable-memory query: 7.33); 22,001 points: 75.9 → 70.3 (65.4) | `check_cost.py`, `h2.out` |
| 4 | Saved-gradient cycle, MiB allocated after each of 12 calls | current: 208.3, 272.3, 208.3, 272.3, 208.3, 272.3, 336.3, 208.3, 272.3, 208.3, 272.3, 336.3 (144.3 after `gc.collect()`); fixed: 144.3 every time | `gpu_misc.py leak`, `g0.out` |
| 5 | Real OOM in `loss_fxn`, 1 GiB cap | `OutOfMemoryError` (720 MiB request, the 23,000-point batch), `w.grad` = 0.00197 left | `gpu_misc.py lossoom`, `g0.out` |
| 6 | Real backward OOM, 1 GiB cap, 3 runs per variant | 5 restarts, final batch 4,899, integral and gradient bit-equal to a run with that batch; current 0.781, 0.714, 0.773 s; cache kept 0.485, 0.446, 0.486 s; reference 0.041 to 0.043 s | `gpu_misc.py backoom`, `j0.out` |
| 7 | Real OOM in `f`, 1 GiB cap | `take_gradient=False`: final batch 4,094, integral bit-equal; `True`: 4,071, integral equal, gradient equal to rounding (not bit-equal: different batch partition) | `gpu_misc.py fwdoom`, `g0.out` |
| 8 | OOM episode, state start, 6.9 M points (s, 3 runs) | current 6.020, 6.054, 6.228; no sleep 6.034, 5.982, 5.972; cache kept 4.658, 4.638, 4.617; bisection 6.017, 5.877, 5.812; both 4.545, 4.516, 4.548 | `oom.py`, `g1.out` |
| 9 | OOM episode, cold, 6.9 M points (s, 2 runs) | current 7.962, 7.964; bisection 6.766, 6.769; cache kept 5.490, 5.451; both 5.257, 5.221 | `oom.py`, `g1.out`, `h1.out` |
| 10 | Same integral, fixed batch, no failure (s) | 3,881,250: 4.073; 3,493,125: 4.066; 2,070,000: 4.018; 2,626,785: 4.080 | `oom.py ... fixed`, `h1.out` |
| 11 | OOM episode, cold, 11.5 M points (s, 1 run) | current 16.040 (4 failed, 4.241 s); cache kept 10.069 (3 failed, 0.516 s); both 9.307 (1 failed, 0.517 s) | `oom.py`, `i1.out` |
| 12 | Call of 2,070,000 points after the episode (s) | with emptying 2.254, 2.252, 2.248; without 0.989, 0.989, 0.989; fixed batch 1.005 | `g1.out`, `h1.out` |
| 13 | Large starting meshes, cold (s, 7 runs) | see the table of speed-up 2 | `compare.py`, `h3.out`; `main`: `g3.out` |
| 14 | Default mesh, branch vs `main` vs no test calls (5 runs) | see the branch table; no test calls, cold: sin 0.029, 3x512 0.008 / 0.014 (with gradient), 6x2048 0.008 / 0.017 s | `compare.py`, `g2.out`, `k2.out`; CPU `j2.out` |
| 15 | Graph walk | GPU 1,006 nodes: walk 0.93, faster walk 0.67, backward 25.2 ms; 10,006 nodes: 9.46, 6.29, 220.5 ms; login CPU (4 threads) 256 nodes: 0.31, 0.29, 4.53 ms; whole calls within noise | `walk.py`, `g3.out` |
| 16 | New test file on the A100 | 5 passed, 9 xfailed, 16.4 s; on the login CPU 2 passed, 4 skipped, 8 xfailed | `h0.out` |

## Checked and found correct

- **Real CUDA errors leave the process usable.** This holds for errors inside `f` and inside `loss.backward()`, under a 1 GiB cap. After a backward error the restarted integral and gradient are bit for bit those of an uninterrupted run with the final batch size. After an error in `f` without gradients the integral is bit for bit that run's. With gradients it agrees to rounding, because the earlier batches were partitioned differently. Tests: `test_real_backward_out_of_memory_restarts_to_the_reference_gradient` and `test_real_out_of_memory_in_f_is_retried[False, True]`.
- **No endless restart.** A backward pass that fails at every size is attempted 14 times (2,300, 1,725, 1,288, ..., 46, 23 points). It then raises `BackwardOutOfMemoryError` with the gradients put back (`test_backward_that_never_fits_ends_in_an_error_after_few_restarts`).
- **Weak references and `id` keys.** When a leaf's Python object loses its last reference while the autograd graph still holds the tensor, PyTorch keeps the same Python object alive. Checked: no callback fires, and the graph returns an object with the same `id`. Keying by `id` is therefore sound; only the cycle (bug 2) is wrong.
- **`_Restart` cannot escape an inner call.** Every `integrate` call goes through the wrapper, so an inner call inside `f` handles its own restarts. A `BackwardOutOfMemoryError` from an inner call is not retried by the outer call's handlers, because `_is_out_of_memory` returns False for it. Positional arguments, subclass calls and `padaquad.integrate` all pass through `signature.bind`; I checked this by reading the code. Restarts with positional arguments are not tested.
- **The variable-sampling solver with growth.** With gradients the integral is right and the gradient equals the exact value to 5e-16. `conserve_memory=True` from its state gives the right integral.
- **Nothing in the memory code synchronizes with the GPU** except `torch.cuda.synchronize` in `_measure_call`. The whole test-call step, synchronization included, takes 0.3 to 1.4 ms on the default mesh.
- **The peak-memory reset** (suspect 4) is documented in the limitations list.

## What I could not check

- **Genuine out-of-memory conditions without a cap.** A backward error caused by fragmentation, or by other processes, rather than by the cap. `torch.cuda.set_per_process_memory_fraction` caps PyTorch's allocator, but `mem_get_info` does not see the cap, so the check before a backward pass was blind in these tests by construction.
- **Why the cache matters this much.** A near-limit call fails in milliseconds with the cache kept and after 1.5 to 2 s with it emptied, and the first call after emptying is slow. My hypothesis, getting tens of GB back from the driver, is not verified by profiling. PyTorch's `expandable_segments` allocator option may change both; not tried.
- **The variant of speed-up 2 that starts at the first pass** (estimated only), and bisection on the restart path (estimated only).
- **Multi-process `torch.distributed`, a real CPU out-of-memory error** (it can kill the process), **and two threads on one solver.** The last is the same mechanism as bug 3; not run.
- **Whether every `torch.compile` backend lifts parameters to inputs** (bug 5). Checked for `aot_eager` and one `nn.Linear`.

## Confidence

Sure, measured or read in the code:

- Bugs 1 to 4 and their consequences. Bugs 1 and 2 were reproduced on the A100.
- The time saved by keeping the cache, in these three settings.
- The size of the test-call cost and of the memory reads.
- The walk being a few percent of the backward pass.

Less sure:

- **How general speed-up 1's numbers are.** They come from one integrand family (wide tanh networks without gradients) on an 80 GB A100, and the mechanism is not established.
- **Speed-up 2's recommended variant**, which is estimated.
- **Bug 5's code fix**, which rests on one check; the documentation fix is safe.
- **The ratio-inflation note**, which is reasoning only.

## Connections

- PyTorch's `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (CUDA semantics notes, "Optimizing memory usage with `PYTORCH_CUDA_ALLOC_CONF`") targets the fragmentation that near-limit calls meet. It could change how quickly a call slightly too large fails, which matters for speed-ups 1 and 3. Untested here.
