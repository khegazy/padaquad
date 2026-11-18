# Integration state and a batch size that grows: design

Status: design, 2026-11-17, for Kareem's approval. Replaces the plan in
`2026-11-17-memory-benchmark-on-demand-plan.md` (written for option E before
these decisions). Background: `2026-11-17-memory-benchmark-speed-brainstorm.md`
(evidence) and `2026-11-17-integration-state-brainstorm.md` (how this design
was reached).

## Terms

- **Batch size (`max_batch`):** the most points padaquad sends to the
  integrand `f` in one call.
- **Pass:** one turn of the integration loop, which evaluates every
  subinterval that still needs evaluation.
- **Test call:** a call of `f` on N copies of one point, made only to read how
  much memory it uses.
- **State:** the object a call returns so that the next call can start where
  it ended: the mesh, the batch size, and whether the batch size has
  converged.
- **Converged:** the batch size will not grow again (section 5).
- **Usable memory:** the free memory of the GPU minus the share that
  `total_mem_usage` keeps free, `free - (1 - total_mem_usage) * total`
  (`_get_usable_memory`, unchanged).

## 0. Prerequisite: fix today's `conserve_memory=True` bug first

Found by the adversarial review (`2026-11-17-integration-state-critique.md`,
F1) and reproduced on 2026-11-18. Kareem: it goes into the plan as its own
first task, done before the rest; not carried out yet.

- **The bug.** `_evaluate_f_on_split_residual_nodes` (`base.py:2172`, the
  `conserve_memory=True` path) plans a pass for a batch size, then calls
  `_evaluate_f_in_batches`, whose out-of-memory retry can shrink the batch
  inside the pass. The batches then end at other nodes than planned, and the
  code that splits off the carried, partly evaluated subinterval still uses the
  planned numbers. For gk21 with `max_batch=100` and one retry at 75, the
  batches evaluate 75 + 40 = 115 nodes, one whole subinterval more than the 92
  planned; that subinterval is treated as a partly evaluated one, its nodes are
  dropped (`nodes_flat[-C:0]`, an empty slice) and its values carried, and the
  next pass pairs them with other nodes. `integrate` returned 0.38 for an
  integral of 0.54, with `converged=True`. In 7 of 8 cases (gk21 and gk15,
  `max_batch` 60 to 200) the integral was off by 0.05 to 0.29. The existing test
  integrates f = 1, for which a wrong pairing gives the right answer.
- **The fix.** After the batches, put everything evaluated in node order (the
  carried nodes first), then cut at the last whole subinterval: the whole
  subintervals are returned, and the nodes after them (fewer than C) are
  carried. The cut is counted from what was evaluated, not from the plan, so
  it holds whatever batch sizes the retries used. About the same number of
  lines as the code it replaces.
- **The tests**, in `tests/test_oom_recovery.py`, on sin(40 t) + t, whose
  integral is known:
  1. the handler, called directly with out-of-memory errors above 30, 50 and
     80 % of `max_batch` (bosh3, gk15, gk21; `max_batch` 30 to 200): every
     returned subinterval holds its own C nodes with the integrand's values at
     them, the carried nodes are the first nodes of the next subinterval with
     their values, and nothing evaluated is lost;
  2. `integrate` end to end, the same cases, with `conserve_memory` True and
     False: the integral is right to 1e-5 and `converged` is True.
- **Status.** Written and run once on 2026-11-18, then set aside as Kareem
  asked: the patch is `docs/issues/2026-11-18-conserve-memory-fix.patch`
  (241 lines, code and tests; it applies to `main` as of 78c5ccb). Before the
  fix, test 1 failed on its first case; with it, the five test files that use
  `conserve_memory` passed (1,014 passed, 99 skipped, CPU on a GPU node). The
  whole suite was not run.
- **Effect on the design.** With this fix the batch size may change inside a
  `conserve_memory=True` pass, so that path can grow like the others
  (section 5.3). Its batch size still decides which subintervals a pass holds,
  so unlike the default path its results can depend on it (section 7.4).

## 1. Intent

Today every `integrate` call without `max_batch` first runs a memory benchmark
that often takes longer than the integral itself (0.7 to 1.6 s on a CPU node
for integrals that take 10 to 40 ms; about 0.1 s on an A100). Most users never
pass `max_batch`, so most of their run time is spent measuring one number.

The design removes that cost:

1. `integrate` never runs the memory benchmark. A call with no information
   about the batch size makes small test calls (1, 10, 100, ... points, up to
   the order of its first pass) that only check whether `f` runs at those
   sizes, and starts with the largest of them as its batch size.
2. After that, padaquad never tests ahead of time. The batch size grows
   whenever a pass needs more, until the GPU runs out of memory; then it
   shrinks and stops growing.
3. Each call returns a state, and the next call takes it, so what one call
   learned is not lost.

padaquad is designed for GPUs. CPU behavior must work, but its memory risks
are accepted (section 9).

## 2. Goals, non-goals, anti-goals

Goals:

- The time padaquad spends on memory, for an `integrate` call without
  `max_batch`, is a few milliseconds, plus, once per state, the retries after
  the first out-of-memory error (section 5.4).
- A loop of calls (training) passes one object, `state=result.state`, and
  never measures again after the first call.
- No out-of-memory error during a backward pass leaves wrong gradients.

Non-goals:

- Finding the largest batch the machine can hold. The batch size grows only
  as far as the integrals need.
- Changing `solver.get_max_batch(f=..., t=...)`. It keeps today's full
  benchmark and returns a number.
- Making CPU memory safe (section 9).

Anti-goals (failure even if it otherwise works):

- A result that silently depends on whether an out-of-memory error happened,
  beyond what a different batch size already changes today with
  `take_gradient=True` (section 7.4).
- Gradients counted twice.
- Changing the benchmark's warnings beyond what section 8 says (Kareem first
  forbade any edit, then, once `integrate` no longer ran the benchmark, asked
  for the change described there).

## 3. Approaches considered

| Approach | How the batch size grows after the first call | Trade-off | Outcome |
|---|---|---|---|
| Benchmark on demand (option E) | More test calls, 10 times larger each, only when a pass needs them | Measured to work; but every growth costs test calls, and a returned number can be too small for the next call | Replaced |
| State with the test readings kept | As above, continuing from readings stored in the state | Fixes the "too small" problem; still test calls, and more to store and check | Replaced |
| **State, growth until out of memory** | **The pass simply uses a larger batch; an out-of-memory error shrinks it and stops growth** | **No measuring after the first call; needs protection where an out-of-memory error cannot simply be retried (section 6)** | **Chosen by Kareem** |
| Measure memory during every pass (option D) | From readings of the real passes | Reads memory while other work runs; more code | Not chosen |

## 4. The state

A frozen dataclass, `IntegrationState`, exported from `padaquad`.

| Field | Type | Meaning |
|---|---|---|
| `mesh` | `torch.Tensor` or `None` | The mesh to start the next call from: today's `mesh_optimal` (detached, on the solver's device). |
| `max_batch` | `int` | The batch size the next call starts with. |
| `max_batch_converged` | `bool` | True when the batch size will not grow again (section 5). |

- `IntegrationResult` gains `state` and loses `mesh_optimal` and `max_batch`
  (one place for each value). `IntegrationResult.converged` (whether the
  integral met its tolerance) is unrelated, which is why the state's flag is
  named `max_batch_converged`.
- `integrate` returns a new state and never changes the one it was given.
- Editing uses the standard library: `dataclasses.replace(state, ...)`.
- Saving: `dataclasses.asdict(state)` gives a dictionary of a tensor, an int
  and a bool, which `torch.load` reads with its default `weights_only=True`;
  `IntegrationState(**d)` rebuilds it. No new methods.

## 5. How a call chooses the batch size

### 5.1 At the start of a call

| What the call is given | Starting batch size | Converged |
|---|---|---|
| `state=` | `state.max_batch` | `state.max_batch_converged` |
| `max_batch=n` (with or without `state=`) | `n` | True (Kareem: "If the user provides a max_batch then we can treat that as converged") |
| Constructor `max_batch=n`, and no `state=` | `n` | True |
| None of these | Test calls, section 5.2: the largest that ran | False, unless a test call ran out of memory |

Test calls are made only in the last row (Kareem: "Test calls should only be
made if max_batch and state are not present"; "If max_batch is not given and
no state exists, then it needs to run the test calls").

### 5.2 Test calls, but no benchmark, in `integrate`

Kareem (2026-11-17): "Since the memory test is not longer measuring a cost per
sample, we are no longer forced to run the memory measurement function.
Therefore we do not need the error messages. However, you should keep the
function in case people want to call it themselves to measure max_batch." And
later: "If max_batch is not given and no state exists, then it needs to run
the test calls."

Read together (my reading, to confirm): the test calls stay, but they no
longer measure anything.

- With neither `max_batch` nor a state, after today's one-point output check,
  `integrate` calls `f` on 10, 100, ... copies of one point (the one-point
  check is the test call of 1 point), up to the power of 10 that is the order
  of the first pass: the largest power of 10 not above the number of points of
  the first pass (Kareem: "cap the test calls at the same order of the first
  pass"). Examples: a first pass of 253 points (the default starting mesh of
  gk21) gives test calls of 10 and 100; one of 46,000 points (a mesh passed in)
  gives 10 to 10,000. The calls are forward only. Their only purpose is to see
  whether `f` runs at these sizes.
- No memory is read, nothing is fitted, nothing is logged, and nothing is done
  after the calls (no garbage collection, no emptying of the GPU cache, no
  sleep).
- If all run: the batch size starts at the largest test call, not converged,
  and grows by factors of 10 (section 5.3). The first pass is then at most 10
  times the batch size, so it runs as one batch (section 5.3).
- If a test call of B points runs out of memory (Kareem, 2026-11-17): the
  batch size becomes 0.75 B and `f` is called again on that many copies; if
  that also runs out of memory, 0.75 times that, and so on until a call runs.
  This is the same rule as every other out-of-memory retry (sections 6.1 and
  6.2). The size that ran is the batch size, the state is converged, and no
  larger test call is made. A single point that does not fit raises, as
  today.
- Any other error from `f` in a test call is raised as it is.
- `integrate` no longer calls `_setup_memory_checks`: neither of its two
  warnings, and none of its errors (memory growing faster than N², no reading
  above the noise floor, the time limits). The builder may reuse
  `_measure_call`, which already builds the N copies and recognizes an
  out-of-memory error, or write the three calls directly; whichever is less
  code.
- `_setup_memory_checks` stays, unchanged in what it measures, as the engine of
  `solver.get_max_batch(f=..., t=...)`, its only caller. A number from it,
  passed as `max_batch`, is a fixed size (converged, section 5.1).
- `integrate` no longer sets `f_mem_coeffs`. `solver.get_max_batch()` without
  `f` returns the constructor's `max_batch`, else the batch size from the fit
  of the last `get_max_batch(f=...)`, else `None` (mine).
- The benchmark's start warning is shortened (section 8).

### 5.3 During the call

Kareem (2026-11-17): growth "should not jump straight to the largest size,
otherwise if that number is very large and max_batch keeps getting reduced by
0.75 each time, that could take a long time. Let max_batch grow by a factor of
10 every time it needs to grow."

Each time padaquad is about to call `f` (each batch, not each pass):

- **Not converged, and more points are waiting than the batch size:** this
  batch is 10 times the batch size, or the points waiting if fewer.
- **Otherwise:** the batch size is used as it is.

The first pass of a call in particular (Kareem, 2026-11-18): "As long as the
first pass fits in the max_batch or 10*max_batch for not converged, then use
the full first pass." So a first pass of at most the batch size (converged), or
at most 10 times it (not converged), is one batch. A larger first pass, for
example a large mesh passed in, is cut by the rule above. This is the case in
which results with `take_gradient=True` depend on the batch size (section
7.4).

So a pass of 151,087 points with a batch size of 1,000 and
`take_gradient=False` is evaluated in calls of 10,000, then 100,000, then the
41,087 left (if none runs out of memory). Growth per call of `f`, not per
pass, is my reading of "every time it needs to grow"; growth once per pass
would cut that pass into 16 calls of 10,000.

After a batch runs without error, the state's batch size becomes the larger of
its old value and that batch's size, so it always means "the largest batch
known to fit". The out-of-memory handling of section 6 can lower it and set
converged to True. Once converged, the batch size never grows again in this or
any later call that uses the state.

`conserve_memory=True` follows the same rule (mine), which needs the fix of
section 0 first: without it a batch size that changes inside a pass gives
wrong integrals (critique F1).

### 5.4 Cost of the first out-of-memory error

A batch that does not fit is retried at 0.75 times its size until it fits.
With growth by 10, the failed batch is at most 10 times one that already ran,
so there are at most log(10) / log(4/3) ≈ 8 retries (8 were observed). In a
cold call passes at most double, so the failed batch is at most twice one that
ran and about 3 retries suffice (the reviewer's reasoning, not measured).

Measured by the reviewer on an A100 80 GB (critique F4): one episode lost 4.9
to 6.3 s on three models. A call far too large fails in about 5 ms; a call a
little too large runs almost to its end first (1.5 to 2.6 s); emptying the
cache took up to 1.5 s. My earlier estimate of "under a second" was wrong. It
happens once per state, since the state is converged afterwards. The bound of
8 does not hold when a state is reused under other settings (for example a
size learned with `take_gradient=False` used for training): the retries are
then log(ratio) / log(4/3) with no upper limit. On the `take_gradient=True` path these are
forward retries (6.2) or one proportional shrink (6.3), not restarts.

A call does not need to know how many calls came before it (Kareem's question
"how would any call know that it is the 3rd vs the 5th call"): it compares
each pass with the batch size in the state.

## 6. Out-of-memory protection on the GPU

### 6.1 `take_gradient=False`: an error in `f`

Unchanged code path: `_evaluate_f_in_batches` catches the error, frees memory,
and retries the batch at 0.75 times its size. New: the state's batch size
becomes that smaller size, and converged becomes True.

### 6.2 `take_gradient=True`: an error in `f`

Today `_evaluate_f_on_full_nodes` has no retry. Add one, the same way as 6.1:
catch the error, drop the partial outputs, free memory, retry with 0.75 times
as many subintervals, set converged. This is safe because no gradient has been
written yet.

### 6.3 `take_gradient=True`: making sure the backward pass fits

Kareem's rule: the backward pass costs roughly as much memory as the forward
pass, so before it runs, check that this much memory is available, keeping the
`total_mem_usage` share free.

1. Before the batch's call of `f`: reset the GPU's peak-memory counter and read
   the memory in use.
2. After the call: the forward memory F is the peak minus the memory in use
   before it.
3. Read the usable memory U (with the batch's results still held).
4. If U >= F: go on to the backward pass.
5. If U < F: drop this batch's outputs and their autograd graph (no gradient
   has been written), lower the batch to fit (the builder chooses the rule; it
   must shrink by at least the factor 0.75, and a proportional estimate such as
   N (U + F) / (2 F) is acceptable), set converged, and evaluate again.

The check runs on every gradient batch on a GPU, including a batch size the
user fixed (mine): it costs two counter reads, and the alternative, a backward
pass that runs out of memory, is worse. Like 6.1, 6.2 and 6.4, it can lower a
size the user gave.

### 6.4 `take_gradient=True`: an error in the backward pass anyway

Every batch adds its gradient into the `.grad` of the integrand's parameters
as soon as its backward pass runs. When a backward pass fails, the earlier
batches of the call have already added theirs, and the failed pass may have
added part of its own (measured in the speed brainstorm, option D item 5: one
of two parameters already had its gradient). Starting the integral again
without undoing this counts gradients twice. padaquad cannot undo it by itself:
`f` can be any function, so padaquad does not reliably know the model's
parameters (for example `f = lambda t: model(t)`), and zeroing gradients would
also erase what the user added before the call.

**Which tensors are restored (Kareem, 2026-11-18: the graph walk, checked by
tests).** Before every backward pass padaquad walks that batch's autograd
graph and saves the `.grad` of every leaf tensor it reaches for the first time
in the call (every leaf that receives a gradient is `node.variable` of an
accumulate-gradient node). Walking every batch, not only the first, also
covers a parameter that only a later batch reaches (a data-dependent branch).

The walk can miss a parameter only behind a custom autograd function: a
function that uses a tensor it does not take as an input, for example
activation checkpointing with `use_reentrant=True` (checked in PyTorch 2.8.0:
2 of 4 parameters found). Such a function shows up in the graph as a
`BackwardCFunction` node, so padaquad knows when the walk may be incomplete.
With `use_reentrant=False`, PyTorch's recommended setting, the walk found all
4.

**`parameters=` is needed only with a custom autograd function.** It is a new
optional keyword argument of `integrate` (and `padaquad.integrate`), taking an
iterable of tensors or a `torch.optim.Optimizer` (its `param_groups`). The
tensors given are saved at the start of the call, in addition to those the
walk finds. Without a custom autograd function it is never needed.

**On an out-of-memory error in `loss.backward()`:**

1. Put every saved `.grad` back.
2. Drop the call's tensors and free memory, and log a warning: the batch size
   at which the backward pass ran out of memory, the new batch size, and that
   the integral is restarted with the gradients of the failed attempt removed.
3. Call `integrate` again from scratch with the same arguments, except the
   state: the mesh the call started from (the state's mesh as passed in, or
   the fresh random mesh the call made, which is kept for this), a batch size
   of 0.75 times the one that failed, and converged set to True.
4. If the restarted call fails the same way, repeat. When the batch would fall
   below one subinterval's nodes, raise the out-of-memory error.

**Instead of restarting, it raises** `padaquad.BackwardOutOfMemoryError` (a
subclass of `torch.OutOfMemoryError`, carrying the `state` to call again with,
and a message saying that the gradients must be reset first) when padaquad
cannot be sure to undo the gradients: a custom autograd function is in the
graph and no `parameters=` was given; a saved tensor has a hook that runs
after its gradient is added (`register_post_accumulate_grad_hook`); or
multi-GPU training (below).

Restarting from the starting mesh, not the partly refined one, makes the
result the same as a call run at the smaller batch size from the start (mine).
Gradients of the bounds are not affected: they are passed back once after the
loop (`mesh_grad`), which the restarted call redoes.

**Cases that always raise, even with `parameters=`:**

- Multi-GPU training: a `torch.distributed` process group with more than one
  process is running. A restart on one process breaks the gradient averaging
  of the others. Left for the future: `docs/issues/multi-gpu-backward-oom-restart.md`.
- A given parameter has a hook that runs after its gradient is added
  (`register_post_accumulate_grad_hook`, for example an optimizer step inside
  the backward pass): that work cannot be undone. Detectable: such a tensor's
  private attribute `_post_accumulate_grad_hooks` is not `None` (checked in
  PyTorch 2.8.0); it is private, so a test must pin it.

## 7. Arguments given together with a state

| Given with `state=` | Rule | Source |
|---|---|---|
| `mesh_init`, `mesh_final` | Cut or extend the state's mesh, as for a mesh today. | Today's rule |
| `max_batch=n` | Replaces the state's batch size; converged True. | Kareem |
| `mesh=` | `ValueError`: two starting meshes. The message names `dataclasses.replace(state, mesh=m)` and `mesh=m` without the state. | Mine |
| Constructor `max_batch` | The state wins (every per-call value already beats the constructor's). | Mine |
| `take_gradient`, `total_mem_usage` | Used in this call; not stored in the state. | Mine |
| A state with `mesh=None` | Bounds are required; a fresh mesh is made. | Mine |
| Another device | The mesh is moved to the solver's device, as a passed mesh is today. | Today's rule |
| Another integrand | Not detectable; documented: a state belongs to one integrand. | Mine |

### 7.4 Effect on results

- `take_gradient=False`, `conserve_memory=False`: the batch size only cuts a
  pass into calls of `f`, so results do not depend on it.
- `conserve_memory=True`: the batch size decides which subintervals a pass
  holds (`tests/test_error_integral_reference.py` relies on this), so results
  can depend on it.
- `take_gradient=True`: one batch is one turn of the loop, judged against the
  running integral of the batches so far, so results depend on the batch size
  whenever a pass is cut. Measured by the reviewer (critique F3, an emulation
  of this design's growth inside today's loop, on integrands built to be hard):
  - A cold call: identical to an unlimited batch in 9 of 9 comparisons, since
    the first pass runs whole and later passes at most double.
  - A first pass larger than 10 times the batch size (a large mesh passed in),
    or a batch size frozen below the pass size by an out-of-memory error: with
    gk21 the true error stayed far inside the tolerance and only `converged`
    changed; with bosh3 and dopri5 the integral differed by 4 to 166
    tolerances (7,718 in one case), mostly with `converged=False`. In one case
    the limited run was the more accurate one.
  My earlier sentence ("slightly different results (within the tolerance)")
  was wrong.

## 8. Messages

`integrate` logs no benchmark message. The benchmark's two messages now appear
only in `get_max_batch(f=...)`.

**Start warning, new text** (Kareem: "simply saying that this function can be
slow and should be called once and then save max_batch, also the current
warning that it is not optimized for every computer"). Proposed wording, for
Kareem's approval; the first sentence and the second bullet's first two
sentences are today's, word for word:

> PadaQuad is measuring how much memory one integrand (f) evaluation uses, to
> determine max_batch (the maximum number of evaluations per batch).
> - This measurement can be slow. Run it once, save the max_batch it returns,
>   and pass that value to integrate().
> - The memory cost measurement may not be optimized for your machine and may
>   result in an underutilization of your hardware (or, for some integrands, a
>   batch too large for memory); see
>   https://khegazy.github.io/padaquad/user_guide/devices_memory.html#ug-memory-limitations
>   for more details. The only way to guarantee that max_batch makes full use
>   of your hardware's memory is to measure max_batch yourself.

Removed from today's text: why the measurement is needed during an integral,
that it runs on every `integrate()` call, the advice about `result.max_batch`
and `mesh=result.mesh_optimal` (both removed), and the sentence "If you are
not reusing the integrator many times ...". The note for systems whose
readings miss reused memory (macOS, Windows) stays appended as today.

**End message:** unchanged except its last clause, "pass max_batch=N to skip
this measurement in later runs", which becomes "pass max_batch=N to
integrate()" (mine, to confirm: `integrate` no longer runs this measurement,
so there is nothing to skip).

Other messages:

- The existing "Caught OOM ..." warning: unchanged.
- New: the warning of section 6.4, and one warning in 6.2 and 6.3 saying the
  batch was lowered and why (mine).
- No message when the batch size grows (mine).

## 9. CPU

Same rules, without the checks of section 6.3 (no cheap peak counter on a
CPU). An allocation that does not fit can get the process or the Slurm job
killed without a Python error; this is accepted (Kareem: "You can ignore an
issue on CPU, padaquad is designed primarily for gpus") and documented as a
limitation.

## 10. What else changes

- `integrate` (solver method and `padaquad.integrate`) gains the keyword
  argument `state=`.
- Removed: `IntegrationResult.mesh_optimal`, `IntegrationResult.max_batch`.
  Everything that passes them on (README, `repeated_calls.md`,
  `troubleshooting.md`, `devices_memory.md`, `examples/pode/trainer.py`,
  `examples/quadrature_basics.py`, about 30 tests) moves to `state=`.
- `limit_benchmark_time` affects only `get_max_batch(f=...)`.
- Version: stays 2.0.0. 2.0.0 has not been released, and this change is
  merged before it (Kareem, 2026-11-17). The `[2.0.0]` section of
  `CHANGELOG.md` is edited in place to describe `state=` and
  `IntegrationResult.state` as the 2.0.0 way to reuse a call.
  - `IntegrationResult.max_batch` was added on `main` after 1.4.1 and never
    released: its entry is removed from the changelog, with no migration note.
  - `IntegrationResult.mesh_optimal` was released in 1.4.1 (checked in
    `git show v1.4.1:padaquad/results.py`): it is listed under "Removed", with
    the migration `mesh=result.mesh_optimal` → `state=result.state` (or
    `mesh=result.state.mesh`).
- The memory-benchmark limitations page gets the new limitations: the batch
  size grows by trying (up to about 8 failed calls of `f`, once per state, on
  a GPU; section 5.4); the backward
  check assumes backward memory about equals forward memory; CPU (section 9).

## 11. Decisions and who made them

Kareem:

- Growth until out of memory, by a factor of 10 each time it is needed.
- Converged by an out-of-memory error or a given `max_batch`.
- Test calls only with no `max_batch` and no state.
- `integrate` does not run the memory benchmark; the function stays for
  `get_max_batch(f=...)`, with a shortened start warning (section 8).
- Sections 6.1 to 6.3, and in 6.4 a restart of the whole integral from
  scratch.
- `parameters=` is optional; without it a backward out-of-memory error raises.
- Multi-GPU training is left for the future
  (`docs/issues/multi-gpu-backward-oom-restart.md`).
- CPU memory risks accepted.
- 2.0.0 is not released, and this change goes into it.

Mine, to check while reading:

- The test calls only check that `f` runs (no memory is read).
- The exact wording of the new start warning and of the end message's last
  clause (section 8).
- The field names (`IntegrationState`, `max_batch_converged`), the error name
  `BackwardOutOfMemoryError`, and one argument `parameters=` that also takes an
  optimizer.
- Growth per call of `f`, not per pass.
- The restart starts from the call's starting mesh.
- The completeness check of the parameter list before the first backward pass.
- `mesh=` with `state=` raises; the state beats the constructor's `max_batch`.
- No message on growth.
- `conserve_memory=True` grows like the other paths.
- The backward check of 6.3 also runs on a size the user fixed.
- `get_max_batch()` without `f` no longer uses the last `integrate` call.

## 12. Left to the builder

- The exact shrink rule of 6.3, within the bounds given there.
- How the restart of 6.4 is structured in the code (for example an internal
  exception caught at the top of `integrate`), provided the restarted call
  receives the original arguments except the state.
- Test names and fixtures. Tests must cover every row of sections 5.1, 6 and 7
  on CPU with fakes, and sections 6.2 to 6.4 also on a GPU, including: a
  restart with `parameters=` leaves each gradient equal to that of a call run
  at the smaller size from the start; a call without `parameters=` raises
  `BackwardOutOfMemoryError` with a usable `state`; an incomplete parameter
  list raises before any gradient is written.

## 13. Open questions

All resolved on 2026-11-18: see the decisions in
`2026-11-18-memory-state-plan.md`, which amend this design where they differ
(in particular, `result.mesh_optimal` and `result.max_batch` stay, and
`result.state` is built from them). The list below is kept for the record.

From the adversarial review (`2026-11-17-integration-state-critique.md`), not
yet decided:

1. **Which tensors to restore before a restart (6.4).** Kareem asked whether
   walking the graph misses parameters; answer and proposal in the reply of
   2026-11-18, to be written here once decided.
2. **The backward check (6.3, critique F2):** skip it when the output of `f`
   does not require a gradient; keep the rule U >= F or measure the backward
   peak too; activation checkpointing (backward 4 to 6 times the forward
   memory) as a documented limitation.
3. **`total_mem_usage` (critique F5)** no longer keeps memory free except in
   the backward check: document it, or keep the promise.
4. **A batch size that shrank never grows back (critique F6).**
5. **The new error (critique F8):** `_is_out_of_memory` must not treat it as an
   ordinary out-of-memory error, and it must survive pickling.
6. **Smaller points of the critique (F10 to F13):** which per-call attributes
   survive, edge cases of the state, the real amount of test work (107 test
   functions), and GPUs that fall back to system memory instead of failing.
