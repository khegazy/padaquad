# Returning a state instead of `max_batch`: brainstorm

Status: brainstorm, 2026-11-17. Nothing built. Follows
`2026-11-17-memory-benchmark-speed-brainstorm.md` (option E) and
`2026-11-17-memory-benchmark-on-demand-plan.md`.

## Summary

Kareem's idea:

1. On the first call, test `f` on 1, 10, 100 and 1,000 points.
2. After that, let the batch size grow only when a pass needs more.
3. Each call returns a **state** holding the batch size, the optimal mesh and a
   true/false flag saying whether the batch size has converged.
4. The next call takes `state=result.state` in place of `mesh=` and
   `max_batch=`.

My assessment:

- **It works well, and it fixes the problem of the last discussion.** The
  returned batch size could be too small for the next call. With a state, the
  next call simply lets it grow again, once, when a pass needs it, and never
  repeats the tests it already made.
- **The state must also hold the memory readings** (the size and memory of
  each test call, a handful of numbers). Without them the next call cannot
  continue growing the batch size; it would have to start testing from 1
  again. This addition is my idea.
- **"Converged"** (Kareem's definition, section 3.1): False means the batch
  size may still grow in any later call; True means it will not, because the
  user gave `max_batch` or the memory limit was reached. On a CPU the limit
  must be predicted from the readings, not found by running out of memory,
  because there the process can be killed without an error.
- **Recommended rules** (section 5): an argument that says the same thing as a
  part of the state (`mesh=` or `max_batch=` together with `state=`) raises an
  error; the user edits the state instead. Bounds and per-call settings
  (`take_gradient`, `total_mem_usage`) combine with the state. A state made on
  another device, in another dtype or on another kind of machine keeps its
  mesh, and its memory readings are measured again, with a warning.
- **Cost:** it replaces `result.mesh_optimal` and `result.max_batch`, which the
  not-yet-tagged 2.0.0 just introduced (the last tag is `v1.4.1`). If 2.0.0 is
  not released yet, this can go into it; otherwise it is a 3.0.0.

## 1. What the state holds

| Part | What it is | Who uses it |
|---|---|---|
| `mesh` | Today's `mesh_optimal`: detached, on the solver's device. `None` for a state that holds only memory information. | The next call starts from it. |
| `max_batch` | The most points the next call may send to `f` at once, as known at the end of this call. | The user, to read. padaquad recomputes it from the readings. |
| `converged` | True/False; meaning in section 3.1. | The next call: when True, no more test calls. The user, to read. |
| memory readings (private) | For each test call: its number of points, the memory it used, and whether it was above the noise floor. Plus the reader's kind (GPU, Linux, other) and the device and dtype they were measured on. | The next call: it continues the measurement from them. |
| out-of-memory cap (private) | Set when a batch ran out of memory in this call. | The next call, so it does not run out of memory again at the same size. |

Rules for the object itself (mine):

- **It never changes after it is made** (a frozen dataclass). `integrate`
  returns a new state and leaves the one it was given as it was. That keeps
  2.0.0's promise that no call can change what another call reuses, and a
  call that raises leaves the caller a usable state to retry with.
- **Editing is done with the standard library:**
  `dataclasses.replace(state, mesh=None)` or `replace(state, mesh=my_mesh)`.
  No new methods are needed.
- **Saving needs care.** `torch.load` now defaults to `weights_only=True`,
  which loads only tensors, plain Python types, dictionaries "and any types
  added via torch.serialization.add_safe_globals()" (PyTorch documentation of
  `torch.load`, version 2.15). A saved state object would therefore fail to load
  in a checkpoint unless users register the class. The usual way out is the
  pattern of PyTorch's optimizers: `state.state_dict()` returns a plain
  dictionary, and `IntegrationState.from_dict(d)` rebuilds the object.

## 2. How a call uses the state, step by step

1. **Starting mesh:** the state's mesh, cut or extended by any bounds given
   (today's rule for a mesh with bounds). With no mesh in the state, the bounds
   are required, and a fresh random mesh is made.
2. **Memory readings:** taken from the state if they match this call's device,
   dtype and reader (section 5). Otherwise there are none.
3. **Before each pass:**
   - With no readings yet, test `f` on 1, 10, 100 and 1,000 points.
   - Then, while the pass is larger than 10 times the largest test call, make
     the next test call, 10 times larger (2 times after a slow call).
   - The pass's batch size is then the smaller of 10 times the largest test
     call and what the readings predict fits in the memory free now.
4. **At the end:** return a new state with the new mesh, all the readings, the
   batch size at the end, and the flag.

The batch size is recomputed from the memory free at each pass, as today
without `max_batch`. A reused state therefore adapts when other work takes or
frees memory. A reused fixed `max_batch` cannot, which is an improvement over
today's reuse.

## 3. What "converged" means

### 3.1 Kareem's definition (2026-11-17), which replaces 3.2

Kareem: "let max_batch continue to grow until it reaches an OOM. maybe in the
5th call it needs more memory. [...] As long as converged is not true then
max_batch can continue to grow. If the user provides a max_batch then we can
treat that as converged." He also asked how a call would know it is the 3rd
and not the 5th; it cannot, which rules out meaning 2 below.

So `converged` is a property of the state, not of one call:

- **False (the usual case):** the batch size may still grow. Any later call,
  whichever it is, makes a larger test call when one of its passes needs more
  than the state's readings cover. No call needs to know how many came before.
- **True:** the batch size will not grow again. No more test calls. It becomes
  True when:
  1. **the user gave `max_batch`** (it replaces the `fixed` part of section 1:
     one flag instead of two);
  2. **an out-of-memory error was caught**: in a test call, or in a pass with
     `take_gradient=False` (whose batches are retried smaller). The size at
     which it happened, times 0.75, becomes the cap;
  3. **(my addition, to confirm)** the readings predict that the next test call
     would not fit in the memory budget (three quarters of the usable memory),
     so the test call is not made. This is "reaching out of memory" without
     causing it.

Why item 3 is needed, in my view: on a CPU an allocation that does not fit
does not always raise an error. Linux's default memory setting refuses only
"obvious overcommits of address space" (kernel documentation,
https://www.kernel.org/doc/html/latest/mm/overcommit-accounting.html, mode 0,
"This is the default"). An allocation that is accepted and then does not fit
in physical memory can get the process killed, with no Python error to catch
(padaquad's own docs already say this, `devices_memory.md` limitation 10; on a
Slurm node the job's memory limit can do the same). So on a CPU the limit
must be found from the readings, not by running into it. On a GPU an
out-of-memory error is raised and caught cleanly, so items 2 and 3 both work.

What converged=True still allows: the batch size is recomputed at each pass
as the smaller of the stored size and what the readings predict fits in the
memory free now. So it can shrink when other work takes memory, but it never
grows.

Consequences to accept:

- On a large machine most integrals never get close to the limit, so
  `converged` stays False for them. That is normal and costs nothing: it only
  means "may still grow".
- With `take_gradient=True`, an out-of-memory error during a pass is still
  raised, not retried (a backward pass that fails halfway cannot be repeated
  safely). The test calls run forward only and the gradient factor 2.1 is an
  estimate, so this can happen as today. The caller keeps the state it passed
  in, unchanged.
- An out-of-memory error can come from something else using the GPU at that
  moment. The state then stays converged at a size smaller than the machine
  allows, until the user resets it with `replace(state, converged=False)`.

### 3.1b Kareem's refinement (2026-11-17, later message)

Kareem: "Test calls should only be made if max_batch and state are not
present, basically there is no idea of what max_batch is." And on item 3 of
3.1: "why does it need to calculate the usable memory? All it needs to know is
if it is larger than max_batch and if max_batch has converged."

So after the first call, growth comes from the integration's own passes, not
from test calls: when a pass is larger than `max_batch` and the state is not
converged, use the larger batch; if it runs out of memory, back off and mark
the state converged. This is option D of the speed brainstorm (Kareem's idea
2), started from the first call's test calls.

Where this is safe without any memory calculation, and where it is not:

| Situation | What "try the larger batch" does when it does not fit |
|---|---|
| GPU, `take_gradient=False` | An out-of-memory error, caught; the batch is retried at 0.75 times its size (`_evaluate_f_in_batches`). Safe. The work of the failed batch is lost, once. |
| GPU, `take_gradient=True`, error in the forward call of `f` | Raised today (`_evaluate_f_on_full_nodes` has no retry). A retry could be added: nothing has been written to the gradients yet. |
| GPU, `take_gradient=True`, error in the backward pass | Cannot be retried safely: measured earlier (speed brainstorm, option D item 5), a backward pass that raised midway had already written the gradient of one of two parameters, so a retry counts it twice. |
| CPU, any setting | Often no error to catch: the process or the Slurm job can be killed (section 3.1). |

Only the first row works with no prediction at all. In the other rows a
prediction from the first call's readings (does the larger batch fit in the
memory free now?) is what keeps padaquad from making a call that crashes. That
is the only reason to read the usable memory.

### 3.2 The two meanings I first proposed (superseded)

There are two reasonable meanings. They differ in the most common case.

1. **The machine's limit was found.** True when the memory free, and not the
   size of the test calls so far, sets the batch size. After that, testing
   larger sizes cannot raise it.
2. **The batch size stopped changing.** True when this call needed no new test
   call.

Example: a GPU run of the network times sin(1e5 t) (numbers from the
emulation of section 2.6 of the speed brainstorm). The first call's largest
pass had 94,208 points; the readings predict about 758,000 points fit in
memory.

| | Test calls made | Batch size at the end | Meaning 1 | Meaning 2 |
|---|---|---|---|---|
| Call 1 | 1, 10, 100, 1,000, then 10,000 | 100,000 | False (memory allows about 758,000) | False (it grew) |
| Call 2: its first pass is the whole mesh, 151,087 points | 100,000 | 1,000,000 | False (memory allows about 2.3 million) | False (it grew) |
| Call 3 and later: passes of at most 151,087 points | none | 1,000,000 | False | True |

With meaning 1, the flag stays False forever here, although nothing will ever
be tested again. On a CPU node with 251 GB the memory allows millions of
points, so almost every integral would show False. Users would read it as
"something is not finished".

With meaning 2, the flag says what a user wants to know: "this call spent no
time measuring". It can turn False again when an integral grows.

My recommendation: meaning 2, under a name that says it, such as
`max_batch_unchanged`. If meaning 1 is wanted as well, it can be a second
flag, `max_batch_at_memory_limit`. padaquad itself needs neither flag: the
readings already tell it what to do.

## 4. How well it works

Using the same example, compared with what the plan currently does
(`result.max_batch` is the size at the end, passed back as a fixed number):

| | Plan as written | With the state |
|---|---|---|
| Call 2 | The number 100,000 is fixed, so the pass of 151,087 points is cut into 2 calls of `f`. The same happens in every later call. | One test call of 100,000 points, once. Then every pass is one call of `f`. |
| What the user passes | `mesh=r.mesh_optimal, max_batch=r.max_batch` | `state=r.state` |
| `take_gradient` changes between calls | A number measured without gradients is too large with them (the docs warn about this). | The readings do not depend on it: the factor 2.1 for gradients is applied in each call. |
| Free memory drops between calls | The fixed number does not follow it. | The batch size is recomputed from the memory free at each pass. |

What gets better:

- One argument instead of two, for the common loop.
- Measuring happens only when an integral grows, and at most once per factor
  of 10 over a whole training run.
- The memory measurement no longer depends on `take_gradient`.

What gets worse, or stays a risk:

- **Readings can go stale.** If the integrand's memory per point changes over
  a training run (a larger model, a cache that fills), the old readings are
  too low. Reusing `max_batch` today has the same problem. With
  `take_gradient=False` an out-of-memory error is retried with smaller
  batches; with `take_gradient=True` it raises.
- **A state belongs to one integrand, and padaquad cannot check that.** Passing
  the state of a cheap `f` to an expensive one gives too large batches. Only
  documentation can prevent this.
- **More rules to learn**: which arguments combine with a state (section 5).
- **An API change** right after 2.0.0 (see the summary).
- **Saving a state in a checkpoint** needs the dictionary form (section 1).

## 5. Arguments given together with a state

Guiding rule (mine): **raise an error when an argument and the state say the
same thing, and combine them when they say different things.** Two sources
for one value means the user may not know which one was used; the error
message names the one-line fix.

| Given with `state=` | Recommendation | Why |
|---|---|---|
| `mesh_init`, `mesh_final` | Allowed: they cut or extend the state's mesh, as they do a mesh today. | Bounds and mesh say different things. Needed to integrate part of the mesh, or to extend it. |
| `mesh=` | Raise `ValueError`. Message: use `dataclasses.replace(state, mesh=m)`, or pass `mesh=m` without the state. | Two starting meshes. |
| `max_batch=` | **Open, after Kareem's 2026-11-17 message.** Either raise `ValueError` (message: use `replace(state, max_batch=n, converged=True)`), or let the given size replace the state's and return a state that is converged at that size. Kareem's "If the user provides a max_batch then we can treat that as converged" fits the second. | Two batch sizes. |
| Constructor `max_batch` | The state wins. | Every per-call value already beats the constructor's. |
| `take_gradient`, `total_mem_usage` | Allowed; used in this call. | The readings do not depend on them. |
| Bounds in another dtype | Converted to the mesh's dtype, as today. The readings match the mesh's dtype, so there is no conflict. | Today's rule. |
| A state with no mesh, and bounds in another dtype than the readings | The readings are dropped and measured again, with a warning. | Float32 readings are too low for float64. Measuring again costs little. |
| A solver on another device | The mesh is moved to the solver's device. The readings are dropped and measured again, with a warning. Devices are compared by kind and GPU name, not index, so `cuda:0` and `cuda:1` of one node count as the same. | Memory per point differs between a CPU and a GPU. |
| A state loaded on another machine | Same rule: the readings are kept only if the reader is of the same kind (GPU name; on Linux the same error allowance, which depends on the CPU count). | A checkpoint can be resumed elsewhere. |
| Another method or solver | Allowed. | A mesh is just points, and the batch size is in points. |
| Another integrand `f` | Cannot be detected; documented. | See section 4. |
| A converged state (a size the user gave, or a limit found) | Kept: no test call, until the user resets it with `replace(state, converged=False)`. | The state carries that decision to the next call. |

## 6. What else this touches

- `result.mesh_optimal` and `result.max_batch` are removed: one place for each
  value. Keeping them next to `result.state` would hold the same value twice.
- `solver.get_max_batch(f=..., t=...)` can stay as it is, returning a number. A
  user who passes that number with bounds gets back a state that keeps it
  fixed.
- The README, `repeated_calls.md`, `troubleshooting.md`, `examples/pode/trainer.py`,
  `examples/quadrature_basics.py`, and about 30 tests that pass `mesh_optimal`
  or `max_batch` from one call to the next change.
- The plan's Task 3 and Task 5 change; Tasks 1 and 2 (the resumable search) are
  needed either way.

## 7. Questions for Kareem

1. "letting max_batch grow as is needed": I read this as "make the next test
   call, 10 times larger, only when a pass needs more", as in the plan. The
   other reading is "grow it during the integration from the passes
   themselves", which is option D of the speed brainstorm. Which do you mean?
2. Answered: section 3.1. Still to confirm: item 3 there (the limit predicted
   from the readings counts as reached), and what `max_batch=` together with
   `state=` does (section 5).
3. Is 2.0.0 released to users yet? If not, the state can replace
   `mesh_optimal` and `max_batch` before they ever ship.
4. The rules of section 5, in particular raising an error for `mesh=` or
   `max_batch=` together with `state=`.

## 8. Confidence

- Sure: the example's numbers (from the emulation files) and the current
  behavior of padaquad cited above.
- Sure: the `torch.load` default, checked on the PyTorch documentation page.
- Reasoning, not tested: everything about how well the state works (section
  4) and the rules of section 5. None of it was run.
- Not sure how often memory per point drifts during real training runs; that
  sets how much stale readings matter.

## 9. Connections

- PyTorch optimizers keep their state in an object and give it out as a plain
  dictionary (`optimizer.state_dict()`, `load_state_dict()`) for checkpoints;
  section 1 borrows that pattern.
- diffrax lets a solve be resumed by passing state from a previous solve:
  `diffeqsolve(..., solver_state=..., controller_state=...)`, "Generally
  obtained by `SaveAt(solver_state=True)` from a previous solve" (diffrax
  documentation, `diffeqsolve`, https://docs.kidger.site/diffrax/api/diffeqsolve/).
  It is the same idea for an ODE solver.
