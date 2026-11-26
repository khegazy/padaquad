# Speed-up plan: evaluations of f at the start of a call (`startup-evaluations`)

**Status:** reviewed 2026-11-26 (section "Review"); implemented on branch `startup_evaluations`. Code references below the Review are to
commit `0590ccf` (branch `fix_conserve_memory`); `HEAD` is `8df47d5`, which
differs from it only in `docs/conf.py`, so every line number below holds for
both. Prototype, scripts and raw outputs:
`/pscratch/sd/k/khegazy/padaquad_speedup_plans/startup-evaluations/` (scratch
is purged; copy what should be kept).

## Review (2026-11-26, implementing agent, against `73caae8`)

The plan was written against `0590ccf`; `73caae8` (the tip of
`speedups_integration`) contains `call-setup-cost` and `endpoint-evaluations`.
Checked against that code, the git history and runs of the planning scripts
(copied to `$SCRATCH/padaquad_speedup_plans/startup-evaluations/impl/`, run
against an archive of `73caae8`). What was wrong or has changed, and what this
review changes:

1. **Line numbers and the start-up point are stale.** The shape check is now
   `base.py:1051-1096`, the start of the batch size `1110-1125`,
   `_initialize_max_batch` `memory.py:1599-1649`, `_call_f` `747-801`, the row
   check `base.py:2477-2482`, `get_max_batch`'s copy of the dimension check
   `memory.py:641-649`. Since `endpoint-evaluations` the start-up calls do not
   use `mesh_init` but a "probe", the first node of `[mesh_init, mesh_final]`
   (`base.py:1061-1063`), so that `f` is not called at a bound a Gauss-Kronrod
   method never evaluates. The probe, its comment, and the edit that the same
   branch made to `tests/test_memory_state_adversarial.py::test_saved_gradients_do_not_keep_every_batch_alive`
   (`if len(t) > 1`, because the probe carries the bounds' autograd graph) go
   with the calls. Corrected below where the line numbers matter.
2. **The counts are for 23 points per gk21 subinterval.** gk21 now evaluates
   21. Re-run at `73caae8` (`impl/counts_base.txt`): the default gk21 first
   pass is 231 points (not 253), a mesh of 500 subintervals 10,500 (not
   11,500), 2,000 subintervals 42,000. The extra points per call are as the plan
   says (1, 11, 111, 1,111), so a default cold call evaluates 342 points for
   231 used (32 % of its points are discarded; it was 111 of 364). `n_first` is
   now `n_start_steps * self.C` (`call-setup-cost`), a host integer, so the
   plan's `int(mesh_trackers.sum())` (a device read) is not needed.
3. **The out-of-memory chains are re-derived for C = 21** (`impl/oom_base.txt`
   at `73caae8`; the plan's table used 253-point first passes): see the
   "Implemented" section for both sides.
4. **History (task doubt 1), checked:** the design (`6b3cf42`,
   `2026-11-17-integration-state-design.md` sections 5.1-5.3), the critique
   (F9), the brainstorm (option D, items 2 and 5) and the break-and-speed
   review (`59b7f11`, speed-up 2) say what the plan quotes. What the test calls
   protected against: (a) a first call more than 10 times larger than any call
   that ran ("today's caution" of the old benchmark), which matters on a CPU,
   where an allocation that does not fit can kill the process without a Python
   error; (b) the one-point call also warmed `f` up before the benchmark read
   time and memory (the benchmark has left `integrate`). What replaces it:
   nothing for (a): the first call is the first pass, at most 10,000 points,
   retried smaller after an out-of-memory error exactly as every later call;
   the documented limitation "Growth needs a clean failure"
   (`devices_memory.md`) already tells CPU users to pass `max_batch`, and it is
   extended to say that the first call is not preceded by a smaller one. For
   (b): `get_max_batch(f=...)` keeps its own one-point call before the
   benchmark; the loop reads no time, and the one memory reading in the loop
   (the check before a backward pass on a GPU) sees `f`'s one-time
   allocations in its first batch (section 9).
5. **Placement of the moved checks (D3, D4), changed to need no new branch or
   flag in the loop.** The plan adds a block that runs once after the first
   `_evaluate_f_on_mesh`, with a `first_turn` flag. Leaner placements exist:
   - *Output shape (≥ 2 dimensions, one row per point):* in `_call_f`, as the
     plan says, one condition and one message for both (the two messages
     merge). Every call of `f` in the loop and `get_max_batch`'s one-point
     call (D5) go through it.
   - *`error_integral_reference` shape (D4):* inside the existing
     `if error_integral_reference is not None:` of Step 4, where the reference
     replaces the running integral and D is known (`current_integral`). It
     runs on the first pass, before any subinterval is judged, recorded or
     backpropagated, so the error comes before a second pass and before any
     gradient is written, with today's message. It is re-checked on later
     passes at the cost of two shape reads (no device read).
   - *The `trade_speed_for_memory` warning (D3):* after the loop, from
     `record["integral"].requires_grad` (the integral before `y0` is added;
     D3's condition), still only when `state is None`. No flag, no branch in
     the loop. Cost: the warning is logged at the end of the call instead of
     before the first evaluation, and not at all if the call raises. It is
     advice about memory that was already spent; I judge this acceptable and
     report it as a behaviour change.
6. **D5 is taken.** `get_max_batch(f=...)` makes its one-point call through
   `_call_f(..., can_shrink=False)`: one check and one message. A one-point
   out-of-memory error there is raised as `torch.OutOfMemoryError` with
   `_call_f`'s sentence.
7. **D6:** `CHANGELOG.md` `[2.0.0] - 2026-11-25` is untagged (`git tag` ends
   at `v1.4.1`), so its entries are edited in place: the "`integrate` no
   longer runs the memory benchmark" entry (Changed) and the "integrable
   singularity" entry (Fixed), which mentions the start-up calls.
8. **Commits split so that D1 can be dropped.** Removing the one-point call
   (with the moved checks) is one commit; removing the test calls (D1, which
   reverses the user's decision 11 of 2026-11-18) is the next, so the user can
   keep the test calls (alternative D) by dropping it.
9. **Test list to be re-found** at `73caae8` by running the suite (section
   6.2 named 23 tests at `0590ccf`; `tests/test_interior_nodes.py` is new since
   and pins the start-up calls in
   `test_f_gets_c_points_per_subinterval_and_no_mesh_point`). Added: the
   invariant test of 6.1 (with `C = solver.C`, which is 21 for gk21) and a
   check that an integrable singularity at 0 under gk21 without `max_batch`
   makes no call of `f` at the bound, for both values of `take_gradient`.
10. **Stale documentation outside the plan's list:**
    `docs/getting_started/quickstart.md` still says that the first call
    measures memory for "a few seconds" (stale since the benchmark left
    `integrate`); `README.md` ("the first call tries a few small batches") and
    `docs/user_guide/devices_memory.md` ("253 points"); `CLAUDE.md`'s test map
    row of `test_interior_nodes.py` ("start-up calls included").
11. **Stochastic integrands (task doubt 4), checked at `73caae8`:** the order
    is shape check (`f` on the probe), then `_setup_initial_mesh` (draws the
    random mesh with `torch.rand`), then the test calls, then the first pass.
    So for an `f` that draws from the global generator, the starting mesh and
    every later draw change; for a deterministic `f` nothing changes. The advice
    "call `torch.manual_seed` before `integrate()`" stays true (the results
    repeat for a given seed before and after); what changes is which mesh a
    given seed gives with a stochastic `f`, and the mesh no longer depends on
    `f` at all.

## 1. Summary

- **What is slow.** Every `integrate` call evaluates `f` before the
  integration starts and throws the results away: one point for a shape check
  (every call, also with `state=` or `max_batch=`), and, with neither
  `max_batch` nor `state`, test calls on 10, 100 and 1,000 copies of the lower
  bound. That is 1, 11, 111 or 1,111 extra points and 1 to 4 extra calls of
  `f` per call (exact counts in section 2.3).
- **What changes.** Both are removed. The output checks move to the first real
  call of `f`, and a call with neither `max_batch` nor `state` starts at a
  batch size chosen so that its first call of `f` is the same call as today
  (the whole first pass, at most 10,000 points). Every evaluated point then
  belongs to the integration. About 70 lines are removed, about 25 added.
- **Gain** (measured, A100 shared with other agents, gk21, no `max_batch`,
  medians of 9 calls): a 6-layer, 2048-wide network went from 7.4 and 7.8 ms
  to 6.1 and 6.0 ms with the default mesh (two rounds each; 364 to 253 points,
  4 calls to 1), and from 39.9 and 38.8 ms to 35.7 and 34.8 ms with a starting
  mesh of 500 subintervals (12,611 to 11,500 points). For a cheap integrand
  the change is inside the run-to-run noise. For an `f` whose cost is
  proportional to its points, the saving is the share of discarded points: 30 %
  of the points of a one-pass default call (111 of 364).
- **Returned numbers.** Identical bit for bit (integral, error, contributions,
  state mesh, `state.max_batch`, `converged`, gradients) in all 32 measured
  cases, for an elementwise integrand and a small network, with
  `take_gradient` True and False, unless (a) `f` runs out of memory in its
  first call, or (b) `f` draws from PyTorch's global random number generator,
  whose state the shape check advanced before the random starting mesh was
  drawn. Section 6.3 states both.

## 2. Evidence

### 2.1 The code as it is now

1. **The shape check**, `padaquad/quadrature/base.py:1113-1150`. `integrate`
   calls `f(mesh_init.reshape(1, -1), *f_args)` (1118), then
   - raises `ValueError` if the output has fewer than two dimensions
     (1119-1123);
   - logs a warning when `state is None`, `trade_speed_for_memory` is True and
     the output requires grad (1128-1138);
   - checks that `error_integral_reference` has shape `[]`, `[1]` or `[D]`,
     with D the output's last dimension (1141-1149);
   - deletes the output (1150).
   The comment (1113-1115) says the call also "warms f up, so one-time costs
   (CUDA start-up, lazy model loading, cuBLAS workspace) stay out of" the test
   calls. It runs before `_setup_initial_mesh` (1155), which draws the random
   starting mesh with `torch.rand` (`base.py:3575`).
2. **The test calls**, `padaquad/quadrature/memory.py:1599-1649`
   (`_initialize_max_batch`), called at `base.py:1172-1179` only when neither
   a `max_batch` (call, or constructor without a state) nor a `state` is given.
   It calls `_measure_call` (`memory.py:1434-1518`) with readers that do
   nothing (`lambda: None, lambda: 0`, 1643) on n = 10, 100, 1,000 copies while
   n ≤ min(n_first, `_LARGEST_TEST_CALL` = 1000) (`memory.py:72`, 1640), where
   n_first is the number of points of the first pass. It returns the largest
   n that ran (1 when n_first < 10), not converged; an out-of-memory error
   shrinks n by `_shrink_batch` (0.75, `memory.py:75-92`) until a call runs and
   sets converged. `_measure_call` synchronizes the GPU after each call
   (1509-1512) and returns a time and a memory reading that
   `_initialize_max_batch` ignores (it tests only `is None`).
3. **Growth after the start.** `_batch_for(n_waiting)` (`memory.py:1576-1597`)
   returns `min(n_waiting, 10 * _batch)` while `_batch_converged` is False.
   `_evaluate_f_on_mesh` (`base.py:2306-2311`) sizes the next call with it
   (at least one subinterval, C points, with `take_gradient=True`).
   `_evaluate_f_in_batches` (`base.py:2454-2536`) raises `_batch` to every call
   that ran (2531) and shrinks the failed call on an out-of-memory error (2509).
   `_evaluate_f_on_full_nodes` (`base.py:2387-2427`) does the same for
   `take_gradient=True`, plus the check before a backward pass on a GPU.
4. **Consequence for the first pass today.** With the test calls, `_batch` is
   the largest power of ten not above min(n_first, 1000), so `10 * _batch`
   exceeds n_first whenever n_first < 10,000: the first call of the first pass
   is the whole pass. For n_first ≥ 10,000 (a mesh passed in without a state)
   the first call is 10,000 points (9,982 with gk21 and `take_gradient=True`,
   the whole subintervals that fit), and growth continues from there.
5. **Every call of f in the loop** goes through `_call_f`
   (`memory.py:747-801`). The number of rows is checked only on the
   `take_gradient=False` path (`base.py:2521-2526`). `get_max_batch(f=...)`
   makes its own one-point call with its own copy of the two-dimension check
   (`memory.py:641-649`).

### 2.2 Why the test calls exist (recovered from git history)

- The design (`git show 6b3cf42:docs/issues/2026-11-17-integration-state-design.md`,
  sections 1 and 5.2) removed the memory benchmark from `integrate`. The owner
  asked to keep test calls when no `max_batch` or state is given ("If
  max_batch is not given and no state exists, then it needs to run the test
  calls"), capped "at the same order of the first pass". The design's reading,
  marked as the author's: they "only check that `f` runs" at those sizes.
- The memory-benchmark brainstorm (same commit,
  `2026-11-17-memory-benchmark-speed-brainstorm.md`, option D, item 2) gives
  the motive: the old benchmark never made a call more than 10 times larger
  than one that ran, and "two tiny test calls of 10 and 100 points before the
  first pass would keep today's caution". Item 5 adds that on a CPU an
  allocation that does not fit can get the process killed without a Python
  error, so growth there "must be predicted, never tried".
- The critique (`2026-11-17-integration-state-critique.md`, F9) found that on
  a GPU the test calls protect against nothing the first real call does not
  also find, since both paths retry after an out-of-memory error; they matter
  only on a CPU, where the design accepts the risk anyway. It measured their
  cost (0.01 to 3.37 times a small integral on a CPU, 0.01 to 0.34 on an A100)
  and left the choice to the owner: keep them, or "let the first pass be the
  first test". The plan (`2026-11-18-memory-state-plan.md`, decision 11) kept
  them.
- The break-and-speed review (`git show 59b7f11:docs/issues/2026-11-18-memory-state-break-and-speed.md`,
  speed-up 2) measured cold calls from large starting meshes 10 to 32 % faster
  without test calls, proposed starting at the first pass itself, and noted
  that this is the owner's decision. The commit message records it as "not
  done, against the owner's decisions".
- The bug hunt (`git show d0bcbdc:docs/issues/2026-11-18-memory-state-bughunt.md`)
  only checked that the test calls work with both solvers and dtypes.

So removing the test calls reverses an explicit owner decision; it is
decision D1 below.

### 2.3 Measured counts (exact; CPU, login node)

Script `run/count_startup.py` (integrand `1.3 exp(-t) sin(40 t)` and a
1-64-1 tanh network, atol 1e-10, rtol 1e-8, float64; outputs
`run/counts_before.txt`, `run/counts_after_v1.txt`). "Used" is C times the
subintervals returned by `_evaluate_f_on_mesh`, the points that belong to the
integration. The counts are the same for `take_gradient` True and False and
for both integrands.

| Case | Points evaluated today | Calls today | Extra points today | Points after | Calls after |
|---|---|---|---|---|---|
| `max_batch` given | used + 1 | +1 | 1 | used | −1 |
| `state` given | used + 1 | +1 | 1 | used | −1 |
| neither, n_first < 10 (bosh3, 2 subintervals, 8 points) | used + 1 | +1 | 1 | used | −1 |
| neither, 10 ≤ n_first < 100 (gk21, 3 subintervals, 69) | used + 11 | +2 | 11 | used | −2 |
| neither, 100 ≤ n_first < 1,000 (gk21 default mesh, 253) | used + 111 | +3 | 111 | used | −3 |
| neither, n_first ≥ 1,000 (gk21, 50, 500 and 2,000 subintervals) | used + 1,111 | +4 | 1,111 | used | −4 |

Examples: the default gk21 call of a smooth integrand evaluates 364 points
for 253 used (4 calls; after: 253, 1 call); a call with `state=` 139 for 138
(after: 138); a mesh of 500 subintervals 12,611 for 11,500 (6 calls; after:
11,500, 2 calls).

### 2.4 Measured times (A100, GPU 1 of nid008404, shared)

Scripts `run/time_startup.py` and `run/time_heavy.py`, a fresh solver per
call, gk21, no `max_batch`; before and after alternated twice
(`run/gpu_*.txt`, `run/heavy_*.txt`). Medians in ms.

| Integrand, start | Before (two rounds) | After (two rounds) |
|---|---|---|
| `exp(-t) sin(40 t)`, default mesh | 6.28, 5.62 | 5.85, 5.90 |
| 3×512 network, no grad, default mesh | 6.67, 6.03 | 5.79, 6.02 |
| 3×512 network, `take_gradient=True`, default mesh | 8.70, 8.42 | 7.38, 8.00 |
| `exp(-t) sin(40 t)`, `state=` | 5.62, 5.12 | 5.62, 5.71 |
| 6×2048 network, no grad, default mesh | 7.35, 7.84 | 6.07, 6.01 |
| 6×2048 network, no grad, 500 subintervals | 39.86, 38.77 | 35.72, 34.78 |

Only the 6×2048 rows separate from the noise (the two "before" rounds of the
cheap rows differ by up to 0.7 ms). The integrals and `state.max_batch`
printed by these runs are identical before and after.

## 3. The design

### 3.1 What changes

1. **The starting batch size** (`base.py:1172-1179`). Replace the call of
   `_initialize_max_batch` by

   ```python
   n_first = int(mesh_trackers.sum()) * self.C
   # No test calls: the first call of f is the first pass, at most
   # _LARGEST_FIRST_CALL points (_batch_for allows _GROWTH times this size).
   self._batch = -(-min(n_first, _LARGEST_FIRST_CALL) // _GROWTH)
   self._batch_converged = False
   ```

   with `_LARGEST_FIRST_CALL = 10_000` replacing `_LARGEST_TEST_CALL = 1000`
   in `memory.py:69-72`. Why this is the same call as today: `_batch_for`
   gives `min(n_waiting, 10 * _batch)`, and `10 * ceil(m / 10)` lies in
   `[m, m + 9]`, so the first call is `min(n_first, 10,000)` points, which is
   what the test calls lead to when none runs out of memory (section 2.1,
   item 4). After that call `_batch` is raised to its size (`base.py:2427`,
   2531), as today. Measured: the 32 cases of section 2.3 and the
   out-of-memory cases of section 6.3 give the same calls and numbers with
   this expression as with the literal "largest power of ten" value.
2. **`_initialize_max_batch` is deleted** (`memory.py:1599-1649`).
   `_measure_call` stays, used by `_setup_memory_checks` only.
3. **The shape check call is deleted** (`base.py:1113-1150`). Its three uses
   move:
   - *Output dimensions:* into `_call_f` (`memory.py:747`), right after a call
     that did not run out of memory: raise the same `ValueError` when the
     output is not a tensor of at least two dimensions. Move the row-count
     check of `_evaluate_f_in_batches` (`base.py:2521-2526`) there too, so both
     output checks live in the one function every call of `f` in the loop goes
     through, and the `take_gradient=True` path gets the row check it lacks
     today. Both checks read only shapes (no device read).
   - *`error_integral_reference` shape* (corrected by the review, item 5):
     inside the existing `if error_integral_reference is not None:` of Step 4
     (`base.py:1222`), D from `current_integral.shape[-1]`, same message.
   - *The `trade_speed_for_memory` warning* (corrected by the review, item
     5): after the loop, when `state is None and trade_speed_for_memory and
     record["integral"].requires_grad` (decision D3). The plan's once-run
     block with a `first_turn` flag is not used.
4. **Warming f up** is dropped with nothing in its place. It protected the
   timing and memory readings of the test calls (from the era when they were
   the memory benchmark); `_initialize_max_batch` ignores both. The loop never
   times `f`. `get_max_batch(f=...)` keeps its own one-point warm-up
   (`memory.py:641-649`) before `_setup_memory_checks`, so the benchmark is
   unaffected. The one place in the loop that reads memory around a call is
   the check before a backward pass (`base.py:2398-2418`, GPU,
   `take_gradient=True`, output requiring grad): its first batch now also
   counts one-time allocations of `f` (section 9).

### 3.2 Why this is the least new code

It deletes `_initialize_max_batch` (51 lines), the shape-check block (38
lines) and one constant, and adds one expression, a once-run block of about 15
lines (the same checks, moved), and about 6 lines in `_call_f` (of which the
row check is moved, not new). No new function, attribute or parameter. The
batch size keeps its one meaning and its one place (`_batch`, `_batch_for`).

### 3.3 Alternatives considered

| Alternative | Extra points | Extra calls | Returned numbers vs today | Why not recommended |
|---|---|---|---|---|
| **A (recommended):** first call = first pass, at most 10,000 points | 0 | 0 | identical (no out-of-memory error, `f` not drawing random numbers) | — |
| B: first call = whole first pass, no cap (`-(-n_first // _GROWTH)`; the break review's proposal) | 0 | 0 | for n_first > 10,000: `state.max_batch` changes (11,500 instead of 10,000); with `take_gradient=True` the first pass is one batch instead of several, so the integral, error and gradient change (measured: last bits on smooth integrands; the critique's F3 measured up to 166 tolerances on hard ones with bosh3/dopri5, in the direction of an unlimited batch); a network `f` changes in the last bits because cuBLAS rows depend on the batch shape (measured on the A100: 0.018827924127638347 → …344) | Changes numbers on large starting meshes; removes the 10,000-point cap on a first call nothing has tested, which matters on a CPU. A separate decision (D2). |
| C: grow inside the first pass from 1 (`_batch = 1`): real nodes in calls of 10, 100, … | 0 | +2 for the default mesh (10, 100, 143) | `state.max_batch` changes (143 instead of 253, measured), which changes later calls; a network `f` changes at rounding level (measured on CPU: a 69-point pass in calls of 10 and 59 changed the contributions' last bits); the first pass loses the one-call path that keeps f's output without a copy (`base.py:2690-2694`) | Keeps the "never 10 times what ran" caution, but only for `take_gradient=False`: with `take_gradient=True` each call is one batch judged against a partial running integral (section 3.4), so this would need two rules. More calls and changed numbers for a caution that only a CPU needs. |
| D: keep the test calls, remove only the shape-check call | 0 to 1,110 | 0 to 3 | identical | Keeps 98 % of the waste on default calls (110 of 111 points). The fallback if the owner keeps decision 11. |
| E: test calls on a CPU only | as today on a CPU | as today on a CPU | identical on a GPU | Device-dependent behaviour for a risk the user guide already accepts and documents ("Growth needs a clean failure", `devices_memory.md:85-90`). |
| F: keep a one-point call only when `error_integral_reference` is given | 1 with that argument | 1 | identical | Keeps the early error for a wrong reference shape, at one call per `integrate` for the users who pass it, who are mostly training with `take_gradient=True` (`tests/test_error_integral_reference.py`). Decision D4. |

### 3.4 Starting small on the real nodes (task question 1)

- **`take_gradient=False`.** The batch size only cuts a pass into calls of
  `f`; a call may end inside a subinterval (`base.py:2650-2748`). Starting
  small therefore costs extra calls in the first pass but no extra points, and
  the subintervals, their nodes and the accept-or-split decisions are the same.
  Returned numbers are the same bit for bit when each output row of `f` does
  not depend on how many rows the call had (true for elementwise `f`;
  `tests/test_trade_speed_for_memory.py::test_refinement_does_not_depend_on_max_batch`
  pins it), and differ in the last bits otherwise (matrix-multiply kernels
  choose their blocking from the batch shape; measured for a network on CPU
  and GPU). `state.max_batch` changes because it is the largest call that ran.
- **`take_gradient=True`.** Each batch is a whole number of subintervals and
  one turn of the loop: evaluate, judge, record, backward
  (`base.py:1214-1545`). A batch is judged against the running integral of the
  record plus itself (1240-1261, 1275), so in the absolute mode with rtol > 0
  the tolerance `max(atol, rtol |I_running|)` of the first batches is set by a
  partial integral; in the cumulative mode the running sums are partial too.
  The re-check of accepted subintervals that corrects this is off with
  `take_gradient=True` (`enforce_convergence`, 1309-1311). So smaller first
  batches change which subintervals are split, hence the mesh, the integral,
  its error, `converged`, the gradient, and the evaluation count; with
  rtol = 0 in the absolute mode only the order of the sums changes (last bits
  of the integral and gradient). Each extra batch also adds a backward pass
  and its host overhead (the break review measured cold gradient calls from a
  10-point start 30 to 50 % slower: 9.6 → 14.3 ms and 12.1 → 16.5 ms).
- **Options**, therefore: A (one first call covering the first pass up to
  10,000 points, the existing retry shrinking it on an out-of-memory error;
  the same for both values of `take_gradient`), B (the same without the cap),
  C restricted to `take_gradient=False` (two rules), D (keep the test calls).
  A is recommended because it is the only one that changes no number.

### 3.5 What the test calls protect that a failed first call does not (task question 2)

Let t(N) be the time of one call of `f` on N points and n0 = min(n_first,
10,000) the first call.

- **No out-of-memory error** (the common case: the default first pass is at
  most 11 C points, 253 for gk21, 715 for cc65). Today every call without
  `max_batch` or state pays t(1) + t(10) + t(100) [+ t(1000)], plus a GPU
  synchronization after each test call (`memory.py:1509-1512`); every call
  with a state or `max_batch` pays t(1). Design A pays nothing. For a cheap
  `f` that is a few launches per call (inside the noise in section 2.4); for
  an expensive `f` it is 1,110 points per cold call, or 44 % on top of a
  one-pass default call (111/253), and on a GPU a launch-bound `f` (small
  batches) costs about as much per call as a full pass, so even the one-point
  call of every training step is a full call's latency (my reasoning; the
  3×512 `state=` rows of section 2.4 are inside the noise).
- **`f` does not fit n0 points.** Both designs shrink by 0.75 per failed call
  until one fits. Today the chain starts at the first test call that fails
  (at most 10 times one that ran, so at most 8 failures); in design A it
  starts at n0. Simulated on CPU (`run/count_oom.py`, an `f` that raises
  `torch.OutOfMemoryError` above a limit; `run/oom_before.txt`,
  `run/oom_after_v1.txt`):

  | Limit, start | Today: failed calls / failed points / all points / state | A: failed calls / failed points / all points / state |
  |---|---|---|
  | 60 points, default mesh, `take_gradient=False` | 2 / 175 / 495 / (56, converged) | 5 / 771 / 1,024 / (60, converged) |
  | 60 points, default mesh, `take_gradient=True` | 2 / 175 / 495 / (56, converged) | 5 / 736 / 989 / (46, converged) |
  | 500 points, 50 subintervals, False | 3 / 2,312 / 3,995 / (422, converged) | 3 / 2,658 / 3,808 / (484, converged) |
  | 5,000 points, 500 subintervals, False | 3 / 23,125 / 35,736 / (4,219, converged) | 3 / 23,125 / 34,625 / (4,219, converged) |

  A failed call costs between almost nothing (an allocation refused at once,
  5 ms) and almost a full call (a call slightly too large runs nearly to its
  end; 1.5 to 2.6 s on an A100 for a network filling the GPU, critique F4;
  with PyTorch's cache kept after the error, as the code does now since
  `59b7f11`, such calls failed in 3 to 21 ms in the break review's speed-up
  1). So the test calls save failed calls only when `f` cannot hold a first
  pass of a few hundred points, which on an 80 GB GPU means more than about
  300 MB per point for the default gk21 mesh (my arithmetic: 75 GB / 253).
  The cost then comes once per state: the batch size converges, and a call
  given the state never retries. A user who calls without a state each time
  pays it each time, in both designs.
- **On a CPU** an allocation that does not fit can get the process killed
  instead of raising (`devices_memory.md:85-90`, accepted by the owner,
  design section 9). Design A's first call is at most 10,000 points with no
  smaller call before it; today it is at most 10,000 points after a call of
  1,000 ran. Later growth (10 times per call until failure) is unchanged and
  carries the same risk in both designs.
- **Expected cost** (my estimate): with p the probability that the first pass
  does not fit, design A saves (1 − p) × [t(1) + t(10) + t(100) (+ t(1000))]
  per cold call and costs p × (extra failed calls, at most about
  log(n0/limit)/log(4/3) − 2 more) once per state. For any `f` whose first
  pass fits on the device, p = 0 and A is strictly cheaper.

### 3.6 The shape check (task question 3)

Between `base.py:1118` and the first real call (`base.py:1227`), only the
three checks of 1119-1149 read the output; nothing else needs D
(`_setup_initial_mesh`, the batch start, the saved gradients, the record and
`split_counts` do not). After the change:

| Check | Today | After |
|---|---|---|
| Output has ≥ 2 dimensions | before the mesh is built, on 1 point | in `_call_f`, on the first call of the first pass, before anything is integrated; same message |
| Output has one row per point | `take_gradient=False` only, in `_evaluate_f_in_batches` | in `_call_f`, both paths |
| `error_integral_reference` shape `[]`, `[1]` or `[D]` | before the mesh is built | after the first `_evaluate_f_on_mesh`, before the reference is used; same message. With `take_gradient=True` no backward pass has run yet, so no gradient is written |
| `trade_speed_for_memory` warning when the output requires grad | before any evaluation | after the first `_evaluate_f_on_mesh`; still once, still only when `state is None` |
| Warm-up | the one-point call | none needed (section 3.1, item 4) |

The cost of the moved reference check: a wrong shape is reported after the
first pass's evaluations instead of after one point (a slower error for an
expensive `f`; the run fails either way). Decision D4.

## 4. Decisions for the user

- **D1. Remove the test calls (reverses plan decision 11 of 2026-11-18).**
  Recommendation: yes, with design A. Cost: on a CPU the first call (at most
  10,000 points) is no longer preceded by a smaller one; an `f` that cannot
  hold its first pass takes up to about 3 more failed calls once per state
  (section 3.5). Alternative: D (keep them, drop only the one-point call).
- **D2. Cap the first call at 10,000 points (A) or take the whole first pass
  (B).** Recommendation: A now, since it changes no number. B can follow as
  its own change: it makes cold `take_gradient=True` calls from large meshes
  agree with an unlimited batch (in my runs B's `take_gradient=True` result
  for 500 subintervals equals the `take_gradient=False` result bit for bit),
  at the cost of changed numbers there and an uncapped first call.
- **D3. Condition of the `trade_speed_for_memory` warning.** Recommendation:
  `method_output.integral.requires_grad`. It differs from today's test only
  for an integrand whose output does not depend on t (or not differentiably)
  while a bound requires grad: the integral then requires grad through the
  widths and the warning appears, although `f` saved nothing. The exact
  alternative needs the output's flag passed out of `_evaluate_f_on_split_nodes`
  (one more returned value or attribute) for one warning; not worth it in my
  judgement.
- **D4. When a wrong `error_integral_reference` shape is reported.**
  Recommendation: after the first pass (no extra evaluation). Alternative F:
  keep a one-point call only when the argument is given.
- **D5. Merge `get_max_batch`'s copy of the dimension check**
  (`memory.py:645-649`) by letting its one-point call go through `_call_f`
  (with `can_shrink=False`). Recommendation: yes (one source of truth for the
  check and its message). Cost: a one-point out-of-memory error there is then
  raised as `torch.OutOfMemoryError` with `_call_f`'s added sentence ("the
  batch size cannot be reduced") instead of bare.
- **D6. Changelog.** 2.0.0 has no tag yet (`git tag` ends at `v1.4.1`), and
  the design edited the `[2.0.0]` section in place. Recommendation: edit the
  "`integrate` no longer runs the memory benchmark" entry (`CHANGELOG.md:37`)
  in place, and add one line about stochastic integrands (section 6.3).

## 5. Implementation steps

1. **Tests first** (they fail before the change).
   - Add `test_every_point_of_f_belongs_to_the_integration` to
     `tests/test_memory_state.py` (section 6.1).
   - Rewrite the tests that pin the old calls (section 6.2).
   - Run them on the CPU and see them fail for the expected reason (the extra
     1, 10, 100, 1,000-point calls).
2. **`_call_f`** (`padaquad/quadrature/memory.py:747`). After a call that
   did not run out of memory, raise `ValueError` if the integrand output is
   not a tensor with at least two dimensions (message of `base.py:1120-1123`),
   or if its length differs from `len(points)` (message of
   `base.py:2522-2526`, moved). Keep the checks outside the `try` so a
   `ValueError` is never taken for an out-of-memory error. Update the
   docstring (`Raises`). Delete `base.py:2521-2526`. Test:
   `tests/test_bug_regressions.py::test_integrand_output_must_be_two_dimensional`
   for both values of `take_gradient`, and a new row-count case for
   `take_gradient=True`.
3. **`integrate`** (`padaquad/quadrature/base.py`), first commit (the
   one-point call):
   - Delete the shape check (1051-1096) but keep the probe for the test calls.
   - The reference check into Step 4's existing reference branch (1222); the
     warning after the loop (review, item 5).
   Second commit (D1, the test calls):
   - Replace the `else` branch of 1113-1125 by the expression of section 3.1
     with `n_first = n_start_steps * self.C`; delete the probe; update the
     comment at 1110-1112.
   - Tests: `tests/test_recheck_accepted_panels.py::test_reference_shape_is_validated`,
     `tests/test_trade_speed_for_memory.py::test_warning_on_first_call_only`
     and `test_no_warning`, unchanged.
4. **`memory.py`.** Delete `_initialize_max_batch` (1599-1649). Replace
   `_LARGEST_TEST_CALL` (69-72) by `_LARGEST_FIRST_CALL = 10_000` with a
   comment ("the largest first call of `integrate` given no `max_batch` and no
   state; larger calls are reached by growth"). Import it in `base.py` next to
   `_shrink_batch` (base.py:68). Update the `_batch_for` docstring if it
   mentions test calls. D5 if accepted: route `memory.py:644` through
   `_call_f`.
5. **Checks.** `ruff check padaquad/`, `mypy padaquad/` (must report no
   issues), `numpydoc lint $(git ls-files 'padaquad/*.py')`.
6. **Bit-identity evidence** (section 6.3): the count script before and
   after; the grid harness, updated; the snapshot tests.
7. **Whole suite** on a compute node (CPU-pinned and CUDA runs, as in the
   memory notes), comparing the failing set with the baseline branch.
8. **Documentation** (section 8), then `sphinx-build -W`.

## 6. Tests

### 6.1 New: every evaluated point belongs to the integration

In `tests/test_memory_state.py`, parametrized over
`start ∈ {max_batch=10**6, state from a first call, neither with a mesh of 1, 3, 11 (default), 50, 500 subintervals}`,
`take_gradient ∈ {False, True}`, and the solvers `("uniform", "gk21")`,
`("uniform", "bosh3")`, `("variable", "interpolatory3_variable")`:

- wrap `f` to sum `len(t)` over its calls;
- wrap `solver._evaluate_f_on_mesh` (as `_spy_evaluated` does in
  `tests/test_recheck_accepted_panels.py:590-601`) to sum `len(nodes)`;
- assert `points == solver.C * subintervals`, and that the first call of `f`
  has `min(n_first, 10_000)` points (`10_000 // C * C` with
  `take_gradient=True`).

An out-of-memory variant (the `Recorder(limit=...)` of
`tests/test_memory_state.py:43-54`) asserts the same equality for the calls
that ran (failed calls excluded) and pins the retry chain of the first call
(e.g. limit 60 on the default mesh: 253, 190, 142, 106, 80, then 60).

### 6.2 Tests to change (they pin the removed calls)

The prototype (design A, the moved checks; `proto/`) was run on the login
node's CPU against ten affected files (`run/pt_<file>.txt`). Passed unchanged:
`test_snapshots.py` (390), `test_trade_speed_for_memory.py` (311, including
both warning tests), `test_oom_recovery.py` (159),
`test_memory_state_adversarial.py` (21), `test_memory_state_break.py` (14),
`test_error_integral_reference.py` (5). Failed, all because they pin the
removed calls (23 tests):

- `tests/test_memory_state.py` (8):
  `test_test_calls_stop_at_1000_or_the_order_of_the_first_pass` (3 cases)
  becomes "the first call is the first pass, at most 10,000 points" (no
  `[1, *test_calls]` prefix); `test_failed_test_call_shrinks_until_one_runs`
  becomes the first-call retry chain (253, 190, 142, 106, 80, then 60; state
  (60, True)); `test_batch_grows_tenfold_per_call_until_out_of_memory` (2),
  `test_given_max_batch_is_fixed_and_converged` and
  `test_state_carries_mesh_and_batch_size_to_the_next_call` index `f.sizes`
  from 1 (after the one-point call) and move to 0; the module docstring.
- `tests/test_memory_benchmark.py::test_out_of_memory_retry_shrinks_the_failed_batch` (1):
  `[1, 44, 33, 25, 19]` → `[44, 33, 25, 19]`, and its `stub_calls` (which only
  stubs the test calls) goes.
- `tests/test_recheck_accepted_panels.py::test_switch_never_evaluates_a_subinterval_twice`
  (12): `1 + solver.C * len(seen)` → `solver.C * len(seen)` (the invariant of
  6.1).
- `tests/test_bug_regressions.py::test_integrand_test_call_shape` (2) is deleted:
  the call it pins (one point shaped `[1, T]`) no longer exists; the loop's
  calls are `[N, T]` by construction (`base.py:2393`, 2622). The docstring of
  the test at 77-88 ("test calls, then growth") is updated.
- Passing, wording only: the comment at `tests/test_oom_recovery.py:373`
  (the one-point warm-up call), and
  `tests/test_memory_state_adversarial.py::test_test_calls_with_both_solvers_and_dtypes`,
  renamed to what it still checks (the first call with both solvers and
  dtypes).
- Not run against the prototype: the rest of the suite. Step 7 runs it all.
- Tests of `get_max_batch(f=...)` (`tests/test_memory_benchmark.py`) keep
  their leading `1` (its own one-point call stays).

### 6.3 Showing that results are unchanged, and where they change

1. **Count-and-hash script** (`run/count_startup.py`, to be kept with the
   change, for example under `examples/verification/` only if the owner
   wants it there; otherwise run from scratch): prints, for 32 cases, the
   calls, points and the hex of every returned number. Before and after must
   differ only in calls and points. Done on the prototype: 0 of 32 rows differ
   (`run/counts_before.txt`, `run/counts_after_v1.txt`).
2. **Snapshot tests** (`tests/test_snapshots.py`) must pass unchanged. They
   pass `max_batch` (`tests/_helpers.py:174-175`), so they cover the removed
   shape check but not the new start.
3. **Grid harness** (`/pscratch/sd/k/khegazy/padaquad_methodclass_verify/grid_run.py`,
   `grid_run_gpu.py`). It predates the current API and must be updated before
   use: unpack `result, state = solver.integrate(...)`; `mesh_optimal` →
   `state.mesh`; `same_f=True` → `state=state`; `conserve_memory` →
   `trade_speed_for_memory`. Every case passes `max_batch` (lines 76, 84-89,
   113), so it never runs the start this plan changes: add variants without
   `max_batch` (default mesh, and a mesh of 500 subintervals) for both
   `take_gradient` values, CPU and CUDA.
4. **Where numbers change** (state in the CHANGELOG):
   - an out-of-memory error in the first call: the retry chain starts at the
     first call instead of at a test call, so `state.max_batch` can differ
     (60 instead of 56 in the simulation above), and with `take_gradient=True`
     the batches differ, so the integral, error and gradient can differ;
   - an `f` that draws from PyTorch's global random number generator (dropout,
     noise): the shape check advanced the generator before the random starting
     mesh was drawn, so the starting mesh changes and every number changes
     within the tolerance (measured: first interior mesh point 0.0730 →
     0.1359; `run/rng_before.txt`, `run/rng_after_v1.txt`). With
     `random_initial_mesh=False` nothing changes. The test calls also
     advanced the generator before the first pass. After the change the mesh
     depends only on the seed the user set, not on `f`;
   - a stateful `f` (counters, BatchNorm running statistics in training
     mode) sees 1 to 4 fewer calls, and no longer sees 1,000 copies of the
     lower bound.

## 7. How to measure the gain

- **Counts (login node, CPU, exact):** run
  `PYTHONPATH=<checkout> python run/count_startup.py` before and after; the
  "extra" column must be 0 in every row, and the hashes equal.
- **Time (compute node, one GPU):** `run/time_heavy.py` (6×2048 network,
  default mesh and 500 subintervals) and `run/time_startup.py` (cheap and
  3×512 integrands, with and without `state`), alternating before and after
  at least twice. Report medians and the spread between rounds. Expected (from
  section 2.4): 15 to 25 % for an expensive `f` on the default mesh, about
  10 % from a mesh of 500 subintervals, nothing measurable for a cheap `f`.
- The brief's baseline (`profile_baseline.py`) passes `max_batch=1_000_000`,
  so it sees only the one-point call; add a run without `max_batch` if the
  coordinator's tables are redone.

## 8. Documentation to update

- `docs/user_guide/devices_memory.md:24-36` ("How `integrate` chooses it"):
  replace the test-call sentences by "With neither, the first call of the
  integrand evaluates the first pass (at most 10,000 points); the batch size
  grows from there." Line 35-36 ("runs whole when it is at most 10 times the
  batch size") stays true. Add to the limitations (85-90) that the first call
  is not preceded by a smaller one (D1).
- `CHANGELOG.md:37` (D6): the same change; plus "Every point at which
  `integrate` evaluates the integrand now belongs to the integration: the
  one-point output check and the test calls are gone. A stochastic integrand
  that draws from PyTorch's random number generator gets a different random
  starting mesh than before."
- `CLAUDE.md`, "Memory management", first bullet: remove
  `_initialize_max_batch` and `_LARGEST_TEST_CALL`; describe the start
  (`_LARGEST_FIRST_CALL`). "Core algorithm" step 2 is unchanged. The test
  table: `test_memory_state.py` is not listed; add it with the new invariant
  test, and update the row of `test_bug_regressions.py` if it mentions the
  shape-check call.
- Docstrings: `integrate`, parameter `max_batch` (`base.py:803-809`, "f is
  called on 10, 100 and 1,000 points ..."); `_call_f` (new `Raises`);
  `_evaluate_f_in_batches` (the row check moved out); `padaquad.integrate`
  in `_frontend.py` defers to the solver's documentation (checked: it does not
  repeat the text).

## 9. Risks and interactions with the other plans

- **Risks.**
  - The first batch with `take_gradient=True` on a GPU now includes `f`'s
    one-time allocations (lazy weights, cuBLAS workspace) in its forward peak
    (`base.py:2398-2410`). That makes the check before the first backward pass
    more cautious, and the measured backward-to-forward ratio of the first
    batch smaller, so the second batch's check less cautious; a backward
    out-of-memory error is still caught by the restart. Training loops pass
    `state`, and `f` is warm after their first call, so this affects one
    batch per process (my reasoning, not measured).
  - A wrong `error_integral_reference` shape or a bad output is reported
    after one call of first-pass size instead of one point.
  - The checks in `_call_f` run on every call of `f`; they read shapes only
    (no device read), so they add no wait.
- **`call-setup-cost`:** owns the other fixed costs of a call. This plan
  deletes code in the same region (`base.py:1113-1179`); whoever lands second
  rebases. No shared function changes.
- **`endpoint-evaluations`:** may change C or the nodes per subinterval; the
  invariant test of 6.1 should then count the points `f` really receives per
  subinterval (whatever that plan defines), so the two plans must agree on
  the test's expected count.
- **`split-waste`:** may change how many subintervals a pass evaluates; it
  does not interact with the start, and the invariant of 6.1 holds for any
  mesh update.
- **`turn-device-reads`:** touches `_evaluate_f_on_mesh` and
  `_evaluate_f_in_batches`; this plan moves the row check out of the latter
  into `_call_f`. Land in either order; resolve the textual conflict.
- **`compiled-turn`:** a once-run block at the top of the loop is plain
  Python on the first turn only; it reads shapes and one `requires_grad`
  flag, no device values.

## 10. Confidence

- **Sure (measured or read in the code):** the call sites and what each
  check needs; the exact counts of section 2.3; that design A makes the same
  calls of `f` as today when no out-of-memory error occurs (shown by the
  `_batch_for` arithmetic and by 32 cases with identical hashes of every
  returned number); the out-of-memory chains of section 3.5 (simulated); that
  a network `f` changes in the last bits when its call sizes change (measured
  on CPU and GPU), which is why B and C change numbers; the change of the
  random starting mesh for an `f` that draws random numbers.
- **Less sure:** the time gains, measured on a GPU shared with other agents
  (two rounds each; the cheap rows are inside the noise); how often a real
  `f` fails to fit its first pass (no data; the estimate of section 3.5 is
  mine); the effect on the first backward check (reasoning only); the
  complete list of tests to change: ten files were run against the prototype
  (section 6.2), not the whole suite. The prototype is a quick hack (an
  environment variable picks the variant) and not the code to merge.
- **Not checked:** the variable-sampling solver and float32 in the count
  script (the invariant test of 6.1 covers them); `torch.distributed`.

## 11. Implemented (branch `startup_evaluations`, base `73caae8`)

| Step | Commit | What |
|---|---|---|
| Review | `f56d1c1` | this plan, reviewed (section "Review") |
| 2 (`_call_f`) | `eec8c98` | one check of the output in `_call_f` (≥ 2 dimensions and one row per point, one message), for every call of `f`; the row check of `_evaluate_f_in_batches` removed. Fixes `take_gradient=True`, where an `f` returning two rows per point was reshaped into an integral with two components without an error (checked at `73caae8`) |
| D5 | `62bd766` | `get_max_batch(f=...)`'s one-point call through `_call_f`; its copy of the check removed |
| 3 (one-point call) | `1183318`, `07cf507` | the shape-check call removed; the reference check moved into Step 4's reference branch; the `trade_speed_for_memory` warning moved after the loop (Review, item 5); `07cf507` only formats its condition |
| D1, 4 (test calls) | `2d969f1` | `_initialize_max_batch`, the probe and `_LARGEST_TEST_CALL` removed; the batch size starts at `ceil(min(n_first, _LARGEST_FIRST_CALL) / _GROWTH)`; tests and documentation |
| 8 (docs) | `ae66c63` | the failed calls of a first call far above the integrand's limit, in the user guide and the changelog |

**Evidence** (scripts and outputs in
`$SCRATCH/padaquad_speedup_plans/startup-evaluations/impl/`):

- *Bit for bit.* `startgrid.py` (248 integrations without `max_batch`: first
  passes of 8 and 9 points (bosh3, interpolatory3_variable), 10-99, 100-999,
  1,000-9,999 and 10,500 to 42,000 points; gk21, cc17, bosh3,
  interpolatory3_variable; an elementwise integrand with parameters and a
  tanh network; `take_gradient` True and False with the parameters'
  gradients; from the bounds or a mesh, from the returned state, with
  `max_batch`; `trade_speed_for_memory`, `record_evaluations=False`, float32)
  against `73caae8`: 0 differences in 3,294 tensors on the CPU and 0 on an
  A100, and the calls of `f` after the removed start-up calls are the same
  calls (160 cases dropped 1 point, 26 dropped 11, 26 dropped 111, 36 dropped
  1,111). The standard grid (`std_bitgrid.py`, 656 integrations with
  `max_batch`): 0 differences in 6,656 tensors on CPU and CUDA.
  `count_startup.py`: the 32 rows' returned numbers are identical; extra
  points 0 in every row (`counts_new.txt`).
- *Stochastic integrand* (`rng_check.py`): the first interior point of the
  random starting mesh after `torch.manual_seed(0)` moves from 0.0730 to
  0.1359 for an `f` that draws from the global generator; unchanged with
  `random_initial_mesh=False` and for a deterministic `f`.
- *Out of memory* (`oom_quant.py`, simulated limit, CPU, `oom_quant_*.txt`).
  Points evaluated / lost in failed calls / failed calls, base → branch:

  | Start, limit | `take_gradient=False` | `take_gradient=True` | state after (base → branch) |
  |---|---|---|---|
  | default mesh (231), fits | 342 / 0 / 0 → 231 / 0 / 0 | same | (231, False) both |
  | default mesh, limit 150 | 746 / 404 / 2 → 635 / 404 / 2 | 741 / 399 / 2 → 630 / 399 / 2 | (130 or 126, True) both |
  | default mesh, limit 60 | 473 / 175 / 2 → 937 / 706 / 5 | 473 / 175 / 2 → 903 / 672 / 5 | False: 56 both; True: 56 → 42 |
  | 500 subintervals (10,500), fits | 11,611 / 0 / 0 → 10,500 / 0 / 0 | 11,611 → 10,500 | (10,000 or 9,996, False) both |
  | 500 subintervals, limit 5,000 | 34,736 / 23,125 / 3 → 33,625 / 23,125 / 3 | 34,711 → 33,600 (3 failed both) | same |
  | 500 subintervals, limit 150 | 14,210 / 3,465 / 7 → 49,964 / 39,464 / 15 | 14,210 / 3,465 / 7 → 49,560 / 39,060 / 14 | 134 both; True: 134 → 147 |
  | 500 subintervals, limit 60 | 10,742 / 175 / 2 → 50,273 / 39,773 / 18 | 10,742 / 175 / 2 → 49,875 / 39,375 / 17 | 56 both; True: 56 → 42 |
  | state, limit 60 | 627 / 458 / 4 → 626 / 458 / 4 | 610 / 441 / 4 → 609 / 441 / 4 | same |

  So the plan's "up to about 3 more failed calls" (D1) holds for the default
  mesh, but not for a large first pass with an integrand that cannot hold
  1,000 points: each failed call shrinks by a quarter, so a first call of
  10,000 points takes about 8 more failed calls per factor of 10 between 1,000
  and the limit than the test calls did (16 more for a limit of 60). For a
  limit of 1,000 points or more nothing changes, since no test call failed.
  How much a failed call costs in time depends on where in `f` the
  allocation fails (a few ms when it fails at once; up to a full call). The
  retry chain always ends: with `take_gradient=True` at one subinterval (21
  points) and with `False` at one point, both raising
  `torch.OutOfMemoryError` ("cannot be reduced"). The state's batch size with
  `take_gradient=True` is now a whole number of subintervals (42 instead of
  56), the batch that ran.
- *Test suite* at `ae66c63` on nid200333 (CPU run with
  `CUDA_VISIBLE_DEVICES=""`, CUDA run on one A100): base `73caae8` 4,007
  passed, 339 skipped (CPU) and 4,321 passed, 25 skipped (CUDA), no failure;
  branch 4,146 passed, 339 skipped (CPU) and 4,460 passed, 25 skipped (CUDA),
  no failure. `sphinx-build -W` passes.
- *Time* (A100, GPU 0 of nid200333, nothing else running; base and branch
  alternated twice; medians in ms):

  | Integrand, start | Base (two rounds) | Branch (two rounds) |
  |---|---|---|
  | 6×2048 network, no grad, default mesh (342 → 231 points, 4 → 1 calls) | 6.98, 7.21 | 5.75, 5.71 |
  | 6×2048 network, no grad, 500 subintervals (11,611 → 10,500, 6 → 2) | 35.55, 35.51 | 33.98, 31.61 |
  | `exp(-t) sin(40 t)`, default mesh | 5.16, 5.18 | 4.91, 4.95 |
  | 3×512 network, no grad, default mesh | 5.67, 5.75 | 5.04, 5.08 |
  | 3×512 network, `take_gradient=True`, default mesh | 7.70, 7.92 | 7.10, 6.87 |
  | `exp(-t) sin(40 t)`, `state=` | 4.68, 4.69 | 4.61, 4.65 |
  | 3×512 network, no grad, `state=` | 4.93, 4.96 | 4.76, 4.87 |
  | 3×512 network, `take_gradient=True`, `state=` | 7.18, 7.06 | 6.80, 6.44 |

  The integrals printed are identical.

## 12. Not done

- A mitigation for a first call far above the integrand's limit (above):
  for example a first call capped at 1,000 points, or a faster shrink while
  no call has run. Each needs either changed numbers (smaller first batches
  with `take_gradient=True`) or a new branch in the retry; not done, reported
  for a decision.
- D2's alternative B (no 10,000-point cap) is not done, as decided.
