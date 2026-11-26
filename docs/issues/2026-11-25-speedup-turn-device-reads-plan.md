# Plan: fewer device reads per turn (`turn-device-reads`)

Status: plan written 2026-11-25 against `0590ccf`; reviewed and corrected
2026-11-26 against `73caae8` (section "Review"; sections 4 and 5 rewritten).
Items 5, 6 and 7 of
`per-turn-overhead.md` (branch `performance_benchmark`). Line numbers are of
commit `0590ccf` (`padaquad/quadrature/base.py` unless another file is named).
Code quality of the prototypes: prototype; of the planned change:
research-grade.

## Review (2026-11-26, against `73caae8`, branch `turn_device_reads`)

Checked every claim of this plan against the code at `73caae8` (the tip of
`speedups_integration`, after `call-setup-cost` and `endpoint-evaluations`)
and by measurements on an A100 (`nid200337`, torch 2.8.0; scripts and logs in
`$SCRATCH/padaquad_speedup_plans/turn-device-reads/impl/`). Line numbers in
the sections below are still those of `0590ccf`; the steps of section 5 name
functions, not lines. What was wrong or missing, and what changed:

1. **Re-measured base** (`count_turn.py`; gk21, `atol=1e-8`, `rtol=0`,
   `error_norm="max"`, float64, `max_batch=1e6`, [0, 1]): host waits /
   GPU launches per call are `damped_sine` 45 / 256 (1 turn), `chirp`
   648 / 3,739 (14 turns), `many_outputs` 93 / 520 (2 turns), `oscillatory`
   432 / 2,883 (13 turns). On `chirp` the mesh update makes 180 waits
   (12 per turn plus `_keep_rows`), the evaluation path 42, the loop condition
   15 and the loop's mesh-order check 28. `endpoint-evaluations` added the
   boolean index of `method_output.mesh_left` (one more wait per turn), which
   the integer indices of this plan absorb.
2. **The doubled warning of a lone boolean index (2.2) is the prototype
   notice.** `torch.cuda.set_sync_debug_mode("warn")` emits, once, the notice
   "Synchronization debug mode is a prototype feature and does not yet detect
   all synchronizing operations"; the planning script counted every message
   containing "synchron", and the notice contains "synchronizing". Counting
   only "called a synchronizing CUDA operation" (as `_host_waits` in
   `tests/test_base_solver.py` does), `x[mask]` makes one wait
   (`impl/sync_ops2.py`, `impl/logs/sync_ops2.log`). Also measured: indexing a
   CUDA tensor with a CPU mask makes two, and copying a CPU index to the GPU
   one.
3. **D2's guard as written was incomplete.** Applied only to the batch's
   masks, it misses two places where a subinterval is split: the re-check of
   recorded subintervals (`extra_split_idxs`, from `_failed_record_positions`)
   and the optimal mesh (`_get_optimal_mesh` splits every failing
   subinterval). A one-ULP subinterval accepted by the guard fails the
   re-check on the next turn and would be split there (a zero-width
   subinterval, then the same one again, without end), and the state's mesh
   would get a repeated point, which the next call rejects with `ValueError`
   ("mesh must be ... increasing"). The guard now lives in one helper,
   `_splittable`, which also holds the split cap that the mesh update and
   `_failed_record_positions` each wrote out before; both call it, and so
   does `_get_optimal_mesh` through the mesh update.
4. **D1 decided without a switch.** Of the per-turn internal checks, three
   become true by construction and are removed: the loop's mesh-order check
   (existing points move by the strictly increasing `idxs_transfer`; a
   midpoint lies in [left, right] because rounding is monotone, and differs
   from both by the D2 guard), the NaN check of the new mesh (it cannot see a
   gap in a `torch.empty` array; coverage follows from `split_idxs` being
   strictly ascending), and the node-order check of the `take_gradient=True`
   path (nodes are `_compute_nodes` of an increasing mesh at ascending
   fractions, checked when a method is built). The extras' ascending check
   stays under `runtime_checks`: it runs only on turns where the re-check
   split something and guards the disjointness of the record and the waiting
   set, which no construction here enforces. The record-order check in
   `_record_results` (3 waits per turn) belongs to `record-order`, which
   deletes it. The located `ValueError` for a NaN or Inf from `f` stays. So
   no module-level switch and no second code path for the tests.
5. **The legacy mask path and `error_ratios` go.** `_adaptively_increase_mesh`
   derived masks from `error_ratios` when none were given (a second copy of
   `_accept_reject_masks`'s rule, used only by tests and Step 7). It now takes
   `keep_mask` (required) and no `error_ratios` or `remove_mask`; it returns
   `keep_idx` in the slot of `error_ratios[keep_mask]`, and the loop filters
   the error ratios itself. This is the interface `record-order` asks for
   (its section 3.2): `keep_idx` (final, after the cap and the guard) in slot
   8, and `idxs_transfer` a local of that name for it to return in slot 7
   (not returned now: nothing reads it yet).
6. **Step 7's empty batch** costs one read (the plan skipped it with a branch
   on `len(keep_mask)`); it runs at most once per call with a fixed
   `error_integral_reference`, so the branch is not worth its code.
7. **`_accept_reject_masks` keeps its three return values**: its `remove`
   mask is still read by `_record_error_check`; the function belongs to
   `error-ratio-work`.
8. **The `compiled-turn` alternative** (a sorted tensor of waiting indices in
   place of `mesh_trackers`, 3.3) does not remove code: the mesh update would
   build the next waiting set (the batch's unevaluated tail, both halves of
   every split and the extras' halves, moved by `idxs_transfer`, then sorted)
   in about as many lines as the trackers take now, and it changes the
   signatures of `_setup_initial_mesh`, the mesh update, `_get_optimal_mesh`
   and Step 7 and about 40 test calls. It would save the loop's one
   `torch.nonzero` per turn (14 waits on `chirp`). Not done; left for
   `compiled-turn`.
9. **Vague steps rewritten**: section 5 now names, for each commit, the
   functions and the tests.

## 1. Summary

- **What is slow.** On a GPU, the CPU waits for the GPU (a *host wait*: the CPU
  stops until every queued GPU operation has finished, because it needs a value
  computed there) 44 times per turn on `chirp`, and launches 266 small GPU
  operations per turn. A *turn* is one pass of the `while` loop of `integrate`.
  The mesh update (`_adaptively_increase_mesh` with `_keep_rows`) makes 12 of
  these waits per turn, the evaluation path 3, and the loop condition 1.
- **What the plan changes.** (1) The mesh update turns its accept mask into
  integer indices once per turn, with one read, and uses integer indexing
  everywhere else; it allocates with `torch.empty`, drops three needless
  clones, and fills with `index_fill_` instead of assigning Python scalars.
  (2) The loop computes the waiting subintervals once per turn with one
  `torch.nonzero`, which gives the loop condition, their count and `step_idxs`.
  (3) The invariant checks of `runtime_checks` are split from the user-facing
  check of `f`'s output (a decision for the user, section 4).
- **Measured gain (prototype, A100, shared node, indicative).** Steps 1 and 2:
  `chirp` 619 → 425 waits and 3,731 → 3,069 launches per call; wall time
  median 62.4 → 51.6 ms (minimum 59.6 → 49.5 ms). Step 3, if chosen, removes
  about 5 more waits per turn (about 5 ms of `chirp`'s 52 ms).
- **Numbers.** Steps 1 and 2 change no returned number: measured bit for bit
  on three integrands (15 fields), and on a `take_gradient=True` call
  (integral and gradient). Step 3 changes no number either; it only changes
  which internal assertions run.

## 2. Evidence

### 2.1 The code as it is

`_adaptively_increase_mesh` (1905-2145):

- 2041: `split_counts[mesh_idxs]` with the cap; integer indexing, no wait (it
  did not appear in the counts; the cap was off in the baseline).
- 2044: `mesh_trackers[mesh_idxs[keep_mask]] = False`: two waits, one for the
  boolean mask (it runs `nonzero`) and one for assigning a Python scalar into
  an indexed CUDA tensor (measured below).
- 2053: `mesh_idxs[remove_mask]`: one wait.
- 2059-2060: the ascending check of the extra split positions
  (`runtime_checks`, only when the re-check found failures).
- 2061: `mesh_trackers[split_idxs] = True`: a Python scalar again (a wait
  when extras exist).
- 2067: `torch.nan * torch.ones(...)`: two launches, no wait.
- 2080: `idx_offset[split_idxs + 1] = 1`: one wait (Python scalar).
- 2083, 2084, 2102: `.clone()` of the right-hand side of three index
  assignments. The source and the destination are different tensors, so the
  copies are needless (each is one launch and one allocation).
- 2091-2092: `assert torch.sum(torch.isnan(mesh_new)) == 0` under
  `runtime_checks`: one wait.
- 2111, 2112-2114, 2115: three boolean-mask indexings of `method_output`
  fields: three waits.
- 2126: `bool(torch.all(keep_mask))`: one wait.
- 2128, 2131: `_keep_rows` (143-180), whose line 173 indexes with the mask
  (`not in_place`, or an `x` that requires grad or is a view): one wait; its
  in-place path runs `torch.nonzero` at 174: one wait.
- 2134: `nodes[keep_mask...]`: one wait.
- 2142: `self._get_mesh_indices(mesh_new)` (the dictionary; owned by
  `record-order`).
- 2143: `error_ratios[keep_mask]`: one wait.

Evaluation path:

- 1214: `while torch.any(mesh_trackers)`: one wait per turn and one at the end.
- 2306-2308 (`_evaluate_f_on_mesh`): `torch.sum(mesh_trackers) * self.C`, an
  assert on it, and `int(...)`: two waits (ungated).
- 2313-2314: `torch.arange(len(mesh))[mesh_trackers]`: one wait.
- 2230 (`_check_f_output_finite`, called at 2430 and 2527): `finite.all()`,
  one wait per call of `f` when `runtime_checks` is True.

Checks gated by `runtime_checks` (constructor default True, line 264;
per-call override, 1072):

| Where | What it checks | When | Waits |
|---|---|---|---|
| 1379-1386 | consecutive mesh points distinct; 1-D mesh monotonic | every turn | 2 |
| 2059-2060 | extra split positions strictly ascending | turns with re-check failures | 1 |
| 2091-2092 | no NaN left in the new mesh | every mesh update | 1 |
| 2183-2246 | `f`'s output finite; raises a located `ValueError` | every call of `f` | 1 |
| 2394-2397 | nodes ascending (`take_gradient=True` path) | every batch | 1 |
| 4374-4380 | record in strictly increasing mesh order | every turn | 2 (one inside `_mesh_order`) |
| 3177, 3606-3611, 4419-4430, `uniform.py:200`, `variable.py:200, 234` | ordering after merges, starting mesh, final sort | once per call | on `result_device` (CPU by default): CPU time, no GPU wait |

### 2.2 Which operations wait (PyTorch documentation and a measurement)

- "When input is on CUDA, `torch.nonzero()` causes host-device
  synchronization" (PyTorch 2.8 documentation, `torch.nonzero`,
  <https://docs.pytorch.org/docs/2.8/generated/torch.nonzero.html>).
- "By default, GPU operations are asynchronous", and "PyTorch automatically
  performs necessary synchronization when copying data between CPU and GPU"
  (PyTorch 2.8 documentation, CUDA semantics, section "Asynchronous
  execution", <https://docs.pytorch.org/docs/2.8/notes/cuda.html>). `.item()`,
  `bool()`, `int()` and `tolist()` of a CUDA tensor are such copies. The same
  page, section "CUDA Graphs", "Constraints", says "Ops that synchronize the
  CPU with the GPU (e.g., `.item()` calls) are prohibited" during graph
  capture, which is why the `compiled-turn` plan needs the reads gone.
- `torch.cuda.set_sync_debug_mode("warn")` warns on synchronizing operations;
  the documentation warns that "not all synchronizing operations will trigger
  warning or error" (PyTorch 2.8,
  <https://docs.pytorch.org/docs/2.8/generated/torch.cuda.set_sync_debug_mode.html>).
  The counts below are therefore lower bounds.
- `Tensor.index_fill_(dim, index, value)` takes `value` as a number (PyTorch
  2.8, <https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.index_fill_.html>),
  so no tensor is copied to the GPU for it.

I did not find a documentation page stating that boolean-mask indexing or the
assignment of a Python scalar waits, so I measured each operation alone
(torch 2.8.0+cu128, A100, `set_sync_debug_mode("warn")`; script
`$SCRATCH/padaquad_speedup_plans/turn-device-reads_scripts/sync_ops.py`):

| Operation on CUDA tensors | Warnings |
|---|---|
| `x[mask]` (boolean mask) | 2 |
| `torch.nonzero(mask)` | 1 |
| `torch.nonzero_static(mask, size=5)` | 0 |
| `x[int_idx]`, `x.index_select(0, int_idx)` | 0 |
| `b[int_idx] = True` (Python scalar) | 1 |
| `b.index_fill_(0, int_idx, True)` | 0 |
| `x[int_idx] = tensor` | 0 |
| `torch.sort(mask.to(torch.uint8), stable=True)` | 0 |
| `int(mask.sum())`, `bool(torch.all(mask))`, `bool(torch.any(mask))` | 1 each |
| `torch.stack([a, b]).tolist()` | 1 |
| `torch.empty(...)`, `torch.nan * torch.ones(...)` | 0 |
| `mask.to("cpu")` | 1 |

In the integration, each boolean-mask line counted one wait per call, not two
(the counts of 2.3); I do not know why the single operation warns twice.

### 2.3 Measurements of the integration (A100, GPU index 2, shared node)

Setup as the brief's baseline: gk21, `atol=1e-8`, `rtol=0`,
`error_norm="max"`, float64, `take_gradient=False`, `max_batch=1_000_000`,
[0, 1]. Scripts in `$SCRATCH/padaquad_speedup_plans/turn-device-reads_scripts/`:
`count_waits.py` (waits by source line, launches; saves the results),
`compare.py` (bit-for-bit comparison), `time_calls.py`, `time_chirp.py`,
`count_grad.py` (a `take_gradient=True` call), `patch.py` (the prototype
edits), `patch_checks.py` (one check removed at a time). Prototype packages:
`$SCRATCH/padaquad_speedup_plans/turn-device-reads` (unchanged `0590ccf`),
`..._mu` (mesh update only), `..._ev` (mesh update and evaluation path),
`..._full` (the same as `_ev`, with the tests), `..._nofinite`,
`..._nomeshorder`, `..._norecordorder`.

Host waits and launches of one call (counted, exact for this torch version):

| Integrand | Turns | Waits: base → mesh update → + evaluation path | Launches: base → mesh update → + evaluation path |
|---|---|---|---|
| chirp | 14 | 619 → 467 → 425 | 3,731 → 3,207 → 3,069 |
| many_outputs | 2 | 108 → 88 → 82 | 537 → 469 → 451 |
| oscillatory | 13 | 424 → 283 → 244 | 2,886 → 2,385 → 2,258 |

On `chirp`, `_adaptively_increase_mesh` with `_keep_rows` goes from 166 waits
to 14 (one per call), and `_evaluate_f_on_mesh` from 42 to 0; the loop's
`torch.nonzero` takes the place of the 15 waits of line 1214.

`take_gradient=True` (`chirp` times a parameter, `max_batch=230`, so each turn
evaluates only the first subintervals of the waiting set, 18 turns): 716 → 458
waits; the integral and the gradient are the same bit for bit
(`0x1.0283430b44e6fp-1`, `0x1.0283430b44e70p-1`).

Bit for bit: `integral`, `integral_error`, `mesh_quadratures`,
`mesh_quadrature_errors` and `state.mesh` of the three integrands are equal
(`torch.equal`, same shapes) between the base and both prototypes.

Wall time (median of 15 calls; minimum in brackets; indicative, since other
agents shared the node):

| Package | chirp | many_outputs | oscillatory |
|---|---|---|---|
| base | 62.4 (59.6) ms | 12.7 (12.6) ms | 136.1 (105.8) ms |
| mesh update | 65.0 (52.8) ms | 18.6 (11.1) ms | 132.1 (99.5) ms |
| + evaluation path | 51.6 (49.5) ms | 10.9 (10.9) ms | 121.5 (93.5) ms |

Cost of each check on `chirp`, on top of the `_ev` prototype (median of 21
calls; waits from the per-line counts; the differences are near the noise of
about 1 ms):

| Variant | Median (min) | Waits removed |
|---|---|---|
| all checks on (`_ev`) | 52.3 (48.8) ms | 0 |
| `_check_f_output_finite` off | 51.1 (47.6) ms | 14 |
| mesh-order check (1379-1386) off | 50.2 (47.8) ms | 28 |
| record-order check (4374-4380) off | 49.8 (46.1) ms | 25 |
| `runtime_checks=False` (all off) | 47.0 (46.7) ms | 70 |

## 3. The design

### 3.1 Mesh update (`_adaptively_increase_mesh`, `_keep_rows`)

1. **One read for the accept/split partition.** `keep_mask` and `remove_mask`
   are complements in every caller: `_accept_reject_masks` returns
   `keep_mask, ~keep_mask` (4066-4067), the legacy path (2035-2037) builds
   complements, and the cap (2041-2043) keeps them complements. So:

   ```python
   n_keep = int(keep_mask.sum())  # the update's one read
   order = torch.sort((~keep_mask).to(torch.uint8), stable=True).indices
   keep_idx, split_pos = order[:n_keep], order[n_keep:]
   ```

   A stable sort keeps the original order inside each group, so `keep_idx` and
   `split_pos` are the ascending positions that `keep_mask` and `remove_mask`
   select. Every field is then indexed with them (`x[keep_idx]`, a gather).
   Bit for bit: the same rows in the same order are copied, and the sums at
   2116-2119 run over equal tensors of equal shape.
   `remove_mask` becomes redundant: I recommend removing the parameter (one
   source of truth); its callers at 1361 and 3318 stop passing it.
   With an empty batch (Step 7, 1519) `len(keep_mask) == 0` is known on the
   host, so the read is skipped.
2. **`index_fill_` instead of Python-scalar assignment** at 2044, 2061 and
   2080 (`mesh_trackers.index_fill_(0, mesh_idxs[keep_idx], False)`, and so
   on). Same values written; no copy of a scalar to the GPU.
3. **`torch.empty` for `mesh_new` (2067) and `split_counts_new` (2099).**
   Every position is written: mesh point i moves to
   `i + #{s in split_idxs : s < i}` and the midpoint after split position
   `split_idxs[j]` goes to `split_idxs[j] + j + 1`. With `split_idxs` strictly
   ascending these two sets are disjoint and together hold
   `len(mesh) + N_t_add` positions, which the host-side assert at 2093 already
   checks. `split_idxs` is ascending because `mesh_idxs` is (it comes from the
   ascending waiting set, or a prefix of it) and the extras are sorted with it
   (2055). So the NaN fill and the NaN check (2091-2092) go; the check could
   not see anything after `torch.empty` anyway.
4. **Drop the clones** at 2083, 2084 and 2102: the destination is a new tensor,
   so it never overlaps the source (PyTorch refuses an overlapping index
   assignment with a `RuntimeError`, so an overlap could not pass silently).
5. **`all_kept = n_keep == len(keep_mask)`** at 2126: no read.
6. **`_keep_rows(x, keep_idx, in_place)`** takes the integer positions instead
   of the mask. The not-in-place path indexes with them (`x[idx]`); the
   in-place path uses them directly instead of `torch.nonzero` (174). The
   block loop is unchanged, so the "values held once" behaviour of CLAUDE.md is
   kept: kept row j still comes from row `idx[j] >= j`. When `x` is on another
   device than the index (`trade_speed_for_memory`, values on
   `result_device`), `idx.to(x.device)` is one copy, as `keep.to(x.device)` is
   now.
7. **2041** stays: `split_counts[mesh_idxs]` is integer indexing and does not
   wait.

Result: one read per mesh update (measured: 14 waits in 15 calls on `chirp`;
the 15th call runs on `result_device`, the CPU).

### 3.2 Evaluation path and loop condition

The loop reads the waiting set once per turn:

```python
while True:
    step_idxs = torch.nonzero(mesh_trackers).flatten()  # the turn's one read here
    if len(step_idxs) == 0:
        break
    ...
    self._evaluate_f_on_mesh(f, f_args, mesh_eval, step_idxs, take_gradient, ...)
```

`_evaluate_f_on_mesh` takes `step_idxs` (ascending left mesh indices of the
waiting subintervals) in place of `mesh_trackers`; `batches_left =
len(step_idxs) * self.C` is a host integer, and lines 2313-2314 go. Both
evaluation functions already take `step_idxs` (2334, 2538), so their
signatures do not change. On the `take_gradient=True` path,
`_evaluate_f_on_full_nodes` evaluates `step_idxs[:max_mesh_steps]` (2391) and
returns the evaluated prefix (2819); it is the same tensor as before, so this
path is unchanged (measured bit for bit above). Step 7's
`not torch.any(mesh_trackers)` (1492) runs only with a fixed
`error_integral_reference` (`reference_pinned`, evaluated first), so it costs
nothing by default; it stays.

### 3.3 Alternatives considered

- **Keep both `torch.nonzero(keep_mask)` and `torch.nonzero(remove_mask)`**:
  two reads instead of one; rejected.
- **Replace `mesh_trackers` by the sorted tensor of waiting indices** (my own
  idea, not prototyped). The mesh update would compute the next waiting set on
  the device (the old waiting indices after the batch, moved by
  `idxs_transfer`, merged with both halves of every split), and its length
  would be known on the host from `n_keep`, so the loop's `torch.nonzero`
  would go too: one read per turn in total, from the mesh update. It changes
  more signatures (`_setup_initial_mesh`, the mesh update, `_get_optimal_mesh`,
  Step 7 and about 15 tests). Estimated saving: one wait per turn (14 on
  `chirp`). I recommend it as a follow-up only if the `compiled-turn` plan does
  not replace the dynamic shapes first, since a captured turn needs
  fixed-capacity buffers in which neither form is used.
- **`torch.nonzero_static` with a count tracked on the host**: no wait, but
  the count would be a second copy of what `mesh_trackers` holds (against the
  one-source-of-truth rule); rejected.
- **`torch.where(mask)` or `masked_select`**: these also wait (they need the
  output size); no gain.

## 4. Decisions for the user (as taken on 2026-11-26)

The original options are kept below for the record; the review (items 3 and
4) changed both.

- **D1, taken without a switch.** Removed, because the change makes them true
  by construction: the loop's mesh-order check after the update (with D2),
  the NaN check of the new mesh, the node-order check of the
  `take_gradient=True` path. Kept under `runtime_checks`: the located
  `ValueError` for a NaN or Inf from `f` (one read per call of `f`), the
  extras' ascending check (only on turns where the re-check splits), the
  record-order check of `_record_results` (owned and deleted by
  `record-order`), and the checks after the loop (once per call, on
  `result_device`). The `runtime_checks` docstrings, the user guide and the
  CHANGELOG say so, in their own commit.
- **D2, taken: the guard, in its own commit**, for every split (the batch, the
  re-check's extras, the optimal mesh) through `_splittable`, on the device
  with no new read. A subinterval whose midpoint would round onto one of its
  ends is accepted; the call ends with `converged=False` and its warning
  instead of an empty `AssertionError` (`runtime_checks=True`) or a refinement
  that never ends (`runtime_checks=False`).

### Options as first written (0590ccf)

#### D1. What `runtime_checks` covers (recommendation: option c)

The option exists, its default is True, and its docstring (343-349, 855-859)
promises both the located `ValueError` for a NaN or Inf from `f` and the order
checks of nodes and mesh on every pass.

- **(a) Keep as is.** About 5 waits per turn (`chirp`: 70 waits, about 5 ms of
  52 ms with the other steps of this plan done, measured). No change of
  behaviour.
- **(b) One combined read per turn.** Compute every flag on the device and
  read them with the mesh update's `n_keep` read
  (`torch.stack([...]).tolist()`). The finite check would keep the first
  offending `t` values on the device without a read
  (`torch.nonzero_static(bad_rows, size=max_report, fill_value=-1)`, measured
  wait-free) and raise at the read with the same message. Costs: about 25
  more lines; the error is raised after the turn's other calls of `f`, which
  can be expensive; the record-order check runs after that read (Step 6), so
  it would be read one turn late. Zero extra waits.
- **(c) Separate the two kinds of check (recommended).** Keep `runtime_checks`
  for what users are promised and what protects them: the non-finite check of
  `f`'s output (1 read per call of `f`; about 1.2 ms of 52 ms on `chirp`).
  Move the internal invariants, which detect bugs in PadaQuad rather than
  problems in the user's integrand, behind one module-level switch that
  `conftest.py` turns on for the test suite (as proposed in
  `per-turn-overhead.md`, item 6): 2059-2060, 2394-2397, 4374-4380 (unless
  `record-order` removes it) and 1379-1386 (unless D2 replaces it). The
  per-call checks after the loop (3177, 3606-3611, 4419-4430, `uniform.py`,
  `variable.py`) run once per call on the CPU by default; they can stay under
  `runtime_checks`. The docstrings at 343-349 and 855-859 then say
  "raise `ValueError` naming the points where `f` returned NaN or Inf" only.
  Saves about 4 waits per turn (about 4 ms on `chirp`).

The finite check is user-facing and raises a located `ValueError`; none of the
options removes it from the default.

#### D2. The mesh-order check after the update (recommendation: guard)

The check at 1379-1386 can only fail in one way after this plan: a midpoint
`0.5 * (left + right)` that rounds to `left` or `right`, when the two are
adjacent floating-point numbers (the existing points keep their order by
construction). Whether the rounding floor (`_round_floor`) always prevents a
split that deep is something I have not checked.

- **(a) Keep the check** (under D1's choice).
- **(b) A guard instead of a check (my proposal):** before forming the masks,
  accept a subinterval whose midpoint would equal one of its ends, as is done
  for a subinterval at `max_adaptive_splits` (on the device, no read). Results
  are the same bit for bit whenever the current assert would pass; when it
  would fail, the call no longer raises `AssertionError` but finishes, and the
  final check sets `converged=False` with its warning. Cost: about 5 lines and
  a test. This is a change of public behaviour in a corner case, so it is
  yours to decide.

## 5. Implementation steps (one commit each)

1. **Plan review** (this file).
2. **Mesh update with one read** (`padaquad/quadrature/base.py`:
   `_adaptively_increase_mesh`, `_keep_rows`; callers in `integrate` (Step 5
   and Step 7) and `_get_optimal_mesh`). 3.1 items 1-6, without the
   empty-batch branch; `keep_mask` required, `error_ratios` and `remove_mask`
   and the legacy mask path removed; `keep_idx` returned in slot 8 and the
   loop filters `error_ratios` with it; the NaN check of the new mesh goes
   with the `torch.empty` allocation. Tests: `tests/test_adaptive_steps.py`
   (every call passes the loop's accept mask, `_keep_mask`; the explicit-mask
   test becomes `test_keep_mask_drives_accept_split`; the cap test checks
   `keep_idx`), `tests/test_trade_speed_for_memory.py::test_keep_rows_in_place`
   (integer positions).
3. **Evaluation path** (`integrate`'s loop condition; `_evaluate_f_on_mesh`
   takes `step_idxs`). 3.2. Tests: `tests/test_evaluate_f_on_mesh.py` (three
   calls), the spy of `tests/test_interior_nodes.py`.
4. **D2 guard** (`_splittable`, new; used by `_adaptively_increase_mesh` and
   `_failed_record_positions`, which gains `mesh`). Tests: new tests in
   `tests/test_adaptive_steps.py` (a one-ULP subinterval is accepted, alone,
   as an extra and in the optimal mesh's split) and
   `tests/test_split_statistics.py` (an interior non-integrable singularity
   ends with `converged=False`, a strictly increasing state mesh, and no
   error, with `runtime_checks` True and False).
5. **D1** (remove the three checks; `runtime_checks` docstrings of the
   constructor and `integrate`, `docs/user_guide/error_control.md`,
   CHANGELOG). Tests: the suite; `test_runtime_checks_off_skips_checks_and_keeps_result`
   still holds (the record checks remain).
6. **Host-wait tests** (`tests/test_base_solver.py::TestHostWaits`, its
   `_host_waits`): the mesh update reads the device once, with values on the
   solver's device and on the CPU only for the copy of the kept positions;
   `_evaluate_f_on_mesh` reads it only for the finite check of each call of
   `f` (none with `runtime_checks=False`).
7. **Documents** (`CLAUDE.md`, `CHANGELOG.md`) and this file's "Implemented".

## 6. Tests and proof that results are unchanged

- The whole suite (`pytest`), on CPU and on a GPU node.
- `tests/test_snapshots.py` (bit for bit on CPU, float64).
- The bit-identity grid `/pscratch/sd/k/khegazy/padaquad_methodclass_verify/`
  (`grid_run.py`, `grid_run_gpu.py`, `grid_compare.py`). It predates the 2.0.0
  API; to use it: `integrate` returns `(result, state)`; `mesh_optimal` is
  `state.mesh`; `same_f=True` becomes `state=state`; `conserve_memory` becomes
  `trade_speed_for_memory` / `evaluation_buffer` / `record_evaluations`; the
  bounds are tensors on every call. Add variants with `max_adaptive_splits`
  (the cap path, 2041), a fixed `error_integral_reference` (Step 7, the empty
  batch), `enforce_convergence=True` with failures (extras), tracked variables
  and `trade_speed_for_memory=True` (values on the CPU, `_keep_rows` across
  devices).
- Prototype checks already done: 15 of 15 fields equal on three integrands
  (GPU), and integral and gradient equal with `take_gradient=True` (GPU). On
  the `_full` prototype (CPU, login node), `test_snapshots`,
  `test_adaptive_steps`, `test_adaptivity`, `test_recheck_accepted_panels`,
  `test_max_adaptive_splits`, `test_split_statistics`,
  `test_gradient_integration`, `test_prune_and_optimal`, `test_integrals` and
  `test_oom_recovery` gave 1,560 passed and 138 skipped. The prototype keeps the
  `remove_mask` parameter and does not yet update
  `tests/test_evaluate_f_on_mesh.py` or `test_keep_rows_in_place`, which call
  the changed private signatures; they were not run.

## 7. How to measure the gain

- Counts (exact, any GPU node): `count_waits.py <out.pt> 1` and `0` with
  `PYTHONPATH` set to the package before and after; report waits and launches
  per call for `chirp`, `many_outputs`, `oscillatory`, and the per-function
  counts of `_adaptively_increase_mesh`, `_keep_rows`, `_evaluate_f_on_mesh`,
  `integrate`. Expected after steps 1-2: the numbers of 2.3.
- `compare.py before.pt after.pt`: every field bit-identical.
- Time: the brief's `profile_baseline.py` (median of 5) and `time_calls.py`
  (median of 15) on a node not shared with other jobs, since the shared node
  moved medians by several ms.

## 8. Documentation to update

- `CLAUDE.md`, "Memory management" ("A pass's values are held once"):
  `_keep_rows` takes the kept positions; "Core algorithm", step 2: the loop
  reads the waiting set once per turn.
- `CHANGELOG.md`: under "Changed", fewer GPU synchronizations per turn (no
  change of results); with D1(c), what `runtime_checks` now checks.
- With D1(c) or D2(b): the docstrings of the constructor (343-349) and of
  `integrate` (855-859), `docs/user_guide/error_control.md` and
  `docs/user_guide/troubleshooting.md` (both mention `runtime_checks`).

## 9. Risks and interactions with the other plans

- **Risk: masks that are not complements.** The partition in 3.1 item 1
  assumes `remove_mask == ~keep_mask`. Removing the `remove_mask` parameter
  makes this true by construction.
- **Risk: `torch.empty` hides a coverage bug.** The host-side length assert
  (2093) and the strict ascending order of `split_idxs` guarantee coverage;
  the snapshot tests and the grid compare the meshes bit for bit.
- **`record-order`** owns `_get_mesh_indices` (called at 2142), the record and
  the mesh-position bookkeeping. Interface I assume: this plan computes, inside
  the mesh update, `keep_idx` (positions in the batch, ascending) and
  `idxs_transfer` (2079-2082, unchanged); `record-order` may return
  `mesh_idxs[keep_idx]` or `idxs_transfer[mesh_idxs[keep_idx]]` (the kept
  subintervals' left mesh indices in the old or the new mesh) in the slot of
  the return tuple that now holds the dictionary. Both plans edit
  `_adaptively_increase_mesh`; land this plan first (it is local and changes
  no number), then `record-order`. The record-order check (4374-4380) is
  theirs to keep, move to the test switch, or drop.
  *Resolved in the review (item 5):* the mesh update returns `keep_idx`
  (after the cap and the guard) in slot 8, read by the loop; `idxs_transfer`
  is a local of that name, for `record-order` to return in slot 7.
- **`error-ratio-work`** owns `_compute_error_ratios`, `_accept_reject_masks`
  and the re-check. With this plan, `_accept_reject_masks`'s second return
  value (`remove_mask`) has no reader in the loop; they may drop it. The
  re-check's reads (1394 `bool(torch.all(record_keep_mask))`,
  `_failed_record_positions` 1777) are theirs.
- **`compiled-turn`** depends on about one read per turn. Waits per turn on
  `chirp` (14 turns, `runtime_checks=True`), measured on the `_ev` prototype:
  about 30. Of these, this plan keeps 2 (the loop's `torch.nonzero` and the
  mesh update's `n_keep`) plus the finite check (1 per call of `f`) and, with
  D1(a), about 4 invariant reads. `record-order` removes about 20
  (`_mesh_order` ~6, `_insert_sorted_results` ~7, `place` ~5,
  `_get_mesh_indices` 1), `error-ratio-work` about 2 (1394, 1777). With D1(c),
  the tracker-list alternative (3.3) and D1(b) for the finite check, one read
  per turn remains (`n_keep`, which only fixed-capacity buffers remove).
  Launches: this plan removes about 47 of 266 per turn (measured); the rest
  belongs to the record (`record-order`) and the arithmetic
  (`quadrature-arithmetic`, `error-ratio-work`).
- **`split-waste`** may split a subinterval into more than two pieces; then
  the offset arithmetic (2079-2090) changes, and the coverage argument of 3.1
  item 3 must be redone for it.
- **`call-setup-cost`** owns the checks before the loop (3606-3611 in
  `_setup_initial_mesh`); D1 applies to them only if they choose so.

## 10. Confidence

- **Sure (measured):** the wait and launch counts in 2.2 and 2.3 for torch
  2.8.0 on an A100; the bit-for-bit equality of the prototypes on the cases
  run; that the masks are complements in every current caller (read in the
  code).
- **Fairly sure (reasoning):** that every position of `mesh_new` is written
  (the argument in 3.1 item 3); that the `take_gradient=True` path is
  unchanged in general, not only in the case run.
- **Unsure:** the wall-time gains (the node was shared; medians moved by
  several ms between runs; treat 62 → 52 ms on `chirp` as indicative); why a
  lone boolean-mask index warns twice but once inside the integration;
  whether the rounding floor already makes the D2 failure unreachable.
- **Mine, not from a source:** the guard of D2(b), the tracker-list
  alternative, and the split of `runtime_checks` in D1(c) (the last one is
  also proposed in `per-turn-overhead.md`).
