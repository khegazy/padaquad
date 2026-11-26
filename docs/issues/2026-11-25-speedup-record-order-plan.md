# Speed-up plan: keep the record without the mesh dictionary (`record-order`)

Status: reviewed against `cd3ff34` (the tip of `speedups_integration`,
2026-11-26) and corrected (section "Review"); implemented on branch
`record_order` (section "Implemented"). Items 1 and 2 of
`per-turn-overhead.md` (branch `performance_benchmark`). Line numbers in
sections 1-3 are of `padaquad/quadrature/base.py` at commit `0590ccf`, where
the plan was written; the review gives the locations at `cd3ff34`. A CPU
prototype exists (section 7); no GPU measurement was made before the review.

## Review (2026-11-26, against `cd3ff34`)

Five merged changes landed after the plan was written (`git log
0590ccf..cd3ff34`: #74 on `main`, `endpoint_evaluations`, `startup_evaluations`,
`call_setup_cost`, `turn_device_reads`). Checked claim by claim against the
code at `cd3ff34` and by small experiments; what was wrong or is now
different, and what changed in this file:

1. **Slots of the mesh update.** `turn_device_reads` already returns `keep_idx`
   (final, after the split cap and the new guard `_splittable`) in slot 8 of
   `_adaptively_increase_mesh` (index 7, line 2126) and keeps `idxs_transfer`
   as a local (2062). Only slot 7 (index 6, the dictionary,
   `self._get_mesh_indices(mesh_new)` at 2125) changes: it returns
   `idxs_transfer`. The loop's `error_ratios[keep_idx]` (1314) is already in
   place. `_get_optimal_mesh` unpacks all nine slots into `_` except the mesh
   (3333), so its line does not change at all; nothing conflicts with
   `optimal_mesh_cost` there.
2. **The `mesh_left` stopgap** of `endpoint_evaluations` (a subinterval's
   first node is no longer its left mesh point for the Gauss-Kronrod methods)
   is new: `MethodOutput.mesh_left` (`padaquad/results.py` 257-267), set by
   `_calculate_integral` (664) and `_evaluate_f_on_split_nodes` (2788),
   filtered in the mesh update (2098), the record key `mesh_left` and the
   argument `mesh_left` of `_record_results` (4263, 4365). The plan now removes
   all of it: nothing identifies a subinterval by a node or a coordinate any
   more. The `MethodOutput` field and its two tests
   (`tests/test_methods_registry.py:244`, `tests/test_evaluate_f_on_mesh.py:119`)
   go; `_calculate_integral` keeps its `mesh_left` argument (it needs the
   width).
3. **Step 7 needs no running sum.** It runs only when `reference_pinned`,
   which requires `use_absolute_error_ratio` (1060-1062), and in absolute mode
   `_error_ratios_norm` and `_error_ratios_failure_fraction` never read
   `cum_mesh_quadratures` (3900-3906). The plan's row for Step 7 (a cumulative
   sum in mesh order built with `_merge_positions`) is dropped: Step 7 passes
   `None`, and `_record_error_check` and `_failed_record_positions` accept
   `None` (their annotation changes from `torch.Tensor`). This is the same
   finding as the `error-ratio-work` plan's point (1).
4. **The loop's second `_mesh_order`** (1180) looks up
   `method_output.mesh_left`, which is `mesh[step_idxs]`, so it returns
   `step_idxs`; it is replaced by `step_idxs` (as `error-ratio-work` found).
5. **The batch's kept rows enter the record as
   `idxs_transfer[step_idxs[keep_idx]]`** (confirmed: `keep_idx` indexes the
   batch rows that `_adaptively_increase_mesh` keeps in `method_output`, the
   values and the nodes, 2093-2117; `step_idxs` are their left mesh indices in
   the mesh the update received; `idxs_transfer` maps that mesh to the new
   one). **`extra_split_idxs` is sorted with the batch's split positions**
   inside the update whenever it is non-empty (2044-2045), also in Step 7's
   call with an empty batch, so positions in record order need no sort of
   their own; the docstring's "Must be ascending" (1992) becomes "in any
   order".
6. **#74 already did two things the plan proposed.** `_insert_sorted_results`
   no longer clones (4249-4256), and `_flatten_output` writes into one array
   (4580-4590): decision D2 and step 8 are removed. #74 also made
   `_sort_record` keep a record that is already in mesh order without copying
   it (4432-4440, test `test_already_sorted_is_not_copied`). The plan's final
   ordering always wrote a new array; it now keeps that protection (a field
   held in one chunk whose rows are already in mesh order is kept as it is,
   decided by one read of the mesh indices), so a call whose accepted
   subintervals were recorded in one batch (one pass, the common case for
   smooth integrands) copies no values at the end, as now.
7. **The final ordering needs no sort.** The record of a finished integration
   covers its final mesh exactly once (`_get_optimal_mesh`'s docstring, 3267-
   3269; the loop ends when no tracker is True, and every subinterval whose
   tracker became False was recorded or removed with its split). So the mesh
   indices of the N recorded rows are a permutation of 0..N-1, and row i
   belongs at row `mesh_index[i]`: the fields are scattered there, without
   `argsort`. Under `runtime_checks` one read checks that they are a
   permutation (it replaces the plan's "equal to the positions whose tracker
   is False", which is the same statement at the end of the loop).
8. **D1 memory, recomputed.** gk21 now evaluates 21 nodes, not 23 (no end
   nodes since `endpoint_evaluations`). Kept on the GPU until the end of the
   loop with D1(b), in float64 with T = 1: nodes 21 x 8, widths 8, error
   ratios 8, so 184 bytes per accepted subinterval (2.5 MB for oscillatory's
   13,737 subintervals, 184 MB per million), plus the 8-byte mesh index of
   every recorded subinterval, which stays on the solver's device in every
   setting (as `mesh_quadratures` and `mesh_quadrature_errors` already do).
9. **Memory of the values, restated for the current code** (3.4): the per-turn
   merge made a new array of the whole record every turn (twice the record's
   values while it ran, since #74); chunks hold the record once. The end of
   the call is unchanged (two times the values, in `_flatten_output` and, when
   the record is not in order, in the final ordering).
10. **Host waits.** The baseline counts of 2.2 were made before
    `turn_device_reads`. At `cd3ff34` on `chirp` (77 subintervals, 14 turns;
    `impl/count_turn.py`, A100, `runtime_checks=True`), of 412 waits:
    `_insert_sorted_results` 120, `_mesh_order` 102, `place` 78,
    `_get_mesh_indices` 15, `_record_results` 12 (its order check),
    `_sort_record` 2; on `oscillatory` (13,737 subintervals, 13 turns), of 213:
    50, 42, 36, 14, 5 and 2. This plan removes all of these
    except about 2 per turn of `place` (the values and their tracked
    variables still go to `result_device` every turn) and adds about 4 at the
    end of the call (section 7 gives the measured numbers).
11. **Host-wait tests.** `tests/test_base_solver.py::TestHostWaits`
    (`turn_device_reads`) counts every wait of a call, including the
    dictionary's (`test_starting_mesh_reads_once_per_check`,
    `test_mesh_update_reads_once`). A host-wait warning carries the file and
    line of the Python statement that waited (checked on an A100,
    `impl/probe_warn.py`). Fixing the two counts is less code than filtering by
    line, so the counts are fixed; the missing pin (one read per pass at the
    loop's condition) needs the filter, so `_host_waits` gains an optional
    function whose lines it counts.
12. **Tests that call the changed functions, at `cd3ff34`** (re-found; the plan
    listed 25 at `0590ccf`): `tests/test_sorted_insert.py::TestMeshOrder` (2),
    `tests/test_record_and_sort.py::TestRecordResults` (9) and
    `::TestSortRecord` (7), the dictionary assertions and the `mesh_left` of
    `tests/test_adaptive_steps.py` (every test there builds a `MethodOutput`
    with `mesh_left`; the dictionary is asserted in 4 tests and `mesh_left` in
    1), `tests/test_recheck_accepted_panels.py` section E (2 tests, 3 cases),
    `tests/test_bug_regressions.py::test_runtime_checks_off_skips_checks_and_keeps_result`
    (2 cases; it counts `_mesh_order` calls),
    `tests/test_base_solver.py::TestHostWaits` (4 cases),
    `tests/test_solver_methods.py::test_starting_mesh_has_isqrt_segments`
    (unpacks three values of `_setup_initial_mesh`),
    `tests/test_methods_registry.py` and `tests/test_evaluate_f_on_mesh.py`
    (`mesh_left`). Section 6 says, for each, what protection is kept.
13. **Vague steps made concrete**: every step of section 5 now names its
    functions and the tests that prove it; the memory claims of 3.4 name the
    tests and the script that measure them.
14. **Readers of the record during the loop, re-checked** (3.3): the
    cumulative merge, the re-check and `_remove_failed_record_panels`, the
    `loss` sum, Step 7, the `mesh_grad` carry (Step 7 never runs with
    `take_gradient=True`, which `enforce_convergence` refuses, so the carry
    always uses the update of Step 5 in the same turn), the final check and
    the split statistics (from `split_counts`, not from the record). No other
    reader exists (`grep` of `record[` in `padaquad/`).

## 1. Summary

- **What is slow.** The record of accepted subintervals is kept in mesh order
  by looking each subinterval's left mesh point up in a Python dictionary
  that is rebuilt from the whole mesh after every mesh update, and every
  record field (the integrand values included) is rebuilt by a merge on every
  turn. On `oscillatory` (13,737 subintervals) the dictionary functions take
  172 of 239 ms of the profiled call on the GPU and 91 of 128 ms on the CPU
  (measured, `baseline_cuda.txt`, `baseline_cpu.txt`).
- **What changes.** Each recorded subinterval carries its left mesh index as
  an integer on the device. One gather through `idxs_transfer` (the new
  position of every old mesh point, already computed by the mesh update)
  keeps it current. The values, nodes and other fields not read during the
  loop are appended as one chunk per turn and put in mesh order once, after
  the loop, into one preallocated array per field.
- **Expected gain (estimate, not measured).** On `oscillatory` most of the
  172 ms; on `chirp` about 280 of 618 host waits (a host wait is a point
  where the CPU stops until the GPU has finished its queued work).
- **Numbers.** No returned number changes: the CPU prototype returns every
  tensor of 400 integrations bit for bit, including 78 removals of recorded
  subintervals (section 7). The GPU check is a step.

## 2. Evidence

### The code now

- `_get_mesh_indices` (3618-3640) copies the mesh to Python (`mesh.tolist()`)
  and builds `{tuple(mesh point): index}`. Called by `_setup_initial_mesh`
  (3614) and at the end of every `_adaptively_increase_mesh` (2142), which
  runs once per turn, in Step 7 (1519) and in `_get_optimal_mesh` (3307).
- `_mesh_order` (4132-4166) copies left nodes to Python and looks each up.
  Called at 1252-1253 (loop cumulative sums), 1472 (`mesh_grad` carry), 1767
  (`_failed_record_positions`), 4354-4355 and 4375 (`_record_results`), 4412
  and 4421 (`_sort_record`).
- `_merge_positions` (4168-4197) ranks two index sets; it is integer work on
  the device and is not the cost.
- `_insert_sorted_results` (4199-4243) clones the old field, allocates a
  zero array and scatters both, for every field on every turn
  (`_record_results`, 4362-4371). The index copies at 4235-4236 wait when the
  field is on `result_device` and the indices are on the GPU.
- `place` (4320-4343) moves `nodes`, `h`, `y`, `error_ratios`, the tracked
  variables and `loss` to `result_device` on every turn (one wait each on
  CUDA when `result_device="cpu"`); `integral`, `integral_error`,
  `mesh_quadratures`, `mesh_quadrature_errors` stay on the solver's device
  (`_DEVICE_RESIDENT_KEYS` 4084, `_RESULT_INTEGRATION_KEYS` 4091).
- `_sort_record` (4384-4431) sorts the already sorted record after the loop
  (1553), copying every field.
- `_remove_failed_record_panels` (4433-4534) filters every field by a mask
  (Step 5b, 1394-1402; Step 7, 1536-1543).
- `_adaptively_increase_mesh` computes `idxs_transfer` (2079-2082): mesh
  point i of the old mesh is at `idxs_transfer[i]` in the new mesh.

### Measurements

From the coordinating session's baseline (gk21, `atol=1e-8`, `rtol=0`,
`error_norm="max"`, float64, `take_gradient=False`,
`max_batch=1_000_000`), `/pscratch/sd/k/khegazy/padaquad_speedup_profile/`:

| Integrand | Device | Profiled call | `_get_mesh_indices` | `_mesh_order` | `_insert_sorted_results` | `_record_results` (cumulative) | `_sort_record` |
|---|---|---|---|---|---|---|---|
| oscillatory | GPU | 238.9 ms | 126.9 ms (15 calls) | 45.3 ms (32) | 7.7 ms (35) | 30.1 ms | 9.0 ms |
| oscillatory | CPU | 128.4 ms | 52.2 ms (15) | 39.0 ms (32) | 2.5 ms (35) | 20.7 ms | 7.7 ms |
| chirp | GPU | 71.2 ms | (not in top 22) | 4.3 ms (77) | 5.9 ms (85) | 12.9 ms | — |
| chirp | CPU | 22.0 ms | — | 1.1 ms (62) | 1.9 ms (70) | 4.2 ms | — |

Host waits on `chirp` (GPU, 618 in all, 14 turns), by source line:
`_mesh_order` 4162-4163: 102; `place` 4343: 65; `_insert_sorted_results`
4235-4236: 96; `_get_mesh_indices` 3640: 15; `_record_results`: 12;
`_failed_record_positions` 1777: 13; `_sort_record` 4418: about 4.

Counted by me on the CPU (login node, counts only; script
`count_record.py`, section 7): none of the four baseline integrands removes
a recorded subinterval (`_remove_failed_record_panels` is never called), with
`take_gradient` True or False. Removal is a rare path.

## 3. The design

### 3.1 The record in two parts

The loop reads only three per-subinterval fields of the record:
`mesh_quadratures` and `mesh_quadrature_errors` (cumulative sums and the
re-check) and the mesh position. The other fields (`nodes`, `h`, `y`,
`error_ratios`, tracked variables) are only written during the loop and read
after it.

- `record: dict[str, torch.Tensor]`, on the solver's device:
  - the sums `integral`, `integral_error`, `loss` (as now, but `loss` stays
    on the solver's device until the end; decision D1);
  - `mesh_index` (new, `torch.long` [N]): the index in the current mesh of
    each recorded subinterval's left mesh point; it replaces the record key
    `mesh_left`;
  - `mesh_quadratures`, `mesh_quadrature_errors` [N, D].

  These rows are in the order the subintervals were accepted, not in mesh
  order. A batch's rows are appended with `torch.cat`.
- `chunks: dict[str, list[torch.Tensor]]`: for `nodes`, `h`, `y`,
  `error_ratios` and each tracked variable, one tensor per accepted batch, in
  the same row order as `record`. Nothing is merged or copied after a chunk is
  appended, except when a removal joins a field's chunks (3.4).

The constant `_DEVICE_RESIDENT_KEYS` becomes `_RECORD_LOOP_KEYS =
("mesh_index", "mesh_quadratures", "mesh_quadrature_errors")`, the fields kept
as one tensor; one name, one place.

### 3.2 The mesh index

- **Batch rows.** Both evaluation paths return `step_idxs`, the ascending
  indices of the evaluated subintervals' left mesh points in the mesh they
  were evaluated on. With `take_gradient=True` the batch is evaluated on
  `mesh_eval`, a detached copy of `mesh` with the same points, so the indices
  are the same. The accepted rows' indices in the new mesh are
  `idxs_transfer[step_idxs[keep_idx]]`.
- **The mesh update returns `idxs_transfer`** in slot 7 (index 6) in place of
  the dictionary; slot 8 already holds `keep_idx`, the final accepted
  positions (after the split cap and the guard). The tuple keeps its 9 slots.
- **Recorded rows.** After every mesh update (Step 5 and Step 7):
  `record["mesh_index"] = idxs_transfer[record["mesh_index"]]`, one gather. A
  recorded subinterval is split by the update only through the re-check
  (`extra_split_idxs`); its left mesh point is an old mesh point, so it is
  mapped too, and the row is removed right after (Step 5b, Step 7).
- **T > 1.** The index is an integer and does not depend on the number of
  coordinates T. The dictionary was introduced because coordinates have no
  order for T > 1 (`CHANGELOG.md`, the 2026-11 entry on `_mesh_order`). It
  also had a flaw the index does not have: a path through the same point
  twice would map both mesh points to one key
  (`docs/issues/multi-coordinate-t.md`).

### 3.3 Every reader of the record's order during the loop

| Reader | Now (`cd3ff34`) | After |
|---|---|---|
| Cumulative sums of the batch (1176-1188) | `_merge_positions` of two `_mesh_order` lookups | `_merge_positions(record["mesh_index"], step_idxs)`; the rest unchanged. Both index sets are in the mesh the batch was evaluated on (the remap happens after the update). `_merge_positions` ranks any order. |
| Re-check, cumulative sums of the recorded rows (1267) | `cumsum(all)[idxs_keep]` | unchanged: `idxs_keep` is aligned with the record's rows in any order. |
| Re-check positions (`_failed_record_positions`, 1731) | `_mesh_order` lookup | `record["mesh_index"]`; the `mesh_indices` parameter is removed. |
| `extra_split_idxs` "must be ascending" (1992) | ascending because the record was | in record order; sorted inside the update (2045). Docstring corrected. |
| `_remove_failed_record_panels` (Step 5b 1322-1330, Step 7 1466-1473) | mask over mesh-ordered rows | mask over the record's rows; the removed rows are gathered in mesh order (argsort of their `mesh_index`) for the sums and `loss_fxn`, as now (3.5). |
| `loss` bookkeeping | sum on `result_device` | sum on the solver's device, moved at the end (D1); same additions in the same order. |
| Step 7 (1418-1473) | `cumsum(record["mesh_quadratures"])`, valid because the record was sorted | no running sum (absolute mode only, Review item 3): passes `None`. Then the update, the remap, the removal. |
| `mesh_grad` carry (1396-1403) | `_mesh_order(mesh_eval, mesh_indices)` | `mesh_grad[idxs_transfer] = sum(grads)`: `idxs_transfer` is exactly "old mesh point to new position". |
| `_record_error_check` in the loop | rows in mesh order | rows in record order. Its per-row ratios depend only on the row and on the cumulative sum passed in. Its pair ratios (`error_ratios_2steps`) are meaningless in record order, but nobody reads them there (`_, ...` at 1670). In cumulative mode the loop always passes the running sums, so the internal fallback `cumsum(mesh_quadratures)` (which assumes mesh order) is used only after the loop. |

After the loop, `_sort_record` puts everything in mesh order (3.4), and the
final check, `_get_optimal_mesh` and the result read a sorted record, as now.

### 3.4 Devices, memory, and the final ordering

**Where chunks live, and when they move.**

- The values (`y`, tracked variables) go to `result_device` on every turn, as
  now (they exist only with `record_evaluations=True`). This keeps the
  documented GPU behaviour of `result_device="cpu"`: the GPU holds no accepted
  values (`docs/user_guide/devices_memory.md`, section "`result_device`").
- `nodes`, `h`, `error_ratios` and `loss` stay on the solver's device during
  the loop and move once, after the loop (decision D1). `_sort_record` puts
  the chunk fields on the record's device (`result_device`, or the solver's
  with `record_evaluations=False`), where `_get_optimal_mesh` reads `nodes`
  with `y`, as now.
- `trade_speed_for_memory=True` puts the values on `result_device` before the
  decision; the chunk is then already there.
- A chunk must own its memory. With `in_place`, `_keep_rows` returns the
  accepted rows as a view of the front of the pass's array, and `.to()` to
  the same device returns that view, which would keep the whole pass's array
  (rejected rows included) alive until the end. `_record_results` therefore
  copies a chunk whose storage is larger than the chunk
  (`value.untyped_storage().nbytes() > value.nbytes`). That copy happens only
  when the record's device is the device of the pass's array (CPU-only runs,
  `result_device=None`, or `trade_speed_for_memory=True` with a buffer that
  cuts a pass); with the default GPU solver and `result_device="cpu"` the move
  to the host is the copy.

**"A pass's values are held once" is unchanged.** The pass's array, its
in-place compaction and the `del` of the batch's values are not touched. The
record holds the accepted values once (one chunk per turn); the per-turn merge,
which made a new array of the whole record every turn (old record + merged
record while it ran, since #74), is gone.

**Peak of the values on the record's device** (R = the values of every
accepted subinterval at the end, P = one pass's values, K = its accepted
part; reasoning, measured in section 7):

| Moment | Now (`cd3ff34`) | After |
|---|---|---|
| Each turn | R_old + R_new (merge), with P alive on the same device | R_old + P (+ K, the copy of a view, same device only) |
| A removal (rare) | 2 × R (filter) | 2 × R: a field's chunks are joined (`torch.cat`), the list cleared, the kept rows indexed out of the joined array |
| The final ordering | none (#74 skips a sorted record) | none when every field is one chunk in mesh order (one recorded batch); else 2 × R: one array of the final size per field, filled chunk by chunk, each chunk dropped once written |
| Building the result | 2 × R (`_flatten_output`) | unchanged |

**Autograd with `take_gradient=False`.** The record keeps the graph:
`torch.cat`, indexing with integer tensors, `.to()` and indexed assignment
into a new tensor (`out[idx] = chunk`, a differentiable in-place copy) all
pass gradients. `take_gradient=True` detaches before recording, as now.

### 3.5 Bit for bit

Every number is formed by the same operations on the same inputs in the same
order:

- `integral`, `integral_error`, `loss`: running sums over batches in batch
  order, unchanged. `loss` is added on the solver's device instead of
  `result_device`; a basic IEEE 754 addition is correctly rounded on both
  (NVIDIA, *CUDA C++ Programming Guide*, "Floating-Point Standard" section,
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/). Only the order of
  the additions could change a result, and it does not change.
- Cumulative sums: the merged array is built from the same rows in the same
  mesh order, so the same tensor is summed.
- Removal: the removed rows are summed and passed to `loss_fxn` in mesh order,
  as now.
- The result, the final check and `_get_optimal_mesh` read the record after
  the final ordering, with the same rows in the same order on the same
  devices.

### 3.6 What is deleted and what is kept

- Deleted: `_get_mesh_indices`, `_mesh_order`, the `mesh_indices` variable,
  parameter and return value everywhere (`_setup_initial_mesh` returns
  `(mesh, mesh_trackers)`; `_failed_record_positions`, `_record_results`,
  `_sort_record` lose the parameter; `_adaptively_increase_mesh` returns
  `idxs_transfer` in its place); the `mesh_left` stopgap (`MethodOutput`
  field, record key, `_record_results` argument, its filter in the mesh
  update); the `place`/`target` closures; the order checks after each merge
  and in `_sort_record`: the record is no longer merged, so there is nothing
  to check per turn.
- Kept: `_merge_positions` (cumulative sums of the loop) and
  `_insert_sorted_results` (allocating with `new_empty`: every row is
  written, `_merge_positions` gives a permutation).
- Rewritten: `_record_results` (append), `_sort_record` (scatter once into
  mesh order, unless already in order), `_remove_failed_record_panels`
  (record order, chunks).
- New check, under `runtime_checks`, in `_sort_record`: the recorded mesh
  indices are a permutation of 0..N-1 (every subinterval recorded exactly
  once). One read at the end; it replaces the per-turn order checks.

### 3.7 Alternatives rejected

- **Keep the record in mesh order, merged each turn with integer indices**
  (only `_mesh_order` replaced). Removes the dictionary but keeps the work
  per turn proportional to the whole record.
- **A mesh-parallel array "record row of the subintervals starting here",**
  carried through the update like `split_counts`. Same information, but the
  update would carry one more array; the record-parallel index needs one
  gather in the loop.
- **Order by coordinates (`searchsorted` on the left mesh point).** Works only
  for T = 1, which is why the dictionary replaced it.
- **One growable buffer per field (capacity doubling)** instead of chunks.
  More code (capacity, length, growth), the same peak; `compiled-turn` may
  need it later for static shapes.
- **Dropping removed rows lazily** (leave them in the chunks, skip them at the
  end). The values of removed subintervals would stay in memory until the end,
  and the removal must gather them for `loss_fxn` anyway.
- **Joining the chunks and indexing at the end** (`torch.cat(chunks)[order]`).
  Two copies of the values instead of one, with the same peak.
- **Moving every field to `result_device` only at the end.** Fewest waits, but
  the GPU would hold every accepted value until the end, against the
  documented behaviour of `result_device="cpu"`.
- **Asynchronous copies to pinned host memory** (`non_blocking=True`). The
  host copy is valid only after the GPU's queue reaches it (PyTorch tutorial
  "A guide on good usage of non_blocking and pin_memory()", 2024,
  https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html). It would
  rely on a later wait in the loop. Not worth one wait per turn.

## 4. Decisions for the user

- **D1. Where `nodes`, `h`, `error_ratios` and `loss` wait during the loop.**
  - (a) As now: moved to `result_device` every turn (4 waits per turn on CUDA
    with `result_device="cpu"`).
  - (b) **Recommended, and taken:** on the solver's device until the loop
    ends, then moved once. Cost: GPU memory of (C·T + T + 1) · 8 bytes per
    accepted subinterval in float64 (184 bytes for gk21, C = 21, T = 1: 2.5 MB
    for 13,737 subintervals, 184 MB per million), against C·D·8 bytes of
    values per subinterval that still go to the host every turn. With
    `record_evaluations=False` the nodes already waited on the solver's
    device. The returned result is unchanged (same devices).
  - The user guide's `result_device` row ("Holds only the values waiting for
    their quadrature calculation") stays true for values; it now adds that
    the nodes, widths and error ratios of accepted subintervals wait on the
    solver's device until the end. Own commit.
- ~~D2~~ (`_flatten_output` in one copy): done by #74.
- No public API changes. `IntegrationResult` and `IntegrationState` are
  unchanged; `MethodOutput.mesh_left` (internal, added by
  `endpoint_evaluations` after the last release) is removed.

## 5. Implementation plan

All in `padaquad/quadrature/base.py` unless named.

1. **This file**, reviewed (first commit).
2. **The mesh index and the appended record, in one commit** (the record, the
   dictionary and `mesh_left` depend on each other; no intermediate state
   passes the tests).
   - `_adaptively_increase_mesh`: return `idxs_transfer` in slot 7; delete the
     `mesh_left` filter; docstring (slot 7, `extra_split_idxs` in any order).
   - `_setup_initial_mesh`: return `(mesh, mesh_trackers)`; delete
     `_get_mesh_indices` and `_mesh_order`.
   - `MethodOutput.mesh_left` (`padaquad/results.py`), its two setters.
   - `_record_results(record, chunks, take_gradient, results, mesh_index,
     result_device)`: detach when `take_gradient`; sums added; loop fields
     appended with `torch.cat`; other fields appended to their chunk list,
     the values moved to `result_device` (as now) and the other fields left on
     the solver's device (D1(b), see step 3), copied when their storage is
     larger than they are; the tracked-variable key check kept.
   - `_sort_record(record, chunks, device)`: one read decides whether the
     record is in mesh order; fields of one chunk in order are kept; others
     are scattered to row `mesh_index`; chunk fields moved to `device`;
     permutation check under `runtime_checks`.
   - `_remove_failed_record_panels(record, record_keep_mask, chunks, loss_fxn,
     mesh_init, mesh_final, values_device)` (the chunks third, so the test
     spies `spy(record, keep, *args)` work unchanged).
   - `_failed_record_positions`: drop `mesh_indices`, read
     `record["mesh_index"]`; `cum_mesh_quadratures` may be None (also in
     `_record_error_check`).
   - The loop (`integrate`): `chunks = {}`; cumulative merge from
     `record["mesh_index"]` and `step_idxs`; remap after each update; Step 6
     records `idxs_transfer[step_idxs[keep_idx]]`; `mesh_grad` carried with
     `idxs_transfer`; Step 7 passes `None` for the running sum; `_sort_record`
     after the loop.
   - `_insert_sorted_results`: `new_empty` for `zeros`.
   - Tests in the same commit (section 6).
3. **D1(b)** (own commit, so it can be dropped): with step 2 alone the
   chunks of `nodes`, `h`, `error_ratios` and `loss` go to `result_device` each
   turn as now; this commit leaves them on the solver's device until
   `_sort_record` and updates the user guide's `result_device` row.
4. **Host-wait tests** (`tests/test_base_solver.py::TestHostWaits`): fix the
   two counts; add the pin "one read per pass at the loop's condition" and
   "no read by the record per turn" with a line filter.
5. **Bit for bit, CPU and GPU**: the grid (`impl/bitgrid.py`, 1,000-odd
   integrations) against `cd3ff34` on both devices; snapshot tests; full suite
   on both.
6. **Measurement** (section 7): host waits by line, wall time (minimum,
   median, spread), garbage collections per call; memory peaks per stage.
7. **Docs**: `CLAUDE.md`, `CHANGELOG.md`, docstrings; this file's
   "Implemented".

## 6. Tests

What each test that calls a deleted or rewritten function protected, and
where that protection goes:

- `tests/test_sorted_insert.py::TestMeshOrder` (lookup by full coordinate
  vector, T > 1): the lookup is gone; ordering by integer mesh index for T = 2
  is tested in `TestSortRecord` (below). `TestMergePositions` gains a case
  with a record in any order (not ascending). `TestInsertSortedResults`
  unchanged.
- `tests/test_record_and_sort.py::TestRecordResults`: first batch creates the
  record and one chunk per field; a second batch appends (rows in acceptance
  order, `mesh_index` as given); sums accumulate; `take_gradient=True`
  detaches; tracked variables are chunk keys; a changed set of tracked
  variables raises; a chunk that is a view of a larger array is copied (its
  storage equals its size), a whole array is not (same `data_ptr`).
- `tests/test_record_and_sort.py::TestSortRecord`: scrambled `mesh_index` gives
  every field in mesh order, chunks emptied, scalars untouched, all keys
  consistent, T = 2 by mesh index; a record in order whose fields are one
  chunk is not copied (the #74 protection); several chunks in order are
  joined; the chunk fields end on the given device; the permutation check
  raises on a duplicate or missing index with `runtime_checks` and the
  gradient passes through the ordering.
- `tests/test_adaptive_steps.py`: the dictionary assertions become
  `torch.equal(barriers_new[idxs_transfer], barriers)` (T = 1 and T = 2);
  `mesh_left` dropped from the helper and the filter test.
- `tests/test_recheck_accepted_panels.py` section E: built with the new
  `_record_results`; the subset test checks the sums and that `loss_fxn`
  receives the removed rows in mesh order although the record is not; the
  emptying test appends after emptying (also with values on CUDA).
- `tests/test_bug_regressions.py::test_runtime_checks_off_skips_checks_and_keeps_result`:
  counts calls of `torch.sort` (the permutation check) instead of
  `_mesh_order`.
- `tests/test_base_solver.py::TestHostWaits`: counts fixed; new pins.
- `tests/test_solver_methods.py`, `tests/test_methods_registry.py`,
  `tests/test_evaluate_f_on_mesh.py`: two return values; no `mesh_left`.
- **New, end to end** (`tests/test_record_and_sort.py`): every option that
  changes the record's path (`take_gradient` True and False, removals by the
  re-check and by Step 7, cumulative mode, tracked variables,
  `trade_speed_for_memory`, `record_evaluations=False`, `max_adaptive_splits`)
  returns nodes in mesh order and the same numbers with `runtime_checks`
  True and False.
- **Memory**: `tests/test_trade_speed_for_memory.py::test_default_holds_a_pass_once`
  gains `result_device=None` (the record on the GPU): the peak stays below
  the bound for the record's values at the end (2 × R plus the per-turn
  arrays).
- **Unchanged and must pass:** `tests/test_snapshots.py`, every integration
  test file.

**Results unchanged, shown by** the snapshot tests and the bit-for-bit grid.

## 7. Prototype and measurement

Prototype (CPU, login node; correctness and counts only):
`/pscratch/sd/k/khegazy/padaquad_speedup_plans/record-order/`.

- `padaquad/quadrature/base.py`: design A of this plan, except that the split
  cap is applied in the loop (3 lines) instead of returning `keep_idx`, the
  record device of `_sort_record` is inferred from the values' chunk, and
  the D1 choice is (b). `base_orig.py` is the original for diffing.
- `count_record.py`: counts of the record functions per call.
- `bitgrid.py` / `bitcompare.py`: 400 integrations (4 uniform methods and
  `interpolatory3_variable`; every `integrand_dict` integrand plus a
  cancelling peak, an oscillatory and a chirp integrand; `take_gradient`
  True and False; multi-batch; cumulative mode; `max_adaptive_splits`;
  removals; `error_integral_reference` (Step 7); tracked variables;
  `trade_speed_for_memory`; `record_evaluations=False`; `evaluation_buffer`;
  a learnable D = 3 integrand with gradients and a warm start from `state`; a
  bound that requires grad), every returned tensor, the state's mesh and the
  gradients compared with `torch.equal`.
  - Result (measured, CPU, 2026-11-25): 400 integrations, 4,060 tensors,
    0 differences; `_remove_failed_record_panels` was called 78 times in each
    run, so the removal path was exercised. `grid_orig.log`,
    `grid_proto.log`.
  - The prototype applies the cap in the loop instead of returning
    `keep_idx`; both give the same final accept mask, so this does not change
    the conclusion, but the implementation must be run through the grid
    again.
- Tests run on the prototype (CPU, `CUDA_VISIBLE_DEVICES=` so GPU tests
  skip; `tests/test_examples_run.py` and the doctests not run):
  `tests/test_snapshots.py` 390 passed; every other file under `tests/`
  3,476 passed, and 3 more (the spies) after the parameter order of step 6
  was applied (`tests_proto.log`, `tests_proto2.log`).
  The 25 failures are all tests that call a deleted or rewritten function
  directly, the ones section 6 lists: `test_sorted_insert.py::TestMeshOrder`
  (2), `test_record_and_sort.py` (15), `test_adaptive_steps.py` dictionary
  assertions (4), `test_recheck_accepted_panels.py` section E (2),
  `test_bug_regressions.py::test_runtime_checks_off_skips_checks_and_keeps_result`
  (2).

The existing grid harness `/pscratch/sd/k/khegazy/padaquad_methodclass_verify/`
predates the API: `integrate` now returns `(result, state)`,
`conserve_memory` is `trade_speed_for_memory`, `same_f` is gone (warm start
with `state=`), and `mesh_optimal` is `state.mesh`. `bitgrid.py` above is
written for the current API and can replace it for this change.

**How to measure the gain** (on a compute node, not the login node; the
review's harness is in `$SCRATCH/padaquad_speedup_plans/record-order/impl/`):

1. Host waits by source line and GPU launches before and after
   (`count_turn.py`, the `turn_device_reads` script; the baseline numbers are
   in Review item 10), for the five cases of the planning brief.
2. Wall time alternating `cd3ff34` and the branch on one GPU, alone on the
   node: minimum, median and spread of 21 calls, and the number of Python
   garbage collections per call (counted with `gc.callbacks`), since the
   dictionary's tuples are the likely cause of the two-speed timings on
   `oscillatory`.
3. Memory per stage on the GPU with the record there (`result_device=None`,
   the allocator's peak is exact): each turn, a removal, the final ordering.
4. The bit-for-bit grid on CPU and CUDA.

## 8. Documentation

- `CLAUDE.md`: the test table rows of `test_sorted_insert.py` (it still names
  `_get_sorted_indices`, removed earlier) and `test_record_and_sort.py`;
  "Core algorithm" step 2 ("Record accepted results") and step 3 ("Sort the
  record"); "A pass's values are held once" (the record appends, the merge is
  gone); "Uniform vs variable sampling" is unaffected.
- `CHANGELOG.md`, Unreleased, "Changed": the record is kept in acceptance
  order with integer mesh positions and put in mesh order once; no number
  changes; the per-turn copies of the record and the 3 × host peak are gone.
  Also record that the dictionary entry of 2026-11 (`CHANGELOG.md:258`) is
  superseded.
- `docs/user_guide/devices_memory.md`, `result_device` table: with D1(b),
  say that the nodes and widths of accepted subintervals stay on the
  solver's device until the end.
- `docs/issues/multi-coordinate-t.md:83`: the coordinate-key collision is
  fixed.
- Docstrings of every function in 3.6.

## 9. Risks and interactions with the other plans

- **Risk: a reader that assumes mesh order during the loop.** Any new code
  that calls `_record_error_check` without `cum_mesh_quadratures` in
  cumulative mode, or that reads pair ratios of the record, would be wrong.
  The docstrings of `record` and `_record_error_check` must say "acceptance
  order".
- **Risk: a chunk that is a view.** Covered by the copy rule and its test.
- **Risk: GPU bit identity.** Argued; measured by the grid on CUDA (step 5).
- **`turn-device-reads`** (landed): returned `keep_idx` in slot 8 and kept
  `idxs_transfer` as a local for this plan; this plan returns it in slot 7.
  The record-order checks of `_record_results` are deleted here.
- **`error-ratio-work`** (not landed). What it gets from this plan: the mesh
  position of every recorded subinterval as `record["mesh_index"]`, a long
  tensor on the solver's device, in the record's row order; the batch's
  positions are `step_idxs`. In absolute mode (the default) the loop needs no
  ordering work at all: it can skip the merge (`_merge_positions`,
  `_insert_sorted_results`, both cumsums) entirely, leaving one `torch.cat`
  per loop field and one gather of `mesh_index` per turn. In cumulative mode
  the merge stays, bit-identical, from `record["mesh_index"]` and
  `step_idxs`. Two of its statements change: Step 7 already passes `None`
  (done here, its point (1)); and `_failed_record_positions` returns the
  re-split positions in record order, not ascending (`nonzero` over the
  record's rows no longer gives mesh order), which is fine because the mesh
  update sorts them. Its final-check `None` in cumulative mode relies on the
  record being in mesh order after `_sort_record`, which holds.
- **`optimal-mesh-cost`** (in progress beside this one): receives the sorted
  record as now, on the same devices (`nodes` and `y` on the record's
  device, `mesh_quadratures`, `mesh_quadrature_errors` and `integral` on the
  solver's). Its call of `_adaptively_increase_mesh` no longer builds a
  dictionary; its unpacking line (3333) is unchanged by this plan.
- **`call-setup-cost`**: `_setup_initial_mesh` no longer builds the
  dictionary (one `tolist` of the starting mesh per call).
- **`compiled-turn`**: the `torch.cat` of the loop fields changes shapes each
  turn; a capacity buffer for those three fields would replace it (3.7).
- **`split-waste`**: if a subinterval is split into more than two pieces,
  `idxs_transfer` must still be "new position of each old mesh point"; the
  record remap then works unchanged.
- **`quadrature-arithmetic`**: no interaction.

## 10. Confidence

- **Sure (read in the code and checked by the prototype):** the record's
  order is read in the loop only at the places of 3.3; `step_idxs` and
  `idxs_transfer` give every index the dictionary gave, for both gradient
  modes; on the CPU the results are unchanged bit for bit on the grid of
  section 7.
- **Fairly sure (reasoning):** the same on the GPU, because every number is
  formed by the same operations on the same tensors; to be checked (step 9).
- **Fairly sure (reasoning):** the memory table of 3.4. Not measured.
- **Unsure:** the size of the wall-time gain. The estimates subtract the
  profiled time of the deleted functions; cProfile adds overhead to many
  small Python calls (the dictionary building is exactly that), so the real
  gain may be smaller than the profiled 172 ms on `oscillatory`.
- **Not checked:** T > 1 end to end, which CLAUDE.md says is unreachable
  (the order check fails for T > 1); the unit tests with T > 1 are kept.

## Implemented (branch `record_order`, 2026-11-26)

Scripts and logs: `$SCRATCH/padaquad_speedup_plans/record-order/impl/`
(`bitgrid.py`, the `turn_device_reads` grid extended with removals in
cumulative mode and with the record on the solver's device, and tracked
variables with `take_gradient=True`: 856 integrations on the CPU, 864 on
CUDA; `bitcompare.py`; `count_turn.py` host waits by line and GPU launches;
`timing.py` wall time and garbage collections; `memory_stages.py` GPU peaks
per stage; `probe_warn.py`; `logs/`). Node `nid200288`, A100-SXM4 (GPU 0 for
the timings, alone on the node), torch 2.8.0.

| Step (5) | Commit | What |
|---|---|---|
| 1 | `95c66d3` | this file, reviewed |
| 2 | `aa8ac48` | mesh index, appended record, `_sort_record` by scatter (one chunk in order kept), removal with chunks, dictionary and `mesh_left` removed, Step 7 without a running sum; tests rewritten |
| 3 (D1) | `adb8478` | nodes, widths, error ratios, loss on the solver's device until `_sort_record`; user guide's `result_device` row |
| 4 | `9a97db9` | host-wait pins by source line (`_lines`, `_host_waits(fn, where)`) |
| memory test | `0c949a1` | recording a pass on the GPU never allocates the record's values again (fails on `cd3ff34`: 1.03 GB against a bound of 0.80 GB) |
| simplification | `19052b5` | `_record_results(result_device=result_device)`; docstrings |
| 7 | `6c66fb4` | CLAUDE.md, CHANGELOG, user guide (`runtime_checks`), `multi-coordinate-t.md` |

**Numbers unchanged.** The grid against `cd3ff34`: 0 differences of 8,832
tensors (CPU) and 8,928 (CUDA), with 246 and 228 removals of recorded
subintervals in the base runs; after step 2 and after step 4 (the code of
`19052b5` differs only by passing `result_device` where it is unused; the final
run repeats the grid).

**Host waits and GPU launches per call** (`count_turn.py`, gk21, `atol=1e-8`,
`rtol=0`, `error_norm="max"`, float64, `max_batch=1e6`, `runtime_checks=True`;
`cd3ff34` → branch):

| Case | Turns | Waits | Launches |
|---|---|---|---|
| damped_sine | 1 | 30 → 25 | 213 → 214 |
| chirp | 14 | 412 → 103 | 3,229 → 2,823 |
| many_outputs | 2 | 61 → 32 | 443 → 416 |
| oscillatory | 13 | 213 → 77 | 2,310 → 2,123 |
| chirp, `take_gradient=True`, `max_batch=230` | 18 | 422 → 90 | 3,003 → 2,565 |

On `chirp` what remains is the loop (37: its condition and the per-pass
reads), the mesh update (14), the finite check (14), `_record_results` (13:
the values to the host), `_failed_record_positions` (13, `error-ratio-work`),
`_sort_record` (6, once).

**Wall time** (`timing.py`, 21 calls after a warm-up, ms, min / median / max;
two rounds alternating base and branch; round 1 shown, round 2 within 2 ms of
it except `many_outputs`):

| Case | A100 base | A100 branch | CPU (8 threads) base | CPU branch |
|---|---|---|---|---|
| damped_sine | 4.55 / 4.59 / 5.15 | 4.42 / 4.45 / 4.89 | 3.01 / 3.03 / 3.19 | 2.92 / 2.94 / 3.12 |
| chirp | 46.2 / 48.4 / 54.1 | 35.4 / 37.4 / 42.2 | 15.3 / 16.1 / 16.9 | 13.2 / 13.7 / 16.7 |
| many_outputs | 9.00 / 9.34 / 9.75 | 8.47 / 8.59 / 8.98 | 6.08 / 6.27 / 6.75 | 5.71 / 6.04 / 6.19 |
| oscillatory | 89.0 / 123.2 / 143.6 | 36.2 / 38.2 / 40.3 | 72.9 / 105.4 / 114.7 | 28.9 / 29.7 / 31.1 |
| oscillatory, `result_device` the solver's | 76.4 / 101.4 / 115.2 | 28.7 / 28.9 / 29.7 | 74.9 / 105.1 / 127.6 | 28.8 / 30.4 / 33.2 |
| chirp, `take_gradient=True`, `max_batch=230` | 46.8 / 49.1 / 52.1 | 35.9 / 37.9 / 41.9 | 16.9 / 17.8 / 18.8 | 13.3 / 14.4 / 14.9 |

**Garbage collections per call** (`gc.callbacks`): `oscillatory` ran 100.7
collections of generation 0, 9.1 of generation 1 and 0.81 of generation 2 per
call on the base (93, 8.5 and 0.76 on the CPU); none on the branch. Every other
case: none on both, except `chirp` with `take_gradient=True` (0.8 of
generation 0 on both). The base's two-speed timing on `oscillatory` (89 to
145 ms) is gone (36 to 40 ms), consistent with the collections being its
cause (not proven: the spread could have other sources).

**GPU peak per stage** (`memory_stages.py`, multiples of R, the recorded
values; the record stage is the largest peak inside `_record_results`):

| Case, setting | `result_device` | total, base → branch | record stage | final ordering | `_flatten_output` |
|---|---|---|---|---|---|
| split (4096 outputs, 1,500 subintervals), default, trade, trade + buffer | None | 2.51 → 2.30 | 2.51 → 1.25 | 1.24 → 2.24 | 2.25 → 2.25 |
| split, every setting | "cpu" | unchanged (0.97 default, 0.48 trade) | unchanged | 0.24 → 0.24 | unchanged |
| split, `record_evaluations=False` | either | 0.96 → 0.96 | 0.25 → 0.25 | 0.24 → 0.24 | unchanged |
| removal (512 outputs, a tracked variable, 7,831 subintervals), default, trade, trade + buffer | None | 3.68 → 3.68 | 2.98 → 2.34 | 2.17 → 3.19 | 3.02 → 3.02 |
| removal, every setting | "cpu" | 2.29 → 2.29 | unchanged | 1.04 → 1.54 | unchanged |

The record's per-pass peak falls; the final ordering, which the base skipped
for its always-sorted record, now writes the values once when the record is
not in mesh order, which reaches about the peak of `_flatten_output` right
after it, so the call's peak does not rise in any case measured. The removal
stage is small (0.01 R) in both. On the host the same code runs with the record
there; it was not measured separately (the GPU numbers with
`result_device=None` are the allocator's exact peaks of the same operations).

**Tests at the final commit:** the full suite on CUDA and CPU, the grid, the
docs build, ruff, mypy and numpydoc are in the implementing agent's final
report (logs `logs/*_final*`).

### Not done

- The host's peak with `result_device="cpu"` was argued, not measured.
- `compiled-turn`: the loop fields' `torch.cat` per turn changes shapes; a
  capacity buffer would replace it (3.7).
