# Most of an integral's time is bookkeeping, not the integrand

Status: open (speed only; no accuracy problem). Found 2026-11-24 while explaining
the Performance page, where padaquad loses to quadax and scipy on integrands that
need few subintervals. Written so that another agent can pick it up without the
original session. Line numbers are those of commit `453474d`.

## Summary

On a GPU, one `padaquad.integrate` call is limited by the CPU, not by the GPU or by
the integrand. Each turn of the integration loop (one call of the integrand, then
error ratios, splitting and recording) launches about 200 small GPU operations and
makes the CPU wait for the GPU about 45 times, which costs about 3 to 4 ms per
turn whatever the integrand costs. A typical integral takes 5 to 10 turns. quadax,
which compiles its loop, spends about 0.15 ms per subinterval on the same GPU. So
padaquad only wins when each of its turns handles more than about 20 subintervals.
That threshold is our reading of these numbers.

On top of that per-turn cost:

- The record of accepted subintervals is copied to the CPU after every turn.
- That record is kept in mesh order through a Python dictionary.
- A reuse mesh is built after the loop on every call.

The record copy alone is 13 of 35 ms on `many_outputs`.

Most of the causes are local and fixable without changing results (items 1–9
below). Making a turn free of waits, so it can be captured as a CUDA graph or
compiled, is the larger step that follows (item 12).

## Evidence (measured)

All measurements were taken on a Perlmutter node with an NVIDIA A100-SXM4-80GB
and an AMD EPYC 7763, with `gk21`, `atol = 1e-8`, `rtol = 0`, `error_norm="max"`,
the integrands of `examples/speed_test/library_comparison.py`, and `max_batch`
from its `MAX_BATCH` table, so memory is never measured. `max_batch` does not
matter here: on `many_outputs`, values from 1,000 to 3.5 million gave the same
time, because at most 1,104 points wait at once.

One call on the GPU (PyTorch profiler, `torch.cuda.set_sync_debug_mode`):

| Integrand | Wall time | Integrand's time | GPU busy | GPU launches | Waits for the GPU | Turns |
|---|---|---|---|---|---|---|
| `many_outputs` (1,000 outputs) | 35–44 ms | 0.6 ms | 7.0 ms | 983 | 226 | 5 |
| `oscillatory` | 24–27 ms | about 1 ms | 3.6 ms | 1,050 | 237 | 5 |
| `chirp` | 42–48 ms | about 1 ms | 6.2 ms | 1,718 | 361 | 9 |
| `damped_sine` | 6–7 ms | 0.2 ms | 0.7 ms | 206 | 67 | 1 |

Removing one cost at a time, on `many_outputs` (GPU). These changes were made in a
test script only; every variant returned the same integral bit for bit:

| Change | Time |
|---|---|
| none | 35.1 ms |
| record kept on the GPU (`result_device="cuda"`) | 22.3 ms |
| + insertion without the extra `clone` | 21.5 ms |
| + no reuse mesh after the loop (`_get_optimal_mesh` skipped) | 17.6 ms |

For comparison, quadax takes 37.4 ms on this integrand. Skipping the reuse mesh
saved 3–4 ms on the GPU and 2–6 ms on the CPU on every integrand tried.

The 226 waits of `many_outputs`, grouped by their source lines in
`padaquad/quadrature/base.py`:

| Group | Waits | Lines |
|---|---|---|
| Keeping the record in mesh order and copying it to the CPU | 92 (41 %) | 4001–4002 `_mesh_order` (30), 4182 `place` (20), 4074–4075 index copies in `_insert_sorted_results` (24), 3481 `_get_mesh_indices` (6), 4214 assert (3), 4255/4258 `_sort_record` (5), 1592 return of `_failed_record_positions` (4) |
| `_adaptively_increase_mesh` | 55 (24 %) | 1853, 1862, 1888, 1899 (assert), 1918–1941 (one boolean-mask index per field) |
| Setup and wrap-up of each call | 30 (13 %) | `methods/_base.py:87–89` (`to_device`), `base.py:194–203`, 3321, 3449–3454, 3721–3722, 1380–1454 |
| Loop control and per-turn asserts | 20 (9 %) | 1052 `while torch.any`, 1213, 1217, 1227 |
| Evaluation path | 20 (9 %) | 2105–2106 (`batches_left`), 2114 (mask index), 2028 (`finite.all()`) |
| The benchmark's integrand (`k.to(t.device)` in `examples.py:608`) | 6 (3 %) | not library code |

GPU launches by the innermost solver method (`many_outputs`):

- `_adaptively_increase_mesh`: 318.
- `integrate` itself: 91.
- `_reduce_norm` and `_accept_reject_masks`: 80 each.
- `_error_ratios_norm`: 70.
- `_calculate_integral`: 60.
- The rest: 45 or fewer each.

`_get_optimal_mesh` launches nothing on the GPU, because it runs on `result_device`
(the CPU). Its cost is CPU time.

On the CPU device, torch's 128 threads slow these many small operations. With 8
threads:

- `many_outputs` takes 39 ms instead of 54 ms;
- `spectrum` takes 41 ms instead of 58 ms;
- `chirp` takes 20 ms instead of 26 ms.

`_insert_sorted_results` took about 30 of 75 ms on `many_outputs` on the CPU
(measured with timing wrappers around each method, so treat it as approximate).

Run-to-run variation: padaquad's slowest of 7 runs is often 2 to 6 times its
fastest, and medians moved by 30–80 % between two benchmark runs (`damped_sine`,
GPU: 6.8 ms, 12 ms, and 7 ms alone). JAX's thread pool running in the same process
was tested and is not the cause. The cause has not been found.

## The audit, cost by cost

Each item gives the location, what the code does, why it costs time, and a
proposed change. Ideas marked "(ours)" are proposals made here, not taken from a
source.

1. **The record is kept in mesh order on every turn, through Python.**
   - **What it does:** `_get_mesh_indices` (3459) copies the whole mesh to Python
     (`mesh.tolist()`) and builds a dictionary from coordinate tuples to indices.
     `_mesh_order` (3971) copies a set of left nodes to Python, looks each up,
     and builds a CPU tensor. `_merge_positions` (4007) ranks them with
     `argsort`, and `_insert_sorted_results` (4038) rebuilds every record field:
     `clone`, then `zeros`, then two scatters.
   - **How often:** per turn, `_mesh_order` runs six times:
     - twice in the loop (1092–1093), which computes the same merge positions as
       `_record_results`;
     - three times in `_record_results` (4193–4194, plus the ordering assert at
       4213);
     - once in `_failed_record_positions` (1582).

     Twice more in `_sort_record` after the loop (4249, 4257).
   - **The redundancy:** the record is sorted again after the loop anyway
     (`_sort_record`, line 1371).
   - **Proposed change (ours):** let each record row carry its left mesh index as
     an integer tensor on the device. When `_adaptively_increase_mesh` inserts
     midpoints it already computes `idxs_transfer` (line 1890), the new position
     of every old mesh point, so one gather remaps the record's indices on the
     device: no dictionary, no `tolist`.
     - Append each turn's rows (keep a list of chunks; one `torch.cat` at the end).
     - Sort once, by that index, at the end.
     - Code that needs mesh order during the loop (the cumulative mode, the
       re-check positions) gets it from an `argsort` of the index, on the device.
     - This also works for T > 1, which is why the dictionary was introduced.
2. **The record is copied to the CPU after every turn** (`place`, line 4182).
   - **Cause:** `result_device` defaults to `"cpu"` (line 679), "so they do not
     accumulate on the GPU" (`_record_results` docstring). The integrand's values
     at every node (`y`, [N, C, D]) and the nodes go to the CPU each turn, and
     the insertion then runs on the CPU.
   - **Measured:** 13 of 35 ms on `many_outputs`.
   - **Options:**
     - Keep the record on the integration device during the loop and move it
       once at the end.
     - Move it only when it exceeds a share of the free GPU memory.
     - Copy it asynchronously (pinned memory, `non_blocking=True`).

     The trade-off is GPU memory for large D or many subintervals. Decide with
     the memory design (`docs/user_guide/devices_memory.md`).
3. **Work needed only in cumulative mode runs in the default absolute mode.**
   - The loop's insertion of `mesh_quadratures` into the record order and the
     running sum (`cum_mesh_quadratures`, lines 1084–1101) are read only when
     `use_absolute_error_ratio=False` (`_error_ratios_norm`, 3729–3740).
   - The same holds for the re-check's running sum (line 1165).
   - **Proposed change:** skip them in absolute mode.
4. **Pair error ratios are computed every turn but unused in the loop.**
   - `_compute_error_ratios` always computes `pair_errors` and
     `error_ratios_2steps` (3641–3652; norms, rounding floor). The loop only
     checks their length and logs them (1116–1134).
   - Only `_get_optimal_mesh` uses them, after the loop (3119).
   - **Proposed change:** a flag, or a separate call after the loop.
5. **`_adaptively_increase_mesh` (1720): 55 waits and over 300 launches per call.**
   - **What costs:** every field is filtered with a boolean mask (`x[keep_mask]`,
     1918–1941). Each mask index runs `nonzero` and waits for the GPU. The new
     mesh is filled with NaN and then checked with `torch.isnan` (1899).
     `mesh_idxs[keep_mask]` and `mesh_idxs[remove_mask]` each wait again (1853,
     1862). The coordinate dictionary is rebuilt at the end (`_get_mesh_indices`).
   - **Proposed change (ours):**
     - Turn the mask into indices once (`keep_idx = keep_mask.nonzero()`, one
       wait) and use `index_select` with them for every field.
     - Allocate `mesh_new` with `empty`, since every position is written; drop
       the NaN check, or gate it (item 6).
     - With item 1, the dictionary is no longer needed.
6. **Invariant checks read the GPU on every turn.**
   - **Where:** 1213, 1217–1219, 1868, 1899, 2105, 4214, 4217, 4258, 3449, 3452.
   - **Why it costs:** each `assert torch.all(...)` on a GPU tensor waits for the
     GPU.
   - **Proposed change (ours):** keep them, but behind one module flag that the
     test suite turns on (for example `padaquad._CHECKS`, set in `conftest.py`),
     so production calls skip the reads.
7. **The evaluation path waits several times per turn** (`_evaluate_f_on_mesh`,
   2104–2114; `_check_f_output_finite`, 2028; the loop condition, 1052).
   - **What costs:** separate reads of `torch.sum(mesh_trackers)`, the mask index
     `step_idxs[mesh_trackers]` and `finite.all()`.
   - **Proposed change (ours):** compute the waiting count, the finite flag and
     the loop condition on the device, and read them together once per turn
     (one small `tolist()`).
8. **The reuse mesh is built after the loop on every call.**
   - **Where:** line 1425–1426, `_get_optimal_mesh`, with `merge_rule="predicted"`
     interpolating the stored integrand values `y` [N, C, D] in
     `_predict_merged_errors`.
   - **Measured:** 2–6 ms per call, on the CPU (`result_device`), even when the
     caller never uses `state.mesh`.
   - **Options:**
     - Build it lazily, on first access to `state.mesh` (the state keeps
       references to the sorted record).
     - Add an argument that skips it.
     - Run it on the integration device.

     See also `predicted-merge-cost.md`.
9. **Fixed costs of every `padaquad.integrate` call.**
   - **What they are:**
     - `_frontend.py:192` builds a new solver each call: the method is
       deep-copied and its tables are copied to the GPU (`methods/_base.py:87–89`,
       3 copies);
     - `atol` and `rtol` become device tensors (`base.py:194–203`);
     - a random starting mesh is built (`_setup_initial_mesh`, 0.5–0.7 ms);
     - one extra call of the integrand checks its output shape
       (`f(mesh_init.reshape(1, -1))`, line 970).
   - **Proposed change (ours):**
     - Cache the method tables per device.
     - Check the shape on the first batch's output instead of a separate call.
       That also removes one call of `f` per integral, in line with the rule that
       `f` can be expensive.
10. **The number of turns.**
    - **What sets it:** turns grow with the depth of halving, about
      log2(starting width / smallest width), not with the number of subintervals.
    - **Idea (ours, untested):** when a subinterval's error ratio r is far above 1,
      split it into about r^(1/k) pieces (k = `error_exponent`) instead of two,
      so deep refinement needs fewer turns.
    - **Risk:** extra evaluations where the estimate overshoots. Measure the
      evaluation count before adopting it; padaquad's rule is not to trade calls
      of `f` for speed elsewhere.
11. **CPU threads.**
    - Not library code. The user guide could advise fewer torch threads (about 8)
      for integrands that need few evaluations.
    - The benchmark uses all cores, to match XLA.
12. **The larger step: a turn without waits.**
    - **Precondition:** once items 1–7 leave about one read per turn (the loop
      condition), a turn has a fixed sequence of operations on fixed-size buffers.
    - **Two routes:**
      - capture the turn as a CUDA graph (one launch replays it);
      - compile it with `torch.compile`, fusing the elementwise work of
        `_reduce_norm`, `_accept_reject_masks`, `_round_floor` and
        `_calculate_integral`.
    - **Expected effect (ours, an estimate):** a turn would then cost tens to
      hundreds of microseconds, which would let padaquad win on integrands that
      need few subintervals too.
    - **Required:** preallocated mesh and record buffers with a capacity, and
      masked updates instead of boolean indexing.
13. **Run-to-run variation (unexplained).** Candidates to test:
    - Python's garbage collector (time with `gc.disable()`);
    - the CUDA caching allocator (`torch.cuda.memory_stats` around slow runs);
    - CPU frequency and thread wake-up.

## Suggested order

1. Items 3, 4 and 6: small, local, no behaviour change; they remove the unused
   work and the assert reads.
2. Items 1, 2 and 5: the record and mesh update, about 65 % of the waits.
   - Check bit-for-bit results with the grid in the user's `$SCRATCH`
     (`padaquad_methodclass_verify`) and the snapshot tests.
3. Items 7–9: the evaluation path, the reuse mesh and the fixed costs per call.
4. Item 12, then item 10 if turns still dominate.

After each step, re-measure:

- with the GPU-profiler counts above (launches, waits, GPU busy time);
- by rerunning `examples/speed_test/run_benchmark.sh` on an 80 GB A100 node
  (its `MAX_BATCH` table was measured there).

How to measure: wrap the solver's methods with `torch.profiler.record_function`
to attribute launches, and use `torch.cuda.set_sync_debug_mode("warn")` with a
`warnings.showwarning` hook that records the innermost `padaquad/` stack frame to
find each wait.
