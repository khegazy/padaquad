# Plan: the memory peak at the end of `integrate` (`record-peak`)

Status: measured on the base `c70f790` (tip of `speedups_integration`) and on
`main` (`d11df47`, before the speed-up work); design chosen from the
measurements below; implemented on branch `record_peak` (section
"Implemented"). The user's instruction: "Determine the best option based on
speed and robustness."

## 1. The question

The CHANGELOG (2.0.0, "Changed") says that since pull request #74 the values
kept on the host (`result_device="cpu"`) peak at about twice their size. Since
then the record is kept in the order subintervals are accepted, as chunks, and
put in mesh order after the loop (`_sort_record`, plan `record-order`), and
earlier reviewers measured stages above twice the values (the reuse mesh at 2.9
to 3.6 times for `bosh3`, a call at 3.3 times with removals). This plan
measures every stage of the end of a call, finds what holds more than about
twice the values and why, and chooses what to change.

R below is the size of the recorded integrand values in their per-subinterval
form, N x C x D (plus the tracked variables, when there are any); Q = N x D
elements is one per-subinterval array such as the contributions
`mesh_quadratures`. Q = R / C: a twentieth of R for `gk21`, a quarter for
`bosh3`, half for `adaptive_heun`.

## 2. How it was measured

`stage_peaks.py` (in `$SCRATCH/padaquad_speedup_plans/record-peak/impl/`)
wraps the stages of a call (each pass `_evaluate_f_on_mesh`, the mesh update,
`_record_results`, a removal `_remove_failed_record_panels`, the final
ordering `_sort_record`, the checks `_record_error_check`, the reuse mesh
`_get_optimal_mesh` and its helpers, `_flatten_output`, and the gaps between
them, such as the move of the record to `result_device` and the building of
the result) and records the peak above the memory allocated before the call:

- exactly, with the allocator's peak counter, on a second GPU that holds the
  record (`result_device="cuda:1"`), which runs the code path of the host;
- on the host itself (`result_device="cpu"`), with Linux `VmHWM` reset through
  `/proc/self/clear_refs` at every stage boundary, above `VmRSS` before the
  call;
- on the solver's GPU (`cuda:0`) in both runs, and in a third run with the
  record on the solver's GPU (`result_device="cuda:0"`).

Cases (all with `take_gradient=False` unless named): one pass (a smooth
integrand on 200 to 100,000 evenly spaced subintervals), several passes (a
narrow positive peak), removals by the re-check (a narrow peak that cancels
part of the integral), `take_gradient=True` with small batches,
`trade_speed_for_memory` without and with a 32 MiB buffer,
`record_evaluations=False`, one tracked variable as large as the values,
`build_state_mesh=False`, `error_norm="failure_fraction"`, cumulative mode;
for `gk21` and `bosh3` (and `adaptive_heun`, `cc17` at D = 10^4); D = 1, 512
and 10^4. Node: NERSC Perlmutter GPU node, 4 x A100 40 GB, PyTorch 2.8.

## 3. Measurements on the base and on `main`

The call's peak on the record's device, in R (exact allocator, record on a
second GPU). "Stage" is where the base reaches it.

| case | R (MiB) | main | base | stage on the base |
|---|---:|---:|---:|---|
| gk21, D = 512, one pass | 162 | 2.09 | 2.15 | building the result |
| gk21, D = 10^4, one pass | 335 | 2.09 | 2.14 | building the result |
| gk21, D = 10^4, several passes | 183 | 2.09 | 2.14 | building the result |
| gk21, D = 10^4, removals | 300 | 2.09 | 2.15 | building the result |
| gk21, D = 10^4, take_gradient | 207 | 2.09 | 2.14 | building the result |
| gk21, D = 10^4, trade_speed_for_memory | 300 | 2.09 | 2.15 | building the result |
| gk21, D = 10^4, tracked | 599 | 2.02 | 2.07 | building the result |
| gk21, D = 10^4, record_evaluations=False | 300 | 0.13 | 0.14 | building the result |
| bosh3, D = 512, one pass | 31 | 2.75 | 2.53 | building the result |
| bosh3, D = 10^4, one pass | 64 | 2.76 | 2.51 | reuse mesh |
| bosh3, D = 10^4, several passes | 68 | 2.76 | 2.51 | building the result |
| bosh3, D = 10^4, removals | 3,945 | 2.75 | 2.50 | building the result |
| bosh3, D = 10^4, take_gradient | 331 | 2.75 | 2.50 | building the result |
| bosh3, D = 10^4, trade_speed_for_memory | 3,945 | 2.75 | 2.50 | building the result |
| bosh3, D = 10^4, failure_fraction | 845 | - | 3.32 | reuse mesh |
| bosh3, D = 10^4, cumulative | 142 | - | 2.78 | reuse mesh |
| adaptive_heun, D = 10^4, several passes | 463 | 4.51 | 4.01 | reuse mesh |

The base is never above `main` for `gk21` and is 0.25 R below it for `bosh3`
(`main` held a merged copy of the record in every pass and the reuse mesh's
temporaries). The documented "about twice" holds for `gk21` (2.15 R) and not
for methods with few nodes: 2.5 R for `bosh3`, 4 R for `adaptive_heun`. On the
host, `VmHWM` agrees with the allocator for large records (`bosh3`, removals,
3.9 GB: 2.50 R; `gk21`, D = 10^4: 2.15 to 2.2 R) and is higher for records of
tens to hundreds of MB (up to 3.5 R), because glibc keeps freed blocks below
its mmap threshold (up to 32 MiB) resident (section 6). At D = 1 the nodes,
widths and error ratios are as large as the values, and the record's
per-subinterval arrays, not the values, set the peak (7 to 11 R, 1,200 bytes
per subinterval for `gk21` and 290 for `bosh3`).

What is alive at each stage above 2 R (base):

1. **Building the result** (`gk21` 2.15 R, `bosh3` 2.5 R). The record's
   per-subinterval values (R), their joined copy made by `_flatten_output`
   (R for `gk21`, (C - 1)/C R for `bosh3`), the contributions and error
   estimates moved to `result_device` before the join (2Q), and
   `torch.abs` of the error estimates for the result (Q).
2. **The reuse mesh** (`bosh3` 2.5 R, `adaptive_heun` 4 R). The values (R),
   the copies of the contributions and of the absolute error estimates made
   for it (2Q), the merge prediction (Q), the pair errors and the temporary of
   their sum (2Q), and inside the norm's reduction of the error estimates
   `abs` and the square (2Q): R + 6Q at the reduction. The merge then copied
   each input whole (`index_put`) before selecting the kept rows, and the
   check of the merged subintervals held the unmerged inputs, the merged ones
   and two temporaries.
3. **The final ordering** (2.0 R when the record has more than one chunk): the
   chunks (R) and the array in mesh order (R), each chunk freed once written.
   On the host, pages of the new array are committed only as rows are written
   while the chunks are freed, so `VmHWM` reads 1.2 to 1.9 R there.
4. **Removals** stay below the end of the call: at most 1.6 R of the final
   values (joining the chunks while the pass's values are on the host with
   `trade_speed_for_memory`), never the call's peak in these cases.

## 4. Options

**The final ordering.** With more than one chunk the chunks and the ordered
array coexist, 2 R at the moment the array is allocated; no option removes
that without writing the values into a preallocated record during the loop,
whose final size is unknown.

- (a) Accept the copy. 2 R on the allocator (less on the host).
- (b) Write the values in mesh order straight into the joined array of
  `_flatten_output`, so the ordering and the join are one copy. The reuse
  mesh's merge prediction must then read each subinterval's C values through
  an overlapping strided view of the joined array. For a method with nodes at
  both ends (k = 1), the right neighbour's first value would be read from the
  left neighbour's last one: the same point, but evaluated in another call or
  another position of a vectorized kernel, whose result can differ in the
  last bit (on the CPU, PyTorch's vectorized `sin`/`exp` and their scalar tail
  loop can differ by one unit in the last place). The prediction, and with it
  a merge decision at the cutoff, would then no longer be bit-identical. It
  saves one copy of R in time for multi-chunk records of k = 1 methods and no
  memory beyond (c'), below, and needs `_sort_record` to know which fields are
  per node, a strided view and a rewrite of `_flatten_output`'s tests.
- (c) Permute a single chunk that is not in mesh order in place. A single
  chunk out of order does not occur at the end of a call: a removal joins the
  chunks, but the removed subintervals were split, and their halves are
  evaluated and recorded as a new chunk in a later pass. Nothing to gain.

**The join (`_flatten_output`).**

- (c') For a method without nodes at both ends (the Gauss-Kronrod methods,
  among them the default `gk21`), no node is shared and the joined sequence is
  the per-subinterval array itself: a view, no copy, no time. For k = 1, keep
  the copy, but join each field in the record (freeing the per-subinterval
  array) before the contributions and error estimates come to
  `result_device`: R + (C - 1)/C R = 2R - Q.
- In-place compaction for k = 1 (move the rows forward within the array, as
  `_keep_rows` does) would bring the join to R, but needs to know that the
  array is PadaQuad's own (with `result_device` on the solver's device and
  `take_gradient=True` a recorded chunk can be `f`'s own output, detached,
  with no `_base` to detect it), leaves the dropped rows' memory in the
  result's storage, and does not lower the call's peak, which the ordering
  sets at 2 R. Not taken.

**The error estimates.** Take their absolute value in place in the record,
once, before the reuse mesh: the record's array is read by nothing else (the
per-batch results `loss_fxn` received hold their own absolute copies), and it
is detached. Saves Q on the solver's device and Q at the result.

**The reuse mesh.** Keep it on the record's device (moving its [N, D] work to
the solver's GPU would lower the host by up to 4Q but raise the GPU, the
scarcer memory, by as much), and order the work so that few [N, D] arrays are
alive at once:

1. `_get_optimal_mesh` makes its own copies of the contributions and error
   estimates (the caller's argument expressions kept them alive for the whole
   call), so the merged arrays replace them.
2. The norm family's reduced errors (`_reduced_errors`, two temporaries) are
   computed before the prediction exists and passed to `_compute_error_ratios`
   through its existing `reduced_errors` argument (the same function on the
   same inputs: the same bits).
3. The pair errors are added into the prediction in place, in the order of
   the former sum (prediction + e_i + e_{i+1}: the same bits), and freed after
   the pair ratio.
4. `_prune_excess_mesh` selects the kept rows and adds each merged pair's
   right row where its left row lands (`index_put_(..., accumulate=True)`: one
   addition per row, the same bits), instead of copying each input whole.

Expected: R + 4Q at the reduction of the error estimates (2.0 R for `bosh3`,
3.0 R for `adaptive_heun`). Splitting the reuse mesh into blocks of rows would
bring every method to about R + 2Q, but reductions over the D outputs of a
block of rows can be ordered differently by CUDA's reduction kernels than over
all rows (the record is on the GPU with `result_device` on a GPU), so the bits
of the decisions could change; not taken.

## 5. Choice and why

(a) for the ordering, (c') for the join, the absolute error estimates in
place, and the four changes of the reuse mesh. Every returned number and
decision is unchanged bit for bit (no value is read from another place or
summed in another order), the code shrinks or stays the same size (no new
option, no new mode; one early return in `_flatten_output` and a
key test in the join loop), and the call's peak becomes the ordering's 2 R
for every case of the norm family with C >= 4, and R plus the per-subinterval
arrays for one-pass calls of methods without end nodes (the default `gk21`;
typical of warm starts from a state). (b) would save one more copy of R in
time for k = 1 methods only, at the price of the prediction's bit identity and
more code.

## 6. What is not changed, and why

- `"failure_fraction"` and cumulative mode keep more [N, D] temporaries in
  their pair ratio (the per-component ratios, the running sums); `bosh3` stays
  near 2.5 to 3 R there. Lowering them means restructuring
  `_error_ratios_failure_fraction` and the cumulative tolerance for one
  stage; not done.
- Methods with C < 4 (`adaptive_heun`, `fehlberg2`,
  `interpolatory3_variable`): R + 4Q is 3 R and 2.3 R; their result alone is
  (C + 1)/C R.
- glibc keeps freed blocks smaller than its mmap threshold (at most 32 MiB)
  resident, so on the host the chunks of many small batches (for example
  `take_gradient=True`) stay in the resident size after the ordering frees
  them, and `VmHWM` can exceed the allocator's live set by up to R. `malloc_trim`
  after the ordering, or `MALLOC_MMAP_THRESHOLD_`, would return them; it is the
  C library's policy, not PadaQuad's, and is documented rather than changed.
- D = 1: the per-subinterval arrays (nodes, widths, ratios) are as large as the
  values; the peak is stated per subinterval.

## 7. Steps

1. This plan.
2. The join and the error estimates (`integrate`'s end, `_flatten_output`):
   test `TestFlattenOutput.test_no_shared_node_keeps_every_row` (a view) and a
   GPU-peak test of a whole call.
3. The reuse mesh (`_get_optimal_mesh`, `_prune_excess_mesh`): GPU-peak tests
   of `_prune_excess_mesh` and of a whole call with `bosh3`.
4. Documentation: `docs/user_guide/devices_memory.md` (the peaks per case),
   CHANGELOG, CLAUDE.md.
