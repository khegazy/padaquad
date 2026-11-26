# Plan: a turn without host waits, captured or compiled (`compiled-turn`)

Status: feasibility plan (2026-11-25), reviewed and measured against
`bc5b4a1` (2026-11-26), after the eight other speed-up plans landed. Item 12
of `per-turn-overhead.md` (branch `performance_benchmark`), "the larger
step". Line numbers are of `bc5b4a1` (`padaquad/quadrature/base.py` unless
another file is named). Measurements: one A100 node (`nid200441`, Perlmutter,
torch 2.8.0+cu128), nothing else running on the node during timed runs; CPU
runs with 8 threads on the same node. Scripts (prototype quality) and logs:
`/pscratch/sd/k/khegazy/padaquad_speedup_plans/compiled-turn/impl/` (section 7).

A *turn* is one pass of the `while` loop of `integrate`: evaluate the waiting
subintervals, judge them, split or record. A *host wait* is an operation that
makes the CPU stop until the GPU has finished its queue (reported by
`torch.cuda.set_sync_debug_mode("warn")`). A *launch* is one GPU kernel or
copy, as the PyTorch profiler counts them; an *operation* is one ATen operator
that is not a view, as a `TorchDispatchMode` counts them (on CUDA, about one
launch each).

## Review (2026-11-26, against `bc5b4a1`)

The plan was written against `0590ccf`, before any speed-up was implemented,
and without a GPU. Every claim it relies on was checked against the code at
`bc5b4a1` and measured on an A100 (section 2). What was wrong or is now
different, and what changed in this file:

1. **The starting point is half of what the plan assumed.** The plan's turn
   (4.75 ms, 42.5 host waits, 266 launches on `chirp`) is now 2.17 ms, 4.9
   host waits (5 per turn from the second on) and about 165 launches (149
   operations) on the same `chirp` (2.1). Section 2 is replaced by these
   measurements; the per-operation costs the plan derived (11-15 µs per
   launch, 17.5 µs per wait) are replaced by measured ones: a small operation
   costs about 6.5 µs of host time and a host wait about 15 µs more (2.4).
2. **Stage 1 (route (c)) has almost nothing left to remove.** Of its four
   parts: the packed read (one read per turn) would merge reads worth about
   15 µs each, and two of its five reads are forbidden by the task (D2b, the
   finite check) or would add a code path (D2, the values' copy); the sorted
   waiting list would add a sort to save one read; the operation budget test
   would pin launch counts that change with the PyTorch version. Section 5 now
   lists every candidate found in the measured turn with its gain; one was
   implemented (three operations fewer per mesh update, no number changed),
   one test was added (the reads that grow with the number of passes are the
   pinned ones), and the rest are reported with numbers for the user to
   decide.
3. **D2 (copy the values without waiting) does not help the default integrand
   and helps large D a lot.** With D = 1 a turn with the record on the GPU (no
   copy at all) is not faster than with `result_device="cpu"` (2.30 against
   2.17 ms per turn on `chirp`), so the plan's argument "it lets a turn have
   one wait" buys nothing measurable there. With many outputs the copy is the
   largest cost of a turn: the prototype (pinned memory, `non_blocking=True`)
   cut a call of `many_outputs` with 10,000 outputs from 20.1 to 14.3-15.1 ms
   and with 100,000 from 108 to 71-78 ms, bit for bit (5.3). It is not on the
   branch (task decision) and needs a synchronization before every read of the
   record on the host during the loop, which the plan missed: the re-check's
   removal (`_remove_failed_record_panels`) joins the record's chunks on the
   host.
4. **`torch.lerp` gives the same bits on the CPU and the A100.** The plan
   argued that the nodes of route (d) must be computed on the GPU because
   `lerp` "need not give the same last bit"; on 21 million gk21 nodes it did
   (2.5). What must stay on the GPU for bit identity are the sums over
   subintervals and the `"2"` and `"rms"` norms over outputs, which differ.
5. **Route (d) was measured, not estimated.** A proxy (the whole solver on the
   CPU, `f` on the GPU) runs a `chirp` turn in 0.95 ms against 2.1 ms on the
   GPU, and stops paying at about 5,000 mesh points (5.4). And the plan
   missed that on the CPU alone a turn already costs 0.7-0.9 ms: the
   few-subinterval targets of 3.6 are met today by `device="cpu"` (2.1).
6. **`take_gradient=True` turns spend about 0.4 ms in memory statistics.** The
   plan did not look at them. Each call of `torch.cuda.memory_allocated`,
   `max_memory_allocated` or `memory_reserved` builds PyTorch's whole
   statistics dictionary (about 65 µs), and a `take_gradient=True` turn on a
   GPU makes six of them (2.6). Not a GPU operation or host wait, so outside
   this plan's scope; reported as an option (5.2).
7. **Line numbers and code paths re-located.** The plan's table of a turn
   (old 2.1) cited code that `turn_device_reads`, `record_order` and
   `error_ratio_work` rewrote (no `_mesh_order`, no dictionary, no per-turn
   running sums in absolute mode, one read per mesh update). It is replaced by
   the measured table of 2.2.
8. **Decisions as taken** (section 4): D1 accepted; D2, D2b, route (d) and
   stage 3 not implemented, by the task's instruction, with their measured
   gains reported.

The original analysis of quadax (2.7), of the PyTorch documentation (2.8) and
of compiled bits (2.9), and the blockers of capture and compilation (3.1),
are unchanged and kept.

## 1. Summary

- **Where a turn stands now (A100, `chirp`, defaults):** 2.17 ms per turn,
  5 host waits and about 149 operations (165 launches); a call of 14 turns
  takes 35.3 ms, of which 1.0 ms before the loop and 3.8 ms after it. The
  integrand's own work is about 4 operations of a turn. On the CPU (8
  threads) the same turn costs 0.89 ms.
- **No single remaining restructuring removes more than about 2 % of a turn**
  without a new code path: the turn's cost is spread over about 150
  operations of 6-15 µs each and the Python around them (2.4, 5.1). One such
  change was made: the mesh update inserts its midpoints with three
  operations fewer (5.1, item 1), about 1 % of a turn, no number changed.
- **One read per turn is the floor** (D1): `f` must be called on exactly the
  waiting nodes, so the host must know their number before each turn.
- **The options with real gains are outside "restructure only"** and are
  reported for the user: the values' copy without waiting for large D (D2,
  25-35 % of a call at 10,000-100,000 outputs), the bookkeeping on the CPU for
  small meshes (route (d), turns 2.2 times faster on `chirp`), and fewer
  memory-statistics calls with `take_gradient=True` (about 0.4 ms of a
  3.3 ms turn).
- **The few-subinterval targets** (a `sharp_peak` turn under about 1.0 to
  1.15 ms to beat quadax) are not met on the GPU (2.4 ms per turn) and are met
  on the CPU (0.8 ms per turn), section 6.

## 2. Evidence (measured at `bc5b4a1`)

### 2.1 The turn now, per integrand

Script `impl/measure_turn.py` (logs `impl/logs/base_cuda.json`,
`base_cpu.json`): gk21, float64, `atol=1e-8`, `rtol=0`, `error_norm="max"`,
`max_batch=1_000_000`, interval [0, 1], one solver built per call, seed 0;
median of 15 calls after a warm-up. "Turn" is the host time from one entry of
`_evaluate_f_on_mesh` to the next (the last one to the entry of
`_sort_record`); "pre" is the call's start to the first turn, "post" the
first entry of `_sort_record` to the call's end. Integrands: `chirp`
`sin(1/(t + 1e-3))`; `damped_sine` `exp(-t) sin(40 t)`; `many_outputs`
`sin(k t)`, k = 1000 values from 1 to 100; `oscillatory` `sin(2e5 t)`; and
the Performance page's two few-subinterval integrands, copied from branch
`performance_benchmark` (`padaquad/examples.py`) and called through
`padaquad.integrate` as `library_comparison.py` does: `bench_damped_sine`
`exp(-5 t) sin(2π 3.7 t)` and `bench_sharp_peak` `1/(1 + ((t - 0.37)/1e-3)^2)`.

GPU, `take_gradient=False`, default `result_device="cpu"`:

| Integrand | Turns | Subintervals | Call (ms) | Pre (ms) | Turn (ms), median of the turns | Post (ms) | Host waits per turn (after the first) | Operations per turn | Launches per call |
|---|---|---|---|---|---|---|---|---|---|
| chirp | 14 | 77 | 35.3 | 1.02 | 2.17 | 3.80 | 5 | 149 | 2,331 |
| damped_sine | 1 | 11 | 5.0 | 0.99 | 1.74 | 2.24 | (4 in the one turn) | 115 | 191 |
| many_outputs (D = 1,000) | 2 | 13 | 11.2 | 1.25 | 2.72, 3.03 | 4.17 | 5 | 133 | 362 |
| oscillatory | 13 | 13,737 | 39.8 | 1.28 | 1.85-2.82 | 8.90 | 4-5 | 125 | 1,797 |
| bench_damped_sine | 1 | 11 | 5.0 | 1.00 | 1.77 | 2.24 | (4) | 117 | 193 |
| bench_sharp_peak | 7 | 17 | 20.4 | 1.00 | 2.42 | 2.99 | 5 | 149 | 1,207 |

The other variants (same integrands; turn in ms, median of the turns after
the first):

| Integrand | GPU, record on the GPU | GPU, `take_gradient=True` | GPU, `take_gradient=True`, record on the GPU | CPU (8 threads) | CPU, `take_gradient=True` |
|---|---|---|---|---|---|
| chirp | 2.30 | 3.46 | 3.23 | 0.89 | 0.97 |
| many_outputs | 2.43 | 3.78 | 3.26 | 1.33 | 1.56 |
| bench_sharp_peak | 2.37 | 3.28 | 3.23 | 0.83 | 0.90 |

(`take_gradient=True` used `f` times a scalar parameter equal to 1 and
`loss_fxn=lambda r: r.integral.sum()`.) With the record on the GPU the
values are not copied at all, and a D = 1 turn is not faster; with
D = 1,000 the copy costs about 0.6 ms of a 2.7-3.0 ms turn (5.3).

### 2.2 Host waits per turn, by source line

From the second turn on, a default turn waits five times (`chirp`, GPU):

| Line (`bc5b4a1`) | What is read | Per turn | Removable? |
|---|---|---|---|
| `integrate` 1216, `torch.nonzero(mesh_trackers)` | which subintervals wait (the loop's condition and the evaluation's input) | 1 | No: the floor (D1). |
| `_check_f_output_finite` 2404, `finite.all()` | whether `f`'s output is finite (`runtime_checks`) | 1 per call of `f` | Only by D2b (not allowed; 5.2). |
| `_failed_record_positions` 1860, `rec_positions[splittable]` | which recorded subintervals the re-check splits (boolean index) | 1 (from the second turn) | Only by joining the mesh update's read (5.2, item 4). |
| `_adaptively_increase_mesh` 2185, `int(keep_mask.sum())` | how many subintervals are accepted | 1 | No: sizes the new mesh. |
| `_record_results` 4489, `value.to(result_device)` | the accepted values copied to the host (`result_device="cpu"`) | 1 per values field | Only by D2 (not allowed; 5.3). |

With `take_gradient=True` the re-check does not run (4 waits per turn); with
the record on the GPU the copy does not either (4). Before the loop a call
waits 4 times (`_setup_integral_bounds` 3637, 3638, 3662 for bounds given on
the CPU, `_setup_initial_mesh` 3790), after it 16-18 times (the sort, the
split statistics, the state's mesh). No other line of the loop reads the
device.

### 2.3 Operations per turn, by function

`chirp`, GPU, default, 149 operations per turn (`TorchDispatchMode`, innermost
PadaQuad function; `impl/logs/base_cuda.json`, key `ops_by_fn`):

| Function | Operations per turn | What they are |
|---|---|---|
| `_adaptively_increase_mesh` | 40.9 | the masks and the one read; sort; the new mesh, trackers and split counts (index, `index_put_`, `cumsum`, `arange`); the kept rows of contributions, errors, widths, nodes; two sums |
| `_splittable` (twice: the batch, the re-check's record) | 17.4 | the subintervals' ends and midpoints, `!=`, `any`, `stack`, `all` |
| `_accept_reject_masks` (twice) | 15.4 | `isfinite` (4 operations: `abs`, `!=`, `==`, `*`), `<`, `|`, `~` |
| `_RK_integral` | 10.0 | two weighted sums over the C nodes, two sums over subintervals and two additions of `y0 = 0` |
| `integrate` (its own lines) | 8.5 | the loop's `nonzero`, the running integral, the batch result's `abs` and `zeros_like`, the record's mesh indices |
| `_evaluate_f_on_split_nodes` | 8.0 | the mesh points of the waiting subintervals, the pass's two sums, `cat`s |
| `_reduce_norm`, `_error_ratios_norm`, `_reduced_errors`, `_round_floor` | 19.6 | the batch's reduced error and floor; the tolerance and ratio, once for the batch and once for the record |
| `_record_results` | 6.9 | three sums added to, four loop fields appended (`cat`), the values copied |
| `_check_f_output_finite` | 6.0 | `isfinite`, `all`, the read |
| `_midpoint` | 5.9 | two midpoints (the split test and the mesh update) |
| `_call_f` | 4.0 | `f`'s own operations |
| others | 6.7 | `_failed_record_positions`, `_calculate_integral`, `_compute_nodes`, `_keep_rows` |

### 2.4 What an operation and a wait cost

`impl/microbench.py` on the A100 (median host time of 2,000 calls):

| | µs |
|---|---|
| one elementwise addition of 80 float64 numbers | 6.5 |
| ten such additions | 62.9 |
| one addition and a read of its result (one host wait) | 21.8 |
| `torch.isfinite` on 80 numbers (4 launches) | 22.9 |
| `torch.abs(x) < math.inf` (2 launches; the same booleans) | 13.8 |
| `torch.cuda.mem_get_info` | 13.6 |
| `torch.cuda.memory_reserved` and `memory_allocated` (each builds the statistics dictionary) | 129 |

So a small operation costs about 6.5 µs of host time and a host wait about
15 µs more. The turn's 2.17 ms is about 149 operations (about 1 ms at 6.5 µs,
more for indexing) and 5 waits (75 µs); the rest is Python around them
(cProfile, `impl/logs/base_cuda.json`, key `cprofile`: the mesh update, the
evaluation and the re-check are the three largest).

### 2.5 Which bookkeeping operations give the same bits on the CPU and the GPU

`impl/bits_cpu_gpu.py`, float64, `torch.equal` of the CPU and A100 results:

| Operation | Same bits? |
|---|---|
| `torch.lerp` of 1,000,000 subintervals' ends with gk21's fractions (21 million nodes) | yes |
| midpoint `0.5 * (a + b)` | yes |
| `clamp(rtol * x, min=atol)`, `atol + rtol * x` | yes |
| `e / maximum(tol, floor)` | yes |
| `amax(abs(y), -1)` over 1,000 outputs | yes |
| `cumsum` over 4,096 rows (3 columns) | yes |
| `sum` over 4,096 rows | no |
| `sqrt(sum(y**2, -1))` over 1,000 outputs | no |

### 2.6 `take_gradient=True` turns

cProfile of `chirp` with `take_gradient=True` on the GPU: of 3.3-3.5 ms per
turn, the backward pass and its bookkeeping (`_loss_and_backward`) take about
1.1 ms, and the memory checks about 0.4 ms: per batch,
`_evaluate_f_on_full_nodes` (2565-2577) calls `memory_allocated`,
`max_memory_allocated` and `_get_usable_memory` (`mem_get_info`,
`memory_reserved`, `memory_allocated`; `memory.py` 688-696), and
`_loss_and_backward` (`memory.py` 1715-1720) `memory_allocated` and
`max_memory_allocated` again: six builds of the statistics dictionary at about
65 µs each (2.4).

### 2.7 quadax (read in its source, and the Performance page's numbers; unchanged)

quadax (commit `2d825399` of https://github.com/f0uriest/quadax, `main`; the
benchmark ran version 0.3.1, which was not opened) keeps every subinterval in
arrays of fixed length `max_ninter` (`quadax/_adaptive.py`, `_init_state`,
1306-1313) and runs `bounded_while_loop(condfun, bodyfun, state, max_ninter + 1)`
(1767), a `jax.lax.scan` of `max_ninter + 1` iterations whose body is gated by
`jax.lax.cond` (`quadax/_utils.py`, 1299-1328). Each iteration bisects the one
subinterval with the largest error (`jnp.argmax(state["e_arr"])`, 1561) and
calls the rule twice (1573-1576). The benchmark sets `max_ninter` to the
smallest power of 2 at least twice what quadax needs
(`library_comparison.py`, `QUADAX_MAX_NINTER`, branch `performance_benchmark`).

From `docs/performance/results_cuda.json` on `performance_benchmark` (A100,
torch 2.11, quadax 0.3.1, commit `12ebfef`; bisections = (evaluations/21 −
1)/2, my arithmetic):

| Integrand (bench) | quadax | ms per loop iteration | ms per bisection | PadaQuad gk21 then |
|---|---|---|---|---|
| damped_sine | 1.59 ms | 0.32 | 1.59 | 12.19 ms |
| neural_network_wide | 2.78 ms | 0.31 | 1.39 | 9.17 ms |
| sharp_peak | 10.03 ms | 0.30 | 0.91 | 28.80 ms |
| oscillatory | 19.14 ms | 0.15 | 0.30 | 35.40 ms |
| many_outputs | 37.45 ms | 0.15 | 0.30 | 43.15 ms |
| chirp | 462.03 ms | 0.23 | 0.49 | 41.54 ms |

So the issue document's "0.15 ms per subinterval" is quadax's cost per
iteration of its fixed-length loop (counting the idle iterations after
convergence); per subinterval actually created it is about 0.3 ms or more.

### 2.8 What the PyTorch documentation says (PyTorch 2.8; unchanged)

CUDA graphs, https://docs.pytorch.org/docs/2.8/notes/cuda.html, section
"CUDA Graphs": a replay "skips all layers of argument setup and kernel
dispatch"; "Ops that synchronize the CPU with the GPU (e.g., `.item()` calls)
are prohibited"; "Dynamic control flow (based on CPU or GPU data) is
prohibited", "Dynamic shapes are prohibited", "CPU work is not captured";
"data-dependent dynamic control flow means the network isn't capturable
end-to-end"; each capture gets a private memory pool.

`torch.compile`, https://docs.pytorch.org/docs/2.8/generated/torch.compile.html:
with `dynamic=None` a size that changes is recompiled as dynamic;
`mode="reduce-overhead"` uses CUDA graphs only "for CUDA only graphs which do
not mutate inputs"; a frame is recompiled up to
`torch._dynamo.config.recompile_limit` (8) and then falls back to eager.
Dynamic shapes (https://docs.pytorch.org/docs/2.8/torch.compiler_dynamic_shapes.html):
sizes from `.nonzero()` or `.item()` are "unbacked", and "we must graph break
on these operations". CUDA graph trees
(https://docs.pytorch.org/docs/2.8/torch.compiler_cudagraph_trees.html)
re-record "for every unique shape of an input tensor". `torch.cond` is "a
prototype feature" (https://docs.pytorch.org/docs/2.8/generated/torch.cond.html);
`while_loop` (`torch/_higher_order_ops/while_loop.py`, torch 2.8.0) is a
prototype that "only supports **inference**".

### 2.9 Does compiled code give the same bits? (CPU, unchanged)

The planning scripts `compile_bits.py`, `compile_bits_size.py` and
`compile_rows.py` (`/pscratch/sd/k/khegazy/padaquad_speedup_plans/compiled-turn/scripts/`)
compared eager PyTorch with `torch.compile(..., dynamic=False)` (Inductor,
CPU, torch 2.8.0, float64). The elementwise chain of the error ratios and
masks with `"max"` kept its bits; the `"2"` norm over D = 1,000 outputs and
every weighted sum did not (differences at rounding level, 2.2e-16 to 1.8e-11
relative); and a compiled weighted sum with D = 1,000 gave a subinterval
different bits depending on how many subintervals were in the batch, which
would break "identical for any batch size". Inductor's GPU code was not
checked.

## 3. The design

### 3.1 What blocks capture or compilation (unchanged)

1. **The call of `f` sets the shapes.** `f` is arbitrary user code and must
   be called on exactly the waiting nodes (padding would be an evaluation of
   `f` at points whose values are discarded, which the user's rule forbids).
   The host must therefore know their number before each turn's first call of
   `f`: **one host read per turn is the floor** (D1). It also sets the number
   of calls of `f` and their out-of-memory retries.
2. **Sizes change every turn**: the waiting subintervals, the mesh and the
   record all grow (2.1). A CUDA graph needs fixed shapes and addresses;
   `torch.compile` recompiles per size or compiles dynamic kernels.
3. **Data-dependent control flow on the host**: the loop condition, the batch
   loop and its out-of-memory retries, the re-check's removal, Step 7, the
   split cap.
4. **Assertions and debug logging** (`runtime_checks`, `logger.debug` of
   tensors) synchronize, which capture prohibits.
5. **Autograd**: with `take_gradient=False` and an `f` whose output requires
   grad the contributions carry the graph to the end; with `True` each batch
   runs `loss.backward()` and user code. Only detached tensors may enter a
   captured region.
6. **Memory options** decide where the values go; copying them into a static
   buffer would hold them twice, against "a pass's values are held once".
7. **Capacity growth**: fixed shapes mean capacity buffers that must be
   re-allocated and re-captured as the mesh grows (12 to 13,738 mesh points
   on `oscillatory`).

### 3.2 The routes, with what is now measured

- **(c) Restructuring only.** Measured (section 5): every candidate found
  removes 1-12 operations or one wait from a turn, worth 1-4 % each; one was
  implemented.
- **(d) The bookkeeping of small meshes on the CPU, `f` and the reductions
  on the GPU.** Measured by a proxy (5.4): a turn of 0.8-1.0 ms on the few-
  and middle-sized meshes, against 2.0-2.2 ms on the GPU; it stops paying at
  about 5,000 mesh points. Not implemented (a device choice by mesh size is a
  new code path; task decision).
- **(a) CUDA graph of the bookkeeping, (b) `torch.compile`.** Not attempted
  (stage 3; see "Not done").

### 3.3 Why not a fully captured loop (unchanged)

A loop with the turn and `f` inside one graph (as quadax does in JAX) would
need `f` to be capturable, called on a fixed number of points (padding is an
evaluation of `f`: forbidden), and the loop condition inside the graph
(`torch.cond`/`while_loop` are prototypes without training support). quadax
pays for exactly this with a fixed loop length (idle iterations after
convergence) and one subinterval per iteration. Rejected.

## 4. Decisions for the user (as taken)

- **D1. One host read per turn is the floor.** Accepted.
- **D2. Copy the accepted values to `result_device` without waiting.** Not
  on the branch (the task's instruction: `record-order` rejected it, it needs
  pinned host memory as large as the record's values and an ordering
  argument). Prototyped in scratch; the gain is large only for many outputs
  (5.3). Left for the user.
- **D2b. Fold the finite check of `f`'s output into the turn's read.** Not
  implemented (it would raise the located `ValueError` after the pass's other
  calls of `f`, which could spend evaluations of `f` after a NaN). What it
  would save: one host wait per call of `f`, about 15 µs, under 1 % of a turn
  (2.2, 2.4).
- **D3. Route order.** Stage 1 done as far as it goes without new code paths;
  stage 2 measured by a proxy only; stage 3 not attempted.
- **D4. Route (d) automatic or an option.** Open; the proxy's numbers are in
  5.4.
- **D5. Compiled sums.** Not now (2.9).

## 5. Stage 1 against the measurements

### 5.1 To implement

1. **The mesh update inserts its midpoints with three operations fewer per
   halving** (`_adaptively_increase_mesh`, the halving loop). Each split
   subinterval's left mesh point moves to `step_transfer[split_idxs]`, which
   the update computed twice, once as that index for the split counts and once
   as `split_idxs + arange + 1 - 1` for the midpoint's left neighbour. Both
   are now the one tensor `split_left`, and the midpoint's position is
   `split_left + 1`. That `step_transfer[split_idxs[j]] = split_idxs[j] + j`:
   `step_transfer[i] = i + #{k : split_idxs[k] + 1 <= i}` (the `cumsum` of the
   offsets set at `split_idxs + 1`), which for `i = split_idxs[j]` counts the
   `j` smaller split positions because `split_idxs` is strictly ascending.
   Integers only, so every returned number is unchanged; measured: 149 to 146
   operations per `chirp` turn, the same on the post-loop mesh update.
2. **Test** `tests/test_base_solver.py::TestHostWaits::test_loop_reads_only_where_pinned`:
   the host waits that grow with the number of passes are exactly those the
   other tests of the class pin (per pass the loop's condition and the mesh
   update, per call of `f` the finite check, per re-check its one read), found
   by comparing two integrals of 30 and 300 Hz-scale frequencies whose numbers
   of passes differ, without hard-coding what is read before and after the
   loop. Checked to fail when one read per pass is added to the loop body. No
   launch count is pinned.

### 5.2 Candidates not implemented, with their measured or estimated gains

Gains are per turn of `chirp` on the A100 (2.17 ms), from 2.4's costs
(6.5 µs per small operation, 15 µs per wait); none was timed end to end,
since each is below the run-to-run variation of a turn (about 2-3 %).

| # | Candidate | Saves per turn | Why not |
|---|---|---|---|
| 1 | `torch.isfinite(x)` → `torch.abs(x) < math.inf` in `_check_f_output_finite`, `_accept_reject_masks` (twice per turn) and `_error_ratios_failure_fraction` (identical booleans) | 6 launches, about 27 µs (1.2 %); for `f`'s check with 231 × 1,000 values, 54 → 44 µs | A less readable idiom in four places for about 1 %. |
| 2 | Sum the batch's totals once: `_calculate_integral` (through `_RK_integral`) sums the integral and the error and adds `y0 = 0` to each, which the split path discards and sums again from the pass's contributions, and the mesh update sums them a third time for the accepted rows | 6 operations, about 40 µs (1.8 %) | Changes the interface of `_calculate_integral` (abstract in `padaquad/base.py`, with a `y0` argument the package never passes) and `_RK_integral`, which several tests call with `y0`. |
| 3 | Judge the batch and the record in one call (one tolerance, one ratio, one set of masks, one `_splittable` over both) | about 12-20 operations, 80-130 µs (4-6 %; estimate) | Rewrites Steps 4, 4b and 5's inputs, which `error_ratio_work` just reorganised and the running `split_waste` follow-up edits; for `"failure_fraction"` it would join `[R, D]` arrays every turn. |
| 4 | Join the re-check's read (1860) with the mesh update's (2185): the update would take the re-check's mask and read its count with `n_keep` | 1 wait, about 15 µs (0.7 %) | Rewrites the mesh update's read and its four-way branch, which the `split_waste` follow-up edits. |
| 5 | Fold the finite check into the turn's read (D2b) | 1 wait per call of `f`, about 15 µs (0.7 %) | Forbidden by the task (when the error is raised). |
| 6 | The loop's condition from the mesh update's counts (`torch.nonzero_static` with the number of waiting subintervals computed on the host) | 1 wait, about 15 µs (0.7 %) | A second record of the waiting count on the host, kept in step with the trackers by hand. |
| 7 | `_splittable`: compare consecutive points with one `stack`, `!=`, `any`, `all` instead of a list | 2 operations per call, 4 per turn (1.2 %) | `_splittable` is edited by the running `split_waste` follow-up. |
| 8 | One `torch.cuda.memory_stats` call in `_get_cuda_memory` instead of `memory_reserved` and `memory_allocated` (`take_gradient=True` only) | about 65 µs of a 3.3 ms turn (2 %); all six statistics calls about 0.4 ms (12 %) | Host-side, not a GPU operation or wait (outside this plan); reading fewer statistics per batch needs the private `torch._C._cuda_memoryStats` or fewer checks, a change to the backward-memory machinery. |

Items 1, 2 and 7 together would remove about 16 operations, about 7 % of a
turn; with item 3 about 15 %. None reaches the targets of section 6.

### 5.3 D2 prototype: the values to the host without waiting

`impl/patch_d2.py` patches a copy of `bc5b4a1`: `_record_results` copies `y`
and the tracked variables into a new pinned host tensor with
`non_blocking=True`, and the loop synchronizes the stream before
`_sort_record` and at the start of `_remove_failed_record_panels` (which joins
the record's chunks on the host during the loop). `impl/time_d2.py`, GPU,
`result_device="cpu"`, median of 11 calls, base and prototype alternated twice:

| Integrand | Values recorded | Call, base (ms) | Call, prototype (ms) | Turns, base (ms) | Turns, prototype (ms) |
|---|---|---|---|---|---|
| `sin(k t)`, 1,000 outputs | 2.2 MB | 10.1, 10.2 | 8.2, 9.4 | 2.3, 2.7 | 1.8-2.2, 2.1-2.2 |
| `sin(k t)`, 10,000 outputs | 21.8 MB | 20.0, 20.3 | 14.3, 15.1 | 3.4, 3.1 | 2.4, 2.4 |
| `sin(k t)`, 100,000 outputs | 218 MB | 107.9, 109.2 | 70.7, 78.4 | 31-32, 13.8 | 9, 5 |
| `chirp × cos(k t)`, 1,000 outputs (14 turns) | 13.6 MB | 41.8, 44.4 | 39.9, 41.1 | 2.3-2.5 | 2.3 |
| `chirp` (D = 1) | 13 kB | 35.5, 36.0 | 34.6, 35.4 | 2.2 | 2.2 |

Every returned `integral`, `y`, `nodes`, `mesh_quadratures` and
`error_ratios` was equal bit for bit (5 integrands). The pinned host memory
is the record's values (the last column but one), allocated per turn through
PyTorch's pinned-memory cache. Correctness needs a synchronization before
every host read of the record during the loop (two places today; a new one
added later would read unfinished copies silently), so it would want a test
that fails when one is missing. Worth it when the outputs are many (25-35 %
of a call at 10,000-100,000 outputs); nothing for D = 1.

### 5.4 Route (d) proxy: the bookkeeping on the CPU

`impl/route_d_proxy.py`: the same integration (gk21, `atol=1e-8`, `rtol=0`,
`"max"`) with (a) the solver on the GPU (today), (b) the solver on the CPU and
`f` on the GPU (each call of `f` sends its points to the GPU and its values
back: one wait per call), (c) everything on the CPU (8 threads); alternated,
median of 11. In (b) the weighted sums and error control also run on the
CPU, which route (d) would leave on the GPU, so (b) is pessimistic for many
outputs and many subintervals. The CPU solver draws its random starting mesh
on the CPU, so (b) and (c) integrate a different mesh than (a); the proxy
measures time, not bits.

| Integrand | Call (a) GPU (ms) | Call (b) (ms) | Call (c) CPU (ms) | Turn (a) (ms) | Turn (b) (ms) |
|---|---|---|---|---|---|
| chirp | 33.0 | 13.1 | 11.9 | 2.1 | 0.95 |
| sharp_peak | 17.4 | 8.8 | 8.1 | 2.05 | 0.90 |
| damped_sine | 4.7 | 3.0 | 3.0 | 1.55 | 0.79 |
| many_outputs (D = 1,000) | 7.9 | 5.9 | 5.8 | 1.9-2.1 | 1.3-1.5 |
| oscillatory | 30.7 | 24.2 | 23.8 | 1.5-2.3 | 0.7-2.5 |

On `oscillatory`, whose turns sweep the mesh from 12 to 13,738 points, (b)
is faster while the mesh has up to about 1,400 points (1.14 against 1.67 ms),
about equal at 2,700-4,900 (1.75-2.01 against 2.18-2.19 ms), and slower from
about 8,000 (2.48 against 2.19 ms): route (d) stops paying at about 5,000
mesh points, later if its weighted sums stay on the GPU.

Bits (2.5): the nodes, midpoints, tolerances, ratios, `amax` and `cumsum` give
the same bits on the CPU and the A100, so route (d) can keep them on either;
the sums over subintervals and the `"2"`/`"rms"` norms must stay on the GPU.
Whether a full route (d) is bit-identical end to end was not shown (it was not
built). It would add a device choice by mesh size (a code path in every
bookkeeping step, and a mesh that requires grad must stay on the GPU).

### 5.5 Stage 3 (CUDA graphs, `torch.compile`): not attempted

See "Not done".

## 6. The few-subinterval integrands against the plan's targets

The plan's 3.6 asked what a turn must cost to beat quadax's times on the
Performance page (A100; quadax 1.59 ms for `damped_sine`, 10.03 ms for
`sharp_peak`; the page was not re-run, and the integrands here are copies, 2.1).
`sharp_peak` takes 7 turns, so a turn must cost under about (10.03 − F)/7,
with F the fixed cost of a call:

| | GPU (A100), today | CPU (8 threads), today |
|---|---|---|
| F = pre + post of `bench_damped_sine` (one turn) | 1.0 + 2.2 = 3.2 ms | 0.7 + 1.8 = 2.5 ms |
| `bench_damped_sine`, a call (quadax 1.59 ms) | 5.0 ms | 3.25 ms |
| `bench_sharp_peak`, a turn (target about 1.0-1.15 ms) | 2.4 ms | 0.8 ms |
| `bench_sharp_peak`, a call (quadax 10.03 ms) | 20.4 ms | 8.6 ms |

On the GPU the targets are missed by a factor of about 2, and nothing in
section 5.1-5.2 closes that. On the CPU a `sharp_peak` call already beats
quadax's GPU time; a one-turn integral still loses on the fixed cost F, which
no turn change removes (the plan's conclusion stands).

## 7. How to measure, and the scripts

In `/pscratch/sd/k/khegazy/padaquad_speedup_plans/compiled-turn/impl/`:

- `measure_turn.py DEVICE OUT.json [REPEATS]`: per case (integrand ×
  `take_gradient` × `result_device`) turns, wall time per call and per turn,
  host waits by phase and source line, operations by function, launches,
  cProfile (2.1-2.3, 2.6). Logs `logs/base_cuda.json`, `logs/base_cpu.json`.
- `microbench.py`: costs of an operation, a wait, `isfinite`, the memory
  statistics (2.4).
- `bits_cpu_gpu.py`: CPU against GPU bits of the bookkeeping operations (2.5).
- `patch_d2.py`, `time_d2.py`: the D2 prototype (5.3).
- `route_d_proxy.py`: the route (d) proxy (5.4).
- `bitgrid_ct.py`: `error-ratio-work/impl/bitgrid_er.py` extended with
  `split_rule="accelerate"` cases (two halvings per mesh update), with the
  split cap, the record on the GPU, both modes, a learnable integrand and
  `take_gradient=True`; compared with that plan's `bitcompare_er.py`.
- `time_ab.py`, `run_ab.sh`: base and branch alternated on the GPU.

The planning-stage scripts (`count_turns.py`, `count_ops_by_function.py`,
`compile_bits*.py`, `compile_rows.py`, quadax sources) are in `../scripts/`.

## 8. Documentation

No user-visible behaviour changed: no entry in the CHANGELOG, the user guide
or `CLAUDE.md` (which does not describe the mesh update's index arithmetic).
`per-turn-overhead.md` (branch `performance_benchmark`) item 12 should read
that a turn now costs about 2.2 ms on an A100 and 0.9 ms on the CPU, that
capture and compilation are blocked as in 3.1, and that quadax costs about
0.15-0.3 ms per loop iteration (2.7), not per subinterval.

## 9. Risks and interactions

- **`split_waste` follow-up** (running): it edits the four-way threshold in
  `_adaptively_increase_mesh` and `_splittable` and one conditional of Step 5.
  The change of 5.1 is in the halving loop, below the four-way test, and does
  not touch Step 5's arguments; the two should merge without conflict.
- **`timing_variation`** changes no package code.

## 10. Confidence

- **Sure (measured on one A100 node and its CPU, torch 2.8.0):** the turn
  costs, waits, operations and launches of 2.1-2.3; the micro-costs of 2.4;
  the bit comparisons of 2.5; the D2 prototype's times and bit identity on its
  5 integrands; the route (d) proxy's times; that the change of 5.1 removes
  three operations per mesh update and changes no returned number on the
  grids of "Implemented".
- **Fairly sure (reasoning, not timed end to end):** the per-turn gains of
  5.2, built from 2.4's costs; that route (d) proper would be faster than the
  proxy for many outputs.
- **Unsure:** where route (d)'s crossover lies when its sums stay on the GPU
  (only the proxy's, about 5,000 mesh points, was measured); the pinned-memory
  behaviour of D2 over many calls (PyTorch's pinned cache keeps freed blocks);
  run-to-run variation on other nodes.
- **Not checked:** quadax 0.3.1 itself; Inductor's GPU code; the Performance
  page (not re-run).
- **Mine, not from a source:** every candidate of 5.2, the D2 and route (d)
  prototypes, and the reading of quadax's cost per iteration.
