# Plan: the cost of the reuse mesh built after every call (`optimal-mesh-cost`)

Status: plan, reviewed against `cd3ff34` (2026-11-26) and corrected; being
implemented on branch `optimal_mesh_cost`. Sections 2-4 were written against
branch `fix_conserve_memory`, commit `0590ccf`; their `file:line` references
are of that commit. Section "Review" below says what changed since and what
of the plan still holds; Section 5 is the corrected implementation plan.
Code level of the prototypes: prototype (scratch only).

## Review (2026-11-26, against `cd3ff34`)

Five changes landed after the plan was written (`git log --oneline
0590ccf..cd3ff34`): `endpoint-evaluations`, `call-setup-cost`,
`startup-evaluations`, `turn-device-reads` and the memory options change.
Every claim the plan relies on was checked against `cd3ff34`, and the build
was re-measured by part on an A100 node (`nid200337`, alone on the node,
`impl/time_parts.py`, `impl/logs/time_parts_base_cuda.log`; gk21, float64,
`atol=1e-8`, `rtol=0`, `error_norm="max"`, `take_gradient=False`, medians of
11 calls):

| Part (ms) | damped_sine (11 subintervals) | many_outputs (13, D = 1000) | oscillatory (13,737) |
|---|---|---|---|
| Method swap (`_integration_on_device`, both ways) | 0.008 | 0.010 | 0.013 |
| Prediction (`_merge_inputs`), new solver / warm solver | 0.92 / 0.40 | 1.32 / 0.73 | 3.86 / 2.57 |
| of which building the weight vectors (new solver) | 0.47 | 0.54 | 1.16 |
| of which the node check's `_compute_nodes` | 0.03 | 0.03 | 0.14 |
| Two `_compute_error_ratios` | 0.20 | 0.33 | 0.48 |
| Merge (`_prune_excess_mesh`) | 0.23 | 0.32 | 0.77 |
| Split (`_adaptively_increase_mesh`) | 0.26 | 0.29 | 3.55 |
| of which the mesh dictionary (`_get_mesh_indices`) | 0.015 | 0.018 | 2.82 |
| `_get_optimal_mesh`, new solver / warm solver | 1.77 / 1.23 | 2.41 / 1.82 | 9.14 / 7.66 |
| Whole call, with the build / build replaced by the loop's mesh | 4.57 / 3.14 | 9.03 / 7.10 | 117 / 108 (noisy) |

GPU solver with `result_device="cpu"`; the CPU solver gives the same parts
within 10 % and whole calls of 2.93 / 1.45 ms (`damped_sine`) and 6.28 /
4.38 ms (`many_outputs`). What the review found and changed:

1. **Table moves (3.2 item 5): done.** `call-setup-cost` made
   `_integration_on_device` swap shared converted methods (`MethodClass.to`);
   it costs 0.01 ms. Dropped from this plan.
2. **The node-position check is true by construction (D2 changed).** Since
   `endpoint-evaluations` the build takes every width from the final mesh,
   whose subintervals are exactly the recorded ones in order, and every
   recorded node was placed by `_compute_nodes` on the same two mesh points
   (mesh points are only ever inserted, never moved). The check compared the
   nodes with a recomputation of themselves (to 16 eps, for a record made on
   another device). Instead of gating it by `runtime_checks`, it is removed
   with the `nodes` argument of `_merge_inputs`, `_predict_merged_errors` and
   `_get_optimal_mesh` (12 test calls change, the two tests that fed it
   off-fraction nodes go). Gain 0.03-0.14 ms of `_compute_nodes` plus one read
   of the device per chunk.
3. **No registered method repeats a node fraction any more** (`dopri5` has 5
   distinct nodes since `734f7f7`). So `y[lo:hi][:, first, :]` copies every
   node for every registered method. Measured on the node (`impl/ycopy.py`,
   the two copies plus the two products): 0.24 → 0.08 ms at D = 1000, 3.0 →
   0.43 ms at D = 10^4, 0.47 → 0.21 ms at 13,737 subintervals. D5 is taken in
   a form that removes code instead of adding a branch: `_merged_pair_weights`
   returns its vectors over all C nodes, with weight 0 at a repeated fraction
   (whose stored value equals its first occurrence's), so the prediction
   multiplies the stored values as they are (views; a float32 record is still
   cast) and `first` leaves the return value, the cache (`cache[None]`) and the
   call sites. For the registered methods the vectors are the same numbers;
   for a user method that repeats a fraction the products get extra zero
   terms, which can change the prediction at rounding level. Kept only if the
   grid and a direct comparison of every prediction show no changed bit
   (Section 6). A bmm on a view at an offset against the same values copied
   was equal in 916 of 916 shapes on a login node's CPU (`impl/align.py`; C
   from 2 to 65, D from 1 to 10^4).
4. **The split stage (3.2 item 1): dropped.** Of its 3.55 ms on `oscillatory`,
   2.82 ms is the mesh dictionary, which `record-order` (in progress beside
   this plan) deletes everywhere. What remains of `_adaptively_increase_mesh`
   here is 0.26-0.7 ms, most of it `_splittable` and the insertion that a
   direct insertion would need too; the direct insertion would add a second
   insertion path, or a helper cut out of a function that `record-order` and
   `turn-device-reads` edit, for an estimated 0.1-0.2 ms. The guard
   `_splittable` (a midpoint that rounds onto an end is not inserted) is then
   applied as now, through the same call.
5. **D3 (shared weight cache): not done.** The converted method objects are
   shared and documented as never changing (`_Tableau` is frozen); a mutable
   cache on them would be shared by solvers in different threads, and
   `_cached_merged_pair_weights` clears the cache when it is full, so one
   thread's `cache[k]` could miss a key another thread just cleared
   (`KeyError`), a hazard the per-solver cache does not have. It would also
   add as much code as it removes (an attribute set when a method is
   converted, against the solver's attribute and its reset), and it depends on
   the solver's class constants (`_RATIO_QUANTUM`). What a one-shot call
   pays for the empty cache: 0.47-1.16 ms (the table above; 11-82 distinct
   ratios).
6. **Doubt 1 (state.mesh None, not the loop's final mesh): re-checked and
   upheld.** `impl/count_reuse.py` on `cd3ff34` (20 calls from the previous
   state, the integrand changing a little each call): points of `f` in calls
   2-20, optimal mesh against final mesh, 2,562 against 4,767 (`damped_sine`,
   1.86 times) and 3,990 against 9,408 (moving peak, 2.36 times); the final
   mesh only grows (13 and 29 subintervals at call 20, against 8 and 9).
7. **Doubt 2 (lazy build): rejected, confirmed.** A lazy state would have to
   hold the call's `y` (the largest array of the call) and the solver, and
   would no longer be a small frozen object of a tensor, an int and a bool.
8. **Doubt 3 (the second `_compute_error_ratios`).** The split stage reads
   only the per-subinterval ratios; its pair ratio is now discarded by name
   (`_`), so the first call, before the merge, is the one reader of the pair
   ratio, for `error-ratio-work`.
9. **`examples/pode/trainer.py`: the plan's claim is wrong.** The trainer does
   not discard the mesh: at line 613 it sets
   `self.input_times = self.integral_output.mesh_optimal`, which the next
   iteration passes as the state's mesh. That attribute was removed in 2.0.0,
   so the branch raises `AttributeError` when it runs. The example must not
   pass `build_state_mesh=False`; the stale line is fixed to
   `self.integral_state.mesh` (own commit).
10. **The `record_evaluations=False` warning** names the state's mesh; it is
    not logged when the mesh is not built.
11. **A state whose mesh is None** is already accepted and tested with bounds
    (`test_restart_keeps_every_other_argument` passes
    `IntegrationState(None, 100, False)`). Without bounds the error said only
    "Pass mesh=, or both mesh_init and mesh_final"; it now names the state
    built with `build_state_mesh=False`. The restart after a backward
    out-of-memory error builds its own state from the call's starting mesh and
    repeats every argument, so it keeps the switch.
12. **Stale docs** (doubts 4 and 5): `padaquad/results.py`'s module docstring
    said the result holds the warm-start mesh (own commit). The places that
    say the state's mesh is always a tensor: `IntegrationState.mesh`, the
    README reuse section, `repeated_calls.md` (interval table, checkpoints),
    `how_it_works.md` (`ug-optimal-mesh`), the `integrate` docstring; the
    tests and examples read `state.mesh` only after default calls.
13. **D4**: `predicted-merge-cost.md` is corrected (its evidence used
    `same_f`, removed in 2.0.0), with numbers re-measured with `state=`.

## 1. Summary

- **What is slow.** After the integration loop, every call builds the reuse
  mesh (`state.mesh`; called the optimal mesh in the code) with
  `_get_optimal_mesh`, whether or not the caller passes the state to a later
  call. Measured on an A100 node (GPU solver, record on the CPU, one solver per
  call): 2.0 ms of a 6.0 ms call on `damped_sine`, 2.5 ms of 11.5 ms on
  `many_outputs`, 9-14 ms on `oscillatory` (13,737 subintervals).
- **What the plan changes.** (1) An `integrate` argument that skips the build,
  with `state.mesh = None` (a user decision, D1). (2) A cheaper build that
  returns the same mesh bit for bit: the split stage inserts midpoints directly
  instead of calling `_adaptively_increase_mesh`; the prediction uses views of
  the stored values instead of copies when every node is distinct; the
  node-position check runs only with `runtime_checks`. (3) Optionally, the
  merge weight vectors cached per method, dtype and device for the whole
  process instead of per solver (D3, jointly with the `call-setup-cost` plan).
- **Expected gain (measured, indicative; the node was shared).** Skipping:
  1.7 ms (29 %) of `damped_sine`, 2.4 ms (20 %) of `many_outputs` per call on
  the GPU solver; 1.3 ms (48 %) and 2.4 ms (25 %) on the CPU solver. Cheaper
  build (prototype): 9.2 → 5.7 ms on `oscillatory`, 2.6 → 2.0 ms on
  `many_outputs`, no change on `damped_sine`; the shared weight cache would
  save a further 0.4-1.3 ms per one-shot call (estimated from the measured
  cost of building the vectors).
- **Numbers.** The cheaper build changed no returned number in 164
  integrations (two calls each) on a CPU and on a GPU. The skip changes only
  `state.mesh`, and only when the caller asks for it. Option 2 of
  `predicted-merge-cost.md` would change weights at rounding level; it is not
  recommended now (D4).

## 2. Evidence

### 2.1 The code as it is now

All in `padaquad/quadrature/base.py` unless named otherwise.

- **Where the build runs.** `integrate` calls `_get_optimal_mesh` after the
  loop, inside `with self._integration_on_device(optimal_device)` (1608-1621),
  where `optimal_device = record["nodes"].device` (1608): the CPU by default
  (`result_device="cpu"`), the solver's device with
  `record_evaluations=False`. The mesh is moved to the solver's device and
  detached (1629) and returned as `IntegrationState(mesh_optimal, ...)` (1675).
- **Table moves.** `_integration_on_device` (3188-3216) sets `self.device`
  and calls `self.method.to_device(device)` before the block and again after
  it, when the record is on another device than the solver. For a uniform
  method this reassigns `c`, `b` and `b_error` (`padaquad/methods/_base.py:87-89`):
  three copies each way. It mutates the solver's method object, which is
  harmless only because every solver has its own clone
  (`padaquad/methods/__init__.py:72`, `clone` at `_base.py:424-437`).
- **The merge stage.** `_get_optimal_mesh` (3218-3322) calls
  `_merge_inputs` (3276; defined 2840), which with the default
  `merge_rule="predicted"` calls `_predict_merged_errors` (2987). Per chunk of
  pairs, that function:
  - checks that the nodes sit at the method's node fractions (3057-3063):
    one `_compute_nodes` call, a subtraction, a comparison and one
    `bool(...)` read; this check is not gated by `runtime_checks`;
  - looks up the weight vectors (3066-3068) through
    `_cached_merged_pair_weights` (3085), which builds missing ones with
    `_merged_pair_weights` (2907) and keeps them in `self._merge_weight_cache`
    (created empty at 549 in `_set_solver_dtype`; at most
    `_MERGE_CACHE_SIZE = 256` ratios, 2834; a chunk with more distinct ratios
    builds without caching, 3112-3113);
  - copies the two neighbours' values with advanced indexing,
    `y[lo:hi][:, first, :].to(torch.float64)` (3069-3070); `first` lists the
    distinct node fractions, so for every method except `dopri5` (which repeats
    the fraction 1) the indexing selects every node and only copies;
  - applies the vectors with two `torch.bmm` (3073-3075).
- Then `_compute_error_ratios` computes the pair ratio (3277-3282; only
  `error_ratios_2steps` is used), `_prune_excess_mesh` (3129) merges with
  `_rec_remove` (3324) and `_merge_excess_nodes`
  (`padaquad/quadrature/uniform.py:128`, `variable.py:162`; it allocates with
  `self.device`, `uniform.py:182, 190`).
- **The split stage.** A second `_compute_error_ratios` on the merged record
  (3297-3303; its pair ratio is discarded), `_accept_reject_masks` (3304),
  and `_adaptively_increase_mesh` (3307-3320) with no batch data. Of its work
  only the midpoint insertion (2063-2081) is used; it also allocates trackers,
  clones the mesh, and rebuilds the mesh dictionary
  `self._get_mesh_indices(mesh_new)` (2142), which nobody reads here.
- **One-shot calls.** `padaquad.integrate` builds a new solver per call
  (`padaquad/_frontend.py:192-193`), so its weight cache always starts empty.

### 2.2 What promises `state.mesh` makes (these bound what may change)

- `IntegrationState.mesh` docstring (`padaquad/results.py:35-40`): "the
  optimal mesh of the call that returned the state (low-error subintervals
  merged, failing ones split), detached from autograd, on the solver's device,
  its ends equal to the bounds. None makes a fresh mesh on the bounds, which
  must then be given." The class is frozen (`results.py:20`), "never changes
  after it is made", and `dataclasses.asdict(state)` must survive `torch.save`
  / `torch.load(weights_only=True)` (`results.py:27-31`).
- README "Reusing the mesh and batch size" (`README.md:113-126`, included in
  `docs/user_guide/repeated_calls.md`): "`state.mesh`, the optimal mesh (the
  final mesh with low-error subintervals merged), so the next call needs few
  splits"; "The solver keeps nothing between calls".
- `docs/user_guide/repeated_calls.md`, section "Checkpoints": the state is a
  dictionary of a tensor, an int and a bool.
- `docs/user_guide/how_it_works.md:30-36` (label `ug-optimal-mesh`): "After
  the integration converges, PadaQuad builds the optimal mesh, `state.mesh`".
- `integrate` docstring, Returns (`base.py:901-902`): "state : IntegrationState.
  The optimal mesh and the batch size, for a later call's `state=`"; and
  `record_evaluations` (880-890): "the state's mesh is built with
  `merge_rule="power_law"`".
- `tests/test_snapshots.py:115` pins `n_optimal_mesh = len(state.mesh)`.

### 2.3 How the state is used

- `examples/pode/trainer.py:360-372` keeps only the batch size: it replaces
  the mesh with `dataclasses.replace(self.integral_state, mesh=self.input_times)`
  on every iteration. It pays for a mesh it throws away.
- The restart after an out-of-memory error in a backward pass builds its own
  state from the call's starting mesh (`padaquad/quadrature/memory.py:1805`)
  and calls `integrate` again with the same arguments
  (`memory.py:162-226`); it never reads a built `state.mesh`, and a new
  `integrate` argument is passed through it unchanged.
- `tests/test_prune_and_optimal.py` (24 tests of `_get_optimal_mesh` and the
  prediction), `tests/test_bug_regressions.py` (reuse through the state),
  `examples/quadrature_basics.py`, `examples/verification/merge_prediction.py`.

### 2.4 Measurements (2026-11-25)

Node `nid008404`, A100 (GPU index 3), shared with other agents: wall times
are indicative. gk21, `atol=1e-8`, `rtol=0`, `error_norm="max"`, float64,
`take_gradient=False`, `max_batch=1_000_000`, interval [0, 1], same integrands
as the brief. Each part was timed with a device synchronization around it,
only while the post-loop block ran; medians of 7-9 calls. Scripts in
`$SCRATCH/padaquad_speedup_plans/optimal-mesh-cost/` (Section 7).

Time by part of `_get_optimal_mesh` (ms), GPU solver, `result_device="cpu"`
(the default), a new solver per call (`time_optimal_mesh.py`, `out_cuda.txt`;
`_get_mesh_indices` from `time_parts2.py`, `out2_cuda.txt`):

| Part | damped_sine (11 subintervals) | many_outputs (13, D = 1000) | oscillatory (13,737) |
|---|---|---|---|
| Table moves (`_integration_on_device`, both ways) | 0.15 | 0.15 | 0.16 |
| Prediction (`_merge_inputs`), all | 0.89 | 1.26 | 3.76 |
| of which: building weight vectors (`_merged_pair_weights`) | 0.40 (11 ratios) | 0.40 (12) | 0.95 (82) |
| of which: node check's `_compute_nodes` | 0.06 | 0.06 | 0.24 |
| Pair ratio and split ratios (`_compute_error_ratios`, 2 calls) | 0.32 | 0.42 | 0.63 |
| Merge (`_prune_excess_mesh`) | 0.36 | 0.42 | 1.49 |
| Split masks (`_accept_reject_masks`) | 0.04 | 0.04 | 0.08 |
| Split (`_adaptively_increase_mesh`) | 0.23 | 0.23 | 3.53 |
| of which: mesh dictionary (`_get_mesh_indices`) | 0.02 | 0.02 | 3.0-3.7 |
| `_get_optimal_mesh` total | 2.03 | 2.55 | 10.3 |

- **Same solver, second call (cache warm).** The totals drop to 1.54, 2.27
  and 8.73 ms: the vectors are not rebuilt. So the empty cache of a one-shot
  call costs 0.3-1.6 ms.
- **Record on the GPU** (`result_device="cuda"`): 3.9, 4.1 and 7.0 ms. Many
  small operations cost more on the GPU than on the CPU for small meshes; the
  large mesh gains. Running the build on the solver's device is therefore not
  a uniform gain.
- **CPU solver** (`out_cpu.txt`): 1.58, 2.26 and 10.6 ms.
- **The whole call with the build replaced by "return the loop's mesh"**
  (`time_parts2.py`, uninstrumented): GPU solver 5.97 → 4.24 ms
  (`damped_sine`), 11.54 → 9.18 ms (`many_outputs`); `oscillatory` too noisy to
  read (median 146 → 130 ms, minimum 127 → 93 ms). CPU solver 2.78 → 1.44 ms
  and 9.43 → 7.01 ms.
- **Many distinct width ratios** (`many_ratios.py`: random starting mesh of
  10^4 subintervals, `damped_sine`, `atol = rtol = 1e-8`, GPU solver, record on
  the CPU, default thread count): `_get_optimal_mesh` took 60 ms (gk21) and
  260 ms (cc65) on the first call, 27 and 134 ms on the warm call from its state,
  of which building the vectors was about 47 and 250 ms (first) and 21 and
  126 ms (warm); `power_law` took 2.4-4.0 ms. About 9 µs per ratio for gk21
  and 50 µs for cc65 here; `predicted-merge-cost.md` gives 2 and 20 µs on 8
  threads, a different thread setting (not compared further).
- **Batch independence of the vectors** (`batch_dep.py`, CPU and GPU):
  `_merged_pair_weights` gave the same vector for a ratio bit for bit whether
  it was built alone, in blocks of 7, or with 299 others (gk21, cc65, dopri5,
  gk7; float64 and float32; 300 random ratios). This is what makes a shared
  cache safe (Section 3.3).
- **Reusing the loop's final mesh instead of the optimal one**
  (`count_reuse.py`, login node, counts only): 20 calls, each from the
  previous state, with the integrand changing a little each call. Points of
  `f` in calls 2-20: `damped_sine` 2,825 with the optimal mesh against 5,240
  with the final mesh (1.9 times); a moving narrow peak 4,389 against 10,323
  (2.4 times). Without merges the mesh only grows (8 against 13, and 9
  against 29 subintervals after 20 calls).

### 2.5 The prototype of the cheaper build

`proto/` (made by `make_proto.py` from a copy of `0590ccf`) has three edits
in `_get_optimal_mesh` and `_predict_merged_errors`: the node check under
`self.runtime_checks`; `y[lo:hi]` instead of `y[lo:hi][:, first, :]` when
`len(first) == y.shape[1]`; and the split stage as
`split = nonzero(remove_mask)`, `midpoints = 0.5 * (mesh[split] + mesh[split + 1])`,
merged into the mesh in order. `compare_proto.py` ran 164 integrations (gk21,
cc65, dopri5, bosh3, gk7; float64 and float32; absolute and cumulative mode;
9 integrands), each followed by a warm call from its state, and saved both
`state.mesh` and the warm call's integral. Prototype against base: **0 of 164
differ, bit for bit, on the CPU and on the GPU** (`diff_pt.py`). Time of
`_get_optimal_mesh` (GPU solver, record on the CPU, one solver per call; base
then prototype): `damped_sine` 1.57 / 1.71 ms (no change within noise),
`many_outputs` 2.58 / 1.99 ms, `oscillatory` 9.16 / 5.69 ms.

## 3. The design

### 3.1 Not building the mesh when it is not used (question 1)

Three options were compared.

- **(a) An `integrate` argument that skips it** (recommended; D1). With the
  argument off, `integrate` does not enter the block at 1598-1629 and returns
  `IntegrationState(None, self._batch, self._batch_converged)`. Code: one
  keyword argument with its docstring entry, one `if` around the existing
  block, and nothing in `IntegrationState`, which already accepts `mesh=None`
  and documents what it means (`results.py:39-40`). `_frontend.integrate`
  picks the argument up by itself (`_CALL_OPTIONS` is read from the signature,
  `_frontend.py:106-108`). The warning about `record_evaluations=False` and
  `"power_law"` (1059-1067) is skipped when the mesh is not built.
  - **`state.mesh = None` against the loop's final mesh.** The final mesh is
    a valid starting mesh, but a caller who reuses it gets no merges: measured
    1.9-2.4 times the points of `f` over 19 warm calls, with a mesh that only
    grows (Section 2.4). `f` can be very expensive, so handing out a mesh that
    looks reusable but costs twice the evaluations is the wrong default for a
    "skip" switch. None fails loudly instead: a later `state=state` call
    without bounds raises the existing error ("None makes a fresh mesh on the
    bounds, which must then be given"), and with bounds it starts fresh. The
    final mesh is kept alive in either case only as long as needed: with None,
    nothing extra.
- **(b) Building it lazily on first access of `state.mesh`** (rejected). The
  state would have to keep the sorted record's `nodes` [N, C, T], `y`
  [N, C, D], `mesh_quadratures`, `mesh_quadrature_errors`, the integral and the
  loop's mesh, plus a reference to the solver (its method, `merge_rule`,
  tolerances and error norm at the time of the call, which a later
  constructor-level change must not alter). Costs:
  - Memory: `y` is the largest array of the call. With
    `result_device="cpu"` chosen because `y` does not fit the GPU, or with
    `trade_speed_for_memory=True`, a state kept for the next call (the normal
    pattern in a training loop) would hold the previous call's `y` alive after
    the user dropped the result, so two calls' values would be held at once.
    With `record_evaluations=False` only the small fields are needed
    (`"power_law"`), so the cost would differ between options, which is hard
    to document.
  - It breaks three promises: the state is frozen and "never changes after it
    is made"; `dataclasses.asdict(state)` would no longer be a tensor, an int
    and a bool (Checkpoints section); and "the solver keeps nothing between
    calls" (the state would reference the solver).
  - Least code: a dataclass field cannot be a lazy property without a custom
    descriptor, `__getattr__` or a non-field attribute; all of these are more
    code than (a).
- **(c) Keeping it always, but cheaper** (recommended as well, independent of
  D1). Section 3.2. It cannot reach zero: the measured floor on small meshes
  is about 1.1-1.6 ms of small tensor operations on the CPU.

Recommendation: (a) with `state.mesh = None`, default "build" (no change for
existing callers), and (c). The argument's name is the user's choice; a
proposal is `build_state_mesh: bool = True`. Two alternatives: `state_mesh`
(shorter, but reads like a mesh, not a switch); `reuse_mesh` (was the name of a
removed option, `docs/user_guide/troubleshooting.md:78`, so it would confuse).

### 3.2 Making it cheaper when it is built (question 2)

For each item: what changes, the gain, and whether results stay identical bit
for bit.

1. **Split stage without `_adaptively_increase_mesh`** (recommended).
   The stage needs only "insert the midpoint `0.5 * (left + right)` of every
   subinterval in `remove_mask`". Move lines 2063-2081 of
   `_adaptively_increase_mesh` (allocate the new mesh, `idx_offset`, transfer,
   midpoints) into one function `_insert_midpoints(mesh, split_idxs)`
   returning `(mesh_new, idxs_transfer, idxs_new)`, called by
   `_adaptively_increase_mesh` (which still needs `idxs_transfer` and
   `idxs_new` for the trackers and split counts) and by `_get_optimal_mesh`
   (which uses only `mesh_new`). One definition of the midpoint insertion, two
   callers, so the split is justified by reuse. `_get_optimal_mesh` then drops
   the `mesh_idxs`/`mesh_trackers` arguments it builds only for that call and
   the mesh dictionary (3.0-3.7 ms on `oscillatory`). Identical bit for bit:
   the same formula on the same two mesh points (2079 reads
   `mesh_new[idxs_new ± 1]`, which are `mesh[split]` and `mesh[split + 1]`);
   the prototype used a different way to interleave and was identical on all
   164 cases. Owner of `_adaptively_increase_mesh`'s internals is
   `turn-device-reads` (Section 9).
2. **Second `_compute_error_ratios`** (keep; change only through
   `error-ratio-work`). The split decision needs each pruned subinterval's
   ratio against the final integral, and a merged subinterval's ratio cannot
   be derived bit for bit from its halves' ratios (the norm of a sum is not the
   sum of norms; in cumulative mode the running sums are re-added in another
   order). Its pair-ratio output is discarded (3297), and the first call's
   single ratios are discarded (3277): with `error-ratio-work`'s "pair ratio
   on request", call 1 asks for the pair ratio only and call 2 for none.
   Rejected: skipping the split stage when `converged` is True. A merged
   subinterval can still fail when the rounding floor sets the tolerance
   (`_get_optimal_mesh` docstring, 3239-3242), so the skip could change the
   mesh in that case; and the stage costs 0.04-0.6 ms once item 1 is done.
3. **Views of `y` instead of the two indexed copies** (recommended, with a
   check). When `len(first) == C` (every method except `dopri5`),
   `y[lo:hi][:, first, :]` equals `y[lo:hi]`, and `.to(torch.float64)` is a
   no-op for float64 values, so `y_l` and `y_r` become views and no memory is
   allocated per chunk. For float32 values the cast still copies (needed:
   the vectors are float64). The values the products see are identical.
   Caution: the view starts at an offset into `y`, so its memory alignment
   differs from a fresh copy's; Intel oneMKL documents that results of some of
   its routines can depend on the alignment of the data unless its Conditional
   Numerical Reproducibility mode is set (Intel, *oneMKL Developer Guide*,
   "Obtaining Numerically Reproducible Results"; I have not checked which
   routine `torch.bmm` on float64 CPU tensors calls). The 164 cases were
   identical, but they mostly have D = 1; Step 6 adds `many_outputs`-like
   cases (D up to 10^4) to the grid. If any case differs, drop this item: it
   is about 0.5 ms on `many_outputs` (estimated from the prototype's 0.6 ms
   gain, which also includes item 4).
4. **Node-position check under `runtime_checks`** (recommended). It is an
   invariant check (nodes always come from `_compute_nodes`, 3035-3036), like
   the other checks gated by `runtime_checks` (brief, item 6). It only raises,
   so results are identical. Gain 0.1-0.4 ms per call. The two tests that
   expect the error (`test_predicted_needs_nodes_at_the_fractions`,
   `test_predicted_node_check_is_relative_to_each_subinterval`) use a solver
   with `runtime_checks=True` (the constructor default) and keep passing.
5. **Table moves** (0.15 ms; recommended only through the shared method cache
   of `call-setup-cost`). Options:
   - Keep a second copy of the small tables on `result_device`, made once per
     process: this is exactly what a module-level cache of converted methods,
     keyed by (registry method, dtype, device), gives. `_integration_on_device`
     would then swap `self.method` for the cached copy on `device` and back,
     instead of calling `to_device` twice. No copies after the first call in
     a process, and the method object is no longer mutated, which is a
     precondition for `call-setup-cost` to share method objects between
     solvers at all (today `_integration_on_device` mutates the solver's
     method in place, 3211 and 3216).
   - Run the build on the solver's device instead (move the record's fields
     there): rejected. `y` can be far larger than the GPU when
     `result_device="cpu"` was chosen for that reason, and on small meshes the
     GPU run was slower (3.9 against 2.0 ms).
   - Take only the needed numbers to the record's device (`.to(device)` inside
     `_compute_nodes`, `UniformMethodClass.weights` and `_merge_excess_nodes`):
     rejected; it scatters device handling over three functions and still
     copies on every call.
   Identical bit for bit: the tables are the same numbers on the same device.

### 3.3 The weight cache for one-shot calls (question 4)

- **Cost now.** Measured: building the vectors on an empty cache costs
  0.40 ms (11-12 ratios) and 0.95-1.1 ms (60-80 ratios) per call, of a
  2.0-10.6 ms build and a 6-12 ms call on small integrands (5-7 % of the
  call). Almost all of it is the fixed cost of about 40 small tensor
  operations in `_merged_pair_weights`, not the per-ratio arithmetic.
- **A module-level cache** keyed by (method, dtype, device) would give every
  `padaquad.integrate` call, and every new solver, the vectors of earlier
  calls in the process. Requirements:
  - **Key by the registry object, not the method name.** Users add or replace
    methods under a name (`docs/development/adding_a_method.md:52`,
    `UNIFORM_METHODS["simpson"] = simpson`), so a name can map to another
    method later in the process.
  - **Results must not depend on process history.** The vectors for a ratio
    must be the same whether built alone or with others; measured identical
    bit for bit (Section 2.4), on CPU and GPU, for four methods and both
    dtypes. A test must keep this true (Section 6). The current per-solver
    cache already relies on it within a solver.
  - **Bounded**: the existing `_MERGE_CACHE_SIZE` per key, plus a bound on the
    number of keys (methods × dtypes × devices in a process is small).
  - **Concurrency**: two threads may build the same entry twice; dictionary
    assignment in CPython keeps the dictionary valid, and both builds give the
    same vectors, so this wastes time only (my reasoning; PyTorch's own
    thread-safety is not involved).
- **Shared interface with `call-setup-cost`.** If that plan caches converted
  method objects per (registry method, dtype, device) and lets solvers share
  them read-only, the merge cache becomes one attribute of that cached entry,
  and `self._merge_weight_cache` (245, 549) disappears: one source of truth,
  no new module-level dictionary of its own. If that plan keeps one clone per
  solver, a module-level cache only for the vectors is not worth its code for
  0.4-1.3 ms on one-shot calls (D3), and the per-solver cache stays.
  The interface this plan assumes: a function (today `_get_method`,
  `padaquad/methods/__init__.py:72`) that returns the method for
  (sampling type, name, dtype, device), the same object on every call for the
  same registry entry, never mutated after it is returned.

### 3.4 `predicted-merge-cost.md` (question 3)

Its statements against the current code:

- "builds weight vectors for each distinct ratio ... (`_merged_pair_weights`)":
  true (2907-2985, called from 3124).
- "O(C · U) per ratio": true; the temporaries are [R, C, U] and
  [R, 2, C, U] (2965-2983).
- "about 2 µs for gk21 and 20 µs for cc65 on 8 threads": not reproduced here;
  measured 9 and 50 µs per ratio on the GPU node's CPU with the default thread
  count on 10^4-subinterval random meshes, and on small meshes the fixed cost
  (0.4 ms) dominates. The two settings differ, so neither number contradicts
  the other; the document should state the thread count and that a fixed cost
  exists.
- "the vectors are cached on the solver": true (3114-3125).
- "the cache is bypassed when a chunk of pairs has more than 256 ratios":
  true (3112-3113, per chunk of `_MERGE_CHUNK // D` pairs, 3052).
- "The CHANGELOG entry of 1.4.1 ("Cost")": true (`CHANGELOG.md:72`).
- **Stale:** the Evidence section measures warm calls with `same_f=True`,
  which was removed in 2.0.0 (warm calls now pass `state=state`); the numbers
  should be re-measured with `state=` (`many_ratios.py` does this).

Its options:

- **Option 2, the vectors with two batched matrix products** (recommended:
  not now; D4). Summing `b` and `b_err` times the interpolation basis by a
  matrix product changes the order of the additions, so the vectors change by
  rounding-level amounts (the document reports at most 5.6e-17). Effect:
  - Merge decisions: a pair merges when its pair ratio is below `remove_cut`
    (`_prune_excess_mesh`, 3172-3174). A change of order 1e-16 relative moves
    the ratio by about that much, so only pairs whose ratio is within about
    1e-16 of the cutoff can flip. Rare, but not impossible, and a flip changes
    `state.mesh` and so the next call's points and integral.
  - Snapshots: `tests/test_snapshots.py` builds one call per cell and pins
    `integral`, `integral_error` (not affected: the build runs after them) and
    `n_optimal_mesh` (affected by a flip, by one mesh point; it is pinned only
    where the error estimate is above rounding, `test_snapshots.py:185-209`).
    A flip would need a regeneration and an explanation.
  - The grid harness's warm calls would show any flip.
  - Gain: halves the per-ratio cost, which matters only on meshes with many
    distinct ratios (a random starting mesh with large `N_init_steps`; 47 of
    60 ms there for gk21, 250 of 260 ms for cc65). It does nothing for the
    common case, where the fixed cost dominates.
  Since the common case gains nothing and the change is not bit-identical, it
  should wait until a user needs large random starting meshes. If it is done,
  do it as its own change, with the grid harness listing every flipped pair.
- **Option 3, fall back to `"power_law"` above some number of distinct
  ratios** (recommended: no). It changes results for exactly the meshes where
  `"predicted"` matters most (wide subintervals of varied width, the
  pre-asymptotic merges `"power_law"` refuses; `CHANGELOG.md:59-72`), and
  makes the mesh depend on a threshold. With D1, a user who does not want the
  cost can skip the build or choose `merge_rule="power_law"` explicitly.

## 4. Decisions for the user

The plan's options as written (2026-11-25), and, after each, the default
taken in the review (Section "Review"):

- **D1. A switch to skip the state's mesh.** Options: (a) an `integrate`
  argument, `state.mesh = None` when off (recommended); (a') the same, with
  the loop's final mesh (not recommended: 1.9-2.4 times the points of `f` if
  reused); (b) lazy (not recommended: memory, and it breaks three
  documented promises); no switch (keep paying 1.3-2.4 ms per small call).
  Default: build (no change for existing callers). Name: `build_state_mesh`
  proposed. Should `examples/pode/trainer.py` pass it (it discards the mesh)?
  Recommended: yes. *Taken: (a), `build_state_mesh=True` by default, in its
  own commit. The trainer does not discard the mesh (Review, item 9): it does
  not pass the switch.*
- **D2. Gate the node-position check by `runtime_checks`.** Recommended: yes.
  *Taken instead: remove the check and the `nodes` argument (Review, item 2).*
- **D3. Share the merge weight vectors across solvers.** Recommended: only
  together with `call-setup-cost`'s shared method cache; otherwise keep the
  per-solver cache. *Taken: per-solver cache kept (Review, item 5).*
- **D4. Options 2 and 3 of `predicted-merge-cost.md`.** Recommended: option 2
  deferred (rounding-level changes, no gain in the common case); option 3
  rejected. *Taken: neither implemented; the document is corrected.*
- **D5. Views of `y` (item 3 of Section 3.2).** Recommended: yes if the grid
  harness with large D shows no bit change; otherwise drop it. *Taken: yes,
  as vectors over all C nodes (Review, item 3), if the bit checks pass.*

## 5. Implementation plan (corrected 2026-11-26)

Each step is one commit, in this order; after each, run
`tests/test_prune_and_optimal.py` and `tests/test_snapshots.py`. All code is
in `padaquad/quadrature/base.py` unless named otherwise.

1. **Remove the node-position check (D2).** Delete the check in
   `_predict_merged_errors` (the `_compute_nodes` call, `eps`, the
   `ValueError` and its Raises entry) and the `nodes` argument of
   `_predict_merged_errors`, `_merge_inputs` (whose `len(nodes) < 2` becomes
   `len(mesh) < 3`) and `_get_optimal_mesh`; `integrate` stops passing
   `record["nodes"]` (it still reads the record's device from it).
   - Tests (`tests/test_prune_and_optimal.py`): the calls lose `nodes`
     (`_record` returns the values and contributions only); delete
     `test_predicted_needs_nodes_at_the_fractions` and
     `test_predicted_node_check_is_relative_to_each_subinterval`;
     `test_predicted_works_in_bounded_chunks` counts the chunks through
     `_cached_merged_pair_weights` (once per chunk) instead of through the
     check's `_compute_nodes`, and asserts their exact number.
2. **Vectors over all C nodes, values used as stored (D5).**
   `_merged_pair_weights` writes its [R, 2, 2, U] vectors into zeros of
   shape [R, 2, 2, C] at the columns `first` and returns that one tensor;
   `_cached_merged_pair_weights` returns one tensor and loses `cache[None]`
   (the cache's key type becomes `int`); `_predict_merged_errors` uses
   `y[lo:hi].to(torch.float64)` and `y[lo + 1 : hi + 1].to(torch.float64)`.
   - Tests: `test_merged_pair_weights_exact_for_polynomials` applies the
     vectors to the values at all C nodes; `test_predicted_detaches_and_caches`
     counts 2 cache entries (ratios 1 and 2); the eviction and GPU tests lose
     the `None` entry; a new test pins a method with a repeated fraction: its
     prediction equals that of the same method without the repeat (to
     rounding).
   - Proof: the grid and every prediction bit-equal to `cd3ff34` (Section 6),
     CPU and CUDA, D up to 10^4. If any differs, drop this step.
3. **The split stage reads only the per-subinterval ratios** (doubt 3):
   `_get_optimal_mesh`'s second `_compute_error_ratios` unpacks the pair ratio
   as `_`, and a comment at the first says it is the one reader of the pair
   ratio. Folded into step 1's commit (one line).
4. **The switch (D1)**, in its own commit. `integrate(...,
   build_state_mesh=True)`, keyword-only, last; the post-loop block
   (`_integration_on_device` ... `mesh_optimal.to(self.device).detach()`)
   runs only when it is True, otherwise `mesh_optimal = None`; the
   `record_evaluations=False` warning also requires it; the error of a call
   without a mesh and without both bounds names a state built with
   `build_state_mesh=False`. `padaquad.integrate` passes it through
   (`_CALL_OPTIONS` is read from the signature).
   - Test (`tests/test_prune_and_optimal.py`): uniform and variable,
     `take_gradient` both, CPU and CUDA: with the switch off `state.mesh is
     None`, `_get_optimal_mesh` is not called, every `IntegrationResult` field,
     the batch size and the parameter's gradient equal the default call's bit
     for bit; a later call with the state and no bounds raises the
     `ValueError` that names the switch; with bounds it equals a fresh call
     with the state's batch size, bit for bit; through `padaquad.integrate`
     too. `tests/test_memory_state_adversarial.py::test_restart_keeps_every_other_argument`
     passes the switch off and checks that the restarted call returns no mesh.
   - Documentation (Section 8).
5. **`examples/pode/trainer.py`**: `self.integral_output.mesh_optimal` →
   `self.integral_state.mesh` (own commit; a 2.0.0 leftover).
6. **`padaquad/results.py` module docstring** (own commit).
7. **`docs/issues/predicted-merge-cost.md`** (D4): replace the `same_f`
   evidence by numbers measured with `state=` (`impl/many_ratios.py`), state
   the thread count and the fixed cost on small meshes.

Not done (Review): the direct midpoint insertion (item 4), gating the check
(replaced by its removal), the shared weight cache (item 5), the table moves
(done by `call-setup-cost`).

## 6. Tests, and how to show results are unchanged

- Steps 1-3 must change no number. Evidence:
  - `tests/test_snapshots.py` passes unchanged.
  - `impl/bitgrid_om.py` (from `turn-device-reads/impl/bitgrid.py`): every
    first call followed by a warm call from its state; uniform and variable
    methods with and without end nodes; absolute and cumulative mode,
    `max_adaptive_splits`, `merge_rule="power_law"`, `take_gradient` both;
    the memory options and `record_evaluations=False`; float32; D = 3
    learnable integrands with gradients; bounds that require grad; D = 1000
    and 10^4 in float64 and float32 with the record on the CPU and on the
    solver's device; random starting meshes of 10^4 subintervals. It also
    saves every merge prediction (`_merge_inputs`' output), so a rounding
    change shows even where it flips no merge. Base `cd3ff34` against the
    branch, CPU and CUDA, compared by `impl/bitcompare.py`.
  - `impl/align.py`: `torch.bmm` on a view at an offset against the same
    values copied, CPU and CUDA.
- Step 4 changes `state.mesh` only when the switch is off; its test compares
  every result field with the default call bit for bit.
- Not affected by any step: the integral, its error, the record, the split
  statistics and `converged` (all computed before the post-loop block).

## 7. How to measure the gain

Scripts (prototype level) in
`$SCRATCH/padaquad_speedup_plans/optimal-mesh-cost/`
(`/pscratch/sd/k/khegazy/padaquad_speedup_plans/optimal-mesh-cost/`):

- `time_optimal_mesh.py {cuda|cpu}`: time of each part of the build, new
  solver against same solver, `result_device` CPU and GPU (`out_cuda.txt`,
  `out_cpu.txt`).
- `time_parts2.py {cuda|cpu}`: finer parts, and the whole call with the build
  replaced by the loop's mesh (`out2_cuda.txt`, `out2_cpu.txt`).
- `compare_proto.py {cpu|cuda} out.pt time` and `diff_pt.py a.pt b.pt`: the
  164-case bit comparison and the build's time; run from the directory that
  holds the package to test (run `proto/compare_proto.py` for the prototype;
  Python puts the script's directory first on the path).
- `count_reuse.py`: points of `f` over 20 warm calls, optimal against final
  mesh.
- `batch_dep.py` (CPU), `batch_dep_cuda.py`: batch independence of the
  vectors.
- `many_ratios.py cuda`: a random starting mesh of 10^4 subintervals.
- `make_proto.py`: the prototype's edits.

Before and after, on a compute node, one GPU: `time_optimal_mesh.py cuda` for
the build's parts, `time_parts2.py cuda` for the whole call, and the brief's
`profile_baseline.py cuda` for the call's wall time and host waits (the table
moves are 9 waits there). Numbers to compare: `_get_optimal_mesh` total per
integrand, cold and warm; the call's median wall time with and without the
switch. Delete the scratch directory when the work is merged.

## 8. Documentation to update

- `integrate` docstring (`base.py`): the new argument; Returns `state`
  (901-902): "or None when not built"; `record_evaluations` (880-890).
- `IntegrationState.mesh` docstring (`results.py:35-40`): None also when the
  call skipped the build.
- `padaquad/results.py:4-6` module docstring: says `IntegrationResult`
  holds "the mesh to warm-start the next call from"; stale since 2.0.0 (the
  state holds it). Fix in passing.
- README "Reusing the mesh and batch size" (`README.md:113-126`) and
  `docs/user_guide/repeated_calls.md`: one sentence on the switch and what
  None means for the next call.
- `docs/user_guide/how_it_works.md` (`ug-optimal-mesh`, 30-36): "builds"
  → "builds, unless the call skips it".
- `CHANGELOG.md`: a new Unreleased section (the file's top section is 2.0.0):
  Added (the switch, its default, measured gain), Changed (the build is
  faster; same mesh bit for bit).
- `CLAUDE.md`: "Post-convergence" (step 3 of the core algorithm), "Reusing
  the mesh and batch size", and the `merge_rule` paragraph (cache location,
  node check under `runtime_checks`).
- `docs/issues/predicted-merge-cost.md`: the stale `same_f` evidence, the
  new numbers (fixed cost on small meshes; per-ratio cost with thread count),
  and the decisions D3/D4.
- `_cached_merged_pair_weights` docstring (3088-3110), if D3 moves the cache.

## 9. Risks, and interactions with the other plans

- **`record-order`** gives this code the sorted record's `nodes`, `y`,
  `mesh_quadratures`, `mesh_quadrature_errors` and the loop's mesh, as now.
  If it removes the mesh dictionary everywhere, the 3.0-3.7 ms of
  `_get_mesh_indices` in the split stage goes with it, and step 1's own gain
  on `oscillatory` shrinks to about 0.5 ms; step 1 is still worth it for one
  definition of the midpoint insertion and a simpler `_get_optimal_mesh`.
- **`error-ratio-work`**: this plan is the one reader of the pair ratio
  (`error_ratios_2steps`, 3277). Interface assumed: `_compute_error_ratios`
  computes the pair ratio only on request; call 1 (3277) requests only it,
  call 2 (3297) does not request it. With `merge_prediction` given, the pair
  ratio must stay the same bit for bit.
- **`turn-device-reads`** owns `_adaptively_increase_mesh`. Step 1 moves its
  midpoint insertion into `_insert_midpoints`; after step 1 the post-loop no
  longer calls `_adaptively_increase_mesh`, so that plan does not need to keep
  its no-batch-data path for this caller (Step 7 of the loop still uses it
  with an empty batch).
- **`call-setup-cost`**: Section 3.3 (shared method cache; this plan's step 5
  depends on it, and its sharing of method objects depends on
  `_integration_on_device` no longer mutating the method).
- **`compiled-turn`**: not affected (the build runs once, after the loop).
- **Risks.** A user who sets the switch off and then passes the state without
  bounds gets an error on the next call; the docstring and the error message
  must say why. The views of `y` could change bits through memory alignment
  (D5). A shared cache must stay bounded and history-independent (tests in
  step 5).

## 10. Confidence

- **Sure (read in the code, or measured with a bit comparison):** where the
  time goes (Section 2.4, with the noise caveat of a shared node); that the
  three cheaper-build edits gave the same `state.mesh` and warm-call integral
  on 164 cases on CPU and GPU; that the vectors are batch-independent for the
  four methods tested; that reusing the loop's final mesh costs 1.9-2.4 times
  the evaluations in the two series tested; that lazy building would hold `y`
  and conflicts with the documented frozen, saveable state.
- **Less sure:** the exact wall-time gains (one shared node, medians of 7-9);
  that views of `y` stay bit-identical for large D on every CPU (alignment
  dependence of the BLAS routine not checked); the per-ratio costs, which
  depend on the thread count; whether `interpolatory3_variable` vectors are
  batch-independent (not tested; step 5's test covers it); how often option 2
  of `predicted-merge-cost.md` would flip a merge (not measured).
- **My own reasoning, not measured:** the concurrency remark in Section 3.3;
  the claim that a 1e-16 change can flip only pairs within about 1e-16 of the
  cutoff.
