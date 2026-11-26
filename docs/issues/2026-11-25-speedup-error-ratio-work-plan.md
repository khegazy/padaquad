# Plan: error-ratio work the default mode never uses (`error-ratio-work`)

**Status:** reviewed and corrected against `c14d6ff` (2026-11-26, branch
`error_ratio_work`); sections 1 and 2 are the planning record at `0590ccf`
(line numbers there are of that commit), sections 3 onward are the corrected
plan. Prototypes and measuring scripts of the planning stage:
`$SCRATCH/padaquad_speedup_plans/error-ratio-work/`; of the implementation:
its `impl/` directory.

## Review (2026-11-26, against `c14d6ff`)

Eight merged changes landed after the plan was written (`git log
0590ccf..c14d6ff`), among them `turn_device_reads`, `record_order` and
`optimal_mesh_cost`, which rewrote most of the code the plan cites. Every
claim was checked against the code at `c14d6ff` and by measurement on an
A100 (node `nid008229`, `impl/count_fns.py`, `impl/probe.py`,
`impl/probe2.py`, `impl/mem_ratios.py`; logs in `impl/logs/`). What was
wrong or is now different, and what changed in this file:

1. **Item A is smaller than planned.** There is no `_mesh_order` and no
   dictionary any more: the loop merges the batch into the record's order
   with `_merge_positions` (integer mesh indices) and `_insert_sorted_results`,
   which read the device nowhere. In absolute mode this merge and its
   `cumsum` are still computed every turn and never read (measured on
   `chirp`, `"max"`: 8 + 2 launches per turn in the two functions, plus the
   loop's `cumsum` and index, plus a second `cumsum` and index for the
   re-check), but they cost no host wait: the three waits per turn the plan
   counted were `_mesh_order`'s, which `record_order` removed. Step 7
   already passes no running sum (`record_order`). The final check's
   `cumsum` is the helper's own fallback, so it passes `None`. The first
   batch's special case (`len(record) == 0`) is not needed: an empty record
   merges like any other (one `cumsum` and the same values).
2. **The second `_mesh_order` and `MethodOutput.mesh_left`** are gone
   (`record_order` replaced them by `step_idxs` and `mesh_index`); nothing of
   3.1's lookup remains to do.
3. **Item B's interface (decision 3) is changed.** The plan's flag
   `pairs: bool` adds a parameter and a branch in `_compute_error_ratios`
   besides one in each helper. The pair ratio's one reader,
   `_get_optimal_mesh`, now builds the pair errors itself (its errors are
   already absolute; `integrate` passes `torch.abs` of the record) and passes
   them in place of `merge_prediction`, which only it passed: the argument is
   renamed `pair_errors`, `_compute_error_ratios` loses its three lines of
   pair arithmetic, and each helper computes the pair ratio only when given
   pair errors (one early return each). The alternative in the task message,
   computing the pair ratio where it is used, needs the tolerance of each of
   the four cases (absolute or cumulative, norm family or
   `"failure_fraction"`) outside the helpers (a new function returning
   tolerance and floor, or a copy of the tolerance code) and then its own
   family and mode branches for the pair-specific parts (the shifted
   cumulative tolerance, the summed floor, the `amax` of
   `"failure_fraction"`): more code and as many branches. Computing the pair
   ratio's tolerance from `cum[1:]` instead of slicing the full tolerance was
   also rejected: a row's reduction can depend on which rows are in the
   array (item 6), so the merge decisions could change.
4. **Item C is reduced to the loop's read.** At `c14d6ff` the re-check reads
   the device twice per turn: `_failed_record_positions` indexes the mesh
   positions with a boolean mask (one read), and the loop reads
   `bool(torch.all(record_keep_mask))`. The second read is replaced by the
   size of `extra_split_idxs`, which that index already put on the host (no
   new code). The plan's "look up only failures" would read once with
   `nonzero` and once more for `positions[splittable]` (a boolean index waits
   even when empty: `impl/probe.py`), or add a branch; it would save only the
   guard `_splittable` over the record, 11 launches per turn on `chirp`. Not
   done; see "Not done".
5. **Item D's mechanics change.** `MethodOutput` and the mesh update
   (`_adaptively_increase_mesh`, owned by the parallel `split_waste` branch)
   are not touched: the loop filters the stored values with `keep_idx`, as it
   already filters `error_ratios`, and `_record_results` appends them as one
   more of `_RECORD_LOOP_KEYS` (`_DEVICE_RESIDENT_KEYS`, `_insert_sorted_results`
   for the record and the per-turn merge are gone). `_sort_record` orders
   them and `_remove_failed_record_panels` filters them like the other loop
   fields; `integrate` drops them after the final check. Since the
   cumulative-mode fallback (the record's own `cumsum`) needs the
   contributions, it moves from the two helpers into `_compute_error_ratios`,
   so the norm helper takes the stored values alone.
6. **Row reductions.** The plan's `rowwise.py`, rerun on this node: 2 rows
   of `"2"`/`"rms"` differ between a slice and the whole array on CUDA at
   D = 1,000 and on the CPU at D = 100,000, none at D = 2 and 50 (confirmed).
   A second test with views instead of copies found none, so it depends on
   the slice's shape and copy.
7. **Item E is larger than the plan says.** Not only the rounding floor and
   the running sums carry the autograd graph into the error ratios: with
   `take_gradient=False` the running integral does too (`record["integral"]`
   keeps the graph). (The error estimates themselves are already detached by
   `_calculate_integral`; an earlier version of this review said otherwise.)
   The contributions, the recorded integral in the reference and the running
   sums are detached.
8. **Measurements at `c14d6ff`** replace section 2.3's for the decisions:
   section 2.4. The plan's base numbers (619 waits, 3,731 launches on
   `chirp`) are of `0590ccf`; at `c14d6ff` `chirp` takes 103 waits and 2,823
   launches.
9. **Temporaries (task doubt 1)**, counted from the code and measured
   (section 2.4): a call of `_compute_error_ratios` of the norm family holds
   up to 4 arrays of [N, D] at once (|e|, the pair errors and their first
   sum, |s| and the floor, the squares of `"2"`), `"failure_fraction"` up to
   about 8 (also `ones_like` and the broadcast products for both
   tolerances, and the [N, D] ratios of both kinds). The two
   `torch.ones_like` products of `"failure_fraction"` are redundant
   (`torch.maximum` broadcasts, and `x * 1.0` is exact) and are removed with
   item B.
10. **Cumulative sums on CUDA (task doubt 2)**: `torch.cumsum` of a float64
    `[N, 1]` tensor on an A100 is reproducible at N = 30,000 and not at
    N = 40,000 (22 of 30 repeats differ) or above (all differ); `[N, 2]` and
    `[N, 3]` are reproducible to N = 10^6; the CPU is always reproducible;
    `torch.use_deterministic_algorithms(True)` makes it reproducible (0 of 50
    at N = 144,186). The change keeps one `cumsum` per turn for the batch and
    the re-check (the base computed the same sum twice, which could even
    disagree with itself); how the sums are computed is not changed.
11. **Tests that call the changed functions at `c14d6ff`**:
    `tests/test_error_computation.py` (17 calls, 4 read the pair ratio),
    `tests/test_error_masks.py` (17, 6 read it), `tests/test_error_indicator.py`
    (5, none), `tests/test_prune_and_optimal.py` (2, 1 reads it),
    `tests/test_recheck_accepted_panels.py` (`_failed_record_positions` spied
    on; `_make_record` builds a record without the stored values),
    `tests/test_record_and_sort.py` (`_record_results`), and the host-wait
    pins of `tests/test_base_solver.py::TestHostWaits` (unchanged counts).
12. **Vague steps made concrete**: section 5 names, for each step, the
    functions it changes and the tests that prove it.

## 1. Summary

- **What is slow.** Every turn of the integration loop (one pass of the
  `while` loop of `integrate`: evaluate the waiting subintervals, judge them,
  split or record) does error-ratio work whose result nobody reads in the
  default settings: the running sums of cumulative mode in absolute mode
  (item A), the pair error ratios that only the reuse mesh needs (item B), and
  three device reads in the re-check of accepted subintervals (item C). The
  re-check also reduces the whole record's `[N, D]` error and contribution
  arrays again every turn, although a subinterval's reduced error never
  changes (item D), and the returned `error_ratios` carry an autograd graph
  that keeps `[N, D]` arrays alive on the solver's device (item E, new).
- **Gain (measured on an A100 shared with other agents; counts are exact,
  times indicative).** On `chirp` (77 subintervals, 14 turns), the prototype of
  A + B cuts host waits from 619 to 580 and GPU launches from 3,731 to 3,245
  (13 %), and the median time from 58.8 to 51.5 ms; C removes 13 more waits
  and 26 launches (so A + B + C: about 567 and 3,220, estimated from the
  measured parts). Item D cuts one re-check of a record of 2,000 subintervals
  with D = 100,000 outputs from 18.2 ms and a 5.96 GiB temporary peak to
  0.2 ms and no temporary; for D = 1 it costs about 7 launches and 1 wait per
  turn on the present record code.
- **Numbers.** A, B and C change no returned number (shown bit for bit on a
  140-case grid, CPU and CUDA). D can change a re-check decision at rounding
  level for `"2"`, `"rms"` or a callable norm with D > 1 (section 3.4). E
  changes only `result.error_ratios.requires_grad`. D and E are decisions for
  the user (section 4).

## 2. Evidence

### 2.1 The code as it is now

**Item A: cumulative-mode work in absolute mode.** In absolute mode
(`use_absolute_error_ratio=True`, the default) the tolerance is built from the
reference integral only: `_error_ratios_norm` 3888-3891 and
`_error_ratios_failure_fraction` 3973-3976. The running sums are read only in
the cumulative branches, 3893-3905 and 3978-3986. The loop computes them
anyway:

- 1244: `torch.cumsum` of the first batch.
- 1251-1261, every later turn: two `_mesh_order` calls (a device-to-host copy of
  the left mesh points, a Python dictionary lookup per subinterval, and a
  host-to-device copy of the result), `_merge_positions`, `_insert_sorted_results`
  of `record["mesh_quadratures"]` (a clone, a zero array and two scatters of an
  `[N_rec + N_b, D]` array), a `cumsum` and an index. `N_rec` is the number of
  recorded subintervals, `N_b` the number in the batch.
- 1327: a second `cumsum` of the same merged array, for the re-check (it reads
  `idxs_keep` from 1251).
- 1498: a `cumsum` of the record at Step 7. Step 7 runs only while
  `reference_pinned`, which is True only in absolute mode (1096-1098), so this
  sum is never read.
- 1581: a `cumsum` of the record for the final check.

The loop's second `_mesh_order` (1253) recomputes, from the batch's left nodes,
the mesh positions the loop already holds in `step_idxs` (2313-2314 and 2391:
`nodes = _compute_nodes(mesh[step_idxs], ...)`, 2621, so `nodes[:, 0, :]` is
`mesh[step_idxs]` bit for bit, as `_mesh_order`'s docstring, 4138-4144,
explains).

**Item B: pair error ratios.** `_compute_error_ratios` (3732) always builds
`pair_errors` (3800-3804), and both helpers always compute the pair ratio:
3910 and 3913-3914 (`_reduce_norm` of an `[N-1, D]` array) and 3991-3994 and
4004-4005 (two `[N-1, D]` temporaries and an `amax`). Readers:

- `_get_optimal_mesh`'s first call (3277): the only reader.
- The loop (1278-1296): only `len(error_ratios_2steps)` in an assert and a
  debug log.
- `_record_error_check` (1711) and the second call in `_get_optimal_mesh`
  (3297): discarded.

**Item C: device reads of the re-check.** Every turn with a non-empty record,
`_failed_record_positions` (1722) runs `_mesh_order` over the whole record
(1767; one host-to-device copy and a dictionary lookup per recorded
subinterval), indexes it with a boolean mask (1777; one wait), and the loop
then reads `bool(torch.all(record_keep_mask))` (1394; one wait). In almost
every turn nothing fails, and the positions of the non-failing subintervals are
never used.

**Item D: the re-check reduces unchanged data again.** Step 4b (1310-1332)
calls `_failed_record_positions` → `_record_error_check` (1677) →
`_compute_error_ratios` on the whole record, and the final check (1579) does
it once more. For the norm family (`"2"`, `"max"`, `"rms"` or a callable),
`_error_ratios_norm` computes (3877-3914):

$$
r_k = \frac{R(|e_k|)}{\max\big(\tau,\; R(\phi_k)\big)}, \qquad
\phi_k = 50\,\varepsilon\,|s_k| \tag{1}
$$

where $e_k$ is subinterval $k$'s error estimate (a vector of D components),
$s_k$ its contribution to the integral, $\phi_k$ its rounding floor
(`_round_floor`, 3693), $\varepsilon$ the machine epsilon of the dtype, $R$
the norm (`_reduce_norm`, 3646) and $\tau$ the tolerance:
$\tau = \max(\text{atol}, \text{rtol}\,R(I))$ in absolute mode, with $I$ the
reference integral, and $\tau_k = \max(\text{atol}, \text{rtol}\,R(|c_k|))$ in
cumulative mode, with $c_k$ the running integral through subinterval $k$.

$e_k$ and $s_k$ are stored in the record and never change, so $R(|e_k|)$ and
$R(\phi_k)$ never change; only $\tau$ does. Each re-check nevertheless reduces
two `[N_rec, D]` arrays (and, with item B, a third), allocating `[N_rec, D]`
temporaries for `abs`, the floor and the square. The batch's own ratio
computation already produces both reduced vectors (`err` 3877 and `floor`
3882-3886), so they could be stored with the record and never recomputed.

This holds for the whole norm family, including a callable norm, in **both**
modes: in cumulative mode $\tau_k$ must still be rebuilt from the running sums
every turn (one `[N, D]` reduction), but the numerator and the floor need not
(two of three reductions saved). It does not hold for `"failure_fraction"`,
which compares every component with its own tolerance
$\text{atol} + \text{rtol}\,|I_d|$ (3975, 3984, 3996), so nothing smaller than
the `[N, D]` arrays determines its decision.

The record's existing `error_ratios` field cannot hold these values: it is the
ratio at acceptance, against the tolerance of that moment (4100, recorded at
1417), and it is returned to the user as `IntegrationResult.error_ratios`
(1667; documented in `padaquad/results.py:138-141`). It is also read by
`_remove_failed_record_panels` (4516) for the result passed to `loss_fxn`.

**Item E (new): the returned error ratios carry an autograd graph.** With
`take_gradient=False` and an integrand that depends on parameters,
`method_output.mesh_quadratures` requires grad, so the floor (3882-3886) and,
in cumulative mode, the running sum (1261, not detached) put the batch's
error ratios on the autograd graph. `record["error_ratios"]` keeps that graph,
and with it the `[N_b, D]` tensors its nodes saved for a backward pass that
never happens (nobody differentiates an error ratio). The saved tensors are
on the solver's device even when the record is on `result_device`.

### 2.2 Further work of this kind (checked, with readers)

| Where | What | Reader | Action |
|---|---|---|---|
| 3277 | `_get_optimal_mesh`'s first call computes the per-subinterval ratio (an `[N, D]` reduction) and discards it | nothing | Leave to `optimal-mesh-cost`; see section 9 |
| 1414-1416 | `torch.abs(mesh_quadrature_errors)` of the batch, an `[N_b, D]` pass every turn | only a user's `loss_fxn` (the default `_integral_loss` reads only the integral) | Keep: it is part of what `loss_fxn` receives |
| 1505, 1512-1513 | `fails` and `bool(fails.any())` at Step 7 | an INFO log | Keep: once per call, only with a pinned reference |
| 1295, 126, 137 | `logger.debug(..., tensor)` | debug log | Nothing: `%s` formatting is lazy, no cost unless DEBUG is on |
| 1239, 1262-1263, 1266, 1289-1290 | `time.time()` around the merge and the ratios | `speed_logger` (off by default) | Nothing: no device read |
| 1711, 1764 | `_record_error_check`'s ratio, discarded by `_failed_record_positions` | the final check's warning | Nothing: computed anyway for the mask |
| 1286 | `_accept_reject_masks`'s third output | nothing in the loop | Nothing: computed anyway for the masks |

### 2.3 Measurements

All on node `nid008404`, GPU 1 (A100 40 GB), shared with other agents; gk21,
`atol=1e-8`, `rtol=0`, float64, `take_gradient=False`, `max_batch=1_000_000`,
interval [0, 1], the baseline script's method (waits from
`torch.cuda.set_sync_debug_mode("warn")`, launches from `torch.profiler` CUDA
events, peak from `torch.cuda.max_memory_allocated`). Counts are exact and
repeat run to run; times are medians of 5 and indicative only (two runs of the
same code differed by up to 2 ms on `many_outputs`). Script:
`scripts/counts.py`. Prototypes: `item1` (A), `item12` (A + B), `proto`
(A + B + D), `proto4` (A + B + C + D).

| Integrand, norm | Code | Host waits | GPU launches | Wall (ms) | Peak (MiB) |
|---|---|---|---|---|---|
| chirp, `"max"` | base | 619 | 3,731 | 58.8, 58.7 | 0 |
| | A | 580 | 3,441 | 52.5 | |
| | A + B | 580 | 3,245 | 51.5 | |
| | A + B + D | 595 | 3,336 | 52.7, 52.6 | |
| | A + B + C + D | 582 | 3,310 | 52.9 | |
| chirp, `"2"` | base | 618 | 3,843 | 72.0, 70.0 | 0 |
| | A | 579 | 3,553 | 62.8 | |
| | A + B | 579 | 3,329 | 60.6 | |
| | A + B + D | 594 | 3,392 | 61.9, 62.8 | |
| | A + B + C + D | 581 | 3,366 | 61.9 | |
| many_outputs (D = 1,000), `"max"` | base | 108 | 537 | 13.8, 13.9 | 4.6 |
| | A + B | 105 | 487 | 14.9 | 4.6 |
| | A + B + D | 108 | 499 | 15.3, 13.5 | 4.6 |
| chirp × ones(100,000) (D = 1e5, 97 subintervals), `"2"` | base | 618 | 3,950 | 1,712, 1,661 | 1,362.1 |
| | A | 579 | 3,688 | 1,657 | 1,285.0 |
| | A + B | 579 | 3,463 | 1,710 | 1,285.0 |
| | A + B + D | 594 | 3,526 | 1,791, 1,671 | 1,285.0 |

Reading the table:

- **A** removes 3 waits per turn after the first (39 on chirp's 13) and about
  22 launches per turn, and on the D = 1e5 case 77 MiB of peak (the merged
  `[N_rec + N_b, D]` array and its sum).
- **B** removes about 15 launches per turn (two calls of
  `_compute_error_ratios` per turn, about 7 launches each, more for `"2"`).
- **D, as prototyped**, adds 15 waits and 91 launches on chirp. The prototype
  filters the stored vector with the boolean mask (`x[keep_mask]`, one
  `nonzero` and one wait per turn) and merges it into the record with
  `_insert_sorted_results` (a clone, a zero array and two scatters per turn).
  Both costs belong to the code that `turn-device-reads` (one index array
  instead of a mask per field) and `record-order` (no per-turn re-insertion)
  replace; after them, D should add about 2 launches per turn (my estimate,
  not measured).
- **C** removes 1 wait and 2 launches per turn (13 waits and 26 launches on
  chirp).
- On the large-D integrand the wall time is dominated by other work (moving
  the values to the CPU record each turn); no change is visible.

The re-check alone, one call of `_record_error_check` on a hand-built record,
absolute mode, `"2"` (`scripts/recheck_micro.py`, median of 5):

| Record (N × D) | base | A + B | A + B + D |
|---|---|---|---|
| 13 × 1,000 | 0.33 ms | 0.25 ms | 0.20 ms |
| 2,000 × 100,000 (2.98 GiB of record arrays) | 18.19 ms, peak +5.96 GiB | 10.97 ms, +4.47 GiB | 0.21 ms, +0.00 GiB |

The re-check runs every turn, and its temporary peak is twice the record's two
`[N, D]` arrays. For the integrands with 1e5 outputs that the memory options
were built for, this temporary decides whether a turn fits on the GPU.

Item E, measured on CPU (`scripts/grad_check2.py`; f = θ·sin(k t), D = 1,000,
11 subintervals, one turn): the autograd nodes reachable from
`result.error_ratios` but not from `result.integral` save 2 extra `[N, D]`
tensors with `"2"` and 3 with `"max"` (172 and 258 KiB, against 86 KiB for
the record's `[N, D]` array), in both modes. Over several turns these are the
batch's arrays before rejected subintervals are dropped.

**Is the row reduction independent of the other rows?** Item D replaces "row
k reduced inside the record array" by "row k reduced inside its batch array".
`scripts/rowwise.py` compared `sqrt(sum(x**2, -1))` and
`sqrt(mean(x**2, -1))` of slices of a `[2000, D]` float64 array with the same
rows of the whole: no difference for D = 2 and 50 (CPU and CUDA); 2 differing
rows for CUDA at D = 1,000 and for CPU at D = 100,000. So the present code's
re-check of a row can differ in the last bit from the value the batch used to
accept it, and item D removes that difference (section 3.4). `"max"` (an
`amax`) and D = 1 are exact either way.

### 2.4 Measurements at `c14d6ff` (2026-11-26)

Node `nid008229`, A100-SXM4-80GB, torch 2.8.0; gk21, `atol=1e-8`, `rtol=0`,
float64, `take_gradient=False`, `max_batch=1e6` on [0, 1]; counts exact
(`impl/count_fns.py`: host waits from `torch.cuda.set_sync_debug_mode`,
launches as CUDA runtime launch events of `torch.profiler`, each attributed to
the innermost of the functions below that was running); wall time median of 7
on a node shared with my other runs (indicative).

| Case | Turns | Waits | Launches | Wall (ms) |
|---|---|---|---|---|
| `chirp`, `"max"` | 14 | 103 | 2,823 | 56.4 |
| `chirp`, `"2"` | 14 | 103 | 2,935 | 60.0 |
| `chirp`, `"max"`, cumulative, `rtol=1e-8` | 14 | 103 | 2,851 | 67.6 |
| `chirp`, `"failure_fraction"` | 14 | 103 | 3,019 | 61.6 |
| `many_outputs` (D = 1,000), `"max"` | 2 | 32 | 416 | 15.3 |
| `chirp` × ones(10,000), `"2"` | 14 | 103 | 2,949 | 180.0 |

Per call on `chirp`, `"max"` (launches; no wait in any of them):
`_compute_error_ratios` with `_error_ratios_norm` 3 + 17 (the pair ratio
about 9 of them), `_accept_reject_masks` 8; the re-check
(`_failed_record_positions`) 6 own + `_record_error_check` 28 + `_splittable`
11 and one wait, plus the loop's `torch.all` (one wait); the merge of item A
8 (`_merge_positions`) + 2 (`_insert_sorted_results`) + the loop's
`cumsum`s and indexes. With `"failure_fraction"` the helper takes 23.

One re-check (`_record_error_check`) of a hand-built record of 2,000
subintervals and 100,000 outputs, absolute mode, `"2"`: 18.8 ms.

Largest temporary memory inside the error-ratio calls during an integration
(`impl/mem_ratios.py`, 512 outputs, all on the GPU, in arrays of [N, D] of the
call): norm family 4.0 to 5.1 in `_compute_error_ratios` and
`_record_error_check`, 5.0 to 7.8 in `_get_optimal_mesh`; `"failure_fraction"`
7.3 to 9.4, and 9.4 to 12.5 in `_get_optimal_mesh`.

## 3. Design (corrected against `c14d6ff`)

### 3.1 Item A: no running sums in absolute mode

In the loop (`integrate`, Step 3) the running integral is the batch's
integral plus the record's; the running sums are built only when
`not self.use_absolute_error_ratio`: the record (empty on the first turn:
`record.get("mesh_index", step_idxs[:0])`, `record.get("mesh_quadratures",
quads[:0])`) and the batch are merged by `_merge_positions` and
`_insert_sorted_results`, summed once, and indexed twice: the batch's rows
(`cum_mesh_quadratures`) and the recorded rows (`recorded_cum`, detached, for
the re-check). Step 4b passes `recorded_cum` (None in absolute mode). The
final check passes `None` (the record is in mesh order; the helper's fallback
is the same `cumsum`). No number changes; on CUDA the single sum replaces two
sums of the same array, which for N above about 40,000 rows could differ from
each other in the last bit (2.4, review 10).

### 3.2 Item B: the pair ratio only for the merge

`_compute_error_ratios(..., pair_errors=None)` replaces `merge_prediction`;
its pair arithmetic moves to `_get_optimal_mesh`, which builds
`pair_errors = (0 if merge_prediction is None else merge_prediction) + e[:-1]
+ e[1:]` from its absolute errors (the same sum in the same order) and asserts
that the pair ratio came back. Each helper returns its pair ratio only when
`pair_errors` is given (one early return each), else None in that slot. The
loop drops its assert on the pair ratio's length and its debug log of it. The
two `torch.ones_like` products of `_error_ratios_failure_fraction` go
(`torch.maximum` broadcasts the [D] or [N, D] tolerance with the [N, D]
floor; values identical).

### 3.3 Item C: one read in the re-check

The loop's Step 5b tests `len(extra_split_idxs) > 0` (known on the host)
instead of `bool(torch.all(record_keep_mask))`: `record_keep_mask` is
`~splittable` and `extra_split_idxs` is `mesh_index[splittable]`, so "some
recorded subinterval is split" is the same statement. One wait per turn less.

### 3.4 Item E: error control on detached values

The loop passes detached errors and contributions to `_compute_error_ratios`,
merges detached contributions for the running sums, and adds the detached
recorded integral to the running integral. No value changes;
`result.error_ratios.requires_grad` (and that of the `error_ratios` that
`loss_fxn` receives) becomes False when the integrand depends on parameters
or the bounds require grad, and the result no longer keeps the autograd nodes
of the error control alive. Its own commit, in the CHANGELOG under Changed.

### 3.5 Item D: the reduced error and rounding floor kept with the record

For the norm family each recorded subinterval stores its two numbers of
equation (1) that never change, `[R(|e_k|), R(phi_k)]` (`[N, 2]`), and every
later judgement of it (Step 4b, Step 7, the final `converged`) reads them:

- **`_reduced_errors(mesh_quadrature_errors, mesh_quadratures)`** (new, two
  callers: the loop and `_compute_error_ratios`) stacks `R(|e|)` and
  `R(_round_floor(s))` (a zero floor without contributions, which only unit
  tests pass; `max(tol, 0)` is `tol` bit for bit). It replaces the two
  reductions at the top of `_error_ratios_norm`, so the formula stays in one
  place.
- **`_compute_error_ratios(..., reduced_errors=None)`**: the norm family uses
  the given values or computes them; `"failure_fraction"` ignores them (its
  tolerance is per component, so nothing smaller than the [N, D] arrays
  decides). It also takes the cumulative fallback (`cumsum` of the
  contributions) from both helpers, so `_error_ratios_norm` receives the
  `[N, 2]` values and the running sums only.
- **The loop** computes them for the batch (norm family), passes them to
  `_compute_error_ratios`, filters them with `keep_idx` with `error_ratios`,
  and passes them to `_record_results(reduced_errors=...)`, which appends them
  under the key `"reduced_errors"` (one of `_RECORD_LOOP_KEYS`).
- **`_record_error_check`** passes `record.get("reduced_errors")`: the
  re-check, Step 7 and the final check use the stored values, the very
  numbers the batch accepted the subinterval with, so they cannot disagree
  with each other or with the acceptance through rounding. A record without
  them (`"failure_fraction"`, hand-built records in tests) is judged from its
  arrays as before.

Which results can change: for `"2"`, `"rms"` or a callable norm with D > 1, a
recorded subinterval's reduction inside the record array can differ in its
last bit from the same reduction inside its batch array (review 6). Before,
the re-check and the final check used the former; now they use the latter.
A decision changes only when a ratio lies within rounding of 1; then the mesh
(a split or not) and everything after it change. `"max"` (an `amax`) and
D = 1 are exact either way. Shown on the grid (section 6).

The stored values assume that a callable norm reduces each row on its own,
which the batch and the re-check already assumed (they apply the norm to
different sets of rows); `_reduce_norm`'s docstring and the user guide say
so.

### 3.6 Further work of this kind (task doubt 6), checked at `c14d6ff`

| Where | What | Reader | Action |
|---|---|---|---|
| `_get_optimal_mesh`, merge stage | the per-subinterval ratios of `_compute_error_ratios` (with D, the reduction of the errors) | nothing | Kept: skipping them needs a second return shape or a branch; once per call, and not at all with `build_state_mesh=False` |
| loop, Step 6 | `torch.abs(mesh_quadrature_errors)` of the batch for `loss_fxn` | a user's `loss_fxn` | Kept (part of what `loss_fxn` receives) |
| loop | `logger.debug("error_ratios: %s", ...)` | debug log | Nothing: formatting is lazy |
| loop, Steps 3 and 4 | `time.time()` around the merge and the ratios | `speed_logger` (off by default) | Nothing: no device read |
| `_accept_reject_masks` | the reject mask in the loop, `~keep_mask` | nothing in the loop | Nothing: one launch, the function is shared with the re-check |
| `_failed_record_positions` | `_splittable` over every recorded subinterval | only failing ones matter | Not done (review 4) |

## 4. Decisions for the user (as taken)

1. **Item D**: implemented (option (a)), in its own commit.
2. **Item E**: detach, in its own commit, declared in the CHANGELOG.
3. **Item B's interface**: `pair_errors` in place of `merge_prediction`
   (review 3), not a `pairs` flag nor a separate function.

## 5. Implementation steps (one commit each)

1. **This review** (this file).
2. **Item A** (`integrate` Steps 3, 4b and the final check). Proof:
   `tests/test_snapshots.py`, `tests/test_recheck_accepted_panels.py`
   (`test_recheck_cumulative_mode`), `tests/test_error_indicator.py`, the grid.
3. **Item B** (`_compute_error_ratios`, `_error_ratios_norm`,
   `_error_ratios_failure_fraction`, `_get_optimal_mesh`, the loop's Step 4).
   Update the test call sites that read the pair ratio; add a test that
   without `pair_errors` the slot is None and the per-subinterval ratios equal
   those with it. Proof: `tests/test_error_computation.py`,
   `tests/test_error_masks.py`, `tests/test_prune_and_optimal.py`, the grid.
4. **Item C** (`integrate` Step 5b). Proof: the host-wait pins
   (`tests/test_base_solver.py::TestHostWaits`: the re-check now reads once
   per turn, pinned by source line), `tests/test_recheck_accepted_panels.py`,
   the grid with `max_adaptive_splits`.
5. **Item E** (`integrate` Steps 3 and 4). New test: the returned and the
   per-batch `error_ratios` carry no graph, the gradient of the integral is
   unchanged. CHANGELOG.
6. **Item D** (`_reduced_errors`, `_compute_error_ratios`,
   `_error_ratios_norm`, `_record_error_check`, `_record_results`,
   `_RECORD_LOOP_KEYS`, the loop). New tests: the record check with the stored
   values equals the batch's own ratios bit for bit for every `error_norm`,
   both modes, float32 and float64, a NaN row; with them removed it gives the
   same decisions; `_compute_error_ratios` with given values equals the call
   without them. Proof: the grid (with D = 1,000 and 10,000, `"2"`, `"rms"`
   and a callable), the re-check timing on a large record.
7. **Documentation** (section 8), then the full suite, ruff, mypy, numpydoc,
   docs build.

## 6. Tests, and showing that results are unchanged

- The grid `impl/bitgrid_er.py` (every `error_norm` including a callable
  3-norm and `"failure_fraction"` with tolerance 0 and 0.3, absolute and
  cumulative, D = 1, 3, 50, 1,000 and 10,000, `enforce_convergence` on and off,
  `max_adaptive_splits`, the record on either device, a fixed reference both
  above and below the integral, a NaN region, `y0`, float32,
  `take_gradient=True`, a learnable integrand with its gradient, five methods
  including Gauss-Kronrod and a variable method, and a warm call from every
  state; it also records every re-check's positions and failing mask and the
  points of `f`), compared with `impl/bitcompare_er.py` against two runs of the
  base on CUDA (the base's own differences are listed apart) and one on the
  CPU; and `record-order/impl/bitgrid.py`.
- `tests/test_snapshots.py` (CPU, float64).
- The full suite on CPU and CUDA, compared with the base's set of failures.

## 7. How to measure the gain

`impl/count_fns.py` (waits and launches per function, wall time) and
`impl/mem_ratios.py` (temporaries; one re-check of a 2,000 × 100,000 record)
on one GPU, base and branch alternating.

## 8. Documentation to update

- **CLAUDE.md:** "Core algorithm" (running sums only in cumulative mode),
  "Error computation modes" (the pair ratio only for the merge, the stored
  reduced error and floor of the re-check and the final check, the single
  read of the re-check), the test map rows.
- **CHANGELOG.md** (`[2.0.0]`, unreleased): Changed: `error_ratios` no longer
  requires grad (item E); a performance entry for A to D.
- **Docstrings:** `_compute_error_ratios`, the two helpers,
  `_record_error_check`, `_failed_record_positions`, `_reduce_norm` (a
  callable reduces each row on its own), `_record_results`,
  `_RECORD_LOOP_KEYS`, `_get_optimal_mesh`.
- **User guide:** `error_control.md`: a callable norm reduces each row on its
  own; one sentence on what cumulative mode reproduces on a GPU.

## 9. Risks and interactions

- **`split_waste`** (parallel branch) owns `_adaptively_increase_mesh`,
  `_splittable`, `_midpoint` and reads the batch's `error_ratios` at the mesh
  update: this plan changes neither the arguments of the mesh update nor the
  batch's `error_ratios` (values, shape, order); item D filters its own values
  in the loop beside `error_ratios[keep_idx]`.
- **`quadrature-arithmetic`** owns how `_reduce_norm` and `_round_floor`
  compute; item D calls them from one place (`_reduced_errors`).
- **The GPU's `cumsum`** (review 10): unchanged; the user guide says what is
  reproducible.

## 10. Confidence

- **Sure (measured or read):** the readers of each quantity (read at
  `c14d6ff`); the counts of 2.4; the row-reduction and `cumsum` facts of
  review 6 and 10 on this node.
- **Less sure:** wall times (shared node); whether a callable norm in use
  mixes rows (the documentation now forbids it).
- **My own reasoning:** that the re-check agreeing with the acceptance is the
  better behaviour; the choice of the `pair_errors` interface.
