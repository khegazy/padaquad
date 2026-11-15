# Code review: `merge_rule="predicted"` on `improve_pruning`

Reviewed 2026-11-15: commits `c13fcf5`, `db176f2`, `5aadefb` and `387d1bc` on top of
`main` (`2c0c955`). Review commits (not pushed): `b1dd7a6`, `6f1a0dc` (fixes),
`f08b455` (proposed tests) and the commit that adds this file. Scripts and logs are in
`/pscratch/sd/k/khegazy/u028/review2/` (called `review2/` below). Every number below
was measured with those scripts unless it is labelled otherwise.

## Verdict

The merge rule is implemented as the plan describes, and its numerical core is
correct: I found no case where the prediction is wrong for a record that the solver
produces. The cache, the device handling, float32, the variable solver, NaN
integrands, the one-pass `_rec_remove` (CPU and CUDA) and the snapshot changes all
check out.

The claim that the prediction's memory is bounded by working in chunks was false.
There were two causes, both fixed with a failing test first:

- the node-position check ran on the whole record before the chunked loop;
- the chunk size did not depend on the number of outputs D.

One finding is not fixed and needs the author's decision. The cost grows with the
number of distinct width ratios, and a mesh that starts from many random
subintervals has one ratio per pair. Calls are then 2 to 8 times slower than with
`"power_law"` for cheap integrands. The CHANGELOG's cost bullet does not mention this.

The branch is fit to merge after the two fixes, provided the author either accepts
and documents that cost or reduces it (options in Finding 3).

## Findings, by severity

### 1. The node check held three record-sized temporaries, and its tolerance was global (measured; fixed in `b1dd7a6`)

`_predict_merged_errors` checked, before its chunked loop, that every node sits at the
method's node fractions (`base.py:2539` at `387d1bc`):

```text
expected = self._compute_nodes(nodes[:, 0, :], nodes[:, -1, :])
tolerance = 16 * torch.finfo(nodes.dtype).eps * float(nodes.abs().max())
if float((nodes - expected).abs().max()) > tolerance:
```

- **Memory.** `expected`, `nodes - expected` and its absolute value are each the
  size of the record's nodes, which at D = 1 is the size of `y`. The predictor's
  peak was therefore 3.0 times `y`, all of it from the check (`review2/probe3.py`):

  | Case (CPU, 8 threads) | Size of `y` | Peak at `387d1bc` | Peak after `b1dd7a6` |
  |---|---|---|---|
  | gk21, 10^6 subintervals | 175 MB | +528 MB (check alone: +529 MB, 0.19 s of 0.45 s) | +193 MB, 0.35 s |
  | cc65, 4·10^5 | 198 MB | +594 MB | +355 MB, 0.47 s |
  | gk21, 10^6, CUDA, `torch.cuda.max_memory_allocated` (`probe13.py`) | 175 MB | +528 MB | +161 MB |
  | cc65, 4·10^5, CUDA | 198 MB | +595 MB | +370 MB |

- **Tolerance.** One scale for the whole record (16 eps times the largest |node|)
  let a node 1e-6 of its subinterval's width off its fraction pass in a
  subinterval near 0 of a mesh that reaches 1e6.
- **Fix.** The check moved into the chunk loop, and each subinterval is now compared
  with 16 eps times its own largest |node|. The predictions are bit-identical to
  `387d1bc` on random records for gk21, cc65, dopri5 and bosh3 (`probe13.py`).
- **Tests.** `test_predicted_node_check_works_in_chunks` spies on `_compute_nodes`.
  `test_predicted_node_check_is_relative_to_each_subinterval` is the far-from-0
  case. Both failed before the fix.

### 2. The chunk size ignored D, so the temporaries grew with the number of outputs (measured; fixed in `6f1a0dc`)

Each chunk of 2^16 pairs copies both neighbours' stored values to float64, which is
2 · pairs · U · D values. For D = 1 that is bounded, but it grows with D
(`review2/probe14.py`, gk21, 6·10^4 subintervals):

| D, dtype | Size of `y` | Peak before | Peak after `6f1a0dc` |
|---|---|---|---|
| 256, float32 | 1,348 MB | +6,796 MB (5.0 ×) | +96 MB |
| 256, float64 | 2,695 MB | +6,159 MB (2.3 ×) | +142 MB |
| 64, float64 (2·10^5 subintervals) | 2,246 MB | +2,417 MB (1.1 ×) | +146 MB |
| 1, float64 (10^6) | 175 MB | +180 MB | +196 MB (unchanged within noise) |

A chunk now holds `_MERGE_CHUNK // D` pairs, and at least one. The regression test is
`test_predicted_chunks_shrink_with_the_number_of_outputs`, which failed before the fix.

### 3. The cost grows with the number of distinct width ratios (measured; not fixed)

The weight vectors cost O(C · U) per distinct width ratio. That is about 20 µs for
cc65 and 2 µs for gk21 per ratio on 8 threads of a compute node (`probe2.py`, random
meshes of 10^5 subintervals), and 90 µs and 5 µs on a busy login node
(`review2/probe6.py`, 1,024 ratios per block). A
mesh started from many random subintervals has one ratio per pair. The ratios stay
random after merging, so warm calls pay the cost too. The cache cannot help, because
it is bypassed when a chunk has more than 256 ratios.

The table shows realistic calls on a compute node with 8 threads
(`review2/probe5.py`; f = e^{-t} sin 40t + a Lorentzian; the cold call starts from
`N_init_steps` random subintervals, followed by 5 warm calls). The last column is the
number of distinct ratios in the last warm call.

| Method, `N_init_steps`, atol = rtol | `power_law` cold / warm | `predicted` cold / warm | Distinct ratios |
|---|---|---|---|
| gk21, 13 (default), 1e-10 | 8.8 / 2.1 ms | 5.2 / 2.4 ms | 6 |
| cc65, 13, 1e-10 | 5.3 / 2.1 ms | 4.4 / 2.7 ms | 2 |
| cc65, 1,000, 1e-10 | 10.1 / 4.1 ms | 27.7 / 8.9 ms | 30 |
| gk21, 10^4, 1e-8 | 25.1 / 4.6 ms | 89.6 / 11.5 ms | 315 |
| cc65, 10^4, 1e-8 | 31.7 / 5.8 ms | 266 / 31.1 ms | 315 |

Building `mesh_optimal` alone on a random mesh of 10^5 subintervals takes 1.9 s for
cc65 with `"predicted"` and 0.002 s with `"power_law"`; gk21 at 10^6 takes 2.4 s
against 0.008 s (`review2/probe2.py`, compute node).

- **What the docs say.** CHANGELOG, "Cost": "0.2–0.5 ms more per call on meshes of up
  to 64 subintervals", and 0.41 s against 0.31 s at 10^6. Both were measured on
  equal-width meshes, which have few ratios. I confirm both on equal widths (next
  bullet). The worst case is not mentioned.
- **Check of the stated numbers** (compute node, `review2/timings.log`).
  - Small meshes: `"predicted"` costs 0.26–0.47 ms more than `"power_law"` per
    `mesh_optimal` (gk21 0.27–0.34, cc65 0.27–0.47, dopri5 0.26–0.28;
    `pred/overhead.py` on the tree after fix 1).
  - 10^6 equal subintervals, building `mesh_optimal` at `387d1bc`: gk21 0.40 s
    against 0.30 s (the CHANGELOG says 0.41 against 0.31). cc65: 0.77 s against
    0.38 s (the CHANGELOG says 0.60 against 0.39). These are single-node medians of
    3, and repeated runs vary by about 25 %.
  - "About 7 % of the `integrate` call" is ambiguous. The extra cost is 5 % of the
    call for gk21 (0.10 s of 1.99 s) and 15 % for cc65 (0.39 s of 2.53 s).
    Building `mesh_optimal` takes 20–30 % of the call.
- **Options** (my suggestions; the author decides):
  1. Document the cost. A suggested sentence for the Cost bullet: "The weight
     vectors are built once per distinct ratio of neighbouring widths (about 2 µs
     each for gk21 and 20 µs for cc65 on 8 CPU threads). A mesh started from many
     random subintervals has one ratio per pair, so `N_init_steps=10**4` costs about
     0.27 s instead of 0.03 s per cold call with cc65."
  2. Halve the cost and the memory. Apply the weights with two batched matrix
     products instead of materializing the `[R, 2, C, U]` products. A prototype
     (`review2/probe12.py`) takes 2.5 ms instead of 5.2 ms (gk21) and 44.6 ms
     instead of 86.0 ms (cc65) per 1,024 ratios. Its weights differ from the
     current ones by at most 5.6e-17, so it is not bit-identical, and rounding-level
     merge decisions could change. I did not commit it for that reason.
  3. Cap the work. Above some number of distinct ratios per chunk, fall back to
     `"power_law"` for that call with a logged message. This changes results for
     such meshes and needs a decision.

### 4. Pre-existing, outside this work: `same_f=True` after a call with only `mesh=` integrates another interval (measured; not fixed)

A call `integrate(f, mesh=m)` with `m` on [2, 3], followed by
`integrate(f, same_f=True)`, integrates the solver's default interval [0, 1]. The
stored mesh is filtered to the default `mesh_init`/`mesh_final`, nothing is left, and
no warning is given (`review2/probe8.py`). The output is
`cold 12.6965 warm 1.7183`. It is identical on `main` (`2c0c955`), so it is not
caused by this branch. The not-converged warning advises "call again with
`same_f=True`", which leads straight into it. I did not fix it, because which bounds
a warm call should use is a behaviour decision. The proposed test
`test_predicted_far_from_the_origin` passes the bounds explicitly and says why.

### 5. The T > 1 fallback cannot be reached through `integrate()` (from the code, measured; not fixed)

`_merge_inputs` logs a warning and uses `"power_law"` when T > 1
(`base.py:2388`). But `integrate()` with T > 1 already fails in
`_setup_integral_bounds` (`assert mesh_init < mesh_final` on a tensor of two values,
`base.py:2907` and `:2916`, `RuntimeError: Boolean value of Tensor with more than one
value is ambiguous`; `review2/probe4.py`). This is the known blocker in
`docs/issues/multi-coordinate-t.md`. The fallback is only exercised by the direct
unit test. That is fine for now, but "a warning on every call" cannot be checked end
to end until that blocker is removed.

### 6. A width ratio of 0, infinity or NaN converts to int64 without a defined result (measured on CPU; conjecture elsewhere; not fixed)

`key = round(log2(w_R / w_L) * 2**48).to(int64)` (`base.py:2542`) is undefined in C
for ±inf and NaN. On this CPU it gives the most negative int64, so an infinite ratio
is treated as rho = 0. Reaching it needs a width ratio above about 1e308, for example
the mesh [0, 1e-310, 1]. The prediction then compares the merged integral with the
left neighbour's constant value. It came out as 0.718 for e^t, so that pair was not
merged and the result is conservative (`probe4.py`). On CUDA the conversion may
differ (not checked). Zero widths did not come out of `integrate()` in my tries. A
generated starting mesh is checked to be strictly increasing (`base.py:3048–3052`).
A user mesh with a repeated point is rejected with an `AssertionError`. A float32
step function at `atol=0` stopped refining at a width of 4.8e-7 (`probe1.py`). So I
did not fix this. A
one-line guard would be to treat a non-finite `key` as "do not merge".

### 7. Smaller documentation inaccuracies (measured; not fixed)

- `_RATIO_QUANTUM`'s comment says pairs "whose widths are equal up to rounding share
  one set of weight vectors". Equal-width meshes of [0, 1] have 21 distinct keys at
  10^4 subintervals and 29 at 10^5 in float64. They have 26 and 30 in float32,
  because the rounding of a width relative to the width exceeds 2^-48. This costs
  time only (`review2/probe7.py`).
- `_RATIO_BLOCK`'s comment says a block holds "about six [block, C, U] float64
  temporaries (35 MB each for cc65)", which is 210 MB. The predictor's peak on a
  cc65 record with 10^5 distinct ratios is +650 MB (`probe2.py`). The total is
  measured. That the `[R, 2, C, U]` products and the gathered `[P, 2, 2, U]` weights
  of a chunk (136 MB) make up the difference is my conjecture.
- Plan, Implementation notes: "Memory is bounded for any mesh", and Evidence E: "peak
  at most 0.76 × y". At `387d1bc` the peak was 3.0 × `y` at D = 1 and 5.0 × at
  D = 256 (Findings 1 and 2). After the fixes it is bounded per chunk, at 1.1–1.9 ×
  `y` at D = 1 and below 0.1 × for large D.

### 8. Checked and found correct

- **Cache reset.** `_set_dtype` returns early when the dtype is unchanged and
  otherwise rebuilds the method through `_set_solver_dtype`, which empties the cache.
  `set_dtype_by_input` goes through `_set_dtype`. `_integration_on_device` moves only
  the method, and the cache's device check then clears it. No other code reassigns
  `self.method` (grep). The vectors are always float64 and built from the method's
  current dtype, so a stale dtype cannot be used. A CUDA solver alternating
  `result_device="cuda"` and `"cpu"` works and keeps its cache on the last device
  (`test_predicted_on_gpu_with_either_result_device`, run on a GPU).
- **Eviction.** The branch that empties a full cache is right, but no author test
  reached it (see the author's tests). `test_cache_eviction_gives_the_uncached_prediction`
  reaches it. A mutation that drops `missing = key_list` makes it fail.
- **float32.** The node check accepts float32 records far from 0 on both sides
  (±1e3). Warm chains shrink 128 → 2 subintervals where `"power_law"` stays at 123,
  and the error stays within tolerance (`probe1.py`, `probe4.py`).
- **Variable solver.** Warm chains merge (512 → 256 → 128) with no re-split and the
  same accuracy as `"power_law"` (`probe4.py`).
- **NaN integrand** with `error_on_nonfinite=False`. The integral is NaN, so every
  pair ratio is NaN and nothing merges, under every rule; there is no crash. One NaN
  component of D = 2 gives NaN predictions only for that component.
- **Determinism.** The predictions are bit-identical with 1, 4 and 16 CPU threads
  (gk21, cc65, dopri5; `probe4.py`). In the two fragile tests, the decision closest
  to the cutoff is 3.6 % away (`test_rounding_level_warm_starts`, `probe10.py`) and a
  factor of 2 away (`test_kink_warm_starts_stay_accurate`, `probe11.py`). Last-bit
  differences between machines (about 1e-13 of a pair's scale) cannot flip them.
- **`_rec_remove`.** It matches the CPU result on CUDA for lengths 0 to 10^5. No
  caller relied on the former in-place change: the only caller,
  `_prune_excess_mesh`, uses the return value.
- **Snapshots.** Only `n_optimal_mesh` changed, in 90 of 390 cases (86 smaller, 4
  larger: fehlberg2 on `t_squared` loose, and gk15 on `sine_squared` tight, 19 → 20
  and 13 → 14). `integral` and `integral_error` are identical
  (`review2/snapcmp.py` against `git show 2c0c955:tests/test_snapshots_data.json`).
- **Citations.** In the saved texts in `/pscratch/sd/k/khegazy/u028/review/sources/`,
  Berrut and Trefethen (2004), Eq. (3.2), give the barycentric weights (line 142) and
  Eq. (4.2) the second barycentric formula (line 239). Gander and Gautschi (2000),
  Sec. 3, "Adaptive Simpson quadrature", begins with "The idea of adaptive Simpson
  quadrature is old [12]" (line 126), where [12] is Lyness. The numbers cited match.
- **Verification script.** `examples/verification/merge_prediction.py` runs in 4.7
  s, passes, and reports no `[KNOWN]` case on this machine. Its `exp` check
  ("predicted used no more evaluations") is weak: both rules give identical counts
  for every method, because one merge level per call is the bottleneck from 16 equal
  subintervals.
- **Design.** `merge_rule` is read in `_merge_inputs` only. The distinct-node
  computation is not duplicated in the library; the test helper `_distinct_fractions`
  repeats it (see the author's tests). Of the four new methods,
  `_cached_merged_pair_weights` has one caller and could be folded into
  `_predict_merged_errors`. As a separate function the cache logic is easier to
  test, so I would keep it. `merge_rule` being a constructor option only, while
  `error_norm` can be overridden per call, is defensible: `merge_rule` affects only
  `mesh_optimal`, not the returned integral.

## Proposed tests (`tests/test_merge_rule_review.py`)

All 15 cases pass in 1.8 s on a login node (with a GPU), and the two CUDA tests skip
without one.

| Test | What it adds | Keep? |
|---|---|---|
| `test_cache_eviction_gives_the_uncached_prediction` | The only test that reaches the eviction branch; mutation-checked | Yes |
| `test_predicted_warm_starts_with_vector_output` (4 cases) | D = 2 end to end, both norm families, absolute and cumulative. A mutation that makes `failure_fraction` ignore `pair_errors` fails both of its cases (gk15 re-splits 2 starting subintervals) | Yes |
| `test_variable_solver_predicted_warm_chain` | The variable solver merging end to end in the suite | Yes |
| `test_float32_predicted_merges_where_power_law_does_not` | float32 end to end; the mesh sizes are asserted, not only printed | Yes |
| `test_predicted_far_from_the_origin` (4 cases) | The node check on the solver's own records at ±1e3 in both dtypes | Yes, cheap |
| `test_take_gradient_does_not_change_the_merges` | A graph in the record versus detached values give the same `mesh_optimal` (one batch) | Optional; not sensitive to removing the `detach`, which only saves memory |
| `test_nan_integrand_does_not_break_the_prediction` | No crash, and NaN stays in its component | Optional |
| `test_predicted_on_gpu_with_either_result_device` | Cache and device switching on CUDA | Yes (runs in the GPU suite) |
| `test_rec_remove_on_gpu_matches_cpu` | `cummax` on CUDA | Yes |

The three regression tests of the fixes are in `tests/test_prune_and_optimal.py`
(Findings 1 and 2).

## The author's new tests

- **`TestRecRemove.test_input_not_modified`, `test_matches_former_loop`.** Keep.
  They are strong (2,580 masks against the former loop) but CPU only; my CUDA test
  covers the GPU.
- **`test_pair_errors_replace_the_pair_numerator`.** Keep. It is sensitive: the
  `failure_fraction` mutation fails all four of its `failure_fraction` cases.
- **`test_merge_inputs_per_rule`.** Change. For `"predicted"` it asserts
  `pair_errors >= e[:-1] + e[1:]`, which is true by construction (the prediction is
  non-negative and is added to that sum). It would pass with a prediction of 0. The
  exactness test checks the values, so this assertion adds nothing. Suggestion:
  assert that the prediction is strictly positive for `sin(3t)`, or drop that line.
- **The end-to-end tests parametrized over `MERGE_RULES`.** Keep. They double their
  run time, which is still small.
- **`test_merged_pair_weights_exact_for_polynomials`.** Keep; it is the core
  numerical check. `_distinct_fractions` repeats the library's distinct-node code.
  `u = c[first]`, with `first` from `_merged_pair_weights`, would remove the copy.
- **`test_preasymptotic_pair_merges`.** Keep; it separates the rules.
- **`test_kink_warm_starts_stay_accurate`.** Keep. The author mutation-checked it,
  and its margins are large (Finding 8). The captured mesh pins the scenario, which
  is the point.
- **`test_rounding_level_warm_starts`.** Keep. Its margins to the cutoff are at
  least 3.6 % (Finding 8), so it is not fragile across machines.
- **`test_merge_rule_option`.** Keep. Through `integrate()` it can only check that
  the option is accepted, which is all a one-call solver exposes.
- **`test_predicted_falls_back_to_power_law_for_vector_t`.** Keep (see Finding 5).
- **`test_predicted_needs_nodes_at_the_fractions`.** Keep.
- **`test_predicted_detaches_and_caches`.** Change. Its second half is vacuous:
  400 random ratios exceed the cache size, so they take the uncached branch, the
  cache keeps its 3 entries, and `<= _MERGE_CACHE_SIZE + 1` holds whatever the
  eviction code does. My eviction test replaces that half.
- **`test_predicted_chunking_matches_one_pass`.** Keep. After `6f1a0dc` its chunks
  hold 3 pairs (D = 2), at most the cache size of 3, so it now also passes through
  the cached and eviction branches.

## Full suite on the final tree

Run on a compute node from `git archive` of the final commit
(`/pscratch/sd/k/khegazy/u028/suite.sh`; logs `review2/suite_cpu.log` and
`review2/suite_gpu.log`):

- CPU (CI-matched Python 3.12 venv): 3531 passed, 133 skipped, 0 failed (7 min 13 s).
- GPU (popcornn env, one A100): 3542 passed, 122 skipped, 0 failed (7 min 8 s).
- Baseline on `387d1bc` (from the review prompt): CPU 3515 passed / 131 skipped;
  GPU 3524 passed / 122 skipped. My changes add 3 regression tests and 15 proposed
  test cases (2 of them CUDA-only): +16 passed and +2 skipped on CPU, +18 passed on
  GPU, as expected.
- Sphinx with `-W` (`docs_build.sh`, log `review2/docs.log`): built with no
  warnings (exit 0).
- `mypy padaquad/`: no issues. `pre-commit run` on every changed file: passed.
  `ruff check` passed on `padaquad`, `examples` and the changed test files. The
  popcornn ruff (0.17.4) reports format and import differences in five unrelated
  test files that pre-commit's pinned ruff accepts. These are pre-existing and
  untouched.

## What I could not check

- The plan's Evidence A to D (50-digit accuracy study, warm-start chains, pode). I
  did not rerun them. My fixes do not change any prediction value (bit-identical,
  Finding 1; the chunk size changes only how pairs are grouped, and
  `test_predicted_chunking_matches_one_pass` bounds the effect at 1e-13 of a pair's
  scale), so those results should carry over.
- The claim that the prediction is bit-identical to the prototype in float64.
- Behaviour on CPUs other than the AMD EPYC nodes here, such as GitHub's runners.
  The margins measured in Finding 8 make flips unlikely.
- The int64 conversion of an infinite key on CUDA (Finding 6).

## Confidence

- **Sure.** Findings 1, 2 and 3 are measured and repeatable with the scripts named.
  The fixes keep predictions bit-identical (fix 1) or within the chunking bound
  (fix 2). Finding 4 is reproduced on `main`.
- **Less sure.** The attribution of the remaining 650 MB in Finding 7 is a
  conjecture. Timings on shared nodes vary by about 25 % between runs. Whether the
  cost in Finding 3 matters depends on how expensive the integrand is; for pode's
  network integrands it is likely negligible (conjecture, not measured).

## Connections

- Finding 4 interacts with the not-converged warning's remedy text, and with any
  future change to how a warm start chooses its bounds.
- Finding 5 will need an end-to-end test once `docs/issues/multi-coordinate-t.md`
  is resolved.
