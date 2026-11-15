# Review of "Plan: decide mesh merges from a predicted merged error (rule T)"

Reviewed 2026-11-14 against `docs/issues/predicted-merge-rule.md`, `main` at `2b04ea1`,
and the prototype in `/pscratch/sd/k/khegazy/u028/pq_pred`. Every script, log and
source text used here is under `/pscratch/sd/k/khegazy/u028/review/` (`review/`
below). Each finding is labelled **measured** (my run), **from a source** (a paper or
file I opened) or **conjecture** (my reasoning, not checked).

## 1. Verdict

The rule itself holds up better than the plan's weakest evidence suggests, and the
headline claims I could reproduce (Evidence C, D, the pre-asymptotic case, the
quantization) are right. The plan is not ready to execute as written, for four reasons:

1. Two numbers in Evidence B are wrong. The "starting subintervals split (failed
   merges)" counts, 496 for rule 2 and 528 for rule T, are rounding noise from the
   benchmark script; padaquad's own count is 5 and 0. The direction favours rule T,
   but the table and the label must change, and the analysis script never computed
   rule T at all (`pred/warm2_analyze.py:3` lists `rule2, ruleC, ruleG, ruleS`).
2. The dopri5 cost issue (step 9) names the wrong suspect. The hot spot is
   `_rec_remove`, which takes 9.2 of 9.6 s on a 10^5-subinterval call and is hit
   whenever most pairs are flagged for merging. Rule T flags more pairs than rule 2,
   so this cost belongs to this work, and Evidence E measured speed at a tolerance
   (1e-30) where no pair merges, which hides it.
3. "No chain ended less accurate than its cold start" is true at the two tolerances
   tested and false at rounding-level tolerances, which Evidence A excludes by
   construction: gk21 on sin(200 t) at rtol 1e-14 goes from 0.17 to 1.25 times the
   tolerance under rule T while rule 2 stays at 0.17.
4. The plan states that $|\hat Q - (Q_L + Q_R)|$ "predicts the merged subinterval's
   true error". It predicts the error of the merged rule applied to the halves'
   interpolants. Anything the stored values do not see is invisible to it. I
   constructed a case where rule T merges and the warm call returns an error 45 times
   the tolerance with `converged=True`. Rule 2 merges it too, so this is a property of
   every rule that uses stored values, not a regression; but the plan's wording and the
   "0 accuracy-loss pairs for every Gauss-Kronrod and Clenshaw-Curtis method" claim
   describe the 17-integrand set, not the rule, and must say so.

What must change first: fix `_rec_remove` (or budget it) and re-measure Evidence E at a
tolerance where pairs merge; replace the Evidence B split counts with padaquad's
`n_start_steps_split`; add the rounding-level regime to the verification; reword the
true-error paragraph and the safety claim; drop the `_merge_excess_nodes` change and
tighten the `merge_rule` plumbing to one source of truth (Finding 10). The citations
need the corrections in Section 3.

## 2. Findings, by severity

### Finding 1 (high, measured). Evidence B's split counts are an artefact; the real counts are 5 (rule 2) and 0 (rule T)

The plan says: "Starting subintervals split (failed merges) | 496 | 528".

`pred/warm_bench.py:36-47` rebuilds the accepted mesh points as
`start + cumsum(res.h)` and counts a starting subinterval as split when an accepted
point lies strictly inside it. Accumulated rounding in the cumulative sum moves points
by a few ulps, so a point lands "strictly inside" its own subinterval. The original
rows show the symptom: `n_start 5 -> n_accepted 5` with `n_split 3`, which is
impossible (a split adds a subinterval) (`review/warm_chain_detail.log`, gk7 / exp /
1e-10, calls 2-6).

I reran the same 306 chains recording padaquad's own `n_start_steps_split`
(`review/warm_bench_review.py`, logs `review/warm_{rule2,ruleT}.json`,
`review/warm_review_analyze.log`): cold calls identical; warm evaluations 748,243
and 645,858 (0.863, as the plan says); starting subintervals split **5 under rule 2,
0 under rule T**; no chain splits in any warm call after call 1 under either rule; no
warm call is non-converged. The plan's "re-split 8 pairs" in Evidence A and "0 of
4,000 steps" in Evidence D are consistent with this.

Change: replace the row with the padaquad counts, drop the words "failed merges" from
the `n_split` column of the plan and the benchmark, and put `ruleT` into
`warm2_analyze.py` (or use `review/warm_review_analyze.py`).

### Finding 2 (high, measured). The dopri5 slow function is `_rec_remove`, not the add phase, and rule T makes it more likely to run

The plan says: "the add phase of `_get_optimal_mesh`, which splits every failing
subinterval through `_adaptively_increase_mesh`, is the suspect".

`review/profile_dopri5.py` (cProfile, dopri5, 10^5 uniform subintervals,
atol=rtol=1e-30, the setup of `pred/speed.py`; `review/profile_dopri5_100000.txt`):
9.55 s total, of which `_rec_remove` (`padaquad/quadrature/base.py:2496-2539`) takes
9.17 s cumulative, 6.03 s self, with 48,405 passes of its `while` loop, each doing one
`torch.any` and one `torch.concatenate` over the whole mask. `_adaptively_increase_mesh`
takes 0.12 s. The cause is not failing subintervals but merging: at that tolerance every
dopri5 pair ratio is below the cut (k = 5, so the cut is 0.1/16), the mask is all True,
and the loop un-flags about two entries per pass. `mesh_optimal` has 50,076 subintervals,
half the input.

Scaling on an all-True mask (`review/rec_remove_scaling.log`): 0.34, 0.92, 2.56,
6.88 s for N = 10^4, 2·10^4, 4·10^4, 8·10^4, about N^1.45 here and N^2 in the limit
(the pass count grows with N). This is independent of the method and of the rule: gk21
from a 10^5 uniform mesh at atol = rtol = 1e-8 takes 9.2 s under rule T and 8.3 s under
rule 2, against 3.54 s for the plan's 10^6-subinterval gk21 call at 1e-30 where nothing
merges. So Evidence E's "4-15 % of an integration" is the predictor's cost on a call
that never merges; on a call that does merge, the merge path dominates by an order of
magnitude, and rule T merges more often (it exists to).

Change: write the step-9 issue from this profile, and fix `_rec_remove` in or before
this work. Its specification ("a run of flagged entries alternates, TTT -> TFT") is a
parity within runs and can be computed in one pass (run id by `cumsum` of run starts,
position within run, keep even positions); the loop and its O(N) per pass go away.
Re-measure Evidence E at a tolerance where pairs merge.

### Finding 3 (high, measured). The accuracy claim fails at rounding-level tolerances, which Evidence A cannot see

The plan says: "no chain ended less accurate than its cold start (rule 2: 2 chains)" and,
under Confidence, "Conjecture: quantizing $\rho$ to $2^{-48}$ is harmless at tolerances
down to rounding level."

`review/rounding_regime.py` (64-subinterval uniform mesh, atol = 0, rtol in {1e-12,
1e-14, 1e-15}, cold call then four warm calls; `review/rounding_regime.log`):

| case | rule 2, err/tol per call | rule T, err/tol per call |
|---|---|---|
| gk21, sin(200t), rtol 1e-14 | 0.17, 0.17, 0.17, 0.17, 0.17 (64 subintervals kept) | 0.17, **1.25**, 1.25, 1.25, 1.25 (64 -> 59) |
| gk21, sin(200t), rtol 1e-15 | 1.69 throughout | 1.69, **11.16** (64 -> 64 with 3 re-splits) |
| cc65, sin(200t), rtol 1e-14 | 0.17, 1.25, ... (63) | 0.17, **1.79**, ... (54) |
| cc33, sin(200t), rtol 1e-15 | 9.13, 14.55, ... | 9.13, 2.37, ... |
| gk21/cc33/cc65, exp and sin(10t)+1, all three rtol | 64 -> 53-63 | 64 -> 32 -> 16 -> 8 -> 4, err <= 0.26 tol |

On smooth positive integrands rule T is excellent here (it halves the mesh every call
while rule 2 barely merges, as `how_it_works.md` says). On an oscillatory integrand with
cancellation it merges pairs whose true error is at the tolerance, and the warm call is
outside the tolerance while the cold call was inside. Rule 2 does the same on cc65, so
neither rule is safe there, but the plan's claim is "never", and its own Evidence A
excludes this regime: `pred/accuracy_study.py:159` sets the noise floor at
$10^3\,\varepsilon \int|f|$, about 20 times padaquad's rounding floor
$50\,\varepsilon\,|Q|$ (`base.py:2851-2852`), so no pair between the two floors is
ever counted. That is the "window hides failures" case.

Why it happens (conjecture, consistent with the numbers in Finding 5): near the floor
the tolerance is $\max(\tau, 50\varepsilon(|Q_L| + |Q_R|))$ and the cut is a tenth of
it, $5\varepsilon(|Q_L|+|Q_R|)$. The prediction's own rounding noise is about
$\varepsilon\sum_u|\alpha_u||y_u|$, which for a cancelling integrand is set by
$\sum|y|$, not by $|Q|$, and can exceed the cut either way: refuse good merges, or
(as here) accept a merge whose true error is a few times the floor.

Change: add the rounding-level regime to step 8 and to the verification script (at
least rtol 1e-14 on `sin200` and `cancel` for gk21, cc33, cc65), and either accept the
documented limitation or refuse merges whose predicted error is below the pair's floor
$50\varepsilon(|Q_L|+|Q_R|)$ by less than the prediction's noise. The quantization
itself is harmless (Finding 5); the conjecture should be restated as being about the
rule, not the quantization.

### Finding 4 (high, measured). "$|\hat Q - (Q_L + Q_R)|$ predicts the merged subinterval's true error" is wrong as stated; the safety claim describes the test set

The plan says: "$|\hat Q - (Q_L + Q_R)|$ predicts the merged subinterval's **true**
error", and "For every Gauss-Kronrod and Clenshaw-Curtis method except cc5 it has none
[accuracy-loss pairs]".

It predicts the difference between the merged rule *applied to the two interpolants*
and the halves' sum. Any feature that the halves' nodes do not see is absent from
$\hat Q$, $\hat E$, $Q_L + Q_R$, $e_L$ and $e_R$ alike, so $P$ is tiny and the pair
merges; the next call then evaluates $f$ at the merged nodes, which are new points.

Construction (`review/counterexample2.py`, log `review/counterexample2.log`):
$f(t) = e^t + \exp(-((t - c_j)/w)^2)$, a bump of width $w$ centred on merged node
$j$ of the pair $[0, 1/2], [1/2, 1]$, which is not a node of either half. With dopri5,
node $c_2 = 0.3$ (weight 0.449, error weight $-2.8\times10^{-3}$, ratio 158):

| bump width | atol | halves' true err/tol | rule T merges? | warm call: est/tol, true err/tol, converged |
|---|---|---|---|---|
| 1e-7 | 1e-2 | 2.4e-5 | yes ($P = 1.4\times10^{-5}$ against cut 1e-3) | 0.28, **44.9**, True |
| 2e-3 | 1e-2 | 0.35 | yes | 0.28, **44.6**, True |
| 1e-2 | 3e-2 | 0.59 | yes | 0.09, **14.4**, True |

Those are accuracy losses by the plan's own definition (halves inside the tolerance,
merge made, merged subinterval passes its estimate, true error above the tolerance),
at 14-45 times the tolerance, with a bump of width 0.01 on a unit interval. For cc17,
gk21 and cc65 the same construction gives a re-split instead: the error weight at the
hit node is as large as the weight, so the merged subinterval fails its own estimate
(`est/tol` 1e-14 before, 7.0e-2 after for cc17). Rule 2 makes exactly the same
decisions on every row (`review/counterexample2_main.log`), so this is not a
regression; it is the limit of any rule built from stored values, which the plan
acknowledges under "Less sure" but contradicts in "The rule" and in Decision 2
("the remaining losses are marginal (median 1.04-1.10 times the tolerance)" is a
statement about the 17 integrands).

Change: reword to "predicts the error of the merged rule on the halves' interpolants,
which equals the true error when the stored values resolve $f$"; state the condition
once, in the user guide; keep the 17-integrand numbers as what they are. The decision
to make `"predicted"` the default for dopri5 and fehlberg2 is the user's; this finding
does not change it, since rule 2 has the same blind spot with a worse estimate.

### Finding 5 (medium, measured). The quantization is harmless; the prototype's noise amplification is small; float32 works

The plan says: "Conjecture: quantizing $\rho$ to $2^{-48}$ is harmless ... The
prototype agreed with the direct version to 9e-15, but the rounding regime near rtol
≈ 1e-14 was not studied separately" and "Rule T was measured in float64 only".

- Quantization (`review/quantization3.py`, `review/quantization3.log`): for every
  uniform method, $\rho \in \{1, 2, 0.37, 10^{\pm 2}, 10^{\pm 3}\}$ with merged width
  1, $f = \sin 10t + 1$ and $f = e^t$, a half-quantum change of $\log_2\rho$ changes
  $\hat Q$ and $\hat E$ by at most $1.4\times10^{-15}$ of the merged rule's rounding
  scale and at most 0.24 of the pair's rounding floor (fehlberg2, $\rho = 1$); typically
  0.01-0.05 of the floor. Harmless. (My first attempt, `review/conditioning.log`,
  compared the vectors instead of the prediction and shows O(1) changes at $\rho = 1$:
  a merged node sits exactly on the shared mesh point and its weight moves from the
  left vector to the right one. The plan's design therefore relies on
  $y_L[-1] = y_R[0]$ at the shared node; `_compute_nodes` (`base.py:483-484`) computes
  that node as `left + 1.0 * (right - left)` for the left half and `right + 0` for the
  right half, which can differ by an ulp. Harmless at $\varepsilon$ level; worth one
  comment.)
- Noise amplification $\sum_u|\alpha_u||y_u| / \sum_j|b_j||y_j|$ is 1.0-1.9 for all
  methods and ratios, except dopri5 (2.0-3.3) (`review/quantization3.log`, last
  column). So the prediction's rounding noise is at most a few times the merged rule's
  own, and for a positive integrand stays below the cut-times-floor
  ($5\varepsilon|Q|$) by a factor of 2-5. That margin is what Finding 3 loses when
  $|Q| \ll \sum|y|$.
- Barycentric weights (`review/conditioning.log`, `poly64` column): the prototype
  reproduces the merged rule on a random polynomial of degree $U - 1$ to
  $< 1.2\times10^{-15}$ of scale for every method and $\rho$ up to $10^{\pm4}$, so the
  weights for 23-65 Gauss-Kronrod or Chebyshev nodes are well conditioned in float64,
  as Berrut & Trefethen Sec. 5 says for clustered nodes (Section 3 below). With float32
  stored values (`poly32`) the error is $3\times10^{-8}$ of scale, i.e. one float32
  epsilon times the amplification; no blow-up.
- float32 end to end (`review/float32_rule_t.py`, `review/float32_rule_t.log`): gk21,
  cc33, gk7, dopri5 on exp, runge, kink, sin10, gauss_peak at atol = rtol = 1e-5 and
  1e-6, cold plus four warm calls. Rule T: no NaN, no re-split after call 1, every
  error within its cold value, and the smooth cases merge 11 -> 6 -> 3 -> 2 -> 1
  subintervals. Rule 2 cannot merge at all for gk21 and cc33 in float32 (254 and 364
  evaluations on every call, as documented for the round-off regime). This is a point in
  the plan's favour that it did not have.

Change: move the quantization from "conjecture" to "measured" with these numbers; add
the float32 chains to Evidence C or the verification script.

### Finding 6 (medium, measured). Evidence A's counts are not reproducible from run to run, and one of its logs crashed

The plan says: "Require the rule T columns to match the prototype within run-to-run
noise".

`pred/accuracy_study.py:39` seeds one generator and `meshes()` (`:90-106`) draws the
random meshes from it on every (method, integrand) call, so the meshes depend on which
methods ran before. `pred/accuracy_T.log` (run with `ONLY_METHODS`) gives fehlberg2 230
and dopri5 40 accuracy-loss pairs where `accuracy3.log` and `accuracy_factors.log` give
198 and 35; the plan quotes 198 and 35. `accuracy_T.log` also ends in an `IndexError`
(`accuracy_study.py:223`) before printing the factors. The "run-to-run noise" the plan
asks the implementer to match is therefore undefined.

The definition itself is sound: a pair is an accuracy loss when a tolerance exists in
$(\max(P/c_\text{cut}, E_M, T_L + T_R, \text{noise}),\ \min(T_M, 10^{-2}\int|f|))$,
which is exactly "the rule merges, the merged subinterval passes its estimate, the halves
were accurate, the merged one is not"; the count is comparable between rules because every
rule sees the same pairs and window (`accuracy_study.py:158-176`). The window's lower
edge is what Finding 3 is about.

Change: seed per (method, integrand) or generate the meshes once; fix the crash; quote
the counts with the seed.

### Finding 7 (medium, from the code). The memory claim is untested and the detach must move inside

The plan says: "gathers the vectors and applies them with `einsum`, in chunks of 2^16
pairs so that peak memory stays near the record's own size".

The prototype `predict_rule_t` (`pq_pred/padaquad/quadrature/_merge_prediction_fast.py:43-68`)
has no chunking; lines 62-63 copy `y[:-1][:, first, :]` and `y[1:][:, first, :]` to
float64. Measured peak growth during the call (`review/memory_predict2.py`,
`review/memory_predict2.log`): 2.14 x `y` for cc65 with 4·10^5 subintervals, 2.38 x for
gk21 with 10^6, 1.45 x for gk21 with D = 4. For a float32 record the float64 copies
double that. So "near the record's own size" is 1.5-2.4 times the size of `y` in the
prototype; chunking would bring it down, but every Evidence E timing was taken without
it, so step 5's timings have to be redone after chunking.

Two more points on this function:

- `record["y"]` keeps the autograd graph when `take_gradient=False`
  (`base.py:3375`, `:3441`: detached only when the per-batch backward ran). The
  prototype detaches at the call site (`pq_pred .../base.py:2465`); the plan's
  `_predict_merged_errors(nodes, y, mesh_quadratures)` must detach inside, or every
  einsum goes into the graph.
- For a GPU solver the record's `y` and `nodes` are on `result_device`, default
  `"cpu"` (`base.py:622`), and `_get_optimal_mesh` runs there
  (`base.py:1131-1140`). So "the GPU cost is unmeasured" means in practice "the CPU
  cost on the offloaded record"; the vectors are built on the method's device, which
  `_integration_on_device` moves to `result_device` for the duration (`base.py:2368-2397`),
  so devices are consistent.

### Finding 8 (medium, from the code). The `_merge_excess_nodes` change is untested by the prototype and unnecessary

The plan says: "`_merge_excess_nodes` (`uniform.py`, `variable.py`): record the merged
subinterval's error as its pair's `pair_errors` rather than the sum, so the value that
decided the merge is the one stored."

The prototype did not do this: `pq_pred` still records the sum
(`uniform.py:176-179`, `variable.py:211-214` are unchanged), so none of Evidence A-D
covers the change. In the add phase, both values give a ratio below 1: the sum is at
most $P$ (Eq. 3 adds it), and $P/\tau < c_\text{cut} < 1$ for a merged pair; the
rounding floor uses `mesh_quadratures`, which is the same sum either way. So the
behaviour is identical, and the change needs a new argument on both subclasses
(`pair_errors[remove_idxs]`) for nothing. Nothing pins the current error sum:
`test_mesh_quadratures_accumulated` (`tests/test_prune_and_optimal.py:98-110`)
checks the quadrature sum only. Drop step 4's `_merge_excess_nodes` bullet ("least new
code").

### Finding 9 (medium, from the code). Eq. (3) is consistent with both reductions and with cumulative mode; the `pair_errors` plumbing is right

One line: `pair_errors` [N-1, D] replaces `abs_err[:-1] + abs_err[1:]` and is reduced
with `_reduce_norm` (`base.py:3018-3019`) or with `amax` over components
(`base.py:3105-3106`); the tolerance (right subinterval's in cumulative mode,
`base.py:3010`, `:3087`) and the floor (`:3015`, `:3092-3095`) are untouched; a NaN
in `pair_errors` gives a False comparison and no merge. `test_pair_ratio_is_merged_step_ratio`
(`test_prune_and_optimal.py:266-291`) would need a `pair_errors` variant, which the plan
already lists. Correct as planned.

### Finding 10 (medium, against CLAUDE.md). The code plan creates two sources of truth and some new state it does not need

- **The rule is read in two places.** The prototype decides the numerator in
  `_get_optimal_mesh` and the cut in `_prune_excess_mesh` through a separate
  `predicted` flag (`pq_pred .../base.py:2453-2470`, `:2483`, `:2344-2349`); the plan
  keeps that split and adds a three-way string in both. Decide once: a method on the
  solver that returns `(pair_errors, merge_cut)` for the configured rule, consumed by
  `_prune_excess_mesh`. `"sum"` is `"power_law"` with k := 1 (the cut is
  `remove_cut / 2**(k - 1)` in both), so it needs no branch of its own.
- **New method state.** Step 3 stores the distinct node fractions, their first index and
  the barycentric weights on every `MethodClass`, recomputed in `to_device` and copied in
  `clone`. All three derive from `node_fractions()`; `clone()` is a deep copy
  (`methods/_base.py:424-437`) so it copies them already, but `to_device` must move three
  more tensors plus the per-$\rho$ cache of step 5. Keep them derived: one cached
  function of `(node_fractions, rho-key, device)` on the engine side, which is where the
  only caller is. `error_exponent` stays, as the plan says (rule 2 and the methods table).
- **Two quantization constants.** The plan and `predict_rule_t` say $2^{-48}$ in
  $\log_2\rho$ (`_merge_prediction_fast.py:47`, `:56`); the module docstring says
  "relative $2^{-49}$" (`:13`). One number, one place.
- **The `allclose` invariant check** on the stored nodes is fine: the record's nodes are
  always placed by `_compute_nodes` (the variable solver's subsampled merged nodes exist
  only after pruning and never enter a record).

### Finding 11 (medium, measured and from the code). Tests and verification script

- **Pre-asymptotic case** (`review/preasymptotic.py`, `review/preasymptotic.log`):
  gk21 on $1/(1 + 25(t - 1/2)^2)$, mesh $[0, 1/2, 1]$, $|I| = 2\arctan(2.5)/5 =
  0.4761$ (so "atol = rtol·|I| = 1e-2" means rtol = 0.021; write both numbers). Both
  halves pass with estimates 3.17e-9; `pq_pred` returns `mesh_optimal = [0, 1]`, `pq_main`
  returns `[0, 0.5, 1]`; the warm call on the merged subinterval has estimate 3.45e-4 and
  true error 9.88e-8, as the plan states. The test behaves as described.
- **"Never worse than cold" tests** depend on the seeded random starting mesh
  (`base.py:2698`, `torch.rand` on `self.device`): the stream differs between CPU and
  CUDA and the chains the plan cites (dopri5, poly_deg40: 3.99 -> 6.31) are outside the
  tolerance from the cold call, so the test would assert a property of a failed
  integration. Use an explicit starting mesh, or mark the test CPU-only and assert only
  `max(tol, 1.05 * cold)`.
- **Factor 1.05** (Finding 1's rerun, `review/warm_review_analyze.log`): with the factor
  at 1.0 the "worse than cold" count is 17 for both rules, all dopri5 chains already
  outside the tolerance that drift by at most 2 %; the largest warm/cold error ratio among
  chains with a measurable cold error is 7.4 for rule T (gk15, kink_c2: 0.003 -> 0.022 of
  the tolerance) and 2.2 for rule 2. The conclusion "rule T never makes a converged chain
  worse" survives weakening the factor at these two tolerances; it does not survive
  Finding 3's regime.
- **The exactness test** is right; note that dopri5 has 6 distinct fractions (`review/checks_basic.log`), so its interpolant has degree 5.
- **Determinism of the verification script**: evaluations and errors are deterministic
  on CPU with the seed; the plan's budget ("about one minute") does not include a
  `_rec_remove` fix, which is irrelevant at these mesh sizes.

### Finding 12 (low, measured). Evidence C and D reproduce; the rel_l2 claim is correctly hedged

- `pred/snapshots_T.log`: 40 failed, 350 passed, every failure on `n_optimal_mesh`.
- pode (`runs/s{0..4}_pq_{122,main,predT}`, recomputed by `review/pode_reanalyze.py`,
  log `review/pode_reanalyze.log`): evaluations relative to 1.2.2, geometric mean over 5 seeds,
  harmonic 1.325 (main) and 1.042 (rule T), damped 1.518 and 1.053; rel_l2 ratios 1.23 /
  0.80 and 0.96 / 1.00 with per-seed ratios spanning 0.47-2.6, so the plan's "no effect
  of the merge rule on training" is the right reading; worst loss-integral error over
  5 x 400 steps 5.2e-5 (1.2.2), 2.6e-9 (main), 3.7e-6 (rule T) of the tolerance, as
  stated. I could not verify "0 of 4,000 steps re-split" from the run files (they record
  `n_warm`, `n_frozen`, `n_optimal` only). Note damped seed 4 is bit-identical between
  1.2.2 and rule T (same rel_l2 and evaluation count), i.e. every merge decision
  coincided; worth a sentence in the write-up since it is unexpected.

### Finding 13 (low, measured). Structural claims that are right

`VariableMethodClass.weights` rescales the nodes (`methods/_base.py:645-646`); the
weights on $[0, 1 + \rho]$ and on a shifted, scaled copy equal those on $[0, 1]$
(`review/checks_basic.log`). Every registered method's fractions start at 0 and end at 1
(`_check_node_fractions`, `methods/_base.py:276-304`); dopri5 is the only one with a
repeated fraction (two nodes at 1, `runge_kutta.py:105-107`), which `reference_nodes`
removes by value (`_merge_prediction.py:24-41`). Gauss-Kronrod rules carry two
zero-weight padded endpoints (`gauss_kronrod.py:137-146`); they are evaluated and
stored, so they enter the interpolant, which is what the plan assumes (23 distinct nodes
for gk21). The 17 antiderivatives in `pred/cases.py` differentiate back to their
integrands to 2.4e-7 or better by central differences (`review/checks_basic.log`,
section 3; the 1e-7-level cases are the sharp ones, where the finite difference is the
limit, not the formula). The plan's scope statements on `take_gradient`,
`error_integral_reference`, `max_adaptive_splits` and `same_f` are consistent with the
code; none of them is exercised by the evidence (the chains pass `mesh=`, never
`same_f=True`, `warm_bench.py:70-71`), and only the fixed-point test in step 2 would
exercise `same_f`.

### Finding 14 (low, conjecture plus sources). Prior art on coarsening

No adaptive quadrature code I know of coarsens a mesh: QUADPACK's `dqagse`, SciPy's
`quad_vec` and Gonnet's `cquad` bisect only (the first two are the sources already in
`docs/references.bib`; I did not open `cquad` or Chebfun for this review). Coarsening
with a predicted error is standard in ODE step-size control, where the next step grows
by $(\text{tol}/\text{err})^{1/(p+1)}$, which is the same power-law idea as rule 2, and in
adaptive mesh refinement for PDEs. The plan's "Mine, not from the literature" labelling
is therefore right, and could say this in one sentence.

## 3. Citations

### Halves-versus-whole error test

- **Gander & Gautschi, "Adaptive quadrature - revisited", BIT 40(1), 2000, 84-101,
  doi:10.1023/A:1022318402393.** Opened (`review/sources/gander_gautschi_2000.pdf`, text
  in `.txt`). Sec. 2, Eq. (2.1): the conventional stopping test `abs(i1 - i2) <
  tol * abs(i1)` for two estimates, with the $\sqrt{x}$ example where it never
  terminates; Eq. (2.7): the machine-independent criterion `is + (i1 - i2) == is` with
  `is` a rough estimate of the integral. Sec. 3: "The idea of adaptive Simpson
  quadrature is old [12]" with [12] = Lyness 1969; their `adaptsim` uses `i1 = S(h)`
  and `i2 = S(h/2)`, overwritten by one Romberg step `i1 = (16*i2 - i1)/15`. Sec. 1,
  p. 85: "it is impossible to construct a program that is foolproof ... It is easy to
  construct a function f which a given program will not integrate correctly [10]" ([10]
  = Kahan 1980). Sec. 4.4(i): a tolerance relaxation "has already been suggested by Lyness
  in [12, Modification 1]". The failure mode relevant to *merging* is the one in that
  Sec. 1 sentence: two estimates agreeing by accident, which Finding 4 constructs.
  Suggested entry:

  ```
  @article{gander2000,
    author  = {Gander, Walter and Gautschi, Walter},
    title   = {Adaptive quadrature---revisited},
    journal = {BIT Numerical Mathematics},
    volume  = {40}, number = {1}, pages = {84--101}, year = {2000},
    doi     = {10.1023/A:1022318402393},
    checked = {PDF opened 2026-11-14 (copy at users.wpi.edu/~walker/MA3457/HANDOUTS/). Checked Sec. 2 Eqs. (2.1) and (2.7) (stopping tests), Sec. 3 (adaptsim: i1 = S(h), i2 = S(h/2), Romberg (16 i2 - i1)/15; "The idea of adaptive Simpson quadrature is old [12]" = Lyness 1969), Sec. 1 p. 85 (no foolproof program; Kahan [10]), Sec. 4.4(i) (Lyness Modification 1).}
  }
  ```

- **Lyness, "Notes on the adaptive Simpson quadrature routine", J. ACM 16(3), 1969,
  483-495, doi:10.1145/321526.321537.** Not opened (ACM paywall); the bibliographic data
  were confirmed from the ACM author index and Gander & Gautschi's reference [12]. It is
  the source of the *modifications* to adaptive Simpson (Gander & Gautschi cite its
  "Modification 1"), not of the halves-versus-whole test. Cite it only for what it is
  cited for in `gander2000`, or not at all.

- **Kuncir, "Algorithm 103: Simpson's rule integrator", Comm. ACM 5(6), 1962, p. 347,
  doi:10.1145/367766.368179, and McKeeman, "Algorithm 145: Adaptive numerical
  integration by Simpson's rule", Comm. ACM 5(12), 1962, p. 604.** Neither opened
  (paywall). Secondary sources (Wikipedia "Adaptive Simpson's method"; the web summary of
  Gonnet's review arXiv:1003.4629) credit Kuncir with the first published recursive
  adaptive Simpson routine and McKeeman with a trisection variant. I downloaded Gonnet's
  review (`review/sources/gonnet_2010_review.pdf`) but its text extraction with
  ghostscript is unsearchable (no hit even for "Simpson"), so I could not confirm this
  from the review itself. Under the repository's rule ("only sources that were opened and
  checked"), do not add `kuncir1962`, `mckeeman1962` or `lyness1969` to
  `references.bib`; cite `gander2000` and write "the halves-versus-whole test of adaptive
  Simpson quadrature (Gander & Gautschi 2000, Sec. 3), which goes back to the 1962
  Communications of the ACM algorithms" without a key.

### Barycentric interpolation

- **Berrut & Trefethen, "Barycentric Lagrange interpolation", SIAM Review 46(3), 2004,
  501-517, doi:10.1137/S0036144502417715.** Opened
  (`review/sources/berrut_trefethen_2004.pdf`, copy from people.maths.ox.ac.uk/trefethen/).
  Eq. (3.2): $w_j = 1/\prod_{k \ne j}(x_j - x_k)$; Eq. (3.3): the "first form";
  Eq. (4.2): the "second (true) form" $p(x) = \sum_j \frac{w_j}{x - x_j} f_j \big/
  \sum_j \frac{w_j}{x - x_j}$, with the remark that a common factor of the weights
  cancels (what `reference_nodes` uses when it scales by the largest weight); Sec. 5:
  equispaced weights vary like $2^n$ (Eq. (5.1)) and the interpolation is
  ill-conditioned, while nodes clustered at the ends with density $(1 - x^2)^{-1/2}$
  keep the weights of comparable scale; Eq. (5.3): the Chebyshev weights. The prototype
  cites "(4.1)-(4.2)"; (4.1) is only the identity $1 = \ell(x)\sum_j w_j/(x - x_j)$, so
  cite (3.2) and (4.2). Suggested entry:

  ```
  @article{berrut2004,
    author  = {Berrut, Jean-Paul and Trefethen, Lloyd N.},
    title   = {Barycentric {L}agrange interpolation},
    journal = {SIAM Review},
    volume  = {46}, number = {3}, pages = {501--517}, year = {2004},
    doi     = {10.1137/S0036144502417715},
    checked = {PDF opened 2026-11-14 (author copy, people.maths.ox.ac.uk/trefethen/barycentric.pdf). Checked Eq. (3.2) (weights), Eq. (3.3) (first form), Eq. (4.2) (true barycentric formula; common weight factors cancel), Sec. 5 Eq. (5.1) (equispaced weights grow like 2^n, ill-conditioned) and Eq. (5.3) (Chebyshev weights).}
  }
  ```

## 4. The dopri5 profile (item 15)

`review/profile_dopri5.py`, full output `review/profile_dopri5_100000.txt`; dopri5,
10^5 uniform subintervals on [0, 1], $f = e^{-t}\sin 3t$, atol = rtol = 1e-30,
`max_adaptive_splits=0`, 8 threads, `pq_main`. Warm-up call 9.28 s, profiled call
9.55 s; 100,000 accepted subintervals, `mesh_optimal` 50,076, converged.

| function | self s | cumulative s | calls |
|---|---|---|---|
| `_rec_remove` (`base.py:2496`) | 6.03 | 9.17 | 1 (48,405 loop passes) |
| `torch.any` (inside the loop) | 2.16 | 2.16 | 48,405 |
| `torch.concatenate` (inside the loop) | 0.80 | 0.80 | 48,404 |
| `_get_mesh_indices` (`base.py:2740`) | 0.24 | 0.26 | 3 |
| `_adaptively_increase_mesh` (`base.py:1354`) | 0.01 | 0.12 | 2 |
| `_merge_excess_nodes` (`uniform.py:128`) | 0.01 | 0.01 | 1 |

The add phase is not on the list. The time is in the merge phase's conflict resolution:
every pair is flagged (dopri5's cut is 0.1/16 and every ratio is at the rounding
floor), and `_rec_remove` resolves adjacent flags about two per pass with an O(N) pass.
The same function costs 6.9 s on an all-True mask of 8·10^4 entries under either tree,
and gk21 at atol = rtol = 1e-8 from a 10^5 uniform mesh spends the same 9 s
(`review/rec_remove_scaling.log`). For the issue: title it after `_rec_remove`; the
plan's "238 s at 10^6" is this function; gk21 and cc65 show the same growth whenever
their pairs merge, which the speed test's atol = 1e-30 prevented.

## 5. What I could not check

- **Kuncir 1962, McKeeman 1962, Lyness 1969**: paywalled; attribution rests on
  Gander & Gautschi's reference [12] and secondary sources. Gonnet's 2010 review was
  downloaded but its text could not be extracted on this node (no `pdftotext`,
  `pypdf` or `fitz`; ghostscript's `txtwrite` produced unsearchable output for that
  file, though it worked for the other two).
- **GPU**: no GPU on the login node; the prediction runs on `result_device` anyway
  (Finding 7).
- **pode "0 of 4,000 steps re-split"**: the run files do not record it.
- **The full padaquad suite on `pq_pred`**: not run here (the plan says it was not run
  either); only `tests_T.log` (383 passed, 7 files) exists.
- **The prototype's 10^6-subinterval timings**: not repeated (a `_rec_remove` call at
  that size would take about 15 minutes); the 10^5 scaling above is the evidence.
- **Evidence A's 2.57 million pairs**: not rerun (about 2 minutes per rule per run, but
  the point in Finding 6 is reproducibility, which a rerun would not settle without
  fixing the seeding).
- **`Chebfun` and `cquad` source**: not opened (Finding 14 is labelled conjecture).
