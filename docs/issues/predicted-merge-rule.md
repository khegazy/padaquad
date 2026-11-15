# Plan: decide mesh merges from a predicted merged error ("rule T")

Status: plan, not started. Written 2026-11-14 against `main` at `2b04ea1`; revised the
same day after the adversarial review in
[`predicted-merge-rule-review.md`](predicted-merge-rule-review.md) (see
[Response to the review](#response-to-the-review)). Written so that someone who was not
in the original session can execute it. Every number below was measured with the
prototypes in `/pscratch/sd/k/khegazy/u028` (paths in [Files](#files)); nothing in the
repository has been changed yet.

## Summary

- **Problem.** `mesh_optimal` merges two neighbouring subintervals only when
  `2**(k-1) * S < remove_cut` (rule 2, since 1.3.0; `S` and `k` are defined in
  [Terms](#terms)). The factor `2**(k-1)` assumes the error estimate shrinks like
  `h**k`. On wide subintervals that is false: in pode's training loop the merged
  error grew by about 2^8 to 2^9, not the assumed 2^20 for `gk21`. So rule 2
  refuses safe merges, and warm-started callers pay for it: pode used 24–74 % more
  integrand evaluations under 1.3.0 than under 1.2.2 (U-028).
- **A second, independent problem (found by the review).** `_rec_remove`, which
  resolves neighbouring merge candidates, loops about N/2 times over the whole mask
  when most pairs qualify, so building `mesh_optimal` takes 9.1 s of a 9.5 s call at
  10^5 subintervals for every rule. A one-pass version gives identical output and
  takes 2 ms. It is fixed first (step 2).
- **Proposal (rule T).** For each pair, compute the merged subinterval's integral and
  error estimate from the integrand values already stored for its two halves:
  interpolate each half and apply the method's weights at the merged nodes. Merge
  when both the predicted error estimate and the difference between the predicted
  merged integral and the sum of the halves are small. No new integrand evaluations.
- **Evidence (measured).**
  - Against 50-digit exact integrals, over 2.57 million pairs (14 methods, 17
    integrands) and tolerances down to the rounding level:
    - rule T has 86 pairs where a merge can cost accuracy, against 12,058 for
      rule 2 and 23,215 for rule 1;
    - none of rule T's are on a Gauss–Kronrod or Clenshaw–Curtis method;
    - its worst is 4.35 times the tolerance (rule 2: 66);
    - it makes 99 % of the safe merges on the high-order rules, where rule 2 makes
      50–74 %.
  - Warm-start chains (306 chains): 13.7 % fewer evaluations than rule 2; no
    starting subinterval had to be split again (rule 2: 5); no chain ended less
    accurate than its cold start (rule 2: 2).
  - pode (5 seeds): evaluations return to within 4–5 % of 1.2.2 (`main`: +32 % and
    +52 %).
  - padaquad's tests pass; integrals are unchanged in all 390 snapshot cases.
- **Cost (measured, with the `_rec_remove` fix).**
  - A million-subinterval merging call: building `mesh_optimal` takes 0.70 s under
    rule T against 0.52 s under rule 2 (gk21).
  - Small meshes: about 0.7–1.0 ms more per call; step 5 caches what causes it.
  - Memory: the chunked predictor's peak is at most 0.76 times the size of the
    stored integrand values.
- **All three rules stay available** through one constructor option, `merge_rule`,
  with rule T the default (decided by the user, 2026-11-14).
- **Known limit, shared by every rule.** A merge decided from stored values cannot see
  a feature that the stored nodes missed (worked example in [The rule](#the-rule)).

## Response to the review

The review was checked finding by finding; scripts are in `pred/` unless the review's
own (`review/`) is named.

| Review finding | Verdict after checking | Change to the plan |
|---|---|---|
| 1. Split counts 496/528 are an artefact | **Confirmed.** All 250 rule T rows with `n_split > 0` had no subinterval added, which is impossible. padaquad's own count (review rerun) is 5 (rule 2) and 0 (rule T). | Evidence B uses `n_start_steps_split`. |
| 2. The dopri5 hot spot is `_rec_remove` and rule T hits it more | **Confirmed.** The loop takes 0.35 s at 10^4 and 3.2 s at 4·10^4 on an all-True mask; a one-pass version is identical on 4,956 random masks and takes 2 ms at 10^5 (`rec_remove_check.py`). | New step 2 fixes it before rule T; the issue (step 9) is written from the profile. Evidence E is re-measured at a tolerance where pairs merge. |
| 3. "Never worse than cold" fails at rounding-level tolerances | **Partly confirmed.** The chains do cross `rtol·|I|` (gk21, sin 200t, rtol 1e-14: 0.17 → 1.25). But in every such chain, under both rules, the error is at most 2.3 % of padaquad's own summed rounding floor (`rounding_floor_check.py`), and rule 2 crosses the same way (cc65: 0.17 → 1.25). At the level of single merges, with 50-digit references, rule T has no Gauss–Kronrod or Clenshaw–Curtis losses at these tolerances (`accuracy_study2.py`). The first attempt to measure this regime flagged 2,909 losses; mpmath showed those merged subintervals were accurate to ≤ 3 eps of their own ∫\|f\|, and the flags came from float64 reference integrals (`diag_rounding_loss.py`). | Evidence A now includes the rounding regime with 50-digit references. The chain criterion is stated against the accuracy padaquad can certify (Eq. (6)). Rounding-level chains go into the tests and the verification script. |
| 4. The true-error term is overstated | **Confirmed.** The review's narrow-bump case gives dopri5 merges that cost 14–22 times the tolerance, and rule 2 makes identical decisions; for gk21, cc17 and cc65 the merged estimate catches the bump and the subinterval is split again. | "The rule" is reworded; the limit is stated, with this example, in the user guide. |
| 5. Quantization harmless; float32 works | **Confirmed** (from the review's logs; one detail: gk7 on sin10 in float32 went 0.00 → 0.20 of the tolerance, so "within its cold value" should read "within the tolerance"). | Quantization moves from conjecture to measured; float32 chains go into the verification script. |
| 6. Evidence A not reproducible | **Confirmed.** | `accuracy_study2.py` builds its meshes once from one seed; counts are now deterministic. |
| 7. Memory untested; detach must move inside | **Confirmed.** `record["y"]` keeps the graph when `take_gradient=False` (`base.py:3441`). Chunking measured: peak growth 0.01–0.76 × `y` instead of 1.45–2.38 × (`memory_chunked.py`). | The predictor detaches its inputs and works in chunks of 2^16 pairs. |
| 8. The `_merge_excess_nodes` change is untested and unnecessary | **Confirmed** (the prototype never made it; the add phase cannot tell the difference). | Dropped. |
| 9. `pair_errors` plumbing is right | Agreed. | None. |
| 10. Two sources of truth; new method state | **Agreed.** | One method returns the pair errors and the cut for the configured rule; no new state on `MethodClass`; one quantization constant. Rule-dependent data is cached on the solver, next to where the method is built. |
| 11. Tests depend on the random start; pre-asymptotic tolerance unclear | **Confirmed.** Equal-width starts never reproduce the kink failure. A captured explicit mesh does: from it, rule C goes 0.09 → 32.9 times the tolerance, while rule T and rule 2 stay at 0.09 (`kink_capture.py`, mesh in `kink_mesh.txt`). | Tests start from explicit meshes; the pre-asymptotic test states atol and rtol. |
| 12. pode's "0 re-splits" not verifiable | **Not confirmed.** A warm start was split again exactly when `n_frozen > n_warm` (padaquad's loop never merges), and both are in the run files; the count is 0 in all 10 rule T runs. Damped seed 4 is bit-identical between 1.2.2 and rule T. | The method of counting is stated. |
| 13. Structural claims right | Agreed. | None. |
| 14. Prior art | Taken as the review's conjecture. | One sentence in Connections. |
| Citations | **Confirmed** against the saved texts: Gander & Gautschi 2000 Sec. 3 compares one Simpson value with the two halves' and cites Lyness 1969 as its reference [12]; Berrut & Trefethen 2004 Eq. (3.2) gives the weights, Eq. (4.2) the formula, and Eq. (4.1) is only an identity. | Cite `gander2000` and `berrut2004`; do not add Lyness, Kuncir or McKeeman (not opened). |

## Terms

- **Pair**: two neighbouring subintervals $[a, m]$ and $[m, b]$, the *left* and
  *right half*, of widths $h_L = m - a$ and $h_R = b - m$, with
  $\rho = h_R / h_L$ and $H = b - a$.
- **Merged subinterval**: $[a, b]$, which replaces the pair in `mesh_optimal`.
- **Node fractions** $c_j$ ($j = 1..C$): the method's node positions as fractions
  of a subinterval (`method.node_fractions()`); every method includes 0 and 1, and
  only dopri5 repeats a fraction.
- **Weights** $b_j$ and **error weights** $b^\text{err}_j$: from
  `method.weights(nodes)`. A subinterval contributes $Q = h \sum_j b_j y_j$ and has
  the error estimate $e = h \sum_j b^\text{err}_j y_j$.
- **Rounding floor** of a subinterval: $50\,\varepsilon\,|Q|$ (`_round_floor`), with
  $\varepsilon$ the machine epsilon. padaquad raises every tolerance to it.
- **Tolerance** $\tau$ of the merged subinterval: the pair tolerance
  `_compute_error_ratios` already builds (`max(atol, rtol * |I|)`, the right
  subinterval's in cumulative mode, raised to the halves' summed floors).
- **Error ratio**: error divided by $\tau$; a subinterval passes when it is below 1.
- $S = (|e_L| + |e_R|)/\tau$: the pair's summed error ratio. $k$: the method's
  `error_exponent`. $c_\text{cut}$: `remove_cut` (default 0.1).
- **Rules.** Rule 1 (`"sum"`, 1.2.2): merge when $S < c_\text{cut}$. Rule 2
  (`"power_law"`, 1.3.0): merge when $S < c_\text{cut}/2^{k-1}$. Rule T
  (`"predicted"`): Eqs. (1)–(4).
- **Cold call**: an integration from a given or random starting mesh. **Warm call**:
  one started from the previous call's `mesh_optimal`.
- **Accuracy loss** (of a merge, at a tolerance): the rule merges, the merged
  subinterval passes its own error estimate on the next call, its true error is above
  its tolerance, and keeping the two halves would have been within theirs.
- **Re-split**: the merged subinterval fails its own error estimate on the next call
  and is split again; this costs evaluations, not accuracy.
- **Recall** (of a rule): the share of safe merges it makes, where a merge is safe
  when both halves pass and the merged subinterval both passes its estimate and is
  within its tolerance.
- **Rounding regime**: tolerances below $10^3\varepsilon\int|f|$.

## The rule

Each half is replaced by the polynomial through its distinct stored values, in
barycentric form (`berrut2004`: weights Eq. (3.2), formula Eq. (4.2); Sec. 5 shows
the formula is well conditioned for nodes clustered toward the ends, as
Gauss–Kronrod and Chebyshev nodes are). The merged subinterval's nodes are
$x_j = a + c_j H$. At each one, the interpolant of the half that contains it gives

$$
\hat y_j =
\begin{cases} p_L(x_j), & x_j \le m,\\ p_R(x_j), & x_j > m, \end{cases}
\tag{1}
$$

and the merged rule applied to these values gives

$$
\hat Q = H \sum_j b_j \hat y_j, \qquad \hat E = H \sum_j b^\text{err}_j \hat y_j .
\tag{2}
$$

The pair's predicted merged error is, for each output component,

$$
P = \max\!\Big(|\hat E|,\; \big|\hat Q - (Q_L + Q_R)\big|\Big) + |e_L| + |e_R| ,
\tag{3}
$$

and the pair is merged when

$$
P / \tau < c_\text{cut}
\tag{4}
$$

(after the existing norm or `failure_fraction` reduction over components, unchanged).
What each term does:

- $|\hat E|$ predicts whether the merged subinterval will pass its own error
  estimate on the next call. If it would not, the merge only buys a re-split.
- $|\hat Q - (Q_L + Q_R)|$ compares the merged rule, applied to the halves'
  interpolants, with the sum of the two halves. This is the halves-versus-whole test
  of adaptive Simpson quadrature (`gander2000`, Sec. 3, which compares one Simpson
  value on an interval with the sum over its two halves), used here to decide
  coarsening instead of refinement. When the stored values resolve the integrand,
  it approximates the merged subinterval's true error. It is what stops a merge
  whose own error estimate would be blind, as in the next paragraph.
- $|e_L| + |e_R|$ keeps rule 1's sum as a floor and leaves a margin for the halves'
  own error.

**Why the second term is needed (measured).** gk31 on $|t - 1/3|$ at atol 1e-10,
rtol 1e-9, warm-started six times from the 20-subinterval mesh in
`pred/kink_mesh.txt`. Without the second term ("rule C"):

1. the subinterval holding the kink was merged with its neighbour; the merged
   estimate was 0.045 of the tolerance, its true error 1.2 times it;
2. two calls later it was merged with a neighbour 1,600 times wider; every merged
   node with non-zero weight then fell on the straight part, the estimate was exactly
   0, and the true error was 32.9 times the tolerance.

Every later call accepted that subinterval. With the second term, rule T refuses both
merges: $\hat Q$ is the straight-line integral and differs from $Q_L + Q_R$ by about
the true error. Rule 2 also refuses them.

**What no rule built from stored values can see (measured, by the review).** A
feature that lies between the halves' nodes is absent from $\hat Q$, $\hat E$,
$Q_L + Q_R$, $e_L$ and $e_R$ alike. Example: $e^t$ plus a bump of width 0.002–0.01
centred on a merged dopri5 node at $t = 0.3$ on the pair $[0, 1/2], [1/2, 1]$. Every
rule merges, and the next call is evaluated at the merged nodes, which hit the bump:

- for dopri5 the merged estimate still passes (the error weight at that node is 158
  times smaller than the weight) and the result is 14–22 times outside the tolerance;
- for gk21, cc17 and cc65 the merged estimate fails and the subinterval is split
  again.

Rule 2 makes the same decisions on every case. The user guide must say this once: a
merge is only as reliable as the stored values' coverage of the integrand.

**Why it is cheap.** $\hat Q$ and $\hat E$ are linear in the stored values. With
$\ell_{ju}(\rho)$ the Lagrange basis of node $u$ of the containing half evaluated at
merged node $j$, which depends on the pair only through $\rho$,

$$
\hat Q = H\big(\alpha_L(\rho)\cdot y_L + \alpha_R(\rho)\cdot y_R\big),\quad
\alpha_{\cdot,u}(\rho) = \sum_{j\,\in\,\text{half}} b_j\, \ell_{ju}(\rho),
\tag{5}
$$

and the same holds for $\hat E$ with $b^\text{err}$ in place of $b$.

- **Per ratio, not per pair.** The four vectors are built once per distinct $\rho$
  and gathered per pair; $\log_2\rho$ is rounded to a multiple of $2^{-48}$ first, so
  pairs whose widths are equal up to rounding share one set.
- **The rounding is harmless (measured by the review):** half a quantum moves
  $\hat Q$ and $\hat E$ by at most 0.24 of the pair's rounding floor.
- **Merged nodes on the shared mesh point:** a merged node exactly on $m$ is taken
  from the left half (`x <= h_L`). The left half's last value and the right half's
  first value can differ in the last bit, because `_compute_nodes` computes $m$ two
  ways; harmless, but worth a comment in the code.

**What the chain checks compare against.** A warm call is judged against the
accuracy padaquad can certify for it:

$$
\text{err}_\text{warm} \le \max\Big(\tau,\; 1.05\,\text{err}_\text{cold},\;
\sum_i 50\,\varepsilon\,|Q_i|\Big),
\tag{6}
$$

the last term being its own summed rounding floor. Without that term, a cancelling
integral at rtol 1e-14 fails under every rule (Evidence B).

## Evidence (all measured)

### A. Merges against exact integrals, tolerances down to the rounding level

Setup:

- **Scope.** 14 methods (every uniform method and `interpolatory3_variable`) × 17
  integrands × the same 41 meshes (equal, random, graded, bisection trees; drawn once
  from seed 0), giving 183,345 pairs per method.
- **Tolerances.** For every pair, 241 tolerances from $10^{-18}$ to $10^{-2}$ times
  $\int|f|$, each raised to padaquad's rounding floor exactly as the solver does.
- **Reference integrals.** True errors come from closed-form antiderivatives, and
  every flagged pair is rechecked with 50-digit antiderivatives (26,879 pairs).
- **Columns.** "Loss" counts pairs with an accuracy loss at some tolerance; "re-split"
  counts pairs that would be split again at some tolerance; "recall" is defined in
  Terms.
- **Script:** `pred/accuracy_study2.py`; log: `pred/accuracy_study2_mp.log`.

| Method | k | Rule 1: loss | Rule 1: re-split | Rule 1: recall | Rule 2: loss | Rule 2: re-split | Rule 2: recall | Rule T: loss | Rule T: re-split | Rule T: recall |
|---|---|---|---|---|---|---|---|---|---|---|
| adaptive_heun | 2 | 19 | 2 | 0.897 | 5 | 1 | 0.856 | 0 | 0 | 0.836 |
| fehlberg2 | 2 | 17,737 | 3 | 0.936 | 11,655 | 1 | 0.907 | 67 | 0 | 0.880 |
| bosh3 | 3 | 171 | 114 | 0.958 | 32 | 19 | 0.907 | 2 | 1 | 0.905 |
| dopri5 | 5 | 5,161 | 49,531 | 0.994 | 364 | 29 | 0.953 | 17 | 0 | 0.959 |
| gk7 | 7 | 5 | 35,688 | 0.997 | 0 | 24 | 0.923 | 0 | 0 | 0.977 |
| gk15 | 15 | 8 | 3,958 | 0.997 | 0 | 44 | 0.795 | 0 | 5 | 0.993 |
| gk21 | 21 | 8 | 1,878 | 0.997 | 0 | 57 | 0.722 | 0 | 21 | 0.994 |
| gk31 | 31 | 33 | 932 | 0.998 | 0 | 20 | 0.607 | 0 | 33 | 0.994 |
| cc5 | 5 | 18 | 67,533 | 0.991 | 2 | 26 | 0.936 | 0 | 0 | 0.947 |
| cc9 | 7 | 14 | 34,247 | 0.997 | 0 | 22 | 0.939 | 0 | 0 | 0.979 |
| cc17 | 11 | 12 | 6,887 | 0.998 | 0 | 51 | 0.878 | 0 | 8 | 0.992 |
| cc33 | 19 | 21 | 1,722 | 0.998 | 0 | 28 | 0.735 | 0 | 19 | 0.995 |
| cc65 | 35 | 7 | 640 | 0.998 | 0 | 12 | 0.500 | 0 | 36 | 0.996 |
| interpolatory3_variable | 3 | 1 | 107 | 0.955 | 0 | 14 | 0.902 | 0 | 0 | 0.900 |
| **Total** | | **23,215** | **203,242** | | **12,058** | **348** | | **86** | **123** | |

- **Losses inside the rounding regime only:** 1,544 for rule 1, 650 for rule 2, and
  6 for rule T (all fehlberg2).
- **Worst loss, as a multiple of the tolerance:** rule 1 87.9, rule 2 66.0, rule T
  4.35 (fehlberg2) and 3.72 (dopri5).
- **Where rule T's losses are:** only on the three Runge–Kutta methods, whose error
  estimates are unreliable on coarse subintervals (fehlberg2 is a documented
  limitation in `examples/verification/integral_accuracy.py`).
- **Where rule T is more cautious:** it has a few more re-split pairs than rule 2 on
  gk31 (33 against 20) and cc65 (36 against 12), and a lower recall on
  adaptive_heun (0.836 against 0.856).

### B. Warm-start chains

Setup:

- **Chains.** 306 in all: 9 methods (dopri5, gk7, gk15, gk21, gk31, cc9, cc17, cc33,
  cc65) × 2 tolerances (atol/rtol 1e-6/1e-5 and 1e-10/1e-9) × the 17 integrands.
- **Each chain** is one cold call from padaquad's seeded random mesh, then six warm
  calls.
- **Scripts:** `pred/warm_bench.py` for evaluations and errors; the review's
  `review/warm_bench_review.py` and `review/warm_review_analyze.log` for padaquad's
  own `n_start_steps_split`.

| | Rule 2 | Rule T |
|---|---|---|
| Integrand evaluations, warm calls | 748,243 | 645,858 (0.863) |
| Starting subintervals split again (`n_start_steps_split`) | 5 | 0 |
| Warm calls that did not converge | 0 | 0 |
| Chains whose warm calls became less accurate than the cold call (error/tol > 1 and > 1.05 × the cold call's) | 2 (dopri5: tanh_front 1.22 → 1.80, poly_deg40 3.99 → 6.31) | 0 |
| Same with the factor 1.0 | 17 | 17 |
| Largest warm/cold error ratio, chains with cold error > 1e-3 of the tolerance | 2.2 (dopri5, cancel: 0.019 → 0.043 of the tolerance) | 7.4 (gk15, kink_c2: 0.003 → 0.022 of the tolerance) |

- **The factor-1.0 row:** all 17 chains are dopri5 chains already outside the
  tolerance at the cold call, drifting by at most 2 %.
- **Evaluations relative to rule 2, per method:** dopri5 0.999, gk7 0.981, gk15
  0.942, gk21 0.911, gk31 0.662, cc9 0.992, cc17 0.965, cc33 0.906, cc65 0.491.
- **Where the remaining failures come from:** every remaining call outside the
  tolerance comes from a cold-call error, where the integrand's feature falls between
  nodes (for example gk7 on a step function, 585,870 times the tolerance under every
  rule).

**Rounding-level chains** (`pred/rounding_floor_check.py`). Setup: 64 equal
subintervals, atol 0, rtol 1e-14 and 1e-15, one cold call and four warm calls, for
gk21, cc33 and cc65.

- **Every warm error is within Eq. (6)** under both rules: at most 2.3 % of the
  summed rounding floor, and padaquad reports `converged=True` throughout.
- **Both rules cross `rtol·|I|` on sin 200t,** whose integral is 130–205 times smaller
  than $\sum|Q_i|$:
  - rule T, gk21 at 1e-14: 0.17 → 1.25;
  - rule 2, cc65 at 1e-14: 0.17 → 1.25;
  - rule 2, cc33 at 1e-15: 9.13 → 14.55.
- **Where they differ:** on `cancel` and `exp`, rule T merges 64 → 32 → 16 → 8 → 4
  subintervals, while rule 2 keeps 53–63.

### C. padaquad's own tests (prototype `pq_pred`)

- **Seven test files:** `test_prune_and_optimal.py`, `test_error_masks.py`,
  `test_split_statistics.py`, `test_error_computation.py`,
  `test_variable_integration.py`, `test_recheck_accepted_panels.py`,
  `test_multi_d_integrand.py`: 383 passed, 6 skipped. This includes "every merged
  subinterval of `mesh_optimal` passes when re-evaluated" and the warm-start
  fixed-point test.
- **`test_snapshots.py`:** 40 of 390 cases fail, all on `n_optimal_mesh` (36 meshes
  smaller, 4 larger, ratios 0.73–1.08); `integral` and `integral_error` are unchanged
  in every case.
- **Not run:** the full suite.

### D. pode (the U-028 regression problems)

Setup: pode's `scripts/regression.py` configuration (CPU, float64, gk21, atol 1e-6,
rtol 1e-5, 400 epochs), 5 seeds per problem. Ratios are geometric means over seeds
relative to padaquad 1.2.2 (the review recomputed them: `review/pode_reanalyze.log`).

| Problem | Evaluations, `main` | Evaluations, rule T | rel_l2, `main` | rel_l2, rule T |
|---|---|---|---|---|
| harmonic_oscillator | 1.32 | 1.04 | 1.23 | 0.80 |
| damped_oscillator | 1.52 | 1.05 | 0.96 | 1.00 |

- **Re-splits.** None under rule T (0 of 4,000 steps); 14–79 steps per run under
  1.2.2. Counted from the run files: pode's adapting pass never merges, so a warm
  start was split exactly when the evaluated mesh had more subintervals than the
  warm mesh (`n_frozen > n_warm`).
- **Accuracy.** The loss integral stayed inside the tolerance at every step for every
  version; the worst step was 3.7e-6 of the tolerance under rule T.
- **Training error.** rel_l2 varies 8–12 times between seeds, so these ratios show no
  effect of the merge rule on training.
- **A curiosity.** Damped-oscillator seed 4 is bit-identical between 1.2.2 and rule T
  (same rel_l2 and evaluation count), so every merge decision coincided in that run.
  Not investigated.

### E. Speed and memory (CPU, 8 threads; one-pass `_rec_remove` in both trees)

Setup: $f = e^{-t}\sin 3t$ on equal subintervals, atol = rtol = 1e-8 (most pairs
merge), `max_adaptive_splits=0`. Script: `pred/speed_merge.py` with trees
`pq_main2` (rule 2) and `pq_pred2` (rule T). Each cell is the whole `integrate()`
call / building `mesh_optimal`.

| Method | Subintervals | Today (`main`) | Rule 2 + fix | Rule T + fix |
|---|---|---|---|---|
| gk21 | 10^5 | 9.50 s / 9.13 s | 0.31 s / 0.068 s | 0.44 s / 0.127 s |
| cc65 | 10^5 | 9.28 s / 8.95 s | 0.39 s / 0.101 s | 0.49 s / 0.146 s |
| dopri5 | 10^5 | 9.46 s / 9.15 s | 0.27 s / 0.064 s | 0.30 s / 0.079 s |
| gk21 | 10^6 | not run (about 15 min) | 2.91 s / 0.52 s | 3.76 s / 0.70 s |
| cc65 | 10^6 | not run | 4.23 s / 0.87 s | 5.67 s / 1.56 s |

- **Noise.** The `integrate()` times at 10^6 are single runs on a shared login node
  and are noisier than the `mesh_optimal` times.
- **Small meshes.** Building `mesh_optimal` on 2–64 subintervals costs 1.5–1.9 ms
  under rule T against 0.6–0.9 ms under rule 2 (mean of 200 calls). Profiling the
  predictor on 4 subintervals: 0.19 ms recomputes the distinct-node data and 0.30 ms
  builds the per-ratio vectors, both cacheable (step 5).
- **Memory, chunked predictor** (`pred/memory_chunked.py`, the review's method),
  peak growth during the prediction:

  | Case | Size of `y` | Unchunked (review) | Chunked |
  |---|---|---|---|
  | cc65, 4·10^5 subintervals | 198 MB | 2.14 × | 0.76 × |
  | gk21, 4·10^5, D = 4 | 281 MB | 1.45 × | 0.01 × |
  | gk21, 10^6 | 175 MB | 2.38 × | 0.17 × |

  The chunked predictor gives output identical to the unchunked one.
- **Device.** The prediction runs on `result_device` (CPU by default), as the rest of
  `_get_optimal_mesh` does, so a GPU solver pays this CPU cost unless it sets
  `result_device` to the GPU.

### F. float32 (measured by the review, `review/float32_rule_t.log`)

- **Setup:** gk21, cc33, gk7 and dopri5 on five integrands at atol = rtol = 1e-5 and
  1e-6, one cold call and four warm calls.
- **Rule T:** no NaN, and no starting subinterval split again after the first warm
  call. Smooth cases merge 11 → 6 → 3 → 2 → 1. Every chain that ends outside the
  tolerance was already outside at the cold call.
- **Rule 2:** cannot merge at all for gk21 and cc33 in float32, so every call costs
  254 and 364 evaluations.
- **Interpolation accuracy:** the barycentric interpolation reproduces polynomials of
  degree U − 1 to 1.2e-15 of scale in float64, and to 3e-8 with float32 values.

### G. Variants tested and rejected

| Variant | Definition | Why it was rejected |
|---|---|---|
| C | $P = |\hat E| + |e_L| + |e_R|$ | Blind when the merged estimate is fooled: the gk31 kink case reaches 32.9 times the tolerance. |
| G, S | C plus an interpolation-error bound (absolute or signed difference from a half-degree interpolant) | Still blind when the feature is narrower than the merged node spacing (the same kink case). Not linear in the values, so the cached vectors do not apply. |
| Merged floor | Judge a pair against $50\varepsilon|Q_L + Q_R|$ instead of the halves' summed floors | Changed 2 of the 2,909 pairs flagged by the float64-reference run; those flags were artefacts. No reason to change `_error_ratios_*`. |

## Implementation plan

Code level: research-grade. Two commits: the `_rec_remove` fix, then rule T.

1. **Branch.** Create `predicted-merge-rule` from `main`.
2. **Fix `_rec_remove` (first commit).**
   - Replace the loop (`padaquad/quadrature/base.py`, `_rec_remove`) with the
     one-pass form in `pred/patch_rec_remove.py`: keep a flagged entry when its
     distance from the start of its run of flags is even (`cummax` of run starts).
   - Test: copy the old loop into `tests/test_prune_and_optimal.py` as the reference,
     and assert identical output on random masks of length 0–40, 100 and 1000 at
     densities 0, 0.2, 0.5, 0.8, 0.95 and 1. Keep the existing `_rec_remove` tests.
   - Write `docs/issues/rec-remove-quadratic-cost.md` in the format of the other
     issue files, from the review's profile (`review/profile_dopri5_100000.txt`:
     `_rec_remove` 9.17 of 9.55 s, 48,405 loop passes) and the timings in Evidence E.
     State that it is fixed by this commit. This is the "dopri5 issue" the user asked
     for: the cost was never specific to dopri5. It hits any call in which most pairs
     qualify for merging.
3. **Write the failing tests for rule T first.**
   - *Exactness*, in `tests/test_prune_and_optimal.py`. For every method in both
     registries and $\rho \in \{1, 2, 1/2, 0.37, 10^{\pm 3}\}$, take a random
     polynomial of degree $U - 1$, where $U$ is the number of distinct node fractions
     (dopri5: 6), so interpolation is exact. Assert that $\hat Q$ and $\hat E$ equal
     the merged rule applied to the polynomial, to $10^{-12}$ of $H\sum|b||y|$.
   - *Pre-asymptotic merge.* gk21 on $1/(1 + 25(t - 1/2)^2)$, mesh $[0, 1/2, 1]$,
     atol = 1e-2, rtol = 1e-12. $|I| = 0.4761$, so the tolerance is 2.1 % of the
     integral. Measured on both prototype trees (review): the halves' estimates are
     3.2e-9; the merged estimate is 3.45e-4 and its true error 9.9e-8. Assert that
     `mesh_optimal` is `[0, 1]` under `"predicted"` and `"sum"`, and `[0, 1/2, 1]`
     under `"power_law"`. This fails on `main`.
   - *Kink chain from an explicit mesh.* gk31 on $|t - 1/3|$, atol 1e-10, rtol 1e-9,
     starting from the 21 mesh points in `pred/kink_mesh.txt` (float hex). Run six
     warm calls and assert each warm error ≤ Eq. (6). Measured: rule T and rule 2
     stay at 0.09 of the tolerance; rule C reaches 32.9. Also run the test once with
     the second term of Eq. (3) removed, to confirm it then fails.
   - *Rounding-level chain.* gk21 on $\sin 200t$, atol 0, rtol 1e-14, 64 equal
     subintervals, four warm calls. Assert Eq. (6) and `converged`. Measured: the
     error is at most 0.7 % of the summed floor. Also assert that `"predicted"`
     takes $\sin 2\pi t$ from 64 to at most 8 subintervals in four calls; measured: 4.
   - *Pair errors in every mode*, in `tests/test_error_masks.py`. A supplied
     `pair_errors` replaces the pair numerator in the norm family and in
     `failure_fraction`, in absolute and cumulative mode, and the rounding floor still
     raises the pair tolerance.
   - *The `merge_rule` option*, in `tests/test_prune_and_optimal.py` and
     `tests/test_solver_methods.py`:
     - the default is `"predicted"`;
     - an unknown value raises `ValueError` at construction, naming the valid values;
     - the option passes through `adaptive_quadrature()` and `integrate()` like
       `remove_cut`;
     - the existing rule 2 tests (`test_merge_cut_accounts_for_merged_step_error`,
       `test_optimal_mesh_passes_when_reevaluated*`,
       `test_repeated_warm_starts_prune_to_a_fixed_point`) are parametrized over
       `"power_law"` and `"predicted"`.
4. **Engine side** (`padaquad/quadrature/base.py`; no change to `MethodClass`).
   - **Constructor option `merge_rule`** on `AdaptiveQuadrature.__init__`
     (keyword-only, like `remove_cut`): `"predicted"` (default), `"power_law"`,
     `"sum"`. Validate with `ValueError`; store once as `self.merge_rule`; document
     the three values and when to use the older two (reproducing 1.3.0 or 1.2.2
     meshes) in the constructor docstring.
   - **One decision point:**
     `_merge_inputs(nodes, y, mesh_quadratures, mesh_quadrature_errors) -> (pair_errors | None, cut)`.
     - `"sum"` returns `(None, remove_cut)`;
     - `"power_law"` returns `(None, remove_cut / 2**(k-1))`;
     - `"predicted"` returns `(prediction + |e_i| + |e_{i+1}|, remove_cut)`. With
       `T > 1` it logs a warning that it falls back to `"power_law"` and returns
       that rule's values.

     `_get_optimal_mesh(..., y)` calls it once and passes `pair_errors` to
     `_compute_error_ratios` and `cut` to `_prune_excess_mesh`. `"sum"` is
     `"power_law"` with $k = 1$, so it needs no separate branch.
   - **`_compute_error_ratios(..., pair_errors=None)`** forwards to both helpers,
     which use it in place of `abs_err[:-1] + abs_err[1:]`. Tolerance, cumulative
     mode and rounding floor are unchanged.
   - **`_prune_excess_mesh(..., cut)`** replaces its internal `merge_cut` line. Keep
     the derivation of the `2**(k-1)` factor in the docstring, with the condition
     under which it holds.
   - **Call site** (around `base.py:1127`): pass `y=record["y"]`.
   - `_merge_excess_nodes` is not changed.
5. **The predictor** `_predict_merged_errors(nodes, y, mesh_quadratures) -> [N-1, D]`
   on `AdaptiveQuadrature`, as `predict_rule_t` in
   `pq_pred2/padaquad/quadrature/_merge_prediction_fast.py`.
   - Detach the inputs inside, since the record keeps the graph when
     `take_gradient=False`.
   - Process pairs in chunks of 2^16.
   - Define one constant for the quantization ($2^{-48}$ in $\log_2\rho$).
   - Raise `ValueError` if the stored nodes are not at the node fractions (one
     `allclose`; an internal invariant, since `_compute_nodes` places every node).
   - Comment the shared-node choice described in [The rule](#the-rule).
   - Cache on the solver the distinct-node data (fractions, first indices,
     barycentric weights) and a bounded dictionary of per-ratio vectors. Build the
     cache lazily and clear it where `_set_solver_dtype` rebuilds the method and in
     `_integration_on_device`.
   - Target: at most 0.3 ms more per call than `"power_law"` on meshes of up to 64
     subintervals. Measure it with `pred/speed_merge.py`, and inside pode as
     wall-clock time per step.
6. **Documentation.**
   - Rewrite the docstrings of `_prune_excess_mesh` and `_get_optimal_mesh`.
   - Rewrite `docs/user_guide/how_it_works.md` "The optimal mesh and the merge rule"
     around Eqs. (1)–(4), with:
     - one paragraph per `merge_rule` value;
     - the limit of stored-value rules (the bump example);
     - the rounding-level behaviour (Eq. (6)).
   - Update `README.md` (around lines 261 and 299), `padaquad/_frontend.py` option
     lists, and the merge paragraph of `CLAUDE.md`.
   - Add a `CHANGELOG.md` entry with the measured numbers above and the
     `_rec_remove` speed-up.
   - Add `gander2000` and `berrut2004` to `docs/references.bib`, with the entries and
     `checked` fields given in the review (Sec. 3); the quoted passages were
     re-checked against the saved texts. Do not add Lyness, Kuncir or McKeeman.
7. **Verification script** `examples/verification/merge_prediction.py`, in the style
   of `integral_accuracy.py` (module docstring explaining what is checked and why,
   `# %%` cells, run by `test_examples_run.py` and the gallery, exit status 1 on
   failure). Budget: about one minute on CPU. Every chain starts from an explicit
   mesh.
   - **Methods:** every uniform method except `adaptive_heun` and `fehlberg2`, plus
     `interpolatory3_variable`.
   - **Integrands:** `exp`, the Lorentzian $1/(1 + 625(t - 1/2)^2)$ and the kink
     $|t - 1/3|$, at atol 1e-6 and rtol 1e-5.
   - **Runs:** one cold call and four warm calls, under `"predicted"` and
     `"power_law"`.
   - **Rounding level:** gk21, cc33 and cc65 on $\sin 200t$ and $\sin 2\pi t$ at
     rtol 1e-14 from 64 equal subintervals.
   - **float32:** gk21 and cc33 on `exp` and the kink at 1e-5.
   - **Printed per method:** warm-call evaluations under both rules,
     `n_start_steps_split`, and the worst warm error divided by the right-hand side
     of Eq. (6).
   - **Fails when:**
     - a `"predicted"` warm call violates Eq. (6);
     - its warm calls use more evaluations than `"power_law"`'s on `exp`;
     - a float32 chain produces NaN.

     dopri5 is printed with `[KNOWN]` when it fails, as `integral_accuracy.py` does
     for fehlberg2 (its error estimate, not the merge rule).
8. **Verification before merge.**
   - Run the full suite on a compute node, CPU and GPU, and mypy, ruff, numpydoc and
     `sphinx-build -W`.
   - Regenerate the snapshots, and check with a diff of `test_snapshots_data.json`
     that only `n_optimal_mesh` changed.
   - Rerun, against the real implementation:
     - `pred/accuracy_study2.py`: deterministic now, so the rule T counts must equal
       Evidence A unless the implementation differs on purpose;
     - `review/warm_bench_review.py`, for the padaquad split counts;
     - `pred/rounding_floor_check.py`, `pred/speed_merge.py` and
       `pred/memory_chunked.py`.
   - Rerun pode's three regression problems over 5 seeds.
9. **Issue file** (folded into step 2).

## Decisions

Settled by the user on 2026-11-14.

1. **All three rules are options.** `merge_rule` takes `"predicted"` (rule T,
   default), `"power_law"` (rule 2) and `"sum"` (rule 1). The older rules stay so
   that earlier meshes can be reproduced; the documentation says what each does and
   that `"predicted"` is the one to use.
2. **dopri5 and fehlberg2.** Their error estimates are unreliable on coarse
   subintervals, so some merges can lose accuracy under every rule. Rule T reduces it
   (17 against 364 pairs for dopri5, 67 against 11,655 for fehlberg2; worst 3.72 and
   4.35 times the tolerance) but not to zero. Accepted.
3. **A verification script** is required (step 7).
4. **The cost issue** is written in step 2. The review showed that the cause is
   `_rec_remove`, not dopri5, and since rule T makes it more frequent, the fix is part
   of this work.

## Out of scope

- **Cold-call blind spots.** A feature between nodes can fool the error estimate on
  the starting mesh (gk7 on a step: 585,870 times the tolerance). This is a property
  of error estimation (`gander2000`, Sec. 1: no program is foolproof), not of merging.
- **The rounding floor's definition.** It uses the signed contribution, which scipy
  does not (`_round_floor` docstring). It was tested as a cause here and is not one.
- **Merging more than one level per call.** Merging stays one level per call, as
  today.

## Confidence

- **Sure (measured, and checked by an adversarial review):**
  - the `_rec_remove` cost and its one-pass fix;
  - the conclusions of Evidence A, which now include the rounding regime and use
    50-digit references;
  - the corrected split counts;
  - the evaluation savings in Evidence B and D;
  - unchanged integrals in the snapshots;
  - the speed and memory of the chunked predictor.
- **Less sure:**
  - The integrand set is broad but not exhaustive, and the bump example shows what
    any stored-value rule misses.
  - Rule T's extra cost on small meshes is measured outside pode, and the caching
    that should remove most of it is not built yet.
  - The GPU path is unmeasured; by default the prediction runs on the CPU anyway.
  - Evidence A's random meshes use one seed.
- **Mine, not from the literature:**
  - interpolating the stored values to decide merges;
  - the halves-versus-whole comparison used for coarsening;
  - the cached per-ratio weight vectors;
  - Eq. (6) as the criterion for warm-start checks.

## Connections

- The halves-versus-whole comparison is the error test of adaptive Simpson
  quadrature (`gander2000`, Sec. 3); rule T uses it to decide when coarsening is
  safe rather than when refinement is needed.
- According to the review (its conjecture), published adaptive-quadrature codes such
  as QUADPACK and scipy's `quad_vec` only bisect. Coarsening from a power-law
  prediction is standard in ODE step-size control, which is the idea behind rule 2.
- The fast growth of the merged error on wide subintervals (2^8 to 2^18 here) is the
  pre-asymptotic, geometric convergence of Gauss and Clenshaw–Curtis rules on
  analytic integrands (`trefethen2008`).

## Files

Under `/pscratch/sd/k/khegazy/u028/`:

- **Prototype trees** (copies of `main` at `2b04ea1`):
  - `pq_pred/`: rule T, unchunked, with `PADAQUAD_MERGE_RULE` (T by default; C, G, S
    are the rejected variants);
  - `pq_pred2/`: rule T, chunked, with the one-pass `_rec_remove`;
  - `pq_main2/`: rule 2 with the one-pass `_rec_remove`;
  - `pq_main/`, `pq_122/`: `main` and 1.2.2 unchanged.
- **Pair study:** `pred/accuracy_study2.py` (logs `accuracy_study2_mp.log`,
  `accuracy_study2_halves.log`, `accuracy_study2_merged.log`) and
  `pred/diag_rounding_loss.py`. Superseded: `pred/accuracy_study.py`,
  `pred/safety_study.py`.
- **Warm-start chains:** `pred/warm_bench.py` (`warm2_rule{2,C,G,S,T}.json`), and the
  review's `review/warm_bench_review.py` and `warm_review_analyze.log`.
- **Rounding-level chains:** `pred/rounding_floor_check.py`.
- **Kink case:** `pred/kink_capture.py`, `pred/kink_mesh.txt`, `pred/diag_kink.py`.
- **Speed and memory:** `pred/rec_remove_check.py`, `pred/patch_rec_remove.py`,
  `pred/speed_merge.py`, `pred/memory_chunked.py`; older `pred/speed.py` and
  `pred/speed_t.py` (measured at a tolerance where nothing merges).
- **Other:** `pred/cases.py` (integrands and antiderivatives) and
  `pred/find_case2.py` (the Lorentzian case).
- **pode runs:** `runs/`, `logs/`, `drive.py`, `sweep_T.sh`.
- **Review material:** `review/`, including the opened sources in `review/sources/`.
