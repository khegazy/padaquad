# Plan: fewer evaluations spent on subintervals that are then split (`split-waste`)

Status: plan, reviewed against `b43da3d` (2026-11-26; section "Review"). Written
at commit `0590ccf` (branch `fix_conserve_memory`); the line numbers of sections
2 to 9 are those of `0590ccf`, and the functions they name were re-located at
`b43da3d` in the review. Every count of the original plan was measured on the CPU
of the login node (counts are exact there; no wall time was measured, see
section 7). Prototype and scripts: the last section.

## Review (2026-11-26, against `b43da3d`)

Six plans landed between `0590ccf` and `b43da3d` (`git log --oneline
0590ccf..b43da3d`): `endpoint-evaluations` (the Gauss-Kronrod methods have 7, 15,
21, 31 points per subinterval, none at its ends; `dopri5` 5), `startup-evaluations`
and `call-setup-cost` (no start-up calls of f), `turn-device-reads` (one read of
the device per mesh update, `_midpoint`, `_splittable`) and `record-order` (each
recorded subinterval carries its left mesh index, remapped through
`idxs_transfer`). Scripts and outputs of the review:
`$SCRATCH/padaquad_speedup_plans/split-waste/impl/` (`sweep.py`, `analyze.py`,
`refs.py`, `carrier.py`, `d7.py`, `sweep_tuned/`, `stage2/`).

**Re-measured on `b43da3d`** (CPU, the plan's sweep: 9 integrands × atol 1e-6,
1e-8, 1e-10 × seeds 0-2, `rtol=0`, `error_norm="max"`, with the implementation's
`split_rule="accelerate"` against `"halve"`; `sweep_tuned/analysis.txt`):

| Method | Points | Turns | Cells with more points | Errors worse / better |
|---|---|---|---|---|
| gk21 | 0.892 | 0.851 | 0 | 0 / 1 |
| gk15 | 0.858 | 0.851 | 0 | 0 / 0 |
| gk7 | 0.918 | 0.898 | 3 (all the jump, which halving accepts by accident; up to 3.53) | 0 / 3 |
| cc17 | 0.882 | 0.854 | 0 | 0 / 1 |
| cc33 | 0.867 | 0.854 | 1 (`peak_wide`, atol 1e-8, seed 0: 627 to 693, +10.5 %) | 0 / 0 |

These are the plan's `child:4` numbers to the third digit: dropping the
zero-weight end nodes changed the points per subinterval (gk21: 23 to 21) but
not which subintervals are evaluated, so the ratios did not move. Halving still
discards 46 % (`chirp`) and 50 % (`oscillatory`) of gk21's points (2.2's table
now reads 2,961 and 547,281 points). The saving for the default gk21 is about
11 % of the points summed over the sweep, all of it on `chirp`, `oscillatory`
and the Lorentzian; on 6 of the 9 integrands the median cell is unchanged.

**Errors and changes in the plan:**

1. **The insertion (3.1, "Where it changes").** The plan inserts `2**j - 1`
   points per split with `repeat_interleave`, `torch.lerp` and a
   `searchsorted` into the split positions: a second insertion path, a formula
   that can differ from `_midpoint` in the last bit (its D5), and points that
   `_splittable` never checked. Replaced: a split in four is two halvings in
   the same update. The existing insertion block runs a second time on both
   halves of the four-piece subintervals, so the four pieces are bit for bit
   those two successive halvings would make (`_midpoint` everywhere, D5 moot),
   `idxs_transfer` is the composition of the two transfers (the record's mesh
   indices and `mesh_grad` follow it unchanged), and the split counts get + 2
   without new code (D2 (a)).
2. **How the update sees the rule's inputs (3.1, step 2).** The plan adds a
   `split_rule` argument to `_adaptively_increase_mesh` and finds the batch's
   splits in `split_idxs` with `searchsorted`. The rule needs only the batch's
   error ratios: the update gets one new keyword `error_ratios` (None, the
   default, halves), passed only by Step 5 of `integrate` and only with
   `split_rule="accelerate"` and a norm-family `error_norm` (D4). The re-check's
   extra positions, Step 7 and `_get_optimal_mesh` pass none and halve. The rule
   is decided before the batch is sorted, so its groups (accepted, halved, in
   four) come out of the same stable sort, and the number in four joins the
   number accepted in the update's one read (`torch.stack(...).tolist()`). The
   plan's "one more wait per turn for `N_t_add`" (section 9) does not happen:
   since `turn-device-reads`, `len(split_idxs)` is known on the host.
3. **The guard.** The plan's rule did not check that the quarter points differ
   from their neighbours. `_splittable` gets `halvings=2`: it builds the points
   two halvings insert with `_midpoint` and requires every piece to have
   non-zero width, and requires `split count + 2 <= max_adaptive_splits`. A
   failing piece that can be halved but not quartered is halved.
4. **Tests of step 4.** "Every integrand of `integrand_dict` converges with the
   rule" tests nothing: those integrands need at most one level of refinement,
   where the rule never acts. Replaced by integrands where it acts (a
   Lorentzian, a jump, sin(2e4 t)) against their exact integrals, and by the
   guarantees of doubt 5 of the task (section 5).
5. **Carrier (3.2, stage 3) re-derived.** Counted by wrapping f
   (`carrier.py`): the points of f within 2 ULP of a node of a subinterval
   rejected in the previous pass, which a carrier could fill without calling
   f. Plan's 9 integrands, seed 0, `rtol=0`, halving:

   | atol | gk7, gk15, gk21, gk31 | cc5 | cc9 | cc17 | cc33 | cc65 | dopri5 | bosh3 | fehlberg2 | adaptive_heun | interpolatory3_variable |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | 1e-6 | 0 | 40.0 % | 22.2 % | 11.7 % | 6.0 % | 3.0 % | 19.9 % | 62.4 % | 64.9 % | 50.0 % | 66.6 % |
   | 1e-8 | 0 | 40.0 % | 22.2 % | 11.7 % | 6.0 % | 3.0 % | 20.0 % | 62.5 % | 66.6 % | 50.0 % | 66.7 % |

   Every match is exact (bit equal) except 45 and 64 of bosh3's. The
   Gauss-Kronrod methods gain nothing: their parent's centre node is an end of
   both halves, where they have no node. These shares count both halves'
   copies of the parent's centre value, so they are an upper bound if the
   `endpoint-evaluations` design (b) (one evaluation per shared mesh point) were
   also built. The plan's 6.1 % for cc17 counted a different quantity (its
   section 2.6 took the column "b + c" minus "b" of the other plan). Not
   implemented (see "Not done").
6. **D6.** The uniform `_evaluate_adaptive_nodes` and the abstract declaration
   were already deleted by `endpoint-evaluations` (`6201cff`); only the variable
   solver's method remains, with no caller and nothing to override. Its
   deletion is done now, as its own commit, instead of in stage 3.
7. **D7.** Re-checked on the current code over seeds 0-9 (`d7.py`) and written
   up as `docs/issues/accuracy-accidental-acceptance.md`.
8. **Smaller corrections.** Section 2.1 item 3: the cap is applied in
   `_splittable`. Section 2.1 item 7: one reuse method is left, not three
   definitions. Section 6: the grid harness used is
   `$SCRATCH/padaquad_speedup_plans/record-order/impl/bitgrid.py` (current
   API, CPU and CUDA), not `padaquad_methodclass_verify`. Section 9,
   `record-order`: the index remap needs no change (it reads the composed
   `idxs_transfer`).

Decisions taken by default (the brief's rule: the plan's recommendation): D1
(a), the opt-in option; D2 (a), halvings; D3, 1e4 fixed (module constant
`_FOUR_PIECES_RATIO`, condition split count >= 1 in the code); D4 (a),
`failure_fraction` halves; D5 moot; D6 delete now; D7 an issue document.

## 1. Summary

- **Slow:** a failing subinterval is halved and its values dropped; with `gk21`, 46 % (`chirp`) and 50 % (`oscillatory`) of the points of `f` are discarded this way, and turns = halving depth + 1 (measured, 2.2).
- **Change (stage 1, opt-in, D1):** a failing piece of a failed subinterval with error ratio ≥ 1e4 is split into 4 pieces instead of 2 (rule `child:4`, 3.1); all else stays halving.
- **Gain (prototype, 81 cells per method):** 8-14 % fewer points of `f` (gk21 11 %) and 10-15 % fewer turns; one cell in 405 cost more (+11 %); no error got worse.
- **Rejected:** `r^(1/k)` pieces does nothing for `gk21` (it helps `gk7`, `cc17`); a finer starting mesh costs 5-90× on easy integrands; raising the degree is a rewrite.
- **Later:** stage 2 measures on GPU and more integrands; stage 3 builds the carrier of a parent's values that `endpoint-evaluations` asks for (0 for Gauss-Kronrod, 6-34 % for others), closing `variable-sampling-reuse.md`.
- **Numbers:** unchanged bit for bit by default; with the rule on, the mesh, the integral (within tolerance), `max_split_depth` and the state's mesh change.
- **Ideal** (final mesh only) would save about 50 % on the hard integrands; no rule tried gets close. Details of the gains: 2.4.

## 2. Evidence

### 2.1 The code as it is now

All in `padaquad/quadrature/base.py`.

1. **Halving.** `_adaptively_increase_mesh` (1905-2145) splits every rejected
   subinterval (`remove_mask`) and every recorded subinterval that failed the
   re-check (`extra_split_idxs`). It inserts one mesh point per split at
   `0.5 * (left + right)` (2088-2089), sets both halves' trackers True, and gives
   both halves the parent's split count plus 1 (2098-2106). The rejected
   subintervals' values are dropped (`_keep_rows`, 2126-2135; the batch's arrays
   are deleted at 1465).
2. **Fresh nodes.** The next pass places every waiting subinterval's nodes with
   `_compute_nodes` (551-587, `torch.lerp` at the method's node fractions).
3. **Split statistics and the cap.** `max_adaptive_splits` accepts a subinterval
   whose split count reached the cap (2040-2044). After the loop,
   `n_start_steps_split` and `max_split_depth` are read off `split_counts`
   (1560-1563); `results.py:164-166` documents `max_split_depth` as "Largest
   number of times one starting subinterval was halved".
4. **Starting mesh.** `N_init_steps=13` (717) gives s = 3 segments of 4 mesh
   points, 11 subintervals (`_setup_initial_mesh`, 3508-3616).
5. **Error ratio.** In absolute mode (default) a subinterval's ratio is its
   reduced error estimate divided by `max(atol, rtol * |I|)` (`_error_ratios_norm`,
   3813); with `error_norm="failure_fraction"` the accept quantity is the fraction
   of failing components (`_compute_error_ratios` docstring, 3779-3781), not a
   magnitude.
6. **Registered `error_exponent` k** (`padaquad/methods/`): gk7 7, gk15 15,
   gk21 21, gk31 31 (`gauss_kronrod.py:192, 228, 276, 343`); cc5 5, cc9 7,
   cc17 11, cc33 19, cc65 35 (`clenshaw_curtis.py:153-181`); adaptive_heun 2,
   fehlberg2 2, bosh3 3, dopri5 5 (`runge_kutta.py`); interpolatory3_variable 3
   (`interpolatory.py:86`).
7. **The variable solver's reuse method** `VariableAdaptiveQuadrature._evaluate_adaptive_nodes`
   (`padaquad/quadrature/variable.py:75-160`) still has no caller in `padaquad/`
   (`git grep "_evaluate_adaptive_nodes("`: only the three definitions and
   `tests/test_evaluate_and_merge.py`). Its `nodes_new` is still allocated with
   last dimension D (141-143). Section 2.6 checks the rest of
   `variable-sampling-reuse.md`.

### 2.2 Where the discarded evaluations come from (halving, measured)

Setting of the brief's baseline on the CPU: `gk21`, `atol=1e-8`, `rtol=0`,
`error_norm="max"`, float64, `take_gradient=False`, `max_batch=1_000_000`,
`torch.manual_seed(0)`, interval [0, 1]. The CPU's random starting mesh differs
from the GPU's, so `chirp` and `oscillatory` differ from the brief's GPU table
(3,290 and 631,650 points); the CPU counts equal `baseline_cpu.txt` (which adds
the one shape-check point). "Final mesh" (the ideal) is the points of the
returned subintervals only. "Generation" is the split count. Script
`measure_waste.py`, output `out_gk21_halve.txt`.

| Integrand | Points of f | Final mesh | Discarded | Turns | Depth |
|---|---|---|---|---|---|
| damped_sine | 253 | 253 | 0 % | 1 | 0 |
| many_outputs | 345 | 299 | 13 % | 2 | 1 |
| chirp | 3,243 | 1,748 | 46 % | 11 | 10 |
| oscillatory | 599,403 | 299,828 | 50 % | 13 | 12 |
| peak_wide (Gaussian, width 1e-3, at 0.3183) | 529 | 391 | 26 % | 6 | 5 |
| kink (\|t − 1/3\|) | 575 | 414 | 28 % | 8 | 7 |

Per generation (subintervals evaluated / split; a split one discards 23 points):

- `oscillatory`: generations 0-6 split everything (11, 22, 44, 88, 176, 352,
  704); then 1,408/1,359, 2,718/2,043, 4,086/3,359, 6,718/2,891, 5,782/1,976,
  3,952/0. The median log10 error ratio of the split ones stays between 5.7 and
  3.0 through generations 0-10: the estimate barely falls while the oscillation
  is not resolved.
- `chirp`: 11/2, 4/3, 6/5, 10/6, 12/8, 16/8, 16/10, 20/11, 22/10, 20/2, 4/0
  (a refinement front that moves toward t = 0; median log10 ratio of the split
  ones 6.3 falling to 2.6).
- `peak_wide` and `kink`: one or two subintervals split per generation (an
  isolated feature: each split leaves one or two failing pieces).

The **observed exponent** of a split, `k_obs = log2(r_parent / r_child)` (equal
to k = 21 if the estimate scaled as h^21), measured over all parent-child pairs:

| log10 r_parent | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| oscillatory, median k_obs (n) | 19.2 (4,404) | 18.7 (4,142) | 4.6 (3,296) | 5.3 (10,966) | 5.1 (3,048) | 2.2 (164) | 2.4 (30) |
| chirp, median k_obs (n) | 19.5 (22) | 17.7 (14) | 10.5 (24) | 4.8 (38) | 7.0 (24) | 6.0 (6) | 30.7 (2) |

The spread in each column is wide (oscillatory, log10 r_parent = 3: from −6.1
to 27.3). So the theoretical exponent holds only when the parent's ratio is
below about 100; above it the estimate is in the pre-asymptotic regime, as the
`_merge_inputs` docstring says for merges (2856-2865). Gonnet 2012 (ACM
Computing Surveys 44(4), arXiv:1003.4629v2, read on ar5iv) reviews algorithms
that test for this before trusting an estimate (Berntsen and Espelid's
"non-asymptotic" test, Sec. 2.3; Rowland and Varol's ratio test, Sec. 3.2) and
notes that a difference of two rules can be "accidentally small" (Sec. 2.5).

**When does a split into 4 cost nothing extra?** A split into 4 equals halving
and then halving both halves; it wastes evaluations only when at least one half
would have been accepted. From the halving tree I counted, for every split
subinterval, s = the number of levels below it in which every descendant was
split (s ≥ 2: a 4-way split would have been free):

| log10 r | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| oscillatory, fraction with s ≥ 2 (n) | 0.01 (2,202) | 0.10 (2,071) | 0.66 (1,648) | 0.66 (5,483) | 0.82 (1,524) | 1.00 (82) | 1.00 (15) |
| chirp, fraction with s ≥ 2 (n) | 0.00 (11) | 0.00 (7) | 0.50 (12) | 0.68 (19) | 0.67 (12) | 0.33 (3) | 0.00 (1) |

For `peak_wide` and `kink`, s = 1 for every split (an isolated feature). This
table is the reason for a cut at r ≥ 1e3 or higher and for limiting the split to 4
pieces (8 pieces overshoots, measured in 2.4).

**Accuracy observation (not this plan's scope, reported).** The `oscillatory`
error of the halving run, 2.19e-5 against the reference, comes entirely from 49
subintervals accepted in generation 7 (true error 2.19e-5 summed; generation 8:
2.2e-9; later ones about 1e-13). They are about 7e-4 wide, about 22 periods of
sin(2e5 t) on 23 nodes: under-resolved, accepted because the two rules happened
to agree. In the sweep, halving also missed whole features: the Gaussian peak of
width 1e-4 with `gk21` (all 11 starting subintervals have ratio below 1e-150;
the integral, 2.5e-4, is lost), the peak of width 1e-3 with `gk15`, `cc17` and
`gk7` (error 2.5e-3, the whole peak), and the jump with `gk7` (error 1.9e-4).

### 2.3 What the theoretical exponent rule would do (measured)

`r^(1/k)` pieces needs r ≥ 2^k to give more than 2 pieces: 2.1e6 for `gk21`,
2,048 for `cc17`, 128 for `gk7`. Prototype rule `theory`: 2^j pieces with
j = ceil(log2(r) / k), at most 4 pieces.

- `gk21`: identical counts on every integrand except `oscillatory` (599,403 to
  599,265 points, 13 to 12 turns).
- `gk7` (acts at r ≥ 128) and `cc17` (r ≥ 2,048), 81 cells each
  (`theory_*.csv`): points 0.804 and 0.837 of halving summed, turns 0.676 and
  0.761; one cell above halving (`gk7`, kink, +46 %).

The coordinator's doubt is confirmed for the default method: the theoretical
rule changes nothing for `gk21`, and its premise (h^k scaling) is false at the
ratios where it would act (table of k_obs above). For low-degree methods it
works about as well as the `child` rules, through what is in effect a cut at
r ≥ 2^k; more than 4 pieces was not tried with it (8 pieces overshoots with
every rule tried, 2.4).

### 2.4 Prototype rules (measured)

All rules split into 2^j equal pieces (j = 1 is halving, at most j = 2 unless
noted) and give the pieces the split count of the parent plus j, so the split
count keeps meaning "halvings": width = starting width / 2^count. Rules:

- `ratio:c`: 4 pieces when log10 r ≥ c.
- `hist:c`: 4 pieces when log10 r ≥ c and the observed exponent from the parent,
  `log2(r_parent / r) / j_parent`, is below k / 2 (stores the parent's ratio per
  mesh point).
- `child:c`: 4 pieces when log10 r ≥ c and the subinterval is itself a piece of
  a split (split count ≥ 1). No stored history: it uses only the fact that its
  parent failed, which `split_counts` already records.

Seed 0, `gk21`, `atol=1e-8` (points of f / turns):

| Integrand | halve | theory | ratio:2 | hist:2 | hist:3 | ratio:2, 8 pieces | hist:2, 8 pieces |
|---|---|---|---|---|---|---|---|
| damped_sine | 253 / 1 | 253 / 1 | 253 / 1 | 253 / 1 | 253 / 1 | 253 / 1 | 253 / 1 |
| many_outputs | 345 / 2 | 345 / 2 | 437 / 2 | 345 / 2 | 345 / 2 | 621 / 2 | 345 / 2 |
| chirp | 3,243 / 11 | 3,243 / 11 | 3,013 / 6 | 2,737 / 7 | 2,829 / 8 | 3,243 / 5 | 3,565 / 5 |
| oscillatory | 599,403 / 13 | 599,265 / 12 | 504,919 / 8 | 460,391 / 9 | 491,855 / 9 | 576,035 / 6 | 588,731 / 7 |
| peak_wide | 529 / 6 | 529 / 6 | 529 / 4 | 483 / 4 | 483 / 4 | 621 / 3 | 575 / 4 |
| kink | 575 / 8 | 575 / 8 | 575 / 7 | 575 / 8 | 575 / 8 | 621 / 6 | 575 / 8 |

8 pieces overshoots (more points than halving in most rows): rejected.

**Sweep** (`sweep.py`, `analyze.py`): 9 integrands (the six above plus
`sqrt(t)`, a jump at 0.3183 and the Lorentzian 1/(1 + 1e6 (t − 0.4)^2)) × atol
1e-6, 1e-8, 1e-10 × seeds 0, 1, 2 = 81 cells per method, `rtol=0`. Columns:
points summed over the 81 cells relative to halving; turns likewise; cells with
more points than halving; the largest ratio of one cell. An error is counted
"worse" when it exceeds max(10 × halving's error, 10 × atol).

| Method | Rule | Points | Turns | Cells with more points | Worst cell | Errors worse / better |
|---|---|---|---|---|---|---|
| gk21 | hist:2 | 0.945 | 0.727 | 9 | 1.11 | 0 / 1 |
| gk21 | hist:3 | 0.894 | 0.795 | 3 | 1.11 | 0 / 0 |
| gk21 | ratio:2 | 0.905 | 0.679 | 8 | 1.45 | 0 / 2 |
| gk15 | hist:2 | 0.860 | 0.735 | 2 | 1.07 | 0 / 1 |
| gk15 | hist:3 | 0.835 | 0.799 | 1 | 1.07 | 0 / 0 |
| gk15 | ratio:2 | 0.897 | 0.680 | 6 | 1.27 | 0 / 3 |
| gk7 | hist:2 | 0.832 | 0.815 | 5 | 3.53 (*) | 0 / 5 |
| gk7 | hist:3 | 0.848 | 0.872 | 4 | 3.53 (*) | 0 / 4 |
| gk7 | ratio:2 | 0.813 | 0.668 | 1 | 1.46 | 0 / 0 |
| cc17 | hist:2 | 0.821 | 0.743 | 1 | 1.10 | 1 / 2 |
| cc17 | hist:3 | 0.812 | 0.805 | 0 | 0.99 | 0 / 1 |
| cc17 | ratio:2 | 0.877 | 0.673 | 5 | 1.27 | 1 / 3 |
| cc33 | hist:2 | 0.842 | 0.736 | 7 | 1.11 | 0 / 0 |
| cc33 | hist:3 | 0.834 | 0.790 | 2 | 1.22 | 0 / 1 |
| cc33 | ratio:2 | 0.879 | 0.688 | 8 | 1.35 | 0 / 1 |
| gk21 | child:3 | 0.894 | 0.795 | 3 | 1.11 | 0 / 0 |
| gk21 | child:4 | 0.892 | 0.851 | 0 | 1.00 | 0 / 1 |
| gk15 | child:3 | 0.835 | 0.793 | 3 | 1.14 | 0 / 0 |
| gk15 | child:4 | 0.858 | 0.851 | 0 | 1.00 | 0 / 0 |
| gk7 | child:3 | 0.836 | 0.845 | 4 (*) | 3.53 (*) | 0 / 4 |
| gk7 | child:4 | 0.918 | 0.898 | 3 (*) | 3.53 (*) | 0 / 3 |
| cc17 | child:3 | 0.813 | 0.793 | 0 | 1.00 | 0 / 1 |
| cc17 | child:4 | 0.882 | 0.854 | 0 | 1.00 | 0 / 1 |
| cc33 | child:3 | 0.834 | 0.790 | 2 | 1.22 (**) | 0 / 1 |
| cc33 | child:4 | 0.867 | 0.854 | 1 | 1.11 | 0 / 0 |

(*) `gk7` on the jump: halving accepted the jump by accident (seed 0: error
1.9e-4 with 171 points; seed 1, atol 1e-10: 1.4e-7); the rules found it (error
1.2e-9 at atol 1e-8). Every `gk7` cell of the `child` rules above halving is one
of these. Not an overshoot. (**) `cc33` `oscillatory`, atol 1e-6, seed 1:
halving's error 8.5e-4 against the rule's 1.2e-5. The real overshoots of
`child:3` are the Gaussian peak (+46 to +66 points: two subintervals) and
`many_outputs` with `gk15` (+68 points, +14 %).

Per integrand (median over the 9 cells of a method): `damped_sine`,
`many_outputs`, `sqrt` and `kink` are unchanged for every `hist` rule; `chirp`
0.82-0.91, `oscillatory` 0.79-0.99, `lorentz` 0.92-1.00, `peak_wide` 1.00
(between 0.88 and 1.11). Turns fall on every integrand that needs more than one
level. The full per-integrand table is `analysis_4methods.txt`.

**Relative tolerance** (`rtol=1e-8`, `atol=1e-14`, `rtol.csv`): `gk21` hist:3
points 0.811, turns 0.750, one cell above halving; `cc17` hist:3 0.833 / 0.768,
none above.

**Where the error ratio cannot be used.** With
`error_norm="failure_fraction"` the accept quantity is a fraction in [0, 1]
(2.1 item 5), so a cut at 1e4 never triggers: the rule falls back to halving
there unless the per-component ratios are used (D4).

### 2.5 A finer starting mesh (direction 2, measured)

Halving, median over seeds 0-2, `atol=1e-8` (`ninit.csv`; points / turns):

| Integrand | N_init 13 | 50 | 200 | 1000 |
|---|---|---|---|---|
| gk21 damped_sine | 253 / 1 | 1,265 / 1 | 4,807 / 1 | 22,793 / 1 |
| gk21 chirp | 3,243 / 12 | 4,071 / 12 | 7,475 / 9 | 25,093 / 7 |
| gk21 oscillatory | 599,403 / 13 | 592,227 / 11 | 567,525 / 9 | 558,325 / 7 |
| gk21 step | 1,173 / 21 | 2,001 / 17 | 5,497 / 16 | 23,345 / 13 |
| cc17 oscillatory | 776,135 / 14 | 786,267 / 12 | 772,837 / 10 | 762,127 / 8 |

A finer default multiplies the cost of every easy integrand (damped_sine: 5× at
50, 90× at 1000) and saves at most 7 % on the hardest one: rejected as a default.
Before the first pass nothing is known about `f`; after it, every subinterval's
error ratio is known, and using it is exactly the split rule of 2.4 applied to
generation 0, which the measurements show to be the risky generation (the
`ratio` rules overshoot on `many_outputs` there; `child` and `hist` skip it).
A warm start (`state=`) already removes the waste on repeated calls; what
remains for a first call is the waste measured in 2.2, which only a split rule
reduces. The two do not add up: `hist:3` with N_init 1000 gives 497,283 points
on `oscillatory` (`gk21`), against 491,855 with N_init 13.

### 2.6 Reuse of the parent's values (direction 3, computed and checked)

**Which nodes of a parent coincide with nodes of its halves** (`coincide.py`,
fractions c of the parent against 0.5 c and 0.5 + 0.5 c of the halves, float64,
tolerance 1e-12):

- Every Gauss-Kronrod and Clenshaw-Curtis method (gk7, gk15, gk21, gk31, cc5,
  cc9, cc17, cc33, cc65), fehlberg2 and interpolatory3_variable: fractions 0,
  1/2 and 1 only. The smallest gap between any other parent node and a node of a
  half is 2.8e-4 (gk21) to 7.3e-2 (cc5): no interior node coincides, as the
  coordinator expected.
- bosh3: 0, 1/2, 3/4 and 1 (its 3/4 node is the right half's node 1/2).
- dopri5: 0 and 1; adaptive_heun: 0 and 1.
- With 4 pieces the same three fractions coincide (a Clenshaw-Curtis node sits
  at 1/4 only if n is divisible by 3, and every registered n is a power of 2).

**What reuse is worth.** For Gauss-Kronrod the three coinciding nodes are the two
zero-weight padded ends and the centre node, which becomes a zero-weight end of
both halves (`w.py`: b = b_error = 0 at fractions 0 and 1 for gk7 and gk21).
Once `endpoint-evaluations` stops evaluating zero-weight nodes (its design a),
reuse saves nothing for Gauss-Kronrod. For methods with weight at the ends the
`endpoint-evaluations` plan counted it (its section 2.3, column "b + c" minus
"b"): `cc17` `oscillatory` 47,293 of 776,135 points (6.1 %), `cc5` 21.4 %,
`interpolatory3_variable` `chirp` 34.3 %, `adaptive_heun` 25 %.

**`variable-sampling-reuse.md` against the current code.**

- Still true: no caller (2.1 item 7); the D-for-T allocation; the tracked
  variables are not returned; the split point is the middle node.
- Stale: points 2 and 9 mention `same_f` warm starts (removed; a warm start is
  `state=`), and the list of places to change predates `trade_speed_for_memory`,
  `record_evaluations` and the in-place pass array.
- **Its premise no longer gives anything beyond shared end points.** For both
  registered variable methods the parent's nodes are exactly fractions 0, 1/2, 1
  (interpolatory3_variable) or 0, 1 (adaptive_heun), which are nodes of the
  halves at their standard fractions anyway. The reuse method's new nodes,
  midpoints between old nodes, are the halves' standard nodes. So reconnecting it
  costs C − 1 new points per split, the same as a uniform split that reuses the
  three shared points. One mechanism for every method (stage 3) replaces it, and
  the two `_evaluate_adaptive_nodes` methods, their abstract declaration and
  their tests can be deleted.
- **Fit with this branch.** A pass's values live in one [K, C, D] array
  (CLAUDE.md, "A pass's values are held once"), filled by calls of `f` that cover
  whole consecutive rows. Reuse needs the parent's rows kept past the pass
  (today deleted at 1465), and the calls to skip the reused nodes, so rows are no
  longer "C consecutive points of `f`'s output": the `endpoint-evaluations` plan
  costed that change for its design (b) (60 to 100 lines in
  `_evaluate_f_on_split_nodes`, the one-call no-copy path lost). With
  `trade_speed_for_memory` or `record_evaluations=False` the kept values must stay
  on the solver's device (at most 3 D values per split). With
  `take_gradient=True` reuse is not possible without keeping the parent's graph
  (`variable-sampling-reuse.md` point 4); the carrier runs only with
  `take_gradient=False`.

### 2.7 Nested and doubly adaptive refinement (direction 4, literature)

- Gonnet 2010, "Increasing the reliability of adaptive quadrature using explicit
  interpolants", ACM TOMS 37, art. 26 (arXiv:1006.3962, read on ar5iv):
  Algorithm 3 is doubly adaptive. It raises the degree of a subinterval's
  Clenshaw-Curtis rule through nested node sets (degrees 4, 8, 16, 32, so 5, 9,
  17, 33 points; the nodes "can be re-used when increasing the degree") and
  bisects when the highest degree is reached or the interpolation coefficients
  change by more than a fraction 0.1 between degrees. Both of its algorithms
  always bisect (two pieces), never more, and the halves are evaluated at fresh
  nodes; the parent's interpolant is used only for the halves' error estimates.
  Its Tables I and II compare evaluation counts with QUADPACK's DQAGS, Matlab's
  quadl and Espelid's da2glob: the new algorithms are "more reliable, albeit in
  some cases less efficient".
- Gonnet 2012 (`gonnet2012`), read on ar5iv: trisection (Sec. 2.1, an early
  recursive integrator; McKeeman 1962 as far as the stripped citations show),
  Oliver 1972 (Sec. 2.3: doubles the Clenshaw-Curtis degree unless the
  coefficients decay too slowly, Eq. (17)), nested "RMS" rule pairs that allow
  evaluations "to be re-used after bisection" and a choice between bisection and
  higher degree from the ratio of successive error estimates (Sec. 2.4), de Boor's
  CADRE (Sec. 3.1, the first doubly adaptive algorithm, tests ratios of
  successive differences against their expected values) and Rowland and Varol
  (Sec. 3.2, Eq. (36): a ratio of successive differences checked against 2^(2q)
  before the asymptotic error model is trusted). The `hist` rule is this last
  idea applied to the split decision (my reading; none of these papers splits
  into 4 pieces on such a test).
- Espelid 2007, "Algorithm 868: Globally doubly adaptive quadrature — reliable
  Matlab codes", ACM TOMS 33(3), art. 21: **not opened** (ACM returned 403; only
  a database summary was reachable). Not relied on.
- **Checked here:** `cc5 ⊂ cc9 ⊂ cc17 ⊂ cc33 ⊂ cc65` as node sets (`nested.py`);
  no Gauss-Kronrod method is a subset of another.
- **Fit with PadaQuad.** Raising the degree per subinterval means a different C
  per subinterval. Every array of a pass, the record, the weights, the merge
  prediction and `_RK_integral` assume one C ([N, C, D]). It is a rewrite, not a
  change: rejected for now. Within one method, the user already chooses the
  degree; the counts of 2.2 show where a higher degree pays (`cc33` evaluates
  580,899 points on `oscillatory` against `cc17`'s 776,135).

## 3. The design

### 3.1 Stage 1: split a failing piece of a failed subinterval into 4

The rule (prototype `child:4`; constants fixed in the code, D3):

1. A rejected subinterval of the batch with split count ≥ 1 and error ratio
   r ≥ 1e4 is split into 4 equal pieces; every other split is a halving
   (including recorded subintervals that fail the re-check, the splits of Step 7,
   and those of `_get_optimal_mesh`).
2. With `max_adaptive_splits`, the pieces are limited so that the split count
   never exceeds the cap (4 pieces only when count + 2 ≤ cap).
3. Each of the 4 pieces gets split count parent + 2.
4. With `error_norm="failure_fraction"`, halving (D4).

Why `child` rather than `hist`: `child:3` gives the same totals as `hist:3`
for `gk21` (0.894 / 0.795, the same 3 cells) and nearly the same for the other
methods, so the stored parent ratio adds nothing measurable. Why the cut 1e4
(`child:4`) rather than 1e3: it keeps most of the saving in points (8-14 %
against 11-19 %) and has one real overshoot cell in 405 against 7. Why no
stored history: `split_counts` is already carried per mesh point; `hist` needs
one more per-mesh-point tensor carried through `_adaptively_increase_mesh`
(2098-2108's transfer, a second time) and returned from it.

**Where it changes** (corrected in the review; `padaquad/quadrature/base.py` at
`b43da3d`):

- `_adaptively_increase_mesh` gets the keyword `error_ratios` (the batch's
  ratios; None, the default, halves). Before the batch is sorted, each of its
  subintervals gets a number of halvings: 0 (accepted, as now), 1, or 2 when it
  fails, `error_ratios >= _FOUR_PIECES_RATIO`, its split count is at least 1 and
  `_splittable(..., halvings=2)` allows it. The stable sort of that number gives
  the accepted, halved and four-piece positions as slices, and the update's one
  read of the device returns two numbers (accepted, in four) instead of one.
- The existing insertion block (offsets, `idxs_transfer`, `_midpoint`, trackers,
  split counts + 1) runs once, or a second time on both halves of the four-piece
  subintervals (their left mesh points after the first halving, and the
  midpoints right after them). `idxs_transfer` returned is the composition of
  the two transfers. The four pieces are bit for bit those of two successive
  halvings, and their split counts are parent + 2 (D2 (a)).
- `_splittable` gets `halvings` (default 1): the points `halvings` successive
  halvings insert (`_midpoint`), every piece of non-zero width, and split count
  + `halvings` at most the cap.
- `integrate` passes `error_ratios` at Step 5 only with
  `split_rule == "accelerate"` and a norm-family `error_norm` (D4).

Least new code: no new function or state; one keyword and one loop around the
existing insertion. The re-check, Step 7 and `_get_optimal_mesh` call the same
function without `error_ratios` and halve.

### 3.2 Stage 3: one carrier of a parent's values to its pieces

Built only after stage 1 is measured, after `record-order` and after
`endpoint-evaluations` design (a). Design (owned here, as that plan asks):

- In `_adaptively_increase_mesh`, for each split subinterval of the batch, keep
  its values at the node fractions that coincide bit for bit with a piece's node
  (computed once per method from the node fractions: 0, 1/2, 1 for most; 3/4 too
  for bosh3), in a small per-pass store keyed by the pieces' mesh positions.
  Recorded subintervals that fail the re-check hand theirs from the record.
- In `_evaluate_f_on_split_nodes`, skip those nodes in the flat list given to
  `_evaluate_f_in_batches` and scatter the kept values into the pass's array;
  the same machinery serves the `endpoint-evaluations` design (b) (a mesh point
  shared by two waiting subintervals is evaluated once).
- Only with `take_gradient=False`; tracked variables travel with the values.

Expected: 0 for Gauss-Kronrod after `endpoint-evaluations` (a); 6.1 % for `cc17`
`oscillatory`; 21-34 % for `cc5` and the variable and low-order methods (counts
of that plan). This plan gives the design and the acceptance numbers; its
implementation steps are sketched in section 5, stage 3, and must be planned in
detail together with `endpoint-evaluations` (b).

### 3.3 Rejected alternatives

- **`r^(1/k)` pieces with the registered exponent**: no effect for `gk21`; where
  it acts the h^k premise is false (2.3).
- **More than 4 pieces**: overshoots (2.4).
- **`ratio:c` without the "piece of a failed subinterval" condition**: acts on
  generation 0, where one halving often suffices (`many_outputs` +27 % with
  `gk21`, `damped_sine` +27 % with `cc17`).
- **A finer default starting mesh**: 5-90× the points on easy integrands (2.5).
- **Reconnecting `_evaluate_adaptive_nodes`**: saves nothing that the stage 3
  carrier does not save for every method (2.6).
- **Doubly adaptive (raise the degree before splitting)**: needs a C per
  subinterval; a rewrite (2.7).
- **Predicting the halves' error from the parent's interpolant** (as the merge
  prediction does for merges): my idea, not prototyped. In the asymptotic regime
  it predicts what `r / 2^k` already predicts; in the pre-asymptotic regime the
  parent's interpolant is not accurate, so it predicts nothing.

## 4. Decisions for the user

- **D1. Ship the 4-piece rule, and how.** (a) A constructor and `integrate`
  option, `split_rule="halve"` (default) or `"accelerate"`; every current number
  stays bit for bit by default. (b) Make it the default after stage 2: changes
  every result that needs more than one level of refinement (within tolerance),
  saves 8-14 % of the points of the sweep and 10-15 % of the turns, but costs
  up to 11 % more points in a single measured cell. (c) Do not ship it.
  **Recommended: (a) now, (b) decided on the stage 2 measurements.** The rule
  "never trade extra evaluations of f for speed" argues against (b) unless the
  wider benchmark shows no cell above halving by more than a few percent.
- **D2. What the split count counts.** (a) Halvings: a 4-way split adds 2, so
  `max_split_depth`, `max_adaptive_splits` and width = W / 2^count keep their
  meaning and their documentation. (b) Generations (adds 1): matches turns but
  changes the meaning of three documented quantities. **Recommended: (a).**
- **D3. Constants.** Cut 1e4 (`child:4`: fewest overshoots) or 1e3 (`child:3`:
  3-7 percentage points more saving, up to 3 overshoot cells per method), 4
  pieces, condition "split count ≥ 1"; fixed in the code (one module constant
  each, documented) or exposed. **Recommended: 1e4, fixed**; only these values
  were measured.
- **D4. `error_norm="failure_fraction"`.** (a) Halving always. (b) Use the
  largest finite per-component ratio as r. **Recommended: (a)** until (b) is
  measured.
- **D5. The midpoint formula.** Keep `0.5 * (l + r)` for halving (bit for bit) or
  use `torch.lerp` for every piece (one formula, possibly one ULP different
  midpoints; needed anyway by stage 3's bit-equality of shared nodes).
  **Recommended:** keep today's formula in stage 1; decide with
  `endpoint-evaluations` in stage 3.
- **D6. `variable-sampling-reuse.md`.** **Recommended:** close it as superseded
  by the stage 3 carrier, and delete `_evaluate_adaptive_nodes` (both solvers and
  the abstract method) with its tests in `tests/test_evaluate_and_merge.py`.
  Alternative: keep the issue open for a variable method with non-standard node
  positions, which none of the registered methods has.
- **D7. Report the accuracy observation** (2.2: accidental acceptance, missed
  peaks) as its own issue. Recommended: yes; it is outside this plan.

## 5. Implementation plan

### Stage 1 (opt-in rule; corrected in the review)

1. **Option** (its own commit, with the rule). `padaquad/quadrature/base.py`:
   `AdaptiveQuadrature.__init__` and `integrate` take `split_rule` (constructor
   default `"halve"`, stored as `init_split_rule`; the call's value overrides it
   as for `max_adaptive_splits`), validated by `_check_split_rule` against
   `_VALID_SPLIT_RULES` with `ValueError`. `padaquad/_frontend.py` needs no
   change (`integrate()` routes an argument of the call to the call). Test:
   `tests/test_split_statistics.py::test_invalid_split_rule_raises`.
2. **Rule and insertion** (3.1). `_FOUR_PIECES_RATIO = 1e4` (module constant),
   `_splittable(halvings=2)`, `_adaptively_increase_mesh(error_ratios=...)`, the
   Step 5 call. Tests, `tests/test_adaptive_steps.py::TestFourPieces`:
   - one update with subintervals accepted, halved (ratio 9e3; split count 0) and
     in four (ratio 1e5 and 1e4, split count >= 1) equals, bit for bit, two
     halving updates (mesh, trackers, split counts; `idxs_transfer` the
     composition);
   - without `error_ratios`, halving whatever the ratio;
   - the cap: one halving below it, halving;
   - `extra_split_idxs` halved in the same update as a four-way split;
   - a subinterval two ULPs wide (float64 and float32) is halved, not split in
     four;
   - `tests/test_base_solver.py::TestHostWaits::test_mesh_update_reads_once`
     with four-way splits: still one read.
3. **Halving unchanged.** With the default, `tests/test_snapshots.py` passes
   without regeneration and the bit grid (section 6) shows 0 differences
   against `b43da3d`, CPU and CUDA.
4. **Integration tests.** `tests/test_split_statistics.py`: exact integrals of
   a Lorentzian, a jump and sin(2e4 t) for gk21, gk15, gk7, cc17, both
   `take_gradient`; fewer points and passes on sin(2e4 t) and the same run bit
   for bit on a kink; the same integral and state mesh for `max_batch` 1e6 and
   7, with and without `trade_speed_for_memory`, uniform and variable; a
   second call from the state, absolute and cumulative, float64 and float32;
   `failure_fraction` the same as halving bit for bit; gradients of the bounds
   and of a parameter with both `take_gradient`.
   `tests/test_max_adaptive_splits.py::test_four_way_splits_respect_the_cap`
   (K 2^n subintervals, depth n, fewer passes).
   `tests/test_recheck_accepted_panels.py::test_recheck_with_four_way_splits`
   (re-check splits in the same update, no subinterval evaluated twice).
5. **Docs** (section 8), in the same commit series.

### Stage 2 (measure, then decide D1)

6. Run `sweep.py` with `RULES=halve,child:3,child:4` on the methods of
   section 2.4 plus gk31, cc65, dopri5, bosh3, and on the `integrand_dict` integrands and the
   performance-benchmark integrands (`examples/speed_test/`), cumulative mode
   (`use_absolute_error_ratio=False`) and `take_gradient=True` (counts only).
7. Time on a GPU node (section 7). Decide D1 (b) from 6 and 7.

### Stage 3 (carrier of parent values): not implemented

8. Not built (review, item 5: 0 for the Gauss-Kronrod methods, 6-12 % for
   cc33 and cc17; a second way of filling a subinterval's values in every
   evaluation path). See "Not done".
9. D6, done now and separately: delete `VariableAdaptiveQuadrature._evaluate_adaptive_nodes`
   and `tests/test_evaluate_and_merge.py`, and rewrite
   `docs/issues/variable-sampling-reuse.md` and the pages it lists.

## 6. Tests and how results are shown unchanged

- **Default unchanged, bit for bit.** `tests/test_snapshots.py` must pass without
  regeneration. The grid harness `/pscratch/sd/k/khegazy/padaquad_methodclass_verify/`
  must report 0 differences between `main`'s archive and the branch for the
  default rule. To use it, update its calls to the current API (no `same_f`,
  `integrate` returns `(result, state)`, keyword-only arguments after
  `mesh_final`); I did not open it, so the exact edits are to be listed when it
  is run.
- **Prototype check already done:** the prototype with `halve` reproduced every
  count of the unpatched code (2.2 table and the sweep), and 422 of 423 tests of
  `test_snapshots.py`, `test_adaptive_steps.py`, `test_max_adaptive_splits.py`,
  `test_split_statistics.py` passed; the one failure is the prototype's own
  device bug (its stored history lived on the CUDA device the login node exposes
  while `_get_optimal_mesh` ran on the CPU), not halving. With `child:4` on
  (after fixing that bug), `test_max_adaptive_splits.py`,
  `test_split_statistics.py`, `test_adaptive_steps.py`,
  `test_recheck_accepted_panels.py`, `test_snapshots.py` and
  `test_scipy_agreement.py`: 832 passed, 5 failed. The 5 are direct calls of
  `_adaptively_increase_mesh` in `test_adaptive_steps.py` with
  `split_counts=None`, which the prototype's rule indexes (TypeError); the
  implementation must treat `split_counts=None` as "halve" (step 2). The
  snapshot cells all passed unchanged: none of their integrands has a piece of a
  failed subinterval with error ratio ≥ 1e4, so even with the rule on they keep
  their bits (`tests_child4.txt`).
- **With `"accelerate"`:** numbers change; tests compare with exact values and
  with halving's counts (step 4), never with pinned bits.
- `mypy padaquad/` and `ruff` clean; numpydoc on the changed docstrings.

## 7. How to measure the gain

- **Counts (any node, CPU):** `sweep.py` (environment `RULES`, `METHODS`,
  `ATOLS`, `SEEDS`, `NINIT`, `RTOL`) and `analyze.py`; before = `halve`, after =
  `child:4` (prototype) or `split_rule="accelerate"` (implementation). Numbers:
  points of f, turns, cells above halving, errors.
- **Time (GPU node only):** `/pscratch/sd/k/khegazy/padaquad_speedup_profile/profile_baseline.py`
  with `split_rule="accelerate"` added to `run`, on `chirp` and `oscillatory`;
  numbers: wall median of 5, turns, host waits, launches. Estimate to check (mine,
  from the brief's 4.9 ms per turn on `chirp`): 3-4 fewer turns save 15-20 ms of
  68 ms; the per-turn cost will fall with `record-order` and `turn-device-reads`,
  and the gain in turns with it. The saving in points matters most when `f` is
  expensive, where it is the fraction measured.
- Not measured in this plan (allocation ended): any wall time.

## 8. Documentation to update

- `CLAUDE.md`: "Core algorithm" step 2 (split at the midpoint → into 2 or 4
  pieces with `split_rule`), the split statistics, and the test table
  (`test_adaptive_steps.py`).
- `README.md` lines 230 and 319 and `docs/user_guide/error_control.md:16`
  ("split at its midpoint"), `docs/glossary.md:73` ("split in half").
- `docs/user_guide/troubleshooting.md:12, 36` and `padaquad/results.py:164-166`:
  "halved" stays true under D2 (a); add that a 4-way split counts as two.
- `AdaptiveQuadrature.__init__` and `integrate` docstrings: the option.
- `CHANGELOG.md`: "Added: `split_rule`".
- Stage 3: the files listed in `variable-sampling-reuse.md` "Documentation to
  update".

## 9. Risks and interactions

- **Overshoot.** A 4-way split of a subinterval whose halves would have passed
  costs 2 subintervals more. Measured worst single cell +11 % for `child:4`,
  +7 to +22 % for `child:3` (2.4); the sum over the sweep is a saving. The
  constants were chosen on 9 integrands; stage 2 is there because they may not
  generalize.
- **Accuracy.** The rule changes which subintervals are accepted. No cell got
  worse beyond the criterion for `child:3`, `child:4` or `hist:3`; several got
  better, by avoiding accidental acceptances (2.2). Convergence guarantees are unchanged:
  every accepted subinterval still passed its own test, the re-check still runs,
  `max_adaptive_splits` is respected, and no subinterval is evaluated twice (the
  pieces are new subintervals, evaluated once).
- **Batch-size independence.** The rule uses only a subinterval's own ratio and
  split count, so in absolute mode with `rtol=0` it does not depend on the batch.
  With `rtol > 0` or `take_gradient=True` the ratio already depends on the
  running integral and so on batching; the rule adds no new dependence.
- **`record-order`.** If it replaces the dictionary by mesh indices carried
  across splits, its index update must handle `2^j − 1` inserted points per
  split (the same `idx_offset` cumsum). I assume it keeps the insertion in
  `_adaptively_increase_mesh`.
- **`turn-device-reads`, `compiled-turn`.** Stage 1 adds a `searchsorted`, a
  `repeat_interleave` and a sum read to the host (`N_t_add`, one wait per turn,
  as `len(split_idxs)` already is). A turn captured with fixed-size buffers must
  allow up to 3 new points per split.
- **`endpoint-evaluations`.** Stage 3 is the carrier it asks for (its D5); the
  midpoint formula (D5 here) must be decided once for both.
- **`optimal-mesh-cost`.** `_get_optimal_mesh` still halves; a 4-way split
  leaves pairs of pieces that its merge step can merge one level per call.
- **`startup-evaluations`, `call-setup-cost`.** No overlap.

## 10. Confidence

- **Sure (measured, exact counts):** the waste per generation (2.2); the
  theoretical rule does nothing for `gk21` (2.3); the prototype counts and the
  sweep (2.4, 2.5); which node fractions coincide (2.6); the nesting of the
  Clenshaw-Curtis node sets.
- **Sure (read in the code):** the locations in 2.1; that
  `variable-sampling-reuse.md`'s statements on the reuse method still hold and
  which are stale.
- **Less sure:** that the constants (1e4, 4 pieces) generalize beyond the 9
  integrands and 5 methods measured; the time saved per turn on a GPU (estimated
  from the brief, not measured); the cost of stage 3 in code (estimated from the
  `endpoint-evaluations` plan's estimate for its design b).
- **Not checked:** Espelid 2007 (not opened); the stripped citation names of
  Gonnet 2012's ar5iv text (I name McKeeman only as the review's cross-references
  suggest); cumulative mode, `take_gradient=True`, T > 1 with the new rule.
- **My own ideas, not from a paper:** the `child` and `hist` rules, the
  "s ≥ 2" overshoot criterion, and the carrier design (shared with the
  `endpoint-evaluations` plan).

## Prototype and scripts

- Package copy: `$SCRATCH/padaquad_speedup_plans/split-waste/` (from HEAD),
  `padaquad/quadrature/base.py` patched by
  `$SCRATCH/padaquad_speedup_plans/split-waste_scripts/patch_proto.py` (rules read
  from `PADAQUAD_SPLIT_RULE`, `PADAQUAD_SPLIT_JMAX`) plus the `child:` rule.
- Scripts and outputs in `$SCRATCH/padaquad_speedup_plans/split-waste_scripts/`:
  `measure_waste.py` (per-generation picture, `out_gk21_halve.txt`),
  `sweep.py` (+ `references.json`: gk31, atol 1e-13), `analyze.py`,
  `analyze_ninit.py`, `coincide.py`, `nested.py`, `w.py`; results
  `sweep_*.csv`, `child_*.csv`, `theory_*.csv`, `ninit.csv`, `rtol.csv`, `tests_child4.txt`,
  `analysis_4methods.txt`.
