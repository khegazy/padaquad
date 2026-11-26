# Plan: stop evaluating the integrand where it adds no information (`endpoint-evaluations`)

Status: plan reviewed and corrected on 2026-11-25/26 (section 0), implemented on
branch `endpoint_evaluations` from `d11df47` (section 11). Written at commit
`0590ccf` (branch `fix_conserve_memory`); the line numbers of sections 2 to 9 are
of that commit, and section 0 gives the locations at `d11df47` where they moved.
Prototypes and measurements are in
`$SCRATCH/padaquad_speedup_plans/endpoint-evaluations/` (listed in section 2.6).

## 0. Review of 2026-11-25/26

Done before any code, against `d11df47` (the tip of `main`; it differs from
`0590ccf` in `quadrature/base.py` from `_insert_sorted_results` on, including a
rewritten `_flatten_output`). The user's decisions at the time of the review:
D1 (i) decided; D2 (i) by default (compact layout); D3 as recommended, but only
what is true; D4 implemented last, in its own commit, so it can be dropped; D5
out of scope; D6: `record-order` has not landed and will not land first, so the
fallback applies; D7 open (default: a diagnosis that changes no number).

### What was wrong or missing

1. **Step 0 cannot be met; the D6 fallback was under-specified.** The fallback
   ("a record field with each subinterval's left mesh point; six call sites
   read it") does not say where a batch's left mesh points come from after the
   accept-or-split decision. The final accept mask is not available to the loop:
   `_adaptively_increase_mesh` turns some splits into accepts at
   `max_adaptive_splits` (`base.py:2040-2043`) and returns only the filtered
   arrays, not the mask. So the left mesh points must travel through that filter
   with the other per-subinterval numbers. There are also more readers than six:
   `nodes[:, 0, :]` is read at `base.py:1252`, `1253`, `1767`, `4357`, `4358`,
   `4378`, `4414` and `4426`, and the 1-D order checks read `[:, 0, 0]` at `4382`
   and `4430`. Corrected step 1 below.
2. **The start-up calls of `f` still evaluate a bound.** Section 1 ("Also fixed")
   and D3 say that Gauss-Kronrod then never evaluates `f` at a mesh point. Not so:
   the shape check calls `f(mesh_init)` (`base.py:1118`), and without
   `max_batch` or `state`, `_initialize_max_batch` (`memory.py:1599`, called at
   `base.py:1173`) calls `f` on 10, 100 and 1,000 copies of `mesh_init`. Neither
   checks finiteness (read in the code; `_measure_call` and the shape check have
   no `isfinite`), so `1/sqrt(t)` still works, but an `f` that raises at the
   bound (for example one that validates its domain) fails, and the claim is
   false as stated. Corrected in step 3: the start-up calls use the first node of
   `[mesh_init, mesh_final]` (`_compute_nodes`), which is `mesh_init` bit for bit
   for every method with a node at fraction 0 (so nothing changes for them) and
   an interior point for Gauss-Kronrod. The probe is computed from the bounds, not
   from the starting mesh, because building the starting mesh first would change
   the random mesh for an `f` that draws random numbers.
3. **`_check_node_fractions` and `measure_degree`.** The plan required the ends
   only in `VariableMethodClass.__init__`, but `measure_degree` calls the same
   check on fractions a user passes for a variable method, and the variable
   weights rescale by the first and last node (`_base.py:645-646`). The end
   requirement must be part of the check for variable methods (an override that
   calls the general check). The general check: 1-D, at least one fraction, every
   fraction in [0, 1], non-decreasing.
4. **`_flatten_output` was rewritten on `main`** (one copy into a preallocated
   output, `base.py:4541-4566`). The condition is redone on that code: join shared
   ends only when the first fraction is 0 and the last is 1. A method with only
   one end among its nodes (allowed by the new check) shares no point with its
   neighbour, so its rows are joined by a plain reshape, like Gauss-Kronrod.
5. **Merge prediction: extrapolation is still needed.** Section 9 item 2 says the
   merged centre node is then extrapolated, and the doubt list asked whether no
   merged node lies beyond the outermost node. Checked: for two neighbours of
   equal width (rho = 1) the merged node at fraction 1/2 lands at x = 1, the
   shared mesh point, which is no longer a node of either neighbour, so the left
   neighbour's polynomial is evaluated 0.0022 of its width beyond its last
   Kronrod node (gk21: 1 - (1 + 0.99566)/2). The second (true) barycentric
   formula used there (Berrut and Trefethen 2004, SIAM Review 46(3), Eq. (4.2),
   key `berrut2004`, cited in the code) gives the interpolating polynomial at
   every x that is not a node, inside or outside the nodes, because the identity
   it divides by, Eq. (4.1), holds for every such x (derivation; the equations
   as recorded in the `checked` field of `berrut2004`). So no new case is needed;
   only the comment that says the shared point is a node of both neighbours is
   wrong afterwards. The prediction changes, and with it
   the state's mesh in a few cells (to be measured, step 4).
6. **`_merge_excess_nodes` and `_evaluate_adaptive_nodes`.** Confirmed: the
   merged nodes from `_merge_excess_nodes` are read only as `nodes_pruned[:, 0, :]`
   (`base.py:3292`); nothing else reads them. Both overrides and the abstract
   declaration go; `_prune_excess_mesh` drops the middle mesh point of each merged
   pair. `UniformAdaptiveQuadrature._evaluate_adaptive_nodes` (dead: no caller in
   `padaquad/`) takes a split midpoint from the first and last node, which is
   wrong for an unpadded method, so it is deleted with its abstract declaration
   and its tests; the variable solver's method (correct for variable methods,
   which keep both ends) stays for the `split-waste` plan.
7. **Bounds that require grad** were not discussed. The width must come from the
   same tensors the batch's nodes are built from: `mesh[idxs]` and
   `mesh[idxs + 1]` of the `mesh` the evaluation function receives, which with
   `take_gradient=True` is the detached copy `mesh_eval` whose gradient is
   carried to the mesh after each batch (`base.py:1209-1217`, `1467-1474`). The
   gradient with respect to a bound flows through the width and the nodes; it is
   tested against the closed form for both bounds, `take_gradient` True and False.
   The order in which autograd adds the width's and the nodes' contributions to
   a mesh point changes (the width no longer passes through `torch.lerp`), so a
   bound's gradient may change at rounding level even for methods with end
   nodes: measured in step 1 (section 11), not assumed.
8. **"The evaluation, batching, in-place pass array, out-of-memory and
   tracked-variable paths do not change at all."** Checked each: they work in
   units of `self.C` points per subinterval and never read node positions, so a
   smaller C needs no change there. Two edits in the evaluation functions remain:
   they pass the subintervals' mesh points to `_calculate_integral`, and the split
   path's list of widths per block is replaced by one difference of mesh points.
   The order check on the flattened nodes of `_evaluate_f_on_full_nodes`
   (`base.py:2394-2397`) stays meaningful: without shared ends, the nodes of
   consecutive subintervals are strictly increasing (the last Kronrod node of
   one is below the mesh point, the first of the next above it). Every use of
   `self.C` (`base.py:1177`, `1236`, `2306-2311`, `2319`, `2420-2421`, `2433`,
   `2442`, `2620`; `memory.py:1254`, `1816`) means "points `f` receives per
   subinterval", which is what C becomes.
9. **Tests and documents the plan did not list:** `tests/test_record_and_sort.py`,
   `tests/test_sorted_insert.py` (record built without the new field),
   `tests/test_adaptive_steps.py` (builds `MethodOutput`),
   `tests/test_evaluate_and_merge.py`, `tests/test_prune_and_optimal.py` (new
   signatures), `tests/test_evaluate_f_on_mesh.py:111`,
   `tests/test_trade_speed_for_memory.py` (spy on `_calculate_integral`);
   `padaquad/base.py` (the abstract `_calculate_integral`),
   `padaquad/quadrature/__init__.py` (names the abstract methods);
   `docs/development/adding_a_method.md` (says fractions must start at 0 and end
   at 1), `docs/glossary.md`, `docs/user_guide/concepts.md`,
   `examples/quadrature_basics.py` (counts evaluations from `result.y`).
10. **dopri5 (D4) arithmetic, checked.** In exact rationals,
    11/84 - 649/6300 - 1/60 = (825 - 649 - 105)/6300 = 71/6300; equivalently the
    embedded weight at fraction 1 is 649/6300 + 1/60 = 754/6300 and
    b - b_embedded = 825/6300 - 754/6300. The tableau was compared with the
    installed torchdiffeq 0.2.5 (`torchdiffeq/_impl/dopri5.py`, `c_sol` and
    `c_error`): identical to the current table. The plan's 71/6300 is right.
11. **D7 (divergent integral at a bound)** must restore an early diagnosis
    without changing any returned number; see step 5 below and section 11.

### What changed in this plan

- Section 5 is replaced by the corrected steps below (5'); sections 1 to 4 and 6
  to 10 are kept as written at `0590ccf`, with the corrections above taking
  precedence.

## 5'. Implementation plan as corrected (each step one commit)

1. **Subinterval geometry from the mesh** (changes no number, except possibly a
   bound's gradient at rounding level, item 7).
   - `_RK_integral(h, y, tableau_b, y0)` takes the widths [N, T] and returns
     `(integral, contributions)`.
   - `_calculate_integral(t, y, mesh_left, mesh_right, y0=None)` computes
     `h = mesh_right - mesh_left` and returns it with `mesh_left` in the
     `MethodOutput` (new field `mesh_left` [N, T], internal); `t` still feeds
     `method.weights` (variable methods). The abstract declaration in
     `padaquad/base.py` follows.
   - `_evaluate_f_on_full_nodes` and `_evaluate_f_on_split_nodes` pass the same
     `mesh[idxs]`, `mesh[idxs + 1]` they give `_compute_nodes`; the split path
     drops `width_parts`.
   - `_adaptively_increase_mesh` filters `method_output.mesh_left` with the final
     accept mask, like `h`.
   - The record gets the key `mesh_left` (detached; `_record_results(...,
     mesh_left=...)`); every reader of `nodes[:, 0, :]` listed in item 1 reads it.
     The `record-order` plan later replaces this key by its integer `mesh_index`
     at the same places, and the `MethodOutput` field by `step_idxs[keep_idx]`.
   - Tests: `test_rk_integral.py`, `test_gradient.py`, `test_evaluate_f_on_mesh.py`,
     `test_oom_recovery.py`, `test_recheck_accepted_panels.py`,
     `test_record_and_sort.py`, `test_sorted_insert.py`, `test_adaptive_steps.py`,
     `test_trade_speed_for_memory.py`. Proof: snapshots and the grid
     (`bitgrid.py`, extended with dopri5, gk7, cc5, float32 and a device
     argument), CPU and CUDA, bit for bit.
2. **The optimal mesh from the final mesh** (changes no number).
   `_get_optimal_mesh` checks `len(mesh) - 1 == len(record)` (`ValueError`);
   `_merge_inputs` and `_predict_merged_errors` take the mesh (widths in float64
   from it; the at-the-fractions check compares the nodes with
   `_compute_nodes(mesh[:-1], mesh[1:])`); `_prune_excess_mesh` takes and returns
   the mesh, dropping the middle point of each merged pair; the two
   `_merge_excess_nodes` overrides, `UniformAdaptiveQuadrature._evaluate_adaptive_nodes`
   and both abstract declarations are deleted (item 6). Tests:
   `test_prune_and_optimal.py`, `test_evaluate_and_merge.py`. Proof: as step 1.
3. **Node fractions without end points; results; start-up point** (changes no
   number). `_check_node_fractions` and the variable override (item 3);
   `_flatten_output` (item 4); the start-up probe (item 2). Tests: a uniform
   method without end nodes (the 2-point Gauss rule) integrates polynomials of
   degree 3 exactly on uneven meshes and returns `N·C` rows; a variable method
   without ends is refused; `test_methods_registry.py` keeps the 0-and-1
   requirement only for variable methods. Proof: as step 1.
4. **Unpad Gauss-Kronrod** (numbers change at rounding level: D2). Delete the
   padding (`gauss_kronrod.py`), regenerate the Gauss-Kronrod snapshot cells
   only, update the tests that count 23 points, and the documents (section 8,
   item 9). New tests: evaluation counts (C = 9, 17, 21, 33; no point of `f`,
   start-up calls included, equals a mesh point), integrable singularities at a
   bound (`1/sqrt(t)`, `log(t)`, with and without `max_batch`), `result.y` equals
   `f(result.nodes)` row for row with `N·C` rows, bound gradients against the
   closed form. Measure: subinterval counts, integral and error changes, state
   mesh changes, against `d11df47`, CPU and CUDA.
5. **D7: a diagnosis for a divergent integral at a bound** (changes no number;
   own commit). A logged warning, once per call, when refinement goes deeper than
   a convergent integrand needs, naming where; the threshold is chosen after
   measuring how deep convergent strongly singular integrands go (section 11).
6. **dopri5 without its zero-weight node and its repeated node** (D4; own commit,
   last). `c = [0, 3/10, 4/5, 8/9, 1]`, `b` without its two zeros, error weight
   71/6300 at fraction 1; regenerate the dopri5 snapshot cells.

## 1. Summary

- **What is slow.** PadaQuad calls `f` at points whose values change nothing. Every
  Gauss-Kronrod method pads each subinterval with two nodes of weight zero at its
  ends, so `gk21` (the default) calls `f` 23 times per subinterval where 21 carry
  weight: 8.7 % of all evaluations (gk7 22.2 %, gk15 11.8 %, gk31 6.1 %).
  `dopri5` calls `f` at a node of weight zero (fraction 1/5) and twice at the same
  point (fraction 1): 2 of its 7 evaluations (28.6 %). Methods with real weights
  at both ends (Clenshaw-Curtis, the other Runge-Kutta methods,
  `interpolatory3_variable`) evaluate each shared end point again for every
  subinterval that touches it, including the halves of a split subinterval, whose
  parent had already evaluated both ends and the midpoint (cc17 11.8 %, cc5 40 %,
  fehlberg2 67 %, adaptive_heun 75 %). All of these numbers are counted, not
  estimated (section 2.3).
- **What the plan changes (recommended).** (1) The Gauss-Kronrod tables lose their
  padding, and the engine takes a subinterval's width and ends from the mesh, not
  from its first and last node. (2) The `dopri5` table drops the zero-weight node
  and folds its two nodes at 1 into one. (3) Re-use of end values across
  subintervals (designs b and c, section 3.3) is deferred to one mechanism shared
  with the `split-waste` plan.
- **Expected gain.** Counted: 8.7 % fewer evaluations of `f` for every `gk21` call
  (GPU baseline: `oscillatory` 631,650 to 576,724 points; `chirp` 3,290 to 3,004),
  and 8.7 % less memory for the values a pass holds. Estimated (not measured): wall
  time falls by about 8.7 % of the time spent in `f`, so by nearly 8.7 % when `f`
  is expensive and by little on the cheap benchmark integrands.
- **Numbers that change.** With the recommended compact layout, `gk21` integrals and
  error estimates change at rounding level (largest change measured: 1.6e-14 of
  |I| for `gk21`, 7.2e-11 of |I| for `gk7` on a strongly cancelling integral);
  no subinterval count changed in the 48 measured cells. The state's mesh changed
  in 1 of 32 Gauss-Kronrod cells (CPU; the same 1 of 32 on the GPU with the
  zero-padded layout), by 1 to 3 of about 77,000 subintervals.
  `result.nodes` and `result.y` lose the mesh points for Gauss-Kronrod. A
  zero-padded alternative keeps every integral and error estimate bit for bit
  (measured on CPU and GPU) at a memory cost (decision D2).
- **Also fixed.** An integrand singular at a bound (for example `1/sqrt(t)` at 0)
  no longer raises under Gauss-Kronrod, which never needed that point (measured:
  converges to 2 within the tolerance).

## 2. Evidence

### 2.1 The code as it is now

All in `padaquad/quadrature/base.py` unless named.

1. **Zero-weight padding.** `padaquad/methods/gauss_kronrod.py:136-146` appends
   fractions 0 and 1 with `b = b_error = 0` to every Gauss-Kronrod table. The
   comment gives the reason: `_RK_integral` computes the width as
   `h = t[:, -1] - t[:, 0]` (line 125), so the ends must be nodes.
   `_check_node_fractions` (`padaquad/methods/_base.py:276-304`) requires
   `c[0] == 0` and `c[-1] == 1` for every method, and
   `tests/test_methods_registry.py:224-244` enforces the padding.
2. **dopri5.** `padaquad/methods/runge_kutta.py:105-131`:
   `c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]`, `b[1] = 0` and `b_error[1] = 0` (a node
   with no weight at all, not listed in the task brief), and `b[6] = 0`,
   `b_error[6] = -1/60` at the repeated fraction 1. The comment at
   `base.py:2939-2940` notes the repeat.
3. **End nodes are mesh points, bit for bit.** `_compute_nodes` (551-587) uses
   `torch.lerp`, which returns `mesh_left` at c = 0 and `mesh_right` at c = 1
   exactly. So neighbouring subintervals evaluated together both call `f` at their
   shared mesh point.
4. **Split midpoints.** `_adaptively_increase_mesh` inserts
   `0.5 * (left + right)` (2089). The parent's node at c = 1/2 is
   `torch.lerp(left, right, 0.5)` = `right - 0.5 * (right - left)` (docstring of
   `_compute_nodes`, 563-565). Measured: the two were equal bit for bit for every
   split in every run of section 2.3 (column Dn = 0). My reasoning, not checked
   against PyTorch's kernels: both are the correctly rounded midpoint whenever
   `right - left` is exact (for example when `left >= right / 2 > 0`), and can
   differ by one unit in the last place otherwise.
5. **Every use of the first and last node as geometry** (what the padding exists
   for):
   - `_RK_integral` width, 125; `_calculate_integral` (589-629) has two callers,
     `_evaluate_f_on_full_nodes` (2451) and `_evaluate_f_on_split_nodes` (2775).
   - Record order through the left node: `_mesh_order(nodes[:, 0, :])` at
     1252-1253, 1767, 4354-4355, 4375, 4412, 4421; order checks on
     `record["nodes"][:, 0, 0]` at 4379 and 4425 (the `record-order` plan removes
     these).
   - `_predict_merged_errors`: `left_end` and `widths` from `nodes[:, 0, 0]` and
     `nodes[:, -1, 0]` (3043-3044) and the check that nodes sit at the fractions,
     `_compute_nodes(chunk[:, 0], chunk[:, -1])` (3058).
   - `_get_optimal_mesh`: `mesh_pruned` from `nodes_pruned[:, 0, :]` (3292-3294),
     after `_merge_excess_nodes` (`uniform.py:168-170`, `variable.py:206-208`)
     placed merged nodes from `nodes[i, 0]` and `nodes[i + 1, -1]`. Those merged
     nodes are used for nothing but their first entry.
   - `_flatten_output` (4536-4560) joins subintervals by dropping each later
     subinterval's first entry, which assumes shared ends; `result.nodes`,
     `result.y` and `result.tracked_variables` come from it (1657-1664).
   - `VariableMethodClass.weights` (`_base.py:645-646`) rescales by the first and
     last node; both variable methods have real end nodes and are unaffected.
   - `UniformAdaptiveQuadrature._evaluate_adaptive_nodes` (`uniform.py:112-121`)
     takes the midpoint from the end nodes; `integrate` never calls it (its
     docstring, 85-87).
6. **Where the end values themselves are read.** `_merged_pair_weights`
   (2907-2985) interpolates each neighbour through its values at all distinct
   fractions, the padded ends included; for two equal neighbours the merged
   method's centre node (fraction 1/2, the largest Kronrod weight) lands exactly on
   the shared mesh point (2960-2964). So the padded values are not entirely
   without use: they enter the merge prediction of the state's mesh. They enter
   no integral and no error estimate.
7. **Non-finite values.** `_check_f_output_finite` (2183-2246) raises when `f` is
   NaN or Inf at any evaluated node. With `runtime_checks=False` a non-finite
   value at a zero-weight node gives `0 * inf = NaN` in the subinterval's sum.

QUADPACK's 21-point routine evaluates `f` only at the centre and at
`centr ± hlgth * xgk(j)`, j = 1..10, with every `xgk(j) < 1`
(`dqk21.f`, lines 134-166; Piessens and de Doncker 1983, Netlib,
https://netlib.org/quadpack/, key `quadpack_netlib`; I downloaded and read those
lines on 2026-11-25). It never evaluates the ends.

### 2.2 The singular-bound claim, checked

`scripts/endpoint_singularity.py`, gk21, CPU, float64, `atol=1e-6`, `rtol=0`,
`max_batch=100_000`, `max_adaptive_splits=60` (logs `singularity_base.log`,
`singularity_proto.log`):

| Integrand on [0, 1] | Today, `runtime_checks=True` | Today, `False` | Zero-weight ends skipped |
|---|---|---|---|
| `1/sqrt(t)` (exact 2) | `ValueError` at t = 0 | integral NaN, not converged | 1.99999952, converged, 37 subintervals |
| `log(t)` (exact -1) | `ValueError` at t = 0 | integral NaN, not converged | -0.99999990, converged, 18 subintervals |

The start-up calls of `f` at `mesh_init` (shape check, 1118; `_initialize_max_batch`)
do not check finiteness, so a call without `max_batch` also converges
(`scripts/singularity_no_max_batch.py`).

### 2.3 Evaluations counted per method

`scripts/count_evaluations.py`: one call per cell on the login node's CPU,
float64, `atol=1e-8`, `rtol=0`, `error_norm="max"`, `take_gradient=False`,
`max_batch=1_000_000`, [0, 1], `torch.manual_seed(0)`. It captures the nodes of
every pass (one turn of the `while` loop) and classifies every evaluated point;
the classes add up to the points `f` received (asserted). Full output:
`counts_all.txt`, `counts_all_b.txt`.

- **a**: zero-weight nodes plus repeated fractions (dopri5).
- **b**: a needed point that an earlier subinterval of the same pass also
  evaluates (de-duplication within a pass).
- **b + c**: a needed end point already evaluated in this pass or an earlier one
  (in practice: the split parent's two ends and its midpoint node).
- **interior**: a needed interior point evaluated by an earlier pass (only
  bosh3's fraction 3/4 and a few dopri5 points): the `split-waste` plan's.

The CPU's random starting mesh differs from the GPU's, so the CPU counts for
`chirp` and `oscillatory` differ from the brief's GPU table; the GPU counts in
section 2.4 match the brief exactly.

| Method | Integrand | Points evaluated | Per subinterval | a | b | b + c | After a, b, c |
|---|---|---|---|---|---|---|---|
| gk21 | damped_sine | 253 | 23 | 22 (8.7 %) | 0 | 0 | 231 |
| gk21 | many_outputs | 345 | 23 | 30 (8.7 %) | 0 | 0 | 315 |
| gk21 | chirp | 3,243 | 23 | 282 (8.7 %) | 0 | 0 | 2,961 |
| gk21 | oscillatory | 599,403 | 23 | 52,122 (8.7 %) | 0 | 0 | 547,281 |
| gk7 / gk15 / gk31 | all four | | 9 / 17 / 33 | exactly 2 per subinterval: 22.2 / 11.8 / 6.1 % | 0 | 0 | |
| dopri5 | chirp | 6,741 | 7 | 1,926 (28.6 %) | 912 (13.5 %) | 1,438 (21.3 %) | 3,377 (49.9 % saved) |
| cc5 | oscillatory | 1,785,495 | 5 | 0 | 332,296 (18.6 %) | 714,186 (40.0 %) | 1,071,309 |
| cc9 | oscillatory | 1,246,707 | 9 | 0 | 135,181 (10.8 %) | 277,034 (22.2 %) | 969,673 |
| cc17 | oscillatory | 776,135 | 17 | 0 | 44,005 (5.7 %) | 91,298 (11.8 %) | 684,837 |
| cc33 | oscillatory | 580,899 | 33 | 0 | 17,420 (3.0 %) | 35,194 (6.1 %) | 545,705 |
| cc65 | oscillatory | 447,785 | 65 | 0 | 6,687 (1.5 %) | 13,766 (3.1 %) | 434,019 |
| adaptive_heun | damped_sine | 148,514 | 2 | 0 | 74,184 (50.0 %) | 111,379 (75.0 %) | 37,135 |
| fehlberg2 | damped_sine | 13,881 | 3 | 0 | 4,574 (33.0 %) | 9,242 (66.6 %) | 4,639 |
| bosh3 | chirp | 32,404 | 4 | 0 | 7,779 (24.0 %) | 16,190 (50.0 %) | 16,214 (4,027 more are interior, `split-waste`) |
| interpolatory3_variable | chirp | 39,363 | 3 | 0 | 12,739 (32.4 %) | 26,230 (66.6 %) | 13,133 |

What the counts show (read from the counts and the code, my interpretation):

- For Gauss-Kronrod only design (a) matters: its ends carry no weight, so sharing
  them saves nothing once they are skipped.
- For methods with a node at 1/2, every split's two halves need 4 end values (left
  end, midpoint twice, right end), all already evaluated by the parent: b + c is
  exactly 4 per split (cc5 `oscillatory`: 178,544 splits, 714,176 = 4 × 178,544,
  plus 10 shared points of the starting mesh). Sharing within a pass alone (b)
  recovers less than half of that, because most shared points were evaluated in
  the previous pass.
- The brief's estimate for Clenshaw-Curtis ("cc5: up to 1 of 5 points") counted
  only neighbours within a pass; with the parent's values it is 2 of 5.

### 2.4 Prototype of design (a), checked bit for bit

Prototype: `pkg/` (a copy of `0590ccf`; the edits are marked `PROTOTYPE` in
`pkg/padaquad/quadrature/base.py`). It keeps every array laid out at the 23
positions, calls `f` only at the positions whose `b` or `b_error` is nonzero, and
writes exact zeros at the others (`_pad`); the merge prediction interpolates only
through the evaluated positions; `_flatten_output` keeps only evaluated positions.
With `PROTO_COMPACT=1`, `_calculate_integral` instead sums over the 21 evaluated
positions only, which is the arithmetic of the recommended design.

`scripts/dump_results.py` (baseline copy and prototype, same seed) and
`scripts/compare_results.py`: gk7, gk15, gk21, gk31, dopri5, cc17 × damped_sine,
many_outputs, chirp, oscillatory, a Gaussian peak; `take_gradient` False and True
(True with a learnable scale `theta`, scalar integrands only): 48 cells on each of
CPU and GPU (A100, `nid008404`). Outputs: `compare_zero_fill_cpu.txt`,
`compare_zero_fill_cuda.txt`, `compare_compact_cpu.txt`,
`absdiff_compact_cpu.txt`.

| | Zero-padded layout (CPU, GPU) | Compact layout (CPU) |
|---|---|---|
| Points of `f` | gk21 −8.66 to −8.70 % (GPU `oscillatory` 631,650 → 576,724, `chirp` 3,290 → 3,004, `many_outputs` 346 → 316, `damped_sine` 254 → 232, the brief's figures); gk7 −22.2 %, gk15 −11.7 %, gk31 −6.0 %, dopri5 −14.3 % (only the 1/5 node), cc17 0 | same |
| `integral`, `integral_error` | bit-identical in 48 of 48 cells on both devices | rounding-level changes: largest \|ΔI\|/\|I\| 1.6e-14 for gk21, 7.2e-11 over all cells (gk7 `oscillatory`, whose integral is a cancelling sum of 75,000 terms); largest \|ΔE\|/\|I\| 3.8e-11 |
| Number of subintervals | unchanged in every cell | unchanged in every cell |
| State's mesh | changed in 5 of 48 cells on the CPU (gk7 `oscillatory` 74,363 → 74,362; dopri5 `chirp` 486 → 485 and `peak` 30 → 29, both modes), 1 of 48 on the GPU (gk7 `oscillatory` 77,394 → 77,397) | same cells as the zero-padded layout |
| `theta.grad` (`take_gradient=True`) | rounding-level change in 10 of 18 (CPU) and 7 of 18 (GPU) cells, largest relative 5.6e-16 | same |
| `result.nodes`, `result.y` | the mesh points are gone (gk21: N·21 rows instead of N·22 + 1); every remaining (node, value) pair is one of today's, bit for bit | same |

The state's mesh changes because the merge prediction can no longer interpolate
through the end values (section 2.1, item 6); the gradient changes because the
backward pass sums over fewer rows. The compact sum changes bits because removing
two exact zeros from a 23-term reduction changes how PyTorch groups the remaining
terms: `scripts/sum_bits.py` finds 48 to 58 % of rows different at `D = 1` on both
devices (logs `sum_bits_cpu.log`, `sum_bits_cuda.log`), and the same for `dopri5`'s
7-term against 5-term sum.

### 2.5 What is not measured

Wall time. The allocation the brief named (job 62011151, `nid008404`) ended
during this work, and the brief allows no other node; section 7 is the
measurement to run. The compact layout was compared on the CPU only. Everything
above is a count or a bit comparison.

### 2.6 Prototype files

All under `/pscratch/sd/k/khegazy/padaquad_speedup_plans/endpoint-evaluations/`:
`pkg/` (prototype package), `scripts/count_evaluations.py`,
`scripts/dump_results.py`, `scripts/compare_results.py`, `scripts/abs_diff.py`,
`scripts/sum_bits.py`, `scripts/endpoint_singularity.py`,
`scripts/singularity_no_max_batch.py`, and the logs named above. The large `.pt`
dumps were deleted; `dump_results.py` regenerates them (about 3 minutes per run on
the CPU). Delete the directory when the plan is done.

## 3. The design

Terms: a **pass** is one turn of the loop of `integrate` (evaluate the waiting
subintervals, judge them, split or record). The **evaluated positions** of a
method are the positions of its node fractions at which `f` is called.

### 3.1 Design (a), recommended form: node fractions are only the points `f` needs

1. **Tables.** The Gauss-Kronrod builder returns the 2n + 1 Kronrod fractions
   without padding (delete `gauss_kronrod.py:136-146`'s two concatenations).
   `dopri5`'s table becomes `c = [0, 3/10, 4/5, 8/9, 1]`, `b` without its two
   zeros, and the error weight at 1 equal to the sum of the two old ones,
   `71/6300` as one rational (11/84 − 649/6300 − 1/60 = 71/6300, derivation),
   if the user agrees (D4).
2. **Geometry from the mesh.** The width of a subinterval is
   `mesh[i + 1] − mesh[i]`, passed to `_calculate_integral` and `_RK_integral`
   instead of computed from the nodes. After the loop, `_predict_merged_errors`
   and `_get_optimal_mesh` read the ends from the final mesh, whose subintervals
   are exactly the recorded ones, in order. `_get_optimal_mesh` drops the middle
   mesh point of each merged pair instead of computing merged nodes it uses only
   for their first entry; the two `_merge_excess_nodes` overrides then do the same
   thing (sum the pair's contribution and error) and become one method in the
   base class, and the abstract method goes. This removes code.
3. **Validation.** `_check_node_fractions` requires fractions in [0, 1], at least
   one, non-decreasing; `VariableMethodClass` additionally requires 0 and 1,
   because its `weights` rescale by the end nodes.
4. **Results.** `_flatten_output` drops the first entry of every later subinterval
   only when the method's fractions start at 0 and end at 1. `result.nodes`,
   `result.y` and `result.tracked_variables` then list exactly the points `f` was
   evaluated at (for gk21, `N·21` rows); the per-batch result `loss_fxn` receives
   has `[N, 21, ...]`, aligned as today.
5. Everything that counts points per subinterval already uses `self.C =
   method.n_nodes` (1177, 1236, 2306-2319, 2420-2433, 2620; `memory.py:1254, 1816`),
   which becomes 21 for gk21 without further change.

Why this is the least code: the evaluation, batching, in-place pass array,
out-of-memory and tracked-variable paths do not change at all, because a method
simply has fewer nodes. The new code is one argument (the width), a relaxed check
and a condition in `_flatten_output`; the merged-node computations and one abstract
method are deleted. The padding, its comment, the test that enforces it, and the
explanation in the user guide go too.

What it depends on: the record must not be ordered by `nodes[:, 0]` (section 9,
`record-order`).

### 3.2 Alternatives for design (a) that I rejected (or offer as fallback)

- **Zero-padded layout** (prototyped): keep the 23 positions as geometry and in
  every value array, call `f` at the 21 weighted positions, store exact zeros at
  the others. Bit-identical integrals (measured), no dependence on `record-order`.
  Rejected as the default because (1) a pass covered by one call of `f` can no
  longer use `f`'s output as the pass's array without a copy (`base.py:2690-2694`),
  so the pass's values are held twice for a moment, which undoes the "hold a
  pass's values once" property this branch built (`tests/test_trade_speed_for_memory.py`
  checks a GPU peak below 1.8 times a pass's values); (2) every held value array
  is 23/21 as large as needed; (3) the arrays carry placeholder zeros that
  `loss_fxn`, `_flatten_output` and the merge prediction must each know to skip.
  It is the fallback if the user wants bit-identity (D2), or if `record-order`
  does not land.
- **Compact values, padded geometry**: nodes `[N, 23, T]`, values `[N, 21, D]`.
  Same arithmetic change as 3.1 without its clean-up, and nodes and values no
  longer align row for row in what `loss_fxn` receives. Rejected.
- **A NaN marker at skipped positions**: `0 · NaN = NaN` poisons every sum.
  Rejected.
- **Bit-identity with compact storage**: form the 23-wide product per block of
  rows inside `_RK_integral` (zeros scattered into a block-sized temporary) so the
  reduction groups as today. Possible (my idea, not prototyped), but it is code
  whose only purpose is to keep last bits that the `quadrature-arithmetic` plan
  may change anyway. Rejected unless the user asks for it.

### 3.3 Designs (b) and (c): shared end points of methods with real end weights

These save nothing for Gauss-Kronrod after (a), and a lot for the
Clenshaw-Curtis, Runge-Kutta and variable methods (section 2.3).

- **(b) Evaluate each distinct point of a pass once.** Before the calls of `f`,
  drop each subinterval's left end when its left neighbour is in the same pass
  (the points are equal bit for bit, section 2.1 item 3); after, gather the value
  into both subintervals.
  - Cost in code: the calls no longer cover "C consecutive rows per subinterval",
    so `_evaluate_f_on_split_nodes`'s whole-subinterval and leftover logic
    (2650-2752) must count points per subinterval (C or C − 1); the one-call
    no-copy path is lost (a gather builds the pass's array), and the tracked
    variables need the same gather. Estimate: 60 to 100 lines changed.
  - Cost on the device per pass: one comparison of `step_idxs` with its shifted
    self, one index build, one gather of `K·C` rows (a copy of the pass's values).
  - `_evaluate_f_in_batches` (2454) is unchanged: it calls `f` on whatever flat
    list it is given.
  - `take_gradient=True`: a batch is one call of `f` and one backward pass, so a
    value shared by two subintervals of the same batch is in one graph and its
    gradient is the sum of both uses: correct. Neighbours in different batches
    are in different calls and are not shared.
  - In-place pass array: the gather writes into it; the shortcut that uses `f`'s
    output as the array is lost.
- **(c) Carry the parent's ends (and its node at 1/2) to its halves.** Keep, for
  every split subinterval, its values at fractions 0, 1/2 (when it has one) and 1,
  on the solver's device, and use them for the halves' ends in the next pass.
  - `take_gradient=True`: not possible. The parent's values were part of a batch
    whose `backward()` freed their graph (the same problem as
    `docs/issues/variable-sampling-reuse.md`, point 4); detaching them would drop
    their gradient. So (c) runs only with `take_gradient=False`.
  - Re-checked subintervals (`enforce_convergence`) have their values in the
    record, possibly on `result_device`: 2 or 3 rows per split are copied back.
  - With `record_evaluations=False` or `trade_speed_for_memory=True` the end values
    must still be kept on the solver's device: at most `(K + 1)·D` values per pass.
  - The midpoint must be the parent's node bit for bit; compute the inserted
    midpoint with `torch.lerp(left, right, 0.5)` (the formula of `_compute_nodes`),
    which may move a midpoint by one unit in the last place in rare cases
    (section 2.1, item 4), or reuse only where the bits are equal (one
    `torch.where`).
- **Recommendation: do not build (b) and (c) in this plan.** The `split-waste`
  plan must build exactly the machinery (c) needs, a store of a split parent's
  values handed to its halves with the same `take_gradient`, tracked-variable,
  re-check and memory questions, for the parent's interior values. Built twice,
  it would be two mechanisms for one task. I recommend that `split-waste` builds
  one carrier for "every node of a new subinterval that coincides bit for bit with
  a node of its parent", which covers the ends, the midpoint and bosh3's 3/4 in
  one rule, and that (b) is added to it as "a mesh point shared by two waiting
  subintervals is evaluated once". The counts of section 2.3 are the acceptance
  numbers for it (decision D5).

### 3.4 Interaction with non-finite values and nondeterministic integrands

- Skipping a zero-weight node changes no sum mathematically. It changes results
  only where `f` is NaN or Inf at a mesh point: today the subinterval becomes NaN
  (or `runtime_checks` raises), after the change it is finite. Measured in 2.2.
- A singularity exactly at an interior mesh point is then no longer reported by
  `runtime_checks` under Gauss-Kronrod; the loop behaves as it does today for a
  singularity between mesh points. Not tested beyond 2.2 (risk 9.4).
- An integrand whose value at a point depends on the other rows of its call (for
  example a network whose matrix-multiplication kernel is chosen by batch size)
  can change in the last bits whenever the call sizes change, and they change here
  (fewer points per call). This is true of any change of `max_batch` today.
- Sharing values (b, c) would make two subintervals use one value where they now
  use two calls' values; for a deterministic `f` in one call these are equal,
  for a nondeterministic `f` (dropout, atomics) they are not, and results change.

## 4. Decisions for the user

- **D1. What `result.nodes`, `result.y` and `result.tracked_variables` contain for
  Gauss-Kronrod.**
  - (i) Only the points `f` was evaluated at: `N·C` rows for gk21 with C = 21, no
    mesh points. **Recommended**: every row is a real evaluation, and the shapes
    follow from one rule ("the evaluated points, each once").
  - (ii) Keep the mesh points with NaN in `y`: users who plot `y` get gaps, and
    every consumer must handle NaN. Not recommended.
  - Either way the `IntegrationResult` docstring's `P = C + (N − 1)(C − 1)` holds
    only for methods with both ends among their nodes.
  - **Decided by the user, 2026-11-25: (i).** Not evaluating unnecessary points
    is preferred, even though `result.nodes` then no longer contains
    `mesh_init`, `mesh_final` or the other mesh points.
- **D7 (added 2026-11-25 by the coordinating session; open). A divergent
  integral at a bound** no longer raises at once (section 9, item 8: 1,011
  turns for `1/t`, then an error naming a subnormal t). Options: (a) a message
  and stop when a subinterval has been halved more often than any convergent
  integrand needs (the threshold must be chosen and tested; `1/sqrt(t)` at
  `atol=1e-6` reached depth 37 or less in section 2.2); (b) a finite default
  for `max_adaptive_splits`; (c) leave it, and document it. Not yet decided.
- **D2. Rounding-level change of integrals versus memory.**
  - (i) Compact layout (3.1): integrals and error estimates change at rounding
    level once (snapshots regenerated); values held use 8.7 % less memory.
    **Recommended**, ideally landed together with any bit-changing step of the
    `quadrature-arithmetic` plan so the snapshots change once.
  - (ii) Zero-padded layout (3.2): bit-identical integrals (measured), but a
    pass covered by one call is held twice for a moment and every held array is
    9.5 % larger than needed.
- **D3. Integrands singular at a bound.** After (a), Gauss-Kronrod never
  evaluates `f` at a mesh point, so integrable singularities at the bounds work
  (2.2), while Clenshaw-Curtis and the Runge-Kutta methods still evaluate the
  ends and raise. **Recommended**: document this in the methods page as a reason
  to choose Gauss-Kronrod, with the `1/sqrt(t)` example as a test. The
  alternative, keeping a raise for consistency, has no use I can see.
- **D4. Change `dopri5`'s table** (drop the zero-weight node at 1/5, fold the two
  nodes at 1): 28.6 % fewer evaluations for `dopri5`; its integral and error
  estimate change at rounding level (`sum_bits.py`), its `result.nodes` lose two
  rows per subinterval, and the table no longer reads as the Runge-Kutta tableau
  of torchdiffeq. **Recommended: yes**, with a comment that records the original
  tableau and the folding. Alternative: leave the legacy method untouched.
- **D5. Shared end points (b, c)** for the Clenshaw-Curtis, Runge-Kutta and
  variable methods. **Recommended**: build them as part of the `split-waste`
  carrier (3.3). Alternatives: a separate mechanism here (two mechanisms for one
  task), or not at all (gk21 users gain nothing from it either way).
- **D6. Order of work.** **Recommended**: after `record-order`. If `record-order`
  is dropped, either add a record field with each subinterval's left mesh point
  (six call sites of `record["nodes"][:, 0, :]` read it instead) or take the
  zero-padded layout (D2 ii).

## 5. Implementation plan (recommended path; superseded by section 5')

Each step is one commit; steps 1 to 3 must change no number, bit for bit.

0. **Prerequisite.** `record-order` has landed: no code reads
   `nodes[:, 0, :]` or `record["nodes"][:, 0, 0]` to order or check the record
   (today 1252-1253, 1472, 1767, 4354-4355, 4375-4380, 4412, 4421-4426).
1. **Width from the mesh.**
   - `_RK_integral(h, y, tableau_b, y0)` takes the widths `[N, T]` instead of `t`;
     `_calculate_integral(t, y, h, y0=None)` passes them (`t` still feeds
     `method.weights`, which the variable methods need).
   - `_evaluate_f_on_full_nodes`: `h = mesh[idxs + 1] − mesh[idxs]`;
     `_evaluate_f_on_split_nodes`: the same for `step_idxs`, sliced with the nodes
     at 2775.
   - Update `MethodOutput.h`'s docstring (`results.py:249-250`) and
     `_RK_integral`'s (107-109).
   - Tests: `tests/test_rk_integral.py`, `tests/test_gradient.py`,
     `tests/test_oom_recovery.py:292`, `tests/test_recheck_accepted_panels.py:821`
     pass widths. Proof: snapshots and the grid harness bit-identical (section 6).
     The width is bit-identical because `torch.lerp` returns the mesh points
     exactly at fractions 0 and 1 (2.1, item 3).
2. **Geometry from the mesh after the loop** (coordinate with `optimal-mesh-cost`,
   which owns this code).
   - `_predict_merged_errors(mesh, y, mesh_quadratures)` (or `nodes` for the
     fraction check plus `mesh`): widths in float64 from the mesh; the
     at-the-fractions check compares `nodes` with `_compute_nodes(mesh[:-1],
     mesh[1:])`. Assert `len(mesh) − 1 == len(record)` where `_get_optimal_mesh`
     is called (the record covers the final mesh once every tracker is False).
   - `_merge_inputs` reads T and N from the mesh.
   - `_get_optimal_mesh`: `mesh_pruned = mesh[keep]` with `keep[merge_idxs + 1] =
     False`; `_prune_excess_mesh` returns the merged pair indices and the summed
     contributions and errors; one base implementation replaces the two
     `_merge_excess_nodes` overrides (`uniform.py:128-208`, `variable.py:162-246`)
     and the abstract declaration (`base.py:671-701`).
   - Delete `UniformAdaptiveQuadrature._evaluate_adaptive_nodes` (dead,
     `uniform.py:69-126`) and its tests in `tests/test_evaluate_and_merge.py`,
     or leave the decision to `split-waste`, which owns the variable one; it
     would compute wrong midpoints for an unpadded method.
   - Tests: `tests/test_prune_and_optimal.py`, `tests/test_evaluate_and_merge.py`
     (merge part). Proof: bit-identical snapshots and harness, including
     `state.mesh`.
3. **Fractions without ends allowed for uniform methods; results.**
   - `_check_node_fractions` (`_base.py:276-304`): uniform methods need fractions
     in [0, 1], non-decreasing; `VariableMethodClass.__init__` also requires 0 and
     1. `measure_degree` calls the same check (381).
   - `_flatten_output`: join shared ends only when the method's fractions start
     at 0 and end at 1.
   - Replace `test_uniform_methods_have_canonical_first_and_last_nodes`
     (`tests/test_methods_registry.py:224-244`) by a test that every method
     integrates `f(t) = t` and `f(t) = 1` exactly on uneven meshes (the regression
     it was written for); keep the 0-and-1 assertions only for variable methods
     (`test_methods_registry.py:186-187`).
   - Proof: bit-identical (no method changes yet).
4. **Unpad Gauss-Kronrod** (`gauss_kronrod.py:136-146`; module docstring 22-24;
   builder docstring 90-92).
   - Numbers change (D2): regenerate the Gauss-Kronrod cells of
     `tests/test_snapshots_data.json` and say why in the commit.
   - Tests with gk21's 23 hard-coded switch to `solver.C`:
     `tests/test_bug_regressions.py:268`, `tests/test_memory_state.py:81, 90, 189,
     313`, `tests/test_memory_state_break.py:226, 336-337`,
     `tests/test_memory_benchmark.py:1305, 1835, 1844, 1945`,
     `tests/test_memory_state_adversarial.py:216-217, 232, 251`,
     `tests/test_trade_speed_for_memory.py:36-37, 214, 222` (check each: some
     choose batch sizes relative to C on purpose).
   - New tests (section 6).
5. **dopri5 table** (if D4 is yes): `runge_kutta.py:96-131`; update the module
   docstring (7 nodes) and `uniform.py:5-6, 39-41`. Regenerate the dopri5 snapshot
   cells.
6. **(b) and (c)**: not here (D5); hand the counts of section 2.3 to
   `split-waste`.

## 6. Tests, and how to show what changed

- **New: evaluation counts.** Wrap `f` and count points per pass: every pass
  calls `f` at `C` points per waiting subinterval, with C = 9, 17, 21, 33 for
  gk7, gk15, gk21, gk31 (5 for dopri5); for every Gauss-Kronrod method no
  evaluated point equals a mesh point. (The totals of section 2.3 are the
  reference: gk21 `oscillatory` on the CPU 547,281 points.)
- **New: singular bound.** `1/sqrt(t)` and `log(t)` on [0, 1] with every
  Gauss-Kronrod method, `runtime_checks=True`: converged, within the tolerance of
  2 and −1; with `max_batch` given and not given.
- **New: results.** `result.y` equals `f(result.nodes)` row for row; for
  Gauss-Kronrod, `len(result.nodes) == N·C`; for methods with both ends,
  `N(C − 1) + 1` as today; tracked variables aligned.
- **Changed:** `tests/test_methods_registry.py` (3.1, step 3),
  `tests/test_interpolation.py::test_endpoints` (endpoints only for methods with
  0 and 1), `tests/test_tableau.py` (weights sum to 1 still holds),
  `tests/test_exactness.py` (degrees unchanged; the measurement runs on the
  unpadded table).
- **Bit-identity of steps 1 to 3:** `tests/test_snapshots.py`, and the grid
  harness in `/pscratch/sd/k/khegazy/padaquad_methodclass_verify/`
  (`grid_run.py`, `grid_run_gpu.py`, `grid_compare.py`), run before and after
  on CPU and GPU. The harness predates the current API and must be updated first:
  `integrate` returns `(result, state)` (it assigns one value at lines 77, 99,
  106, 115 of `grid_run.py`); `same_f=True` is gone (pass `state=` from the first
  call); the field `mesh_optimal` is now `state.mesh` (`FIELDS`, line 17).
- **Steps 4 and 5 (numbers change):** run `scripts/dump_results.py` and
  `scripts/compare_results.py` before and after; expect subinterval counts
  unchanged in every cell, `integral` and `integral_error` within a few units in
  the last place of the sum of |contributions| (measured bound 1.6e-14 · |I| for
  gk21), the state's mesh unchanged in most cells (measured: 1 of 32 Gauss-Kronrod
  cells per device, by at most 3 subintervals). Then regenerate the snapshot
  cells of the changed methods only, and check that `n_optimal_mesh` changes only
  where the comparison above showed it.
- `examples/verification/integral_accuracy.py` and `polynomial_degree.py` must
  still pass unchanged.

## 7. How to measure the gain

- **Counts (decisive, any node):** `scripts/count_evaluations.py gk21 dopri5`
  before and after; and `/pscratch/sd/k/khegazy/padaquad_speedup_profile/profile_baseline.py cuda`
  on a GPU node, whose "f points" must fall from 254, 346, 3,290 and 631,650 to
  232, 316, 3,004 and 576,724 (the zero-padded prototype gave exactly these on
  the A100; the compact design gives the same counts unless a decision flips).
- **Wall time (compute node only, never the login node):** the same script's
  median times (expect a small change for the cheap integrands, where `f` is a
  small part of a pass); and an expensive integrand, the Fourier-feature networks
  of `examples/speed_test/trade_speed_for_memory.py` with gk21 and a fixed
  `max_batch`, where the time of the calls of `f` should fall by about 8.7 %
  (estimate) and the GPU peak of the pass's values by about 8.7 %.

## 8. Documentation to update

- `docs/user_guide/methods.md:27-29` (the 23 points), and a short note on
  singularities at the bounds (D3).
- `padaquad/methods/gauss_kronrod.py` module docstring (22-24) and the builder's
  docstring and comment (90-92, 136-146).
- `padaquad/results.py`: `IntegrationResult` (86-88 the formula for P; 119-131
  `nodes`, `y`, `tracked_variables`), `MethodOutput.h` (249-250).
- `padaquad/methods/_base.py`: `_Tableau.c` (46-48), `MethodClass.node_fractions`
  (209-219), `_check_node_fractions` (276-291).
- `padaquad/quadrature/base.py`: `_RK_integral` (104-109), `_flatten_output`,
  `_mesh_order`'s docstring if it survives `record-order` (4138-4141).
- `docs/user_guide/devices_memory.md:210` (gk21 "with 23 nodes").
- `CLAUDE.md`: "Tensor shape conventions" (C = 23 for gk21 "with endpoint
  padding"), the `test_methods_registry.py` row, "Uniform vs variable sampling"
  (merging), and "Core algorithm".
- `docs/references.bib`: extend the `checked` field of `quadpack_netlib` with
  "dqk21.f lines 134-166: f evaluated at centr and centr ± hlgth·xgk(j),
  j = 1..10, every xgk(j) < 1; no end point" (read on 2026-11-25).
- `CHANGELOG.md`: "Changed": Gauss-Kronrod methods evaluate `f` at 2n + 1 points
  per subinterval, not 2n + 3 (gk21: 21, 8.7 % fewer calls' points), and
  `result.nodes`/`y`/`tracked_variables` no longer contain the mesh points for
  them; dopri5 at 5 points (D4); integrals change at rounding level. "Fixed":
  integrable singularities at a bound no longer raise under Gauss-Kronrod.
- The methods table (`docs/_ext/methods_table.py`, column "Nodes per
  subinterval") updates itself from `n_nodes`.

## 9. Risks and interactions with the other plans

1. **`record-order`** (prerequisite): I assume every record row carries its
   subinterval's mesh index as an integer, kept correct across splits, and that
   `_mesh_order`, `_get_mesh_indices` and the order checks on
   `record["nodes"][:, 0, 0]` are gone. Without that, the unpadded nodes break
   the record's order (fallback in D6).
2. **`optimal-mesh-cost`** owns `_get_optimal_mesh` and the merge prediction.
   My requirement: they take subinterval ends and widths from the mesh, not from
   `nodes[:, 0]` and `nodes[:, -1]`, and the merge prediction interpolates through
   the evaluated nodes only (for gk21 the merged centre node, on the shared mesh
   point, is then extrapolated by 0.2 % of the width from the outermost Kronrod
   node; measured effect in 2.4). Collapsing `_merge_excess_nodes` (step 2) also
   simplifies their work.
3. **`quadrature-arithmetic`** owns the weighted sums. Step 1 changes
   `_RK_integral`'s signature (width passed in); step 4 changes its bits. If they
   also change bits, land both before one snapshot regeneration.
4. **`split-waste`** owns reuse of a split parent's interior values and splitting
   into more than two. I leave to it: the carrier of parent values, which should
   also carry the ends and the midpoint (3.3, D5); the `take_gradient=True`
   restriction; and the dead `_evaluate_adaptive_nodes` methods.
5. **`startup-evaluations`**: the shape-check call and the start-up test calls
   evaluate `f` at `mesh_init`, a bound. They do not check finiteness today
   (2.2), so singular bounds still work; any replacement should keep that (or use
   an interior point).
6. **`turn-device-reads`, `compiled-turn`**: fewer nodes per subinterval reduce
   every node and value array of a pass by 8.7 % for gk21; nothing else
   interacts. The evaluated positions are fixed per method and need no device
   read.
7. **Users' code** that slices `result.nodes` by `C − 1` for Gauss-Kronrod, or
   reads the value at a mesh point from `result.y`, breaks (D1).
8. **A singularity exactly at a mesh point** is no longer reported by
   `runtime_checks` under Gauss-Kronrod (3.4). For a non-integrable singularity
   the loop then refines toward it as it does today for one between mesh points;
   I did not test what that does at the last representable widths.

   Tested afterwards by the coordinating session (2026-11-25, CPU, float64,
   gk21, `atol=1e-6`, `rtol=0`, `1/t` on [0, 1], this plan's prototype in
   `pkg`; script `divergent_bound.py`, kept beside this plan's other scripts):

   | Code | `max_adaptive_splits` | Outcome |
   |---|---|---|
   | Today | None | `ValueError` naming t = 0, in turn 1 |
   | Today, `runtime_checks=False` | None | integral NaN, `converged=False`, 1 turn |
   | Prototype | 60 | integral 53.5, `converged=False`, 61 turns |
   | Prototype | None | `ValueError` naming t = 2.9e-309, after 1,011 turns |
   | Prototype, `runtime_checks=False` | None | integral inf, `converged=False`, 1,011 turns |

   So a divergent integral at a bound loses today's immediate, located error:
   without a split cap the loop halves toward the bound until `1/t` overflows
   at a subnormal t. The implementation needs an early, clear stop for this
   case (a decision for the user; see D7).

## 10. Confidence

Sure (measured or read in the code):

- The waste: exactly 2 evaluations per subinterval for every Gauss-Kronrod method
  and 2 per subinterval for dopri5, and the shared-end counts of section 2.3
  (counted, the classes add up to the points `f` received).
- Skipping the zero-weight nodes with the zero-padded layout leaves every
  integral and error estimate bit-identical on the measured 48 cells per device;
  the compact layout changes them only at rounding level and changed no
  subinterval count there.
- The state's mesh and `take_gradient=True` gradients change slightly in both
  layouts, for the reasons given.
- Gauss-Kronrod with a singularity at a bound raises today and converges once the
  ends are not evaluated.
- QUADPACK's dqk21 never evaluates the ends (read in the source).

Not sure:

- Wall-time gain (not measured; estimated as up to 8.7 % of the time in `f`).
- That steps 1 and 2 are bit-identical everywhere: argued from `torch.lerp`'s
  exact ends and from the record covering the final mesh; to be proven by the
  harness.
- The size of the merge-prediction change on integrands other than the five
  measured, and on float32.
- The code estimate for (b) (60 to 100 lines) and how much of (c) the
  `split-waste` carrier will already provide.
- Whether `0.5 * (l + r)` and `torch.lerp(l, r, 0.5)` can differ in practice
  (never in the measured runs; my reasoning says only when `r − l` is inexact).

## 11. Implemented (branch `endpoint_evaluations`, 2026-11-26)

Scripts, logs and grid files: `$SCRATCH/padaquad_speedup_plans/endpoint-evaluations/impl/`
(`bitgrid.py` is the `record-order` grid extended with dopri5, gk7, cc5, float32
cells and a device argument: 656 integrations; `bitcompare.py` bit for bit;
`scripts/numcompare.py`, `scripts/samemesh.py` for the steps that change numbers).
Node `nid200297` (A100-SXM4-80GB), CPU runs there or on the login node.

| Step (5') | Commit | What it does | Numbers |
|---|---|---|---|
| review | `d03ab9a` | section 0 and 5' | none |
| 1 | `89eadbf` | width and left mesh point from the mesh (`_calculate_integral(t, y, mesh_left, mesh_right)`, `_RK_integral(h, ...)`, `MethodOutput.mesh_left`, record key `mesh_left`) | grid CPU and CUDA: every tensor equal except the gradient with respect to a bound, at most 1.8e-14 relative (CPU) and 1.5e-14 (CUDA) |
| 2 | `6201cff` | optimal mesh from the final mesh; `_merge_excess_nodes` (both), the uniform `_evaluate_adaptive_nodes` and both abstract declarations deleted | grid CPU: 0 differences against step 1 |
| 3 | `ec6e3ca` | fractions without end nodes allowed; `_flatten_output`; start-up probe | grid CUDA (steps 1-3 against `d11df47`): only the bound gradients differ |
| lean | `28057af` | follow-up after the user's instruction on fewer code paths: one fraction check for every method (no variable override), one `_flatten_output` expression, variable weights rescale to the first and last fraction, the length check in `_get_optimal_mesh` removed | grid CPU and CUDA: 0 differences against steps 2 and 3 |
| 4 | `4315f2d` | Gauss-Kronrod unpadded; merge prediction takes the mean of both neighbours at their shared mesh point; documents; new `tests/test_interior_nodes.py`; Gauss-Kronrod snapshot cells | see below |
| 6 (D4) | `734f7f7` | dopri5 at 5 nodes, error weight 71/6300 | see below |
| test fix | `3c54432` | `assert_step_continuity` checks the widths, not nodes at the bounds | none |

Numbers changed by step 4, against the lean commit (cells whose accepted
subintervals are the same): Gauss-Kronrod integrals at most 2.4e-16 (float64)
and 6.7e-8 (float32) of the sum of |contributions|, error estimates at most
2.4e-16 and 9.1e-9; CPU and CUDA alike. Subinterval counts changed only with
`take_gradient=True` and `max_batch=200` or 300, where a batch now holds another
number of whole subintervals (CPU: gk21 oscillatory 168 -> 171, gk7 2298 -> 2299
and 353 -> 354; CUDA: gk21 125 -> 127, gk7 2286 -> 2283 and 357 -> 356). Methods
with end nodes: bit-identical in all 656 cells on both devices, so the mean at
the shared point changed nothing there. Snapshots: 42 of 120 Gauss-Kronrod
integrals and all 120 error estimates differ, at most 2.4e-16 relative; no
`n_optimal_mesh` changed. Without the mean at the shared point,
`test_predicted_far_from_the_origin[f32]` failed: the merge prediction of gk21 on
[1000, 1000.5] in float32 was 100 times larger (1.4e-7 against 1.5e-9; with the
mean 1.5e-9), so the state's mesh kept 59 instead of fewer than 16 subintervals.

Numbers changed by D4: dopri5 only; same-mesh cells at most 1.7e-16 (float64)
and 1.7e-8 (float32); subinterval counts change in 2 to 3 cells with
`take_gradient=True` and a small `max_batch` (batching) and the state's mesh in 4
(CPU) and 1 (CUDA) cells by one or two subintervals. Snapshots: 10 of 30 dopri5
integrals (at most 1.7e-16) and all 30 error estimates (at most 6.1e-18 of
|I|); no `n_optimal_mesh` changed.

Points of `f` and wall time (gk21, `atol=1e-8`, `rtol=0`, `error_norm="max"`,
`max_batch=10**6`, float64; `scripts/timing.py`, median of 7):

| Integrand | Points `d11df47` -> branch | GPU ms | CPU (8 threads) ms |
|---|---|---|---|
| damped_sine | 254 -> 232 | 5.91 -> 5.96 | 3.62 -> 3.32 |
| chirp | 3,290 -> 3,004 (CPU 3,244 -> 2,962) | 59.7 -> 61.9 | 19.3 -> 17.9 |
| oscillatory | 631,650 -> 576,724 (CPU 599,404 -> 547,282) | 130 -> 142 | 108.8 -> 107.8 |
| sin(2e4 t) + MLP of width 2048 (`scripts/timing_mlp.py`, atol 1e-9) | 82,502 -> 75,328 | 130.8 -> 122.3 and 129.3 -> 121.7 (two runs) | not run |

The point counts are those section 1 predicted. With an expensive integrand the
GPU time falls by 6.5 %; with the cheap ones, where `f` is a small part of a
pass, the GPU differences are within the run-to-run spread (chirp ranged from
57.3 to 61.8 ms across the step-by-step runs, all on the same node).

Tests: full suite on `nid200297` before the test fix: CUDA 4,196 passed, 25
skipped, 56 failed; CPU 3,906 passed, 315 skipped, 56 failed; the 56 are the
`test_step_continuity` cases of `test_integrals.py` and `test_data_types.py`
(Gauss-Kronrod), fixed by `3c54432` (those two files then: 728 passed on CUDA
and on CPU). `d11df47` on the same node: CUDA 4,249 passed, 25 skipped; CPU
3,959 passed, 315 skipped; no failure.

## 12. Not done

- **D7 (a divergent integral at a bound).** Not implemented: every form needs a
  new code path in the loop, which the user's instruction excludes. Measured
  (gk21, CPU, `max_adaptive_splits=None`, `scripts/d7_depths.py`):
  `t^-0.5` converges at depth 26 (`atol=1e-6`) and 52 (`1e-10`); `t^-0.9` at 193
  and 326; `t^-0.99` (integrable) needs more halvings than float64 has and
  raises `ValueError` at a subnormal t after 1,021 turns (4 s); `1/t` at 0
  raises after 1,011 turns (3 s) naming t = 2.9e-309; `1/(1 - t)` at 1 raises
  after 43 turns naming t = 1.0 (a node rounds onto the bound). So no depth
  separates convergent from divergent integrands. Options for the user: (a) a
  warning when a subinterval at a bound passes a depth, worded "may diverge"
  (false positives for `t^-0.9`); (b) a finite default for `max_adaptive_splits`
  (would stop `t^-0.9` short of its tolerance); (c) detect an error estimate
  that does not fall when the subinterval at a bound is halved (state across
  turns); (d) leave it, documented. The branch does (d): a paragraph on the
  methods page.
- Removing the de-duplication of repeated fractions in `_merged_pair_weights`
  (`first`, the `None` cache key): possible once D4 is kept, since no registered
  method then repeats a fraction, but repeated fractions are still allowed by
  `_check_node_fractions` and tested (`test_repeated_node_fractions_accepted`).
- Designs (b) and (c) (D5): out of scope.
