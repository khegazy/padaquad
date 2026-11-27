# No solver reuses integrand values when it splits a subinterval

Status: open. Found 2026-11-12 on branch `website_docs_update` (base `main` at
95936ce); rewritten 2026-11-26 on branch `split_waste`, when the unused reuse
method was deleted. Written so that another agent can pick it up without the
original sessions.

## Summary

The variable-sampling solver (`VariableAdaptiveQuadrature`) computes each
subinterval's weights from where its nodes are, so that a split could keep the
parent's integrand values in its halves and evaluate `f` only at new nodes. This
reuse does not exist, for either solver: when a subinterval fails its
tolerance, `_adaptively_increase_mesh` inserts its midpoint into the mesh (and
the midpoints of both halves for a split in four, `split_rule="accelerate"`)
and drops its values, and the next pass evaluates `f` at fresh nodes in every
piece, placed by `_compute_nodes` at the method's node fractions.

The method that implemented reuse for the variable solver,
`VariableAdaptiveQuadrature._evaluate_adaptive_nodes`, had no caller since
commit `da752ea` (2025-02-22) and was deleted on branch `split_waste` with its
tests (`tests/test_evaluate_and_merge.py`). Nothing replaced it. It could not
have been reconnected as it was:

- It assumed that a subinterval's first and last nodes are its ends (it
  duplicated the middle node as the halves' shared end). The Gauss-Kronrod
  methods have no nodes at the ends since the `endpoint-evaluations` plan
  (`docs/issues/2026-11-25-speedup-endpoint-evaluations-plan.md`, deleted
  once done, in commit `3033e24`), and a
  `MethodClass` may now put its node fractions anywhere in [0, 1].
- It overrode nothing: that plan deleted the uniform solver's version and the
  abstract declaration in `AdaptiveQuadrature` (commit `6201cff`).
- It allocated its nodes with last dimension `D` (outputs of `f`) instead of
  `T` (coordinates of `t`), returned no tracked variables, and called `f`
  outside the batching and out-of-memory handling of `_evaluate_f_in_batches`.

A separate observation, unchanged: every subinterval's nodes sit at the
method's node fractions, so `interpolatory3_variable` always has its middle
node at the center, where its weights are Simpson's rule. Its general
position-dependent weights (`sanderse2019` in `docs/references.bib`,
Eq. (119)) are never used in a real integration.

## What reuse would save (measured)

For both registered variable methods the parent's nodes are at fractions 0,
1/2, 1 (`interpolatory3_variable`) or 0, 1 (`adaptive_heun`), which are nodes
of the halves at their standard fractions anyway. So reuse for them is the same
as for any method: hand the parent's values at the fractions that coincide with
a node of a piece down to that piece (the "carrier" of
`docs/issues/2026-11-25-speedup-split-waste-plan.md`, section 3.2; deleted
once done, in commit `3033e24`). Counted on
`b43da3d` by wrapping `f` (the points of `f` within 2 ULP of a node of a
subinterval rejected in the previous pass; the plan's 9 integrands on [0, 1],
seed 0, `atol=1e-8`, `rtol=0`, halving; script
`$SCRATCH/padaquad_speedup_plans/split-waste/impl/carrier.py`):

| Method | Share of the points of f a carrier would fill |
|---|---|
| `interpolatory3_variable` | 66.7 % |
| `adaptive_heun` | 50.0 % |
| `fehlberg2`, `bosh3`, `dopri5` | 66.6 %, 62.5 %, 20.0 % |
| `cc5`, `cc9`, `cc17`, `cc33`, `cc65` | 40.0 %, 22.2 %, 11.7 %, 6.0 %, 3.0 % |
| `gk7`, `gk15`, `gk21` (default), `gk31` | 0 |

The Gauss-Kronrod methods gain nothing: the parent's centre node is an end of
both halves, where they have no node. The shares count both halves' copies of
the parent's centre value; with the `endpoint-evaluations` design (b) (one
evaluation per mesh point shared by two waiting subintervals) the two savings
overlap.

## History

1. Runtime check (2026-11-12): wrapping the reuse method with a counter and
   integrating `exp(-200 (t - 0.5)^2)` on [0, 1] with
   `integrate(f, "interpolatory3_variable", sampling="variable", ...)`,
   `atol = rtol = 1e-10`, `device="cpu"`, `max_batch=100_000`,
   `take_gradient=False`, `torch.manual_seed(0)`: 2495 subintervals, about 6
   points of `f` per subinterval (not 3), and 0 calls of the method.
2. Before `da752ea`, `integrate()` called `_adaptively_add_y`, which called
   `_evaluate_adaptive_y` (later renamed `_evaluate_adaptive_nodes`) in
   `torchpathdiffeq/parallel_solver.py`; reuse was live. `da752ea`
   (2025-02-22, "Integration sets (#11)") introduced the mesh-based
   `_adaptively_add_steps`, which splits by adding mesh points, and kept the
   reuse path only in functions nothing called. `3063c37` (2025-03-07)
   deleted those callers; `a7365ae` (2026-03-18) added the method's unit
   tests; `bde5e72` (2026-05-12) renamed it and moved it into
   `padaquad/quadrature/`, and CLAUDE.md described split-reuse as working,
   but the loop was never reconnected.

## What would be needed

A design, not a prescription; confirm each point against the code.

1. **Which values.** For each method, the node fractions of a parent that
   equal a node fraction of a piece (0, 1/2, 1 for most; 3/4 too for `bosh3`;
   none for Gauss-Kronrod), computed once.
2. **Keep them past the pass.** Today a pass's values live in one [K, C, D]
   array, and the rejected rows are dropped in `_adaptively_increase_mesh`
   (`_keep_rows`) and deleted after recording. The kept values must stay on the
   solver's device, also with `trade_speed_for_memory=True` (values move to
   `result_device` as they are integrated) and `record_evaluations=False`
   (values dropped once integrated), in a store keyed by the pieces' mesh
   positions and remapped through `idxs_transfer` like the record's mesh
   indices. Recorded subintervals that fail the re-check
   (`_remove_failed_record_panels`, Step 5b and Step 7) hand theirs from the
   record, which is on `result_device` unless `result_device` is the solver's
   device.
3. **Evaluate only the rest.** `_evaluate_f_on_split_nodes` passes the flat
   list of every waiting subinterval's nodes to `_evaluate_f_in_batches`; it
   would skip the carried nodes and scatter the carried values into the pass's
   array, so a row is no longer C consecutive points of `f`'s output (the
   one-call path with no copy is lost for such passes).
4. **Bit equality.** A parent's node at fraction 1/2 is `torch.lerp(left,
   right, 0.5)` (`_compute_nodes`), the halves' end is `_midpoint` =
   `0.5 * (left + right)`; they can differ in the last bit. In the count above
   every match was exact except 45 to 64 of `bosh3`'s. Decide whether a value
   one ULP away is acceptable or compute the shared nodes from the mesh points.
5. **Gradients (`take_gradient=True`).** Each batch's backward frees its
   graph, so a carried value would lose its gradient path unless the parent's
   graph is kept until the piece is accepted (more memory). The simplest
   choice: carry only with `take_gradient=False`.
6. **Tracked variables** travel with the values, so `result.tracked_variables`
   stays aligned with `result.nodes`.
7. **Memory.** Carried values add at most a few D values per split; with
   `trade_speed_for_memory` they are on the solver's device, which the batch
   sizing does not count.

## Tests to add

- Evaluation count per method against the table above, and the same integral
  (to the tolerance) and mesh as without the carrier.
- `take_gradient=True` (no carrier, or exact gradients against the closed form
  as in `tests/test_autodiff_consistency.py`); tracked variables; `D > 1`;
  `max_batch` below one subinterval's nodes and the out-of-memory retry
  (`tests/test_oom_recovery.py`); `trade_speed_for_memory=True`,
  `record_evaluations=False`; the re-check and Step 7.

## Documentation to update when fixed

These say that reuse does not happen; change them:

- `README.md`, "Available methods" ("The integration loop currently places
  every subinterval's nodes…").
- `docs/glossary.md`, "uniform sampling" and "variable sampling".
- `docs/user_guide/methods.md`, "How to choose".
- `CLAUDE.md`, "Uniform vs variable sampling".
- The module docstring of `padaquad/quadrature/variable.py` and the docstring
  of `VariableAdaptiveQuadrature`.
- `tests/test_variable_integration.py` module docstring.
- `CHANGELOG.md`: a "Changed" entry with the counts.

## References

- `sanderse2019` in `docs/references.bib`, Eq. (119): the weights of
  `interpolatory3_variable` for nodes 0, a, 1.
