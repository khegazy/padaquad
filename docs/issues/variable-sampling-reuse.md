# Variable-sampling solver does not reuse integrand evaluations when it splits

Status: open. Found 2026-11-12 on branch `website_docs_update` (base `main` at
95936ce). Written so that another agent can pick it up without the original
session.

## Summary

The variable-sampling solver (`VariableAdaptiveQuadrature`) exists to save
integrand evaluations when evaluating `f` is expensive: when a subinterval fails
its tolerance and is split in half, the halves should reuse the integrand values
already computed at the parent's nodes and evaluate `f` only at the new nodes.

`integrate()` does not do this. Since commit `da752ea` (2025-02-22,
"Integration sets (#11)"), both solvers split a failing subinterval by inserting
its midpoint into the mesh and then evaluating `f` at fresh nodes in both
halves, discarding the parent's values. The method that implements reuse,
`VariableAdaptiveQuadrature._evaluate_adaptive_nodes`, still exists and is
unit-tested, but nothing in the integration loop calls it.

Consequences:

- The variable solver costs as many evaluations as the uniform solver; the
  feature it exists for is missing.
- Every subinterval's nodes sit at the method's node fractions, so
  `interpolatory3_variable` always has its middle node at the center, where its
  weights are Simpson's rule. Its general position-dependent weights (Sanderse
  and Veldman 2019, Eq. (119)) are never used in a real integration.

## Evidence

1. No caller. `git grep -n "_evaluate_adaptive_nodes(" -- padaquad` finds only
   the two definitions (`padaquad/quadrature/uniform.py`,
   `padaquad/quadrature/variable.py`) and the abstract declaration in
   `padaquad/quadrature/base.py`. The only calls are in
   `tests/test_evaluate_and_merge.py`.
2. Runtime check (2026-11-12): wrapping
   `VariableAdaptiveQuadrature._evaluate_adaptive_nodes` with a counter and
   integrating `exp(-200 (t - 0.5)^2)` on [0, 1] with `interpolatory3_variable`,
   `atol = rtol = 1e-10`, `device="cpu"`, `max_batch=100_000`,
   `take_gradient=False`, `torch.manual_seed(0)`: the result had 2495
   subintervals, `f` was evaluated at 14938 points (about 6 per subinterval, not
   3), and the wrapped method was called 0 times.
3. History (checked with `git log -G` and by parsing old versions):
   - Before `da752ea`, `integrate()` called `_adaptively_add_y`, which called
     `_evaluate_adaptive_y` (the method later renamed `_evaluate_adaptive_nodes`)
     in `torchpathdiffeq/parallel_solver.py`; reuse was live.
   - `da752ea` (2025-02-22, "Integration sets (#11)") introduced the mesh-based
     `_adaptively_add_steps`, which splits by adding mesh points, and kept the
     reuse path only in functions nothing called (`_adaptively_add_steps_orig`,
     `__depricated_adaptively_add_steps`).
   - `3063c37` (2025-03-07, "Cleaning code (#13)") deleted those dead copies.
   - `bde5e72` (2026-05-12, "Quadrature alignment (#27)") re-added
     `_evaluate_adaptive_nodes` on both solvers with unit tests, and CLAUDE.md
     described split-reuse as working, but the loop was never reconnected.

## The existing reuse method

`VariableAdaptiveQuadrature._evaluate_adaptive_nodes(f, idxs_add, y, nodes,
f_args)` (`padaquad/quadrature/variable.py`):

- For each subinterval to split (`idxs_add`, shape `[R]`), it computes the
  midpoints between consecutive nodes, `(nodes[:, 1:] + nodes[:, :-1]) / 2`
  (shape `[R, C-1, T]`), evaluates `f` only there, and interleaves old and new
  values into `2C` slots, duplicating the middle slot as the shared endpoint.
- For `C = 3`, nodes `[p0, p1, p2]` give `[p0, m01, p1, p1, m12, p2]`, reshaped
  into the halves `[p0, m01, p1]` and `[p1, m12, p2]`. For `C = 2`
  (`adaptive_heun`), `[p0, p1]` gives the halves `[p0, m]` and `[m, p1]`.
- The split point is the parent's middle node, which is the midpoint only when
  the node fractions are symmetric about 1/2 (true for both variable methods).
- Cost per split: `C - 1` new evaluations (2 for `interpolatory3_variable`)
  instead of `2C` (6), or `2C - 1` if a shared endpoint were reused.

Known defects of the method as it stands (fix them when reconnecting it):

- `nodes_new` is allocated with last dimension `D` (the integrand's number of
  outputs) instead of `T`. With `D = 2`, `T = 1` it returns nodes of shape
  `(2R, 3, 2)`. The unit tests use `D = T = 1`, so they miss it.
- It returns `nodes` and `y` per half, but not the tracked variables the
  integrand may return (see "Tracked variables" below).
- The interleaving assumes an odd `C` with a middle node, or `C = 2`; it should
  validate `C` or be written generally.

`UniformAdaptiveQuadrature._evaluate_adaptive_nodes` re-evaluates both halves at
the fixed node fractions; it is also unused and can stay as the uniform path or
be removed.

## Where the loop splits today

All in `padaquad/quadrature/base.py`:

- `AdaptiveQuadrature.integrate()` runs the loop.
- `_adaptively_increase_mesh(...)` turns failing subintervals into two by
  inserting midpoints into `mesh` and setting `mesh_trackers` (True = still to
  evaluate); it also maintains `split_counts` (for `max_adaptive_splits` and
  the split statistics). The failed subintervals' `y` values are dropped here.
- `_evaluate_f_on_mesh(...)` evaluates the tracked subintervals. It dispatches
  to `_evaluate_f_on_full_nodes` (`take_gradient=True`),
  `_evaluate_f_on_split_nodes` (default with `take_gradient=False`) and
  `_evaluate_f_on_split_residual_nodes` (`conserve_memory=True`), and places
  nodes with `_compute_nodes(...)` at the node fractions.
- `_record_results(...)` records accepted subintervals in mesh order.
- With `take_gradient=False`, accepted subintervals are re-checked as the
  running integral changes (`_remove_failed_record_panels`) and split again;
  their recorded `y` values are dropped too.

## What has to change

A design, not a prescription; the next agent should confirm each point against
the code.

1. **Keep the values of subintervals that will be split.** When
   `_adaptively_increase_mesh` (or the re-check) splits a subinterval of the
   variable solver, keep its `nodes` `[C, T]` and `y` `[C, D]` (and tracked
   variables) with the two halves, instead of discarding them. A natural place
   is a pending store keyed by mesh position that `_evaluate_f_on_mesh` reads.
2. **Evaluate only the new nodes.** For a half that comes from a split, the
   nodes are the parent's nodes plus midpoints (the layout above); call `f` only
   at the midpoints, inside the existing batching (`_evaluate_f_in_batches`,
   which handles `max_batch` and out-of-memory retries). Fresh subintervals
   (starting mesh, `same_f` warm start) keep using `_compute_nodes`.
3. **Weights from positions.** `VariableMethodClass.weights(nodes)` already
   computes weights from each subinterval's actual nodes, so no method change is
   needed. With bisection of symmetric node fractions the halves' nodes stay
   symmetric (for `interpolatory3_variable`, at 0, 1/2, 1 again); non-central
   positions appear only after merges or non-midpoint splits.
4. **Gradients (`take_gradient=True`).** Today each batch is back-propagated and
   its results detached before the next batch. A reused value from an earlier
   batch has either already been back-propagated through (when its subinterval
   was rejected it contributed no loss, so its graph was never used) or freed.
   Reusing it in a later accepted subinterval needs its autograd graph to stay
   alive until that subinterval is accepted, or the gradient of the reused
   values is lost. Decide whether to (a) keep the graph of pending values (more
   memory, correct gradient), (b) disable reuse with `take_gradient=True`, or
   (c) re-evaluate only the reused values' gradient. Test
   `∇θ ∫ f_θ` against the closed form (as `tests/test_autodiff_consistency.py`
   does) with reuse on.
5. **Memory.** Pending values add `R * C * D` stored values for `R` pending
   splits; account for them in the batch budget (`get_max_batch`,
   `_get_max_f_evals`) or document the bound.
6. **Tracked variables.** If `f` returns `(values, tracked_variables)`, the
   reused nodes must carry their tracked variables too, so
   `result.tracked_variables` stays aligned with `result.nodes`.
7. **Split statistics and caps.** `split_counts`, `max_adaptive_splits`,
   `n_start_steps_split` and `max_split_depth` must count reused splits the same
   way.
8. **Re-check path.** Subintervals removed by the re-check
   (`_remove_failed_record_panels`) should also hand their values to their
   halves.
9. **Optimal mesh.** `_merge_excess_nodes` (variable) merges two subintervals
   by keeping every other node of the `2C - 1` combined nodes; it already works
   on recorded nodes and needs no change, but check that `mesh_optimal` and
   `same_f=True` warm starts still start from fresh nodes.
10. **Uniform solver.** Unaffected: its methods have fixed node fractions. (Methods
    with nodes at both ends could reuse the two endpoint values; that is a
    separate, smaller optimization.)

## Tests to add

- Evaluation count: integrate a peaked integrand with the variable solver and
  count points; each split should cost `C - 1` evaluations. For
  `interpolatory3_variable`, total evaluations ≈ 3 × (starting subintervals) +
  2 × (number of splits). Compare with the uniform solver on the same mesh.
- Accuracy unchanged: `tests/test_variable_integration.py`,
  `examples/verification/integral_accuracy.py`, and the variable cells of
  `tests/test_snapshots.py` (their `integral` should agree to the tolerance;
  `n_optimal_mesh` may change, so regenerate snapshots only for that reason
  and say so).
- Gradients with `take_gradient=True` and `False`, against the closed form.
- `D > 1` integrands (catches the `D`/`T` allocation bug) and tracked
  variables.
- `max_batch` smaller than one subinterval's new nodes, and the
  out-of-memory retry path (`tests/test_oom_recovery.py` has a simulated
  out-of-memory integrand).
- `conserve_memory=True` and `max_adaptive_splits`.

## Documentation to update when fixed

These now say that reuse does not happen; change them back:

- `README.md`, "Available methods" section ("The integration loop currently
  places every subinterval's nodes…").
- `docs/glossary.md`, "variable sampling".
- `docs/user_guide/methods.md`, "How to choose" and "The variable-node method".
- `CLAUDE.md`, "Uniform vs variable sampling".
- Docstrings of `VariableAdaptiveQuadrature` and its
  `_evaluate_adaptive_nodes` (`padaquad/quadrature/variable.py`), and the
  module docstring of `padaquad/quadrature/__init__.py`.
- `tests/test_variable_integration.py` module docstring.
- `CHANGELOG.md`: a "Fixed" entry.

## References

- Sanderse, B. and Veldman, A. E. P. (2019), Constraint-consistent Runge-Kutta
  methods for one-dimensional incompressible multiphase flow, J. Comput. Phys.
  384, 170-199, Eq. (119) (checked in arXiv:1809.06114v2, p. 21): the weights
  of `interpolatory3_variable` for nodes 0, a, 1.
