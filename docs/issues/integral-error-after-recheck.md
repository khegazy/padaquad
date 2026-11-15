# `integral_error` is wrong after the re-check removes subintervals

Status: open. Found 2026-11-12 on branch `website_docs_update` (base `main` at
95936ce). Written so that another agent can pick it up without the original
session.

## Summary

`IntegrationResult.integral_error` is meant to be the sum of the accepted
subintervals' **signed** error estimates. With `take_gradient=False`, padaquad
re-checks accepted subintervals as the running integral changes and splits again
those that now fail (`_remove_failed_record_panels` in
`padaquad/quadrature/base.py`). When it removes them, it subtracts the
**absolute values** of their error estimates from the signed total, because the
record stores absolute values. After any removal, `integral_error` is no longer
the signed sum over the subintervals that remain, and it can be wrong by a large
factor. The integral itself is not affected (its update is exact).

## Reproduction

Script (run from the repository root with the project's Python):

```python
import math
import torch
from padaquad import UNIFORM_METHODS, integrate

f = lambda t: torch.sin(t) + 1e-3
lo = torch.tensor([0.0], dtype=torch.float64)
hi = torch.tensor([2 * math.pi], dtype=torch.float64)
method = UNIFORM_METHODS["dopri5"]

torch.manual_seed(0)
r = integrate(f, "dopri5", atol=1e-9, rtol=1e-3, mesh_init=lo, mesh_final=hi,
              device="cpu", max_batch=14, take_gradient=False,
              conserve_memory=True)

# Recompute each kept subinterval's signed error estimate from its endpoints.
h = r.h.reshape(-1)
left = lo + torch.cumsum(h, 0) - h
c, b_err = method.tableau.c, method.tableau.b_error
signed = torch.stack([hh * (b_err * f((l + c * hh)[:, None]).reshape(-1)).sum()
                      for l, hh in zip(left, h)])
print(float(r.integral_error), float(signed.sum()))
```

Result on 2026-11-12 (CPU, float64): the re-check removed 2 subintervals;
`integral_error` was **-2.054897e-04**, while the signed sum over the kept
subintervals is **-3.436877e-06** (about 60 times smaller). The same integral
with `take_gradient=True`, where no re-check runs, gives an `integral_error`
equal to the signed sum. `conserve_memory=True` is needed here only to make the
loop evaluate one small batch per pass, so that the running integral changes
between passes and the re-check removes something.

Why removals happen in this example: in absolute mode the tolerance is
`max(atol, rtol * |running integral|)`. The first batches cover the positive half
of `sin`, so the running integral is large and the tolerance loose; when the
negative half is added, the total shrinks toward `2π · 1e-3`, the tolerance
tightens, and early subintervals fail the re-check.

## Cause

In `AdaptiveQuadrature.integrate()` (`padaquad/quadrature/base.py`):

- Each batch's accepted subintervals are recorded through an `IntegrationResult`
  built with `mesh_quadrature_errors=torch.abs(method_output.mesh_quadrature_errors)`
  (around line 1040), while `integral_error=method_output.integral_error` is the
  batch's signed sum. So the record holds absolute per-subinterval errors and a
  signed total.
- `_remove_failed_record_panels` (around line 3525) then does
  `record["integral_error"] -= err[removed].sum(0)` with `err` the absolute
  values. Its docstring already says the result "is an approximation".
- `_adaptively_increase_mesh` (around line 1550) recomputes
  `method_output.integral_error = torch.sum(method_output.mesh_quadrature_errors, 0)`
  after dropping rejected subintervals from a batch. This sum is of signed
  values (checked): `_calculate_integral` returns signed errors, and the
  `torch.abs` above is applied later.

## Proposed fix

1. Record the **signed** per-subinterval error estimates: give
   `_record_results` the signed `method_output.mesh_quadrature_errors`. Do not
   drop the `torch.abs` from the per-batch `IntegrationResult` itself (around
   line 1040): that object is also passed to `loss_fxn(intermediate_results)`,
   and its `mesh_quadrature_errors` is documented as absolute values
   (`padaquad/results.py`), so user loss functions must keep receiving
   absolute values.
2. In `_remove_failed_record_panels`, subtract the signed sum of the removed
   subintervals; `integral_error` then equals the signed sum over the kept
   subintervals exactly, as `integral` does.
3. Take absolute values where they are needed, so no other behavior changes:
   - the per-batch result passed to `loss_fxn` keeps its `torch.abs` (step 1);
   - the result field `mesh_quadrature_errors` is documented as absolute values
     and is built with `torch.abs(record["mesh_quadrature_errors"])` (around
     line 1181) — keep that;
   - the not-converged warning already takes `torch.abs` (around line 1285);
   - the error ratios take `torch.abs` (`_compute_error_ratios`, around line
     2906);
   - **the optimal mesh**: `_get_optimal_mesh` → `_prune_excess_mesh` →
     `_merge_excess_nodes` sums the two merged subintervals' errors. With signed
     values those sums could cancel and change which pairs merge. Pass
     `torch.abs(...)` into `_get_optimal_mesh` (around line 1130) so the merge
     decisions stay bit-identical.
4. Alternatively, keep absolute values in the record and also keep the signed
   values in a second record key; that costs memory for every subinterval and
   duplicates data, so (1)-(3) is preferred.

## Tests

- A regression test from the reproduction above: after a run with removals
  (assert that `_remove_failed_record_panels` removed at least one subinterval,
  for example with a counting wrapper), `integral_error` equals the signed sum
  of the kept subintervals' estimates recomputed from `result.h` and the
  method's `b_error`, to float64 rounding.
- The snapshot data (`tests/test_snapshots_data.json`) does not change and
  does not guard this bug: an independent review (2026-11-14) ran every
  snapshot cell through a patched copy that applies steps 1-3, and no
  `take_gradient=False` cell has a re-check removal, so no `integral_error`
  changed. The regression test above is the only guard. `tests/test_snapshots.py`
  must still pass unchanged, which checks that `integral` and `n_optimal_mesh`
  did not move.
- In the same patched copy, the reproduction gave `integral_error`
  -3.4368769904242198e-06 against a signed sum of -3.4368769904242333e-06, and
  `integral` and `mesh_optimal` were identical to the unpatched code for all
  four combinations of `take_gradient` and `conserve_memory`.

## Documentation to update when fixed

These now describe the approximation; change them:

- `IntegrationResult.integral_error` docstring (`padaquad/results.py`).
- `_remove_failed_record_panels` docstring.
- `docs/user_guide/troubleshooting.md`, "The error estimate seems unreliable".
- `CHANGELOG.md`: the [Unreleased] "Fixed" entry that documents the
  approximation; replace it with the fix.
