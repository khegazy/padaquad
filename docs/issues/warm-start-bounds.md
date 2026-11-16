# `same_f=True` after a call with only `mesh=` integrates the default interval

Status: fixed on branch `fix_warm_start` (option 2 below), not yet merged. Found
2026-11-15 by the code review of `merge_rule="predicted"` (PR #69). The bug is
identical on `main` at 2c0c955 and at 51269c9 (1.4.1). Line numbers below are on
`main` 51269c9 unless stated otherwise.

## Summary and recommendation

A call that gives its interval only through `mesh=` is integrated correctly. A
following call with `same_f=True` and no bounds used to integrate the solver's
default interval: [0, 1], unless the constructor set `mesh_init` / `mesh_final`.
It gave no warning and returned `converged=True`.

The fix (option 2) changes one rule. Under `same_f=True`, a call that passes no
`mesh` takes any bound it does not give from the stored mesh: `mesh_previous[0]`
for `mesh_init` and `mesh_previous[-1]` for `mesh_final`. Those are the bounds of
the last call that returned. A call without bounds therefore repeats the previous
interval and reuses every stored point. Nothing else changes:

- bounds that are given still cut the stored mesh;
- a `mesh=` still sets the bounds;
- a first call (nothing stored) still uses the constructor's bounds.

No new state is stored. The decisions the maintainer may want to revisit are
Q2 (stored bounds win over constructor bounds set on purpose) and Q1b (the
exception type).

## The problem

Reproduction (`/pscratch/sd/k/khegazy/u028/verify3/scripts/warm_bounds.py`): a
cold call with `mesh=linspace(lo, hi, 9)`, then `same_f=True` without bounds, on
a `gk21` solver with `max_batch=10_000`. Output on `main` 51269c9:

```text
mesh on [2.0, 3.0]: exact 12.696481, cold 12.696481, warm 1.718282 on [0.0, 1.0], warm converged=True
  stored mesh: [2.0, 2.25, 2.5, 2.75, 3.0]
  warm mesh:   [0.0, 1.0]
mesh on [0.5, 1.5]: exact 2.832968, cold 2.832968, warm 1.718282 on [0.0, 1.0], warm converged=True
  stored mesh: [0.5, 0.75, 1.0, 1.25, 1.5]
  warm mesh:   [0.0, 0.75, 1.0]
```

1.718282 is e − 1, the integral of e^t over [0, 1]. In the second case the warm
mesh mixes the default bounds with part of the stored mesh. The not-converged
warning advised "call again with same_f=True to start from mesh_optimal"
(`padaquad/quadrature/base.py:1376`). After a call that passed only `mesh=`,
following that advice led straight into the bug.

## Cause

1. `integrate` resolves the bounds in `_setup_integral_bounds`
   (`padaquad/quadrature/base.py:2878`). With a mesh, they are its ends.
   Without one, a missing bound came from the constructor's `init_mesh_init` /
   `init_mesh_final` (`padaquad/base.py:210-219`), which default to [0] and [1].
2. With `same_f=True`, `_setup_initial_mesh` (`quadrature/base.py:2935`) keeps
   only the stored points inside the bounds (line 2987). It then adds the bounds
   if they are missing. A stored mesh on [2, 3] has no point in [0, 1], so the
   warm call started from a fresh mesh on [0, 1].
3. The stored mesh is written by every call that returns
   (`self.mesh_previous = mesh_optimal`, line 1203). No bounds are stored
   separately.

Each rule was documented on its own. Together they gave an integral over an
interval the caller never asked for.

## Options compared

1. **Warn** when a warm call's bounds were not given and stored points are
   discarded. The answer is still the integral over the wrong interval; a
   warning in a training loop is easily missed. It also does not make the
   not-converged remedy correct.
2. **Take a missing bound from the stored mesh** (chosen). The natural call,
   `same_f=True` and nothing else, does what the flag says: the same problem
   again, from the stored mesh. Every call site in this repository keeps its
   behaviour (Q9). Cost: about ten lines in one function. This option was
   proposed by the previous agent; it is not from a published source.
3. **Raise** in that situation and ask for the bounds. This is safe, but it
   breaks the remedy flow (the warning's advice would raise), and it rejects the
   call whose meaning is clearest.

Option 2 does not need option 1's warning in addition; see Q10.

## Design questions

**Q1. One bound given.** The other bound comes from the stored mesh, not the
constructor. Taking it from the constructor would recreate the mixing of the
second reproduction case. A given bound outside the stored range extends or cuts
the stored mesh, as explicit bounds always have. If the result is not
increasing (for example `mesh_init=4` after a call on [2, 3]), the call raises
`ValueError`, and the message says where the missing bound came from. Tests:
`test_warm_start_one_bound_takes_the_other_from_stored_mesh` ([2, 3] → [2, 3.5]
→ [2.5, 3.5]) and `test_warm_start_given_bound_beyond_stored_mesh_raises`.

**Q1b. `assert` or `ValueError`** (maintainer decision). The bound-order check
was an `assert`, twice: once in the mesh branch and once at the end of
`_setup_integral_bounds`. `python -O` removes asserts, and a call with
reversed bounds then runs on. On `main` under `-O`:

- A cold call on [1, 0] returns the signed integral (−1.718). It stores an
  out-of-order `mesh_optimal` with points outside the interval:
  `[1.0, 1.212, 1.227, 0.368, …]`.
- The call `same_f=True, mesh_init=4` after a call on [2, 3] uses the bounds
  [4, 1], because `main` takes the missing upper bound from the constructor. It
  returns −51.88, which is e − e⁴, the correct signed integral over [4, 1], with
  the mesh `[4.0, 2.0, 2.5, 3.0, 1.0]`.

So under `-O` the harm is a stored mesh that is out of order, not a wrong
returned value in these probes. The final reviewer corrected this; an earlier
version of this page compared −51.88 with the integral over [4, 3]. Probes:
`final_review/rev_O.py` and `rev_O2.py`, both under
`/pscratch/sd/k/khegazy/warm_bounds/`. The fixed tree raises `ValueError` in both
cases. The fix keeps one check, as a `ValueError`. No test expected
`AssertionError` from it. For T > 1, the comparison still raises `RuntimeError`
("Boolean value of Tensor with more than one value is ambiguous") at the same
place, as before.

**Q2. Constructor bounds set on purpose.** The stored bounds win, whether the
constructor's bounds were given or defaulted. Reasons:

- The two choices differ only when some call overrode the constructor's
  interval. Then the most recent call is the better guess for "the same
  problem".
- If given constructor bounds won instead, the exact bug would remain for users
  who set them: constructor [0, 1], call `mesh=` on [2, 3], warm call without
  bounds discards every stored point.
- `same_f` already lets the stored `max_batch` win over the caller's and the
  constructor's (`quadrature/base.py`, `force_max_batch`).
- Telling given from defaulted bounds would need new state, or a `None` sentinel
  in `init_mesh_init` that `_set_dtype`, two tests and the "None means 0"
  docstrings would have to handle.

The mirror case weighs against this choice. Suppose the constructor is given
[0, 1] on purpose, and training calls use `same_f=True` without bounds. One
`same_f=False` call, for example a validation call, then runs on [0, 2]. The next
training call integrates [0, 2] and returns 6.389 with `converged=True` and no
message; on `main` it integrated [0, 1] and returned 1.718
(`/pscratch/sd/k/khegazy/warm_bounds/final_review/validation_call.py`, found by
the final reviewer). The caller marked the validation call as a different
integrand, so "the most recent call is the better guess" is weakest here. The
user guide and the CHANGELOG state this behaviour. A caller who wants fixed
bounds should pass them on every call, as the README and the user-guide
examples do. Choosing between the two cases is the maintainer's decision.

This argument is the author's and the reviewers' reasoning, not from a source.
Test: `test_warm_start_without_bounds_repeats_previous_interval[...-ctor_given]`.

**Q3. Are `mesh_previous[0]` and `[-1]` exactly the previous bounds?** Yes, bit
for bit, on CPU in 90 of 90 probe cells: every method in both registries,
float64 and float32, with a fresh mesh, an over-resolved mesh (merges), and a
`max_adaptive_splits=1` call that does not converge
(`/pscratch/sd/k/khegazy/warm_bounds/endpoints_new_main.log`, re-run on 51269c9).
The reasons in the code:

- `mesh_optimal` is `[nodes_pruned[:, 0, :], mesh[-1]]` (line 2807).
- `_compute_nodes` places nodes with `torch.lerp` (line 535), whose end nodes are
  the mesh points exactly.
- The uniform merge re-places nodes from the pair's outer ends; the variable
  merge keeps node 0.
- Splitting only inserts midpoints.
- Device moves copy bits.

On CUDA, with `result_device="cpu"` and `None`, the property is checked by the new
test `test_mesh_optimal_ends_are_the_bounds` (120 CUDA cells; results below).
Under option 2 the warm call's bounds are the stored mesh's own ends, so the cut
in `_setup_initial_mesh` keeps every stored point even if Q3 failed. Q3 only
decides whether "the previous interval" is literally true. After a dtype switch,
the reused bound is the stored value rounded to the new dtype (see Risks).

**Q4. Where the change belongs.** In `_setup_integral_bounds`, which gains a
`same_f` argument. The bound fallback in `SolverBase._check_variables`
(`padaquad/base.py:433`) was never reached from `integrate`: `integrate` calls it
after `_setup_integral_bounds`, which never returns `None`. After the fix, the two
copies would have disagreed. A separate commit (5a92e70) removes the bounds from
`_check_variables`, which now returns `(f, y0)`. It can be dropped on its own.

**Q5. `mesh=` with `same_f=True`.** Unchanged. The bounds are the mesh's ends,
and the stored mesh is cut to them. `test_same_f_prefers_stored_mesh_and_max_batch`
uses one interval throughout, so it did not pin this. The new
`test_same_f_with_mesh_takes_bounds_from_mesh` does (stored [2, 3], call on
[2.5, 4]).

**Q6. T > 1.** No change. `integrate` with T > 1 still stops at the bound-order
check (see `docs/issues/multi-coordinate-t.md`), so no T > 1 call stores a mesh.
`mesh_previous[0]` has shape [T], like the constructor's bounds.

**Q7. First call with `same_f=True`.** Unchanged: with nothing stored, the
constructor's bounds (or [0, 1]) are used.
`test_first_same_f_call_uses_constructor_bounds` passes on `main` and on the fix.

**Q8. The remedy text.** After the fix, "call again with `same_f=True`" points at
the right interval for every way a solver can be called. It did nothing for the
one-shot `padaquad.integrate()`: that function builds a new solver per call
(`padaquad/_frontend.py:128`), so `same_f=True` starts fresh there. The remedy
now reads "call the same solver again with same_f=True, or pass
mesh=mesh_optimal, to start from mesh_optimal". The second half works for both;
it was suggested by the reviewer. In cumulative mode the remedy may still fail
to refine, because a failing subinterval can be merged. That is unrelated and
predates this change (CLAUDE.md, warm-start section).

**Q9. Compatibility.** Every `same_f=True` call in `tests/`, `examples/`,
`docs/`, `README.md` and `padaquad/` on `main` was classified (table in
`/pscratch/sd/k/khegazy/warm_bounds/plan_review.md`, finding 13, re-checked on
51269c9):

- Each call either passes bounds, or passes `mesh=`, or follows a call on
  [0, 1].
- No call site warm-starts without bounds after a call on another interval, so
  none changes behaviour.
- Three places passed bounds only to work around this bug. They now pass none:
  `test_capped_call_then_warm_starts_converge`, the helper `_warm_errors`, and
  `test_predicted_far_from_the_origin` (both in `tests/test_prune_and_optimal.py`).
  `_warm_errors`'s docstring said so explicitly.

The downstream user `pode` was checked only in a local copy
(`/pscratch/sd/k/khegazy/u028/padaquad-pinn`, commit 9d52074 of 2026-11-11). It
never passes `same_f`, `reuse_mesh` or `mesh_previous`, and it always passes
both bounds (`src/pode/integrators/padaquad_policy.py:442-443`). The working
copy on OLCF Frontier could not be read from this machine.

**Q10. A warning when explicit bounds discard every stored point?** No. Explicit
bounds are the caller's choice. Disjoint windows ([0, 1], then [2, 3]) would warn
on every call. Adjacent windows ([0, 1], then [1, 2]) keep the shared point, so
they would not trigger an "every point" warning, but would trigger any looser
one.

## Behaviour after the fix

"Stored" means `same_f=True` and `solver.mesh_previous is not None`. A
`same_f=False` call counts as "not stored" here, but it still updates the stored
mesh, so its interval is the next bound-less warm call's interval.

| Call has `mesh` | Stored | Bounds in the call | Constructor bounds | Bounds used | Starting mesh |
|---|---|---|---|---|---|
| yes | no | any (must match the mesh) | any | `mesh[0]`, `mesh[-1]` | the caller's mesh |
| yes | yes | any (must match the mesh) | any | `mesh[0]`, `mesh[-1]` | stored mesh cut to the bounds, padded |
| no | no | both | any | given | fresh |
| no | no | one or none | given | missing ones from the constructor | fresh |
| no | no | one or none | defaulted | missing ones are 0 / 1 | fresh |
| no | yes | both | any | given | stored mesh cut to the bounds, padded |
| no | yes | one | any | **changed:** the other from the stored mesh | stored mesh cut, padded |
| no | yes | none | any | **changed:** `mesh_previous[0]`, `mesh_previous[-1]` | the whole stored mesh |
| any | any | resolved bounds not increasing | any | **changed:** `ValueError` (was `AssertionError`) | — |

Notes:

- Given bounds pass through `set_dtype_by_input` (`quadrature/base.py:775`) and can
  switch the solver's dtype; omitted bounds never do.
- Explicit bounds equal to the stored ends give the same starting mesh, optimal
  mesh and integral as omitted bounds (reviewer's probe,
  `review/plan_tests_fix.log`).
- `result.mesh_init` / `result.mesh_final` report the bounds used; the new tests
  assert them.

## Changes on `fix_warm_start`

| Commit | Content |
|---|---|
| e5b3632 | `test_mesh_optimal_ends_are_the_bounds` (Q3): 60 CPU cells, 120 CUDA cells. |
| 0de21da | The fix in `_setup_integral_bounds`; one `ValueError`; the remedy text; six new regression tests in `tests/test_bug_regressions.py`; three workarounds removed. |
| 5a92e70 | `_check_variables` fills in `f` and `y0` only (Q4); droppable. |
| 49116d1 | Docs: `integrate` docstring (bounds, `same_f`, Raises), constructor bound entries, `SolverBase` class docstring, `docs/user_guide/repeated_calls.md`, CLAUDE.md, CHANGELOG `[Unreleased]`. |
| 1517b9b | Final-review fixes: a `same_f=False` assertion in the reproduction test (it fails when the `same_f` condition is dropped from the fallback), the `python -O` rationale corrected, the mirror case of Q2 in the CHANGELOG, wording in the error message, `SolverBase` docstring and `multi-coordinate-t.md`. |

The README needed no change: its training-loop sketch passes bounds.

Test-first evidence (`/pscratch/sd/k/khegazy/warm_bounds/tests_before_fix.log`).
Before the fix, the new and edited tests ran on the login node, on CPU, with the
memory benchmark stubbed: 17 failed and 19 passed. The failures were the
reproduction (8 cases), one bound given (2), the `ValueError` test, the float32
test, the capped warm-start test, and `test_predicted_far_from_the_origin` (4).
The passes were the tests that pin unchanged behaviour and the untouched
`_warm_errors` callers. After the fix, all 36 passed. The three edited test files
as a whole: 508 passed, 12 skipped.

## Results

Full suite (`tests/` and the docstring examples) on a Perlmutter GPU node,
2026-11-15. Four runs went in parallel: each tree on CPU (the CI-matched
Python 3.12 venv) and on one A100 (the popcornn environment). The trees were
`main` 51269c9 and `fix_warm_start` at 49116d1 plus this page. Logs are in
`/pscratch/sd/k/khegazy/warm_bounds/`.

| Tree | CPU | GPU |
|---|---|---|
| `main` 51269c9 | 3628 passed, 134 skipped, 0 failed | 3640 passed, 122 skipped, 0 failed |
| `fix_warm_start` 49116d1 | 3704 passed, 253 skipped, 0 failed | 3835 passed, 122 skipped, 0 failed |
| `fix_warm_start` 1517b9b (after the final review) | 3704 passed, 253 skipped, 0 failed | 3835 passed, 122 skipped, 0 failed |

- **Collected tests.** The trees differ only by the 195 new test IDs (3762 →
  3957): 180 cells of `test_mesh_optimal_ends_are_the_bounds` and 15 new
  regression cases.
- **GPU.** The 195 new tests all pass.
- **CPU.** The 120 CUDA cells skip, so 75 new tests should pass. The observed
  change is +76 passed and +119 skipped: one existing test skipped on `main`
  and ran on the fix. Most likely it is one of the `test_evaluate_f_on_mesh.py`
  skips that depend on live free memory, which varies with four suites running
  at once. This is a conjecture; the test was not identified.
- **Docs.** The build with `-W` (Sphinx, including the example gallery)
  succeeded: `docs_fix.log`, `exit=0`.
- **CUDA tests on two GPUs.** The new end-point and warm-start tests passed:
  217 passed (`cuda_checks.log`). `tests/test_base_solver.py` with two GPUs
  visible, including the tests that need two GPUs: 39 passed (`two_gpu.log`).

## Risks

- **Dtype switches.** A float32 bound given after a float64 call switches the
  solver to float32, so the reused stored bound is the float32 value of the old
  one (`test_warm_start_one_bound_float32`). The other direction reuses
  `float64(float32(0.1)) = 0.10000000149011612`.
- **Aliasing.** The reused bound is a view into `mesh_previous`, which is the same
  object as the previous result's `mesh_optimal`. An in-place edit of either by
  the caller changes the other. The same hazard already existed for
  `mesh_optimal`.
- **A `same_f=False` call on another interval moves later warm calls** (Q2,
  mirror case): bound-less `same_f=True` calls after it integrate its interval,
  not the constructor's.
- **Callers relying on the old fallback.** That is, a bound-less warm call meant
  to integrate the constructor's interval after a call on another interval. None
  was found in this repository or in the local `pode` copy.
- **Exception type.** Reversed bounds now raise `ValueError` instead of
  `AssertionError`; code that caught `AssertionError` would miss it.
- **Device moves.** `set_device` after construction leaves `mesh_previous` on
  the old device. `_setup_integral_bounds` moves the reused bound, but the cut in
  `_setup_initial_mesh` would then compare tensors on two devices. This is
  reasoned from the code, not run. It predates the fix and is not made worse.

## Follow-ups (outside this fix)

- The other checks on user input in `_setup_integral_bounds` are still
  `assert`s: the mesh's number of dimensions, and that a given bound matches
  `mesh[0]` / `mesh[-1]`. The reason for the `ValueError` (Q1b) applies to them
  too. This was found by the final reviewer.

## Not checked

- The `pode` working copy on Frontier.
- Multi-process (SLURM) paths in `distributed.py`. In the full GPU suites each
  suite saw one GPU; the two-GPU device tests were run separately (Results).

## Confidence

- **Sure:**
  - the cause and the code path;
  - that the fix gives the right integrals and bounds in every case tested (the
    reproduction, one bound, constructor bounds given, `mesh=`, the first call,
    float32, off the origin in float32 and float64, the capped remedy loop);
  - the bit-exact end points on CPU;
  - the `python -O` behaviour on `main` (Q1b);
  - the classification of the call sites in this repository.
- **Reasoned, not measured:** that taking the stored bounds is the better default
  for users (Q2); that no outside user depends on the old fallback.
- **Not measured:** see "Not checked".
