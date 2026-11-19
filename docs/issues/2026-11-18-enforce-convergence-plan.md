# `enforce_convergence` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `integrate` argument `enforce_convergence`. It controls the existing re-check of accepted subintervals and extends it to runs with a fixed `error_integral_reference`, so that `converged` is True whenever the method can reach the tolerance, without evaluating any subinterval twice.

**Architecture:** The re-check already exists (Step 4b of `AdaptiveQuadrature.integrate`, PR #47). On every pass, it judges each recorded subinterval against the same reference integral as the new batch, using the stored integrals and error estimates (no call of `f`). It splits the failures in the same mesh update (`extra_split_idxs`). This plan has two parts:
- **The flag.** `enforce_convergence` replaces the gate's hard-coded `not take_gradient`.
- **The switch-over.** With a fixed reference, once no subinterval is left to evaluate, the reference is dropped.
  - The record is judged once against its own integral, which is complete at that point, from the stored values.
  - The failures are split directly, in a mesh update with no new batch, and removed from the record.
  - The next pass evaluates only their halves, judged together with the whole record against the running integral.

**Tech Stack:** Python 3.11+, PyTorch; pytest; mypy (strict, CI-enforced); ruff; numpydoc; Sphinx.

**Code level:** research-grade (kept and shipped).

**Spec:** the decisions below, confirmed by the user on 2026-11-18 and 2026-11-19 (no separate spec file).

## Design decisions (the spec)

1. **The flag.** `integrate(..., enforce_convergence: bool | None = None)` is per call only, like `take_gradient`, on which its default depends.
   - **None** resolves to `not take_gradient`.
   - **True** turns the re-check on.
   - **False** turns it off, including with `take_gradient=False`.
2. **True with `take_gradient=True` raises `ValueError`**, before `f` is called. Gradients are never undone.
3. **No re-check of any kind with `take_gradient=True`.** This follows from 1 and 2.
4. **Fixed `error_integral_reference` (absolute mode) with enforcement on.**
   - The loop runs against the fixed reference as today.
   - When no subinterval is left to evaluate, set `error_integral_reference = None` and `reference_pinned = False`.
   - Judge every recorded subinterval against `record["integral"]`, from the stored values.
   - Split the failures that are below `max_adaptive_splits` directly in the mesh, and remove them from the record.
   - The loop continues as it would without a reference, with the Step 4b re-check on, until it converges.
   - If nothing fails, nothing changes, and the result is bit-identical to `enforce_convergence=False`.
5. **No subinterval is evaluated twice** (user, 2026-11-19: "I do not want to spare code by running new evaluations. There will be times where f is very expensive to evaluate.") This rules out a smaller design that marks the failures for evaluation again instead of splitting them.
   - The review measured that design at about 8 lines instead of about 45.
   - It costs one extra evaluation of each failing subinterval, plus one extra pass of the loop.
   - On `sin(2πt)` over [0, 1] (`bosh3`, reference 1.0) it made 2192 subinterval evaluations instead of 2079, and 113 subintervals were evaluated twice.

## Global Constraints

- **Error message.** The `ValueError` text, exactly: `"enforce_convergence=True cannot be used with take_gradient=True: each batch's gradient is backpropagated as soon as the batch is accepted, so an accepted subinterval that later fails its tolerance cannot be split again without counting its gradient twice. Pass take_gradient=False and differentiate result.integral, or leave enforce_convergence as None."`
- **Type check.** A value that is neither None nor a Python `bool` raises `TypeError` naming the argument. The check is `isinstance(value, bool)`, so `np.bool_` is rejected too.
- **No evaluation without new information.** Never evaluate `f` on a subinterval whose values are already stored. Decisions about recorded subintervals use the stored `mesh_quadratures` and `mesh_quadrature_errors`.
- **When results change.** Snapshots must NOT change: `tests/test_snapshots.py` passes without `TPD_REGENERATE_SNAPSHOTS`. With the defaults, a result changes exactly when all of these hold:
  - `take_gradient=False`;
  - a fixed `error_integral_reference` is given, in absolute mode;
  - at the switch-over, at least one recorded subinterval fails its tolerance against the final integral.

  That needs a reference larger in magnitude than the final integral. Even a reference equal to the true integral can trigger it, when the computed integral is slightly smaller.
- **Vocabulary.** Public docstrings and user pages say "subinterval", never "panel" or "step" (`docs/glossary.md`, checked by `tests/test_docs_vocabulary.py`). Log messages keep their existing wording ("steps").
- **Checks.** Every new def is fully annotated (`mypy padaquad/` clean). `ruff check padaquad/` and `numpydoc lint` must be clean.
- **Where tests run.** Use the `popcornn` interpreter, `PY=~/.conda/envs/popcornn/bin/python`. Targeted test files may run on a login node. The full suite runs only on a compute node (salloc, `$SCRATCH` paths, absolute test paths).
- **Seeding.** In every test, call `torch.manual_seed(0)` immediately before EACH run, because the starting mesh is random. Unseeded, two runs differ (the review measured integrals of -4.66e-11 against -8.42e-10).

## Review Focus

1. **No subinterval evaluated twice**, through the switch-over, with and without `conserve_memory` and with a split cap. Expected: every subinterval passed to `f` is distinct. Test: Task 2, `test_switch_never_evaluates_a_subinterval_twice`. Checked on the review's copies on 2026-11-19: 0 repeats with the direct split, and 113 repeats (83 with cap 4) with the rejected design.
2. **A split cap at the switch-over.** Failing subintervals at `max_adaptive_splits` stay in the record and are not split. Expected: the call terminates, `converged=False`, and the warning names `max_adaptive_splits`, not the reference, which has been dropped by then. Test: Task 2, `test_switch_respects_split_cap`. Cap 4 is needed: with cap 2, every subinterval is already at the cap before the switch-over, so nothing is exercised.
3. **A CUDA solver with `result_device="cpu"`.** The record's `nodes` live on the CPU, while the error masks, `split_counts` and the mesh live on the GPU. Expected: the same result as on the CPU. Test: Task 2, `test_too_large_fixed_reference_converges_cuda`.
4. **Vector integrand, `error_norm="failure_fraction"`, vector reference.** Expected: each output component is judged against its own final integral, and the run converges. Test: Task 2, `test_switch_vector_failure_fraction`.
5. **NaN in the integrand** (`error_on_nonfinite=False`). A NaN integral makes every tolerance non-finite, and non-finite subintervals are accepted, never split. Expected: the call terminates with `converged=False`. Test: Task 2, `test_switch_with_nonfinite_terminates`.

A further check, kept as a test though not in the list above: with a reference no larger than the integral, the switch-over must find nothing, and the result must be bit-identical (`test_small_fixed_reference_needs_no_switch`).

---

### Task 1: The `enforce_convergence` argument

**Files:**
- Modify: `padaquad/quadrature/base.py`.
  - The `integrate` signature (line ~674): add the parameter after `error_integral_reference`.
  - The `integrate` docstring:
    - Add a parameter entry.
    - In the Raises section (line ~805), add `TypeError` for a non-bool value and the new `ValueError`.
    - Amend "Every argument left as None falls back to the constructor's value, except the bounds" to also except `enforce_convergence`.
  - The argument checks: after `self._check_max_adaptive_splits(max_adaptive_splits)` at line ~874, not the constructor's call at line 347.
  - The Step 4b gate (line 1091).
  - `_warn_not_converged` (lines 1414-1505): signature, remedies, and the docstrings of its `take_gradient` and `reference_pinned` parameters; also its call (line ~1285).
- Test: `tests/test_recheck_accepted_panels.py` (add `import logging`), `tests/test_solver_methods.py`.

**Interfaces:**
- Produces:
  - `AdaptiveQuadrature.integrate(..., enforce_convergence: bool | None = None)`. Inside `integrate`, after the checks, the local `enforce_convergence` is a `bool`.
  - `_warn_not_converged(self, record, final_ratios, final_fails, nonfinite, max_adaptive_splits, take_gradient, enforce_convergence: bool, reference_pinned)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_recheck_accepted_panels.py`, using the file's `_solver`, `_run` and `_cancelling`.

```python
def test_enforce_convergence_false_turns_the_recheck_off():
    torch.manual_seed(0)
    result, _ = _run(_solver(), _cancelling, take_gradient=False, enforce_convergence=False)
    assert result.converged is False

def test_enforce_convergence_none_and_true_agree():
    torch.manual_seed(0)
    a, _ = _run(_solver(), _cancelling, take_gradient=False)
    torch.manual_seed(0)
    b, _ = _run(_solver(), _cancelling, take_gradient=False, enforce_convergence=True)
    assert torch.equal(a.integral, b.integral)
    assert torch.equal(a.h, b.h)

def test_enforce_convergence_with_take_gradient_raises():
    calls = []
    def f(t, *a):
        calls.append(1)
        return _cancelling(t)
    with pytest.raises(ValueError, match=r"take_gradient=True.*twice"):
        _run(_solver(), f, take_gradient=True, enforce_convergence=True)
    assert calls == []          # raised before any call of f

@pytest.mark.parametrize("value", [1, "yes"])
def test_enforce_convergence_rejects_non_bool(value):
    with pytest.raises(TypeError, match="enforce_convergence"):
        _run(_solver(), _cancelling, take_gradient=False, enforce_convergence=value)

def test_warning_suggests_enforce_convergence(caplog):
    torch.manual_seed(0)
    with caplog.at_level(logging.WARNING, logger="padaquad"):
        _run(_solver(), _cancelling, take_gradient=False, enforce_convergence=False)
    assert "enforce_convergence=True" in caplog.text
```

In `tests/test_solver_methods.py`, add `{"enforce_convergence": False}` to the parameter list of `test_call_options_are_accepted` (line ~295). `_CALL_OPTIONS` reads the signature, so `_frontend.py` needs no change.

- [ ] **Step 2: Run them and see them fail.** Run `$PY -m pytest tests/test_recheck_accepted_panels.py -k enforce -v`. Expected: FAIL with `TypeError: ... unexpected keyword argument 'enforce_convergence'`.

- [ ] **Step 3: Implement.**
  - **Resolve and validate the flag.** If None, it becomes `not take_gradient`. If not a `bool`, raise `TypeError`. If True with `take_gradient`, raise the `ValueError` from Global Constraints.
  - **Gate.** Step 4b becomes `recheck_active = enforce_convergence and len(record) > 0 and not reference_pinned`. Its comment points at the argument instead of "Only when not taking gradients".
  - **Warning remedies.** Keep "raise max_adaptive_splits". Then, when `not enforce_convergence and not reference_pinned`:
    - with `take_gradient`, append the existing "pass take_gradient=False (accepted steps are then re-checked)";
    - otherwise, append "pass enforce_convergence=True".

    The `not reference_pinned` stays until Task 2, because until then a fixed reference still turns the re-check off. The reference remedy is unchanged.
  - **Docstrings.**
    - The `integrate` entry says what None, True and False do, that True with `take_gradient=True` raises, and links `ug-convergence`.
    - The `_warn_not_converged` docstrings of `take_gradient` and `reference_pinned` say the re-check follows `enforce_convergence`.

- [ ] **Step 4: Run them and see them pass.** Run `$PY -m pytest tests/test_recheck_accepted_panels.py tests/test_split_statistics.py tests/test_solver_methods.py -v`. Expected: all PASS. `test_split_statistics.py` needs no change in this task.

- [ ] **Step 5: Commit.** Message: `Add integrate(enforce_convergence=...), default not take_gradient; True with take_gradient=True raises`.

### Task 2: Drop a fixed reference once every subinterval is accepted

**Files:**
- Modify: `padaquad/quadrature/base.py`.
  - Move the body of Step 4b (lines 1106-1122) into `_failed_record_positions`, placed next to `_record_error_check`.
  - Add Step 7 as the last block of the loop body, after the `collect_mesh_grad` block at line ~1242.
  - Update the comment at lines 901-905 and the sentence "Skipped when a fixed error_integral_reference pins the denominator" in the Step 4b comment (line ~1090).
  - In the `error_integral_reference` docstring entry, say the reference is dropped once every subinterval is accepted when `enforce_convergence` is on.
  - In `_warn_not_converged`, drop `and not reference_pinned` from the Task 1 remedy condition.
  - Reword the docstring of `_remove_failed_record_panels` ("the re-check only runs with take_gradient=False") in terms of `enforce_convergence`.
- Modify: `tests/test_split_statistics.py`, two existing tests.
  - `test_converged_uses_the_final_integral_not_the_given_reference` (line 437) is about the definition of `converged`. Add `enforce_convergence=False`.
  - `test_warning_with_a_pinned_reference_suggests_the_reference` (line 551; `take_gradient=True`). `take_gradient=False` is now a real remedy, so replace `assert "take_gradient=False" not in message` with `assert "take_gradient=False" in message`, and update its docstring.
- Test: `tests/test_recheck_accepted_panels.py`.

**Interfaces:**
- Consumes: the local `enforce_convergence: bool` from Task 1.
- Produces: `_failed_record_positions(self, record: dict[str, torch.Tensor], cum_mesh_quadratures: torch.Tensor, integral: torch.Tensor, mesh_indices: dict[tuple[float, ...], int], split_counts: torch.Tensor | None, max_adaptive_splits: int | None) -> tuple[torch.Tensor, torch.Tensor]`.
  - It returns `(split_positions, record_keep_mask)`: the mesh positions of the recorded subintervals that fail and are below the split cap, and the mask that keeps every other recorded subinterval.
  - The body is today's Step 4b lines, unchanged, including `assert split_counts is not None` inside the cap branch.
  - `split_counts` must be typed `torch.Tensor | None`, because the loop declares it Optional (base.py:978). With `torch.Tensor`, mypy fails at the Step 4b call (found by the review).
  - Its two callers are Step 4b and Step 7.

- [ ] **Step 1: Write the failing tests.** All runs use `_run(..., take_gradient=False)`, and each run is seeded immediately before it.
  - Measured with today's code (2026-11-18): `_cancelling` with reference 1.0 gives `converged=False`, 113 subintervals and an integral of -2.4e-7; with reference 0.0, `converged=True`.
  - Measured on the review's copy with this design:
    - Reference 1.0 converges with 1045 subintervals and 2079 subinterval evaluations, none repeated.
    - With `max_adaptive_splits=4`: 83 subintervals with enforcement off and 166 with it on; `converged=False` both times.

```python
def _spy_evaluated(solver):
    """Count each evaluated subinterval, keyed by its first and last node."""
    seen = collections.Counter()
    evaluate = solver._evaluate_f_on_mesh
    def spy(*args, **kwargs):
        out = evaluate(*args, **kwargs)
        nodes = out[0].detach()
        seen.update(zip(nodes[:, 0, 0].tolist(), nodes[:, -1, 0].tolist()))
        return out
    solver._evaluate_f_on_mesh = spy
    return seen

@pytest.mark.parametrize("conserve_memory", [True, False])
@pytest.mark.parametrize("max_adaptive_splits", [None, 4])
def test_switch_never_evaluates_a_subinterval_twice(conserve_memory, max_adaptive_splits):
    solver = _solver()
    seen = _spy_evaluated(solver)
    torch.manual_seed(0)
    _run(solver, _cancelling, take_gradient=False, conserve_memory=conserve_memory,
         error_integral_reference=1.0, max_adaptive_splits=max_adaptive_splits)
    assert len(seen) > 0
    assert max(seen.values()) == 1

@pytest.mark.parametrize("conserve_memory", [True, False])
def test_too_large_fixed_reference_converges(conserve_memory):
    torch.manual_seed(0)
    off, _ = _run(_solver(), _cancelling, take_gradient=False, conserve_memory=conserve_memory,
                  error_integral_reference=1.0, enforce_convergence=False)
    torch.manual_seed(0)
    on, _ = _run(_solver(), _cancelling, take_gradient=False, conserve_memory=conserve_memory,
                 error_integral_reference=1.0)
    assert off.converged is False             # today's behavior, kept by False
    assert on.converged is True
    assert abs(float(on.integral.reshape(-1)[0])) < 1e-5
    assert on.h.numel() > off.h.numel()

@needs_cuda
def test_too_large_fixed_reference_converges_cuda():
    # As above (conserve_memory=True) with a solver built like _solver but
    # device="cuda" (result_device stays "cpu"); compare with the CPU run:
    # converged True, same number of subintervals, integrals within 1e-12.

def test_small_fixed_reference_needs_no_switch():
    # Replaces test_recheck_skipped_with_fixed_reference_still_valid; keep its asserts.
    torch.manual_seed(0)
    a, _ = _run(_solver(), _cancelling, take_gradient=False, error_integral_reference=0.0)
    torch.manual_seed(0)
    b, _ = _run(_solver(), _cancelling, take_gradient=False, error_integral_reference=0.0,
                enforce_convergence=False)
    assert a.converged is True
    assert torch.equal(a.integral, b.integral)
    assert torch.equal(a.h, b.h)

def test_switch_respects_split_cap(caplog):
    torch.manual_seed(0)
    off, _ = _run(_solver(), _cancelling, take_gradient=False, error_integral_reference=1.0,
                  max_adaptive_splits=4, enforce_convergence=False)
    torch.manual_seed(0)
    with caplog.at_level(logging.WARNING, logger="padaquad"):
        on, _ = _run(_solver(), _cancelling, take_gradient=False,
                     error_integral_reference=1.0, max_adaptive_splits=4)
    assert on.converged is False
    assert on.h.numel() > off.h.numel()       # the switch-over split something
    last = caplog.records[-1].getMessage()    # the enforced run's warning
    assert "max_adaptive_splits" in last
    assert "error_integral_reference" not in last

def test_switch_vector_failure_fraction():
    torch.manual_seed(0)
    r, _ = _run(_solver(error_norm="failure_fraction"), _vector_cancelling,
                take_gradient=False,
                error_integral_reference=torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert r.converged is True                # |integral of component 1| = 0.335 < 1
    assert torch.allclose(r.integral, _vector_truth(), atol=1e-5)

def test_switch_with_nonfinite_terminates():
    f = lambda t, *a: torch.where(t > 0.9, torch.nan, _cancelling(t))
    torch.manual_seed(0)
    r, _ = _run(_solver(error_on_nonfinite=False), f, take_gradient=False,
                error_integral_reference=1.0)
    assert r.converged is False
```

Add `import collections` to the test file.

- [ ] **Step 2: Run them and see them fail.** Run `$PY -m pytest tests/test_recheck_accepted_panels.py -k "fixed_reference or switch" -v`. Expected: `test_too_large_fixed_reference_converges`, `test_switch_respects_split_cap` and `test_switch_vector_failure_fraction` FAIL. The others pin behavior that must not change, and may already pass. `test_switch_never_evaluates_a_subinterval_twice` passes today trivially, because nothing is re-checked; it guards Step 4.

- [ ] **Step 3: Extract `_failed_record_positions`.** Step 4b calls it as `extra_split_idxs, record_keep_mask = self._failed_record_positions(record, torch.cumsum(all_mesh_quadratures.detach(), 0)[idxs_keep], reference_integral, mesh_indices, split_counts, max_adaptive_splits)`. Keep Step 4b's explanatory comment, about the merged record and batch cumulative sum, at the call site. Then run `$PY -m pytest tests/test_recheck_accepted_panels.py tests/test_snapshots.py` and `$PY -m mypy padaquad/`, and confirm the extraction changes nothing.

- [ ] **Step 4: Add Step 7** as the last block of the loop body.

```python
# --- Step 7: drop a fixed reference once every subinterval is accepted ---
if enforce_convergence and reference_pinned and not torch.any(mesh_trackers):
    error_integral_reference, reference_pinned = None, False
    split_positions, record_keep_mask = self._failed_record_positions(
        record,
        torch.cumsum(record["mesh_quadratures"].detach(), 0),
        record["integral"].detach(),
        mesh_indices,
        split_counts,
        max_adaptive_splits,
    )
    if split_positions.numel() > 0:
        (_, _, _, _, mesh, mesh_trackers, mesh_indices, _, split_counts) = (
            self._adaptively_increase_mesh(
                method_output=None,
                error_ratios=torch.empty(0, dtype=self.dtype, device=self.device),
                y_step_eval=None,
                nodes=None,
                mesh=mesh,
                mesh_idxs=torch.empty(0, dtype=torch.long, device=self.device),
                mesh_trackers=mesh_trackers,
                split_counts=split_counts,
                max_adaptive_splits=max_adaptive_splits,
                extra_split_idxs=split_positions,
            )
        )
        record = self._remove_failed_record_panels(record, record_keep_mask, loss_fxn)
```

`keep_mask` and `remove_mask` are left out on purpose. With None, `_adaptively_increase_mesh` derives them from the empty `error_ratios` (base.py ~1641). `split_positions` must be on the device of `mesh`, as Step 4b's `extra_split_idxs` already are.

The comment above the block must say four things:
- **No new evaluation.** The record is judged from its stored integrals and error estimates. The failures are split here, and only their halves are evaluated.
- **Why here.** The loop condition is checked next. The halves are evaluated on the next pass and judged together with the whole record against the running integral (Step 4b), so the check of all subintervals happens in the same pass as the split decision.
- **The split cap.** Failing subintervals at `max_adaptive_splits` are not split and stay in the record.
- **What it relies on.**
  - The record is non-empty and in mesh order (asserted in `_record_results`).
  - A reference is fixed only in absolute mode.
  - `split_node_state` holds nothing when every tracker is False (checked by the review in `_evaluate_f_on_split_residual_nodes`).

- [ ] **Step 5: Run them and see them pass.** Run `$PY -m pytest tests/test_recheck_accepted_panels.py tests/test_error_integral_reference.py tests/test_split_statistics.py tests/test_snapshots.py -v` and `$PY -m mypy padaquad/`. Expected: all PASS, and mypy reports no issues.
  - The review's copy with this design passed 108 tests in these files plus 390 snapshot tests.
  - If a step-count comparison in `test_error_integral_reference.py` fails, report the numbers to the user. Do not loosen the assertion.

- [ ] **Step 6: Commit.** Message: `enforce_convergence with a fixed error_integral_reference: drop the reference once every subinterval is accepted and split the failures from their stored errors`.

### Task 3: Documentation and full verification

**Files:**
- Modify:
  - `docs/user_guide/error_control.md`, section `ug-error-reference` (lines ~114-118). The re-check is governed by `enforce_convergence`, and a fixed reference is dropped once every subinterval is accepted.
  - `docs/user_guide/troubleshooting.md`.
    - Lines 14-17: the cause "The tolerance changed after acceptance" names `enforce_convergence`.
    - Line ~65: "With `take_gradient=False`" becomes "With `enforce_convergence`".
  - `README.md` "How it works" step 4 (line ~320): one sentence. With `enforce_convergence` (the default when `take_gradient=False`), accepted subintervals are judged again in every batch, from their stored error estimates.
  - `tests/test_recheck_accepted_panels.py`: the module docstring says the re-check follows `enforce_convergence`, and that `take_gradient=True` is the baseline without the re-check because it turns enforcement off.
  - `tests/test_snapshots.py` line ~63, the docstring of `_case_key`: "runs only on the take_gradient=False path" becomes "is on by default only with take_gradient=False".
  - `CHANGELOG.md`, a new `## [Unreleased]` section.
    - **Added:** `enforce_convergence`.
    - **Changed:** with `take_gradient=False` and a fixed `error_integral_reference`, a subinterval that fails against the final integral is now split, so the result can have more subintervals; pass `enforce_convergence=False` for the previous result.
  - `CLAUDE.md`: the test-table row of `test_recheck_accepted_panels.py`, and one paragraph under "Error computation modes" on the flag and the switch-over.
- No version bump in this plan. Ask the user whether this ships as 2.1.0.

- [ ] **Step 1: Write the pages and docstrings.** Run `$PY -m pytest tests/test_docs_vocabulary.py tests/test_examples_run.py -v`. Expected: PASS.
- [ ] **Step 2: Run the static checks.** `$PY -m mypy padaquad/`, `$PY -m ruff check padaquad/`, and `numpydoc lint $(git ls-files 'padaquad/*.py')`. Expected: no issues.
- [ ] **Step 3: Build the docs.** `sphinx-build -W --keep-going -b html docs docs/_build/html`. Expected: build succeeds with no warnings.
- [ ] **Step 4: Run the full suite on a compute node** (salloc, absolute paths). Expected: no failure beyond those that fail on `main` with the same command. Diff the failure sets; do not compare counts.
- [ ] **Step 5: Commit.** Message: `Document enforce_convergence`.

## Known limitations (not in scope)

- **`integral_error` became approximate** whenever subintervals were removed from the record. Fixed on this branch after the second review (2026-11-19): the record keeps signed estimates; `docs/issues/integral-error-after-recheck.md` was deleted.
- **A custom `loss_fxn` gave a wrong `result.loss`** after removals. Fixed on this branch after the second review (2026-11-19): `_remove_failed_record_panels` subtracts `loss_fxn` of the removed subintervals.

## Review record (2026-11-18, revised 2026-11-19)

An adversarial review applied Tasks 1 and 2 to a scratch copy and ran the tests. Each critique below was checked again before this revision.

| # | Critique | Check | Change |
|---|---|---|---|
| 1 | `_failed_record_positions(split_counts: torch.Tensor)` fails mypy | `split_counts` is declared `torch.Tensor \| None` (base.py:978) | Parameter typed `torch.Tensor \| None`; assert stays inside the cap branch |
| 2 | Test code blocks did not seed; `import logging` missing | Seeding absent from the old blocks; no `import logging` in the file | Seeds in every block; Global Constraint; import added |
| 3 | A smaller switch-over (re-evaluate the failures) gives the same results | Re-ran both copies: same subinterval counts; 113 subintervals evaluated twice | Adopted on 2026-11-18, then rejected on 2026-11-19 by the user (decision 5); a test now forbids repeated evaluations |
| 4 | Cap 2 does not exercise the switch-over | Review's table: 44 = 44 subintervals | Cap 4, and assert more subintervals than with enforcement off |
| 5 | No coverage of `conserve_memory=False`; device mix | The review ran the default path; the device mix is untested | Parametrized `conserve_memory`; CUDA test |
| 6 | Task 1 alone gives a wrong remedy; stale docstrings | Read `_warn_not_converged` | `not reference_pinned` kept until Task 2; docstrings listed |
| 7 | Raises section lacks the new errors | Raises section at line ~805 lists only `ValueError` and `BackwardOutOfMemoryError` | Added to Task 1 |
| 8 | Wording tied to `take_gradient` not in Task 3 | Found in the test module docstring and `test_snapshots.py:63`; README step 4 has no re-check wording | Added; README gets one new sentence |
| 9 | "Results change only for a too-large reference" is imprecise | Agreed | Exact condition in Global Constraints |
| 10 | The comment must say how the placement meets requirement 4 | Agreed | Step 4 comment requirements |
| 11 | Custom `loss_fxn` wrong after removals | Read `_remove_failed_record_panels` | Known limitations |
