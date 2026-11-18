# Memory state and a growing batch size: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `integrate` without `max_batch` stops running the memory benchmark: it
makes a few forward test calls, then grows the batch size by factors of 10 until
`f` runs out of memory, and returns a state that the next call takes as
`state=`.

**Architecture:** The batch size of a call lives in two solver attributes,
`_batch` (the largest batch known to fit) and `_batch_converged` (it will not
grow again), set at the start of `integrate` from `max_batch`, a `state`, the
constructor or the test calls. Every call of `f` is sized by one method,
`_batch_for(n_waiting)`. Out-of-memory errors in `f` shrink it (the existing
retry, extended to the `take_gradient=True` path). Before each backward pass the
gradients are saved by walking the batch's autograd graph; an out-of-memory error
in the backward pass restores them and restarts `integrate`.

**Tech stack:** Python 3.11+, PyTorch, pytest.

**Spec:** `docs/issues/2026-11-17-integration-state-design.md`, as amended by the
decisions below. Evidence: `2026-11-17-integration-state-critique.md`.

## Decisions taken in this plan (Kareem delegated them on 2026-11-18)

Kareem: "You do not need to wait for me to make decisions. Use your best
judgement while prioritizing speed, robustness, and ease of use for the user."
Each line answers a critique finding (F numbers).

1. **F1:** fix `conserve_memory=True` first (design section 0).
2. **Result fields (mine, fewer lines and no break):** `IntegrationResult` keeps
   `mesh_optimal` and `max_batch` and gains `max_batch_converged`; `result.state`
   is a property that builds an `IntegrationState` from those three fields. One
   source of truth: the state is not stored a second time. (Design section 4
   proposed removing the two fields; that would rewrite about 120 lines of tests
   and documentation for no gain.)
3. **`IntegrationState`:** frozen dataclass, `eq=False` (a tensor field makes `==`
   raise, F11), fields validated in `__post_init__`.
4. **F2, the check before the backward pass:** runs only on a GPU, only when the
   output of `f` requires a gradient, inside `_evaluate_f_on_full_nodes` right
   after the call of `f` (before the mesh is touched). Rule: usable memory U >=
   r F, where F is this batch's forward peak and r the measured ratio of the
   previous batch's backward peak to its forward peak (1 for the first batch,
   Kareem's rule). Measuring r costs two counter reads and catches activation
   checkpointing (r of 4 to 6) from the second batch on. On failure the batch
   shrinks to min(0.75 N, N U / (r F)) nodes in whole subintervals, and the
   state converges.
5. **F3:** first pass rule (Kareem) and section 7.4 as amended.
6. **F4:** keep the factor 0.75 (Kareem's); document the measured cost.
7. **F5:** document that `total_mem_usage` now limits only the backward check
   (and `get_max_batch(f=...)`); the forward calls of `integrate` grow until an
   error. Docstrings and user guide say so.
8. **F6:** a converged size never grows back; documented, with the one-line
   reset `dataclasses.replace(result.state, max_batch_converged=False)`.
9. **F7, graph walk (Kareem asked for it):** before every backward pass, walk the
   loss's autograd graph and save `.grad` of every leaf seen for the first time
   in the call. `parameters=` (tensors or an optimizer) is needed only when the
   graph contains a custom autograd function (a `BackwardCFunction` node, for
   example reentrant activation checkpointing), which can hide parameters.
   A backward out-of-memory error then restarts `integrate`, unless: a custom
   function is present and no `parameters=` was given; a saved tensor has a
   post-accumulate-grad hook; or `torch.distributed` runs with more than one
   process. In those cases it raises `BackwardOutOfMemoryError` with a ready
   `state`.
10. **F8:** `_is_out_of_memory` returns False for `BackwardOutOfMemoryError`; the
    error defines `__reduce__` so it survives pickling.
11. **F9:** test calls 10, 100, ... up to the order of the first pass (Kareem).
12. **F10:** `_oom_max_batch`, `_cap_max_batch`, `f_mem_coeffs` and
    `_get_max_f_evals` stay, used only by `get_max_batch`; `integrate` no longer
    touches them. `force_max_batch` disappears from `_evaluate_f_on_mesh`.
13. **F13:** document that a GPU driver that falls back to system memory never
    fails, so the batch grows to the size of the passes.
14. **Messages:** `integrate` logs no benchmark message. The start warning of
    `get_max_batch(f=...)` is shortened (design section 8); its end message says
    "pass max_batch=N to integrate()".

## Global constraints

- Least new code; reuse `_measure_call`, `_release_memory`, `_is_out_of_memory`,
  `_get_usable_memory`, `_split_f_output`, the existing retry loop.
- Every function annotated, NumPy docstrings, `mypy padaquad/`, `ruff`,
  `numpydoc lint` clean.
- Tests on a compute node only (`salloc ... -t 00:30:00 --job-name=interactive`),
  popcornn python, absolute paths; compare failing sets with `main`.
- Branch `change_memory`; commit per task; push to `origin change_memory`.

## Review focus (tests that pin the risky cases)

1. A backward out-of-memory error after earlier batches: gradients after the
   restart equal those of an uninterrupted run (CPU, injected error).
2. A custom autograd function hiding a parameter: without `parameters=` the
   call raises `BackwardOutOfMemoryError` with a usable `state`; with it, it
   restarts and the gradients are right.
3. A data-dependent branch that reaches a new parameter in a later batch: it is
   saved and restored too.
4. `conserve_memory=True` with growth and with retries: integral right.
5. A state reused with other settings (a size larger than fits): the retries
   converge and the integral is right.

---

### Task 1: fix `conserve_memory=True` after a retry (design section 0)

**Files:** `padaquad/quadrature/base.py` (`_evaluate_f_on_split_residual_nodes`),
`tests/test_oom_recovery.py`.

- [ ] Apply the saved fix and its two tests (already written and run).
- [ ] Run `tests/test_oom_recovery.py` and the files that use `conserve_memory`.
- [ ] Commit "Fix conserve_memory=True after an out-of-memory retry".

### Task 2: the state, the batch attributes, the test calls, growth

**Files:** `padaquad/results.py`, `padaquad/__init__.py`,
`padaquad/quadrature/memory.py`, `padaquad/quadrature/base.py`, tests.

**Interfaces:**
- `results.IntegrationState(mesh: Tensor | None, max_batch: int, max_batch_converged: bool)`.
- `IntegrationResult.max_batch_converged: bool | None`; property `state -> IntegrationState | None`.
- `memory._shrink_batch(n: int) -> int` = `max(1, int(min(round(0.75 n), n - 1)))`
  (moved out of `_evaluate_f_in_batches`, now used in three places).
- `_MemoryManagement._batch_for(self, n_waiting: int) -> int` = `min(n_waiting, B)`
  if converged else `min(n_waiting, 10 B)`.
- `_MemoryManagement._first_batch(self, f, f_args, node, n_first, total_mem_usage) -> tuple[int, bool]`:
  test calls of 10, 100, ... <= `n_first` through `_measure_call` with no-op
  readers; an out-of-memory error shrinks by `_shrink_batch` until a call runs.
- `integrate(..., state=None, parameters=None)`.

- [ ] Tests first (in `tests/test_memory_state.py`, a new file for the new
  behavior): test calls stop at the order of the first pass; a failed test call
  shrinks by 0.75 until it runs and converges; growth by 10 per call of `f`
  (default and `conserve_memory` paths); an OOM converges and lowers the size;
  `max_batch=` and the constructor's give converged; `state=` with `mesh=`
  raises; `state=` and `max_batch=` together: the given size wins; the state of
  one call passed to the next starts from its mesh and size; `result.state`
  survives `dataclasses.asdict` and `torch.load(weights_only=True)`;
  `IntegrationState` rejects a size below 1; no benchmark message in
  `integrate`.
- [ ] Implement; remove the benchmark call, `reported_max_batch` and
  `force_max_batch` from `integrate` and `_evaluate_f_on_mesh`.
- [ ] Update the tests whose subject changed (`test_memory_benchmark.py`
  through `integrate`, `test_oom_recovery.py` on `_oom_max_batch`,
  `test_evaluate_f_on_mesh.py` dispatcher, `test_bug_regressions.py`).
- [ ] Commit.

### Task 3: the `take_gradient=True` path

**Files:** `base.py` (`_evaluate_f_on_full_nodes`, the backward step of
`integrate`), `memory.py` (`BackwardOutOfMemoryError`, `_is_out_of_memory`, the
graph walk), `padaquad/__init__.py`, tests.

- [ ] Tests first: forward OOM retried and converged; backward check shrinks
  (fake counters on CPU through monkeypatching the two readings); backward OOM
  with gradients restored and the restart giving the reference gradients
  (sequential and data-dependent-branch integrands); custom autograd function
  without and with `parameters=`; a post-accumulate hook raises; the error is
  not an ordinary OOM for `_is_out_of_memory` and survives pickling; on a GPU
  (skip without CUDA) a real backward check on a small network.
- [ ] Implement.
- [ ] Commit.

### Task 4: messages, documentation, examples

- [ ] Shortened start warning and end clause of `_setup_memory_checks`.
- [ ] Docstrings of `integrate`, `IntegrationResult`, `padaquad.integrate`,
  `get_max_batch`; `README.md`, `docs/user_guide/devices_memory.md`
  (`ug-memory` and the limitations), `repeated_calls.md`, `troubleshooting.md`,
  `examples/pode/trainer.py` and `examples/quadrature_basics.py` (pass
  `state=result.state`), `CHANGELOG.md` (2.0.0 section), `CLAUDE.md`.
- [ ] Docs build with `-W`, `test_examples_run.py`, `test_docs_vocabulary.py`.
- [ ] Commit.

### Task 5: whole suite, compare with `main`, push

- [ ] Full suite on the node for the branch and for `main` in parallel, one GPU
  each; the failing sets must match or shrink.
- [ ] mypy, ruff, numpydoc.
- [ ] Push `change_memory`.

### Task 6: two adversarial reviews (after the plan)

- [ ] Bug hunt: keep its good tests, fix what it finds.
- [ ] Break-and-speed review of everything memory related: fix what it finds.
- [ ] Push.
