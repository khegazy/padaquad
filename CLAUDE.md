# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

**padaquad** is a PyTorch library for **adaptive numerical
quadrature**: it computes definite integrals
$\int_a^b f(t)\,dt$ for a known integrand $f$ by evaluating many
quadrature panels in parallel batches on GPU, with full autograd through
the integration loop. Version 2.0.0 in `pyproject.toml` (Alpha), Python 3.11+.

**Critical distinction.** The integrand `f` depends only on `t` (and
optional extra args), NOT on accumulated state `y`. This independence
between panels is what enables parallel evaluation. This is numerical
quadrature, not ODE solving — for state-coupled $\dot y = f(t, y)$ users
should use `torchdiffeq` / `torchode` / `diffrax` directly.

## Build & test commands

```bash
# Install in development mode (with venv recommended)
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pre-commit install

# Run all tests and docstring examples (3396 pass and 129 skip on a CPU-only Mac; GPU-only tests skip there)
.venv/bin/pytest -v

# Run all tests with coverage (bare --cov: the settings in pyproject.toml measure
# the imported padaquad, editable or installed; --cov=padaquad would measure the
# ./padaquad checkout and report 0% under a non-editable install, as in CI)
.venv/bin/pytest --cov --cov-report=xml

# Run a single test file
.venv/bin/pytest tests/test_integrals.py -v

# Run a specific parametrized case
.venv/bin/pytest tests/test_integrals.py -k "dopri5 and damped_sine" -v

# Run with snapshot regeneration (after intentional behavior changes)
TPD_REGENERATE_SNAPSHOTS=1 .venv/bin/pytest tests/test_snapshots.py

# Lint with ruff
.venv/bin/ruff check padaquad/

# Type check (enforced in CI by .github/workflows/tests.yaml; must report no issues)
.venv/bin/mypy padaquad/

# Docstring checks (pre-commit hook numpydoc-validation; config in pyproject.toml)
numpydoc lint $(git ls-files 'padaquad/*.py')

# Build the documentation (warnings are errors; CI: .github/workflows/docs.yaml)
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
sphinx-build -b linkcheck docs docs/_build/linkcheck
```

`pytest` also runs the docstring Examples (`--doctest-modules`, testpaths
include `padaquad/`); the repository-root `conftest.py` seeds torch and quiets
the padaquad log for them.

## Documentation

- Sphinx + PyData theme in `docs/` (`conf.py`; pages in MyST Markdown). The
  API reference is generated from the NumPy docstrings (autodoc, autosummary,
  numpydoc; `autodoc_typehints = "none"`, so types come from the docstrings;
  `autoclass_content = "both"`, so the solvers' constructor options, documented
  in `AdaptiveQuadrature.__init__`, appear on the class pages).
- Writing rules for docstrings and pages: `docs/development/docstrings.md`;
  vocabulary: `docs/glossary.md` (subinterval, mesh, node, method; never
  panel/barrier/step for a subinterval). `tests/test_docs_vocabulary.py`
  checks the public docstrings and user pages.
- One bibliography, `docs/references.bib`: only sources that were opened and
  checked, each with a `checked` field saying what was checked (not printed on
  the site). Each method's
  `references` attribute holds its keys.
- Generated at build time, never typed by hand: the methods table
  (`docs/_ext/methods_table.py`, from the registries; tested by
  `tests/test_docs_methods_table.py`) and the Performance table
  (`docs/_ext/performance_table.py`, from `docs/performance/results_<device>.json`,
  written by `examples/speed_test/torchdiffeq_comparison.py`; one section per
  file, tested by `tests/test_docs_performance_table.py`), and, by the same
  extension, the memory options tables of the user guide (from
  `docs/performance/memory_options.json`, written by
  `examples/speed_test/trade_speed_for_memory.py`).
- README sections reused on the site are marked
  `<!-- include-start: NAME -->` / `<!-- include-end: NAME -->`; the Python
  blocks of the README and of `docs/development/adding_a_method.md`
  (`DOCUMENTS` in `tests/test_examples_run.py`) run in that test unless
  preceded by `<!-- not-tested -->`.
- The gallery runs the scripts listed in `filename_pattern` (`docs/conf.py`):
  the `examples/*.py` tutorials and `examples/verification/*.py`; matplotlib
  figures become the cards' thumbnails.

mypy runs with every function annotated and every body checked
(`disallow_untyped_defs`, `check_untyped_defs`, `disallow_any_generics`,
`no_implicit_optional`; `[tool.mypy]` in `pyproject.toml`). It must run where
torch is installed: with `ignore_missing_imports = true` a missing torch turns
every tensor type into `Any` and the check passes without checking anything.
The package ships `py.typed`.

## Test structure

Tests are in a flat `tests/` directory using `@pytest.mark.parametrize`
for method × integrand combinations. Shared constants and helpers live
in `tests/_helpers.py` (not `conftest.py`, because
`--import-mode=importlib` in pyproject.toml prevents importing conftest
directly). The `conftest.py` adds `tests/` to `sys.path` so `_helpers`
is importable.

The test suite is split between **integration** (end-to-end runs of the
solver) and **unit** (one function at a time with hand-crafted inputs):

| File | What it tests |
|---|---|
| `test_integrals.py` | Numerical accuracy: each uniform method × each integrand in `integrand_dict`, plus time ordering, mesh ordering, step continuity |
| `test_variable_integration.py` | Same but for variable methods (`adaptive_heun`, `interpolatory3_variable`) |
| `test_data_types.py` | float32 and float64 handling |
| `test_adaptivity.py` | Step adding (coarse mesh grows) and removal (dense mesh shrinks), mesh convergence |
| `test_tableau.py` | Weights sum to 1 (uniform tables and the position-dependent rule) |
| `test_path_integral.py` | `integrate()` wrapper matches direct solver construction |
| `test_chemistry.py` | Wolf-Schlegel 2D potential: parallel vs scipy.integrate.quad |
| `test_exactness.py` | **Polynomial degree** of every method via `MethodClass.measure_degree` (Legendre polynomials): exact through `method.degree` and not one degree further; error-estimate degree = `error_exponent - 2`; stored float64 tables checked in 60 digits (mpmath) |
| `test_scipy_agreement.py` | Cross-validation against `scipy.integrate.quad` on canonical smooth integrands |
| `test_convergence_rate.py` | Empirical convergence-rate slope matches each method's theoretical rate |
| `test_autodiff_consistency.py` | $\nabla_\theta\int f_\theta dt$ via backprop equals $\int \nabla_\theta f_\theta dt$ |
| `test_error_indicator.py` | Behavior of absolute vs cumulative error indicators (Bug B8 investigation) |
| `test_bug_regressions.py` | Regression tests for bugs B4, B6 fixed in Phase 1; reuse by passing the returned state (no state on the solver, `same_f` and constructor bounds gone); the interval rules (bounds required, cut, outside extends, rounding snap, one-ULP cut, mesh validation, keyword-only options) |
| `test_snapshots.py` | Golden-value regression net for the file-split refactor; CPU-only float64 |
| `test_public_api.py` | Smoke test of every public symbol and `IntegrationResult` fields |
| `test_rk_integral.py` | `_RK_integral`: constant / linear / multi-step / multi-dim / zero-width / variable-b inputs |
| `test_base_and_methods.py` | `get_sampling_type`, `_Tableau` dtype conversion, `_get_method` factory (unknown name raises `ValueError`), `error_exponent`, weight formulas |
| `test_error_computation.py` | `_error_norm`, `_rec_remove`, `_compute_error_ratios_absolute`/`_cumulative` |
| `test_interpolation.py` | `_compute_nodes`: endpoints, shapes, monotonicity, tableau-c matching |
| `test_sorted_insert.py` | `_get_sorted_indices` and `_insert_sorted_results` |
| `test_base_solver.py` | `SolverBase._set_dtype`, `set_dtype_by_input`, `_check_variables` (f and y0; the bound defaults through `_setup_integral_bounds`), `_integral_loss`, device selection (`_resolve_device`, indexed GPUs, no global side effects) |
| `test_methods_extra.py` | Variable-mode `adaptive_heun` (the uniform heun object) weights, the generic `VariableMethodClass` (normalized nodes reach the weight function, shape check), node-fraction validation, `to_device`, dtype conversion |
| `test_solver_methods.py` | `adaptive_quadrature` factory, `method.weights` through both solvers, per-step node count `solver.C` |
| `test_adaptive_steps.py` | `_adaptively_increase_mesh`: pass/fail/mixed error ratios, midpoint placement |
| `test_record_and_sort.py` | `_record_results` and `_sort_record` |
| `test_evaluate_and_merge.py` | `_evaluate_adaptive_nodes` + `_merge_excess_nodes` for both uniform and variable solvers |
| `test_prune_and_optimal.py` | `_prune_excess_mesh`, `_get_optimal_mesh`, pair ratio = merged step's ratio; end to end: the state's mesh re-evaluates with no failing step (every uniform method, cumulative mode), repeated warm starts prune to a fixed point; `mesh_optimal` ends at the bounds bit for bit and is increasing (every method, both dtypes, CPU and CUDA) |
| `test_y0_and_f_args.py` | `y0` additive offset (`result.integral = y0 + ∫f`) and `f_args` forwarding to `f(t, *f_args)` |
| `test_gradient.py` / `test_gradient_integration.py` | Per-batch backward, learnable integrand training loops |
| `test_error_integral_reference.py` | The `error_integral_reference` argument of `integrate()`, with `take_gradient=True` (the one path where a batch is judged before the rest of its pass is evaluated) |
| `test_error_masks.py` | Vector-error machinery (norms, `failure_fraction`, rounding floor) in float32 and float64 |
| `test_evaluate_f_on_mesh.py` | `_evaluate_f_on_mesh` and its two evaluation paths; the returned `MethodOutput` equals one `_calculate_integral` call bit for bit however the calls of `f` were cut |
| `test_trade_speed_for_memory.py` | `trade_speed_for_memory`, `evaluation_buffer`, `record_evaluations`: the same calls of `f`, mesh and integral (bit for bit) with every setting, for every `max_batch`, uniform and variable, CPU and CUDA; when the values are integrated (once per pass, per call, per buffer); the values' device in every result `loss_fxn` receives; lower GPU peak; the `ValueError`s; the four warnings, on the first call only (no state); without values, `state.mesh` is the `power_law` one; every feature unchanged |
| `test_examples_run.py` | Runs `examples/*.py` (with output markers, or self-checking with asserts: `SELF_CHECKING_EXAMPLES`, `MPLBACKEND=Agg`), every script in `examples/verification/`, and every ```python block of the files in its `DOCUMENTS` (the README, `docs/development/adding_a_method.md`; except blocks preceded by `<!-- not-tested -->`) |
| `test_max_adaptive_splits.py` | The `max_adaptive_splits` depth cap |
| `test_memory_benchmark.py` | The memory benchmark, `_setup_memory_checks`; `get_max_batch(f=..., t=...)` (measurement, argument checks) and the measured `result.max_batch` |
| `test_methods_registry.py` | Registry consistency: table shapes and dtypes, `degree` / `error_exponent` types |
| `test_multi_d_integrand.py` | Integrands with D > 1 outputs |
| `test_oom_recovery.py` | The out-of-memory catch-and-retry path (direct calls of `_evaluate_f_on_split_nodes`, including retries that end inside a subinterval; `integrate` with and without `trade_speed_for_memory`) |
| `test_recheck_accepted_panels.py` | Re-check of accepted steps against the drifting tolerance (with `take_gradient=False` the drift is between passes: `_cancelling_peak` hides its cancelling part in a narrow peak the first passes miss); `enforce_convergence` (default `not take_gradient`, True with `take_gradient=True` raises, False turns it off); a fixed `error_integral_reference` dropped once every step is accepted (converges, never evaluates a step twice, split cap, CUDA, `failure_fraction`, NaN) |
| `test_solver_isolation.py` | Solvers do not share method objects or state |
| `test_tracked_variables.py` | The `tracked_variables` feature |
| `test_docs_performance_table.py` | The Performance page's rendering of `docs/performance/results_*.json`: no file, one, two (name order, hardware headings), unreadable files named in the error; the memory options tables from `docs/performance/memory_options.json` (every setting of every integrand, defaults first, missing and unreadable files) |
| `test_split_statistics.py` | `n_start_steps` / `n_start_steps_split` / `max_split_depth` on integrands with known answers; `converged` against a hand-computed tolerance check (absolute and cumulative, with `y0`), in `failure_fraction` mode (including a NaN component), against the final integral, and its warning (numbers, output component, remedies); `max_adaptive_splits` type check |

**Snapshot tests** are the regression net for internal refactors: the
file `tests/test_snapshots_data.json` pins `integral`, `integral_error`,
and `n_optimal_mesh` for every (method × integrand × tolerance) cell.
`n_optimal_mesh` is checked only where the stored error estimate is above
float64 rounding level (|error| > 10·eps·|I|); below it, last-bit differences
between CPUs change the mesh-merge decisions and the count varies between CI
runners. Regenerate with `TPD_REGENERATE_SNAPSHOTS=1 pytest tests/test_snapshots.py`.

## Architecture

### Package layout

```
padaquad/
├── __init__.py            # public API exports
├── base.py                # SolverBase, steps enum, get_sampling_type
├── results.py             # IntegrationResult, MethodOutput dataclasses
├── methods/               # quadrature method registries
│   ├── _base.py             # MethodClass (ABC), UniformMethodClass,
│   │                        #   VariableMethodClass, _Tableau
│   ├── runge_kutta.py       # adaptive_heun, fehlberg2, bosh3, dopri5
│   ├── gauss_kronrod.py     # gk7, gk15, gk21, gk31  (with builder)
│   ├── clenshaw_curtis.py   # cc5, cc9, cc17, cc33, cc65  (cosine-series weights)
│   └── interpolatory.py     # interpolatory3_variable; variable registry
├── quadrature/            # adaptive integration engine
│   ├── base.py              # AdaptiveQuadrature ABC + main integrate loop,
│   │                        #   node placement, _calculate_integral, _RK_integral
│   ├── memory.py            # _MemoryManagement mixin: memory benchmark,
│   │                        #   batch-size budget, OOM recognition
│   ├── uniform.py           # UniformAdaptiveQuadrature
│   └── variable.py          # VariableAdaptiveQuadrature
├── _frontend.py           # user-facing integrate() and adaptive_quadrature() (build a solver, run it)
├── examples.py            # integrand_dict (test integrands) + wolf_schlegel
└── distributed.py         # multi-GPU / SLURM support (internal)
```

### Class hierarchy

```
DistributedEnvironment (distributed.py)
  └── SolverBase (base.py)
        └── AdaptiveQuadrature (quadrature/base.py; also inherits the
            _MemoryManagement mixin from quadrature/memory.py)
              ├── UniformAdaptiveQuadrature (quadrature/uniform.py)
              └── VariableAdaptiveQuadrature (quadrature/variable.py)

MethodClass (ABC, methods/_base.py)        one quadrature rule
  ├── UniformMethodClass (methods/_base.py)    fixed _Tableau
  └── VariableMethodClass (methods/_base.py)   node fractions + weight_function
```

The two concrete solvers (`UniformAdaptiveQuadrature` and
`VariableAdaptiveQuadrature`) are what user code instantiates, either
via `adaptive_quadrature(...)` factory or directly. They differ only in
`sampling_type` (which selects the method registry) and in
`_evaluate_adaptive_nodes` / `_merge_excess_nodes`. Node placement and
`_calculate_integral` live once in `AdaptiveQuadrature` and reach the
rule only through `MethodClass.node_fractions()` and
`MethodClass.weights(nodes)` (both families return weights `[N, C, 1]`).
Every registered rule is an instance of one of the two generic classes,
built from data (a `_Tableau`; or node fractions and a module-level
`weight_function`), never of a rule-specific subclass; both classes reject
malformed node fractions with `ValueError`. `adaptive_heun` is one object in
both registries.

Each method stores `degree` (highest polynomial degree integrated exactly,
validated int >= 0) and `error_exponent` (k = degree of the lower-degree method
in the error estimate + 2, validated int >= 2); `order` is the property
`degree + 1`. `MethodClass.measure_degree(nodes=None)` measures both degrees
from the weights (Legendre polynomials, float64, CPU) and is what
`tests/test_exactness.py` and `examples/verification/polynomial_degree.py`
call. `examples/verification/integral_accuracy.py` checks whole integrations
against exact answers; `fehlberg2`'s error estimate ignores curvature and is a
documented known limitation there.

### Public API

```python
from padaquad import (
    integrate,  # one-shot quadrature
    adaptive_quadrature,  # factory for repeated calls
    UniformAdaptiveQuadrature,  # concrete uniform-sampling solver
    VariableAdaptiveQuadrature,  # concrete variable-sampling solver
    IntegrationResult,  # return-type dataclass
    UNIFORM_METHODS,  # registry of uniform method names
    VARIABLE_METHODS,  # registry of variable method names
    integrand_dict,
    wolf_schlegel,  # test integrands with analytical solutions
    steps,  # sampling-mode enum
)
```

### Method portfolio

```
methods/runge_kutta.py:    adaptive_heun, fehlberg2, bosh3, dopri5
methods/gauss_kronrod.py:  gk7, gk15, gk21 (default), gk31
methods/clenshaw_curtis.py: cc5, cc9, cc17, cc33, cc65
methods/interpolatory.py:  interpolatory3_variable (+ adaptive_heun, the same object)
```

For smooth integrands at academic-grade precision, prefer `gk21`
(default) or `cc33`. RK methods are kept for backwards-compatibility and
as low-order baselines. The low-point methods (`gk7`, `cc5`, `cc9`) fill
the gap between the RK rules and the headline `gk15`/`cc17`: useful when
the adaptive controller refines aggressively and a high-node rule wastes
evaluations on smooth regions.

### Tensor shape conventions

- **N**: number of integration steps in a batch
- **C**: number of quadrature points per step (`method.n_nodes`,
  e.g. 4 for bosh3, 7 for dopri5, 23 for gk21 with endpoint padding)
- **T**: dimensionality of time (usually 1, but supports multi-D)
- **D**: dimensionality of `f`'s output

Key tensors: `nodes: [N, C, T]`, `y: [N, C, D]`, weights from
`method.weights(nodes)`: `b, b_error: [N, C, 1]`, stored tables
`tableau.c/b/b_error: [C]`, `y0: [D]`, mesh barriers: `[M, T]`.

### Core algorithm (`AdaptiveQuadrature.integrate()` in `quadrature/base.py`)

`mesh` is the boundary array dividing `[mesh_init, mesh_final]` into
integration steps (panels). Between consecutive barriers, C
quadrature points (`nodes`) are placed per the tableau. `mesh_trackers`
is a boolean array where `True` means the panel still needs evaluation.

1. **Initialize barriers.** When `mesh=None`, generate
   ~$\sqrt{N_\text{init}}$ evenly-spaced top-level segments with random
   sub-barriers (random by default — uniform aliases on
   uniform-feature integrands). With a `mesh` (for example an earlier
   call's `state.mesh`), start from it, cut to any bounds inside it
   (`_setup_integral_bounds`).
2. **Main loop.** While any `mesh_trackers` are `True`:
   - Size the next call of `f` with `_batch_for` (the call's batch size, or
     10 times it while not converged; see Memory management).
   - `_evaluate_f_on_mesh` evaluates and integrates: with `take_gradient=True`
     (`_evaluate_f_on_full_nodes`) the whole panels that fit in one call;
     with `take_gradient=False` (`_evaluate_f_on_split_nodes`) every waiting
     panel, its flattened nodes in calls of at most the batch size (from the
     generator `_evaluate_f_in_batches`; a call may end inside a panel and
     the next finishes it). Nodes via `_compute_nodes`; integral + error via
     `_calculate_integral` (weighted sum with the rule's `b` and `b_error`
     weights from `method.weights`; the error estimates are detached), which
     the split path runs on the completed panels once per pass (`buffer_bytes`
     = inf, the default) or, with `trade_speed_for_memory=True`, whenever
     `evaluation_buffer` bytes of them wait (0, after every call, when None);
     bit-identical either way (every row depends only on its panel). It then
     sends their values and tracked variables to `values_device`: the
     solver's device by default (the record moves the accepted ones to
     `result_device` after the decision), `result_device` with
     `trade_speed_for_memory`, None (dropped) with `record_evaluations=False`
     (then `state.mesh` uses `merge_rule="power_law"`: `_merge_inputs` with
     `y=None`). Nodes stay on the solver's device. Both return `(nodes, y,
     tracked, step_idxs, MethodOutput)`; the pass's integral is
     `sum(mesh_quadratures)`. `integrate` raises `ValueError` for
     `trade_speed_for_memory` with `take_gradient=True` and for a misused
     `evaluation_buffer`, and logs four warnings only when no `state` is given
     (the first call of a series).
   - Compute error ratios (absolute or cumulative mode).
   - `_adaptively_increase_mesh`: panels with ratio `< 1` are accepted;
     panels with ratio `>= 1` are split at midpoint into two
     sub-panels for re-evaluation.
   - Record accepted results into the running record.
   - If `take_gradient`: per-batch `loss.backward()`.
3. **Post-convergence.** Sort the record; read the split statistics off
   `split_counts` (per-barrier count of halvings, 0 = starting step);
   judge every recorded step against the final integral with
   `_record_error_check` (the same helper the in-loop re-check uses) to set
   `converged`, logging `_warn_not_converged` when it is `False` (a NaN/Inf
   error counts as failing); then compute the optimal mesh (`state.mesh`) via
   `_get_optimal_mesh` (prune low-error pairs, refine
   high-error gaps), returned for the caller to pass to a later call.

### Error computation modes

Two orthogonal axes control the adaptive accept/reject decision: the
**tolerance reference** (`use_absolute_error_ratio`) and the **vector-error
reduction scheme** (`error_norm`).

**Tolerance reference** (`use_absolute_error_ratio`):
- **Absolute** (default): denominator is `max(atol, rtol * |total_integral|)` (`atol + rtol * |I_d|` for `"failure_fraction"`, below). Every panel uses the same denominator. Best for path integrals where the total is the meaningful quantity.
- **Cumulative**: the same with `|cumsum_to_step|` (the running integral through the panel) in place of `|total_integral|` (traditional ODE-style). Per-panel ratios *decrease* as integration progresses.

**Vector-error reduction** (`error_norm`, modeled on
`scipy.integrate.quad_vec`'s `norm`): for vector-valued integrands (`D > 1`)
the per-step error must be reduced to one accept/reject decision. Two families,
reducing to `|error| / max(atol, rtol*|I|) < 1` (norm family) or
`|error| / (atol + rtol*|I|) < 1` (`"failure_fraction"`) when `D == 1`:
- **Norm family** — *reduce-then-compare* (scipy style): reduce the error
  vector to a scalar with the norm, compare to `max(atol, rtol * norm(integral))`
  (scipy `quad_vec` uses `max(epsabs, epsrel*norm(I))`).
  Accept when `< 1`. Options: `"2"` (default, L2 = `sqrt(sum(e²))`), `"max"`
  (L∞), `"rms"` (`sqrt(mean(e²))`, padaquad's historical reduction), or a
  callable reducing the last (`D`) axis.
- **`"failure_fraction"`** — *per-component*: each output element is compared
  against its own `atol + rtol*|I_d|`; a panel is accepted when the fraction of
  failing elements is `<= mesh_failure_tolerance` (default 0.0 ⇒ every element
  must pass). This bounds the error of *every* output dimension rather than an
  aggregate.

**Re-check of accepted steps** (`enforce_convergence`, an `integrate`
argument; None means `not take_gradient`, True with `take_gradient=True` raises
`ValueError`): Step 4b judges every recorded step, from its stored
`mesh_quadratures` / `mesh_quadrature_errors` (no call of `f`), against the same
reference as the batch, via `_failed_record_positions`, and the failures below
`max_adaptive_splits` are split in the same mesh update (`extra_split_idxs`)
and removed from the record (`_remove_failed_record_panels`). A fixed
`error_integral_reference` turns Step 4b off; with `enforce_convergence`, Step 7
drops it once no tracker is True: the record is judged against its own
integral, the failures are split directly (`_adaptively_increase_mesh` with an
empty batch) and the loop continues against the running integral. No step is
ever evaluated twice (a user rule: `f` can be very expensive). The plan and
its review record were deleted once done; they are in commit `f6f74d3`
(`docs/issues/2026-11-18-enforce-convergence-plan.md`).

Both families accept a panel with any non-finite (NaN/Inf) element (splitting
cannot fix it; see `_adaptively_increase_mesh`) and apply a scipy-style
machine-precision **rounding floor** (`_round_floor`) so refinement stops once a
panel's error is at/below round-off. The scheme is set at construction
(`error_norm=`, `mesh_failure_tolerance=`) and overridable per call in
`integrate()`. `_compute_error_ratios` dispatches to `_error_ratios_norm` /
`_error_ratios_failure_fraction`; `_accept_reject_masks` turns the result into
keep/split masks consumed by `_adaptively_increase_mesh`.

`error_ratios_2steps`: the pair ratio of consecutive step pairs, used only by
`_get_optimal_mesh` to merge pairs: a pair error divided by the tolerance the
merged step gets (in cumulative mode the right step's `|cumsum|`, where the
merged step ends). The pair error is `|e_i| + |e_{i+1}|` plus, for
`"predicted"`, the merge prediction; `_merge_inputs` returns that prediction (or
None) and the cutoff for the constructor option `merge_rule`:
- `"predicted"` (default): pair error `_predict_merged_errors(...) + |e_i| +
  |e_{i+1}|`, cutoff `remove_cut` (default 0.1). The prediction interpolates
  each step's stored values (barycentric, through its distinct node fractions)
  onto the merged step's nodes and applies the method's weights:
  `max(|E_hat|, |Q_hat - Q_i - Q_{i+1}|)`, the predicted merged error estimate
  and the halves-versus-whole difference. Both are linear in the stored
  values, so `_merged_pair_weights` builds four weight vectors per width ratio
  (log2 ratio quantized to 2^-48), cached on the solver
  (`_merge_weight_cache`, reset in `_set_solver_dtype`). With T > 1 it falls
  back to `"power_law"` with a warning.
- `"power_law"` (padaquad 1.3.0 and 1.4.0): summed errors, cutoff `remove_cut / 2^(k-1)`
  (assumes the estimate scales as h^k, k = `error_exponent`; on wide steps it
  grows far less, so safe merges are refused).
- `"sum"` (1.2.2): summed errors, cutoff `remove_cut`.
`_rec_remove` (one pass: keep every other flag of each run) ensures no two
adjacent pairs are both flagged, so one call merges at most one level. The
merged step records the pair's summed error, at most the pair error, so the add
phase of `_get_optimal_mesh` does not split it (unless the rounding floor sets
the tolerance). Plan, evidence and the adversarial reviews: the commits of PR
#69; open: `docs/issues/predicted-merge-cost.md` (time per distinct width ratio).

### Uniform vs variable sampling

- **Uniform** (`quadrature/uniform.py`): nodes at fixed fractional
  positions within each panel (tableau `c` values). Tableau `b` values
  are constants.
  - *Splitting*: discards old evaluations, places fresh quadrature
    points at standard `c` positions in the new sub-panels.
  - *Merging*: re-interpolates C nodes at standard `c` positions
    spanning `[A.start, B.end]`.
- **Variable** (`quadrature/variable.py`): weights computed from each
  panel's actual nodes by `method.weights(nodes)`
  (`VariableMethodClass.weight_function`; for `interpolatory3_variable`
  the Sanderse-Veldman weights, `_interpolatory3_weights`).
  - *Splitting*: the integration loop does NOT reuse evaluations. Like the
    uniform solver it inserts the midpoint into the mesh and places fresh
    nodes in both halves with `_compute_nodes`, at the node fractions, so
    `interpolatory3_variable` always runs as Simpson's rule (a = 1/2).
    `_evaluate_adaptive_nodes`, which would interleave old and new nodes into
    two C-point halves, exists and is unit-tested but is never called by
    `integrate()` (checked 2026-11-12: 0 calls on a run with 2495
    subintervals). It also allocates its node array with last dimension D
    instead of T (latent; wrong shape only when D != T).
  - *Merging*: concatenates points from both panels (C + C-1 = 2C-1,
    shared boundary not duplicated), then subsamples every other point
    to get C. `method.weights(nodes)` recomputes the right weights for
    the new layout.

### Memory management

- **The batch size of `integrate`** (no benchmark; the design, plan and
  reviews were deleted once done and are in commit `6b3cf42`,
  `docs/issues/2026-11-17-*.md` and `2026-11-18-memory-state-plan.md`; the
  two reviews of the code in `d0bcbdc` and `59b7f11`). Per call, `_batch` (largest batch known
  to fit) and `_batch_converged` (will not grow again), set at the start of
  `integrate`: from `max_batch` (call, or constructor when no state) →
  converged; from `state=` (`IntegrationState`, `results.py`); else from
  `_initialize_max_batch` (forward calls on 10, 100 and 1,000 = `_LARGEST_TEST_CALL`
  copies of `mesh_init`, stopping at the order of the first pass, through `_measure_call` with no-op readers;
  an OOM shrinks by `_shrink_batch` = 0.75 until one runs → converged).
  `_batch_for(n_waiting)` sizes every call of `f`: `min(n_waiting, B)` if
  converged, else `min(n_waiting, 10 B)` (`_GROWTH`); `_evaluate_f_in_batches`
  raises `_batch` after each call that ran and, on OOM, sets it to
  `_shrink_batch(failed)`, converged. `integrate` returns `(result, IntegrationState(mesh_optimal,
  _batch, _batch_converged))`; the result has no mesh_optimal or max_batch.
- **`take_gradient=True`** (`_evaluate_f_on_full_nodes`, one batch per loop
  turn): a forward OOM drops the outputs and retries with fewer
  subintervals; on a GPU, when `f`'s output requires grad, the check
  `usable >= _backward_ratio * F` (F = forward peak via
  `reset_peak_memory_stats`/`max_memory_allocated`, `_backward_ratio` = the
  previous batch's backward peak / forward peak, 1 at first) else drops and
  shrinks to `min(0.75 N, N usable / need)`. Each batch's loss and backward
  pass run in `_loss_and_backward` (memory.py): before `loss.backward()`,
  `_save_grads` walks the loss graph and saves `.grad` of every leaf first
  seen in the call (plus `parameters=` at the start), and on a GPU the
  backward peak sets `_backward_ratio`; it returns no loss on an OOM. A
  backward OOM breaks the loop; `_raise_after_backward_oom` restores the
  gradients and raises `_Restart`, which `_restart_after_backward_oom`
  turns into a new call with the original arguments and
  `state=IntegrationState(start_mesh, _shrink_batch(failed), True)`, or
  raises `BackwardOutOfMemoryError` (custom
  autograd function without `parameters=`, a post-accumulate-grad hook,
  `torch.distributed` world size > 1, or a batch below C).
  `_is_out_of_memory` is False for that error.
- `_setup_memory_checks` (the benchmark) runs only in
  `get_max_batch(f=...)`; `f_mem_coeffs`, `_get_max_f_evals`,
  `_oom_max_batch` and `_cap_max_batch` serve only `get_max_batch`.
- All of this lives in `quadrature/memory.py` (`_MemoryManagement` mixin,
  `_is_out_of_memory`); the batch loop `_evaluate_f_in_batches` stays in
  `quadrature/base.py`.
- `_setup_memory_checks` fits the memory of ONE call of `f` on N points,
  `f_mem_coeffs` = s·((a + E) + b·N + d·N²) in GB (s =
  `_GRADIENT_MEMORY_SAFETY_FACTOR` = 2.1 when `take_gradient`, else 1), and
  returns the batch size that fit gives for the memory free at its end. Its
  only caller: the public `get_max_batch(f=..., t=...)`, which makes a
  one-point call first, clears `_oom_max_batch`, and lets a user measure
  before any integral. The reader (`_peak_memory_reader` →
  `_Reader`: reset, read, E, noise floor, counts_reuse): CUDA peak counter
  (floor 10 KB, counts reused memory); Linux `VmHWM` reset through
  `/proc/self/clear_refs`, after glibc `malloc_trim(0)` when available
  (floor 4 MiB; counts reused memory only with the trim; E = 2 × CPUs in the
  affinity mask × max(32, 2 × online CPUs) pages); elsewhere an RSS
  before/after difference (floor 4 MiB, misses reused memory). Every reading
  is at least the bytes of the input and output tensors (`_tensor_bytes`).
  Readings ≥ the floor are *reliable*; only they are fitted (a + b·N + d·N²,
  a, b, d ≥ 0, `_fit_up_to_quadratic`, scipy `nnls`). The fit is *ready*
  with ≥ 3 reliable readings whose fitted memory grows by ≥ the floor from the
  first to the last; before, the worst case y(kN) ≤ k²·y(N) through the
  reliable reading with the largest N (one local `memory_model`, returning
  the coefficients and `ready`, used for sizing and the stored fit).
  N_next = min(step·N, n_mem, n_time): step 2 after a call > 0.1 s on a
  reader that counts reused memory, else 10; n_mem from 0.75 of usable
  memory; n_time from 1 s (≥ 2N for the first two calls and while not ready),
  always capped by the time left of the 60 s total (`limit_benchmark_time`,
  a constructor option; False makes the total infinite). Stops: 60 s with
  the fit ready (warns), c = s·(M + E)/N changes < 1 % twice with the last 3
  readings reliable and the fit ready, a call > 1 s with the fit ready,
  N = 1e9, N_next < 2N.
  Raises `RuntimeError` (readings listed, asks for `max_batch`) when
  M > 3·(N/N_i)²·(M_i + E) for the earlier reliable reading i with the
  smallest bound (every reader), when the fit is not ready at 60 s or the
  next needed call fits in memory but is predicted past the 60 s (both name
  `limit_benchmark_time`), or when none is reliable at the end. Not ready at
  the memory budget / OOM / 1e9: the cautious worst-case fit. An OOM at
  N > 1 sets `_oom_max_batch = _OOM_BACKOFF·N`; on a reader that misses
  reused memory `_oom_max_batch` is also capped at max(largest test N, C).
  The benchmark is a heuristic for the common case (memory linear in N, or
  at most N²); its known limitations, with measured and simulated numbers,
  are listed once, in `docs/user_guide/devices_memory.md` (section
  "Limitations of the memory benchmark", label `ug-memory-limitations`;
  the start warning links to it). Read them before changing the benchmark,
  and update them when a change fixes or adds one.
  The batch loop's out-of-memory retry shrinks the batch that failed, not
  `_oom_max_batch` (which can be far above it).
- `get_max_batch()` without `f`: the constructor's `max_batch`, else
  `_get_max_f_evals` = `_largest_n(f_mem_coeffs, _MEMORY_BUDGET·usable)` from
  the last `get_max_batch(f=...)`, both through `_cap_max_batch`
  (`min(value, _oom_max_batch)`); no `default_max_batch` (removed).
- Out-of-memory errors are recognized by `_is_out_of_memory` (GPU OOM, CPU
  `DefaultCPUAllocator` refusal, CUDA "out of memory" and CUDA-library
  `*ALLOC*_FAILED` statuses, `MemoryError`). Every call of `f` in the
  integration loop goes through `_call_f` (memory.py), the one place that
  catches them: it returns None after `_release_memory` (CUDA cache only on
  a CUDA solver, and only for an error from outside PyTorch's allocator), or
  raises `torch.OutOfMemoryError` ("cannot be reduced") when the caller cannot
  shrink. Its two callers retry smaller: `_evaluate_f_in_batches`
  (`take_gradient=False`) and the `while output is None` loop of
  `_evaluate_f_on_full_nodes` (`take_gradient=True`).
- **A pass's values are held once** (`take_gradient=False`):
  `_adaptively_increase_mesh` does not filter (a copy) the per-node arrays of a
  batch whose subintervals are all accepted; the loop deletes the batch's
  values (`kept_y`, `intermediate_results`) after recording. The settings'
  memory and time are measured by
  `examples/speed_test/trade_speed_for_memory.py` (every setting, from speed to
  memory, on a light integrand and two Fourier-feature networks of ~10,000 and
  ~1,000,000 parameters; asserts the same integral and subintervals bit for
  bit; writes `docs/performance/memory_options.json`, rendered by its
  `render_markdown` into `docs/user_guide/devices_memory.md`, section
  `ug-trade-speed-for-memory`).
- `_get_usable_memory = free - buffer`, where
  `buffer = (1 - total_mem_usage) * total_memory`.
- Supports both CUDA (`torch.cuda.mem_get_info`) and CPU
  (`psutil.virtual_memory`).

### Gradient / loss support

- `take_gradient=True` triggers `loss.backward()` after each accepted
  batch. This is essential when the autograd graph for the entire
  integration would not fit in GPU memory; per-batch backward
  accumulates gradients into `theta.grad` without holding the full
  graph. A bound that requires grad reaches every batch through the one
  autograd graph of the mesh (its ends and the midpoints split from them),
  so with it each batch is evaluated on a detached copy of the mesh; the
  copies' gradients are summed on the mesh points (`mesh_grad`, carried to
  the split mesh after every batch) and passed through the mesh's graph once,
  after the loop (once per batch, the time grew with the square of the number
  of batches).
- `loss_fxn` defaults to returning the integral itself
  (`_integral_loss` in `base.py`). Linear in the integral, so the
  default path is correctness-safe under per-batch backward.
- Results from each batch after backward are `.detach()`-ed before
  being added to the running record (so the graph for that batch can
  be released).
- **The gradient as extra outputs** (`examples/gradient_as_output.py`): `f`
  returns its gradient with respect to the parameters as extra output
  components (`torch.func.grad` on detached parameters), integrated with
  `take_gradient=False` and no autograd graph; a callable `error_norm`
  chooses which components decide a split. It leaves out the terms of bounds
  that depend on the parameters. `trade_speed_for_memory=True` with an
  output that requires grad logs a warning pointing to it (the graph keeps
  what `f` saved on the device, so moving the values saves little).

### `y0` additive offset

- `y0` is an optional initial value of the integral accumulator. The
  returned `result.integral` is `y0 + ∫f(t)dt`. Default is zeros.
- `_calculate_integral`'s `y0` defaults to zero (None), as the evaluation
  functions call it, so each batch returns only its step contributions. The user-supplied
  `y0` is added once at the final `IntegrationResult`.
- `result.y0` echoes the value that was used (after dtype/device
  coercion) so callers can recover what offset was applied.
- `f_args` is a tuple forwarded positionally to the integrand:
  `f(t, *f_args)`. Used by `examples/pode/` for path parameters.

### Reusing the mesh and batch size

- The solver keeps no warm-start state between calls (`same_f`,
  `mesh_previous`, `previous_max_batch` and the constructor's bounds were
  removed). A call reuses an earlier one only through what it is passed:
  `state=state`, the second value `integrate` returns (its mesh is the optimal mesh, detached, on the solver's
  device, ends equal to the bounds bit for bit; its batch size and converged
  flag), or `mesh=` / `max_batch=` directly. `mesh=` with `state=` raises.
- Interval rules, all in `_setup_integral_bounds` (returns
  `(mesh, mesh_init, mesh_final)`): a call needs a mesh or both bounds; a
  given mesh must be `[M, T]`, M >= 2, increasing (T = 1) or with distinct
  consecutive points (T > 1; unreachable end to end, the order check fails
  for T > 1); bounds are tensors of shape [T] on every call (with or without
  a mesh; checked in `_setup_integral_bounds`, non-tensors already in
  `set_dtype_by_input`). With a mesh, a missing bound is its end, and every
  given bound becomes an end point of the starting mesh, so it gets its
  gradient as any mesh end (replacing the bound by the detached
  `mesh_optimal` end lost it): the mesh's points between the bounds are kept
  and the bounds added, which cuts the mesh at a bound inside it and extends
  it to a bound outside it (the integration refines the new subinterval). A
  bound that is not bit-equal to its end also drops the points within
  `4 eps max(|p|, |bound|)` of it (eps of the coarser of the bound's and the
  mesh's dtypes; relative, since a tolerance scaled by the larger mesh end
  snapped 1e-13 to 0 on [0, 1000]), including that end when within rounding
  (a one-ULP subinterval made `interpolatory3_variable` return NaN). A mesh
  alone, or bounds equal to its ends, leave it unchanged. Bounds not
  increasing raise `ValueError`. `_setup_initial_mesh` only builds a fresh
  mesh. With `take_gradient=True` and a mesh that requires grad (a bound
  that requires grad), each batch is evaluated on a detached copy of the
  mesh, and the summed gradient is passed through the mesh's graph once,
  after the loop.
- Every `integrate` argument after `mesh_final` is keyword-only.
- A non-converged call (e.g. under `max_adaptive_splits`) returns a
  `mesh_optimal` that splits its failing steps, so repeated calls from it
  refine further. In cumulative mode `_prune_excess_mesh` can instead
  merge a failing step with a right neighbour whose running integral (and so
  tolerance) is much larger.

### Dtype management

- `SolverBase._set_dtype` maintains two tolerance levels: `atol` /
  `rtol` (integration error control, can be float32) and
  `atol_assert` / `rtol_assert` (geometric assertions like time
  ordering, always looser).
- `float64` and `float32` are the supported runtime dtypes. `float16`
  is refused at construction (Bug B4 fix): its ~1e-3 precision floor
  exceeds typical adaptive tolerances, so the previous loose-tolerance
  path silently produced wrong answers.
- Methods are **cloned per solver instance** (`_get_method` calls
  `MethodClass.clone()`, a deep copy, for both registries); dtype/device
  mutations stay isolated to one solver and never propagate via shared
  singletons. The previous
  global-singleton hazard (where a float32 round-trip could
  irreversibly degrade `UNIFORM_METHODS["dopri5"].tableau.b` for the
  rest of the test session) is gone.

## Dependencies

- **Runtime**: torch, numpy, scipy, einops, psutil.
- **Dev**: pytest, pytest-cov, mypy, pre-commit, ruff, typos,
  torchdiffeq (for the speed-test benchmark only).
