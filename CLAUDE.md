# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

**PadaQuad** is a PyTorch library for **adaptive numerical
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
the PadaQuad log for them.

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
| `test_bug_regressions.py` | Regression tests for bugs B4, B6 fixed in Phase 1; `runtime_checks` (old name `error_on_nonfinite` raises; False skips the record-coverage and finite checks with the same result); reuse by passing the returned state (no state on the solver, `same_f` and constructor bounds gone); the interval rules (bounds required, cut, outside extends, rounding snap, one-ULP cut, mesh validation, keyword-only options); a subinterval too narrow to halve ends the call with `converged=False` (a jump with `atol=rtol=0`, both `runtime_checks`, both `take_gradient`, a second call from the state) |
| `test_snapshots.py` | Golden-value regression net for the file-split refactor; CPU-only float64 |
| `test_public_api.py` | Smoke test of every public symbol and `IntegrationResult` fields |
| `test_rk_integral.py` | `_RK_integral`: constant / linear / multi-step / multi-dim / zero-width / variable-b inputs; the weighted sum in blocks of rows (`_BLOCK_BYTES`) equals one sum bit for bit, N = 0 included |
| `test_base_and_methods.py` | `get_sampling_type`, the tableau's dtype conversion (`MethodClass.to`), `_get_method` factory (unknown name raises `ValueError`), `error_exponent`, weight formulas |
| `test_error_computation.py` | `_reduce_norm`, `_round_floor`, `_rec_remove`, `_compute_error_ratios` (absolute, cumulative, `failure_fraction`; the pair ratio only with `pair_errors`) |
| `test_interpolation.py` | `_compute_nodes`: endpoints, shapes, monotonicity, tableau-c matching |
| `test_sorted_insert.py` | `_merge_positions` (ranks of mesh indices, the record's in any order) and `_insert_sorted_results` (a new array; the old record is read, not cloned) |
| `test_base_solver.py` | `SolverBase._set_dtype`, `set_dtype_by_input`, `_check_variables` (f and y0; the bound defaults through `_setup_integral_bounds`), `_integral_loss`, device selection (`_resolve_device`, indexed GPUs, no global side effects); `atol`/`rtol` stored as floats, `ValueError` when not one finite number >= 0 (a list or tensor per output component included), unchanged by a dtype round trip (the error ratios of a float64 call after a float32 one); host waits on CUDA (none to build a solver, one for the check of a fresh starting mesh, one per mesh update (the number accepted and the number split in four, also with four-way splits) plus one per array kept on the CPU, none in `_evaluate_f_on_mesh` but the finite check of each call of `f`; counted by source line (`_lines`): one per pass at the loop's condition and one when none is left, none in `_record_results` with the record on the GPU and one per value field with `result_device="cpu"`, two in `_sort_record`; the error control of a pass, one per re-check) |
| `test_methods_extra.py` | Variable-mode `adaptive_heun` (the uniform heun object) weights, the generic `VariableMethodClass` (normalized nodes reach the weight function, shape check), node-fraction validation, `MethodClass.to` (device and dtype) |
| `test_solver_methods.py` | `adaptive_quadrature` factory, `method.weights` through both solvers, per-step node count `solver.C` |
| `test_adaptive_steps.py` | `_adaptively_increase_mesh`: pass/fail/mixed accept masks (`_keep_mask`, the loop's rule), midpoint placement, the returned `keep_idx` and `idxs_transfer` (T = 1 and 2), extra split positions in any order, the split cap, a subinterval one ULP wide accepted (`_splittable`, T = 1 and 2); four-way splits with `error_ratios` (`TestFourPieces`): bit for bit two halving updates, the rule's conditions, the cap, extra positions halved, two ULPs wide halved |
| `test_record_and_sort.py` | `_record_results` (appends in acceptance order: one tensor per loop field, one chunk per batch for the rest; a view of a larger array is copied) and `_sort_record` (rows to their mesh index, T = 1 and 2; one batch in order is not copied; chunk fields moved to the given device; a record that does not cover the mesh once raises; the gradient passes); a removal and the final ordering copy one field at a time (GPU peak); end to end, every path of the record (removals, Step 7, cumulative, tracked, trade, no record, split cap, `take_gradient`, gk21) returns mesh order and the same numbers without `runtime_checks`; `_flatten_output` equals the concatenation, values and gradient |
| `test_prune_and_optimal.py` | `_prune_excess_mesh`, `_get_optimal_mesh`, pair ratio = merged step's ratio; `build_state_mesh=False` (no build, `state.mesh` None, the rest bit for bit, the next call needs bounds); end to end: the state's mesh re-evaluates with no failing step (every uniform method, cumulative mode), repeated warm starts prune to a fixed point; `mesh_optimal` ends at the bounds bit for bit and is increasing (every method, both dtypes, CPU and CUDA) |
| `test_y0_and_f_args.py` | `y0` additive offset (`result.integral = y0 + ∫f`) and `f_args` forwarding to `f(t, *f_args)` |
| `test_gradient.py` / `test_gradient_integration.py` | Per-batch backward, learnable integrand training loops; `error_ratios` (returned and per batch) hold no autograd graph |
| `test_error_integral_reference.py` | The `error_integral_reference` argument of `integrate()`, with `take_gradient=True` (the one path where a batch is judged before the rest of its pass is evaluated) |
| `test_error_masks.py` | Vector-error machinery (norms, `failure_fraction`, rounding floor) in float32 and float64; the float tolerances act as 0-d tensors of the values' dtype, bit for bit (CPU and CUDA); pair errors set only the pair ratio; the record is judged by the reduced errors of acceptance, bit for bit the batch's ratios and masks (every `error_norm`, both modes, a NaN row) |
| `test_evaluate_f_on_mesh.py` | `_evaluate_f_on_mesh` and its two evaluation paths; the returned `MethodOutput` equals one `_calculate_integral` call bit for bit however the calls of `f` were cut |
| `test_trade_speed_for_memory.py` | `trade_speed_for_memory`, `evaluation_buffer`, `record_evaluations`: the same calls of `f`, mesh and integral (bit for bit) with every setting, for every `max_batch`, uniform and variable, CPU and CUDA; when the values are integrated (once per pass, per call, per buffer); the values' device in every result `loss_fxn` receives; lower GPU peak; the `ValueError`s; the three warnings, on the first call only (no state), none for `result_device` on the solver's device; without values, `state.mesh` is the `power_law` one; every feature unchanged; the default holds a pass once (GPU peak < 1.8 × the pass's values); an out-of-memory error at the pass's array names the remedies; `_keep_rows` in place equals indexing |
| `test_examples_run.py` | Runs `examples/*.py` (with output markers, or self-checking with asserts: `SELF_CHECKING_EXAMPLES`, `MPLBACKEND=Agg`), every script in `examples/verification/`, and every ```python block of the files in its `DOCUMENTS` (the README, `docs/development/adding_a_method.md`; except blocks preceded by `<!-- not-tested -->`) |
| `test_max_adaptive_splits.py` | The `max_adaptive_splits` depth cap; with `split_rule="accelerate"`, the same K 2^n subintervals in fewer passes |
| `test_memory_benchmark.py` | The memory benchmark, `_setup_memory_checks`; `get_max_batch(f=..., t=...)` (measurement, argument checks, `evaluation_buffer` kept free twice) and the measured `result.max_batch` |
| `test_memory_state.py` | The batch size of `integrate` and the state: the first call is the first pass (at most 10,000 points), every point `f` is called at belongs to the integration (C per evaluated subinterval; every start, both `take_gradient`, with simulated out-of-memory errors), growth by 10, the retry chain of a first call that runs out of memory, a given `max_batch`, the state's mesh and batch size; backward out-of-memory restarts |
| `test_methods_registry.py` | Registry consistency: table shapes and dtypes, `degree` / `error_exponent` types |
| `test_interior_nodes.py` | Gauss-Kronrod (nodes all interior): C points of `f` per subinterval, none at a mesh point, no call of `f` before the first pass, `1/sqrt(t)` and `log(t)` at a bound converge, result rows = evaluated points, bound gradients against the closed form |
| `test_multi_d_integrand.py` | Integrands with D > 1 outputs |
| `test_oom_recovery.py` | The out-of-memory catch-and-retry path (direct calls of `_evaluate_f_on_split_nodes`, including retries that end inside a subinterval; `integrate` with and without `trade_speed_for_memory`) |
| `test_recheck_accepted_panels.py` | Re-check of accepted steps against the drifting tolerance (with `take_gradient=False` the drift is between passes: `_cancelling_peak` hides its cancelling part in a narrow peak the first passes miss); `enforce_convergence` (default `not take_gradient`, True with `take_gradient=True` raises, False turns it off); a fixed `error_integral_reference` dropped once every step is accepted (converges, never evaluates a step twice, split cap, CUDA, `failure_fraction`, NaN); the norm family's record carries each step's `reduced_errors` into every re-check and the final check; the re-check with `split_rule="accelerate"` (fewer passes, no step evaluated twice) |
| `test_solver_isolation.py` | Solvers of different dtypes keep their own tables; the registry is never changed; `MethodClass.to` returns one shared copy per (method, device, dtype), sharing no memory with its source, dropped with its source (deep copies and pickles of a solver, CUDA); `_Tableau` frozen; no integration writes into a method's tensors (version counters); a replaced registry entry reaches new solvers only; `_integration_on_device` restores the solver's method object, and leaves the solver's device and method when the conversion raises (CUDA) |
| `test_tracked_variables.py` | The `tracked_variables` feature |
| `test_docs_performance_table.py` | The Performance page's rendering of `docs/performance/results_*.json`: no file, one, two (name order, hardware headings), unreadable files named in the error; the memory options tables from `docs/performance/memory_options.json` (every setting of every integrand, the field's `EXAMPLE_SETTINGS` in order of `max_batch`, defaults first, missing and unreadable files) |
| `test_split_statistics.py` | `n_start_steps` / `n_start_steps_split` / `max_split_depth` on integrands with known answers; `converged` against a hand-computed tolerance check (absolute and cumulative, with `y0`), in `failure_fraction` mode (including a NaN component), against the final integral, and its warning (numbers, output component, remedies); `max_adaptive_splits` type check; `split_rule="accelerate"`: validation, exact integrals (Lorentzian, jump, sin(2e4 t); gk21, gk15, gk7, cc17; both `take_gradient`), fewer points and passes on sin(2e4 t), unchanged on a kink, independent of `max_batch` (uniform and variable, with and without `trade_speed_for_memory`), warm start (absolute, cumulative, float32), `failure_fraction` halves, bound and parameter gradients, no four-way test before the first split, the documented worst case (T_20 on one starting subinterval: 11 against 7 subintervals) |

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
`sampling_type` (which selects the method registry). Node
placement, `_calculate_integral` and the merging of the optimal mesh
(`_prune_excess_mesh` drops a merged pair's shared mesh point) live once in
`AdaptiveQuadrature` and reach the
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
  e.g. 4 for bosh3, 5 for dopri5, 21 for gk21, whose nodes are all
  interior: a subinterval's width and place come from the mesh, never from
  its first and last node)
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
2. **Main loop.** While any `mesh_trackers` are `True` (one
   `torch.nonzero(mesh_trackers)` per turn gives the condition and `waiting`,
   the ascending left mesh indices `_evaluate_f_on_mesh` takes):
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
     solver's device by default (`_record_results` moves the accepted ones to
     `result_device` after the decision), `result_device` with
     `trade_speed_for_memory`, None (dropped) with `record_evaluations=False`
     (then `state.mesh` uses `merge_rule="power_law"`: `_merge_inputs` with
     `y=None`). Nodes stay on the solver's device. Both return `(nodes, y,
     tracked, step_idxs, MethodOutput)`; the pass's integral is
     `sum(mesh_quadratures)`. `integrate` raises `ValueError` for
     `trade_speed_for_memory` with `take_gradient=True` and for a misused
     `evaluation_buffer`, and logs three warnings only when no `state` is given
     (the first call of a series).
   - Compute error ratios (`_compute_error_ratios`, absolute or cumulative
     mode) from detached values, so `error_ratios` holds no autograd graph.
     The running sums only in cumulative mode: the record (empty on the first
     turn) and the batch merged in mesh order by their mesh indices
     (`_merge_positions`, `_insert_sorted_results`), one `cumsum` for the
     batch and the re-check. For the norm family the batch's reduced error
     and rounding floor (`_reduced_errors`, [N, 2]) are computed once and
     recorded (`reduced_errors`); every later judgement of the subinterval
     reads them.
   - `_adaptively_increase_mesh(method_output, keep_mask, ...)`: the steps of
     `keep_mask` (`_accept_reject_masks`) are accepted, the others split at
     the midpoint into two sub-steps for re-evaluation, except those
     `_splittable` refuses (at `max_adaptive_splits`, or one ULP wide: the
     midpoint rounds onto an end), which are accepted. With
     `split_rule="accelerate"` (constructor and `integrate` option, default
     `"halve"`), Step 5 also passes `error_ratios` (not with
     `"failure_fraction"`, nor before the call's first split, when the mesh
     has its starting length and no step can qualify): a failing step with split count >= 1 and ratio >=
     `_FOUR_PIECES_RATIO` (1e4) that `_splittable(..., halvings=2)` allows is
     split in four, by running the same insertion block a second time on both
     halves (the pieces of two successive halvings; split count + 2;
     `idxs_transfer` composed). The re-check's extra positions, Step 7 and
     `_get_optimal_mesh` pass no ratios and halve. One read of the device
     per update (the number accepted and the number in four); a stable sort
     of the number of halvings gives `keep_idx` (returned, slot 8) and the
     split positions, and every field is indexed with integers;
     `idxs_transfer` maps each old mesh point to its new position (returned,
     slot 7). The mesh stays strictly increasing by construction, so no order
     check runs.
   - Record accepted results (`_record_results`), in the order they are
     accepted, not in mesh order: each recorded subinterval carries its left
     mesh index (`record["mesh_index"]`, `idxs_transfer[step_idxs[keep_idx]]`
     when recorded, remapped through `idxs_transfer` after every mesh update;
     integers, for any T). The sums and the loop's fields
     (`_RECORD_LOOP_KEYS`: `mesh_index`, `mesh_quadratures`,
     `mesh_quadrature_errors`, read by the cumulative sums through
     `_merge_positions`, removals and the re-check, and `reduced_errors`
     (norm family only; dropped after the final check)) are one tensor each on the
     solver's device; `nodes`, `h`, `y`, `error_ratios` and the tracked
     variables are one chunk per batch in `chunks` (the values on
     `result_device`, the rest on the solver's device until the end), never
     merged during the loop.
   - If `take_gradient`: per-batch `loss.backward()`.
3. **Post-convergence.** Put the record in mesh order (`_sort_record`: it
   covers the final mesh once, so row i goes to row `mesh_index[i]`; one
   chunk already in order is kept; checked under `runtime_checks`; the chunk
   fields end on the record's device); read the split statistics off
   `split_counts` (per-barrier count of halvings, 0 = starting step; a split
   in four adds 2);
   judge every recorded step against the final integral with
   `_record_error_check` (the same helper the in-loop re-check uses) to set
   `converged`, logging `_warn_not_converged` when it is `False` (a NaN/Inf
   error counts as failing); then, unless `build_state_mesh=False` (then
   `state.mesh` is None), compute the optimal mesh (`state.mesh`) via
   `_get_optimal_mesh` (prune low-error pairs, refine
   high-error gaps), returned for the caller to pass to a later call.

### Error computation modes

Two orthogonal axes control the adaptive accept/reject decision: the
**tolerance reference** (`use_absolute_error_ratio`) and the **vector-error
reduction scheme** (`error_norm`).

**Tolerance reference** (`use_absolute_error_ratio`):
- **Absolute** (default): denominator is `max(atol, rtol * |total_integral|)` (`atol + rtol * |I_d|` for `"failure_fraction"`, below). Every panel uses the same denominator. Best for path integrals where the total is the meaningful quantity.
- **Cumulative**: the same with `|cumsum_to_step|` (the running integral through the panel) in place of `|total_integral|` (traditional ODE-style). Per-panel ratios *decrease* as integration progresses. On CUDA, `torch.cumsum` of a float64 `[N, 1]` tensor is not reproducible run to run from about N = 40,000 (A100, torch 2.8; `[N, 2]` and wider, and the CPU, are), so a cumulative-mode call with one output and that many subintervals can differ between runs in the last bits of its ratios (user guide, `ug-error-reference`); `torch.use_deterministic_algorithms(True)` makes it reproducible.

**Vector-error reduction** (`error_norm`, modeled on
`scipy.integrate.quad_vec`'s `norm`): for vector-valued integrands (`D > 1`)
the per-step error must be reduced to one accept/reject decision. Two families,
reducing to `|error| / max(atol, rtol*|I|) < 1` (norm family) or
`|error| / (atol + rtol*|I|) < 1` (`"failure_fraction"`) when `D == 1`:
- **Norm family** — *reduce-then-compare* (scipy style): reduce the error
  vector to a scalar with the norm, compare to `max(atol, rtol * norm(integral))`
  (scipy `quad_vec` uses `max(epsabs, epsrel*norm(I))`).
  Accept when `< 1`. Options: `"2"` (default, L2 = `sqrt(sum(e²))`), `"max"`
  (L∞), `"rms"` (`sqrt(mean(e²))`, PadaQuad's historical reduction), or a
  callable reducing the last (`D`) axis.
- **`"failure_fraction"`** — *per-component*: each output element is compared
  against its own `atol + rtol*|I_d|`; a panel is accepted when the fraction of
  failing elements is `<= mesh_failure_tolerance` (default 0.0 ⇒ every element
  must pass). This bounds the error of *every* output dimension rather than an
  aggregate.

**Re-check of accepted steps** (`enforce_convergence`, an `integrate`
argument; None means `not take_gradient`, True with `take_gradient=True` raises
`ValueError`): Step 4b judges every recorded step, from its stored
`mesh_quadratures` / `mesh_quadrature_errors` (no call of `f`; for the norm
family from its stored `reduced_errors`, the numbers it was accepted with, so
only the tolerance is new and no [N, D] array is reduced), against the same
reference as the batch, via `_failed_record_positions` (the re-check's one
read of the device: the positions of the failures that `_splittable` allows;
the loop learns from their count, on the host, whether any is split), and
those failures are split in the same mesh update (`extra_split_idxs`)
and removed from the record (`_remove_failed_record_panels`). Step 7 and the
final `converged` check use the same `_record_error_check`, so in absolute
mode the three cannot disagree through rounding (in cumulative mode the
tolerance, the norm of the running sums, is reduced as rows of different
arrays and can still differ in its last bit for D > 1). A fixed
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

`error_ratios_2steps`: the pair ratio of consecutive step pairs, computed and
used only by `_get_optimal_mesh` to merge pairs (it passes `pair_errors` to
`_compute_error_ratios`, which skips the pair ratio without them): a pair error divided by the tolerance the
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
  (log2 ratio quantized to 2^-48), one entry per node (0 at a repeated
  fraction, so they apply to the stored values as they are: no copy of a
  contiguous float64 record; a strided one, such as a transposed output of
  `f`, is made contiguous per chunk, since `bmm` rounds differently on it), cached
  on the solver (`_merge_weight_cache`, reset in `_set_solver_dtype`; a
  one-shot call pays 0.5-1.2 ms to fill it). The widths come from the final
  mesh; the nodes are not read. With T > 1 it falls
  back to `"power_law"` with a warning.
- `"power_law"` (PadaQuad 1.3.0 and 1.4.0): summed errors, cutoff `remove_cut / 2^(k-1)`
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
  - *Merging* (optimal mesh): the pair's shared mesh point is removed and
    the merged subinterval carries the pair's summed contribution and error;
    no nodes are built for it (the same for both solvers).
- **Variable** (`quadrature/variable.py`): weights computed from each
  panel's actual nodes by `method.weights(nodes)`
  (`VariableMethodClass.weight_function`; for `interpolatory3_variable`
  the Sanderse-Veldman weights, `_interpolatory3_weights`).
  - *Splitting*: the integration loop does NOT reuse evaluations. Like the
    uniform solver it inserts the midpoint into the mesh and places fresh
    nodes in both halves with `_compute_nodes`, at the node fractions, so
    `interpolatory3_variable` always runs as Simpson's rule (a = 1/2).
    The uncalled reuse method `_evaluate_adaptive_nodes` was deleted; what
    reuse would need: `docs/issues/variable-sampling-reuse.md`.
  - *Merging*: as for uniform sampling.

### Memory management

- **The batch size of `integrate`** (no benchmark; the design, plan and
  reviews were deleted once done and are in commit `6b3cf42`,
  `docs/issues/2026-11-17-*.md` and `2026-11-18-memory-state-plan.md`; the
  two reviews of the code in `d0bcbdc` and `59b7f11`). Per call, `_batch` (largest batch known
  to fit) and `_batch_converged` (will not grow again), set at the start of
  `integrate`: from `max_batch` (call, or constructor when no state) →
  converged; from `state=` (`IntegrationState`, `results.py`); else
  `ceil(min(n_first, _LARGEST_FIRST_CALL = 10,000) / _GROWTH)`, not converged,
  so the first call of `f` is the first pass, at most 10,000 points (no test
  calls and no output-check call: every point `f` is called at belongs to the
  integration; the output's shape is checked in `_call_f`, the
  `error_integral_reference` shape in Step 4, the `trade_speed_for_memory`
  warning after the loop).
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
  `get_max_batch(evaluation_buffer=X)` subtracts 2X bytes from `usable`
  (`reserved`, GB, in `_get_max_f_evals` and the benchmark's result; not in
  its test calls); `integrate` does not.
- `_RK_integral` forms `b * y` in blocks of at most `_BLOCK_BYTES`
  (128 MiB) of rows, bit-identical to one sum, so the quadrature calculation
  needs no temporary as large as the values.
- Out-of-memory errors are recognized by `_is_out_of_memory` (GPU OOM, CPU
  `DefaultCPUAllocator` refusal, CUDA "out of memory" and CUDA-library
  `*ALLOC*_FAILED` statuses, `MemoryError`). Every call of `f` in the
  integration loop goes through `_call_f` (memory.py), the one place that
  catches them: it returns None after `_release_memory` (CUDA cache only on
  a CUDA solver, and only for an error from outside PyTorch's allocator), or
  raises `torch.OutOfMemoryError` ("cannot be reduced") when the caller cannot
  shrink. Its two callers retry smaller: `_evaluate_f_in_batches`
  (`take_gradient=False`) and the `while output is None` loop of
  `_evaluate_f_on_full_nodes` (`take_gradient=True`). It is also the one
  check of the output's shape (`ValueError` unless [N, D], one row per point).
- **A pass's values are held once** (`take_gradient=False`):
  when the values stay on the solver's device for the pass (`values_device`
  is it, or None with no buffer) and f's output does not require grad, `_evaluate_f_on_split_nodes` writes each call's whole subintervals
  into one [K, C, D] array for the pass (`in_place`; K is known before the
  first call, since nothing is split during a pass) and integrates from its
  rows, with no list of calls to join (when one call covers the pass, its
  output is the array, no copy; `_keep_rows` copies from a view such as it); `_adaptively_increase_mesh` keeps the
  accepted rows by moving them to the front of that array
  (`_keep_rows(x, keep_idx, in_place=not take_gradient)`, integer positions,
  blocks of `_BLOCK_BYTES`; a
  view, so no copy), or keeps the array as it is when all are accepted; the
  record appends that as its chunk (copied by `_record_results` when it is a
  view of a larger array, so the pass's array is freed), and the loop deletes
  the batch's values (`kept_y`, `intermediate_results`) after recording. An out-of-memory error allocating the pass's array is raised
  as `torch.OutOfMemoryError` naming `trade_speed_for_memory` and
  `expandable_segments` (user guide `ug-pass-array-oom`). The settings'
  memory and time are measured by
  `examples/speed_test/trade_speed_for_memory.py` (every setting, from speed to
  memory, on a light integrand, two Fourier-feature networks of ~10,000 and
  ~1,000,000 parameters, with every setting and no `max_batch` (the batch grows until f
  runs out of memory), and a `SteppedField` of 512² outputs with
  `FIELD_SETTINGS` (`max_batch` 3,000, 4,000, 5,550 × `trade_speed_for_memory`,
  `record_evaluations=False`; on a 40 GB A100 the default runs out of memory in
  f at 4,000 and 5,550 and retries smaller, rendered as "ran out of memory" in bold); out-of-memory
  rows are recorded; asserts the same integral to 10×
  the tolerance and records whether it is the same bit for bit; writes `docs/performance/memory_options.json`, rendered by its
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
  device, ends equal to the bounds bit for bit, or None with
  `build_state_mesh=False`, which skips the build; its batch size and converged
  flag), or `mesh=` / `max_batch=` directly. `mesh=` with `state=` raises. A
  state whose mesh is None needs both bounds (the error names the switch).
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

- `atol` / `rtol` (integration error control): Python floats, checked to be
  one finite number >= 0 in `SolverBase.__init__`, rounded by each operation
  to the dtype of the tensor they meet, so a float32 solver given float64
  integrand values uses them unrounded.
- `float64` and `float32` are the supported runtime dtypes. `float16`
  is refused at construction (Bug B4 fix): its ~1e-3 precision floor
  exceeds typical adaptive tolerances, so the previous loose-tolerance
  path silently produced wrong answers.
- Methods are **converted once per process and shared**: `_get_method`
  returns `registry[name].to(device, dtype)`, a copy (never sharing memory
  with the registry entry) built on first use and kept in `_CONVERTED`
  (`methods/_base.py`), a `WeakKeyDictionary` of the source object (so a
  copied or unpickled solver's conversions go with it) to a dict by device and
  dtype;
  every solver of that method, device and dtype uses the same object.
  Nothing converts in place (`clone`, `to_dtype` and `to_device` were
  removed; `_Tableau` is frozen), so the registries never change. A method's
  tensors must never be written in place: the write would reach every solver
  sharing it (PyTorch has no read-only tensors; `tests/test_solver_isolation.py`
  checks that no integration writes, by the tensors' version counters).
  `_integration_on_device` swaps `self.method` for its conversion to
  `result_device` and back, without moving any tensor.

## Dependencies

- **Runtime**: torch, numpy, scipy, psutil.
- **Dev**: pytest, pytest-cov, mypy, pre-commit, ruff, typos,
  torchdiffeq (for the speed-test benchmark only).
