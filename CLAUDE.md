# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

**padaquad** is a PyTorch library for **adaptive numerical
quadrature**: it computes definite integrals
$\int_a^b f(t)\,dt$ for a known integrand $f$ by evaluating many
quadrature panels in parallel batches on GPU, with full autograd through
the integration loop. Version 1.3.0 in `pyproject.toml` (Alpha), Python 3.11+.

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

# Run all tests (3242 pass and 138 skip on a CPU-only Mac; GPU-only tests skip there)
.venv/bin/pytest tests/ -v

# Run all tests with coverage
.venv/bin/pytest --cov=padaquad --cov-report=xml

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
```

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
| `test_bug_regressions.py` | Regression tests for bugs B1, B4, B6 fixed in Phase 1, and the `same_f` flag |
| `test_snapshots.py` | Golden-value regression net for the file-split refactor; CPU-only float64 |
| `test_public_api.py` | Smoke test of every public symbol and `IntegrationResult` fields |
| `test_rk_integral.py` | `_RK_integral`: constant / linear / multi-step / multi-dim / zero-width / variable-b inputs |
| `test_base_and_methods.py` | `get_sampling_type`, `_Tableau` dtype conversion, `_get_method` factory (unknown name raises `ValueError`), `error_exponent`, weight formulas |
| `test_error_computation.py` | `_error_norm`, `_rec_remove`, `_compute_error_ratios_absolute`/`_cumulative` |
| `test_interpolation.py` | `_compute_nodes`: endpoints, shapes, monotonicity, tableau-c matching |
| `test_sorted_insert.py` | `_get_sorted_indices` and `_insert_sorted_results` |
| `test_base_solver.py` | `SolverBase._set_dtype`, `set_dtype_by_input`, `_check_variables`, `_integral_loss`, device selection (`_resolve_device`, indexed GPUs, no global side effects) |
| `test_methods_extra.py` | Variable-mode `adaptive_heun` (the uniform heun object) weights, the generic `VariableMethodClass` (normalized nodes reach the weight function, shape check), node-fraction validation, `to_device`, dtype conversion |
| `test_solver_methods.py` | `adaptive_quadrature` factory, `method.weights` through both solvers, per-step node count `solver.C` |
| `test_adaptive_steps.py` | `_adaptively_increase_mesh`: pass/fail/mixed error ratios, midpoint placement |
| `test_record_and_sort.py` | `_record_results` and `_sort_record` |
| `test_evaluate_and_merge.py` | `_evaluate_adaptive_nodes` + `_merge_excess_nodes` for both uniform and variable solvers |
| `test_prune_and_optimal.py` | `_prune_excess_mesh`, `_get_optimal_mesh`, pair ratio = merged step's ratio; end to end: `mesh_optimal` re-evaluates with no failing step (every uniform method, cumulative mode), repeated warm starts prune to a fixed point |
| `test_y0_and_f_args.py` | `y0` additive offset (`result.integral = y0 + ∫f`) and `f_args` forwarding to `f(t, *f_args)` |
| `test_gradient.py` / `test_gradient_integration.py` | Per-batch backward, learnable integrand training loops |
| `test_error_integral_reference.py` | The `error_integral_reference` argument of `integrate()` |
| `test_error_masks.py` | Vector-error machinery (norms, `failure_fraction`, rounding floor) in float32 and float64 |
| `test_evaluate_f_on_mesh.py` | `_evaluate_f_on_mesh` and its evaluation paths |
| `test_examples_run.py` | Runs `examples/*.py`, every script in `examples/verification/`, and every ```python block of the README (except blocks preceded by `<!-- not-tested -->`) |
| `test_max_adaptive_splits.py` | The `max_adaptive_splits` depth cap |
| `test_memory_benchmark.py` | The memory benchmark, `_setup_memory_checks` |
| `test_methods_registry.py` | Registry consistency: table shapes and dtypes, `degree` / `error_exponent` types |
| `test_multi_d_integrand.py` | Integrands with D > 1 outputs |
| `test_oom_recovery.py` | The out-of-memory catch-and-retry path |
| `test_recheck_accepted_panels.py` | Re-check of accepted steps against the drifting tolerance |
| `test_solver_isolation.py` | Solvers do not share method objects or state |
| `test_tracked_variables.py` | The `tracked_variables` feature |
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
├── integrate.py           # integrate() free function + adaptive_quadrature factory
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
   uniform-feature integrands). When `same_f=True` and a mesh is
   stored, seed from the stored `mesh_optimal` of the last call that
   returned.
2. **Main loop.** While any `mesh_trackers` are `True`:
   - Determine `max_steps` from GPU memory budget or explicit `max_batch`.
   - Select up to `max_steps` unevaluated panels.
   - Place C nodes per panel via `_compute_nodes`.
   - Evaluate `f` on all flattened nodes; reshape to `[N, C, D]`.
   - Compute integral + error via `_calculate_integral` (weighted sum
     with the rule's `b` and `b_error` weights from `method.weights`; the
     error estimates are detached).
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
   error counts as failing); then compute `mesh_optimal` via
   `_get_optimal_mesh` (prune low-error pairs, refine
   high-error gaps), cache for reuse on the next call.

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

Both families accept a panel with any non-finite (NaN/Inf) element (splitting
cannot fix it; see `_adaptively_increase_mesh`) and apply a scipy-style
machine-precision **rounding floor** (`_round_floor`) so refinement stops once a
panel's error is at/below round-off. The scheme is set at construction
(`error_norm=`, `mesh_failure_tolerance=`) and overridable per call in
`integrate()`. `_compute_error_ratios` dispatches to `_error_ratios_norm` /
`_error_ratios_failure_fraction`; `_accept_reject_masks` turns the result into
keep/split masks consumed by `_adaptively_increase_mesh`.

`error_ratios_2steps`: summed error ratio S of consecutive step pairs, used
only by `_get_optimal_mesh` to merge pairs. A merged step of width 2h has an
error of about 2^(k-1)·S (the method's error estimate scales as h^k, k =
`error_exponent` on each method), so a pair is merged when
`S < remove_cut / 2^(k-1)` (`remove_cut` default 0.1): the merged step's
predicted ratio stays below `remove_cut`. S uses the tolerance the merged step
gets: in cumulative mode the right step's `|cumsum|` (the merged step ends
there). `_rec_remove` ensures no two adjacent pairs are both flagged, so one
call merges at most one level. The merged step records the pair's summed error,
so the add phase of `_get_optimal_mesh` does not split it (unless the rounding
floor sets the tolerance). The prediction assumes
the h^k regime; when the estimate is round-off (high-order methods, tight
tolerances, float32) S does not shrink with h and pairs rarely reach the cut.

### Uniform vs variable sampling

- **Uniform** (`quadrature/uniform.py`): nodes at fixed fractional
  positions within each panel (tableau `c` values). Tableau `b` values
  are constants.
  - *Splitting*: discards old evaluations, places fresh quadrature
    points at standard `c` positions in the new sub-panels.
  - *Merging*: re-interpolates C nodes at standard `c` positions
    spanning `[A.start, B.end]`.
- **Variable** (`quadrature/variable.py`): nodes at arbitrary positions.
  Weights computed from each panel's actual nodes by `method.weights(nodes)`
  (`VariableMethodClass.weight_function`; for `interpolatory3_variable`
  the Sanderse-Veldman weights, `_interpolatory3_weights`).
  - *Splitting*: reuses existing evaluations by inserting new midpoints
    between each pair of consecutive points and interleaving old + new
    into a 2C-length array, reshaped into two C-point sub-panels.
    Avoids redundant `f` calls.
  - *Merging*: concatenates points from both panels (C + C-1 = 2C-1,
    shared boundary not duplicated), then subsamples every other point
    to get C. `method.weights(nodes)` recomputes the right weights for
    the new layout.

### Memory management

- All of this lives in `quadrature/memory.py` (`_MemoryManagement` mixin,
  `_is_out_of_memory`); the batch loop `_evaluate_f_in_batches` stays in
  `quadrature/base.py`.
- `_setup_memory_checks` measures only c = `f_unit_mem_size`, the memory
  cost of ONE evaluation of `f`. It does not choose a batch size. It calls
  `f` on N = 1, 10, 100, … points, measures M (memory that call used) and
  sets c = s·(M + E)/N (s = `_GRADIENT_MEMORY_SAFETY_FACTOR` = 2.1 when
  `take_gradient`, else 1). M: CUDA peak counters; on Linux CPUs the
  resettable process peak (`/proc/self/clear_refs` + `VmHWM`) plus the
  kernel's counter error E; elsewhere an RSS before/after difference.
  Stops: c changes < 1 %, call slower than 1 s,
  N = 1e9, or 10·M > usable memory.
- Unless `max_batch` or `same_f=True` fixes it, the batch size is
  computed at every evaluation from live memory: `get_max_batch` →
  `_get_max_f_evals = usable_memory / c` (capped at 1e9).
- Out-of-memory errors are recognized by `_is_out_of_memory` (GPU OOM, CPU
  `DefaultCPUAllocator` refusal, CUDA "out of memory" and CUDA-library
  `*ALLOC*_FAILED` statuses, `MemoryError`);
  the shared batch loop `_evaluate_f_in_batches` shrinks the batch, and
  `_release_memory` cleans up (CUDA cache only on a CUDA solver).
- `_get_usable_memory = free - buffer`, where
  `buffer = (1 - total_mem_usage) * total_memory`.
- Supports both CUDA (`torch.cuda.mem_get_info`) and CPU
  (`psutil.virtual_memory`).

### Gradient / loss support

- `take_gradient=True` triggers `loss.backward()` after each accepted
  batch. This is essential when the autograd graph for the entire
  integration would not fit in GPU memory; per-batch backward
  accumulates gradients into `theta.grad` without holding the full
  graph.
- `loss_fxn` defaults to returning the integral itself
  (`_integral_loss` in `base.py`). Linear in the integral, so the
  default path is correctness-safe under per-batch backward.
- Results from each batch after backward are `.detach()`-ed before
  being added to the running record (so the graph for that batch can
  be released).

### `y0` additive offset

- `y0` is an optional initial value of the integral accumulator. The
  returned `result.integral` is `y0 + ∫f(t)dt`. Default is zeros.
- Per-batch `_calculate_integral` calls inside the loop use `y0=zeros`
  so each batch returns only its step contributions. The user-supplied
  `y0` is added once at the final `IntegrationResult`.
- `result.y0` echoes the value that was used (after dtype/device
  coercion) so callers can recover what offset was applied.
- `f_args` is a tuple forwarded positionally to the integrand:
  `f(t, *f_args)`. Used by `examples/pode/` for path parameters.

### Warm-starting (`same_f=True`)

- Every call that returns (converged or not) stores its mesh
  (`mesh_previous` = `mesh_optimal`) and its `max_batch`
  (`previous_max_batch`) together. A call that raises stores nothing.
- With `same_f=True` and stored values, the stored mesh (filtered
  to `[mesh_init, mesh_final]`, padded if needed) and `max_batch` replace
  the caller's `mesh` / `max_batch`, and the memory benchmark is skipped.
  With nothing stored (first call), the caller's values or the defaults
  are used.
- The solver does not identify integrands (no `id(f)` matching); the
  caller says when the integrand is the same. Default is
  `same_f=False`.
- A non-converged call (e.g. under `max_adaptive_splits`) stores a
  `mesh_optimal` that splits its failing steps, so repeated `same_f=True`
  calls refine further. In cumulative mode `_prune_excess_mesh` can instead
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
