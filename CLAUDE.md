# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

**torchpathdiffeq** is a PyTorch library for **adaptive numerical
quadrature**: it computes definite integrals
$\int_a^b f(t)\,dt$ for a known integrand $f$ by evaluating many
quadrature panels in parallel batches on GPU, with full autograd through
the integration loop. v0.2.0 (Alpha), Python 3.10+.

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

# Run all tests (1020 currently)
.venv/bin/pytest tests/ -v

# Run all tests with coverage
.venv/bin/pytest --cov=torchpathdiffeq --cov-report=xml

# Run a single test file
.venv/bin/pytest tests/test_integrals.py -v

# Run a specific parametrized case
.venv/bin/pytest tests/test_integrals.py -k "dopri5 and damped_sine" -v

# Run with snapshot regeneration (after intentional behavior changes)
TPD_REGENERATE_SNAPSHOTS=1 .venv/bin/pytest tests/test_snapshots.py

# Lint with ruff
.venv/bin/ruff check torchpathdiffeq/

# Type check
.venv/bin/mypy torchpathdiffeq/
```

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
| `test_tableau.py` | Tableau b weights sum to 1 (uniform and variable methods) |
| `test_path_integral.py` | `integrate()` wrapper matches direct solver construction |
| `test_chemistry.py` | Wolf-Schlegel 2D potential: parallel vs scipy.integrate.quad |
| `test_exactness.py` | **Polynomial-exactness** for every method (each method is exact through its claimed degree, e.g. gk21 through degree 31) |
| `test_scipy_agreement.py` | Cross-validation against `scipy.integrate.quad` on canonical smooth integrands |
| `test_convergence_rate.py` | Empirical convergence-rate slope matches each method's theoretical rate |
| `test_autodiff_consistency.py` | $\nabla_\theta\int f_\theta dt$ via backprop equals $\int \nabla_\theta f_\theta dt$ |
| `test_error_indicator.py` | Behavior of absolute vs cumulative error indicators (Bug B8 investigation) |
| `test_bug_regressions.py` | Regression tests for bugs B1, B2, B4, B6 fixed in Phase 1 |
| `test_snapshots.py` | Golden-value regression net for the file-split refactor; CPU-only float64 |
| `test_public_api.py` | Smoke test of every public symbol and `IntegrationResult` fields |
| `test_rk_integral.py` | `_RK_integral`: constant / linear / multi-step / multi-dim / zero-width / variable-b inputs |
| `test_base_and_methods.py` | `get_sampling_type`, `_Tableau` dtype conversion, `_get_method` factory, weight formulas |
| `test_error_computation.py` | `_error_norm`, `_rec_remove`, `_compute_error_ratios_absolute`/`_cumulative` |
| `test_interpolation.py` | `_compute_nodes`: endpoints, shapes, monotonicity, tableau-c matching |
| `test_sorted_insert.py` | `_get_sorted_indices` and `_insert_sorted_results` |
| `test_base_solver.py` | `SolverBase._set_dtype`, `set_dtype_by_input`, `_check_variables`, `_integral_loss` |
| `test_methods_extra.py` | Variable subclass `tableau_b`, `to_device`, dtype conversion |
| `test_rk_solver_methods.py` | `adaptive_quadrature` factory, `_get_tableau_b`, `_get_num_tableau_c` |
| `test_adaptive_steps.py` | `_adaptively_increase_mesh`: pass/fail/mixed error ratios, midpoint placement |
| `test_record_and_sort.py` | `_record_results` and `_sort_record` |
| `test_evaluate_and_merge.py` | `_evaluate_adaptive_nodes` + `_merge_excess_nodes` for both uniform and variable solvers |
| `test_prune_and_optimal.py` | `_prune_excess_mesh`, `_get_optimal_mesh` |
| `test_y0_and_f_args.py` | `y0` additive offset (`result.integral = y0 + ∫f`) and `f_args` forwarding to `f(t, *f_args)` |
| `test_gradient.py` / `test_gradient_integration.py` | Per-batch backward, learnable integrand training loops |

**Snapshot tests** are the regression net for internal refactors: the
file `tests/test_snapshots_data.json` pins `integral`, `integral_error`,
and `n_optimal_mesh` for every (method × integrand × tolerance) cell.
Regenerate with `TPD_REGENERATE_SNAPSHOTS=1 pytest tests/test_snapshots.py`.

## Architecture

### Package layout

```
torchpathdiffeq/
├── __init__.py            # public API exports
├── base.py                # SolverBase, steps enum, get_sampling_type
├── results.py             # IntegrationResult, MethodOutput dataclasses
├── methods/               # quadrature method registries
│   ├── _base.py             # _Tableau, MethodClass
│   ├── runge_kutta.py       # adaptive_heun, fehlberg2, bosh3, dopri5
│   ├── gauss_kronrod.py     # gk7, gk15, gk21, gk31  (with builder)
│   ├── clenshaw_curtis.py   # cc5, cc9, cc17, cc33, cc65  (with FFT-based weights)
│   └── interpolatory.py     # variable adaptive_heun, interpolatory3_variable
├── quadrature/            # adaptive integration engine
│   ├── base.py              # AdaptiveQuadrature ABC + main integrate loop
│   ├── uniform.py           # _UniformAdaptiveQuadratureBase
│   └── variable.py          # _VariableAdaptiveQuadratureBase
├── runge_kutta.py         # _RK_integral, UniformAdaptiveQuadrature,
│                          # VariableAdaptiveQuadrature, adaptive_quadrature factory
├── integrate.py           # integrate() free function (one-shot wrapper)
├── examples.py            # integrand_dict (test integrands) + wolf_schlegel
└── distributed.py         # multi-GPU / SLURM support (internal)
```

### Class hierarchy

```
DistributedEnvironment (distributed.py)
  └── SolverBase (base.py)
        └── AdaptiveQuadrature (quadrature/base.py)
              ├── _UniformAdaptiveQuadratureBase (quadrature/uniform.py)
              │     └── UniformAdaptiveQuadrature (runge_kutta.py)
              └── _VariableAdaptiveQuadratureBase (quadrature/variable.py)
                    └── VariableAdaptiveQuadrature (runge_kutta.py)
```

The two concrete classes (`UniformAdaptiveQuadrature` and
`VariableAdaptiveQuadrature`) are what user code instantiates, either
via `adaptive_quadrature(...)` factory or directly. They add the RK-
specific `_calculate_integral` to whichever sampling-mode base they
inherit from.

### Public API

```python
from torchpathdiffeq import (
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
methods/interpolatory.py:  adaptive_heun (variable), interpolatory3_variable
```

For smooth integrands at academic-grade precision, prefer `gk21`
(default) or `cc33`. RK methods are kept for backwards-compatibility and
as low-order baselines. The low-point methods (`gk7`, `cc5`, `cc9`) fill
the gap between the RK rules and the headline `gk15`/`cc17`: useful when
the adaptive controller refines aggressively and a high-node rule wastes
evaluations on smooth regions.

### Tensor shape conventions

- **N**: number of integration steps in a batch
- **C**: number of quadrature points per step (from the rule's tableau,
  e.g. 4 for bosh3, 7 for dopri5, 23 for gk21 with endpoint padding)
- **T**: dimensionality of time (usually 1, but supports multi-D)
- **D**: dimensionality of `f`'s output

Key tensors: `nodes: [N, C, T]`, `y: [N, C, D]`, `tableau_b: [1, C, 1]` or
`[N, C, 1]`, `y0: [D]`, mesh barriers: `[M, T]`.

### Core algorithm (`AdaptiveQuadrature.integrate()` in `quadrature/base.py`)

`mesh` is the boundary array dividing `[mesh_init, mesh_final]` into
integration steps (panels). Between consecutive barriers, C
quadrature points (`nodes`) are placed per the tableau. `mesh_trackers`
is a boolean array where `True` means the panel still needs evaluation.

1. **Initialize barriers.** When `mesh=None`, generate
   ~$\sqrt{N_\text{init}}$ evenly-spaced top-level segments with random
   sub-barriers (random by default — uniform aliases on
   uniform-feature integrands). When `reuse_mesh=True`, seed from the
   cached `mesh_optimal` of the previous successful run.
2. **Main loop.** While any `mesh_trackers` are `True`:
   - Determine `max_steps` from GPU memory budget or explicit `max_batch`.
   - Select up to `max_steps` unevaluated panels.
   - Place C nodes per panel via `_compute_nodes`.
   - Evaluate `f` on all flattened nodes; reshape to `[N, C, D]`.
   - Compute integral + error via `_calculate_integral` (RK weighted
     sum using `b` and `b_error` tableaux).
   - Compute error ratios (absolute or cumulative mode).
   - `_adaptively_increase_mesh`: panels with ratio `< 1` are accepted;
     panels with ratio `>= 1` are split at midpoint into two
     sub-panels for re-evaluation.
   - Record accepted results into the running record.
   - If `take_gradient`: per-batch `loss.backward()`.
3. **Post-convergence.** Sort the record, compute `mesh_optimal` via
   `_get_optimal_mesh` (prune low-error pairs, refine
   high-error gaps), cache for reuse on the next call.

### Error computation modes

Two modes selected by `use_absolute_error_ratio`:
- **Absolute** (default): `error_ratio = |step_error| / (atol + rtol * |total_integral|)`. Every panel uses the same denominator. Best for path integrals where the total is the meaningful quantity.
- **Cumulative**: `error_ratio = |step_error| / (atol + rtol * |cumsum_to_step|)`. The denominator grows with the running integral, mimicking traditional ODE error control. Per-panel ratios *decrease* as integration progresses.

`error_ratios_2steps`: combined error of consecutive step pairs, used for
merging. When `error_ratio_2steps < remove_cut` (default 0.1), pairs are
merged. `_rec_remove` ensures no two adjacent pairs are both flagged.

### Uniform vs variable sampling

- **Uniform** (`quadrature/uniform.py`): nodes at fixed fractional
  positions within each panel (tableau `c` values). Tableau `b` values
  are constants.
  - *Splitting*: discards old evaluations, places fresh quadrature
    points at standard `c` positions in the new sub-panels.
  - *Merging*: re-interpolates C nodes at standard `c` positions
    spanning `[A.start, B.end]`.
- **Variable** (`quadrature/variable.py`): nodes at arbitrary positions.
  Tableau `b` values computed dynamically via
  `method.tableau_b(c)` (Sanderse-Veldman formulas).
  - *Splitting*: reuses existing evaluations by inserting new midpoints
    between each pair of consecutive points and interleaving old + new
    into a 2C-length array, reshaped into two C-point sub-panels.
    Avoids redundant `f` calls.
  - *Merging*: concatenates points from both panels (C + C-1 = 2C-1,
    shared boundary not duplicated), then subsamples every other point
    to get C. The `tableau_b(c)` call recomputes the right weights for
    the new layout.

### Memory management

- `_setup_memory_checks` benchmarks `f` with increasing `N`, measures
  per-evaluation memory cost (with a 2.1× safety factor).
- `_get_max_f_evals = usable_memory / per_eval_size`.
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

### Warm-starting (`reuse_mesh=True`)

- Solver caches `mesh_optimal` from the previous successful call.
- On the next call with `reuse_mesh=True`, the cached mesh is filtered
  to `[mesh_init, mesh_final]`, padded if needed, and used as the
  initial mesh.
- Integrand identity is sanity-checked via `id(f)` (Phase 1 fix; the
  prior `__name__` check collided across all lambdas). On id mismatch
  the solver warns but proceeds.
- Default is `reuse_mesh=False` so that calling the same solver on a
  new integrand never silently reuses a stale mesh.

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
  `MethodClass.clone()`); dtype/device mutations stay isolated to one
  solver and never propagate via shared singletons. The previous
  global-singleton hazard (where a float32 round-trip could
  irreversibly degrade `UNIFORM_METHODS["dopri5"].tableau.b` for the
  rest of the test session) is gone.

## Dependencies

- **Runtime**: torch, numpy, scipy, einops, psutil.
- **Dev**: pytest, pytest-cov, mypy, pre-commit, ruff, typos,
  torchdiffeq (for the speed-test benchmark only).
