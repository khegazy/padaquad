# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-05-30

### Added

- `max_adaptive_splits`: a configurable ceiling on adaptive refinement depth.
  Each panel tracks how many times it has been split (children of a depth-`N`
  panel are depth `N+1`); once a panel reaches `max_adaptive_splits` it is
  accepted even if it still fails the error tolerance, instead of being split
  further. Defaults to `None` (uncapped). Can be set at solver construction or
  passed to the integration call, with the per-call value taking priority.
  Applies to both uniform and variable sampling.

### Changed

- Capped panels retain their failing `error_ratios` (>= 1) in the result, so
  regions that hit the depth cap are visible; the run still completes normally.

## [1.0.1] - 2026-05-30

### Added

- Tracked variables: the integrand `f` may now return a 2-tuple `(integrand, tracked_variables)` to emit extra per-node quantities that are evaluated at every quadrature node but **not** integrated. They are carried through the adaptive loop and returned at the accepted nodes in the new `IntegrationResult.tracked_variables` field, a tuple of tensors each shaped `[N, C, *var_dims]` and aligned with `nodes`/`y`. Values are detached (diagnostic-only). Works across both sampling modes, both `take_gradient` paths, and `float32`/`float64`, and supports non-float (e.g. integer/boolean) tracked tensors.

### Changed

- `IntegrationResult` gained the optional `tracked_variables` field (defaults to `None`).
- The sorted-insert step of the record is now dtype/device-aware, enabling non-float tracked variables to be recorded and sorted alongside the integrand evaluations.

### Compatibility

- Fully backward compatible and opt-in: an integrand returning a bare tensor behaves exactly as before, with `result.tracked_variables = None`. A single tracked tensor is wrapped into a 1-tuple, and `(integrand, None)` is accepted.

## [1.0.0] - 2026-05-30

First stable release. The project, formerly `torchpathdiffeq` (an ODE / path-integral library), has been rebuilt and renamed to **`padaquad`**, a PyTorch library for **parallelized adaptive numerical quadrature**: it computes definite integrals $\int_a^b f(t)\,dt$ for a known integrand `f` by evaluating many quadrature panels in parallel batches on GPU/CPU, with full autograd through the integration loop.

The integrand `f` depends only on `t` (and optional extra args), **not** on accumulated state `y`. This is numerical quadrature, not ODE solving — for state-coupled $\dot y = f(t, y)$ use `torchdiffeq` / `torchode` / `diffrax`.

### Added

- Parallel, memory-aware batched evaluation: quadrature panels are evaluated in batches sized automatically to the available GPU/CPU memory budget.
- Adaptive mesh refinement: panels that miss tolerance are split at their midpoint and re-evaluated; over-resolved regions are merged. The converged mesh can be cached and warm-started (`reuse_mesh=True`) for repeated calls on the same integrand.
- Method portfolio (default `gk21`):
  - Gauss–Kronrod: `gk7`, `gk15`, `gk21`, `gk31`
  - Clenshaw–Curtis: `cc5`, `cc9`, `cc17`, `cc33`, `cc65`
  - Runge–Kutta baselines: `adaptive_heun`, `fehlberg2`, `bosh3`, `dopri5`
  - Variable-sampling rules: `adaptive_heun`, `interpolatory3_variable`
- Uniform and variable sampling modes (fixed tableau-`c` node positions, or arbitrary node positions with weights computed dynamically).
- Full autograd through integration, including a memory-frugal per-batch-backward mode (`take_gradient`) for training learnable integrands whose full graph would not fit in memory.
- `y0` additive offset (`result.integral = y0 + ∫f`) and positional `f_args` forwarding to `f(t, *f_args)`.
- Public API: `integrate`, `adaptive_quadrature`, `UniformAdaptiveQuadrature`, `VariableAdaptiveQuadrature`, `IntegrationResult`, `UNIFORM_METHODS`, `VARIABLE_METHODS`, `integrand_dict`, `wolf_schlegel`, `steps`.

### Changed

- **Renamed package `torchpathdiffeq` → `padaquad`.** Update imports accordingly (`from padaquad import ...`).
- Reframed the public API around quadrature: `IntegrationResult` exposes integral- and mesh-oriented fields (`integral`, `integral_error`, `nodes`, `mesh_optimal`, `mesh_init`, `mesh_final`, `mesh_quadratures`, `mesh_quadrature_errors`, `error_ratios`, …).
- `float64` and `float32` are the supported runtime dtypes; `float16` is refused at construction.

### Removed

- ODE-centric framing and semantics. `padaquad` does not integrate state-coupled ODEs; use a dedicated ODE library for those.

## [0.0.2]

### Added

- Added `.pre-commit-config.yaml`

### Changed

- Use `ruff` for formatting

## [0.0.1]

### Added

- The initial release!
