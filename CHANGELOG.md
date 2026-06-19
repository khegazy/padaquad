# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-06-16

### Changed

- **Accepted integration panels are now re-validated against the drifting error tolerance.** In absolute-error mode the accept/reject denominator (`max(atol, rtol * |I|)`) depends on the running integral, which changes as more panels are evaluated — so a panel accepted early could later violate the *current* tolerance yet remain accepted, silently breaking the error guarantee. Each iteration the solver now re-checks every already-accepted (recorded) panel against the same `reference_integral` the current batch uses; any that now fail are removed from the record and bisected in the same mesh update (`_adaptively_increase_mesh` gained an internal `extra_split_idxs` argument), and their refined children are re-evaluated. At convergence **every** recorded panel satisfies the final tolerance. Runs only on the `take_gradient=False` path (where no per-batch backward can be double-counted and the full-integral denominator is available) and is skipped when a fixed `error_integral_reference` pins the denominator. The `take_gradient=True` path is unchanged. **This can change the converged mesh and integral** for multi-batch, drifting-denominator integrands (notably cancelling ones, where the denominator collapses toward `atol` as the integral cancels).
- **Snapshot golden values are keyed per `take_gradient` mode and regenerated.** The two modes legitimately diverge once the re-check engages (they only ever coincided when every panel fit in a single batch); `take_gradient=True` values are unchanged.

### Fixed

- **Default-loss detection in the record bookkeeping** compared bound methods by identity (`loss_fxn is self._integral_loss`), but `obj.m is obj.m` is `False` for bound methods, so the recorded `loss` was never rebuilt after a panel removal. It now compares the underlying function; a custom `loss_fxn` is still left as-is.
- **`_rec_remove` recursion overflow.** The post-convergence prune helper was tail-recursive and hit Python's recursion limit on the large, finely-refined meshes the re-check produces for cancelling integrands at tight `atol`. Converted to an equivalent `while` loop — identical output (snapshots unaffected), no depth limit.

## [1.2.0] - 2026-06-14

### Changed

- **Error tolerance is now computed against the complete integral (notable for memory-heavy integrands).** The default `take_gradient=False` evaluation path (`_evaluate_f_on_split_nodes`) now evaluates *every* pending panel in a single pass before any accept/reject decision, so the integral is complete the first time a panel is judged. The absolute-mode tolerance denominator (`max(atol, rtol * |I|)`) therefore uses the full integral from the start, instead of the running (incomplete) integral that accumulated batch by batch. This yields better mesh-split decisions, but **can change integration results for integrals large enough to span multiple batches** (`total_nodes > max_batch`): the early panels are now judged against the true total rather than a too-small partial sum, so the converged mesh and integral can differ from prior versions. Small integrals that fit in one batch are unaffected. `max_batch` continues to bound memory by chunking the `f` calls; it no longer fragments the error decision.

### Added

- **`conserve_memory`** argument on `integrate()` (and the public `integrate()` wrapper), default `False`. When `True`, the solver uses the previous memory-conserving path (`_evaluate_f_on_split_residual_nodes`): it evaluates a bounded number of panels per loop iteration and carries a partial-panel residual forward across iterations, so peak memory is bounded by `max_batch` rather than by the whole mesh. This restores the pre-1.2.0 incremental behavior (and the running-integral error denominator it implies). Use it when the full set of node evaluations does not fit in memory.

### Fixed

- **Bug:** `_evaluate_f_on_split_nodes` iterated an undefined `tracked_list` (instead of `tracked_lists`) when concatenating tracked variables across batches, raising `NameError` whenever an integrand emitted tracked variables on the default path.

## [1.1.1] - 2026-06-13

### Added

- **`result_device`** argument on `integrate()` (and the public `integrate()` wrapper), default `'cpu'`: the large per-step records (`nodes`, `y`, `h`, `mesh_quadrature_errors`, `error_ratios`, `integral_error`, `loss`, `tracked_variables`) are detached and moved to `result_device` as they are recorded, so they no longer accumulate on the integration device. This bounds GPU memory on fine meshes — previously the full record grew on the GPU for the entire run even though most fields are only needed in the returned result. The two fields read back inside the loop (`integral` and `mesh_quadratures`) stay on the integration device. When `take_gradient=False` the autograd graph is retained across the device move, so the returned integral remains differentiable (gradients flow back to parameters on the integration device).

### Changed

- **Record ordering is now by mesh position (robust to vector-valued time).** Recorded panels are merged/sorted by each panel's position in the mesh (its left barrier looked up in a `mesh_indices` map) instead of by the first node's first time-coordinate. The previous coordinate-0 key is not a valid total order when `t` is a vector (`T > 1`, e.g. path integrals) and could mis-order the record; the mesh order is correct for any `T`. For scalar time (`T == 1`) the order is unchanged and results are byte-identical (snapshots unaffected).
- **Default result device (notable for CUDA users):** with `result_device` defaulting to `'cpu'`, `integrate()` now returns the large result tensors on CPU by default; pass `result_device='cuda'` (or your device) to keep them on the GPU. The "mesh family" (`mesh_optimal`, `mesh_init`, `mesh_final`) always stays on the integration device because `mesh_optimal` is the warm-start mesh fed back into the solver and its endpoints are `mesh_init`/`mesh_final`. On a CPU-only solver `result_device` is a no-op.
- **Device-consistent internals:** the error-ratio helpers and adaptive-mesh refinement (`_adaptively_increase_mesh`, `_get_optimal_mesh`) are now device-following (tolerances and allocations follow their inputs), and the merge/sort paths place index tensors on each field's device, so a record split across devices stays consistent.
- **Coordinate-0 assumptions removed for multi-dimensional `t`:** post-refinement/record monotonicity assertions are reframed (the 1-D strictly-ascending checks are retained, guarded to `T == 1`), initial sub-barriers are generated along the segment vector, and the `reuse_mesh` warm-start filter uses an all-coordinate bounding box instead of a coordinate-0 range.

### Fixed

- The `max_path_change` early-exit `IntegrationResult` is now device-consistent with the normal-completion path.

### Removed

- Internal `_get_sorted_indices` (coordinate-based binary-search merge), superseded by mesh-position placement (`_mesh_order` + `_merge_positions`).

## [1.1.0] - 2026-06-09

### Added

- **`error_integral_reference`** argument on `integrate()`: in absolute-error mode the rtol denominator is the running (incomplete) integral, which over-refines early panels when the integrand's mass is concentrated at later times. This argument accepts a Python float or tensor close to the true total, using it as the denominator from the first batch onward. Has no effect in cumulative mode.
- **`error_norm`** argument (constructor + per-call override): pluggable vector-error norm schemes replacing the fixed RMS reduction, modeled on `scipy.integrate.quad_vec`'s `norm`:
  - Norm family (`"2"` default, `"max"`, `"rms"`, or a callable) — reduce-then-compare: collapse the error vector with the named norm, compare to the tolerance.
  - `"failure_fraction"` — per-component control: each output element is compared against its own tolerance; a panel is accepted when the fraction of failing elements is ≤ `mesh_failure_tolerance` (default `0.0`, requiring every element to pass).
- **Rounding floor** (`max(tol, 50 * eps * |step|)`): refinement halts at the working-dtype precision wall instead of refining indefinitely when the requested tolerance is below the dtype's resolution.
- **`error_on_nonfinite`** flag (default `True`): raises a `ValueError` naming the offending `t` when `f` returns NaN/Inf, preventing silent wrong results.
- **`max_adaptive_splits`**: configurable ceiling on adaptive refinement depth (added in v1.0.2; documented here for completeness).

### Changed

- **Error tolerance formula (breaking for fine tolerances):** the accept/reject threshold for the norm family has changed from `atol + rtol * |I|` (additive) to `max(atol, rtol * |I|)` (maximum), aligning with `scipy.integrate`. This tightens tolerance when both terms are comparable; snapshot values have been regenerated. The `failure_fraction` scheme retains the additive form.
- `atol` and `rtol` are now stored as dtype-aware tensors in `SolverBase` and re-cast in `_set_dtype`.
- `y0`, `mesh_init`, and `mesh_final` defaults now honor the solver's dtype rather than hard-coding float64.
- `max_batch < C` (`take_gradient=False`): sub-panel batch budgets are now supported. The split path processes one panel per iteration, evaluating its `C` nodes across several `max_batch`-sized sub-batches and carrying remainders forward. The final integral is bit-identical to a `max_batch ≥ C` run with the same seeded mesh. `max_batch == 0` at the dispatcher is rescued to `1` (with a warning); a direct unit call to `_evaluate_f_on_split_nodes` raises `AssertionError`. `take_gradient=True` below `C` continues to raise an error.
- Non-finite error ratios (from singular integrands) are routed into the accept mask rather than hanging the refinement loop indefinitely.
- GPU device handling: user-provided meshes, `f_args` tensors, and internal assertions are now consistently placed on `self.device`, fixing 850 test failures on CUDA machines.
- `max_batch` is cached between calls when `f` is the same function, avoiding a full re-benchmark on every `integrate()` call.
- Out-of-memory errors during integration and memory benchmarking are caught; `max_batch` is lowered and the evaluation retried.

### Fixed

- **Bug:** the integral loss was computed as `sum(step²)` instead of `(integral)²`.
- **Bug:** misplaced parenthesis in the tolerance expression.
- **Bug:** `VariableAdaptiveQuadrature` did not call `method.to_device(self.device)` at construction, leaving method tensors on CPU when the solver was on CUDA.
- **Bug:** `_merge_excess_nodes` used `np.allclose` for a device-agnostic sanity check, which fails on CUDA tensors; replaced with `torch.allclose`.
- **Bug:** non-finite error ratios caused an infinite adaptive refinement loop.

### Removed

- `error_calc_idx` argument (superseded by the callable form of `error_norm`).

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
