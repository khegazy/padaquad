# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **`max_path_change`.** The constructor argument and the early exit it controlled are gone; `max_adaptive_splits` bounds refinement instead. Passing `max_path_change=` now raises `TypeError`. To migrate:
  - To limit how much a call refines, set `max_adaptive_splits`. The call returns `converged=False` when it stopped short of the tolerance, and repeated `same_f=True` calls refine further from `mesh_optimal`, which splits the failing steps (as the early exit's stored mesh did; in cumulative mode a failing step can instead be merged with a right neighbour whose running integral, and so tolerance, is much larger).
  - To see how much of the starting mesh failed, read `n_start_steps_split` / `percent_start_steps_split` (below). With `max_adaptive_splits=0` nothing is split, so read `converged`, or `(result.error_ratios >= 1).float().mean()`, the fraction of steps that failed when accepted (with `error_norm="failure_fraction"`, `error_ratios` holds failure fractions, so compare with `> mesh_failure_tolerance` instead).
  - What is lost is the conditional behaviour "refine fully only when few starting steps fail".
- **No positional constructor arguments.** `max_path_change` was the 4th positional constructor argument. Every constructor option is now keyword-only (see Changed), so passing any of them by position raises `TypeError`. A `max_adaptive_splits` that is not an integer (`int` or numpy integer) raises `TypeError` (a negative one `ValueError`), so a float meant for `max_path_change` cannot act as a cap.

- **Support for Python 3.10.** padaquad now requires Python 3.11 or newer (`requires-python = ">=3.11"`), and CI tests Python 3.11 and 3.12. numpy 2.2.6 is the last numpy release that installs on Python 3.10, and its type stubs give `np.ndarray` no default type arguments, so the CI type check (`disallow_any_generics`) failed on it. The numpy used in CI (`tests/requirements.txt`) moves from 2.2.6 to 2.4.6, the newest release that supports Python 3.11.
- **`_get_num_tableau_c()`** on both solvers. It recomputed `solver.C`; use `solver.C`.
- **`padaquad.runge_kutta`** (its contents moved as described under Changed), **`_VariableSubclass`**, **`_VARIABLE_SECOND_ORDER`**, **`_Interpolatory3Variable`** (use `VARIABLE_METHODS["interpolatory3_variable"]`) with its `b_delta`, `_b0`, `_ba`, `_b1` and `tableau_b` (now the instance's `weight_function`), **`n_tableau_c`** (use `method.n_nodes`), **`_get_tableau_b`** (use `solver.method.weights(nodes)`), **`_Tableau.clone`** (use `MethodClass.clone`), and the solver attributes **`order`** (never read; use `solver.method.order`) and **`Cm1`** (use `solver.C - 1`).
- **`_sort_evals_into_mesh` and `_sort_evals_into_mesh_full_nodes`**, which nothing called.

### Added

- **Split statistics on `IntegrationResult`.** `n_start_steps` (steps in the starting mesh: the caller's `mesh`, the mesh stored by `same_f=True`, or a fresh one), `n_start_steps_split` (starting steps split at least once, including splits by the re-check of accepted steps), the property `percent_start_steps_split` (`100 * n_start_steps_split / n_start_steps`), and `max_split_depth` (the largest number of times any one step was halved). They are read off the per-step split counter, which is now always kept (it was kept only when a cap was set); results are unchanged. Splits made afterwards to build `mesh_optimal` are not counted. All three are `None` in the per-batch result passed to `loss_fxn`.

- **Type checking is enforced in CI, and padaquad ships its types.** `mypy padaquad/` runs in the `tests` workflow (both Python versions) with every function annotated and every function body checked (`disallow_untyped_defs`, `check_untyped_defs`, `disallow_any_generics`, `no_implicit_optional`; see `[tool.mypy]` in `pyproject.toml`). Before, about 37 functions had no annotations, so mypy skipped their bodies. The package now contains the `py.typed` marker (PEP 561) that `pyproject.toml` already listed but that did not exist, so type checkers in projects that use padaquad see its annotations, and `padaquad/__init__.py` defines `__all__`. `mypy` is pinned to `~=2.1.0` in the `dev` extras so a new mypy release cannot fail CI on its own.

- **`MethodClass.measure_degree()` measures a method's polynomial degree from its tables.** It integrates the Legendre polynomials P_k(2t - 1), k = 0 … 2C, over one subinterval with the method's weights and with the lower-degree weights of its error estimate, and returns both measured degrees and the errors (`DegreeMeasurement`). `UNIFORM_METHODS["gk21"].measure_degree()` gives degree 31 and error-estimate degree 19. Legendre polynomials replace the powers t^k of the old exactness test: a method's error on (2t − 1)^(d+1) is its Legendre error times (k!)² 2^k / (2k)!, which is 2.6e-17 for gk31 and below float64 rounding (for powers of t itself the factor is (k!)² / (2k)!, smaller still), so the power test reported degree 63 for gk31, 37 for cc33 and at least 130 for cc65. `tests/test_exactness.py` now checks every method in both directions (exact through its degree, error at least 1e-8 one degree further; the smallest such error is 4.2e-6, for cc65), checks every `error_exponent`, and checks the stored tables in 60-digit arithmetic (mpmath, installed with torch through sympy).
- **Verification scripts.** `examples/verification/polynomial_degree.py` measures every method's degree and error-estimate degree; `examples/verification/integral_accuracy.py` integrates every function in `integrand_dict` with every method at tolerances 1e-6 and 1e-10 and checks each actual error against (number of subintervals) × tolerance. Both exit with status 1 when a check fails, and the test suite runs them.

- **Documentation site.** A Sphinx site (PyData theme) replaces the MkDocs template, which pointed at pages that did not exist: getting started, a user guide, an API reference generated from NumPy-style docstrings, the example scripts with their output, a Performance page, development notes, these release notes, a glossary and a bibliography (`docs/references.bib`, holding only sources that were opened and checked). The methods table is generated from `UNIFORM_METHODS` and `VARIABLE_METHODS`. The build (`sphinx-build -W`) fails on any warning.
- **A benchmark against torchdiffeq that runs and measures fairly.** `examples/speed_test/torchdiffeq_comparison.py` failed on any machine without CUDA (`device = "cuda"` was hard-coded) and passed torchdiffeq an argument it does not have (`mesh=`). It now compares torchdiffeq `dopri5`, padaquad `dopri5` and padaquad `gk21` on six integrands with known integrals at two tolerances, recording median time over repeated runs, integrand evaluations, absolute error, hardware, versions and git commit, in `docs/performance/results_<device>.json`. On an Apple M2 Max CPU padaquad was 1.6 to 46 times faster at the same tolerance. The README's "often two orders of magnitude faster" is replaced by these measured numbers; GPU runs have not been measured.

- **`MethodClass.references`.** Each method carries the keys of `docs/references.bib` for the sources of its nodes and weights (a tuple of strings, validated); the documentation's methods table cites them.
- **Docstring checks.** Every docstring in `padaquad/` follows the NumPy format and passes numpydoc validation (all checks except See Also, Examples and extended summary; pre-commit hook `numpydoc-validation`) and ruff's pydocstyle rules. `pytest` also runs the docstring examples (`--doctest-modules`), and `tests/test_docs_vocabulary.py` checks the public docstrings and user pages for the documentation's vocabulary rules.

- **`CITATION.cff`** with the software's citation metadata (checked against the Citation File Format 1.2.0 schema; a test keeps its version equal to the package's).
- **Development pages**: how to add a quadrature method (with a worked Simpson's-rule example) and how to publish the documentation.
- **Issue write-ups** in `docs/issues/`: the variable-sampling solver does not reuse evaluations on split, and `integral_error` is wrong after the `take_gradient=False` re-check removes subintervals.

### Changed

- **`converged` reports whether the returned integral meets the tolerance.** It was `True` on every call except a `max_path_change` early exit, including calls where `max_adaptive_splits` left steps failing. Every accepted step is now judged against the final integral (in cumulative mode, the final running integral through the step), the reference `mesh_optimal` is built with (also when `error_integral_reference` is given), and `converged` is `False` if any fails or has a NaN/Inf error ratio. A step accepted at the `max_adaptive_splits` cap usually fails this check, but can pass it. A warning then reports the number and percentage of failing steps, the largest error ratio, and the failing steps' summed error estimate next to the total over all finite steps and `|integral|` (for D > 1, all three for the output component with the largest relative failing error), followed by the remedies that apply to the call's settings. Steps are accepted against the running integral, and with `take_gradient=True` (the default) accepted steps are never re-checked, so `converged=False` also appears without a cap. Over 576 uncapped `take_gradient=True` runs (bosh3, dopri5, gk7, gk21, cc9, cc17; six integrands on [0, 1], three of them cancelling; an rtol-dominated and an atol-dominated tolerance; two seeds; absolute and cumulative mode; the `"2"` and `"failure_fraction"` norms; eight steps per batch), 47 (8.2 %) returned `converged=False`, all in absolute mode, where the tolerance follows the running total; the median worst ratio of the `"2"`-norm failures was about 1e4, from cancelling integrands whose total is near 0. The same 576 runs with `take_gradient=False` all converged. Integrals are unchanged; the result only reports what it did not report before. `error_ratios` still holds each step's ratio at acceptance, so it can show every step passing while `converged` is `False`. `converged` is `None` in the per-batch result.

- **`IntegrationResult` is keyword-only and its fields are required.** Fields typed `torch.Tensor` used to default to `None`. They are now required, and the dataclass accepts keyword arguments only, so a result built without one of them raises `TypeError` instead of carrying a `None` that fails later. Three fields can still be `None`: `mesh_optimal` (in the per-batch result passed to `loss_fxn`, where no optimal mesh exists yet; always a tensor on a returned result), `tracked_variables` (when the integrand returns none), and `loss` (in the per-batch result before `loss_fxn` runs). The per-batch result now carries `y0` as a zero tensor instead of the integer `0`. Code that builds an `IntegrationResult` itself must pass every required field by keyword.
- **A missing integrand raises `ValueError`.** `integrate()` with no `f`, here or at construction, now raises `ValueError` from `_check_variables`; it was an `AssertionError`.
- **`max_batch` is always a Python `int` inside the integration loop.** When the memory budget exceeded the remaining work, `max_batch` became a 0-d tensor holding that count; it is now `min(max_batch, int(remaining))`, the same value. No change in results.
- **One abstract `MethodClass` for every quadrature rule.** The uniform rules (`MethodClass` with a `_Tableau`) and the variable rules (`_VariableSubclass` with `tableau_b(c)`) were two unrelated class families, and each solver carried its own copy of node placement, method setup, the integral computation and the weight lookup. `MethodClass` is now an abstract base declaring `node_fractions()`, `weights(nodes)`, `to_dtype`, `to_device`, `n_nodes` and `clone()`; Two generic classes implement it from data, and every registered rule is an instance of one of them: `UniformMethodClass` from a fixed `_Tableau`, and `VariableMethodClass` from node fractions and a module-level `weight_function` of the normalized node positions (`interpolatory3_variable` is `VariableMethodClass` with `_interpolatory3_weights`). Both reject malformed node fractions (not 1-D, not from 0 to 1, or decreasing) with `ValueError`, and `VariableMethodClass.weights` rejects a weight function that does not return `[N, C]`. Node placement, `_set_solver_dtype` and `_calculate_integral` now exist once, in `AdaptiveQuadrature`. Consequences for code that uses these internals:
  - `MethodClass(order, error_exponent, tableau)` can no longer be constructed; use `UniformMethodClass`.
  - Both registries hold instances that `_get_method` clones per solver, so `VARIABLE_METHODS[name]()` no longer works; `VARIABLE_METHODS["adaptive_heun"]` is the same object as `UNIFORM_METHODS["adaptive_heun"]` (its two nodes are the panel ends, so its weights cannot depend on positions).
  - `method.weights(nodes)` returns `b` and `b_error` shaped `[N, C, 1]` for every rule (the uniform weights were `[C, 1]`, adaptive heun's `[1, C, 1]`); stored tables are `[C]`, including adaptive heun's, which was `[1, 2]`.
  - An unknown method name raises `ValueError` listing the valid names, from `_get_method`; it was an `AssertionError` in the solver or a `KeyError` in `_get_method`.
  - `UniformAdaptiveQuadrature` and `VariableAdaptiveQuadrature` are defined in `padaquad.quadrature`, and `adaptive_quadrature` in `padaquad.integrate`; `from padaquad import ...` is unchanged.

  No numerical result changes: 1,280 integrations (both sampling modes, float32 and float64, gradients, D = 3, several batches, `conserve_memory`, warm starts) give bit-identical tensors.
- **The variable solver detaches its error estimates,** as the uniform solver already did (they only drive accept/split decisions). With `take_gradient=False` and a learnable integrand, its `integral_error` and `mesh_quadrature_errors` no longer require grad; the values are unchanged.
- **The quadrature tables are built without numpy.** Gauss-Kronrod tables are built with torch and Clenshaw-Curtis tables with Python floats and `math.cos` (same summation order), so padaquad no longer imports numpy and every table is bit-identical to before.
- **Both solver types build their method in one place.** The variable solver built its method in `__init__` with its own copy of `_get_method`'s steps; both solvers now build it in `_set_solver_dtype`, which `SolverBase.__init__` calls when it sets the dtype (the uniform solver's `_setup_method`, which only `_set_solver_dtype` and `__init__` called, is merged into it). `AdaptiveQuadrature.__init__` no longer resets `method`, `order`, `C` and `Cm1` to `None` after they were built, and the uniform solver no longer builds its method twice. A dtype change now rebuilds the variable method from its float64 definition rather than converting the existing one. No change in results: the variable methods' coefficients (0, ±1/2, 1) are exact in float32 and float64.
- **Tracked variables are stored as one record entry each** (`tracked_variables_0`, `tracked_variables_1`, …) instead of a list under one key, so every record value is a tensor and the record code has no special case for them. When the set of tracked variables differs between two recorded batches, `_record_results` now raises `ValueError` naming both sets of record keys; before, tracked variables that first appeared in a later batch were dropped without a message, and tracked variables that disappeared raised `TypeError`. An integrand that returns an empty tuple of tracked variables now gets `tracked_variables=None`, the same as an integrand that returns none, in both the per-batch and the returned result; before, the returned result held `()`. No other change in results.
- **`SolverBase.integrate` declares the arguments every solver accepts,** in `AdaptiveQuadrature.integrate`'s order (`f, y0, mesh, mesh_init, mesh_final`, then keyword-only `f_args`). It listed `mesh_init` and `mesh_final` before `mesh`, the opposite of the only implementation. `_merge_excess_nodes` is declared as an abstract method of `AdaptiveQuadrature` with its real signature (`_compute_nodes` is now one shared method; see the `MethodClass` entry above).

- **Methods store their polynomial degree; `order` is `degree + 1`.** `UniformMethodClass(degree, error_exponent, tableau)` and `VariableMethodClass(degree, error_exponent, c, weight_function)` take `degree`, the highest polynomial degree the method integrates exactly, in place of `order`, which is now a read-only property returning `degree + 1` (Gonnet 2012, ACM Comput. Surv. 44(4), Sec. 1 (arXiv:1003.4629v2, p. 3)). Both reject a `degree` that is not an int ≥ 0 and an `error_exponent` that is not an int ≥ 2 with `ValueError`. The degrees: adaptive_heun 1, fehlberg2 1, bosh3 2, dopri5 4; gk7 11, gk15 23, gk21 31, gk31 47; cc5 5, cc9 9, cc17 17, cc33 33, cc65 65; interpolatory3_variable 2 (3 with its middle node at the center). `order` therefore changes from 11/23/47 to 12/24/48 for gk7/gk15/gk31 and from 5…65 to 6…66 for cc5…cc65; the README and code comments that gave those methods a degree one too low are corrected. No algorithm reads `order`; integrals are unchanged.

- **The solver constructor is keyword-only and names every option.** `AdaptiveQuadrature.__init__` (inherited by `UniformAdaptiveQuadrature` and `VariableAdaptiveQuadrature`) now lists every option: `f`, `method`, `atol`, `rtol`, `mesh_init`, `mesh_final`, `y0`, `remove_cut`, `max_batch`, `total_mem_usage`, `max_adaptive_splits`, `use_absolute_error_ratio`, `error_norm`, `mesh_failure_tolerance`, `error_on_nonfinite`, `dtype`, `device`, `output_speed_info`, plus the multi-GPU options of `DistributedEnvironment`. Before, the options were spread over three constructors joined by `*args`/`**kwargs`, and positional arguments filled `remove_cut`, `max_batch`, … in order. Passing any option by position now raises `TypeError`; an unknown keyword raises `TypeError` naming the solver class (it named `DistributedEnvironment.__init__`). `SolverBase.__init__` takes its options as required keywords, so each default is defined once. `remove_cut >= 1` raises `ValueError` (it was an `assert`).
- **`adaptive_quadrature(sampling_type, method, **options)`.** `method` is now the second positional argument, as in `integrate(f, method, ...)`. Before, `adaptive_quadrature("uniform", "dopri5")` put `"dopri5"` into `remove_cut` and failed with `TypeError: '<' not supported between instances of 'str' and 'float'`.
- **`padaquad.integrate(f, method, sampling, **options)` accepts every solver option.** Each keyword goes to the solver's `integrate` call if it is one of that call's arguments, and to the constructor otherwise; the defaults are the solver's (one source). `f_args`, `same_f`, `N_init_steps`, `random_initial_mesh`, `loss_fxn`, `error_integral_reference` and `conserve_memory` used to raise `TypeError` because they were sent to the constructor. All options after `sampling` are keyword-only. `sampling` accepts every name `adaptive_quadrature` accepts. `method=None` (the default of the constructor, the factory and `integrate()`) means the sampling type's default method, from `DEFAULT_METHODS` in `padaquad/base.py`: `"gk21"` for uniform sampling and `"interpolatory3_variable"` for variable sampling. Before, the one default `"gk21"` made `adaptive_quadrature("variable")`, `VariableAdaptiveQuadrature()` and `integrate(f, sampling="variable")` raise `ValueError: Unknown method 'gk21'`.
- **Sampling-type names are case-insensitive** in `get_sampling_type` (and so in `adaptive_quadrature`), as `integrate(sampling=...)` already was.

### Fixed

- **The re-check of accepted steps used the wrong running integral in cumulative mode.** With `use_absolute_error_ratio=False` and `take_gradient=False`, each recorded step is re-judged against `|running integral through it|`; that sum was taken over the recorded steps only, leaving out the batch just evaluated, so steps to the right of a batch step were judged against the wrong value and could be left failing. It now uses the running integral of the record and the batch together. Cumulative-mode results can change; absolute-mode results do not.

- **Known limitation documented: `fehlberg2`'s error estimate ignores curvature.** Its weights [1/512, 255/256, 1/512] and the lower-degree weights [1/256, 255/256, 0] make nearly the same error on a curved integrand (0.25098 and 0.24902 for t² on [0, 1], exact 1/3), so their difference, the error estimate, keeps only (f(end) − f(start)) / 512 times the width. On `damped_sine` at tolerance 1e-6, subintervals with estimates near 1e-7 have true errors near 1e-4, and the total error (1.0e-4) is larger than 84 subintervals × 1e-6. The behaviour is unchanged; `examples/verification/integral_accuracy.py` reports it as `[KNOWN]` and the README warns about it.

- **A failed constructor no longer prints a second error.** `SolverBase.__del__` called `end_process` on a half-built object, so a constructor error was followed by an ignored `AttributeError: ... has no attribute 'is_distributed'`.

- **Two Gauss-Kronrod digits beyond float64 precision.** gk21 `wgk(3)` differed from Netlib QUADPACK in its 33rd digit and the gk7 node 0.4342437… from its 28th significant digit; both now match an 80-digit computation. The float64 tables are unchanged.
- **Descriptions corrected to match the code.** The variable-sampling solver does not reuse integrand values when it splits a subinterval (`_evaluate_adaptive_nodes` is not called by `integrate()`), so `interpolatory3_variable` always has its middle node at the center, where it is Simpson's rule; the README, docstrings and glossary said otherwise. `adaptive_heun`'s error weights subtract the right-endpoint rectangle rule, not Euler's method (same size, opposite sign). `integral_error` is approximate with `take_gradient=False` when the re-check removes a subinterval (the absolute value of its estimate is subtracted from a signed sum). No behavior changed.

## [1.3.0] - 2026-11-10

### Added

- **`adaptive_quadrature` defaults to uniform sampling.** `sampling_type` was a required argument, so `adaptive_quadrature(method="gk21", ...)`, as written in the README's warm-start example, raised `TypeError: missing 1 required positional argument: 'sampling_type'`. It now defaults to `DEFAULT_SAMPLING_TYPE` (`"uniform"`, defined in `padaquad/base.py`), the same constant that now sets the default of the `padaquad.integrate()` wrapper's `sampling` argument, so both defaults change together.

### Changed

- **The memory benchmark now measures the memory cost of one integrand evaluation.** `_setup_memory_checks` measures one number, c (`f_unit_mem_size`), the memory one evaluation of `f` uses; the batch size is still computed at every evaluation from live free memory (usable memory ÷ c). Before, nothing was measured: when a test call took longer than 0.1 s the code assumed 100·N points would fill memory, so c was about 100 times too small and batches about 100 times too large. The 1.2.2 entry below claimed the result "never extrapolates past the largest tile actually validated"; that was wrong. Now, for N = 1, 10, 100, … points, M (the memory one call used) is measured and c = s·(M + E)/N, with s = 2.1 for `take_gradient=True` (a heuristic for backpropagation) and 1 otherwise:
  - **CUDA:** peak allocated memory during the call minus allocated memory before it (`torch.cuda.reset_peak_memory_stats` / `max_memory_allocated` on the solver's device). This counts memory `f` frees before returning, such as the intermediates of a Jacobian computed inside `f`. It resets the device's peak counters, including `max_memory_reserved`.
  - **CPU on Linux:** the process's peak resident memory (`VmHWM` in `/proc/self/status`), reset before each call by writing `5` to `/proc/self/clear_refs`, which also resets the peak your own `getrusage` or `/usr/bin/time` report. `ru_maxrss` is not used because it also includes the peak of the process this one was started from, so in a worker started by a larger process (`subprocess`, multiprocessing `spawn`, a Jupyter kernel) it would read no memory use. The kernel's per-CPU counters can read low by up to E = cores·max(32, 2·cores) pages (`lib/percpu_counter.c`), so E is added (1.5 MiB on 12 cores, 512 MiB on 256 cores).
  - **Other CPUs (macOS, Windows):** resident memory after the call minus before. Memory `f` frees before returning is missed there.

  The search stops when c changes by less than 1 % from N/10 to N, a call takes longer than 1 s, N reaches 1e9, or the next call is predicted not to fit (10·M > usable memory). After an out-of-memory error at N, the c measured at the previous N is kept if that N was at least 100; otherwise c = max(previous c, s·F/(0.75·N)), F the free memory before the call. The one-point check of `f` in `integrate()` now runs before the benchmark, so one-time costs (CUDA start-up, lazy model loading) stay out of it. A single evaluation that does not fit raises a clear `torch.OutOfMemoryError` instead of tripping an internal `assert`; any other failure of the measurement stops the run with a `RuntimeError` saying the memory could not be measured and that `max_batch` must be passed, with the original error attached.
- **`integrate()` now takes its dtype from `mesh_final` when that is the only bound given.** `integrate()` passed `mesh_final=mesh_init` to `set_dtype_by_input`, so when `mesh` and `mesh_init` were both `None` the dtype of `mesh_final` was ignored and the integral ran in the solver's construction dtype. A call that passes only `mesh_final` (including through the `padaquad.integrate()` wrapper) now follows the rule that already applied to `mesh` and `mesh_init`: the solver switches to `mesh_final`'s dtype. This can lower precision: `mesh_final=torch.tensor([2.0])` (float32 by default) on a float64 solver now integrates in float32. An integer `mesh_final` passed on its own now raises `ValueError: Given dtype must be torch.float64 or torch.float32; got torch.int64`, as an integer `mesh_init` already did; pass a floating-point tensor.
- **An unknown `sampling_type` now raises `ValueError` with a message, instead of a bare `KeyError`.** `adaptive_quadrature('bogus')` (and `get_sampling_type('bogus')`) raised a bare `KeyError: 'bogus'`; they now raise `ValueError` naming the bad value and listing the valid names. A sampling type that exists but is not adaptive (`steps.FIXED`, `'fixed'`) or a value of the wrong type also raises `ValueError` naming it and the two supported types; previously that was a `ValueError` with no message. Code that caught `KeyError` here must catch `ValueError`.
- **padaquad now requires `torch>=2.4`.** The out-of-memory handling (`_is_out_of_memory` and the batch-reduction re-raise) uses the top-level `torch.OutOfMemoryError`, which first exists in PyTorch 2.4.0 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.4.0), [#121702](https://github.com/pytorch/pytorch/pull/121702)). `pyproject.toml` listed `torch` with no lower bound, so on an older torch the first out-of-memory error would have become an `AttributeError`. 2.4 is the first version where that name exists, not a tested minimum: the code has been run on torch 2.8, and CI installs the latest release.
- **The solver no longer changes process-wide PyTorch state.** It used to call `torch.cuda.set_device` for CUDA and `torch.set_default_device` for CPU (and `torch.cuda.init`), which affected all other code in the process. Only distributed runs still call `torch.cuda.set_device`, which NCCL needs. `solver.device` is now always one `torch.device`; a CUDA device always carries an explicit index.
- **The memory benchmark's messages.** When it starts, a warning explains that `max_batch` is being measured because adaptive refinement changes the evaluation batch at each step, that the measurement is slow and runs on every `integrate()` call without `max_batch`, and how to skip it (pass `max_batch`, or `same_f=True`). When it ends, a warning reports the time taken, why the search stopped, the bytes per evaluation, and the `max_batch` the memory usable at that moment allows for the current `take_gradient`, to pass as `max_batch` in later runs. Both are logged at WARNING level, so they are silent if your application configures logging above WARNING.
- **The two split evaluation handlers share one batch loop.** `_evaluate_f_on_split_nodes` and `_evaluate_f_on_split_residual_nodes` carried byte-identical copies of the batched `f` evaluation loop with its out-of-memory retry; both now call `_evaluate_f_in_batches`. The memory cleanup after an out-of-memory error or a memory benchmark is one method, `_release_memory`, which empties the CUDA cache only on a CUDA solver (previously it did so whenever any GPU was visible, even for a CPU solver). Dead commented-out code in the residual handler and in `integrate()` was removed. No change in results.
- **Memory management moved to `padaquad/quadrature/memory.py`.** The memory benchmark, the free-memory queries, the batch-size budget (`get_max_batch`, `_get_max_f_evals`) and `_is_out_of_memory` now live in a `_MemoryManagement` mixin that `AdaptiveQuadrature` inherits, so every method is still available on the solver. The benchmark is split into `_setup_memory_checks` (the search and its stop rules), `_peak_memory_reader` (how memory is measured on each device) and `_measure_call` (one test call), with comments explaining each step. No change in results.
- **`same_f` replaces integrand matching by `id(f)`.** The solver used to remember `id(f)` of the last converged call and, on a match, reuse the stored `max_batch` (skipping the memory benchmark); ids are unreliable (a freed id is reused, and closures and bound methods get a new id on every call). The caller now says so with `integrate(..., same_f=True)`: the mesh and `max_batch` stored by the last call that returned replace the caller's `mesh` / `max_batch`, and the benchmark is skipped. With nothing stored (first call) the caller's values or the defaults are used. Both values are written together by every call that returns (converged, or stopped by `max_path_change`, which now stores the mesh it stopped on with that batch's failing steps split, and returns it as `mesh_optimal`, so the next call refines further instead of starting from, and stopping on, the same mesh) and never by a call that raises; `previous_max_batch` therefore holds the `max_batch` of the last returning call, not of the last evaluation. Without `same_f=True` or `max_batch`, the memory benchmark now runs on every call.
- **`max_path_change` measures the whole starting mesh.** It used the fraction of failing steps in each batch, so the two halves of one split step could end a call early, and a rough region in the first batch could end it even when the rest of the mesh passed. It now counts the failing steps of the starting mesh across all batches (split halves are not counted) against the starting step count, and stops as soon as that fraction reaches `max_path_change`. It also runs on the stored mesh used by `same_f=True` (`reuse_mesh=True` never ran it on the cached mesh), never on a freshly generated mesh.
- **The mesh-merging cutoff accounts for how a step's error grows when two steps merge.** `mesh_optimal` merged a pair of steps when their summed error ratio S was below `remove_cut`, without evaluating the merged step. A merged step of width 2h has an error of about 2^(k-1)·S when the method's error estimate scales as h^k, so 7–22 % of those merged steps failed when the same integrand was re-evaluated on `mesh_optimal` (dopri5, gk7, cc17 in our tests). A pair now merges only when S < `remove_cut` / 2^(k-1), so the merged step's predicted ratio stays below `remove_cut`; none failed in the same tests. Each method carries k as `error_exponent` (2 for adaptive Heun and fehlberg2, 3 for bosh3, 5 for dopri5, 7/15/21/31 for gk7–gk31, 5/7/11/19/35 for cc5–cc65, and 3 for `interpolatory3_variable`). `mesh_optimal` is larger in 150 of the 390 snapshot cases (up to 75 %, 8 → 14 barriers for cc17 on `damped_sine` at the tight tolerance); integrals are unchanged, because merging happens only after convergence. For high-order methods at tight tolerances, or in float32, the error estimate is round-off, which does not shrink as h^k, so few pairs reach the cutoff: from a 512-step start on smooth integrands, 8 repeated `same_f=True` calls leave gk31 at 7–30 steps and cc65 at 74–264, where gk21 reaches 2–4.
- **`get_max_batch()` defaults to the solver's `total_mem_usage`** instead of a second, fixed default of 0.9.

### Removed

- **`integrate(reuse_mesh=...)` and the `previous_f_id` attribute.** Use `same_f=True`. Migration: with `reuse_mesh=True` an explicit `mesh` won over the cached one; with `same_f=True` the stored mesh and `max_batch` replace the caller's. `same_f` takes `reuse_mesh`'s position, so a positional `reuse_mesh` argument now sets `same_f`.
- **The `device_type` and `device_ids` solver attributes.** Use `solver.device` (a `torch.device`) instead.

### Fixed

- **Cumulative error mode merged steps that failed when re-evaluated.** With `use_absolute_error_ratio=False`, the summed error ratio of a pair of steps (used only to build `mesh_optimal`) was divided by the tolerance of the larger of the two running integrals, `max(|cum_i|, |cum_{i+1}|)`. The merged step ends where step i+1 ends, so it is judged against `|cum_{i+1}|`; where the running integral falls toward zero (e.g. `cos(2πt)` near `t = 1`), merged steps of `mesh_optimal` had error ratios up to 11 when re-evaluated. The pair now uses step i+1's tolerance in both error-norm families. Absolute mode (the default) and all integrals are unchanged.
- **The out-of-memory batch reduction crashed on the first out-of-memory error.** The retry path in the split evaluation handlers computed `torch.int(torch.round(0.75 * self._oom_max_batch))`: `_oom_max_batch` starts as `None`, so the first out-of-memory error raised `TypeError` instead of shrinking the batch, and `torch.int` is a dtype, not a function. The batch is now seeded from the live `max_batch` and shrunk with `max(1, min(round(0.75 * prev), prev - 1))`, which strictly decreases (`round(0.75 * 2) == 2` would retry forever). This path had no tests; `tests/test_oom_recovery.py` now covers it with a simulated out-of-memory integrand.
- **CPU and CUDA-library out-of-memory errors are recognized.** A refused CPU allocation raises a plain `RuntimeError` naming `DefaultCPUAllocator`, numpy raises `MemoryError`, a failed `cudaMalloc` outside PyTorch's allocator raises "CUDA error: out of memory", and CUDA libraries report their own allocation statuses (`CUBLAS_STATUS_ALLOC_FAILED` and the cuSOLVER, cuSPARSE, cuFFT and cuDNN equivalents); none of these is a `torch.OutOfMemoryError`, so the batch loop and the benchmark crashed instead of shrinking the batch. They are now handled the same way. Any other `RuntimeError` from `f` still propagates unchanged.
- **Callers can choose the device, including the GPU index.** `device="cuda:1"` raised `ValueError` and a `torch.device` raised `AttributeError`, so every solver ran on `cuda:0` (or `cuda:<local_rank>` when distributed). `device` now accepts `"cpu"`, `"cuda"`, `"cuda:N"`, a `torch.device`, or `None` (auto-detect), both on the solver and in the `padaquad.integrate()` wrapper. One normalizer, `_resolve_device`, handles `device` and `result_device` and raises `ValueError` for an unsupported type, unavailable CUDA, a GPU index that is not visible, or (distributed only) an explicit index that differs from the rank's GPU. For `result_device`, an index-less `"cuda"` means the solver's GPU and `None` keeps results on the solver's device.
- **Cross-device crash in the accepted-panel re-check.** When `result_device` differed from the integration device, removing a re-split panel rebuilt the recorded `loss` on the integration device while new batches were recorded on `result_device`, so the next recorded batch raised a cross-device addition error. The loss now stays on its own device.
- **The integrand check call now passes `t` shaped `[1, T]` for any number of time coordinates `T`.** Before integrating, `integrate()` calls `f` once on the lower bound to check its output. That point was built with `torch.tensor([[mesh_init]])`, which only works when `mesh_init` holds one number (`T = 1`) and raises `ValueError: only one element tensors can be converted to Python scalars` for `T > 1`. It is now `mesh_init.reshape(1, -1)`. No user could reach the old failure: `T > 1` is still stopped earlier by the `mesh_init < mesh_final` check in `_setup_integral_bounds`, so `T > 1` integration remains unsupported end to end. For `T = 1` the shape (`[1, 1]`) is unchanged. The check call now receives the caller's own bound tensor instead of a detached copy, so an `f` that modifies `t` in place would change the caller's `mesh_init`; such an `f` already fails inside the solver, which also passes `f` views of its node array.

## [1.2.2] - 2026-06-22

### Changed

- **The per-evaluation memory benchmark (`_setup_memory_checks`) was rewritten to be stable and self-limiting.** It now grows the probe batch by powers of ten until the integrand either takes measurably long (`eval_time > 0.1s`), overshoots available memory, or reaches the `1e9` search ceiling, and derives `f_unit_mem_size` from the usable-memory budget at that point. The result is floored so the implied `max_batch` never extrapolates past the largest tile actually validated — previously the time-based branch could imply a `max_batch` ~100× the tested size and risk an out-of-memory error on the first real batch. On an out-of-memory event during the probe the batch is backed off (`0.75 * N`) and retried instead of aborting, and the benchmark now honors the caller's `total_mem_usage` fraction. It logs a warning when it runs (it is slow if repeated) and reports the resolved `max_batch` on completion.

### Added

- **`get_max_batch(total_mem_usage=0.9, default_max_batch=None)`** on the solver: resolves the per-batch evaluation cap *without* re-benchmarking memory. Precedence is an explicit `default_max_batch` (e.g. a per-call `max_batch`), then the construction-time `init_max_batch`, then the budget implied by a prior memory benchmark (`f_unit_mem_size`), all clamped by any `_oom_max_batch` discovered after an out-of-memory event. This lets calling code capture `max_batch` after a first run and feed it back into later solvers/calls to skip the slow benchmark.

## [1.2.1] - 2026-06-16

### Changed

- **Accepted integration panels are now re-validated against the drifting error tolerance.** In absolute-error mode the accept/reject denominator (`max(atol, rtol * |I|)`) depends on the running integral, which changes as more panels are evaluated — so a panel accepted early could later violate the *current* tolerance yet remain accepted, silently breaking the error guarantee. Each iteration the solver now re-checks every already-accepted (recorded) panel against the same `reference_integral` the current batch uses; any that now fail are removed from the record and bisected in the same mesh update (`_adaptively_increase_mesh` gained an internal `extra_split_idxs` argument), and their refined children are re-evaluated. At convergence **every** recorded panel satisfies the final tolerance. Runs only on the `take_gradient=False` path (where no per-batch backward can be double-counted and the full-integral denominator is available) and is skipped when a fixed `error_integral_reference` pins the denominator. The `take_gradient=True` path is unchanged. **This can change the converged mesh and integral** for multi-batch, drifting-denominator integrands (notably cancelling ones, where the denominator collapses toward `atol` as the integral cancels).
- **Snapshot golden values are keyed per `take_gradient` mode and regenerated.** The two modes legitimately diverge once the re-check engages (they only ever coincided when every panel fit in a single batch); `take_gradient=True` values are unchanged.

### Fixed

- **Default-loss detection in the record bookkeeping** compared bound methods by identity (`loss_fxn is self._integral_loss`), but `obj.m is obj.m` is `False` for bound methods, so the recorded `loss` was never rebuilt after a panel removal. It now compares the underlying function; a custom `loss_fxn` is still left as-is.
- **`_rec_remove` recursion overflow.** The post-convergence prune helper was tail-recursive and hit Python's recursion limit on the large, finely-refined meshes the re-check produces for cancelling integrands at tight `atol`. Converted to an equivalent `while` loop — identical output (snapshots unaffected), no depth limit.

## [1.2.0] - 2026-06-14

### Changed

- **Error tolerance is now computed against the complete integral (notable for memory-heavy integrands).** The `take_gradient=False` evaluation path (`_evaluate_f_on_split_nodes`) now evaluates *every* pending panel in a single pass before any accept/reject decision, so the integral is complete the first time a panel is judged. The absolute-mode tolerance denominator (`max(atol, rtol * |I|)`) therefore uses the full integral from the start, instead of the running (incomplete) integral that accumulated batch by batch. This yields better mesh-split decisions, but **can change integration results for integrals large enough to span multiple batches** (`total_nodes > max_batch`): the early panels are now judged against the true total rather than a too-small partial sum, so the converged mesh and integral can differ from prior versions. Small integrals that fit in one batch are unaffected. `max_batch` continues to bound memory by chunking the `f` calls; it no longer fragments the error decision.

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
