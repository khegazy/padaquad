# Plan: bring the `performance_benchmark` work onto PadaQuad 2.0.0

Status: plan, not started (2026-11-26). Supersedes the execution of the earlier
plan on branch `performance_benchmark` (pushed up to `5ab55f5`, plus `fd6c1dd`
local), which was built on `d13b2af` (#72). Since then `main` has gained #73
(memory options), #74 (host memory) and #75 (the speed-up work, PadaQuad
2.0.0, released 2026-11-25).

## What the branch holds (14 commits, 34 files)

- **Homepage:** no install block; "What PadaQuad is for" and the comparison
  table (with diffrax and quadax) under the buttons; the separate page removed.
- **Benchmark:** `examples/speed_test/library_comparison.py` (renamed from
  `torchdiffeq_comparison.py`) against torchdiffeq, diffrax, scipy and quadax:
  - 9 integrands, 6 of which need many subintervals;
  - one tolerance for all;
  - PadaQuad's `max_batch` from a table measured on the 80 GB A100
    (`MAX_BATCH`);
  - a time limit per case.

  `run_benchmark.sh` builds the environment and runs it; the `bench` extra
  lists the compared libraries.
- **Performance page:** results on the 80 GB A100 and the EPYC 7763, "One
  tolerance for all", and "Why PadaQuad is slower where it is".
- **Integrands:** every test integrand takes `xp` (torch, NumPy, JAX);
  `hard_integrand_dict` adds `oscillatory`, `chirp`, `sharp_peak`, `spectrum`,
  `kink`, `many_outputs`.
- **Gallery:** six "Harder integrals" scripts, `neural_network_integral.py` and
  `path_optimization_muller_brown.py`, with figures.
- **Issue** `docs/issues/per-turn-overhead.md` (rewritten in `fd6c1dd`, not
  pushed, as "what 2.0.0 did, what is left").

## What changed on `main` that matters here

1. **Speed.** A turn of the loop went from about 4.75 to 2.2 ms on the GPU
   (the issue's "What was done"). Every number on the branch's Performance page
   is now stale, and so is its section "Why PadaQuad is slower", whose causes
   1–4 and the run-to-run variation (the garbage collector) are fixed.
2. **New options** a benchmark has to take a position on:
   - `split_rule="accelerate"` (opt-in four-way splits, fewer turns);
   - `build_state_mesh=False` (skips building the mesh for a later call);
   - `runtime_checks`;
   - `trade_speed_for_memory`.
3. **Fewer points per subinterval.** Gauss-Kronrod methods no longer evaluate
   the two end points, and `dopri5` takes 5 points instead of 7, so the
   evaluation counts change.
4. **Spelling.** Prose says "PadaQuad" (pages, docstrings, README); code names
   stay `padaquad`.
5. **Benchmark-script changes made on `torchdiffeq_comparison.py`**, which this
   branch renamed:
   - `gc.collect()` before each case;
   - every run's time in the results (`times_s`);
   - the CPU affinity and `OMP_NUM_THREADS` in the metadata.
6. **Shared pieces added on both sides.** `main` also added
   `SELF_CHECKING_EXAMPLES`, `MPLBACKEND=Agg`, matplotlib in `dev` and `docs`,
   and the gallery's matplotlib scraper, for its new `gradient_as_output.py`.
   It also added a memory-options table to `docs/_ext/performance_table.py` and
   a user-guide section "CPU threads" (`ug-cpu-threads`).
7. **New issue** `accuracy-accidental-acceptance.md` (subintervals accepted
   because the error estimate is small by chance). It covers what I saw on
   many sqrt kinks, but not that case.

A trial merge of `main` into the branch (in a separate worktree) conflicts in
12 files, all docs, configuration, the benchmark script and tests:

- `CHANGELOG.md`, `CLAUDE.md`, `README.md`, `pyproject.toml`;
- `docs/conf.py`, `docs/_ext/performance_table.py`, `docs/index.md`,
  `docs/getting_started/what_it_is_for.md`, `docs/performance.md`;
- `examples/speed_test/torchdiffeq_comparison.py`;
- `tests/test_docs_performance_table.py`, `tests/test_examples_run.py`.

No library code conflicts, and the API the branch uses (`integrate`,
`get_max_batch(f=...)`, `state=`) is unchanged.

## Decisions (answered 2026-11-26)

D1: (a), merge `main` into `performance_benchmark`, then pause for review.
D2: (b), the extra `gk21` row. D3: (b), a fixed thread count, default 8.
D4: keep the two Fourier-feature networks separate (the recommendation).
Step 1 (the merge) done as `569484c`, local; under adversarial review.

## Decisions as they were put

- **D1. Which branch.**
  - (a) Merge `main` into `performance_benchmark`, push, and open a pull
    request from it. It keeps the 14 commits and the pushed branch.
    *Recommended.*
  - (b) Rebuild the work on `website_update` (currently identical to `main`).
    Choose this if that branch is meant for this work.
- **D2. PadaQuad's settings in the benchmark.**
  - (a) Defaults only, for `dopri5` and `gk21`, as now.
  - (b) Defaults, plus one row `gk21` with `split_rule="accelerate"` and
    `build_state_mesh=False`. The other libraries build no mesh for a later
    call, so skipping it is the like-for-like comparison, and four-way splits
    help exactly the integrands that need many subintervals. *Recommended.*
- **D3. CPU threads for the torch-based solvers** (PadaQuad, torchdiffeq).
  - (a) All cores, as now.
  - (b) One fixed count from a command-line option, default 8, recorded in the
    results. The speed-up review found 8 within 8 % of the fastest for PadaQuad
    in every case measured, and the user guide now says so (`ug-cpu-threads`).
    *Recommended.*
- **D4. Two Fourier-feature networks.** `main`'s `trade_speed_for_memory.py`
  has a `FourierMLP` (torch only, float32, linearly spaced frequencies); the
  benchmark's `feature_network` needs torch, NumPy and JAX forms from one
  definition. *Recommended:* keep them separate. They serve different
  measurements, and sharing would force the memory script onto the
  three-module builder. I flag it because of the one-source-of-truth rule.

## Steps

1. **Merge** (per D1) and resolve the 12 conflicts:
   - **Prose:** take `main`'s text and re-apply the branch's changes, spelled
     "PadaQuad".
   - **Duplicated pieces** (point 6 above): keep one copy of each.
     `SELF_CHECKING_EXAMPLES` lists `gradient_as_output.py`, the branch's two
     tutorials and `examples/hard_integrals/*.py`; `filename_pattern` lists all
     of them.
   - **`docs/_ext/performance_table.py`:** keep `main`'s version (with the
     memory table) and point `BENCHMARK` at `library_comparison.py`.
   - **`torchdiffeq_comparison.py`:** delete it (the rename), after step 2
     ports its changes.
   - **`CHANGELOG.md`:** the branch's entries go under `[Unreleased]` above
     2.0.0.
   - **`CLAUDE.md`:** `main`'s version, plus the branch's lines: the benchmark
     bullet, `hard_integrand_dict` in the package layout, the gallery line, and
     the test rows of `test_hard_integrands.py`, `test_examples_run.py` and
     `test_docs_performance_table.py`.
2. **Benchmark script** (`library_comparison.py`; item 2 of the issue's "What
   is left"):
   - `gc.collect()` before each case's warm-up;
   - `times_s` in every row;
   - the warm-up with the timed call's options;
   - the thread count per D3 (`--torch-threads`), with the count, CPU set,
     `OMP_*` and `GOMP_SPINCOUNT` in the metadata;
   - the range beside the median in PadaQuad's table;
   - the extra row per D2;
   - in `run_benchmark.sh`, `srun --cpu-bind=cores` when `SLURM_JOB_ID` is
     set.

   The old results are all re-measured in step 5, so rendering a file without
   `times_s` is not needed.
3. **Prose that describes the old internals:**
   - the examples' and pages' quoted evaluation counts, which change with the
     2.0.0 point counts (gk21's old 23-point count is never written in them; a
     search found none);
   - the Performance page's "Why PadaQuad is slower" (rewritten in step 5);
   - the gallery texts (re-read for counts).
4. **Checks on a compute node:**
   - ruff, mypy and numpydoc;
   - the full suite (one GPU each, as before);
   - the gallery scripts;
   - `sphinx-build -W`;
   - re-read the figures, since the subinterval counts change the width plots.
5. **Measure** on an 80 GB A100 node (`-C "gpu&hbm80g"`), from a clean commit:
   - re-measure `MAX_BATCH` (`--measure-max-batch`; `get_max_batch` changed
     with `evaluation_buffer`);
   - re-check `QUADAX_MAX_NINTER` in a dry run;
   - run `run_benchmark.sh`;
   - rewrite "What the results show" by hand, checking every number against
     the files;
   - re-profile the integrands where PadaQuad still loses, if any, with the
     method in the issue's "How to measure".
6. **Issue file:**
   - carry `fd6c1dd`'s `per-turn-overhead.md` over and mark items 1–2 of "What
     is left" done;
   - add the many-kinks case to `accuracy-accidental-acceptance.md` if it still
     happens on 2.0.0 (one run, 100 kinks, `gk21`, ten seeds).
7. **Push and open a pull request to `main`**, with the results files committed
   from a clean tree.

Cost: one 1-hour allocation for the checks and the measurements (the full
benchmark takes about 20 minutes on both devices).

## Not in this plan

Items 3–6 of the issue's "What is left": a turn without host waits, the
bookkeeping on the CPU during `f`, the smaller items, and the
`quadrature-arithmetic` plan. They change the library, not the benchmark.
