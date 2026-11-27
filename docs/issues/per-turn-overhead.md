# Most of an integral's time is bookkeeping, not the integrand

Status: partly addressed (2026-11-26). Found 2026-11-24 while explaining the
Performance page, where PadaQuad loses to quadax and scipy on integrands that
need few subintervals. The original audit (13 items, line numbers of commit
`453474d`) is in commit `5ab55f5`. Ten plans built on it were implemented,
reviewed and merged into branch `endpoints_call_setup` (PR #75, PadaQuad
2.0.0); the plans were deleted once done and are in commit `3033e24`
(`docs/issues/2026-11-25-speedup-<slug>-plan.md`, each with a "Not done"
section). This file now lists what is left.

## What was done

| Audit item | Outcome in 2.0.0 (plan) |
|---|---|
| 1. Record kept in mesh order through a Python dictionary | Done: each recorded subinterval carries its integer mesh index, remapped on the device; the record is put in mesh order once, after the loop (`record-order`) |
| 2. Record copied to the CPU every turn | Partly: nodes, widths and error ratios stay on the solver's device until the end; the values still move every pass. Copying them without waiting (pinned memory) was reviewed and left out by the user's decision (`compiled-turn`, D2) |
| 3. Cumulative-only work in absolute mode | Done (`error-ratio-work`) |
| 4. Pair error ratios every turn | Done: computed only where the state's mesh merges (`error-ratio-work`) |
| 5. `_adaptively_increase_mesh`: 55 waits | Done: one read per mesh update, integer indexing, no NaN check (`turn-device-reads`) |
| 6. Invariant checks reading the GPU every turn | Done: the per-pass order checks are gone (the mesh is in order by construction); the remaining checks are under `runtime_checks`: one read at the end of a call, besides the finite check of each call of `f` |
| 7. Waits in the evaluation path | Mostly: one `torch.nonzero` per turn gives the loop condition and the waiting subintervals; the finite check still reads once per call of `f` (D2b below) |
| 8. Reuse mesh built on every call | Done: `integrate(build_state_mesh=False)` skips it, and the build is cheaper (`optimal-mesh-cost`); a lazy build was not done |
| 9. Fixed costs per call | Done: solvers share one converted method per device and dtype, `atol`/`rtol` are Python floats, the one-point check call of `f` is gone (`call-setup-cost`) |
| 10. Number of turns | Done another way: opt-in `split_rule="accelerate"` splits in four when the error ratio is at least 2^k (`split-waste`) |
| 11. CPU threads | Done: user guide section "CPU threads" |
| 12. A turn without waits | Not done (below) |
| 13. Run-to-run variation | Done: the cause was Python's garbage collector, set off by the mesh dictionary; a call of `sin(2e5 t)` (13,737 subintervals) ran about 100 collections and now runs none (`timing-variation`, with item 1) |

The Gauss-Kronrod methods also no longer evaluate `f` at two end points of
weight zero, and `dopri5` evaluates 5 points instead of 7
(`endpoint-evaluations`).

Measured by the reviews: a turn of `chirp` (gk21, A100) went from 4.75 ms, 42.5
host waits and 266 launches (the plan's figures, at `0590ccf`) to 2.17 ms, 4.9 host waits and about
165 launches (at `bc5b4a1`; `compiled-turn` plan, Review item 1). On the
planning profile (A100, gk21; not the settings of the Performance page), one
call went from 5.6 to 4.3 ms (`damped_sine`), 13.5-15.1 to 11.0 ms
(`many_outputs`), 67 to 40.6 ms (`chirp`) and 229 to 37-39 ms
(`oscillatory`); on the CPU with 8 threads from 3.2 to 2.8, 6.8 to 5.2, 17.4
to 11.7 and 106 to 21 ms.

## What is left

1. **Re-run the Performance page.** Done 2026-11-26: measured at `f8e78b2`
   (PadaQuad 2.0.0 merged into `performance_benchmark`) on the 80 GB A100 node.
   On the GPU `gk21` is now the fastest of the five libraries on all six
   integrands that need many subintervals; it still loses where few are needed
   and, on the CPU, to the compiled JAX solvers on cheap integrands. The page's
   "Why PadaQuad is slower there" gives the profile: a fixed cost of 4.5 ms (GPU)
   and 3.4 ms (CPU) for a one-pass call, and 2.6-3.0 ms (GPU) or about 1 ms
   (CPU) per further pass.
2. **Benchmark script changes.** Done 2026-11-26 in
   `examples/speed_test/library_comparison.py` and `run_benchmark.sh`
   (`569484c` and `f8e78b2`): `gc.collect()` before each case, every run's time
   (`times_s`), the warm-up calls the same closure as the timed runs,
   `--torch-threads` (default 8) for PadaQuad and torchdiffeq, the OpenMP
   variables and the CPU count in the metadata, the range beside the median in
   PadaQuad's table, and each device run as an `srun --cpu-bind=cores` step. Not
   done: `GOMP_SPINCOUNT` is recorded only when set, not set by the script.
3. **A turn without waits (item 12)**, not attempted. The `compiled-turn`
   review found that capture needs fixed shapes and no host reads, while a
   turn's sizes come from its one read and change every turn, so only the
   bookkeeping between reads could be captured, on capacity buffers
   re-captured as the mesh grows (several hundred lines). `torch.compile`
   changed the bits of every sum and made a weighted sum depend on the batch
   size on the CPU, which breaks identical results for any batch size; only
   the elementwise chains (about 35 of a turn's 146 operations) could be
   compiled. The same Python and operations run on the CPU in 0.9 ms per turn
   against 2.2 ms on the GPU.
4. **Running the bookkeeping on the CPU while `f` runs on the GPU** (route (d)
   of the `compiled-turn` plan): measured with a proxy only, a turn of 0.8-1.0
   ms instead of 2.0-2.2 ms on meshes up to about 1,400 points, break-even
   between about 5,000 and 8,000. It would add a device choice by mesh size;
   bit identity end to end is not shown.
5. **Smaller items**, each measured and left out:
   - one read per call of `f` for the finite check (D2b, about 15 µs per call);
   - a sorted list of waiting indices (`turn-device-reads`, 14 waits per call
     on `chirp`);
   - five builds of PyTorch's memory statistics per `take_gradient=True` batch
     on a GPU (about 0.3 ms per batch; they read different moments);
   - judging the batch and the record together (`compiled-turn` 5.2 item 3,
     an estimated 4-6 % of a turn);
   - the GPU count asked at every solver construction, also for a CPU solver
     (60 µs against 15 µs per construction; recorded in
     `docs/issues/distributed-process-group-per-solver.md`, on
     `endpoints_call_setup`).
6. **The arithmetic of a turn in fewer passes** (`quadrature-arithmetic`
   plan): written but not executed, by the user's choice; the plan is an
   untracked file in the user's checkout.

## How to measure

Wrap the solver's methods with `torch.profiler.record_function` to attribute
launches, and use `torch.cuda.set_sync_debug_mode("warn")` with a
`warnings.showwarning` hook that records the innermost `padaquad/` stack frame
to find each wait. Time on a compute node, never a login node.
