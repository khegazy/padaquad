# Plan: run-to-run variation and CPU threads (`timing-variation`)

Status: written 2026-11-25 against `0590ccf` as a diagnosis from counts only;
reviewed against `3eb4507` (the tip of `speedups_integration`, 2026-11-26) and
corrected (section "Review"); the measurement protocol run on a compute node
(section 2.6); implemented on branch `timing_variation` (section
"Implemented"). Items 11 and 13 of `per-turn-overhead.md` (branch
`performance_benchmark`). Line numbers in sections 2.1-2.5 are of `0590ccf`
unless the review gives new ones. Code quality of the scripts in
`$SCRATCH/padaquad_speedup_plans/timing-variation/impl/`: prototype.

## Review (2026-11-26, against `3eb4507`)

107 commits landed after the plan was written (`git log 0590ccf..3eb4507`),
among them all the other speed-up plans. Checked claim by claim against the
code at `3eb4507`, by counts on the login node (CPU, Python 3.12.13, torch
2.8.0) and, for every time, on a compute node (section 2.6). What was wrong or
is now different, and what changed in this file:

1. **The main diagnosis is confirmed and its cause is gone.** `record-order`
   (`aa8ac48`) deleted `_get_mesh_indices` and `_mesh_order`; its implementer
   counted about 100 young, 9 middle and 0.8 full collections per call of
   `sin(2e5 t)` before and none after, and the two-speed time of 89 to 145 ms
   became 36 to 40 ms on an A100 (`2026-11-25-speedup-record-order-plan.md`,
   "Implemented"). Re-counted here at `3eb4507` (`impl/count_now.py`,
   `impl/logs/count_now_login.log`; CPU, 8 threads, gk21, `atol=1e-8`,
   `rtol=0`, `error_norm="max"`, `max_batch=1e6`, 3 calls each after a
   warm-up): for `damped_sine`, `many_outputs`, `chirp` and `sin(2e5 t)`, with
   `take_gradient` False and True, **no collection of generation 1 or 2 in any
   call, and one of generation 0 in 1 of the 24 calls**. The Python memory a
   call allocates peaks at 18 to 30 KiB above its start for 11 to 13,036
   subintervals (`tracemalloc`), and `chirp` (76 subintervals, 11 passes) and
   `sin(2e5 t)` (13,036 subintervals, 13 passes) make the same number of
   Python calls within 3 % (1,361 and 1,391 Python calls; `sys.setprofile`):
   nothing in the loop is done per subinterval in Python. Block G of the
   protocol (the `record-order` prototype) is dropped: that code is now the
   code measured.
2. **What still makes young collections.** Each call leaves 12 objects in
   reference cycles, so a young collection runs about every 58 calls: they are
   a `ctypes.CDLL` of `libnvidia-ml.so.1` and its function pointers, made by
   `torch.cuda.device_count()` (`torch/cuda/__init__.py` 826, NVML query),
   which `DistributedEnvironment.set_device` (`padaquad/distributed.py` 343)
   calls at every solver construction, also for a CPU solver
   (`impl/cycles.py`, `impl/cdll_origin.py`). Not per subinterval and not
   timed here; reported, not changed (it is in the constructor, which another
   plan owns, and a fix needs a new branch).
3. **Line numbers of 2.1 have moved.** `_release_memory` is
   `padaquad/quadrature/memory.py` 719-741; its callers are 224 (restart
   after a backward out-of-memory error), 812 (`_call_f`, after an
   out-of-memory error of `f`), 1230 (end of the memory benchmark of
   `get_max_batch(f=...)`) and 1518 (`_measure_call`, after an out-of-memory
   error in a benchmark call). The merge prediction's `torch.bmm` is
   `padaquad/quadrature/base.py` 3252, and the state's mesh is built where the
   record's nodes are (`optimal_device`, 1696).
4. **"The library sets no process-wide state" needs one qualification.**
   True for threads and the collector (grep of `padaquad/` at `3eb4507`: no
   `set_num_threads`, `set_num_interop_threads`, `gc.disable`, `gc.freeze`,
   `gc.set_threshold`, no `OMP_*` variable). In a distributed job
   (`torch.distributed`, several processes) `padaquad/distributed.py` sets
   `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `LOCAL_RANK` and `RANK`
   (256-262, 310-311) and the current CUDA device (347), as such a job needs.
   The user guide says exactly that.
5. **Torch's default thread count (2.5) was wrong for small CPU sets.** The
   plan said the default is "the number of physical cores of the machine, not
   of the mask" (from the login node: 128 threads with 244 CPUs in the mask).
   On the GPU node (`impl/logs/default_threads.log`, a count): 64 threads for
   the whole node (64 cores, 128 logical CPUs); 8 for a process allowed 8
   logical CPUs, by `taskset -c 0-7`, `srun -c 8` or `srun -c 8
   --cpu-bind=cores`; 16 under `srun -c 16`; and **8 under `taskset -c
   0-3,64-67`, the two hardware threads of 4 cores**, so the default follows
   the CPU set and can put two threads on one core. `srun -c 8
   --cpu-bind=cores` gives exactly that set (CPUs 0-3 and 64-67), and `srun
   -c 16 --cpu-bind=cores` 8 cores (0-7 and 64-71) (`impl/logs/srun_cpuset.log`,
   second node `nid200273`). (That the default is the
   smaller of the CPU set and the number of physical cores is my inference
   from these five counts and the login node's, not from torch's source.)
6. **Known doubts, checked.** `KMP_BLOCKTIME` does not apply: the plan's 2.5
   already said so (torch here maps GNU's `libgomp.so.1`), and no script sets
   it. "128 threads" on a GPU node are 64 cores with two hardware threads each
   (`lscpu` on `nid001584`: 1 socket, 64 cores per socket, 2 threads per core,
   NUMA nodes 0-15/64-79, 16-31/80-95, 32-47/96-111, 48-63/112-127; GPU 0 is
   on NUMA node 3): the protocol labels that configuration "two threads per
   core".
7. **The test of 6.1 would not catch what it is meant to catch.** It counted
   collections of generation 0. The young counter is allocations minus
   deallocations of tracked objects, so a loop that creates and drops one
   object per subinterval never triggers a collection; only objects kept
   alive do. It also depends on the collector's version (3.14 changes the
   generations; plan 2.3). The test now compares a call on n subintervals with
   one on 256 n, on two counts that do not depend on the collector: the Python
   lines executed (`sys.settrace` line events, which count every turn of a
   loop) and the peak of Python-allocated memory (`tracemalloc`). It is in
   `tests/test_base_solver.py` beside the host-wait counts of a call
   (`TestHostWaits`), not in `test_record_and_sort.py` (the dictionary is gone
   and the test is about the whole call). See 6.1.
8. **The protocol scripts were stale.** `variation_protocol.py` wrapped the two
   deleted functions; every block of `run_protocol.sh` after A froze the
   collector (`--gc freeze`), which users do not do and which no longer
   matters; block G pointed at the prototype. Corrected copies in `impl/`:
   phases of the current loop, the collector left as users run it except in
   block A, torch's default thread count recorded before any setting,
   `scaling_cur_freq` as fallback for the frequency, 30 runs per series
   (50 did not fit with the suites in one allocation), two new blocks: I
   (the same configuration in 5 separate processes, for the spread between
   processes) and J (each glibc setting of block H alone). Section 7.
9. **D3.3 is changed by the measurements.** "One thread per physical core of
   the allocation" (64 on a GPU node) was the slowest bound setting on two
   integrands of four; 4 to 8 bound threads were fastest for PadaQuad (2.6).
   D3.3 now recommends a fixed, bound, stated count.
10. **Step 4's location kept.** The section is a subsection "CPU threads" of
    "Device and precision" in `docs/user_guide/devices_memory.md` (where the
    device and `result_device` are described), with a pointer from
    `troubleshooting.md` ("A call is slower than expected, or its time
    varies"), where a user who sees a slow or variable call looks first.
11. **D3 on this branch is smaller than planned.** `torchdiffeq_comparison.py`
    already renders the range beside the median; it now also keeps every
    run's time (`times_s`), collects garbage before each case and records
    `cpu_affinity` and `omp_num_threads`. `trade_speed_for_memory.py` (a
    memory benchmark reporting the best of 2 runs) keeps every run's time
    (`all seconds`), collects garbage before each example and records
    `torch threads`. Both only add keys; the docs extension and its tests read
    none of them, so the committed results files render unchanged. The
    changes to `library_comparison.py` and `run_benchmark.sh` belong to the
    branch `performance_benchmark` and are left as recommendations ("Not
    done").

## 1. Summary

- **What varied (before `record-order`).** `sin(2e5 t)` was two-speed on
  both devices (GPU median 216.9 ms, minimum 120.8 ms; the baseline of 2.2),
  because Python's garbage collector ran about 100 times per call, once in
  most calls over every object of the process, set off by the mesh
  dictionary's objects. `record-order` deleted the dictionary.
- **What varies now (measured, 2.6).** Within one process and 30 calls, the
  longest call is 1.01 to 1.37 times the median for every integrand on the
  GPU solver and for every bound thread count on the CPU solver, and the
  medians of five processes with the same setting agree within 0.9 to 7.5 %.
  What remains: (a) PyTorch's default of 64 unbound threads, on the CPU, gives
  `sin(2e5 t)` a tail (longest call 2.5 times the median) and costs up to
  33 % of the median against 4 to 8 bound threads; 128 threads (two per
  core) cost up to 60 % and give tails of up to 3.9 times; (b) the first call
  that keeps the record on the GPU pays about 190 ms of first-use cost, which
  one warm-up with the default `result_device="cpu"` does not remove; (c) on
  the CPU, a process runs `sin(2e5 t)` either with about 6 minor page faults
  per call or with 1,855 to 2,979, and its median is then 2 to 3 ms longer
  (22.4 to 23.2 ms against 19.8 to 20.4 with 8 bound threads); the collector
  does not decide which (adversarial review, 2.6).
- **What does not contribute any more:** the collector (no full collection
  in 6,320 timed calls, at most 1.45 ms of collection in any call), the CUDA
  caching allocator (a device allocation in 3 timed calls, each the first with
  the record on the GPU), NUMA placement (GPU 0's domain, the
  farthest or none, within 2.3 %), and the random starting mesh (every call of a series
  evaluated the same points).
- **What the plan changes.** No library code. A test that the Python work of
  a call does not grow with its subintervals (D4); a user-guide section on
  CPU threads from these measurements (D2) with the statement that PadaQuad
  changes no thread or collector setting (D1); the benchmark scripts of this
  branch keep every run's time and the thread settings (D3).
- **Numbers.** No returned number changes. Every integrand returned one
  integral, bit for bit, across every thread count, binding, collector and
  allocator setting of the protocol.

## 2. Evidence

### 2.1 The code at `0590ccf` (corrected in the review, items 1-4)

- **The dictionary** (deleted by `record-order`). `_get_mesh_indices`
  returned `{tuple(row): idx for idx, row in enumerate(mesh.tolist())}`, one
  list and one tuple per mesh point, after every mesh update; `_mesh_order`
  made one list and one tuple per looked-up subinterval.
- **No thread or collector settings.** See review item 4. The only use of `gc`
  is `gc.collect()` in `_release_memory` (review item 3), with
  `torch.cuda.empty_cache()` when asked; it frees objects in reference cycles
  that may hold the failed call's tensors, and runs only on the four rare
  paths of review item 3, never in the calls measured here.
- **The state's mesh runs where the record is.** With the default
  `result_device="cpu"` the merge prediction's `torch.bmm` runs on the CPU, so
  a GPU process first uses cuBLAS on the GPU when a call keeps the record
  there (`result_device="cuda"` or `None`).

### 2.2 The baseline (measured by the coordinating session at `0590ccf`)

`/pscratch/sd/k/khegazy/padaquad_speedup_profile/` (`profile_baseline.py`,
`baseline_cuda.txt`, `baseline_cpu.txt`; 5 runs per series, reseeded before
each):

| `sin(2e5 t)` series | GPU median / min / max (ms) | CPU median / min / max (ms) |
|---|---|---|
| plain | 216.9 / 120.8 / 232.8 | 115.0 / 80.4 / 121.8 |
| `runtime_checks=False` | 204.4 / 103.7 / 216.4 | 99.8 / 65.5 / 110.6 |
| `result_device` = solver's | 208.8 / 104.3 / 239.6 | 111.3 / 80.7 / 112.3 |
| `record_evaluations=False` | 207.6 / 104.3 / 208.5 | 106.6 / 74.8 / 107.5 |
| `enforce_convergence=False` | 230.2 / 116.0 / 233.7 | 109.9 / 75.1 / 110.6 |

Other series: `damped_sine` with `result_device` the solver's, GPU: median
8.0 ms, one run of 271.8 ms; `many_outputs`, same variant: median 14.6 ms,
maximum 22.2 ms. The issue document (branch `performance_benchmark`, earlier
code): the slowest of 7 runs often 2 to 6 times the fastest; medians moving
30-80 % between benchmark runs; on the CPU device 8 threads faster than 128.

### 2.3 What the collector does (Python 3.12)

From the Python 3.12 documentation of `gc` (https://docs.python.org/3.12/library/gc.html)
and the CPython source at v3.12.13
(https://github.com/python/cpython/blob/v3.12.13/Modules/gcmodule.c), as read
by the plan's author: a young collection starts when allocations minus
deallocations of tracked (container) objects exceed *threshold0* (700); a full
collection runs only when `long_lived_pending >= long_lived_total / 4`, and
costs time proportional to the objects the process tracks; `gc.freeze()` moves
every tracked object out of future collections. Python 3.14 changed the
oldest generation (incremental collection; restored in 3.14.5, per the
current `gc` documentation). Consequence used by review item 7: an object
created and freed before the next is created leaves the counter where it was,
so a loop of such objects never triggers a collection.

### 2.4 Counts at `0590ccf` (login node, CPU, 8 threads)

`gc_counts.py` and its logs (`counts_head*.log`, `counts_record_order.log`):
`sin(2e5 t)` 88-99 / 8-9 / 0-1 collections per call (generations 0/1/2), a
full collection in 21 of 28 calls, each examining 158,772-172,412 objects;
started inside `_get_mesh_indices` (646) and `_mesh_order` (176); none on the
`record-order` prototype. Other integrands: none.

### 2.5 Environment facts

- **Perlmutter GPU node** (`nid001584`, features `gpu,a100,hbm40g`): one AMD
  EPYC 7763 with 64 cores, two hardware threads per core (logical CPUs n and
  n + 64), 4 NUMA domains of 16 cores; four NVIDIA A100 with 40 GB; GPU 0 on
  NUMA domain 3 (cores 48-63). `scaling_governor` `performance`,
  `cpuinfo_max_freq` 3.53 GHz.
- **Torch's default thread count:** review item 5.
- **How torch parallelizes an operation** (headers of torch 2.8.0, as read by
  the plan's author): `at::parallel_for` runs serially below a grain size
  (32,768 elements for element-wise operations) or with one thread; above it,
  `#pragma omp parallel` wakes every thread of the pool.
- **The OpenMP runtime** is GNU's (`libgomp.so.1`): `OMP_WAIT_POLICY` and
  `GOMP_SPINCOUNT` apply, `KMP_BLOCKTIME` does not; `OMP_PROC_BIND` defaults to
  true only when `OMP_PLACES` or `GOMP_CPU_AFFINITY` is set (GCC libgomp
  manual, https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fPROC_005fBIND.html).
- **CUDA lazy loading:** torch sets `CUDA_MODULE_LOADING=LAZY` when unset
  (`torch/cuda/__init__.py`, torch 2.8.0), so a kernel's code is loaded at its
  first launch.
- **glibc:** freed blocks above `M_MMAP_THRESHOLD` go back to the system, and
  free memory at the top of the heap above `M_TRIM_THRESHOLD` is trimmed
  (mallopt(3), https://man7.org/linux/man-pages/man3/mallopt.3.html); memory
  given back costs a page fault per page when used again.

### 2.6 Measurements at `3eb4507` (compute node)

Node `nid001584` (review item 6), job 62038201, alone on the node for every
timed run (the base commit's test suites ran after the protocol, not during
it). torch 2.8.0+cu128, Python 3.12.13, PadaQuad of this worktree at
`3eb4507`. Every call: gk21, `atol=1e-8`, `rtol=0`, `error_norm="max"`,
float64, `take_gradient=False`, `max_batch=1_000_000`, on [0, 1], after
`torch.manual_seed(0)`; per process and integrand one untimed warm-up call,
then 30 timed calls (5 in block B). GPU runs on GPU 0. Raw data:
`impl/results/protocol1/` (blocks A-H, F) and `impl/results/protocol2/`
(blocks I, J): `runs.jsonl` (one line per call), `summary.txt`,
`analysis.txt` (`summarize_variation.py`), `tables.md` (`tables.py`: every
series), `node_<block>.txt`.

The integrands: `damped_sine` exp(-t) sin(40 t), 11 subintervals, 1 pass,
231 points; `many_outputs` sin(k t) for 1,000 values of k, 13 subintervals, 2
passes, 315 points; `chirp` sin(1/(t + 0.001)), 76 subintervals (77 on the
GPU), 11 passes (14), 2,961 points (3,003); `sin(2e5 t)` ("oscillatory"),
13,036 subintervals (13,737), 13 passes, 547,281 points (576,723). The CPU
and GPU meshes differ because the random starting mesh is drawn on the
solver's device.

Times below are min / median / max in ms over the series (n = 30 unless
stated). "Bound n": `OMP_NUM_THREADS=n OMP_PROC_BIND=close OMP_PLACES=cores
taskset -c 0-(n-1)` (cores 0 to n - 1, one thread per core); "near n": the
same on the first n cores of GPU 0's NUMA domain (48 onward); "default": no
variable, no binding (64 threads, 128 logical CPUs allowed); "128":
`OMP_NUM_THREADS=128`, no binding (two threads per core).

**Block C, CPU solver, threads:**

| Threads | damped_sine | many_outputs | chirp | sin(2e5 t) |
|---|---|---|---|---|
| bound 1 | 2.68 / 2.70 / 3.01 | 6.63 / 6.67 / 6.89 | 11.44 / 11.51 / 12.47 | 27.04 / 31.68 / 37.39 |
| bound 2 | 2.64 / 2.68 / 3.17 | 5.51 / 5.64 / 5.88 | 11.24 / 11.29 / 12.29 | 22.78 / 26.58 / 30.87 |
| bound 4 | 2.63 / 2.68 / 2.96 | 4.94 / 5.13 / 5.40 | 11.28 / 11.46 / 12.42 | 20.45 / 23.38 / 27.60 |
| bound 8 | 2.72 / 2.85 / 3.12 | 4.98 / 5.23 / 5.43 | 11.60 / 12.13 / 12.81 | 20.63 / 22.97 / 26.87 |
| bound 16 | 2.94 / 2.99 / 3.19 | 5.68 / 5.72 / 5.82 | 12.54 / 12.57 / 13.63 | 21.88 / 24.22 / 27.41 |
| bound 32 | 2.97 / 3.00 / 3.25 | 5.96 / 6.00 / 6.30 | 12.76 / 12.80 / 13.89 | 23.45 / 24.91 / 26.28 |
| bound 64 | 2.99 / 3.03 / 3.26 | 6.38 / 6.44 / 6.69 | 12.67 / 12.74 / 13.73 | 24.05 / 26.34 / 33.96 |
| default (64) | 2.66 / 2.70 / 3.02 | 6.03 / 6.07 / 6.23 | 12.81 / 12.84 / 13.95 | 27.89 / 30.64 / 76.84 |
| default, `taskset -c 0-7` (8) | 2.63 / 2.66 / 2.99 | 4.92 / 5.07 / 5.44 | 11.50 / 11.98 / 14.37 | 20.39 / 22.92 / 24.45 |
| 128 | 3.63 / 3.67 / 14.26 | 7.57 / 7.77 / 7.81 | 15.89 / 15.93 / 40.76 | 32.94 / 36.76 / 47.32 |
| bound 8, `OMP_WAIT_POLICY=passive` | 2.67 / 2.79 / 3.07 | 5.16 / 5.54 / 5.96 | 11.45 / 11.83 / 12.83 | 23.54 / 24.60 / 27.14 |
| bound 64, `OMP_WAIT_POLICY=passive` | 3.14 / 3.17 / 3.91 | 8.22 / 8.39 / 8.61 | 13.19 / 13.33 / 19.49 | 35.11 / 36.84 / 38.60 |

**Block D, GPU solver, threads** (the record and the state's mesh on the CPU,
`result_device="cpu"`):

| Threads | damped_sine | many_outputs | chirp | sin(2e5 t) |
|---|---|---|---|---|
| near 1 | 4.28 / 4.30 / 4.58 | 7.54 / 7.61 / 8.14 | 30.77 / 30.90 / 32.53 | 29.89 / 30.01 / 32.85 |
| near 8 | 4.28 / 4.31 / 4.76 | 7.54 / 7.63 / 9.04 | 31.03 / 31.91 / 33.95 | 28.74 / 29.65 / 31.06 |
| near 16 | 4.26 / 4.29 / 4.62 | 8.07 / 8.10 / 8.35 | 33.82 / 33.95 / 34.24 | 30.94 / 31.07 / 32.52 |
| bound 64 | 4.28 / 4.32 / 4.83 | 8.33 / 8.40 / 9.42 | 34.13 / 34.27 / 34.54 | 31.87 / 31.98 / 33.35 |
| default (64) | 4.32 / 4.35 / 5.68 | 8.25 / 8.75 / 10.15 | 34.37 / 35.05 / 40.10 | 34.35 / 34.61 / 36.86 |
| 128 | 4.34 / 4.38 / 4.69 | 10.00 / 10.07 / 10.77 | 35.47 / 35.89 / 42.49 | 35.61 / 35.75 / 47.87 |

Process CPU time over wall time (median): with 64 default threads, 1.00 on
`damped_sine` (the pool never woke), 33 to 84 on the others on the CPU solver
and 12 to 63 on the GPU solver (idle threads spinning); with 8 bound threads
6.4 to 10.7 (CPU) and 1.0 to 8.3 (GPU).

**Block I, the same setting in 5 processes** (medians of the 5 processes,
lowest-highest, and their difference; largest max/median of a process):

| Setting | damped_sine | many_outputs | chirp | sin(2e5 t) |
|---|---|---|---|---|
| GPU, default | 4.30-4.48 (4.4 %), 1.37 | 8.19-8.37 (2.2 %), 1.07 | 34.05-34.75 (2.1 %), 1.12 | 32.16-34.58 (7.5 %), 1.09 |
| GPU, near 8 | 4.27-4.33 (1.4 %), 1.16 | 7.68-7.89 (2.7 %), 1.21 | 31.73-32.58 (2.7 %), 1.09 | 29.39-30.21 (2.8 %), 1.09 |
| CPU, bound 8 | 2.80-2.92 (4.3 %), 1.11 | 5.13-5.17 (0.9 %), 1.08 | 11.92-12.14 (1.8 %), 1.11 | 22.35-22.77 (1.9 %), 1.26 |
| CPU, default | 2.65-2.68 (1.1 %), 1.11 | 5.97-6.08 (1.9 %), 1.04 | 12.62-12.74 (0.9 %), 1.09 | 30.30-30.75 (1.5 %), 2.48 |

The first GPU process of the allocation (block A, `gc_default_cuda`) had a
`many_outputs` median of 9.87 ms, 18 % above the 8.16 to 8.75 ms of every
later process with the same threads (blocks D, F, H, I); `damped_sine` and
`chirp` did not differ. One process, so not explained (the GPU's clocks were
at 210 MHz, idle, when the node file was written just before it).

**Block B, first use** (the baseline's sequence: plain, then the four
variants, 5 runs each, the variants without a warm-up of their own; first run
over the median of the others):

| Variant, integrand | lazy loading (default) | `CUDA_MODULE_LOADING=EAGER` + one `torch.bmm` first |
|---|---|---|
| `result_device` the solver's, damped_sine | 198.13 ms against 6.12 (32.6 times) | 15.00 against 5.92 (2.5 times) |
| `result_device` the solver's, many_outputs | 17.45 against 9.10 (1.9 times) | 10.16 against 9.12 (1.1 times) |
| every other variant and integrand | 0.90 to 1.16 times | 0.89 to 1.13 times |

Three of the 6,320 timed calls made a device allocation (`num_device_alloc`,
one each): the first call with the record on the GPU, `damped_sine` in the
lazy process (198.13 ms) and `sin(2e5 t)` in both (31.2 and 31.3 ms, against
medians of 29.8 and 29.9: 5 %). So the 271.8 ms of 2.2 was a first-use
cost of keeping the record on the GPU (kernels loaded at first launch, cuBLAS
set-up), about 190 ms on this node, which a warm-up with the default
`result_device` does not remove.

**Blocks A, H, J: collector and glibc, CPU solver, bound 8, `sin(2e5 t)`**
(the other integrands had 0 page faults and did not change by more than
their spread):

| Setting | min / median / max (ms) | minor page faults per call (median) |
|---|---|---|
| collector default (A) | 20.16 / 22.50 / 25.40 | 2,656 |
| `gc.disable()` (A) | 19.87 / 20.50 / 22.87 | 0 |
| `MALLOC_TRIM_THRESHOLD_=4 GiB` and `MALLOC_MMAP_THRESHOLD_=32 MiB` (H) | 19.95 / 20.49 / 21.47 | 0 |
| only `MALLOC_TRIM_THRESHOLD_` (J) | 24.90 / 26.27 / 30.03 | 4,796 |
| only `MALLOC_MMAP_THRESHOLD_` (J) | 23.27 / 24.54 / 30.31 | 5,254 |
| neither (J) | 20.28 / 22.53 / 25.77 | 1,933 |

Within each CPU series of `sin(2e5 t)` with page faults, the time rises with
the faults (Spearman rank correlation +0.44 to +0.95 for the bound thread
counts of block C). The collector itself costs nothing measurable: no full
collection in any of the 6,320 timed calls, and at most 1.45 ms in
collections in any call. `many_outputs` also had 2,500 to 3,900 faults and
6.9 to 7.4 ms (against 5.1) with only one of the two settings. On the GPU
solver every setting had 0 faults.

Each row of this table is one process, so it does not separate a setting from
the state a process happens to be in. The adversarial review repeated it in
14 processes (8 bound threads, gk21, `sin(2e5 t)`, 30 calls after a warm-up;
`adv/faults.py`, `adv/logs/timings.log` and `adv/logs/faults2.log` under
`$SCRATCH/padaquad_speedup_plans/timing-variation/`). A process had either a
median of 5 to 6.5 faults per call and a median time of 19.8 to 20.4 ms, or
a median of 1,855 to 2,979 faults and 22.4 to 23.2 ms. Many faults came in 1
of 5 processes with the defaults (2,979), in 2 of 5 with `gc.disable()`
(1,855 and 2,914) and in none of 4 with both glibc variables set; no
collection ran during any of the 420 timed calls. So `gc.disable()` does not
remove the faults, and the attribution to the collector in the first version
of this section was wrong; whether both glibc variables prevent them is not
settled by 4 processes.

**Block E, NUMA placement** (GPU solver, 8 threads, `numactl
--cpunodebind --membind`): GPU 0's domain, the farthest and unbound medians
within 2.3 % on all four integrands (`many_outputs`: 7.91, 7.73, 7.89 ms;
`sin(2e5 t)`: 29.96, 30.00, 30.02 ms).

**Block F, drift**: block A's GPU configuration at the end of the allocation:
damped_sine 4.28, many_outputs 8.16, chirp 33.95, `sin(2e5 t)` 31.87 ms
(medians), against 4.30, 9.87, 34.89, 34.41 at the start (see block I for the
first process).

**CPU frequency** (`cpuinfo_avg_freq` of the main thread's CPU, readable on
the compute node): about 3.5 GHz with 1 or 2 bound threads, 3.3 GHz with 8,
3.14 GHz with 16 or more or unbound. Within series the rank correlation of
time with frequency was between -0.81 and +0.60, mostly negative. Part of the
thread-count effect is therefore the processor's boost with few busy cores
(inference).

## 3. The design

### 3.1 Candidate causes, settled

1. **The garbage collector.** Was the cause at `0590ccf` (review item 1);
   **excluded now**: no full collection in 6,320 timed calls, at most 1.45 ms
   of collection per call; D4's test keeps it so.
2. **The CUDA caching allocator.** **Excluded**: a device allocation in only
   3 timed calls, each the first that keeps the record on the GPU (part of
   item 3); no retry.
3. **First-use costs a warm-up does not remove.** **Confirmed**: about 190 ms
   in the first call that keeps the record on the GPU (block B), down to 9 ms
   with eager loading and one `torch.bmm` first. A benchmark's warm-up must use
   the same `result_device` as the timed calls.
4. **CPU frequency and thread wake-up.** **Contributes, as part of 5**: fewer
   busy cores, higher clock; idle threads spin (CPU time 33 to 84 times wall
   time with 64 default threads on the CPU solver).
5. **The thread count and binding.** **Confirmed, the largest remaining
   effect**: section 2.6, blocks C and D; and the source of the long tails
   (default unbound threads on the CPU, 128 threads).
6. **NUMA placement.** **Excluded** (block E, within 2.3 %, the spread of
   one series).
7. **Other processes on the node.** Not separable here: our allocation held
   the whole node; the system's own processes (`nv-hostengine`, `ldmsd`;
   load average 5-6 before the first run) were present throughout.
   Involuntary context switches per call were 0 to 11, except 41 and 58 with
   128 and 64 unbound threads on `sin(2e5 t)`.
8. **The random starting mesh.** **Excluded**: every call of a series
   evaluated the same points and subintervals.
9. **glibc returning memory.** **Contributes on the CPU for many
   subintervals**: a process has either about 6 or 1,855 to 2,979 faults per
   call of `sin(2e5 t)`, 2 to 3 ms apart (2.6), not decided by the
   collector; not understood well enough to advise users (one glibc
   setting alone made it worse).

### 3.2 What follows for the library

No library change. The two remaining effects users can act on are the thread
count (D2) and, for benchmarks, a warm-up that matches the timed call (3.1
item 3). The library must not set the thread count or the collector (D1), and
does not. Rejected: the library disabling or freezing the collector during
`integrate` (process-wide, and the cause is gone); the library setting threads
(process-wide, and the best count depends on the integrand and the machine).

## 4. Decisions for the user

- **D1. The library's policy on process-wide state.** (a) **Recommended and
  done:** keep it (no thread or collector settings), said in the user guide
  with what the one `gc.collect()` is for. (b) A helper or context manager for
  threads or the collector. Rejected.
- **D2. Thread advice in the user guide.** (a) **Recommended and done:** a
  subsection with the default count, how to read and set it, the Slurm
  binding, and the measured tables of 2.6 for the CPU and the GPU solver.
  (b) A sentence with the issue's old numbers. Rejected.
- **D3. How benchmarks report spread.**
  1. **Store every run's time.** Recommended; done in this branch's two
     scripts.
  2. **Show the range in PadaQuad's own table too** (`library_comparison.py`,
     branch `performance_benchmark`). Recommended; not done here.
  3. **Threads and binding** (corrected by the review, item 9): one fixed
     count for every torch-based solver, at most one per physical core, bound
     (`srun --cpu-bind=cores`), stated in the metadata with the CPU set and the
     OpenMP variables; not `len(os.sched_getaffinity(0))` (128 on a GPU node,
     two per core, the slowest setting measured). 8 bound threads were within
     2 % of the fastest for PadaQuad on the CPU's many-value integrands and
     within 4 % with the GPU solver; within 8 % on the CPU's few-value
     integrands.
  4. **The collector:** `gc.collect()` before each case; never disable or
     freeze it in timed runs. Done in this branch's two scripts.
  5. **Warm-up with the timed settings** (new, 3.1 item 3): the warm-up of a
     case must use the timed call's `result_device` and options.
  6. **More runs:** not needed: with bound threads the longest of 30 calls was
     at most 1.29 times the median and medians of five processes agreed
     within 7.5 %.
- **D4. A regression test that the Python work of a call does not grow with
  its subintervals** (6.1). Recommended and done.

## 5. Implementation plan

1. **Re-count and correct the protocol** (review). `impl/count_now.py`,
   `impl/cycles.py`, `impl/cdll_origin.py`; corrected `run_protocol.sh`,
   `variation_protocol.py`, `summarize_variation.py` (copied unchanged),
   `tables.py`. Commit: this file.
2. **Run the protocol** in one allocation, alone on the node: blocks A-F, H,
   then I and J after the base commit's suites (`impl/alloc1.sh`,
   `impl/alloc1b.sh`). Results into 2.6.
3. **D4:** `TestPythonWork` in `tests/test_base_solver.py` (6.1). Show that it
   fails on mutated copies (`impl/mutate.sh`) and passes on Python 3.11 and
   3.12 and under coverage.
4. **D3:** `examples/speed_test/torchdiffeq_comparison.py` (`measure`:
   `gc.collect()`, `times_s`; `metadata`: `cpu_affinity`, `omp_num_threads`),
   `examples/speed_test/trade_speed_for_memory.py` (`run_example`:
   `gc.collect()`, `all seconds`; `main`: `torch threads`), one sentence in
   `docs/performance.md`. Test: `tests/test_docs_performance_table.py`
   renders the committed (old-format) files unchanged.
5. **D2 and D1:** subsection "CPU threads" (label `ug-cpu-threads`) of
   "Device and precision" in `docs/user_guide/devices_memory.md`; a pointer
   in `docs/user_guide/troubleshooting.md`; `CHANGELOG.md`; `CLAUDE.md`
   (memory management: the library's settings and the one `gc.collect()`;
   test map row of `test_base_solver.py`). Tests:
   `tests/test_docs_vocabulary.py`, `sphinx-build -W`.

## 6. Tests

### 6.1 New (D4)

`TestPythonWork.test_same_for_256_times_the_subintervals`
(`tests/test_base_solver.py`), parametrized over gk21 (`atol=1e-10`) and
`interpolatory3_variable` (`atol=1e-2`) and `take_gradient` False and True,
CPU, float64, `rtol=0`, `max_batch=1e8`: integrate
cos(8 pi t) + exp(cos(2 pi t)) from a given mesh of n unit subintervals of
[0, n], n = 16 and n = 4096, one solver, after a warm-up call with n = 16.
Every unit subinterval holds the same periods, so both calls take the same
decisions in the same passes; the test checks that premise (256 times the
subintervals, at least one split). It asserts that the call on 4096 executes
the same Python lines within 1 % of the subintervals added (`sys.settrace`
line events, restored to any earlier trace function, such as coverage's) and
that its peak of Python-allocated memory (`tracemalloc`) exceeds the small
call's by less than one byte per subinterval added (a list of one pointer per
subinterval takes 8).

Measured at `3eb4507` (`impl/margins.py`, `impl/logs/margins.log`): 64 and
16,384 subintervals (gk21) and 192 and 49,152 (`interpolatory3_variable`);
lines equal exactly (2,800 to 5,864 per call) on Python 3.11.15 and 3.12.13;
peaks of 11.4 to 13.8 KiB, the large call never more than 0.4 KiB above the
small one (threshold 16 to 48 KiB). Mutations (`impl/mutate.sh`, scratch
copies, `impl/logs/mutation.log`), each a line inserted at the top of the
loop of `integrate`:

| Mutation | Result |
|---|---|
| none | 4 passed |
| m1: the old dictionary, `{tuple(row): i for i, row in enumerate(mesh.tolist())}` | 4 failed (28,560 more lines; 3.4 MB more Python memory) |
| m2: a loop reading each mesh point as a float, one object at a time | 4 failed (28,560 more lines) |
| m3: no loop, `self._kept = mesh.tolist()` | 4 failed (2.3 MB more Python memory) |

It also passes under `pytest --cov` (the CI's coverage, Python 3.11 venv of
`$SCRATCH/padaquad_ci`), with no warning. Not counted: Python work inside a C
extension that never returns to the interpreter and allocates no Python
object.

### 6.2 Results unchanged

No library code changes. The protocol recorded the integral of every call:
one value per integrand and device across every configuration.

## 7. How to measure

Scripts in `/pscratch/sd/k/khegazy/padaquad_speedup_plans/timing-variation/impl/`
(the planning versions beside them are superseded):

- `run_protocol.sh [blocks]`, inside an allocation that holds the whole node:
  one process per configuration, `RUNS` timed calls (default 30) per
  integrand after one warm-up; `results/<name>/runs.jsonl`, node files,
  `summary.txt`, `analysis.txt`.
- `variation_protocol.py`: one configuration; per call wall time, main-thread
  and process CPU time, host time per phase (`_evaluate_f_on_mesh`,
  `_compute_error_ratios`, `_adaptively_increase_mesh`, `_record_results`,
  `_failed_record_positions`, `_sort_record`, `_get_optimal_mesh`),
  collections per generation and time in them, CUDA allocator counters, page
  faults, context switches, start and end CPU and its frequency, calls and
  points of `f`, subintervals, the integral; torch's default thread count.
- `summarize_variation.py` (unchanged), `tables.py` (the tables of 2.6).

| Block | Configurations | Settles |
|---|---|---|
| A `gc` | GPU (default threads) and CPU (bound 8) × collector default / `gc.disable()` | 1, allocator counters for 2 |
| B `firstuse` | the baseline's sequence with variants, 5 runs: lazy loading; `CUDA_MODULE_LOADING=EAGER` + one `torch.bmm` first | 3 |
| C `threads_cpu` | CPU: bound 1-64, 128 (two per core), default, default under `taskset -c 0-7`, `OMP_WAIT_POLICY=passive` at bound 8 and 64 | 5 |
| D `threads_gpu` | GPU: near 1, 8, 16, bound 64, 128, default | 5 for GPU solvers |
| E `binding` | GPU, 8 threads, NUMA near / far / unbound | 6 |
| H `malloc` | both glibc settings, CPU bound 8 and GPU | 9 |
| F `drift` | block A's GPU configuration again | 4, 7 |
| I `repeat` | GPU default, GPU near 8, CPU bound 8, CPU default, 5 processes each | spread between processes |
| J `malloc_split` | each glibc setting alone and neither, CPU bound 8 | 9 |

Blocks A-H and F took about 10 minutes, I and J about 4.

## 8. Documentation updated

- `docs/user_guide/devices_memory.md`: subsection "CPU threads" (D1, D2).
- `docs/user_guide/troubleshooting.md`: "A call is slower than expected, or
  its time varies", a pointer to it.
- `docs/performance.md`: what the benchmark records about spread and threads.
- `CHANGELOG.md` ("Added"), `CLAUDE.md` (memory management; test map).

## 9. Risks and interactions with the other plans

- **`compiled-turn`** (in progress beside this plan) may remove operations
  from a pass, which changes how much the thread count matters; the tables of
  the user guide name the commit they were measured at. A compiled function
  or captured graph also adds first-use costs that a warm-up on another
  integrand or setting does not cover (3.1 item 3).
- **The thread tables are one node type.** Another CPU (more or fewer cores,
  another cache layout) can have another best count; the user guide says so.

## 10. Confidence

- **Sure (measured or counted):** every number of 2.6 and 6.1 (one node,
  one allocation, alone on the node); no collection of generation 1 or 2 and
  no per-subinterval Python work at `3eb4507`; torch's default thread counts
  of review item 5; the library sets no thread or collector setting (grep).
- **Fairly sure:** that the thread count and binding are now the largest
  user-controllable source of time and spread on this node (two independent
  blocks, C/D and I, agree).
- **Not explained:** the 18 % slower `many_outputs` in the allocation's first
  GPU process; what gives a CPU process the many page faults of
  `sin(2e5 t)` (2.6: not the collector; with both glibc variables none of 4
  processes had them) and why either glibc setting alone adds faults;
  the rule behind torch's default
  thread count (inferred from counts, not read in torch's source).
- **Not checked:** other node types; Python 3.13 and 3.14 (the test's
  counts do not use the collector, but were run only on 3.11 and 3.12);
  `take_gradient=True` timings (counted, not timed).

## Implemented (branch `timing_variation`, 2026-11-26)

Scripts, raw results and logs: `$SCRATCH/padaquad_speedup_plans/timing-variation/impl/`.

| Step (5) | Commit | What |
|---|---|---|
| 1, 2 | `d5548b3` | this file, reviewed, with the protocol's measurements (2.6) |
| 3 (D4) | `fad0628` | `TestPythonWork` in `tests/test_base_solver.py` |
| 4 (D3) | `f8da744` | both benchmark scripts keep every run's time, collect garbage before each case, record threads; `docs/performance.md` |
| 5 (D1, D2) | `c543780` | user guide "CPU threads", Troubleshooting pointer, CHANGELOG, CLAUDE.md |

No change to `padaquad/`. The test suites of the base commit (`3eb4507`, on
the node of 2.6, after the protocol): CUDA 4,622 passed, 25 skipped; CPU
(`CUDA_VISIBLE_DEVICES=""`) 4,288 passed, 359 skipped; no failure
(`impl/logs/suite_base_*.log`). The suites, docs build and checks at the
final commit are in the implementing agent's final report
(`impl/logs/suite_final_*.log`, `impl/logs/docs_final.log`). The base
suites above ran with an uncommitted draft of `TestPythonWork` already in the
worktree; the clean base comparison is from an archive of `3eb4507`
(`impl/logs/suite_base2_*.log`).

### Not done

- **`performance_benchmark` (not this branch's work).** For
  `examples/speed_test/library_comparison.py`: in `measure`, `gc.collect()`
  before the warm-up and `times_s` (every run's time) in the result; make
  the warm-up use the timed call's options, including `result_device` (3.1
  item 3); in `main`, replace `torch.set_num_threads(len(os.sched_getaffinity(0)))`
  (128 on a GPU node, two threads per core, the slowest setting of 2.6) by
  one fixed count for every torch-based solver, at most one per physical
  core (8 was within 8 % of the fastest for PadaQuad in every case of
  2.6), from a command-line option; in `metadata`, the count, the CPU set,
  `OMP_*` and `GOMP_SPINCOUNT`; in `render_markdown`, the range beside the
  median in PadaQuad's own table. In `run_benchmark.sh`, start the Python
  process with `srun --cpu-bind=cores` (with `-c` twice the number of cores
  wanted, on a node with two hardware threads per core) when `SLURM_JOB_ID`
  is set. A results file without `times_s` must still render (as on this
  branch: only keys are added).
- **The NVML library loaded at every solver construction** (review item 2):
  `DistributedEnvironment.set_device` calls `torch.cuda.device_count()` even
  for a CPU solver, which loads `libnvidia-ml.so.1` through `ctypes` in a
  process that has not initialized CUDA and leaves 12 objects in reference
  cycles per construction. The adversarial review measured 60 µs per CPU-solver
  construction with it and 15 µs with `self.local_rank and` before the
  `device_count()` call (11 objects in cycles, then none; compute node, CUDA
  not initialized); it is a per-call, not a per-subinterval, cost, and the
  fix (calling it only for a CUDA device) adds
  a branch to the constructor, so it is left to the call-setup work.
- **glibc page faults on the CPU** (3.1 item 9): a process runs `sin(2e5 t)`
  with about 6 or 1,855 to 2,979 faults per call, 2 to 3 ms apart; the
  collector does not decide it (many faults in 2 of 5 processes with
  `gc.disable()`), and none of 4 processes with both glibc variables had them;
  not explained, no advice given.
- **The 18 % slower first GPU process** of the allocation (block I) has one
  instance and no explanation.
