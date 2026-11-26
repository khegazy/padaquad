# Every solver starts and ends `torch.distributed`'s default process group

Status: open. Recorded 2026-11-26 from a review of the speed-up work and checked
on commit `40bcfa4` (branch `robustness_fixes`; `padaquad/distributed.py` is the
same as on `main`) with torch 2.8.0. Written so that another agent can pick it
up without the original session. Nothing in `padaquad/` was changed for it.

## Summary

When the environment says the process is part of a distributed job (the
`RANK` variable is set, as `torchrun` does), every PadaQuad solver creates
PyTorch's default process group when it is constructed and destroys it when
it is deleted. PadaQuad never communicates through that group. As a result:

- a second solver cannot be constructed while a first one exists;
- `padaquad.integrate`, which builds a solver per call, creates and destroys a
  process group on every call;
- a solver cannot be constructed in a program that created its own process
  group (a `torch.distributed` training loop), and the failed construction
  destroys the program's group;
- deleting a solver destroys a group that the program may still use.

A second, smaller finding from the same module: every solver construction asks
for the number of GPUs, which loads the NVIDIA management library and leaves
objects for the garbage collector, also for a CPU solver.

## Terms

- **Process group**: the set of processes of a distributed PyTorch job that
  communicate with each other (`torch.distributed`). The **default process
  group** is the one `torch.distributed.init_process_group` creates; there is
  one per process, and calling `init_process_group` again while it exists
  raises `ValueError`. `torch.distributed.destroy_process_group()` removes it.
- **Rank** and **world size**: a process's number in the job and the number of
  processes. `torchrun` starts the processes and sets the environment variables
  `RANK`, `LOCAL_RANK` (the rank within one node), `WORLD_SIZE`, `MASTER_ADDR`
  and `MASTER_PORT` in each.
- **Collective operation**: an operation every process of the group takes part
  in, such as `all_reduce` (used by `DistributedDataParallel`, DDP, to average
  gradients).
- **Backend**: the library a group communicates through: `"nccl"` (GPUs),
  `"gloo"` (CPUs and GPUs) or `"mpi"`.
- **NVML**: the NVIDIA Management Library (`libnvidia-ml.so.1`), which PyTorch
  queries for the number of GPUs before CUDA is initialized.

## When PadaQuad treats a job as distributed

`DistributedEnvironment` (`padaquad/distributed.py`) is the root of the solver
classes: `SolverBase.__init__` (`padaquad/base.py`) calls
`DistributedEnvironment.__init__`, which receives the constructor's extra
keywords (`backend`, default `"nccl"`; `master_addr`; `master_port`;
`is_slurm`, default False; `AdaptiveQuadrature.__init__` passes them through
`**distributed_options`).

`DistributedEnvironment.check_distributed` decides:

- by default, the job is distributed when the environment variable `RANK` is
  set to any value other than `-1`
  (`int(os.environ.get("RANK", "-1")) != -1`). `torchrun` sets it in every
  process it starts, also with one process;
- with `is_slurm=True`, when `SLURM_NTASKS` is above 1. That path then raises
  `ValueError` in `init_slurm_environment` ("Find new way to get hostname"),
  before any process group is created, so it cannot run today.

Then, for a distributed job, `init_distributed_process` reads `RANK`,
`LOCAL_RANK` and `WORLD_SIZE` (a `KeyError` if one is missing) and calls
`torch.distributed.init_process_group(backend=..., world_size=...,
init_method="env://")`. `SolverBase.__del__` calls
`DistributedEnvironment.end_process`, which calls
`torch.distributed.destroy_process_group()` for a distributed job.
`SolverBase.__del__` runs it whenever the object has the attribute
`is_distributed`, which `check_distributed` sets before the group is created,
so it also runs for a solver whose construction failed in
`init_process_group`.

No code in `padaquad/quadrature/` calls a collective operation. The only other
use of `torch.distributed` is a check: after an out-of-memory error in a
backward pass, `_raise_after_backward_oom` (`padaquad/quadrature/memory.py`)
refuses to restart the integral when `torch.distributed.is_initialized()` and
the world size is above 1 (see `docs/issues/multi-gpu-backward-oom-restart.md`).

## Evidence

The script `dist_group.py` (kept in
`/pscratch/sd/k/khegazy/padaquad_speedup_plans/robustness-fixes/impl/`) runs one
process with the gloo backend on the CPU, with the environment that `torchrun`
would set for one process:

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29535 \
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<checkout> \
python -u dist_group.py
```

Its solvers are built with `adaptive_quadrature("uniform", "gk21",
device="cpu", backend="gloo")`; `padaquad.integrate` is called with
`device="cpu", backend="gloo"`. What it printed (torch 2.8.0, Perlmutter login
node), with `torch.distributed.is_initialized()` after each step:

1. **Two solvers.** Constructing the first solver creates the group
   (`is_distributed` True, rank 0, world size 1). Constructing a second while
   the first exists raises `ValueError: trying to initialize the default
   process group twice!`. Once that failed object is freed, its `__del__`
   destroys the group (`is_initialized()` False), although the first solver
   still exists. Deleting the first solver then prints `Exception ignored in:
   <function SolverBase.__del__ ...>` with an `AssertionError` from
   `destroy_process_group` (`assert pg is not None`, as there is no group left).
2. **`padaquad.integrate`.** Inside each call the group exists (the integrand
   records `is_initialized()` True), and after the call returns it is gone
   (False): every call creates and destroys a group. The solver is freed as
   the call returns, by reference counting (no `gc.collect()` was needed).
3. **A group the program created first** (`init_process_group(backend="gloo",
   world_size=1, rank=0)`, as a training script does after `torchrun`
   starts it). Constructing a solver raises `ValueError: trying to
   initialize the default process group twice!`, and after the failed object
   is freed the program's group is destroyed (`is_initialized()` False).
4. **A group the program creates while a solver exists.** The program's
   `init_process_group` raises the same `ValueError`; the program can use the
   solver's group (`all_reduce` works), but after `del solver` the next
   `all_reduce` raises `ValueError: Default process group has not been
   initialized`.

## Why it matters

A user who trains a model with `torch.distributed` (for example DDP, started by
`torchrun`) and calls PadaQuad inside the training loop, for example to
integrate a learned integrand, has `RANK` set and a process group of their own.
Constructing a solver there raises `ValueError` and destroys the user's group,
so the training loop's next collective operation fails. If the user constructs
a solver before creating their group, the user's `init_process_group` raises
instead, and deleting the solver later destroys the group the training loop
uses. `padaquad.integrate` cannot be used in such a program at all. A program
with `RANK` set and no group of its own can use one solver at a time, and pays
for creating and destroying a group (a rendezvous through `MASTER_ADDR` and
`MASTER_PORT`) with every solver; with `padaquad.integrate`, on every call. The
group PadaQuad creates is never used by PadaQuad itself.

## Other process-wide effects of `padaquad/distributed.py`

- `init_distributed_process` writes `os.environ["MASTER_ADDR"]` and
  `os.environ["MASTER_PORT"]` when `master_addr` and `master_port` are given
  to the constructor.
- `init_slurm_environment` would write `MASTER_ADDR`, `MASTER_PORT`,
  `WORLD_SIZE`, `LOCAL_RANK` and `RANK`; it raises before reaching those
  lines today (the code after the `raise` is unreachable).
- `set_device` calls `torch.cuda.set_device(self.device)` for a distributed job
  on a CUDA device, which changes the current CUDA device of the whole
  process (the device that tensors created with `device="cuda"` and NCCL
  operations use), for the rest of the program.
- `check_distributed` raises `AssertionError` when the requested backend is
  not available (for example the default `"nccl"` in a PyTorch build without
  NCCL), and `SyntaxWarning` (raised, not warned) for a backend other than
  `"nccl"`, `"mpi"` and `"gloo"`.

## Second finding: the GPU count at every construction

`set_device` computes its default GPU index as
`self.local_rank if torch.cuda.device_count() > 1 else 0`, so every solver
construction calls `torch.cuda.device_count()`, also for a CPU solver and for
a job that is not distributed. Before CUDA is initialized, PyTorch does not
cache that count (`torch/cuda/__init__.py`, `device_count`: "Do not cache the
device count prior to CUDA initialization") and asks NVML for it, which loads
`libnvidia-ml.so.1` through `ctypes`. Measured with the script
`construct_cost.py` of the review of the `timing-variation` plan
(`/pscratch/sd/k/khegazy/padaquad_speedup_plans/timing-variation/adv/`; 200
constructions of a CPU solver in a fresh process with CUDA visible but not
initialized; `asis` is the code, `fix` skips the call when `local_rank` is 0):

- 11.0 objects left in reference cycles per construction as the code is, and
  0.0 with the call skipped (measured again for this issue on a login node,
  `python construct_cost.py asis cpu` and `... fix cpu`, at commit `40bcfa4`);
- 60 µs per construction as the code is and 15 µs with the call skipped (the
  reviewer's measurement on a compute node, recorded in
  `docs/issues/2026-11-25-speedup-timing-variation-plan.md`, section "Not
  done"; login-node times are not reliable and are not quoted here).

Since `padaquad.integrate` builds one solver per call, every call pays this.
The same plan (section "Review", item 2) found these objects to be what still
makes Python's garbage collector run a young collection, about every 58 calls
of `padaquad.integrate`. For a CUDA solver the count is cached
once CUDA is initialized, so the cost falls on CPU solvers.

## Options (not chosen)

For the process group:

1. **Never create or destroy a group.** Read the rank, the local rank and the
   world size from an existing group (`torch.distributed.is_initialized()`)
   or from the environment, and use them only to choose the GPU. PadaQuad does
   not communicate, so it needs no group of its own. Removes most of
   `init_distributed_process`, `end_process` and `SolverBase.__del__`. A
   program that relied on PadaQuad to create its group (none is known; no test
   or example does) would have to create it itself.
2. **Create a group only when none exists, and destroy only a group PadaQuad
   created.** Keeps the current behaviour for a program without a group, but
   needs a process-wide owner count (several solvers share the one default
   group), and `padaquad.integrate` would still create and destroy a group per
   call.
3. **Make it explicit.** A function such as `padaquad.init_distributed(...)`,
   or a constructor option, that a user calls when PadaQuad should set up the
   group; by default the solver does not touch `torch.distributed`.

For the GPU count: ask for it only when the device is CUDA and `local_rank` is
above 0 (the reviewer's `fix`, one extra condition in `set_device`), or let
`_resolve_device` choose the default index, since it already asks for the
count when it checks a CUDA index.

## Checks before choosing

- Run two processes under `torchrun --nproc_per_node=2` with a DDP-wrapped
  model whose loss calls `padaquad.integrate`, as today (expected to raise as
  in step 3 above) and with the chosen option.
- Search users' code and the examples for `backend=`, `master_addr=`,
  `master_port=` or `is_slurm=` passed to a solver: those are the only callers
  that use the process group setup on purpose.
- Decide what `set_device` should do with `LOCAL_RANK` when the job is not
  distributed for PadaQuad but the program runs several processes on one node.
