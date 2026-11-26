# Devices and memory

## Device and precision

`device` selects where the integrand is evaluated: `"cpu"`, `"cuda"`,
`"cuda:N"` or a `torch.device`; `None` picks the GPU if one is available. The
solver never changes PyTorch's global default device. `result_device` is where the per-subinterval arrays of the result go (nodes,
integrand values, contributions, errors), to keep them off the GPU, and, with
`trade_speed_for_memory=True`, where the integrand values go during the
integration ({ref}`ug-trade-speed-for-memory`); `None` keeps them on the
solver's device. The integral, its error estimate and the
state's mesh stay on the solver's device.

The solver computes in the dtype of the bounds or mesh given to `integrate`,
float64 or float32. A float32 tensor such as `torch.tensor([1.0])` therefore
makes the integration run in float32.

(ug-cpu-threads)=
### CPU threads

A call of `integrate` is a sequence of many small tensor operations in every
pass: the calls of `f`, the error control, the mesh update and the record.
PyTorch can split one operation among several CPU threads, which speeds up a
large operation and slows down a small one, because starting the threads costs
time and idle threads keep their cores busy while they wait for work. The
thread count therefore matters on the CPU, and also with a GPU solver, whose
record (with the default `result_device="cpu"`) and state's mesh are built on
the host.

`torch.get_num_threads()` returns the count. On the node measured below, with
`OMP_NUM_THREADS` unset, it was 64 (the node's 64 cores, which have two
hardware threads each, 128 logical CPUs) in a process allowed to run on every
logical CPU, and 8 in a process allowed 8 logical CPUs, even when those were
the two hardware threads of 4 cores: `srun -c 8 --cpu-bind=cores` gave a job
step exactly that, so PyTorch put two threads on each core. Set the count
before the first integral: with `torch.set_num_threads(n)` in Python, or with
`OMP_NUM_THREADS=n` in the environment before Python starts. On a Slurm
cluster, bind each task to its cores and use at most one thread per core: on
this node, `srun -c 16 --cpu-bind=cores` gave 8 cores (16 logical CPUs), on
which `OMP_NUM_THREADS=8`, `OMP_PLACES=cores` and `OMP_PROC_BIND=close` (the
binding of the tables below) place one thread on each.

PadaQuad itself never changes the thread count or the settings of Python's
garbage collector, and outside a distributed job (several processes joined by
`torch.distributed`) it sets no environment variable. Its only use of the
collector is a `gc.collect()` after an out-of-memory error, before it retries
with fewer points: it frees objects that refer to each other in a cycle and
may still hold the failed call's tensors. The same collection runs at the end
of the memory benchmark of `get_max_batch(f=...)` and before an integral is
restarted after an out-of-memory error in a backward pass.

The tables give the median time of 30 calls in ms, with the shortest and
longest in parentheses, measured on one NERSC Perlmutter GPU node (one AMD
EPYC 7763 with 64 cores; an NVIDIA A100 with 40 GB), PyTorch 2.8.0, Python
3.12, PadaQuad commit `3eb4507`. Each call: `gk21`, `atol=1e-8`, `rtol=0`,
float64, `max_batch=1_000_000`, `take_gradient=False`, after
`torch.manual_seed(0)`; one process per thread count, with one call of each
integrand before its 30 timed ones. "Bound" means one thread per core, on n
cores (`OMP_PROC_BIND=close`, `OMP_PLACES=cores` and `taskset`; with the GPU
solver, cores beside the GPU); "64, default" is PyTorch's own count with
nothing set; 128 is two threads per core. The integrands, on [0, 1]:

- `exp(-t) sin(40 t)`: 11 subintervals, one pass of 231 points;
- `sin(k t)` for 1,000 values of k from 1 to 100 (1,000 outputs): 13
  subintervals, 2 passes, 315 points with 1,000 values each;
- `sin(1 / (t + 0.001))`: 76 subintervals, 11 passes, 2,961 points;
- `sin(200,000 t)`: 13,036 subintervals, 13 passes, 547,281 points.

On the GPU the random starting mesh differs, so the last two had 77 and 13,737
subintervals there.

**CPU solver**

| Threads | `exp(-t) sin(40 t)` | 1,000 outputs | `sin(1 / (t + 0.001))` | `sin(200,000 t)` |
|---|---|---|---|---|
| 1, bound | 2.70 (2.68-3.01) | 6.67 (6.63-6.89) | 11.5 (11.4-12.5) | 31.7 (27.0-37.4) |
| 2, bound | 2.68 (2.64-3.17) | 5.64 (5.51-5.88) | 11.3 (11.2-12.3) | 26.6 (22.8-30.9) |
| 4, bound | 2.68 (2.63-2.96) | 5.13 (4.94-5.40) | 11.5 (11.3-12.4) | 23.4 (20.5-27.6) |
| 8, bound | 2.85 (2.72-3.12) | 5.23 (4.98-5.43) | 12.1 (11.6-12.8) | 23.0 (20.6-26.9) |
| 16, bound | 2.99 (2.94-3.19) | 5.72 (5.68-5.82) | 12.6 (12.5-13.6) | 24.2 (21.9-27.4) |
| 32, bound | 3.00 (2.97-3.25) | 6.00 (5.96-6.30) | 12.8 (12.8-13.9) | 24.9 (23.5-26.3) |
| 64, bound | 3.03 (2.99-3.26) | 6.44 (6.38-6.69) | 12.7 (12.7-13.7) | 26.3 (24.1-34.0) |
| 64, default | 2.70 (2.66-3.02) | 6.07 (6.03-6.23) | 12.8 (12.8-14.0) | 30.6 (27.9-76.8) |
| 128 | 3.67 (3.63-14.3) | 7.77 (7.57-7.81) | 15.9 (15.9-40.8) | 36.8 (32.9-47.3) |

**GPU solver**

| Threads | `exp(-t) sin(40 t)` | 1,000 outputs | `sin(1 / (t + 0.001))` | `sin(200,000 t)` |
|---|---|---|---|---|
| 1, bound | 4.30 (4.28-4.58) | 7.61 (7.54-8.14) | 30.9 (30.8-32.5) | 30.0 (29.9-32.9) |
| 8, bound | 4.31 (4.28-4.76) | 7.63 (7.54-9.04) | 31.9 (31.0-34.0) | 29.7 (28.7-31.1) |
| 16, bound | 4.29 (4.26-4.62) | 8.10 (8.07-8.35) | 34.0 (33.8-34.2) | 31.1 (30.9-32.5) |
| 64, bound | 4.32 (4.28-4.83) | 8.40 (8.33-9.42) | 34.3 (34.1-34.5) | 32.0 (31.9-33.4) |
| 64, default | 4.35 (4.32-5.68) | 8.75 (8.25-10.2) | 35.1 (34.4-40.1) | 34.6 (34.4-36.9) |
| 128 | 4.38 (4.34-4.69) | 10.1 (10.0-10.8) | 35.9 (35.5-42.5) | 35.8 (35.6-47.9) |

On this node:

- **Few values per pass** (`exp(-t) sin(40 t)` and `sin(1 / (t + 0.001))`,
  a few hundred points with one output): 1 to 4 threads were within 2 % of the
  fastest on the CPU; PyTorch's default of 64 took up to 14 % longer, and 128
  up to 41 % longer.
- **Many values per pass** (1,000 outputs, and `sin(200,000 t)`): 4 to 8
  threads were within 2 % of the fastest on the CPU; 1 thread took up to 38 %
  longer, the default of 64 up to 33 % and 128 up to 60 %. With the default,
  the longest of the 30 calls of `sin(200,000 t)` took 2.5 times their
  median, against at most 1.3 times with bound threads.
- **With the GPU solver**, 1 to 8 threads were within 4 % of the fastest on
  every integrand; the default of 64 took up to 17 % longer.
- Every thread count and binding returned the same integral, bit for bit.

The best count depends on the integrand, the machine and what else runs on
it: time a few counts on your own integral before a long run.

(ug-memory)=
## Batch size and the memory benchmark

PadaQuad evaluates the nodes of many subintervals in one call of the
integrand. The number of points per call is the batch size, `max_batch`.

**How `integrate` chooses it.** A `max_batch` given to the call (or to the
constructor, when no `state` is given) is used as it is and never grows. A
`state` (the state an earlier call returned, {ref}`ug-reuse`) gives the batch
size to start from. With neither, the first call of the integrand is the
first pass, at most 10,000 points (for the default starting mesh of gk21, 231
points), and the batch size grows from there. No call of the integrand is
made for anything but the integral: every point it is called at belongs to
it.

**How it grows.** While the batch size has not converged, a call of the
integrand that has more points waiting than the batch size tries 10 times as
many (or all that are waiting). The first pass of a call therefore runs whole
when it is at most 10 times the batch size. When a call runs out of memory, it
is evaluated again at three quarters of its size, repeatedly, until it fits;
that size becomes the batch size, and the batch size has then converged
(`state.max_batch_converged`): it never grows again, in this call or in any
call given its state. Reaching the limit this way costs a few failed calls of
the integrand once per state (measured on an A100, for a network filling the
GPU: 0.6 s more than a run with the right batch size from the start, 1.4 s
from a cold call).

**With `take_gradient=True`.** PadaQuad backpropagates every batch right after
evaluating it. On a GPU, before each backward pass it compares the memory
still usable (keeping `1 - total_mem_usage` of the total free) with what the
backward pass is predicted to need: the batch's forward peak memory, times the
ratio of backward to forward peak memory the previous batch measured (1 for
the first batch). If it does not fit, the batch is dropped before any gradient
is written and evaluated again, smaller. If a backward pass still runs out of
memory, PadaQuad puts back the gradients the call added, lowers the batch
size, and restarts the integral from its starting mesh, with a warning. It
finds those gradients by walking each batch's autograd graph. A custom
autograd function (for example activation checkpointing with
`use_reentrant=True`, or a function compiled with `torch.compile`) can hide
parameters from that walk; pass
`parameters=model.parameters()` (or the optimizer) in that case, else
PadaQuad raises `BackwardOutOfMemoryError`, whose `state` is the state to call
again with after resetting the gradients. It raises it also when a parameter
has a post-accumulate-grad hook, or when `torch.distributed` runs more than one
process. An out-of-memory error in `loss_fxn` is handled the same way as one in
the backward pass. One solver serves one thread: an integrand may integrate with
the same solver (a double integral), but two threads must not share one.

**Measuring before any integral.** `solver.get_max_batch(f=f, t=t,
take_gradient=...)` runs the memory benchmark: it fits the memory of calls on
1, 10, ... points as a + b N + d N² and returns the largest batch whose fitted
memory fits in three quarters of the usable memory. It can take up to a minute
(`limit_benchmark_time=False` removes that limit) and has the limitations
listed below; measure once and pass the value as `max_batch`.
`solver.get_max_batch()` without `f` returns the constructor's `max_batch`,
else the value of the last measurement with `f`.

With `take_gradient=False`, `total_mem_usage` does not limit the batch size:
calls of the integrand grow until they run out of memory. Out-of-memory errors
from PyTorch's allocators and from CUDA libraries are recognized; any other
error from the integrand is raised unchanged.

(ug-memory-limitations)=
### Limitations

**Of the growing batch size in `integrate`**

- *Growth needs a clean failure.* The batch grows until the integrand raises
  an out-of-memory error. On a CPU an allocation that does not fit can get the
  process killed instead (Linux accepts most allocations and kills the process
  when memory runs out), and a GPU driver that falls back to system memory
  (NVIDIA's on Windows) slows down instead of failing. The first call (the
  first pass, up to 10,000 points) is not preceded by a smaller one, and
  every later call is at most 10 times one that ran. PadaQuad is meant for
  GPUs; on a CPU, pass `max_batch` when memory matters.
- *A first call far above the limit.* Each failed call shrinks by a quarter,
  so reaching the limit takes about 8 failed calls per factor of 10 between
  the first call and the limit: for an integrand that holds 60 points, 5
  failed calls from the default first pass of `gk21` (231 points) and 18 from
  a first pass of 10,000 points (a mesh of 500 subintervals). This happens
  once per state; pass `max_batch` or the state to avoid it.
- *Memory is not kept free.* The forward calls fill the memory until they
  fail; other processes on the same GPU can run out of memory meanwhile.
  `total_mem_usage` limits only the check before a backward pass.
- *A size that shrank never grows back* (see {ref}`ug-reuse`).
- *Results with `take_gradient=True`.* A pass larger than the batch size is
  spread over several batches, each judged against the running integral of
  the batches before it, so the result can differ from a run with larger
  batches (with low-order methods by more than the tolerance, usually with
  `converged=False`).
- *The check before a backward pass* resets the GPU's peak-memory counter
  (`torch.cuda.reset_peak_memory_stats`) before every batch, so a user reading
  `torch.cuda.max_memory_allocated()` around an `integrate` call sees the peak
  of its last batch only. Its first batch assumes the backward pass needs as
  much memory as the forward pass, which activation checkpointing breaks (4 to
  6 times as much, measured); the restart handles that batch.
- *A restart repeats side effects* of the integrand and of `loss_fxn`
  (counters, random draws, running statistics) for the batches before the
  failure.

**Of `get_max_batch(f=...)`**


The benchmark is meant for the common case: an integrand whose memory grows
linearly with the number of points N, or at most like N², measured where every
allocation can be seen (a GPU, or Linux with glibc). It cannot be right for
every integrand on every machine. It can choose a batch far smaller than what
fits, which leaves the hardware underused, and for some integrands a batch
larger than what fits. **Passing your own `max_batch`, chosen for the
integrand and the machine, avoids all of this**: the benchmark does not run,
and the batch is the value given. Reusing the benchmark's result, through
`solver.get_max_batch()`, only skips the time it takes; the
batch keeps the limitations below. The known
limitations follow. Numbers marked "simulated" come from fake memory readings
and a fake clock (`tests/test_memory_benchmark.py` and review scripts);
numbers marked "measured" come from real runs on an Apple-silicon Mac. None
was measured on Linux or on a GPU.

**Memory readings**

1. *Hidden memory on macOS and Windows.* On a CPU, memory is read as the
   memory the operating system has given the process, so memory the integrand
   takes from memory the process freed earlier is missed. Only Linux with
   glibc can return such memory first (`malloc_trim`). In a process that has
   already evaluated the integrand, readings on macOS and Windows can be far
   too low. Measured, in 14 repeated `integrate` calls before the batch was
   capped at the largest test call: 6 batches needed 1.2 to 35 times the
   usable memory, and the test calls themselves exceeded it in 7 calls, by up
   to 6.9 times. The cap now bounds the batch, but not the test calls.
2. *The cap underuses memory.* On macOS and Windows the batch is at most the
   largest test call, which can be well below what fits when the memory per
   point stops changing early. Measured, first call in a fresh process:
   `torch.sin` got 41.8 million points where the fit allowed 143 million; a
   small network with `take_gradient=True` got 1.24 million where the fit
   allowed 2.0 million.
3. *A preloaded allocator on Linux is not detected.* With jemalloc or tcmalloc
   loaded in place of glibc's allocator, `malloc_trim` still exists but does
   not reach that allocator, so readings can be far too low while the
   benchmark trusts them (calls twice as large as the previous one, and no
   cap). With noisy readings the fit can then follow a slope far too small
   (simulated: 1 of 60 runs gave a batch needing 25 times the usable memory).
4. *False "faster than N²" errors.* A reading of 0 is raised to the bytes of
   the call's input and output. From about a million points on, that passes
   the 4 MiB noise floor while far below the true memory, so the next call
   looks like growth faster than N² and raises (simulated, with 30 % of the
   readings 0 and memory 1 MiB + 1 KiB per point: 85 of 150 runs).
5. *Memory that is not counted, or counted wrongly.* On macOS and Windows,
   memory the integrand allocates and frees inside one call is missed. On a
   GPU, memory allocated outside PyTorch's allocator (custom CUDA code, CuPy)
   is missed. Every reading covers the whole process or device, so memory
   other threads allocate during a test call is counted as the integrand's.
6. *One input value.* Every point of a test call has the same value of t, so
   an integrand whose memory depends on t is measured at that value only. The
   fit is not updated during the integration (for example when the integrand
   fills a cache); only the free memory is read again for each batch.
7. *A single large reading.* A call that uses much more memory than its
   neighbors (memory allocated only at that size) stays in the fit and makes
   it cautious (simulated: 1 MiB at N = 2 among readings of 1 KiB per point
   gave a batch of 290,767 points where about 79 million fit).

**The fit**

8. *Growth faster than N² is caught late or not at all.* It is caught only
   once a reading exceeds three times what growth like N² allows from an
   earlier reading. Memory growing like N^p shows this across readings k times
   apart in N when k^(p − 2) > 3: N³ needs readings more than 3 times apart,
   N^2.2 more than 243 times apart. A large fixed cost delays this further
   (simulated, with 100 GB usable: 512 MiB + 10,000 bytes·N³ was never caught
   and gave a batch needing 1.48 times the usable memory). The call that shows
   the growth can already be too large.
9. *A small N² term can be missed.* The search stops when the memory per point
   changes by less than 1 % over two calls. An N² term too small to change it
   that much, and hidden from the fit by noise, is missed; at a batch k times
   larger than the last test call it adds k times more memory per point.
10. *The gradient is estimated, and the margin is fixed.* The test calls run
    the integrand forward only; with `take_gradient=True` the fitted memory is
    multiplied by 2.1, an estimate of what backpropagation needs. Batches use
    three quarters of the usable memory, and the remaining quarter is the only
    margin for errors of the fit. On a CPU a batch that does not fit can be
    killed by the operating system without a Python error.

**Time**

11. *Time growing faster than N² overruns the 60 s.* Before three calls, the
    next call is assumed to take at most 4 times as long when N doubles; from
    three calls on, the times are fitted up to N². Faster growth is predicted
    too low, and a running call cannot be interrupted (simulated: time growing
    like 1e-6·N³ s ended at 73 s, like 1e-3·N⁴ s at 170 s).
12. *Slow integrands raise.* With the 60 s limit, an integrand whose calls take
    more than 10 s each always raises: a fit needs three calls, and the third
    is predicted to take 4 times the second. A slow integrand with little
    memory per point, which needs many doublings of N to reach the noise floor,
    raises too (simulated: 5 s per call and 1 KiB per point needs 15 calls,
    75 s). `limit_benchmark_time=False` removes the limit, and then nothing
    bounds the time.

**Other**

13. *An out-of-memory error can leave too small a batch.* An out-of-memory
    error during the benchmark at N points caps the batch at 0.75·N, which can
    be below the nodes of one subinterval (gk21, with 21 nodes, running out of
    memory at N = 10 gives 7). `integrate(take_gradient=True)` then fails,
    because it evaluates whole subintervals.
14. *A wrong reason near 1e9 points.* When the calls double near 1e9 points,
    the end message says the next call would not fit in memory when the 1e9
    limit ended the search.
15. *Silent messages.* The benchmark's messages are logged at WARNING level, so
    they do not appear when the application sets logging above WARNING.

(ug-trade-speed-for-memory)=
## Memory efficiency and speed options for `integrate()`

PadaQuad's memory management can integrate any integrand as long as a single
integrand evaluation fits on your device. However, memory management can come
at the cost of speed when it is not needed. Specifically, improved memory
management speeds up integration with expensive integrand evaluations while
slowing down integration with cheap integrand evaluations. This section
describes the four arguments that control memory management and tests various
combinations of them on four integrands of increasing cost.

None of these options change which points are evaluated, the subintervals, or
the integral. Every combination measured below gave the same integral and
subintervals, bit for bit.

This section uses the words of the {doc}`../glossary` (subinterval, node) and
these:

Integrand values
: The outputs of the integrand `f` at the nodes.

Pass
: One iteration of the integration loop. In a pass, subintervals are
  evaluated, in calls of `f` of at most `max_batch` points; then each
  subinterval is accepted or split.

Solver device
: The device on which the solver evaluates `f` and computes the integral. The
  options matter when it is a GPU, so the text below says GPU.

The options apply only with `take_gradient=False`. With `take_gradient=True`
each call of `f` holds whole subintervals and its quadrature is calculated at
once.

### `trade_speed_for_memory`

When False, the quadrature is calculated once per pass, after all subintervals
are evaluated, which is fastest for cheap integrands. When True, it is
calculated early, so integrand values can be deleted or moved to
`result_device` to free GPU memory; `evaluation_buffer` sets how early (after
every call of `f` by default). For expensive integrands, True can be faster,
because it frees up memory that can be used for evaluation while paying a
small cost of calculating the quadrature from the integrand evaluations more
often. It applies only with `take_gradient=False`, and saves GPU memory only
when `result_device` is another device or `record_evaluations` is False.

| Value | Speed | GPU memory | Host memory |
|---|---|---|---|
| `False` (default) | Fastest for cheap integrands: one quadrature calculation per pass. | Holds all integrand values of the pass, once: each call of `f` writes its values into one array for the pass. | Receives the values of accepted subintervals after each pass. |
| `True` | Can be faster for expensive integrands, because the freed memory can be used for evaluating `f`. Slower for cheap integrands, because each extra quadrature calculation costs time: about 0.2 ms per call of `f` on an A100 without a buffer. | Holds the values of about one call of `f`, or up to `evaluation_buffer` bytes. | Receives the values of every evaluated subinterval right away; values of subintervals that are later split are deleted. The peak is the same as with False. |

The cost per call was measured with the script below at commit cc77e8b, on 75
calls of 46 points (gk21): 52 ms against 34 ms by default on an A100, and
23 ms against 16 ms on a CPU. With a 256 MB buffer it took the default's time.

### `evaluation_buffer`

Sets how many bytes of integrand values can wait on the GPU before the
quadrature is calculated and the values are moved. It balances speed and
memory between the two values of `trade_speed_for_memory` and is meant for
advanced users. Use it only with `trade_speed_for_memory=True`; otherwise it
raises `ValueError`. When measuring `max_batch` with `get_max_batch(f=...)`,
pass the same `evaluation_buffer`: it keeps twice that many bytes free, for
the waiting values and their copy when they are joined for the quadrature
calculation.

| Value | Speed | GPU memory | Host memory |
|---|---|---|---|
| `None` (default) | The quadrature is calculated after every call of `f`, so its cost is paid on every call. | Holds the values of about one call of `f`. | No effect. |
| a number of bytes | The quadrature is calculated once that many bytes of values are waiting, so its cost is paid less often. | Holds up to the buffer plus one call's values. If a pass never fills the buffer, the quadrature is calculated once per pass, as with `trade_speed_for_memory=False`. | No effect. |

### `result_device`

Where the integrand values of accepted subintervals and the per-subinterval
results are stored. The integral and its error estimate stay on the solver
device. It works independently of the other options.

| Value | Speed | GPU memory | Host memory |
|---|---|---|---|
| `"cpu"` (default) | Copying values to the host takes time that grows with their size; for integrands with many outputs it is most of the run time. | Holds only the values waiting for their quadrature calculation, and, until the end of the call, the small per-subinterval results of the accepted subintervals: nodes, widths, contributions, error estimates and error ratios (with `gk21` in float64, about 200 bytes per subinterval for one output). | Holds the values of every accepted subinterval. |
| `None` | Fastest, since nothing is copied. `None` means the solver device. | Holds the values of every accepted subinterval until the end, which gives the highest peak. | Nothing. |

The result `loss_fxn` receives for each batch has its `y` and
`tracked_variables` on `result_device` with `trade_speed_for_memory=True`, and
on the solver device without it; its other tensors stay on the solver device.

### `record_evaluations`

Whether integrand values are kept after their quadrature is calculated. The
quadrature alone decides which subintervals are accepted or split, so the
values are needed only if you use them (`result.y`) or to build the best mesh
for a later call.

| Value | Speed | GPU memory | Host memory |
|---|---|---|---|
| `True` (default) | Values are copied to `result_device`, which takes time when it is the host. | Set by the other options. | Holds the values of every accepted subinterval when `result_device="cpu"`. |
| `False` | Fastest with `result_device="cpu"`, since nothing is copied. `result.y` and `result.tracked_variables` are `None`. | Values are deleted right after their quadrature is calculated. With `trade_speed_for_memory=True` this lowers the peak even when `result_device=None`. | Holds no values. |

With False, the mesh returned for a later call is built with
`merge_rule="power_law"`, because the default `"predicted"` merge rule needs
the stored values. `"power_law"` merges fewer wide subintervals, so the later
call may start with more subintervals ({ref}`ug-reuse`).

### Errors and warnings

`integrate` raises `ValueError`, before any call of `f`, when
`trade_speed_for_memory=True` is given with `take_gradient=True`, and when
`evaluation_buffer` is given with `trade_speed_for_memory=False` or is not an
int of at least 1. It logs a warning on the first call of a series (no `state`
given; later calls that pass the state stay quiet) when:

- `trade_speed_for_memory=True` has no `evaluation_buffer`;
- `trade_speed_for_memory=True` and the integrand's output requires grad (from
  its parameters or from bounds that require grad): the autograd graph keeps
  what `f` saved for its backward pass on the GPU until the integral is
  differentiated, so moving the values may save little (logged at the end of
  the call, once the integral is known to require grad);
- `record_evaluations=False` with the solver's default `merge_rule="predicted"`.

(ug-pass-array-oom)=
### Out of memory with many outputs

With `trade_speed_for_memory=False`, the integrand values of a pass are held in
one array. For an integrand with many outputs this array can take many
gigabytes, all in one piece of GPU memory. PyTorch keeps the GPU memory it
frees in a cache of pieces of the sizes it used before. Near the GPU's limit,
the cache can hold enough free memory in total but no piece large enough, and
the allocation fails. PadaQuad then raises `torch.OutOfMemoryError` naming the
pass's size and two remedies:

- Pass `trade_speed_for_memory=True` with `record_evaluations=False` or
  `result_device="cpu"`, so the GPU holds about one call's values at a time.
- When PyTorch's message reports much memory "reserved by PyTorch but
  unallocated", set the environment variable
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA is first used
  (for example, before the script starts). PyTorch then grows one piece of
  memory as needed, instead of reserving pieces of fixed sizes that later
  requests may not fit ({external+torch:std:ref}`PyTorch's CUDA memory notes
  <cuda-memory-envvars>`). PadaQuad cannot set it for you: it applies to the
  whole process and must be set before CUDA starts.

On a 40 GB A100 (2026-11-25), an integrand with 262,144 outputs
(`SteppedField` in `examples/speed_test/trade_speed_for_memory.py`, side 512)
with `max_batch=500` and `record_evaluations=False` failed with the default
setting: it asked for 16.85 GiB while 18.82 GiB were reserved but unallocated.
With `expandable_segments:True` it completed in 8.3 to 9.1 s, against 8.0 to
8.2 s with `trade_speed_for_memory=True`. Near the limit, the default can
also be slower: at `max_batch=2000` it took 5.3 to 5.8 s against 3.7 to 4.1 s,
mostly in reserving GPU memory for arrays of changing size, and
`expandable_segments:True` did not close that gap.

### The gradient as extra outputs

To get the gradient of an integral without an autograd graph, let the
integrand return its gradient with respect to the parameters as extra outputs,
computed with `torch.func.grad` from detached parameters (or under
`torch.no_grad()`), and integrate with `take_gradient=False`. The integral of
those outputs is the gradient of the integral when the bounds do not depend on
the parameters; a bound b that does adds f(b) times the derivative of b (minus
the same at the lower bound), one call of `f`. `error_norm` chooses which
outputs decide a split ({ref}`ug-error-norm`), so the mesh can be refined for
the gradient's accuracy. {doc}`../auto_examples/gradient_as_output` shows both;
{ref}`ug-autodiff` describes the other ways to differentiate an integral.

### Measurements

`examples/speed_test/trade_speed_for_memory.py` integrates four integrands
over [0, 1]:

- a light integrand with 2 outputs;
- two networks, multilayer perceptrons with two tanh hidden layers, fixed
  random weights and gradients turned off, whose inputs are sin(ωt) and
  cos(ωt) for 64 frequencies ω from 1 to 20,000;
- an expensive field: a state of 64 numbers advanced by 2,000 sequential
  steps, then expanded into a field of 512 × 512 outputs, 1 MB per point, so
  a pass's integrand values take tens of GB.

The light integrand and the networks are integrated with every combination of
the four options and no `max_batch`: PadaQuad grows each call of `f` until `f`
runs out of memory. The field is integrated with `max_batch` 3,000, 4,000 and
5,550, each with `trade_speed_for_memory` False and True, and
`record_evaluations=False`. The script records the actual batch (below) and
the number of calls; for the field, the number of points evaluated; the GPU's peak
memory above what was allocated before the call, during the integration loop
and over the whole call (which also includes the final error check, the
state's mesh and the result); for the light integrand and the networks, the
host's peak resident memory, sampled every 5 ms; and the wall time. The tables
below are rendered from `docs/performance/memory_options.json`, which the
script writes.

In the tables of the light integrand and the networks, rows are ordered by
`trade_speed_for_memory`, then `evaluation_buffer`, `result_device` and
`record_evaluations`, with the default value first. Values that differ from
the default are in bold, so the first row is the default setting. In the
field's table, rows are ordered by `max_batch`.

The column "Actual Batch" is the number of points per call of `f` that
PadaQuad ended with. `max_batch` is where it starts; when a call of `f` runs
out of memory, PadaQuad discards that call's values, evaluates the same
points again at three quarters of the batch size, repeating until a call
fits, and keeps that smaller batch size for the rest of the integration.
Without `max_batch` it also lets each call grow to up to 10 times the
largest call so far, limited by the points waiting, until a call runs out of
memory, so for the light integrand and the networks the Actual Batch is the
largest call that ran. In the field's table, "ran out of memory" in the
Actual Batch column means that a call of `max_batch` points ran out of memory
and PadaQuad continued with a smaller batch; the points include the points of
the calls that were evaluated again.

What the tables show:

- **Light integrand and networks.** Their integrand values take at most a few
  hundred MB, so every setting makes the same calls of `f` and uses about the
  same GPU memory, and the defaults are appropriate. For the large network,
  copying the values to the host is most of the time with
  `result_device="cpu"`, which `record_evaluations=False` avoids.
- **Expensive field.** With `trade_speed_for_memory=False`, the pass's values
  stay on the GPU while `f` is called, so a call of 4,000 or 5,550 points runs
  out of memory. PadaQuad evaluates it again at three quarters of its size
  until it fits, which costs 4,000 and 9,712 extra points and leaves the calls at about
  3,000 points. With `True` every batch size fits, with a GPU peak of 16 to
  26 GB against about 32 GB, and the integration takes a third to almost half less
  time (3.5 to 4.0 s against 5.7 to 6.6 s). The time saved does not come
  from the larger calls, since `True` is fastest at 3,000 points; an earlier
  profile found most of it in the time the default spends reserving GPU
  memory for the pass's values ({ref}`ug-pass-array-oom`).

```{include} ../reference/generated/memory_options_table.md
```
