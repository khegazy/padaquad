# Devices and memory

## Device and precision

`device` selects where the integrand is evaluated: `"cpu"`, `"cuda"`,
`"cuda:N"` or a `torch.device`; `None` picks the GPU if one is available. The
solver never changes PyTorch's global default device. `result_device` is where the per-subinterval arrays of the result go (nodes,
integrand values, contributions, errors), to keep them off the GPU; `None`
keeps them on the solver's device. The integral, its error estimate and the
state's mesh stay on the solver's device.

The solver computes in the dtype of the bounds or mesh given to `integrate`,
float64 or float32. A float32 tensor such as `torch.tensor([1.0])` therefore
makes the integration run in float32.

(ug-memory)=
## Batch size and the memory benchmark

padaquad evaluates the nodes of many subintervals in one call of the
integrand. The number of points per call is the batch size, `max_batch`.

**How `integrate` chooses it.** A `max_batch` given to the call (or to the
constructor, when no `state` is given) is used as it is and never grows. A
`state` (the state an earlier call returned, {ref}`ug-reuse`) gives the batch
size to start from. With neither, the integrand is called, forward only, on 10,
100, ... copies of the lower bound, up to the order of the first pass (for the
default starting mesh of gk21, 253 points: calls of 10 and 100), and the batch
size starts at the largest that ran. These calls read no memory; they only
check that the integrand runs.

**How it grows.** While the batch size has not converged, a call of the
integrand that has more points waiting than the batch size tries 10 times as
many (or all that are waiting). The first pass of a call therefore runs whole
when it is at most 10 times the batch size. When a call runs out of memory, it
is evaluated again at three quarters of its size, repeatedly, until it fits;
that size becomes the batch size, and the batch size has then converged
(`state.max_batch_converged`): it never grows again, in this call or in any
call given its state. Reaching the limit this way costs a few failed calls of
the integrand once per state (measured on an A100: about 5 to 6 s for networks
filling the GPU).

**With `take_gradient=True`.** padaquad backpropagates every batch right after
evaluating it. On a GPU, before each backward pass it compares the memory
still usable (keeping `1 - total_mem_usage` of the total free) with what the
backward pass is predicted to need: the batch's forward peak memory, times the
ratio of backward to forward peak memory the previous batch measured (1 for
the first batch). If it does not fit, the batch is dropped before any gradient
is written and evaluated again, smaller. If a backward pass still runs out of
memory, padaquad puts back the gradients the call added, lowers the batch
size, and restarts the integral from its starting mesh, with a warning. It
finds those gradients by walking each batch's autograd graph. A custom
autograd function (for example activation checkpointing with
`use_reentrant=True`) can hide parameters from that walk; pass
`parameters=model.parameters()` (or the optimizer) in that case, else
padaquad raises `BackwardOutOfMemoryError`, whose `state` is the state to call
again with after resetting the gradients. It raises it also when a parameter
has a post-accumulate-grad hook, or when `torch.distributed` runs more than one
process.

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
  (NVIDIA's on Windows) slows down instead of failing. padaquad is meant for
  GPUs; on a CPU, pass `max_batch` when memory matters.
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
    be below the nodes of one subinterval (gk21, with 23 nodes, running out of
    memory at N = 10 gives 7). `integrate(take_gradient=True)` then fails,
    because it evaluates whole subintervals.
14. *A wrong reason near 1e9 points.* When the calls double near 1e9 points,
    the end message says the next call would not fit in memory when the 1e9
    limit ended the search.
15. *Silent messages.* The benchmark's messages are logged at WARNING level, so
    they do not appear when the application sets logging above WARNING.

## `conserve_memory`

`conserve_memory` matters only with `take_gradient=False`. By default each pass
of the loop evaluates every subinterval still waiting, in calls of at most
`max_batch` points, and holds all of their values at once. With
`conserve_memory=True` each pass evaluates one batch of `max_batch` points; a
subinterval whose nodes do not all fit is finished in the next pass. This
holds fewer values at once but needs more passes. With `take_gradient=True`
each pass already evaluates only the whole subintervals that fit in one batch.
