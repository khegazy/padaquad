# Devices and memory

## Device and precision

`device` selects where the integrand is evaluated: `"cpu"`, `"cuda"`,
`"cuda:N"` or a `torch.device`; `None` picks the GPU if one is available. The
solver never changes PyTorch's global default device. `result_device` is where the per-subinterval arrays of the result go (nodes,
integrand values, contributions, errors), to keep them off the GPU; `None`
keeps them on the solver's device. The integral, its error estimate and
`mesh_optimal` stay on the solver's device.

The solver computes in the dtype of the bounds or mesh given to `integrate`,
float64 or float32. A float32 tensor such as `torch.tensor([1.0])` therefore
makes the integration run in float32.

(ug-memory)=
## Batch size and the memory benchmark

padaquad evaluates the nodes of many subintervals in one call of the
integrand. The number of points per call is the batch size, `max_batch`.

When no `max_batch` is given (to `integrate` or to the constructor) and no
stored value is used ({ref}`ug-same-f`), a **memory benchmark** measures how
the memory of one call of the integrand grows with the number of points N. It
calls the integrand on 1 point, then on more, and fits the memory of the calls
as a + b N + d N², so an integrand whose memory grows like N² (for example, one
that computes a gradient per point inside itself) is measured too. Only
readings of at least 10 KB on a GPU, or 4 MiB on a CPU, count (smaller CPU
readings are mostly noise), and the benchmark keeps going until three readings
do and the memory they show has grown by at least that much. Each next call is
at most ten times larger (twice as large once calls take more than 0.1 s on a
GPU or on Linux, whose readings count all the memory a call uses), and no
larger than the fit predicts to fit in three quarters of the usable memory and
to take at most 1 s; that 1 s limit is stretched only as far as needed to get
those readings. No call is started that is predicted to end past 60 s for the
whole benchmark; `limit_benchmark_time=False` (a constructor option, also
accepted by `padaquad.integrate`) removes this limit for an integrand so slow
that the benchmark needs longer. The
batch size is then computed at every evaluation from the memory free at that
moment: the largest batch whose fitted
memory fits in three quarters of the usable memory, which is the free memory
minus `(1 - total_mem_usage)` times the total memory. With
`take_gradient=True` the fitted memory is multiplied by a safety factor, a
margin for the memory that backpropagation needs.

The benchmark raises an error that lists what it measured and asks for
`max_batch` when it cannot measure the memory: when the readings do not
allow a fit within 60 s (the error names `limit_benchmark_time`), when no
reading reaches the noise floor, or when a call uses more than three times the
memory that growth like N² allows from an earlier reading that reached the
noise floor (memory growing faster than N², which the fit cannot follow).
Faster growth that it does not catch is underestimated
({ref}`ug-memory-limitations`).

On a CPU, memory is measured as the memory the operating system has given the
process. Memory the integrand takes from memory the process freed earlier
does not show up there. On Linux, padaquad first returns such memory with
glibc's `malloc_trim`, so the readings count it (not with another C
library, or with an allocator such as jemalloc loaded instead). On macOS and
Windows that is not possible: in a process that has already evaluated the
integrand, readings can be far too low, so the batch size is capped at the
largest call the benchmark made (it ran). See {ref}`ug-memory-limitations` for
what this costs and what it leaves unbounded; pass `max_batch` there when the
memory of a batch matters.

The benchmark takes a few seconds and logs warnings, and it runs on every call
without `max_batch`. To run it once, integrate once without `max_batch`, read
`max_batch = solver.get_max_batch()`, and pass that value to later calls or
new solvers; or pass `same_f=True`. `get_max_batch()` returns the batch size
the solver would use now, without running the benchmark. Both reuse the
benchmark's result, so they save its time but keep its limitations
({ref}`ug-memory-limitations`).

With `take_gradient=False`, if an evaluation runs out of memory, the batch is
shrunk to three quarters and retried; the smaller size stays in force until
the next `integrate` call. With `take_gradient=True` such an error is raised.
Out-of-memory errors from PyTorch's allocators and from CUDA libraries are
recognized; any other error from the integrand is raised unchanged.

(ug-memory-limitations)=
### Limitations of the memory benchmark

The benchmark is meant for the common case: an integrand whose memory grows
linearly with the number of points N, or at most like N², measured where every
allocation can be seen (a GPU, or Linux with glibc). It cannot be right for
every integrand on every machine. It can choose a batch far smaller than what
fits, which leaves the hardware underused, and for some integrands a batch
larger than what fits. **Passing your own `max_batch`, chosen for the
integrand and the machine, avoids all of this**: the benchmark does not run,
and the batch is the value given. Reusing the benchmark's result, through
`solver.get_max_batch()` or `same_f=True`, only skips the time it takes; the
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
