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
Faster growth that it does not catch (growth only slightly faster than N², N³
before two readings are more than three times apart in N, or growth hidden by a
large fixed cost) is underestimated, so pass `max_batch` for such an integrand.

On a CPU, memory is measured as the memory the operating system has given the
process. Memory the integrand takes from memory the process freed earlier
does not show up there. On Linux, padaquad first returns such memory with
glibc's `malloc_trim`, so the readings count it (not with another C
library, or with an allocator such as jemalloc loaded instead). On macOS and
Windows that is not possible: in a process that has already evaluated the
integrand, readings can be far too low. The batch size is then capped at the
largest call the benchmark made (it ran), which can be well below what fits
for an integrand whose memory per point stops changing early. The benchmark's
own calls are not bounded this way: in repeated calls on a Mac they needed up
to 7 times the usable memory, and an error can appear for an integrand whose
memory grows only like N². Pass `max_batch` there when the memory of a batch
matters.

The benchmark takes a few seconds and logs warnings, and it runs on every call
without `max_batch`. To run it once, integrate once without `max_batch`, read
`max_batch = solver.get_max_batch()`, and pass that value to later calls or
new solvers; or pass `same_f=True`. `get_max_batch()` returns the batch size
the solver would use now, without running the benchmark.

With `take_gradient=False`, if an evaluation runs out of memory, the batch is
shrunk to three quarters and retried; the smaller size stays in force until
the next `integrate` call. With `take_gradient=True` such an error is raised.
Out-of-memory errors from PyTorch's allocators and from CUDA libraries are
recognized; any other error from the integrand is raised unchanged.

## `conserve_memory`

`conserve_memory` matters only with `take_gradient=False`. By default each pass
of the loop evaluates every subinterval still waiting, in calls of at most
`max_batch` points, and holds all of their values at once. With
`conserve_memory=True` each pass evaluates one batch of `max_batch` points; a
subinterval whose nodes do not all fit is finished in the next pass. This
holds fewer values at once but needs more passes. With `take_gradient=True`
each pass already evaluates only the whole subintervals that fit in one batch.
