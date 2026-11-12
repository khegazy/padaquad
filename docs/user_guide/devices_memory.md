# Devices and memory

## Device and precision

`device` selects where the integrand is evaluated: `"cpu"`, `"cuda"`,
`"cuda:N"` or a `torch.device`; `None` picks the GPU if one is available. The
solver never changes PyTorch's global default device. `result_device` (default
`"cpu"`) is where the per-subinterval arrays of the result go (nodes,
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
much memory one evaluation of the integrand takes, by calling it on 1, 10,
100, … points. The batch size is then computed at every evaluation from the
memory free at that moment: the usable memory is the free memory minus
`(1 - total_mem_usage)` times the total memory (default `total_mem_usage` =
0.9). With `take_gradient=True` the measured cost is multiplied by 2.1, a
margin for backpropagation.

The benchmark takes a few seconds and logs warnings, and it runs on every call
without `max_batch`. To run it once, integrate once without `max_batch`, read
`max_batch = solver.get_max_batch()`, and pass that value to later calls or
new solvers; or pass `same_f=True`. `get_max_batch()` returns the batch size
the solver would use now, without running the benchmark.

If an evaluation runs out of memory, the batch is shrunk to three quarters and
retried; the smaller size stays in force until the next `integrate` call.
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
