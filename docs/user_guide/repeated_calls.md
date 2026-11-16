# Repeated calls

A solver built with `adaptive_quadrature` keeps the mesh and batch size of its
last call, so later calls on a similar integrand, for example in a training
loop, can start where the last one ended.

(ug-same-f)=
## Warm starts with `same_f`

Pass `same_f=True` when the integrand is the same as in the previous call, or
changes only slightly. The call then starts from the mesh and `max_batch`
stored by the last call on this solver that returned, in place of the caller's
`mesh` and `max_batch`, and skips the memory benchmark. The integration bounds
are set as follows:

- Bounds given in the call (`mesh_init`, `mesh_final`, or the ends of a `mesh`)
  are used; the stored mesh is cut to them, and their endpoints are added if
  missing.
- A bound not given is the stored mesh's end, so a call without bounds repeats
  the interval of the last call that returned, even if the constructor was
  given other bounds. With only `mesh_final` given, `mesh_init` is the stored
  mesh's first point, and the other way round. Bounds that are not increasing
  raise `ValueError`.
- If nothing is stored yet, the caller's values or the constructor's are used,
  so the flag can be passed from the first iteration.

With `same_f=False` (the default) the caller's values are used, but the stored
values are still updated. A `same_f=False` call on another interval, for
example a validation call in a training loop, therefore sets the interval that
the next `same_f=True` call without bounds integrates.

The stored mesh is the previous call's `mesh_optimal` ({ref}`ug-optimal-mesh`),
also available as `solver.mesh_previous`; the stored batch size is
`solver.previous_max_batch`. The stored batch size was computed for the
previous call's integrand and `take_gradient` setting. A value found with
`take_gradient=False` leaves no memory for backpropagation, so pass
`same_f=False` on a call that changes `take_gradient`.

```python
import torch
from padaquad import adaptive_quadrature

solver = adaptive_quadrature("uniform", "gk21", device="cpu", max_batch=10_000)
bounds = {
    "mesh_init": torch.tensor([0.0], dtype=torch.float64),
    "mesh_final": torch.tensor([1.0], dtype=torch.float64),
}
for scale in (1.0, 1.1, 1.2):
    result = solver.integrate(lambda t: torch.cos(scale * t), same_f=True, **bounds)
    print(round(float(result.integral), 6))
```

## Offsets and extra arguments

- `y0` is added to the returned integral: `result.integral` is
  `y0` + $\int f\,dt$, and `result.y0` echoes the value used.
- `f_args` is a tuple passed to the integrand after `t`: the solver calls
  `f(t, *f_args)`.

(ug-tracked-variables)=
## Tracked variables

The integrand may return a pair `(values, tracked_variables)`, where
`tracked_variables` is a tuple of tensors with one row per point. They are
evaluated at every node but not integrated, and come back in
`result.tracked_variables`, detached and aligned row by row with `result.nodes`
and `result.y`. In the per-batch result passed to `loss_fxn` they keep one row
per subinterval (shape `[N, C, ...]`).
