# Multi-coordinate integration variable (T > 1) is untested end to end and blocked in `integrate()`

## Summary

padaquad is built so that the integration variable `t` can have more than one
coordinate (`T > 1`, for example a point $(x, y)$ moving along a path). However, no
test runs `integrate()` with `T > 1`, and today every such call fails at the first
step. With that failure bypassed, the integration runs, but what it computes for
vector-valued integrands has never been defined or documented.

## Terms

- **`t`**: the integration variable passed to the integrand `f(t)`, shaped `[N, T]` for a batch of `N` points.
- **`T`**: the number of coordinates in one value of `t`. An ordinary integral $\int_a^b f(t)\,dt$ has `T = 1`.
- **`D`**: the number of outputs of `f`, which returns `[N, D]`.
- **`mesh_init` / `mesh_final`**: the start and end points $a$ and $b$, each shaped `[T]`.

## Evidence that T > 1 is intended

- The shape convention throughout the code is `nodes: [N, C, T]` and mesh barriers `[M, T]` (`CLAUDE.md`: "T: dimensionality of time (usually 1, but supports multi-D)").
- The ordering checks in `padaquad/quadrature/base.py` are written for any `T`.
  - The rule for any `T` is "consecutive barriers are distinct".
  - The monotonic check runs only when `T == 1` (around lines 778–787, with the comment "multi-dimensional path" at line 780, and 1963–1967).
- The random initial mesh places sub-barriers "on the line for any T" (around line 1949): the straight segment from $a$ to $b$.
- Unit tests exercise `T = 2` directly, without going through `integrate()`:
  - `tests/test_adaptive_steps.py` (vector-mesh splitting, around lines 244–330 and 547)
  - `tests/test_sorted_insert.py:24`
  - `tests/test_record_and_sort.py:158` and `:263`

## The blocker

`_setup_integral_bounds` (`padaquad/quadrature/base.py`, lines 1834 and 1843) checks
`assert mesh_init < mesh_final`. For `T = 1` that comparison is one True/False value. For `T = 2`
it is two, and Python's `assert` raises:

```python
import torch
from padaquad import adaptive_quadrature

s = adaptive_quadrature("uniform", method="gk21", device="cpu", dtype=torch.float64)
f = lambda t: t[:, :1] * t[:, 1:2]          # f(x, y) = x * y, D = 1

# Path 1: limits only
s.integrate(f, mesh_init=torch.tensor([0., 0.]), mesh_final=torch.tensor([1., 2.]), max_batch=10000)
# RuntimeError: Boolean value of Tensor with more than one value is ambiguous   (base.py:1843)

# Path 2: explicit mesh
mesh = torch.stack([torch.linspace(0, 1, 5), torch.linspace(0, 2, 5)], -1).double()
s.integrate(f, mesh=mesh, max_batch=10000)
# RuntimeError: Boolean value of Tensor with more than one value is ambiguous   (base.py:1834)
```

(torch 2.8.0, padaquad 1.3.0.)

A second blocker sat behind this one: the integrand check call built its point with
`torch.tensor([[mesh_init]])`, which fails for `T > 1`. That is fixed in 1.3.0
(commit "fix: build the integrand test call as [1, T] from mesh_init").

## What T = 2 computes once the assert is bypassed

These results come from replacing `_setup_integral_bounds` on the solver instance (in
memory) so that it returns the bounds unchecked. The path is $(x, y) = (s, 2s)$ for
$s \in [0, 1]$, with `max_batch=10000`:

| Integrand | Result | Meaning |
|---|---|---|
| `D = 1`, $f = xy$ (uniform `gk21`; variable `adaptive_heun`; `take_gradient` True and False) | `[0.6667, 1.3333]` | $\left(\int f\,dx,\ \int f\,dy\right) = \left(\tfrac{2}{3}, \tfrac{4}{3}\right)$, one integral per coordinate. Correct for that reading. |
| `D = 2 = T`, $F = (x, y)$ | `[0.5, 2.0]` | $\left(\int F_1\,dx,\ \int F_2\,dy\right)$: the components are paired one-to-one, **not** summed into the line integral $\int F\cdot dr = 2.5$. |
| `D = 3` | `RuntimeError: The size of tensor a (2) must match the size of tensor b (3)` at `padaquad/runge_kutta.py:82` | `_RK_integral` multiplies the panel width `h` (`[N, T]`) by the weighted sum of `f` (`[N, D]`). The two shapes only combine when `D` is 1 or `T`. |

Analytic check of row 1: along the path, $f = 2s^2$, $dx = ds$ and $dy = 2\,ds$, so
$\int f\,dx = \int_0^1 2s^2\,ds = 2/3$ and $\int f\,dy = \int_0^1 4s^2\,ds = 4/3$.

## Other known limits

- `_RK_integral`'s docstring says `RK_steps` has shape `[N, D]`. For `T > 1` with `D = 1` it is `[N, T]`.
- `_get_mesh_indices` keys barriers by their coordinate tuple, so a path that passes through the same point twice would collide.
- The memory-benchmark path (no `max_batch`) has not been run with `T > 1`.

## What is needed

1. **Decide what the integral means for `T > 1`.** Options include per-coordinate integrals $\int f\,dx_k$, the scalar line integral $\int f\,|dr|$, or the vector line integral $\int F\cdot dr$. Document the choice, including which `D` are allowed.
2. **Replace the scalar `mesh_init < mesh_final` asserts with the existing any-T pattern:** require distinct endpoints for any `T`, and require ordering only when `T == 1`.
3. **Add end-to-end `T = 2` tests** through `integrate()`, with analytic answers, for both sampling types and both `take_gradient` modes, plus one test without `max_batch` to exercise the memory benchmark.
4. **Fix the `_RK_integral` docstring shapes.**
