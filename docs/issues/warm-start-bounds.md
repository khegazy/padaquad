# `same_f=True` after a call with only `mesh=` integrates the default interval

Status: open. Found 2026-11-15 by the code review of `merge_rule="predicted"`
(PR #69); it is identical on `main` at 2c0c955, so that PR did not cause it. Written so that another agent can pick it up without the original session.

## Summary

A call that gives its interval only through `mesh=` is integrated correctly. A
following call with `same_f=True` and no bounds integrates the solver's default
interval, [0, 1] unless the constructor set `mesh_init` / `mesh_final`, and no
warning is given:

```python
import torch
from padaquad import adaptive_quadrature

s = adaptive_quadrature(
    "uniform", "gk21", device="cpu", dtype=torch.float64, max_batch=10**5
)
m = torch.linspace(2, 3, 5, dtype=torch.float64)[:, None]
print(s.integrate(torch.exp, mesh=m).integral.item())  # 12.6965 = e^3 - e^2
print(s.integrate(torch.exp, same_f=True).integral.item())  # 1.7183  = e - 1
```

(Output measured on 2026-11-15 on the tree of PR #69.) The warning for a call that
did not converge advises calling again with `same_f=True`, which leads straight
into this.

## Cause

- `_setup_integral_bounds` (`padaquad/quadrature/base.py:2868`) takes the bounds
  from `mesh` when one is given, and otherwise from the constructor's
  `init_mesh_init` / `init_mesh_final` (`padaquad/base.py:210–219`), which default
  to [0] and [1].
- With `same_f=True`, `_setup_initial_mesh` (`base.py:2925`) cuts the stored mesh
  (`mesh_previous`, set at `base.py:1192`) to those bounds and pads it with them.
  A stored mesh on [2, 3] has no point in [0, 1], so a fresh mesh on [0, 1] is
  used.

Each rule is documented on its own; the two together give an integral over an
interval the caller never asked for.

## Options (not decided)

1. Warn when the stored mesh and the bounds of a `same_f=True` call do not
   overlap, or when bounds that were not given discard stored points.
2. Take the bounds of a `same_f=True` call without bounds from the stored mesh's
   ends. This needs rules for one bound given, for bounds set in the constructor
   on purpose (the constructor does not record whether its bounds were given or
   defaulted), and for T > 1.
3. Leave the behaviour and document it next to `same_f`.

Which bounds a warm call should use is a behaviour decision for the maintainer.
