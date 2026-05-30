"""
Quadrature method definitions, organized by family.

Each family lives in its own submodule:

  - ``runge_kutta``: classical embedded RK pairs (adaptive_heun,
    fehlberg2, bosh3, dopri5).
  - ``gauss_kronrod``: Gauss-Kronrod pairs (gk15, gk21, gk31).
  - ``clenshaw_curtis``: Chebyshev-node nested rules (cc17, cc33, cc65).
  - ``interpolatory``: variable-node rules with weights computed from
    actual node positions (the variable-sampling family).

This ``__init__`` aggregates all family registries into the two
top-level dicts the solver consumes — ``UNIFORM_METHODS`` for
fixed-node methods and ``VARIABLE_METHODS`` for variable-node methods —
and exposes the ``_get_method`` factory used by the parallel solver.

Each method is specified by a Butcher tableau containing:

- **c** (nodes): Fractional positions within a step where the integrand
  is evaluated. ``c[0] = 0`` is the step start, ``c[-1] = 1`` is the
  step end.
- **b** (weights): Weights for combining evaluations into the integral
  estimate. The integral over one step is ``h * sum(b_i * f(t_i))``.
- **b_error**: Difference between the primary and embedded-method
  weights. The error estimate is ``h * sum(b_error_i * f(t_i))``.

There are two families of methods:

**Uniform methods** (``UNIFORM_METHODS``): The tableau ``c`` values are
fixed constants. Quadrature points within each step are always at the
same fractional positions regardless of step size. Simple and efficient.

**Variable methods** (``VARIABLE_METHODS``): The tableau ``b`` weights
are recomputed dynamically based on the actual positions of the
quadrature points (the ``c`` values). This allows quadrature points to
be at arbitrary positions within each step, which is useful when
refining the mesh by inserting new points between existing ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from padaquad.base import steps

from ._base import MethodClass, _Tableau
from .clenshaw_curtis import CC_METHODS
from .gauss_kronrod import GK_METHODS
from .interpolatory import (
    _VARIABLE_SECOND_ORDER,
    INTERPOLATORY_METHODS,
    _Interpolatory3Variable,
    _VariableSubclass,
)
from .runge_kutta import RK_METHODS

if TYPE_CHECKING:
    import torch

# Aggregate registries. Keys are user-facing method names; values are
# (uniform) MethodClass singletons or (variable) classes that produce
# instances per solver.
UNIFORM_METHODS = {**RK_METHODS, **GK_METHODS, **CC_METHODS}
VARIABLE_METHODS = {**INTERPOLATORY_METHODS}


def _get_method(
    sampling_type: steps,
    method_name: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> MethodClass | _VariableSubclass:
    """
    Retrieve and initialize an integration method by name and sampling type.

    Each solver gets its OWN copy of the method's tableau. The
    ``UNIFORM_METHODS`` dict holds canonical singletons at float64,
    and we ``clone()`` them here so that downstream dtype/device
    mutations stay isolated to one solver instance. Without this
    cloning, two solvers running concurrently — or sequentially with
    different dtypes — would corrupt each other's tableau values
    (most notably, lower-precision conversions of the singleton are
    irreversible).

    Variable methods are constructed fresh per-solver too (they
    already manage their own per-instance state).

    Args:
        sampling_type: Whether to use uniform or variable sampling.
        method_name: Name of the method (e.g. 'dopri5',
            'interpolatory3_variable').
        device: Device to place the method's tensors on.
        dtype: Floating-point dtype for the method's tensors.

    Returns:
        An initialized method object with tensors on the correct
        device/dtype, independent of any other solver instance.
    """
    if sampling_type == steps.ADAPTIVE_UNIFORM:
        # Clone the canonical float64 singleton so dtype/device
        # mutations below don't affect other solvers.
        method = UNIFORM_METHODS[method_name].clone()
    else:
        method = VARIABLE_METHODS[method_name](device)

    method.to_device(device)
    method.to_dtype(dtype)
    return method


__all__ = [
    "CC_METHODS",
    "GK_METHODS",
    "INTERPOLATORY_METHODS",
    "RK_METHODS",
    "UNIFORM_METHODS",
    "VARIABLE_METHODS",
    "_VARIABLE_SECOND_ORDER",
    "MethodClass",
    "_Interpolatory3Variable",
    "_Tableau",
    "_VariableSubclass",
    "_get_method",
]
