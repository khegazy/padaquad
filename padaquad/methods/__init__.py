"""
Quadrature method definitions, organized by family.

Each family lives in its own submodule:

  - ``runge_kutta``: classical embedded RK pairs (adaptive_heun,
    fehlberg2, bosh3, dopri5).
  - ``gauss_kronrod``: Gauss-Kronrod pairs (gk7, gk15, gk21, gk31).
  - ``clenshaw_curtis``: Chebyshev-node nested rules (cc5, cc9, cc17,
    cc33, cc65).
  - ``interpolatory``: rules with weights computed from the actual node
    positions (the variable-sampling registry).

This ``__init__`` aggregates all family registries into the two
top-level dicts the solver consumes — ``UNIFORM_METHODS`` for
fixed-node methods and ``VARIABLE_METHODS`` for variable-node methods —
and exposes the ``_get_method`` factory used by the parallel solver.
Every registry value is a ``MethodClass`` instance (see ``_base.py``).

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

from ._base import MethodClass, UniformMethodClass, _Tableau
from .clenshaw_curtis import CC_METHODS
from .gauss_kronrod import GK_METHODS
from .interpolatory import INTERPOLATORY_METHODS, _Interpolatory3Variable
from .runge_kutta import RK_METHODS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

# Aggregate registries. Keys are user-facing method names; values are
# float64 CPU ``MethodClass`` instances that ``_get_method`` clones per solver.
UNIFORM_METHODS: dict[str, UniformMethodClass] = {
    **RK_METHODS,
    **GK_METHODS,
    **CC_METHODS,
}
VARIABLE_METHODS: dict[str, MethodClass] = {**INTERPOLATORY_METHODS}


def _get_method(
    sampling_type: steps,
    method_name: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> MethodClass:
    """
    Retrieve and initialize an integration method by name and sampling type.

    Each solver gets its OWN copy of the method. Both registries hold
    canonical instances at float64, and we ``clone()`` them here so that
    downstream dtype/device mutations stay isolated to one solver instance.
    Without this cloning, two solvers running concurrently — or sequentially
    with different dtypes — would corrupt each other's values (most notably,
    lower-precision conversions of the canonical instance are irreversible).

    Args:
        sampling_type: Whether to use uniform or variable sampling.
        method_name: Name of the method (e.g. 'dopri5',
            'interpolatory3_variable').
        device: Device to place the method's tensors on.
        dtype: Floating-point dtype for the method's tensors.

    Returns:
        An initialized method object with tensors on the correct
        device/dtype, independent of any other solver instance.

    Raises:
        ValueError: If ``method_name`` is not in the registry of
            ``sampling_type``.
    """
    registry: Mapping[str, MethodClass] = (
        UNIFORM_METHODS if sampling_type == steps.ADAPTIVE_UNIFORM else VARIABLE_METHODS
    )
    if method_name not in registry:
        raise ValueError(
            f"Unknown method {method_name!r} for {sampling_type.name} sampling; "
            f"choose one of {list(registry)}."
        )
    method = registry[method_name].clone()
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
    "MethodClass",
    "UniformMethodClass",
    "_Interpolatory3Variable",
    "_Tableau",
    "_get_method",
]
