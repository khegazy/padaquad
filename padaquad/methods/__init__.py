"""
Quadrature methods and the two registries the solvers choose them from.

``UNIFORM_METHODS`` maps each method name of the uniform-sampling solver to a
``UniformMethodClass`` instance with fixed node fractions and weights:

- Runge-Kutta pairs: ``adaptive_heun``, ``fehlberg2``, ``bosh3``, ``dopri5``
  (module ``runge_kutta``).
- Gauss-Kronrod pairs: ``gk7``, ``gk15``, ``gk21`` (the default), ``gk31``
  (module ``gauss_kronrod``).
- Clenshaw-Curtis methods: ``cc5``, ``cc9``, ``cc17``, ``cc33``, ``cc65``
  (module ``clenshaw_curtis``).

``VARIABLE_METHODS`` maps each method name of the variable-sampling solver to
a ``MethodClass`` instance whose weights may depend on the node positions:
``adaptive_heun`` (the same object as in ``UNIFORM_METHODS``) and
``interpolatory3_variable`` (module ``interpolatory``).

The stored instances use float64 on the CPU. Each solver works on its own copy,
converted to the solver's dtype and device, so the registries are never
modified. Read a method's ``degree``, ``order`` and ``n_nodes``, or measure its
degree with ``measure_degree()``.

Each method is defined by three arrays of length C (the number of nodes per
subinterval):

- ``c``, the node fractions: node i of a subinterval [l, r] is at
  ``l + c_i * (r - l)``, with ``c[0] = 0`` and ``c[-1] = 1``.
- ``b``, the weights: the method's result on a subinterval of width h is
  ``h * sum_i(b_i * f(t_i))``.
- ``b_error``, the error weights: ``b`` minus the weights of a lower-degree
  embedded method on the same nodes. The error estimate is
  ``h * sum_i(b_error_i * f(t_i))``.

For a variable method, ``b`` and ``b_error`` are computed for each subinterval
from its node positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from padaquad.base import steps

from ._base import (
    DegreeMeasurement,
    MethodClass,
    UniformMethodClass,
    VariableMethodClass,
    _Tableau,
)
from .clenshaw_curtis import CC_METHODS
from .gauss_kronrod import GK_METHODS
from .interpolatory import INTERPOLATORY_METHODS
from .runge_kutta import RK_METHODS

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

# Registries (described in the module docstring). Keys are method names; values
# are float64 CPU ``MethodClass`` instances that ``_get_method`` clones per solver.
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

    Each solver gets its own copy of the method: both registries hold float64
    instances, which are cloned here so that dtype and device changes stay
    with one solver. Without the copy, two solvers would change each other's
    weights, and converting the shared instance to float32 would lose
    precision for every later solver.

    Parameters
    ----------
    sampling_type : steps
        Whether to use uniform or variable sampling.
    method_name : str
        Name of the method (e.g. 'dopri5',
        'interpolatory3_variable').
    device : str | torch.device
        Device to place the method's tensors on.
    dtype : torch.dtype
        Floating-point dtype for the method's tensors.

    Returns
    -------
    MethodClass
        An initialized method object with tensors on the correct
        device/dtype, independent of any other solver instance.

    Raises
    ------
    ValueError
        If ``method_name`` is not in the registry of
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
    "DegreeMeasurement",
    "MethodClass",
    "UniformMethodClass",
    "VariableMethodClass",
    "_Tableau",
    "_get_method",
]
