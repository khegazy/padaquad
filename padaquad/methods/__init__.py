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

The stored instances use float64 on the CPU and are never modified. A solver
uses a copy converted to its device and dtype (``MethodClass.to``), built once
per process and shared by every solver of that method, device and dtype. Read a method's ``degree``, ``order`` and ``n_nodes``, or measure its
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
# are float64 CPU ``MethodClass`` instances that ``_get_method`` converts (``to``).
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
    Return the named method of a sampling type, converted to a device and dtype.

    Both registries hold float64 CPU instances, which are never modified; the
    method returned is ``registry[method_name].to(device, dtype)``, a copy
    built once per process and shared by every solver of that method, device
    and dtype (see ``MethodClass.to``).

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
        The converted method; its tensors must not be modified in place.

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
    return registry[method_name].to(device, dtype)


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
