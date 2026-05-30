"""
Base abstractions for quadrature methods: Butcher tableau and method wrapper.

This module is part of the ``methods/`` subpackage that organizes the
library's quadrature method definitions by family. Each family lives in
its own module:

- ``runge_kutta.py``: classical embedded RK pairs (adaptive_heun,
  fehlberg2, bosh3, dopri5).
- ``gauss_kronrod.py``: Gauss-Kronrod pairs (gk15, gk21, gk31).
- ``clenshaw_curtis.py``: Clenshaw-Curtis methods (cc17, cc33, cc65).
- ``interpolatory.py``: variable-node rules where weights are computed
  from the actual node positions.

This file provides the data structures shared across families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class _Tableau:
    """
    Butcher tableau for a Runge-Kutta integration method.

    Stores the coefficients that define how quadrature points are placed
    within a step (c) and how integrand evaluations are weighted to produce
    the integral (b) and error estimate (b_error).

    Attributes:
        c: Node positions as fractions of step size. c[0]=0 is the step start,
            c[-1]=1 is the step end. Shape: [C] where C is the number of
            quadrature points per step.
        b: Weights for the primary (higher-order) integral estimate.
            Shape: [C] for uniform methods, [1, C] for some methods.
        b_error: Difference between primary and embedded method weights.
            The error estimate is h * sum(b_error * f(t)). Shape: same as b.
    """

    c: torch.Tensor
    b: torch.Tensor
    b_error: torch.Tensor

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert all tableau tensors to the specified dtype.

        Mutates in place. Use ``clone()`` first if you want to leave
        the original tensors at their pre-conversion precision (which
        is irreversibly lossy when going to lower-precision dtypes).
        """
        self.c = self.c.to(dtype)
        self.b = self.b.to(dtype)
        self.b_error = self.b_error.to(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move all tableau tensors to the specified device."""
        self.c = self.c.to(device)
        self.b = self.b.to(device)
        self.b_error = self.b_error.to(device)

    def clone(self) -> _Tableau:
        """Return a deep copy of this tableau.

        Used by ``_get_method`` so that each solver instance gets its
        own tableau and dtype/device mutations on one solver do not
        propagate to other solvers via shared singletons.
        """
        return _Tableau(
            c=self.c.clone(),
            b=self.b.clone(),
            b_error=self.b_error.clone(),
        )


class MethodClass:
    """
    A complete uniform-sampling RK method: order + tableau.

    Wraps a ``_Tableau`` with metadata about the method's convergence order.
    The order determines how quickly the error decreases as step size shrinks
    (error ~ O(h^order)).

    Attributes:
        order: Convergence order of the method.
        tableau: The Butcher tableau defining the method's coefficients.
    """

    order: int
    tableau: _Tableau

    def __init__(self, order: int, tableau: _Tableau) -> None:
        """Initialize with the method's convergence order and Butcher tableau."""
        self.order = order
        self.tableau = tableau

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tableau to the specified dtype."""
        self.tableau.to_dtype(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move tableau to the specified device."""
        self.tableau.to_device(device)

    def clone(self) -> MethodClass:
        """Return a copy with a deep-cloned tableau but the same order."""
        return MethodClass(order=self.order, tableau=self.tableau.clone())
