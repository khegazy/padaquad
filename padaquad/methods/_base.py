"""
Base abstractions for quadrature methods: the abstract rule and the fixed table.

This module is part of the ``methods/`` subpackage that organizes the
library's quadrature method definitions by family. Each family lives in
its own module:

- ``runge_kutta.py``: classical embedded RK pairs (adaptive_heun,
  fehlberg2, bosh3, dopri5).
- ``gauss_kronrod.py``: Gauss-Kronrod pairs (gk7, gk15, gk21, gk31).
- ``clenshaw_curtis.py``: Clenshaw-Curtis methods (cc5, cc9, cc17, cc33,
  cc65).
- ``interpolatory.py``: rules whose weights are computed from the actual
  node positions.

This file provides the abstract ``MethodClass`` every rule implements and the
two generic classes every registered rule is an instance of:
``UniformMethodClass`` (a fixed ``_Tableau``) and ``VariableMethodClass``
(node fractions and a weight function of the node positions).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Callable

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
            Shape: [C], like c.
        b_error: Difference between primary and embedded method weights.
            The error estimate is h * sum(b_error * f(t)). Shape: [C].
    """

    c: torch.Tensor
    b: torch.Tensor
    b_error: torch.Tensor

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert all tableau tensors to the specified dtype.

        Mutates in place. Clone the owning method first
        (``MethodClass.clone()``) if you want to leave the original tensors
        at their pre-conversion precision (which is irreversibly lossy when
        going to lower-precision dtypes).
        """
        self.c = self.c.to(dtype)
        self.b = self.b.to(dtype)
        self.b_error = self.b_error.to(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move all tableau tensors to the specified device."""
        self.c = self.c.to(device)
        self.b = self.b.to(device)
        self.b_error = self.b_error.to(device)


class MethodClass(ABC):
    """
    A quadrature rule on one panel: the interface every solver uses.

    The base class holds what every rule shares (its orders) and declares the
    operations a solver calls. Two generic subclasses implement them from data:
    ``UniformMethodClass`` from a fixed ``_Tableau``, ``VariableMethodClass``
    from node fractions and a weight function of the node positions. Every
    registered rule is an instance of one of them. Nothing family-specific (a
    tableau, a weight formula) lives on this class.

    Attributes:
        order: Convergence order of the method.
        error_exponent: k such that the per-step error estimate (``b_error``
            weights) scales as h^k for a step of width h: k = d + 2, where d is
            the polynomial exactness of the less accurate rule in the estimate.
            Sets the step-merging cutoff (``_prune_excess_mesh``).
    """

    def __init__(self, order: int, error_exponent: int) -> None:
        """Initialize with the method's convergence order and error exponent."""
        self.order = order
        self.error_exponent = error_exponent

    @abstractmethod
    def node_fractions(self) -> torch.Tensor:
        """
        Node positions of a fresh panel as fractions of its width.

        Ascending, from 0 (the panel's left barrier) to 1 (its right barrier).
        The solver places node i of a panel [left, right] at
        ``left + c_i * (right - left)``.

        Returns:
            Fractions c. Shape: [C], with C the number of nodes per panel.
        """

    @abstractmethod
    def weights(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Weights for the integral and for its error estimate on each panel.

        A panel of width h with integrand values y_i at its nodes contributes
        ``h * sum_i(b_i * y_i)`` to the integral and
        ``h * sum_i(b_error_i * y_i)`` to the error estimate.

        Args:
            nodes: Node positions of each panel. Shape: [N, C, T].

        Returns:
            Tuple (b, b_error), each of shape [N, C, 1] for every rule.
        """

    @abstractmethod
    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert the method's tensors to ``dtype`` in place."""

    @abstractmethod
    def to_device(self, device: str | torch.device) -> None:
        """Move the method's tensors to ``device`` in place."""

    @property
    def n_nodes(self) -> int:
        """Number of nodes per panel (C)."""
        return len(self.node_fractions())

    def _check_node_fractions(self) -> None:
        """
        Validate ``node_fractions()``; called at the end of each constructor.

        Raises:
            ValueError: Unless the fractions are 1-D with at least two entries,
                start at 0, end at 1, and never decrease (repeated values are
                allowed: dopri5 has two nodes at 1).
        """
        c = self.node_fractions()
        if not (
            c.ndim == 1
            and len(c) >= 2
            and float(c[0]) == 0.0
            and float(c[-1]) == 1.0
            and bool((c[1:] >= c[:-1]).all())
        ):
            raise ValueError(
                f"{type(self).__name__}: node fractions must be 1-D, start at 0, "
                f"end at 1 and never decrease; got {c.tolist()}"
            )

    def clone(self) -> Self:
        """
        Return an independent deep copy (no tensor shared with ``self``).

        Used by ``_get_method`` so that each solver gets its own copy of the
        registry's float64 rule: dtype or device changes on one solver never
        reach the registry or another solver (lowering the dtype is lossy).
        """
        return copy.deepcopy(self)


class UniformMethodClass(MethodClass):
    """
    A rule with fixed node fractions and fixed weights: one ``_Tableau``.

    Every panel uses the same fractions ``tableau.c`` and the same weights
    ``tableau.b`` / ``tableau.b_error``, whatever the panel's nodes. Used by
    the Runge-Kutta, Gauss-Kronrod and Clenshaw-Curtis families.

    Attributes:
        tableau: The Butcher tableau defining the method's coefficients.
    """

    def __init__(self, order: int, error_exponent: int, tableau: _Tableau) -> None:
        """
        Initialize with the method's orders and Butcher tableau.

        Raises:
            ValueError: If the tableau's node fractions are malformed (see
                ``MethodClass._check_node_fractions``).
        """
        super().__init__(order, error_exponent)
        self.tableau = tableau
        self._check_node_fractions()

    def node_fractions(self) -> torch.Tensor:
        """The tableau's node fractions ``c``. Shape: [C]."""
        return self.tableau.c

    def weights(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The tableau's weights, the same for every panel.

        Args:
            nodes: Node positions of each panel. Shape: [N, C, T]. Only N is
                used; the weights do not depend on the positions.

        Returns:
            Tuple (b, b_error), each ``[C]`` expanded (a view, no copy) to
            [N, C, 1]. The views share memory with the tableau: do not modify
            them in place.
        """
        n_panels = len(nodes)
        return (
            self.tableau.b.unsqueeze(-1).expand(n_panels, -1, -1),
            self.tableau.b_error.unsqueeze(-1).expand(n_panels, -1, -1),
        )

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert the tableau to ``dtype``."""
        self.tableau.to_dtype(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move the tableau to ``device``."""
        self.tableau.to_device(device)


class VariableMethodClass(MethodClass):
    """
    A rule whose weights are computed from each panel's node positions.

    A fresh panel is evaluated at the node fractions ``c``. The weights come
    from ``weight_function`` applied to the panel's nodes normalized to [0, 1],
    so the rule stays correct when nodes sit at other positions. Each rule is
    an instance with its own ``c`` and ``weight_function`` (see
    ``methods/interpolatory.py``), as each uniform rule is an instance with its
    own ``_Tableau``.

    ``weight_function`` must be a module-level function: ``clone()`` keeps it
    by reference, and solvers can then be pickled (a lambda or a local function
    raises ``PicklingError``). It is stored on the instance; assigning it to the
    class instead would make Python pass ``self`` to it.

    Attributes:
        c: Node fractions of a fresh panel, ascending from 0 to 1. Shape: [C].
        weight_function: Maps normalized node positions [N, C, T] to
            (b, b_error), each of shape [N, C], in the dtype and on the device
            of its input.
    """

    def __init__(
        self,
        order: int,
        error_exponent: int,
        c: torch.Tensor,
        weight_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """
        Initialize with the method's orders, node fractions and weight function.

        Raises:
            ValueError: If ``c`` is malformed (see
                ``MethodClass._check_node_fractions``).
        """
        super().__init__(order, error_exponent)
        self.c = c
        self.weight_function = weight_function
        self._check_node_fractions()

    def node_fractions(self) -> torch.Tensor:
        """The node fractions ``c`` of a fresh panel. Shape: [C]."""
        return self.c

    def weights(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Weights computed from each panel's actual node positions.

        Normalizes each panel's nodes to [0, 1] (shift so the first node is 0,
        divide by the panel width) and applies ``weight_function`` there. The
        weights take the dtype and device of ``nodes``.

        Args:
            nodes: Node positions of each panel. Shape: [N, C, T].

        Returns:
            Tuple (b, b_error), each of shape [N, C, 1].

        Raises:
            ValueError: If ``weight_function`` does not return two [N, C]
                tensors (a [1, C] result would broadcast silently in the
                weighted sum).
        """
        norm_dt = nodes - nodes[:, 0, None]
        norm_dt = norm_dt / norm_dt[:, -1, None]
        b, b_error = self.weight_function(norm_dt)
        expected = norm_dt.shape[:2]
        if b.shape != expected or b_error.shape != expected:
            raise ValueError(
                f"weight_function must return two tensors of shape {tuple(expected)}; "
                f"got {tuple(b.shape)} and {tuple(b_error.shape)}"
            )
        return b.unsqueeze(-1), b_error.unsqueeze(-1)

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert the node fractions to ``dtype``."""
        self.c = self.c.to(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move the node fractions to ``device``."""
        self.c = self.c.to(device)
