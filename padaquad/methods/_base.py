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
from typing import TYPE_CHECKING, NamedTuple, Self

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


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


class DegreeMeasurement(NamedTuple):
    """
    Result of ``MethodClass.measure_degree``.

    Attributes:
        degree: Measured degree of the method (weights ``b``): the highest
            polynomial degree it integrates exactly. -1 if it does not even
            integrate a constant exactly.
        lower_degree: Measured degree of the lower-degree method inside the
            error estimate (weights ``b - b_error``).
        errors: Error of the method on the Legendre polynomial P_k(2t - 1)
            over [0, 1], for k = 0, 1, ..., 2C. Shape: [2C + 1], float64.
        lower_errors: The same for the lower-degree method. Shape: [2C + 1].
    """

    degree: int
    lower_degree: int
    errors: torch.Tensor
    lower_errors: torch.Tensor


class MethodClass(ABC):
    """
    A quadrature rule on one panel: the interface every solver uses.

    The base class holds what every rule shares (its degree and error exponent)
    and declares the operations a solver calls. Two generic subclasses implement
    them from data: ``UniformMethodClass`` from a fixed ``_Tableau``,
    ``VariableMethodClass`` from node fractions and a weight function of the node
    positions. Every registered rule is an instance of one of them. Nothing
    family-specific (a tableau, a weight formula) lives on this class.

    Attributes:
        degree: Highest polynomial degree the rule integrates exactly (it is not
            exact for every polynomial of degree ``degree + 1``). Gonnet 2012,
            ACM Comput. Surv. 44(4), Sec. 2. For a rule whose weights depend on
            the node positions, the degree at general positions.
        error_exponent: k such that the per-step error estimate (``b_error``
            weights) scales as h^k for a step of width h: k = d + 2, where d is
            the degree of the less accurate rule in the estimate. Sets the
            step-merging cutoff (``_prune_excess_mesh``).
    """

    def __init__(self, degree: int, error_exponent: int) -> None:
        """
        Initialize with the method's degree and error exponent.

        Raises:
            ValueError: If ``degree`` is not an int >= 0, or ``error_exponent``
                is not an int >= 2 (k = d + 2 with d >= 0).
        """
        for name, value, minimum in (
            ("degree", degree, 0),
            ("error_exponent", error_exponent, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"{type(self).__name__}: {name} must be an int >= {minimum}, "
                    f"got {value!r}"
                )
        self.degree = degree
        self.error_exponent = error_exponent

    @property
    def order(self) -> int:
        """
        Order of accuracy, ``degree + 1``.

        The composite rule's error on a smooth integrand falls as h^order for
        steps of width h (Gonnet 2012, ACM Comput. Surv. 44(4), Sec. 2: a rule
        of degree n has order n + 1).
        """
        return self.degree + 1

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

    def _check_node_fractions(self, c: torch.Tensor | None = None) -> None:
        """
        Validate node fractions; called at the end of each constructor.

        Args:
            c: Fractions to check. Default: ``node_fractions()``.

        Raises:
            ValueError: Unless the fractions are 1-D with at least two entries,
                start at 0, end at 1, and never decrease (repeated values are
                allowed: dopri5 has two nodes at 1).
        """
        if c is None:
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

    def measure_degree(
        self, nodes: torch.Tensor | None = None, tol: float = 1e-12
    ) -> DegreeMeasurement:
        """
        Measure the polynomial degree of the method and of its error estimate.

        Integrates the Legendre polynomials P_k(2t - 1), k = 0, 1, ..., 2C, over
        one subinterval [0, 1] (C = number of nodes) with the method's weights
        ``b``, and with the weights ``b - b_error`` of the lower-degree method
        inside its error estimate. The exact values are 1 for k = 0 and 0 for
        k >= 1. The measured degree is the last k before the first error larger
        than ``tol``. The calculation runs in float64 on the CPU.

        Legendre polynomials are used instead of the powers t^k because a high
        power is almost entirely made of lower-degree polynomials, which the
        method integrates exactly: its error on t^(d+1) is its error on
        P_(d+1) times (k!)^2 2^k / (2k)! with k = d + 1, about 4e-14 for
        gk31, which float64 cannot resolve. The Legendre errors one degree past
        each method's degree are all at least 4e-6.

        The search stops at k = 2C because no method with C nodes is exact at
        degree 2C (derivation): the polynomial (t - c_1)^2 ... (t - c_C)^2 has
        degree 2C and is zero at every node, so the method gives 0 for it, but
        its integral is positive.

        Args:
            nodes: Node fractions at which to measure a method whose weights
                depend on the node positions: 1-D with C entries, from 0 to 1,
                never decreasing. Default: ``node_fractions()``.
            tol: Errors at most this large count as exact. The default suits
                the float64 tables of ``UNIFORM_METHODS`` and
                ``VARIABLE_METHODS``. A solver's copy of a method lowered to
                float32 is measured as float32 values.

        Returns:
            DegreeMeasurement with the two measured degrees and the errors.

        Raises:
            ValueError: If ``nodes`` is malformed or does not have C entries;
                if ``nodes`` differs from the node fractions while the weights
                do not depend on the node positions (the measurement would
                apply the weights at nodes they were not made for); or if no
                error is larger than ``tol`` up to degree 2C (the weights cannot
                be right).
        """
        f64 = torch.float64
        default = self.node_fractions().detach().to("cpu", f64)
        c = default if nodes is None else nodes.detach().to("cpu", f64)
        self._check_node_fractions(c)
        if c.shape != default.shape:
            raise ValueError(
                f"nodes must have {len(default)} entries (one per node), got "
                f"shape {tuple(c.shape)}"
            )
        b, b_error = (w[0, :, 0].to("cpu", f64) for w in self.weights(c[None, :, None]))
        if nodes is not None and not torch.equal(c, default):
            b_default = self.weights(default[None, :, None])[0][0, :, 0].to("cpu", f64)
            if torch.equal(b, b_default):
                raise ValueError(
                    f"{type(self).__name__}: the weights do not depend on the node "
                    "positions, so the method can only be measured at its own "
                    "node fractions"
                )

        # Legendre polynomials P_0..P_2C at x = 2c - 1 by the three-term
        # recurrence P_(k+1) = ((2k + 1) x P_k - k P_(k-1)) / (k + 1)
        # (NIST DLMF Eq. 18.9.1 with Table 18.9.1).
        x = 2 * c - 1
        max_degree = 2 * len(c)
        legendre = [torch.ones_like(x), x]
        for k in range(1, max_degree):
            legendre.append(
                ((2 * k + 1) * x * legendre[k] - k * legendre[k - 1]) / (k + 1)
            )
        p = torch.stack(legendre[: max_degree + 1])  # [2C + 1, C]
        exact = torch.zeros(max_degree + 1, dtype=f64)
        exact[0] = 1.0  # integral of P_k(2t - 1) over [0, 1] is 1 for k = 0, else 0

        measured = []
        for weights in (b, b - b_error):
            errors = p @ weights - exact
            failing = torch.nonzero(errors.abs() > tol)
            if len(failing) == 0:
                raise ValueError(
                    f"{type(self).__name__}: no error above tol={tol} up to degree "
                    f"{max_degree}; a method with {len(c)} nodes cannot be exact there"
                )
            measured.append((int(failing[0]) - 1, errors))
        (degree, errors), (lower_degree, lower_errors) = measured
        return DegreeMeasurement(degree, lower_degree, errors, lower_errors)

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

    def __init__(self, degree: int, error_exponent: int, tableau: _Tableau) -> None:
        """
        Initialize with the method's degree, error exponent and Butcher tableau.

        Raises:
            ValueError: If ``degree`` or ``error_exponent`` is invalid (see
                ``MethodClass.__init__``), or the tableau's node fractions are
                malformed (see ``MethodClass._check_node_fractions``).
        """
        super().__init__(degree, error_exponent)
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
        degree: int,
        error_exponent: int,
        c: torch.Tensor,
        weight_function: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """
        Initialize with the method's degree, error exponent, node fractions and
        weight function.

        Raises:
            ValueError: If ``degree`` or ``error_exponent`` is invalid (see
                ``MethodClass.__init__``), or ``c`` is malformed (see
                ``MethodClass._check_node_fractions``).
        """
        super().__init__(degree, error_exponent)
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
