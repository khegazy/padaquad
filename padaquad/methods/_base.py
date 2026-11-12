"""
The method interface and the two generic method classes.

Each family of methods lives in its own module of ``padaquad.methods``:

- ``runge_kutta.py``: embedded Runge-Kutta pairs (adaptive_heun, fehlberg2,
  bosh3, dopri5).
- ``gauss_kronrod.py``: Gauss-Kronrod pairs (gk7, gk15, gk21, gk31).
- ``clenshaw_curtis.py``: Clenshaw-Curtis methods (cc5, cc9, cc17, cc33,
  cc65).
- ``interpolatory.py``: methods whose weights are computed from the actual
  node positions (interpolatory3_variable; adaptive_heun is shared with the
  uniform registry).

This module defines the abstract ``MethodClass`` that every method implements
and the two generic classes every registered method is an instance of:
``UniformMethodClass`` (fixed node fractions and weights, a ``_Tableau``) and
``VariableMethodClass`` (node fractions and a weight function of the node
positions).
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
    Fixed node fractions and weights of a uniform method.

    Named after the Butcher tableau of a Runge-Kutta method, whose ``c`` and
    ``b`` rows these are for the Runge-Kutta methods; the Gauss-Kronrod and
    Clenshaw-Curtis methods use the same three arrays.

    Attributes
    ----------
    c : torch.Tensor
        Node fractions: c[0] = 0 is the start of the subinterval, c[-1] = 1
        its end. Shape: [C], with C the number of nodes per subinterval.
    b : torch.Tensor
        Weights of the method; the result on a subinterval of width h is
        h * sum(b * f(t)). Shape: [C].
    b_error : torch.Tensor
        b minus the weights of the embedded lower-degree method; the error
        estimate is h * sum(b_error * f(t)). Shape: [C].
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
    Result of ``MethodClass.measure_degree``: the two measured degrees and the errors.

    Attributes
    ----------
    degree : int
        Measured degree of the method (weights ``b``): the highest
        polynomial degree it integrates exactly. -1 if it does not even
        integrate a constant exactly.
    lower_degree : int
        Measured degree of the embedded lower-degree method used in the
        error estimate (weights ``b - b_error``).
    errors : torch.Tensor
        Error of the method on the Legendre polynomial P_k(2t - 1)
        over [0, 1], for k = 0, 1, ..., 2C. Shape: [2C + 1], float64.
    lower_errors : torch.Tensor
        The same for the lower-degree method. Shape: [2C + 1].
    """

    degree: int
    lower_degree: int
    errors: torch.Tensor
    lower_errors: torch.Tensor


class MethodClass(ABC):
    """
    A quadrature method on one subinterval: the interface every solver uses.

    The base class holds what every method shares (its degree and error
    exponent) and declares the operations a solver calls. Two generic
    subclasses implement them from data: ``UniformMethodClass`` from fixed node
    fractions and weights, ``VariableMethodClass`` from node fractions and a
    weight function of the node positions. Every registered method is an
    instance of one of them.

    Attributes
    ----------
    degree : int
        Highest polynomial degree the method integrates exactly (it is not
        exact for every polynomial of degree ``degree + 1``), as defined in
        Gonnet 2012, Sec. 1 :cite:p:`gonnet2012`. For weights that depend on
        the node positions, the degree at general positions.
    error_exponent : int
        The power k such that the error estimate on a subinterval of width h
        shrinks like :math:`h^k`: k = d + 2, with d the degree of the embedded
        method. Sets the merge cutoff of the optimal mesh
        (:ref:`ug-optimal-mesh`).
    """

    def __init__(self, degree: int, error_exponent: int) -> None:
        """
        Initialize with the method's degree and error exponent.

        Raises
        ------
        ValueError
            If ``degree`` is not an int >= 0, or ``error_exponent``
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
        Order of the method, ``degree + 1``.

        On a smooth integrand the total error of the method applied on
        subintervals of width h falls like :math:`h^{\\text{order}}`; a method of
        degree n has order n + 1 (Gonnet 2012, Sec. 1 :cite:p:`gonnet2012`).
        """
        return self.degree + 1

    @abstractmethod
    def node_fractions(self) -> torch.Tensor:
        """
        Node positions of a new subinterval as fractions of its width.

        Non-decreasing, from 0 (the subinterval's left mesh point) to 1 (its
        right mesh point). The solver places node i of a subinterval
        [left, right] at ``left + c_i * (right - left)``.

        Returns
        -------
        torch.Tensor
            Fractions c. Shape: [C], with C the number of nodes per
            subinterval.
        """

    @abstractmethod
    def weights(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Weights for the integral and for its error estimate on each subinterval.

        A subinterval of width h with integrand values y_i at its nodes
        contributes ``h * sum_i(b_i * y_i)`` to the integral and
        ``h * sum_i(b_error_i * y_i)`` to the error estimate.

        Parameters
        ----------
        nodes : torch.Tensor
            Node positions of each subinterval. Shape: [N, C, T].

        Returns
        -------
        tuple of torch.Tensor
            ``(b, b_error)``, each of shape [N, C, 1].
        """

    @abstractmethod
    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert the method's tensors to ``dtype`` in place."""

    @abstractmethod
    def to_device(self, device: str | torch.device) -> None:
        """Move the method's tensors to ``device`` in place."""

    @property
    def n_nodes(self) -> int:
        """Number of nodes per subinterval (C)."""
        return len(self.node_fractions())

    def _check_node_fractions(self, c: torch.Tensor | None = None) -> None:
        """
        Validate node fractions; called at the end of each constructor.

        Parameters
        ----------
        c : torch.Tensor | None
            Fractions to check. Default: ``node_fractions()``.

        Raises
        ------
        ValueError
            Unless the fractions are 1-D with at least two entries,
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
        Measure the polynomial degree of the method and of its embedded method.

        Integrates the Legendre polynomials :math:`P_k(2t - 1)`,
        k = 0, 1, ..., 2C, over the subinterval [0, 1] (C is the number of
        nodes) with the method's weights ``b``, and with the weights
        ``b - b_error`` of the embedded lower-degree method. The exact values
        are 1 for k = 0 and 0 for k >= 1. The measured degree is the last k
        before the first error larger than ``tol``. The calculation runs in
        float64 on the CPU.

        Legendre polynomials are used instead of the powers :math:`t^k`
        because a high power is almost entirely made of lower-degree
        polynomials, which the method integrates exactly. With k = d + 1, the
        error on :math:`t^k` is the error on :math:`P_k(2t - 1)` times
        :math:`(k!)^2 / (2k)!`, the inverse of the leading coefficient of
        :math:`P_k(2t - 1)` (derivation); for gk31 (k = 48) the factor is
        about 1.6e-28 and the error about 9e-32, which float64 cannot resolve. The Legendre errors one degree
        past each registered method's degree are all at least 4e-6.

        The search stops at k = 2C because no method with C nodes is exact at
        degree 2C (derivation): the polynomial
        :math:`(t - c_1)^2 \\cdots (t - c_C)^2` has degree 2C and is zero at
        every node, so the method gives 0 for it, but its integral is positive.

        Parameters
        ----------
        nodes : torch.Tensor or None
            Node fractions at which to measure a method whose weights depend
            on the node positions: 1-D, C entries, non-decreasing from 0 to 1.
            Default: ``node_fractions()``.
        tol : float
            Errors at most this large count as exact. The default suits the
            float64 methods of ``UNIFORM_METHODS`` and ``VARIABLE_METHODS``;
            float32 weights are measured as their float32 values.

        Returns
        -------
        DegreeMeasurement
            The two measured degrees and the errors.

        Raises
        ------
        ValueError
            If ``nodes`` is malformed or does not have C entries;
            if ``nodes`` differs from the node fractions while the weights
            do not depend on the node positions (the measurement would
            apply the weights at nodes they were not made for); or if no
            error is larger than ``tol`` up to degree 2C (the weights cannot
            be right).

        Examples
        --------
        >>> import torch
        >>> from padaquad import UNIFORM_METHODS, VARIABLE_METHODS
        >>> measured = UNIFORM_METHODS["gk21"].measure_degree()
        >>> print(measured.degree, measured.lower_degree)
        31 19

        A method whose weights depend on the node positions can be measured at
        other node fractions:

        >>> method = VARIABLE_METHODS["interpolatory3_variable"]
        >>> print(method.measure_degree().degree)  # nodes 0, 1/2, 1: Simpson
        3
        >>> nodes = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
        >>> print(method.measure_degree(nodes).degree)
        2
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

        Each solver works on its own copy of the registry's float64 method, so
        dtype or device changes on one solver never reach the registry or
        another solver (lowering the dtype loses precision).
        """
        return copy.deepcopy(self)


class UniformMethodClass(MethodClass):
    """
    A method with fixed node fractions and fixed weights.

    Every subinterval uses the same fractions ``tableau.c`` and the same
    weights ``tableau.b`` and ``tableau.b_error``. Used by the Runge-Kutta,
    Gauss-Kronrod and Clenshaw-Curtis families.

    Attributes
    ----------
    tableau : _Tableau
        The node fractions ``c``, weights ``b`` and error weights
        ``b_error``.
    """

    def __init__(self, degree: int, error_exponent: int, tableau: _Tableau) -> None:
        """
        Initialize with the method's degree, error exponent and tableau.

        Raises
        ------
        ValueError
            If ``degree`` is not an int >= 0, ``error_exponent`` is not an
            int >= 2, or the node fractions are not a 1-D tensor of at least
            two non-decreasing values from 0 to 1.
        """
        super().__init__(degree, error_exponent)
        self.tableau = tableau
        self._check_node_fractions()

    def node_fractions(self) -> torch.Tensor:
        """The tableau's node fractions ``c``. Shape: [C]."""
        return self.tableau.c

    def weights(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The tableau's weights, the same for every subinterval.

        Parameters
        ----------
        nodes : torch.Tensor
            Node positions of each subinterval. Shape: [N, C, T]. Only N is
            used; the weights do not depend on the positions.

        Returns
        -------
        tuple of torch.Tensor
            ``(b, b_error)``, each ``[C]`` expanded (a view, no copy) to
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
    A method whose weights are computed from each subinterval's node positions.

    A new subinterval is evaluated at the node fractions ``c``. The weights come
    from ``weight_function`` applied to the subinterval's nodes rescaled to
    [0, 1], so the method stays correct when the nodes sit elsewhere. Each
    method is an instance with its own ``c`` and ``weight_function`` (see
    ``methods/interpolatory.py``), as each uniform method is an instance with
    its own tableau.

    ``weight_function`` must be a module-level function: ``clone()`` keeps it
    by reference, and solvers can then be pickled (a lambda or a local function
    cannot be pickled). It is stored on the instance; assigning it to the class
    instead would make Python pass ``self`` to it.

    Attributes
    ----------
    c : torch.Tensor
        Node fractions of a new subinterval, non-decreasing from 0 to 1.
        Shape: [C].
    weight_function : Callable
        Maps node positions rescaled to [0, 1], shape [N, C, T], to
        ``(b, b_error)``, each of shape [N, C], in the dtype and on the device
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
        Initialize with the degree, error exponent, node fractions and weight function.

        Raises
        ------
        ValueError
            If ``degree`` is not an int >= 0, ``error_exponent`` is not an
            int >= 2, or ``c`` is not a 1-D tensor of at least two
            non-decreasing values from 0 to 1.
        """
        super().__init__(degree, error_exponent)
        self.c = c
        self.weight_function = weight_function
        self._check_node_fractions()

    def node_fractions(self) -> torch.Tensor:
        """The node fractions ``c`` of a new subinterval. Shape: [C]."""
        return self.c

    def weights(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Weights computed from each subinterval's actual node positions.

        Rescales each subinterval's nodes to [0, 1] (shift so the first node is
        0, divide by the subinterval's width) and applies ``weight_function``
        there. The weights take the dtype and device of ``nodes``.

        Parameters
        ----------
        nodes : torch.Tensor
            Node positions of each subinterval. Shape: [N, C, T].

        Returns
        -------
        tuple of torch.Tensor
            ``(b, b_error)``, each of shape [N, C, 1].

        Raises
        ------
        ValueError
            If ``weight_function`` does not return two [N, C]
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
