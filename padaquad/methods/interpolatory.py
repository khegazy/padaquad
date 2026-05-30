"""
Variable-node interpolatory quadrature rules.

Unlike the uniform-sampling families (RK, Gauss-Kronrod, Clenshaw-Curtis)
which use fixed node positions, these rules accept arbitrary node
placements within each panel and recompute their weights from the actual
positions via a ``tableau_b(c)`` callable.

This flexibility is what enables the parallel solver's variable-sampling
mode to reuse evaluations across mesh refinement: when a step is split,
existing evaluation points end up at non-standard fractional positions
within the new sub-steps, and these methods just recompute the weights
that integrate them correctly.

Methods provided:
  adaptive_heun (variable):  2 nodes, polynomial exactness 1
  interpolatory3_variable:   3 nodes, polynomial exactness 2 (degree 3
      at a=1/2 — Simpson's rule reduction; Sanderse-Veldman formula
      otherwise)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .runge_kutta import _ADAPTIVE_HEUN


class _VariableSubclass(ABC):
    """
    Base class for variable-sampling RK methods.

    Unlike uniform methods where b weights are fixed, variable methods
    compute b weights from the actual quadrature point positions via
    ``tableau_b(c)``.

    Attributes:
        device: The device tensors are stored on.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        """Initialize with an optional device for tensor placement."""
        self.device = device

    @abstractmethod
    def to_device(self, device: str | torch.device) -> None:
        """Move method tensors to the specified device."""

    @abstractmethod
    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert method tensors to the specified dtype."""


class _VARIABLE_SECOND_ORDER(_VariableSubclass):
    """
    Variable-sampling 2nd-order method (adaptive Heun variant).

    For 2nd-order methods, the b weights happen to be independent of the
    quadrature point positions, so this simply returns the fixed adaptive
    Heun tableau b values regardless of the input c.

    Attributes:
        order: Convergence order (2).
        n_tableau_c: Number of quadrature points per step (2).
    """

    order = 2
    n_tableau_c = 2

    def __init__(self, device: str | torch.device | None = None) -> None:
        """Initialize the 2nd-order variable method using adaptive Heun's tableau."""
        super().__init__(device)
        self.device = device
        # Clone instead of aliasing the canonical RK singleton: ``to_dtype``
        # below mutates the tableau in place, so an alias would corrupt the
        # global ``_ADAPTIVE_HEUN`` (the same hazard Phase 4 fixed for the
        # uniform side via MethodClass.clone()).
        self.tableau = _ADAPTIVE_HEUN.tableau.clone()

    def to_device(self, device: str | torch.device) -> None:
        """Move tableau tensors to the specified device."""
        self.tableau.to_device(device)

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tableau tensors to the specified dtype."""
        self.tableau.to_dtype(dtype)

    def tableau_b(self, _c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return b weights for the given quadrature positions.

        For 2nd-order, the weights are constant regardless of c positions.

        Args:
            c: Normalized quadrature positions within each step.
                Shape: [N, C, T] where values are in [0, 1].

        Returns:
            Tuple of (b, b_error) tensors. Shape: [1, C].
        """
        b = self.tableau.b
        b_error = self.tableau.b_error
        return b, b_error


class _Interpolatory3Variable(_VariableSubclass):
    """
    Variable-sampling 3rd-order method (generic Sanderse-Veldman).

    Computes tableau b weights dynamically based on the actual positions
    of the quadrature points within each step. This is the key difference
    from uniform methods: the same set of evaluations at different positions
    can be correctly weighted.

    The method uses the generic 3rd-order formula from Sanderse and Veldman,
    where the b weights are functions of the middle node position 'a':

        b0(a) = 1/2 - 1/(6a)
        ba(a) = 1/(6a(1-a))
        b1(a) = (2-3a)/(6(1-a))

    The error is estimated by comparing against a reference set of weights
    (b_delta = [0.5, 0.0, 0.5], the trapezoidal rule).

    Attributes:
        order: Convergence order (3).
        n_tableau_c: Number of quadrature points per step (3).
        b_delta: Reference weights for error estimation. Shape: [1, 3].
    """

    order = 3
    n_tableau_c = 3

    def __init__(self, device: str | torch.device | None = None) -> None:
        """Initialize the 3rd-order variable method with Sanderse-Veldman weights."""
        super().__init__(device)
        self.device = device
        # Reference weights (trapezoidal rule) used for error estimation
        self.b_delta = torch.tensor(
            [[0.5, 0.0, 0.5]], dtype=torch.float64, device=self.device
        )

    def to_device(self, device: str | torch.device) -> None:
        """Move tensors to the specified device."""
        self.b_delta = self.b_delta.to(device)

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tensors to the specified dtype."""
        self.b_delta = self.b_delta.to(dtype)

    def _b0(self, a: torch.Tensor) -> torch.Tensor:
        """Weight for the left endpoint (c=0). Shape follows input a."""
        return 0.5 - 1.0 / (6 * a)

    def _ba(self, a: torch.Tensor) -> torch.Tensor:
        """Weight for the interior point (c=a). Shape follows input a."""
        return 1.0 / (6 * a * (1 - a))

    def _b1(self, a: torch.Tensor) -> torch.Tensor:
        """Weight for the right endpoint (c=1). Shape follows input a."""
        return (2.0 - 3 * a) / (6 * (1.0 - a))

    def tableau_b(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute b weights dynamically from actual quadrature positions.

        Uses the generic 3rd-order Sanderse-Veldman formula. The middle
        node position 'a' (where 0 < a < 1) determines all three weights.

        Butcher tableau (degree P-1 embedded pair):

            c |      b
            ------------------
            0 | 1/2 - 1/(6a)
            a | 1/(6a(1-a))
            1 | (2-3a)/(6(1-a))

        Args:
            c: Normalized quadrature positions within each step. The middle
                node position a = c[:,1,0] determines the weights.
                Shape: [N, C, T] where C=3, values in [0, 1].

        Returns:
            Tuple of (b, b_error) where:
                - b: Primary method weights. Shape: [N, C].
                - b_error: Difference from reference weights for error
                  estimation. Shape: [N, C].
        """
        # Extract the middle node position 'a' for each step
        a = c[:, 1, 0]
        # Compute the three b weights as functions of 'a'
        b = torch.stack([self._b0(a), self._ba(a), self._b1(a)]).transpose(0, 1)
        # Error is the difference between the computed weights and the
        # reference trapezoidal rule weights
        b_error = b - self.b_delta

        return b, b_error


# Registry of variable-sampling methods, keyed by name.
# These are classes (not instances) since they need per-device initialization.
INTERPOLATORY_METHODS = {
    "adaptive_heun": _VARIABLE_SECOND_ORDER,
    "interpolatory3_variable": _Interpolatory3Variable,
}
