"""
Quadrature rules whose weights depend on the node positions.

Unlike the fixed-table families (RK, Gauss-Kronrod, Clenshaw-Curtis),
these rules accept arbitrary node placements within each panel and
compute their weights from the actual positions (``tableau_b(c)``).
They are the rules of the variable-sampling solver.

Variable-sampling registry (``INTERPOLATORY_METHODS``):
  adaptive_heun:             the uniform ``_ADAPTIVE_HEUN`` object itself.
      Its two nodes are the panel ends, so its weights cannot depend on
      positions and one object serves both sampling modes.
  interpolatory3_variable:   3 nodes, polynomial exactness 2 (degree 3
      at a=1/2 — Simpson's rule reduction; Sanderse-Veldman formula
      otherwise)
"""

from __future__ import annotations

import torch

from ._base import MethodClass
from .runge_kutta import _ADAPTIVE_HEUN


class _Interpolatory3Variable(MethodClass):
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
        error_exponent: Error-estimate exponent (3; see ``MethodClass``).
        c: Node fractions of a fresh panel, [0, 1/2, 1]. Shape: [3].
        b_delta: Reference weights for error estimation. Shape: [1, 3].
    """

    def __init__(self) -> None:
        """Initialize the 3rd-order variable method with Sanderse-Veldman weights."""
        # Estimate against the trapezoidal rule, exact through degree 1.
        super().__init__(order=3, error_exponent=3)
        # Fresh-panel node fractions [0, 1/2, 1], built in float64 and cast by
        # to_dtype. A float64 linspace cast to float32 equals a float32
        # linspace only for 2 and 3 nodes; recheck this if a rule with more
        # nodes is added.
        self.c = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
        # Reference weights (trapezoidal rule) used for error estimation
        self.b_delta = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float64)

    def node_fractions(self) -> torch.Tensor:
        """Evenly spaced fractions [0, 1/2, 1]. Shape: [3]."""
        return self.c

    def weights(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Weights computed from each panel's actual node positions.

        Normalizes each panel's nodes to [0, 1] (shift so the first node is 0,
        divide by the panel width) and evaluates ``tableau_b`` there.

        Args:
            nodes: Node positions of each panel. Shape: [N, C, T].

        Returns:
            Tuple (b, b_error), each of shape [N, C, 1].
        """
        norm_dt = nodes - nodes[:, 0, None]
        norm_dt = norm_dt / norm_dt[:, -1, None]
        b, b_error = self.tableau_b(norm_dt)
        return b.unsqueeze(-1), b_error.unsqueeze(-1)

    def to_device(self, device: str | torch.device) -> None:
        """Move tensors to the specified device."""
        self.c = self.c.to(device)
        self.b_delta = self.b_delta.to(device)

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tensors to the specified dtype."""
        self.c = self.c.to(dtype)
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


# Registry of the variable-sampling solver, keyed by name: rules that are
# valid at the node positions that solver produces. Instances, cloned per
# solver by ``_get_method``. ``adaptive_heun`` is the uniform heun object
# (see the module docstring).
INTERPOLATORY_METHODS: dict[str, MethodClass] = {
    "adaptive_heun": _ADAPTIVE_HEUN,
    "interpolatory3_variable": _Interpolatory3Variable(),
}
