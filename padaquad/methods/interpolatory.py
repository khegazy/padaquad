"""
Quadrature rules whose weights depend on the node positions.

Unlike the fixed-table families (RK, Gauss-Kronrod, Clenshaw-Curtis),
these rules accept arbitrary node placements within each panel and
compute their weights from the actual positions with a weight function.
Each is an instance of the generic ``VariableMethodClass``. They are the
rules of the variable-sampling solver.

Variable-sampling registry (``INTERPOLATORY_METHODS``):
  adaptive_heun:             the uniform ``_ADAPTIVE_HEUN`` object itself.
      Its two nodes are the panel ends, so its weights cannot depend on
      positions and one object serves both sampling modes.
  interpolatory3_variable:   3 nodes, polynomial exactness 2 (degree 3
      at a=1/2 — Simpson's rule reduction; Sanderse-Veldman weights
      otherwise)
"""

from __future__ import annotations

import torch

from ._base import MethodClass, VariableMethodClass
from .runge_kutta import _ADAPTIVE_HEUN


def _interpolatory3_weights(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sanderse-Veldman 3-node weights for normalized nodes [0, a, 1].

    They are the interpolatory weights: the rule integrates 1, x and x^2
    exactly on [0, 1], i.e. sum_i w_i x_i^k = 1/(k+1) for k = 0, 1, 2.

      1. k = 1 minus k = 2:  w_a (a - a^2) = 1/6  =>  w_a = 1/(6a(1-a))
      2. k = 1:              w_1 = 1/2 - a w_a    =>  w_1 = (2-3a)/(6(1-a))
      3. k = 0:              w_0 = 1 - w_a - w_1  =>  w_0 = 1/2 - 1/(6a)

    At a = 1/2 this is Simpson's rule [1/6, 2/3, 1/6] (exact through
    degree 3). The error weights are b minus the trapezoidal rule
    [1/2, 0, 1/2] on the end nodes (exact through degree 1), written out per
    node so no constant tensor is built (or copied to the GPU) per call.

    Args:
        c: Normalized node positions; the middle node is a = c[:, 1, 0].
            Shape: [N, 3, T].

    Returns:
        Tuple (b, b_error), each of shape [N, 3], in the dtype and on the
        device of ``c``.
    """
    a = c[:, 1, 0]
    b0 = 0.5 - 1.0 / (6 * a)
    ba = 1.0 / (6 * a * (1 - a))
    b1 = (2.0 - 3 * a) / (6 * (1.0 - a))
    b = torch.stack([b0, ba, b1]).transpose(0, 1)
    b_error = torch.stack([b0 - 0.5, ba, b1 - 0.5]).transpose(0, 1)
    return b, b_error


# Interpolatory 3-node rule (exactness 2; 3 at a = 1/2), error estimate against
# the trapezoidal rule (exact through degree 1).
_INTERPOLATORY_3 = VariableMethodClass(
    order=3,
    error_exponent=3,
    c=torch.linspace(0.0, 1.0, 3, dtype=torch.float64),
    weight_function=_interpolatory3_weights,
)


# Registry of the variable-sampling solver, keyed by name: rules that are
# valid at the node positions that solver produces. Instances, cloned per
# solver by ``_get_method``. ``adaptive_heun`` is the uniform heun object
# (see the module docstring).
INTERPOLATORY_METHODS: dict[str, MethodClass] = {
    "adaptive_heun": _ADAPTIVE_HEUN,
    "interpolatory3_variable": _INTERPOLATORY_3,
}
