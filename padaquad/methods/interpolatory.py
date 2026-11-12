"""
Methods of the variable-sampling solver, whose weights depend on the node positions.

The uniform methods (Runge-Kutta, Gauss-Kronrod, Clenshaw-Curtis) have fixed
node fractions and fixed weights. The variable-sampling solver instead reuses
integrand values when it splits a subinterval, so its nodes can sit anywhere
inside a subinterval, and the weights are computed from where they are. Each
method here is an instance of ``VariableMethodClass``.

The registry ``INTERPOLATORY_METHODS`` holds:

- ``adaptive_heun``: the same object as the uniform ``adaptive_heun``. Its two
  nodes are the ends of the subinterval, so its weights (the trapezoidal rule)
  cannot depend on the node positions and one object serves both solvers.
- ``interpolatory3_variable``: 3 nodes 0, a, 1 (as fractions of the
  subinterval) with the interpolatory weights of Sanderse and Veldman 2019,
  Eq. (119), Sec. 5.2, with their :math:`c_2` equal to a
  :cite:p:`sanderse2019`. Degree 2 for general a; degree 3 at a = 1/2, where
  the weights are Simpson's rule. The error estimate subtracts the trapezoidal
  rule (degree 1); this pairing is padaquad's choice.

References
----------
- Sanderse and Veldman 2019, Eq. (119), Sec. 5.2, p. 21 of arXiv:1809.06114v2
  :cite:p:`sanderse2019`.
"""

from __future__ import annotations

import torch

from ._base import MethodClass, VariableMethodClass
from .runge_kutta import _ADAPTIVE_HEUN


def _interpolatory3_weights(c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sanderse-Veldman 3-node weights for normalized nodes [0, a, 1].

    Sanderse and Veldman 2019, J. Comput. Phys. 384, Eq. (119), Sec. 5.2
    (arXiv:1809.06114v2, p. 21), with their c2 = a. They are the
    interpolatory weights: the method integrates 1, x and x^2 exactly on
    [0, 1], i.e. sum_i w_i x_i^k = 1/(k+1) for k = 0, 1, 2. Derivation:

      1. k = 1 minus k = 2:  w_a (a - a^2) = 1/6  =>  w_a = 1/(6a(1-a))
      2. k = 1:              w_1 = 1/2 - a w_a    =>  w_1 = (2-3a)/(6(1-a))
      3. k = 0:              w_0 = 1 - w_a - w_1  =>  w_0 = 1/2 - 1/(6a)

    At a = 1/2 this is Simpson's rule [1/6, 2/3, 1/6] (exact through
    degree 3). The error weights are b minus the trapezoidal rule
    [1/2, 0, 1/2] on the end nodes (exact through degree 1; padaquad's
    choice), written out per node so no constant tensor is built (or copied
    to the GPU) per call.

    Parameters
    ----------
    c : torch.Tensor
        Normalized node positions; the middle node is a = c[:, 1, 0].
        Shape: [N, 3, T].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
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


# Interpolatory 3-node method (degree 2; 3 at a = 1/2), weights from Sanderse and
# Veldman 2019, Eq. (119); error estimate against the trapezoidal rule (degree 1,
# padaquad's choice).
_INTERPOLATORY_3 = VariableMethodClass(
    degree=2,
    error_exponent=3,
    c=torch.linspace(0.0, 1.0, 3, dtype=torch.float64),
    weight_function=_interpolatory3_weights,
)


# Registry of the variable-sampling solver, keyed by name: methods that are
# valid at the node positions that solver produces. Instances, cloned per
# solver by ``_get_method``. ``adaptive_heun`` is the uniform heun object
# (see the module docstring).
INTERPOLATORY_METHODS: dict[str, MethodClass] = {
    "adaptive_heun": _ADAPTIVE_HEUN,
    "interpolatory3_variable": _INTERPOLATORY_3,
}
