"""
Clenshaw-Curtis quadrature using Chebyshev nodes.

CC_n integrates polynomials of degree n exactly using n+1 Chebyshev
nodes x_k = cos(k*pi/n) on (-1, 1). Two key properties:

  1. CC nodes are NESTED. The CC_(n/2) grid is exactly the every-other
     subset of the CC_n grid. This makes CC the natural method for
     adaptive doubling refinement.
  2. CC nodes naturally include the panel endpoints (x_0 = 1, x_n = -1
     map to c = 1 and c = 0). No endpoint padding is needed (unlike
     Gauss-Kronrod).

We provide cc17 (n=16), cc33 (n=32), cc65 (n=64). The error indicator
pairs CC_n with CC_(n/2): b_error = CC_n_weights - CC_(n/2)_extended,
where the extension places 0 at CC_n-only nodes.

Reference: Trefethen 2008 SIAM Review (Is Gauss quadrature better than
Clenshaw-Curtis?); Waldvogel 2006 BIT (Fast computation of CC weights).
"""

from __future__ import annotations

import math

import torch

from ._base import MethodClass, _Tableau


def _clenshaw_curtis_weights_unit(n: int) -> list[float]:
    """Compute Clenshaw-Curtis weights on (-1, 1) for n+1 Chebyshev nodes.

    Nodes are x_k = cos(k * pi / n) for k = 0..n; the rule integrates
    polynomials of degree at most n exactly.

    Returns weights of length n+1, indexed as the nodes (i.e. weight at
    cos(k*pi/n) is at position k).

    Computed with Python floats and ``math.cos`` (the C library cosine), in
    this loop order. Do not vectorize the sum over j: reordering the additions
    changes the weights in the last bits, and ``torch.cos`` differs from the C
    library cosine by one unit in the last place on some arguments.
    """
    if n == 0:
        return [2.0]
    # Trefethen 2008 / Waldvogel 2006 cosine-series formula.
    c_aux = [0.0] * (n + 1)
    c_aux[0] = 2.0
    for k in range(2, n + 1, 2):
        c_aux[k] = 2.0 / (1.0 - k * k)
    w = [0.0] * (n + 1)
    pi_n = math.pi / n
    for k in range(n + 1):
        s = c_aux[0] / 2  # j=0 term, halved
        for j in range(1, n):
            s += c_aux[j] * math.cos(j * k * pi_n)
        s += c_aux[n] * math.cos(k * math.pi) / 2  # j=n term, halved
        w[k] = (2.0 / n) * s
    # Halve the endpoint weights (boundary correction)
    w[0] /= 2.0
    w[-1] /= 2.0
    return w


def _build_clenshaw_curtis_tableau(n: int) -> _Tableau:
    """Construct a Clenshaw-Curtis _Tableau on [0, 1] for n+1 nodes.

    The embedded error indicator pairs CC_n with CC_(n/2). The nodes of
    CC_(n/2) are exactly every other node of CC_n (CC nesting), so the
    extension to the CC_n grid places the CC_(n/2) weight at shared
    positions and 0 at the CC_n-only positions.
    """
    assert n >= 2, "CC error pair requires n >= 2"
    assert n % 2 == 0, "CC error pair requires even n for the doubling embed"

    # CC weights on (-1, 1), indexed by k=0..n where x_k = cos(k*pi/n).
    w_cc_high = _clenshaw_curtis_weights_unit(n)  # [n+1]
    w_cc_low = _clenshaw_curtis_weights_unit(n // 2)  # [n//2 + 1]

    # CC_(n/2) nodes coincide with CC_n nodes at indices 0, 2, 4, ..., n.
    # Build the extended CC_(n/2) weights on the CC_n grid.
    w_cc_low_extended = [0.0] * (n + 1)
    w_cc_low_extended[0::2] = w_cc_low

    # Map (-1, 1) -> [0, 1]. The natural CC indexing has x_0 = +1,
    # x_n = -1 (descending), so to get c[0] = 0 and c[-1] = 1 we
    # reverse the index: c[k] = (cos((n-k)*pi/n) + 1) / 2.
    nodes_unit = [(math.cos((n - k) * math.pi / n) + 1.0) / 2.0 for k in range(n + 1)]
    # Force exact endpoint values to avoid sub-eps drift.
    nodes_unit[0] = 0.0
    nodes_unit[-1] = 1.0

    # Reverse the weight arrays to match the reversed node order.
    b_high = torch.tensor(w_cc_high[::-1], dtype=torch.float64) / 2.0
    b_low_ext = torch.tensor(w_cc_low_extended[::-1], dtype=torch.float64) / 2.0
    return _Tableau(
        c=torch.tensor(nodes_unit, dtype=torch.float64),
        b=b_high,
        b_error=b_high - b_low_ext,
    )


# CC_4 (5 nodes): polynomial exactness 4, error vs CC_2 (3 nodes, exactness 2)
_CLENSHAW_CURTIS_5 = MethodClass(
    order=5,  # CC_n exactness = n; convergence rate ~ n + 1 in RK convention
    # Estimate dominated by the embedded CC_(n/2) (n/2 + 1 points), exact
    # through degree n/2 + 1 (n/2 even), so error_exponent = n/2 + 3.
    error_exponent=5,
    tableau=_build_clenshaw_curtis_tableau(4),
)
# CC_8 (9 nodes): polynomial exactness 8, error vs CC_4 (5 nodes, exactness 4)
_CLENSHAW_CURTIS_9 = MethodClass(
    order=9,
    error_exponent=7,
    tableau=_build_clenshaw_curtis_tableau(8),
)
# CC_16 (17 nodes): polynomial exactness 16, error vs CC_8 (9 nodes, exactness 8)
_CLENSHAW_CURTIS_17 = MethodClass(
    order=17,
    error_exponent=11,
    tableau=_build_clenshaw_curtis_tableau(16),
)
# CC_32 (33 nodes): polynomial exactness 32, error vs CC_16 (17 nodes, exactness 16)
_CLENSHAW_CURTIS_33 = MethodClass(
    order=33,
    error_exponent=19,
    tableau=_build_clenshaw_curtis_tableau(32),
)
# CC_64 (65 nodes): polynomial exactness 64, error vs CC_32 (33 nodes, exactness 32)
_CLENSHAW_CURTIS_65 = MethodClass(
    order=65,
    error_exponent=35,
    tableau=_build_clenshaw_curtis_tableau(64),
)

CC_METHODS = {
    "cc5": _CLENSHAW_CURTIS_5,
    "cc9": _CLENSHAW_CURTIS_9,
    "cc17": _CLENSHAW_CURTIS_17,
    "cc33": _CLENSHAW_CURTIS_33,
    "cc65": _CLENSHAW_CURTIS_65,
}
