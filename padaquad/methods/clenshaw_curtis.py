r"""
Clenshaw-Curtis methods: cc5, cc9, cc17, cc33 and cc65.

The Clenshaw-Curtis method CC_n integrates the polynomial that interpolates
``f`` at the n + 1 Chebyshev points :math:`x_k = \\cos(k\\pi/n)`,
k = 0, ..., n, on [-1, 1] (Trefethen 2008, Eq. (2.3), p. 70
:cite:p:`trefethen2008`). Because it integrates the interpolating polynomial,
it is exact for every polynomial of degree n (Trefethen 2008, Sec. 2, p. 69).
For even n, as here, it is exact through degree n + 1 (Trefethen 2008,
Theorem 5.2, Eq. (5.4), p. 78); in short, n + 1 is then odd, and a method with
symmetric nodes and weights integrates every odd power about the center
exactly. ``ccN`` has N = n + 1 nodes and degree N: cc5,
cc9, cc17, cc33 and cc65 have n = 4, 8, 16, 32 and 64.
``tests/test_exactness.py`` measures the degrees from the stored tables.

Two properties padaquad uses:

1. The nodes are nested: the nodes of CC_(n/2) are every other node of CC_n.
   The error estimate pairs CC_n with CC_(n/2) at no extra cost: the error
   weights are the CC_n weights minus the CC_(n/2) weights, with weight 0 at
   the nodes only CC_n has. This pairing is padaquad's choice; Gonnet 2012,
   Sec. 6 :cite:p:`gonnet2012`, discusses nested Clenshaw-Curtis pairs in
   general.
2. The nodes include both ends of the subinterval (:math:`x_0 = 1` and
   :math:`x_n = -1` map to node fractions 1 and 0), so no padding is needed,
   unlike Gauss-Kronrod.

References
----------
- Nodes, degree and weights: Trefethen 2008 :cite:p:`trefethen2008`. The
  weight formula in ``_clenshaw_curtis_weights_unit`` is algebraically the same
  as Trefethen's ``clenshaw_curtis.m`` (p. 70), and ``tests/test_tableau.py``
  checks the weights numerically against the moment equations.
- Waldvogel 2006 and Clenshaw and Curtis 1960 were not checked.
"""

from __future__ import annotations

import math

import torch

from ._base import UniformMethodClass, _Tableau


def _clenshaw_curtis_weights_unit(n: int) -> list[float]:
    """Compute Clenshaw-Curtis weights on (-1, 1) for n+1 Chebyshev nodes.

    Nodes are x_k = cos(k * pi / n) for k = 0..n; the method integrates
    polynomials of degree at most n exactly (n + 1 for even n, by symmetry).
    The weights are w_k = (2 / n) sum''_j a_j cos(j k pi / n), halved at
    k = 0 and k = n, where a_j = 2 / (1 - j^2) for even j, 0 for odd j, and
    sum'' halves the j = 0 and j = n terms.

    Computed with Python floats and ``math.cos`` (the C library cosine), in
    this loop order. Do not vectorize the sum over j: reordering the additions
    changes the weights in the last bits, and ``torch.cos`` differs from the C
    library cosine by one unit in the last place on some arguments.

    Parameters
    ----------
    n : int
        Number of Chebyshev intervals; the method has n + 1 nodes. n = 0
        returns the single weight 2.

    Returns
    -------
    list of float
        Weights of length n + 1, indexed as the nodes: the weight at
        cos(k * pi / n) is at position k.
    """
    if n == 0:
        return [2.0]
    # Cosine-series formula; algebraically the same as clenshaw_curtis.m in
    # Trefethen 2008, SIAM Review 50(1), p. 70 (weights 2/(1 - k^2) on the even
    # Chebyshev coefficients).
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

    The error estimate pairs CC_n with CC_(n/2) (padaquad's choice). The nodes
    of CC_(n/2) are every other node of CC_n, so the CC_(n/2) weights extended
    to the CC_n nodes are the CC_(n/2) weight at shared nodes and 0 at the
    nodes only CC_n has.

    Parameters
    ----------
    n : int
        Number of Chebyshev intervals of CC_n, even and >= 2; the method has
        n + 1 nodes.

    Returns
    -------
    _Tableau
        Float64 node fractions ``c`` (increasing from 0 to 1), weights ``b``
        of CC_n and error weights ``b_error`` = CC_n minus CC_(n/2) weights.
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


# Degrees: CC_n with even n has degree n + 1 (Trefethen 2008, SIAM Review 50(1),
# Theorem 5.2, Eq. (5.4), p. 78); measured in tests/test_exactness.py.
# CC_4 (5 nodes): degree 5, error vs CC_2 (3 nodes, Simpson's rule, degree 3)
_CLENSHAW_CURTIS_5 = UniformMethodClass(
    references=("trefethen2008",),
    degree=5,  # CC_n with even n has degree n + 1; order (convergence rate) 6
    # Estimate dominated by the embedded CC_(n/2) (n/2 + 1 points), exact
    # through degree n/2 + 1 (n/2 even), so error_exponent = n/2 + 3.
    error_exponent=5,
    tableau=_build_clenshaw_curtis_tableau(4),
)
# CC_8 (9 nodes): degree 9, error vs CC_4 (5 nodes, degree 5)
_CLENSHAW_CURTIS_9 = UniformMethodClass(
    references=("trefethen2008",),
    degree=9,
    error_exponent=7,
    tableau=_build_clenshaw_curtis_tableau(8),
)
# CC_16 (17 nodes): degree 17, error vs CC_8 (9 nodes, degree 9)
_CLENSHAW_CURTIS_17 = UniformMethodClass(
    references=("trefethen2008",),
    degree=17,
    error_exponent=11,
    tableau=_build_clenshaw_curtis_tableau(16),
)
# CC_32 (33 nodes): degree 33, error vs CC_16 (17 nodes, degree 17)
_CLENSHAW_CURTIS_33 = UniformMethodClass(
    references=("trefethen2008",),
    degree=33,
    error_exponent=19,
    tableau=_build_clenshaw_curtis_tableau(32),
)
# CC_64 (65 nodes): degree 65, error vs CC_32 (33 nodes, degree 33)
_CLENSHAW_CURTIS_65 = UniformMethodClass(
    references=("trefethen2008",),
    degree=65,
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
