"""
Embedded Runge-Kutta pairs used as quadrature methods.

An explicit Runge-Kutta (RK) method for ordinary differential equations,
applied to an integrand ``f(t)`` that does not depend on the solution, reduces
to a quadrature method: its node fractions are the tableau's ``c`` and its
weights the tableau's ``b``. An RK method of order p then integrates every
polynomial of degree p - 1 exactly. An embedded RK pair adds a second weight
set of lower degree, and the difference of the two results is the error
estimate. Degree of each method and of the lower-degree method in its error
estimate, as measured from the stored weights by ``tests/test_exactness.py``:

- ``adaptive_heun`` (RK order 2, Heun's method): degree 1 (the trapezoidal
  rule); lower degree 0.
- ``fehlberg2`` (RK order 2): degree 1; lower degree 0.
- ``bosh3`` (RK order 3, Bogacki-Shampine): degree 2; lower degree 1.
- ``dopri5`` (RK order 5, Dormand-Prince): degree 4; lower degree 3.

On smooth integrands the Gauss-Kronrod and Clenshaw-Curtis methods reach a
higher degree with the same number of integrand evaluations.

``fehlberg2``'s error estimate is ``h * (f(end) - f(start)) / 512``, which is
blind to the curvature of ``f``: on ``f(t) = t**2 - t`` over [0, 1] it is zero
while the method's error is not.

References
----------
- ``fehlberg2``: Fehlberg 1969, Table XIV "RK1(2)", report p. 29
  :cite:p:`fehlberg1969`.
- ``adaptive_heun``, ``bosh3`` and ``dopri5``: transcribed from torchdiffeq
  :cite:p:`torchdiffeq`; the node fractions ``c`` and weights ``b`` also match
  SciPy's RK23 and RK45 tables :cite:p:`scipy_quadvec`. The primary papers
  (Heun 1900; Bogacki and Shampine 1989; Dormand and Prince 1980) were not
  checked. The source of ``dopri5``'s embedded weights is not known (see the
  comment in the code).
"""

from __future__ import annotations

import torch

from ._base import UniformMethodClass, _Tableau

# Adaptive Heun: second-order RK, 2 nodes per subinterval (its two ends).
# Tableau transcribed from torchdiffeq (Chen 2018, github.com/rtqichen/torchdiffeq,
# _impl/adaptive_heun.py at commit 657943a); Heun 1900 not checked.
_ADAPTIVE_HEUN = UniformMethodClass(
    references=("torchdiffeq",),
    degree=1,  # trapezoidal rule
    # The error weights subtract the right-endpoint rectangle rule [0, 1]
    # (degree 0), not Euler's method [1, 0]: the estimate has the size of
    # trapezoid minus Euler, h |f(1) - f(0)| / 2, with the opposite sign.
    error_exponent=2,
    tableau=_Tableau(
        c=torch.tensor([0.0, 1.0], dtype=torch.float64),
        b=torch.tensor([0.5, 0.5], dtype=torch.float64),
        b_error=torch.tensor([0.5, -0.5], dtype=torch.float64),
    ),
)

# Fehlberg2: second-order RK, 3 nodes per subinterval (start, middle, end).
# Fehlberg 1969, NASA TR R-315, Table XIV "RK1(2)", report p. 29: c = 0, 1/2, 1;
# second-order weights 1/512, 255/256, 1/512 (used for the integral); the error
# weights subtract the first-order weights 1/256, 255/256, 0. The estimate,
# h (f(1) - f(0)) / 512, ignores curvature (see the module docstring and
# examples/verification/integral_accuracy.py).
_FEHLBERG2 = UniformMethodClass(
    references=("fehlberg1969",),
    degree=1,
    error_exponent=2,  # embedded first-order method, degree 0
    tableau=_Tableau(
        c=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        b=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64),
        b_error=torch.tensor([-1 / 512, 0, 1 / 512], dtype=torch.float64),
    ),
)

# Bogacki-Shampine: third-order RK, 4 nodes per subinterval. Tableau
# transcribed from torchdiffeq (_impl/bosh3.py at commit 657943a); c and b match
# SciPy's RK23 (scipy/integrate/_ivp/rk.py, v1.18.1, lines 360-367). The error
# weights subtract the embedded second-order weights 7/24, 1/4, 1/3, 1/8.
# Bogacki and Shampine 1989 not checked.
_BOGACKI_SHAMPINE = UniformMethodClass(
    references=("torchdiffeq", "scipy_quadvec"),
    degree=2,
    error_exponent=3,  # embedded second-order method, degree 1
    tableau=_Tableau(
        c=torch.tensor([0.0, 0.5, 0.75, 1.0], dtype=torch.float64),
        b=torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.0], dtype=torch.float64),
        b_error=torch.tensor(
            [2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype=torch.float64
        ),
    ),
)

# Dormand-Prince: fifth-order RK, 7 nodes per subinterval (two at the right
# end). Tableau transcribed from torchdiffeq (_impl/dopri5.py at commit
# 657943a); c and b match SciPy's RK45 (rk.py, v1.18.1, lines 541-552).
# Dormand and Prince 1980 not checked.
_DORMAND_PRINCE_SHAMPINE = UniformMethodClass(
    references=("torchdiffeq", "scipy_quadvec"),
    degree=4,
    error_exponent=5,  # embedded fourth-order method, degree 3
    tableau=_Tableau(
        c=torch.tensor(
            [0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0], dtype=torch.float64
        ),
        b=torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            dtype=torch.float64,
        ),
        # b_error = b - b_embedded, with the embedded fourth-order set
        # 1951/21600, 0, 22642/50085, 451/720, -12231/42400, 649/6300, 1/60
        # taken from torchdiffeq. Its published source was not found (a search
        # snippet suggests Shampine 1986, not checked). SciPy's RK45 uses a
        # different embedded set (5179/57600, ...). Both satisfy the fourth-order
        # conditions (checked with exact fractions); as quadrature methods both
        # have degree 3.
        b_error=torch.tensor(
            [
                35 / 384 - 1951 / 21600,
                0,
                500 / 1113 - 22642 / 50085,
                125 / 192 - 451 / 720,
                -2187 / 6784 - -12231 / 42400,
                11 / 84 - 649 / 6300,
                -1.0 / 60.0,
            ],
            dtype=torch.float64,
        ),
    ),
)

RK_METHODS = {
    "adaptive_heun": _ADAPTIVE_HEUN,
    "fehlberg2": _FEHLBERG2,
    "bosh3": _BOGACKI_SHAMPINE,
    "dopri5": _DORMAND_PRINCE_SHAMPINE,
}
