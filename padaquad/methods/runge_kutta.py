"""
Embedded Runge-Kutta quadrature methods.

These are RK pairs originally designed for ODE integration, applied here
as quadrature rules on a known integrand f(t). Each pair provides a
high-order weight set for the integral and an embedded lower-order
weight set used to build the per-step error estimate.

Polynomial exactness as quadrature rules:
  adaptive_heun (order-2 RK):    degree 1 (trapezoidal)
  fehlberg2     (order-2 RK):    degree 1
  bosh3         (order-3 RK):    degree 2
  dopri5        (order-5 RK):    degree 4

These methods are useful and historically convenient, but on smooth
integrands the Gauss-Kronrod and Clenshaw-Curtis families in their
sister modules deliver higher polynomial exactness for the same
number of evaluations.
"""

from __future__ import annotations

import torch

from ._base import MethodClass, _Tableau

# Adaptive Heun: 2nd-order method with 2 evaluations per step (endpoints only)
_ADAPTIVE_HEUN = MethodClass(
    order=2,
    tableau=_Tableau(
        c=torch.tensor([0.0, 1.0], dtype=torch.float64),
        b=torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        b_error=torch.tensor([[0.5, -0.5]], dtype=torch.float64),
    ),
)

# Fehlberg2: 2nd-order method with 3 evaluations per step (start, mid, end)
_FEHLBERG2 = MethodClass(
    order=2,
    tableau=_Tableau(
        c=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        b=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64),
        b_error=torch.tensor([-1 / 512, 0, 1 / 512], dtype=torch.float64),
    ),
)

# Bogacki-Shampine: 3rd-order method with 4 evaluations per step
_BOGACKI_SHAMPINE = MethodClass(
    order=3,
    tableau=_Tableau(
        c=torch.tensor([0.0, 0.5, 0.75, 1.0], dtype=torch.float64),
        b=torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.0], dtype=torch.float64),
        b_error=torch.tensor(
            [2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype=torch.float64
        ),
    ),
)

# Dormand-Prince: 5th-order method with 7 evaluations per step
_DORMAND_PRINCE_SHAMPINE = MethodClass(
    order=5,
    tableau=_Tableau(
        c=torch.tensor(
            [0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0], dtype=torch.float64
        ),
        b=torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            dtype=torch.float64,
        ),
        # b_error = b_primary - b_embedded: the difference between the 5th-order
        # and 4th-order method weights, used for error estimation
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
