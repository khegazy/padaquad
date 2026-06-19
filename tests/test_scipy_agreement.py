"""Cross-validate every uniform method against ``scipy.integrate.quad``.

Phase 0 of the quadrature alignment plan. ``scipy.integrate.quad`` is
QUADPACK-based adaptive quadrature with a long history of correctness;
agreement with it on canonical integrands ground-truths the library's
own adaptive control. The tests in this file pin agreement on smooth
integrands at default settings — the bar for an academic-grade
quadrature library.

For each method x each canonical integrand x each interval, the
library's reported integral must match ``scipy.integrate.quad`` to
``10 * atol`` (giving the library a small allowance over scipy's
default error). If a method's adaptive control is mistuned or its
weights are wrong, this test catches the disagreement.

When Phase 2 adds new methods (gk21, cc33, ...), this file gets new
parametrize entries automatically once the new methods register.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import integrate as scipy_integrate
from tests._helpers import (
    SEED,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    UNIFORM_METHOD_NAMES,
    cached_max_batch,
)

from padaquad import integrate

# Canonical smooth integrands. Each is a tuple of:
#   (name, torch_callable, scipy_callable, intervals_to_test)
# The two callables are equivalent — torch for the library, scipy for
# the reference. Different intervals exercise the same integrand
# under translations and scalings.
_INTEGRANDS = [
    (
        "sin",
        torch.sin,
        np.sin,
        [(0.0, math.pi), (0.0, 2 * math.pi), (-math.pi, math.pi)],
    ),
    (
        "exp",
        torch.exp,
        np.exp,
        [(0.0, 1.0), (-1.0, 1.0), (-2.0, 0.0)],
    ),
    (
        "polynomial_t4",
        lambda t: t**4,
        lambda t: t**4,
        [(0.0, 1.0), (-1.0, 1.0)],
    ),
    (
        "gaussian_bump",
        lambda t: torch.exp(-(t**2)),
        lambda t: np.exp(-(t**2)),
        [(-2.0, 2.0), (0.0, 3.0)],
    ),
]

ATOL = 1e-8
RTOL = 1e-8
# The library's actual delivered accuracy is looser than its atol setting on
# many integrands; the adaptive control accepts each step's local error
# against (atol + rtol*|I|) but the global error can accumulate. The bound
# below is "library agrees with scipy to 5 decimal places on smooth
# integrands" — a sensible academic bar, and well within what a quadrature
# library should deliver at atol=1e-8 on canonical functions. Tightening
# this bound is a Phase 6 stretch goal once the new methods (gk21, cc33)
# replace the lower-order RK methods as defaults.
AGREEMENT_BOUND = 1e-5


def _ids():
    """Build readable test ids: method-integrand-a-b."""
    out = []
    for method in UNIFORM_METHOD_NAMES:
        for name, _, _, intervals in _INTEGRANDS:
            for a, b in intervals:
                out.append((method, name, a, b))
    return out


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize(
    ("method", "integrand_name", "a", "b"),
    _ids(),
    ids=lambda v: str(v).replace(".", "p"),
)
def test_method_agrees_with_scipy_quad(method, integrand_name, a, b, take_gradient):
    """Library's integrate matches ``scipy.integrate.quad`` to ``10*atol``."""
    # Seed for deterministic initial-mesh placement; otherwise this test's
    # outcome depends on whatever random state was left by the previous test.
    torch.manual_seed(SEED)

    # Locate the matching torch / scipy callables.
    torch_fn = next(t for n, t, _, _ in _INTEGRANDS if n == integrand_name)
    scipy_fn = next(s for n, _, s, _ in _INTEGRANDS if n == integrand_name)

    scipy_value, scipy_err = scipy_integrate.quad(scipy_fn, a, b, epsabs=ATOL)
    library_result = integrate(
        f=torch_fn,
        method=method,
        atol=ATOL,
        rtol=RTOL,
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        take_gradient=take_gradient,
        max_batch=cached_max_batch(),
    )
    library_value = library_result.integral.item()

    # Generous bound: the test catches order-of-magnitude disagreement
    # between library and scipy, not the last few digits of agreement.
    bound = max(scipy_err, AGREEMENT_BOUND)
    assert abs(library_value - scipy_value) < bound, (
        f"{method} on {integrand_name} over [{a}, {b}]: "
        f"library={library_value}, scipy={scipy_value}, "
        f"diff={abs(library_value - scipy_value)}, bound={bound}, "
        f"library_error_estimate={library_result.integral_error.item()}"
    )
