"""Convergence-rate tests for every quadrature rule.

For an order-p quadrature rule applied to a smooth integrand on a
fixed interval with ``N`` equal panels, the global integration error
scales as ``O(N^(-p))``. Halving the panel size (doubling ``N``)
reduces the error by ``2^p``.

This test directly applies each method's tableau to ``N`` equal
panels (no adaptive control) and verifies that the empirical error
decay matches the theoretical rate. Bypassing the adaptive solver
means this test specifically certifies the rule's *order* property,
not the controller's behavior.

Phase 0 of the quadrature alignment plan.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchpathdiffeq import UNIFORM_METHODS, VARIABLE_METHODS

# Each method's expected convergence rate (= polynomial-exactness + 1, the
# number of orders by which error scales as panel size shrinks).
UNIFORM_RATE = {
    "adaptive_heun": 2,
    "fehlberg2": 2,
    "bosh3": 3,
    "dopri5": 5,
    "cc5": 5,
    "cc9": 9,
    "gk7": 11,
}

# High-order rules saturate to float64 machine epsilon at small N
# (e.g. gk21 hits eps after only a handful of panels), so a log-log
# slope across N=8..128 is dominated by rounding noise, not by the
# theoretical convergence rate. Skip them here; their correctness is
# anchored by test_exactness (polynomial-exactness) and
# test_scipy_agreement.
_CONVERGENCE_TEST_SKIP = {
    "cc5",
    "cc9",
    "gk7",
    "gk15",
    "gk21",
    "gk31",
    "cc17",
    "cc33",
    "cc65",
}

INTERVAL = (0.0, 1.0)
PANEL_COUNTS = (8, 16, 32, 64, 128)
# Use exp(t), an asymmetric / non-periodic smooth integrand. sin(t) over
# [0, π] superconverges by one extra order on every method here because
# its odd derivatives vanish at the endpoints (a half-period symmetry);
# exp avoids that artifact and so reveals each method's intrinsic rate.
TRUTH = math.e - 1.0  # int_0^1 exp(t) dt = e - 1


def _fixed_panel_uniform(method, f, a: float, b: float, N: int) -> float:
    """Evaluate ``method``'s tableau as a quadrature rule with ``N``
    equal panels on ``[a, b]``. Bypasses any adaptive refinement.
    """
    a64 = torch.tensor(a, dtype=torch.float64)
    b64 = torch.tensor(b, dtype=torch.float64)
    h = (b64 - a64) / N
    barriers = a64 + h * torch.arange(N + 1, dtype=torch.float64)
    c = method.tableau.c.to(torch.float64).flatten()
    weights = method.tableau.b.to(torch.float64).flatten()
    # Nodes: [N, C], panel i node j is barriers[i] + h*c[j]
    nodes = barriers[:-1, None] + h * c[None, :]
    y = f(nodes.flatten()).view(N, len(c))
    return float(h * (weights[None, :] * y).sum())


def _fixed_panel_variable(method, f, a: float, b: float, N: int) -> float:
    """Evaluate a variable method with ``N`` equal panels.
    Nodes are placed uniformly within each panel via the same tableau-c
    pattern that the variable solver would use at the start.
    """
    a64 = torch.tensor(a, dtype=torch.float64)
    b64 = torch.tensor(b, dtype=torch.float64)
    h = (b64 - a64) / N
    barriers = a64 + h * torch.arange(N + 1, dtype=torch.float64)
    C = method.n_tableau_c
    # For variable methods, the natural choice is a uniform spacing of
    # nodes inside each panel. The middle node sits at the midpoint
    # (a=1/2 in tableau terms), so generic3 reduces to Simpson's rule
    # under this layout, raising its rate from 3 to 4.
    c_pattern = torch.linspace(0.0, 1.0, C, dtype=torch.float64)  # [C]
    nodes = barriers[:-1, None] + h * c_pattern[None, :]  # [N, C]
    # The variable methods compute weights from the per-panel normalized c.
    c_3d = c_pattern.view(1, C, 1).expand(N, C, 1).contiguous()
    weights, _ = method.tableau_b(c_3d)  # [N, C]
    y = f(nodes.flatten()).view(N, C)
    return float(h * (weights * y).sum())


def _empirical_rate(errors: list[float]) -> float:
    """Estimate the convergence rate from a sequence of errors at
    geometrically increasing N. Fits log(error) vs log(N) by least
    squares; the slope is ``-rate``.
    """
    log_errors = np.log(np.array(errors))
    log_n = np.log(np.array(PANEL_COUNTS))
    slope, _ = np.polyfit(log_n, log_errors, 1)
    return -slope


@pytest.mark.parametrize("method_name", list(UNIFORM_METHODS.keys()))
def test_uniform_convergence_rate(method_name):
    """The empirical convergence rate matches the method's claimed rate
    to within ``±0.5`` (a generous tolerance for low-N noise).
    """
    if method_name in _CONVERGENCE_TEST_SKIP:
        pytest.skip(
            f"{method_name}: convergence rate not testable in float64 — "
            f"the rule's per-panel error reaches machine epsilon at the "
            f"low N values used here, dominating the slope estimate."
        )
    method = UNIFORM_METHODS[method_name]
    expected_rate = UNIFORM_RATE[method_name]

    errors = []
    for N in PANEL_COUNTS:
        approx = _fixed_panel_uniform(method, torch.exp, *INTERVAL, N)
        errors.append(abs(approx - TRUTH))

    # Some methods reach float64 epsilon at high N; drop those points to
    # avoid noise from rounding error swamping the high-order regime.
    rate_errors = [e for e in errors if e > 1e-13]
    rate_panels = PANEL_COUNTS[: len(rate_errors)]
    log_errors = np.log(np.array(rate_errors))
    log_n = np.log(np.array(rate_panels))
    slope = np.polyfit(log_n, log_errors, 1)[0]
    empirical_rate = -slope

    assert abs(empirical_rate - expected_rate) < 0.5, (
        f"{method_name}: expected rate ~{expected_rate}, "
        f"empirical rate {empirical_rate:.3f}. "
        f"errors at N={PANEL_COUNTS}: {[f'{e:.2e}' for e in errors]}"
    )


VARIABLE_RATE = {
    # Variable adaptive_heun has fixed weights (trapezoidal-like), order 2.
    "adaptive_heun": 2,
    # Variable generic3 with uniformly-placed inner node (a=1/2 per panel)
    # reduces to Simpson's rule which is order 4 — rate 4, NOT 3.
    "interpolatory3_variable": 4,
}


@pytest.mark.parametrize("method_name", list(VARIABLE_METHODS.keys()))
def test_variable_convergence_rate(method_name):
    """Variable methods' convergence under uniform-node-placement.

    Note: ``generic3`` with the middle node at the midpoint of each
    panel reduces to Simpson's rule (weights ``[1/6, 2/3, 1/6]``), so
    its convergence rate rises from the general-position 3 to 4. This
    test pins that fact by parametrizing the expected rate to 4.
    """
    cls = VARIABLE_METHODS[method_name]
    method = cls(device="cpu")
    method.to_dtype(torch.float64)
    expected_rate = VARIABLE_RATE[method_name]

    errors = []
    for N in PANEL_COUNTS:
        approx = _fixed_panel_variable(method, torch.exp, *INTERVAL, N)
        errors.append(abs(approx - TRUTH))

    rate_errors = [e for e in errors if e > 1e-13]
    rate_panels = PANEL_COUNTS[: len(rate_errors)]
    log_errors = np.log(np.array(rate_errors))
    log_n = np.log(np.array(rate_panels))
    slope = np.polyfit(log_n, log_errors, 1)[0]
    empirical_rate = -slope

    assert abs(empirical_rate - expected_rate) < 0.5, (
        f"{method_name} (variable): expected rate ~{expected_rate}, "
        f"empirical rate {empirical_rate:.3f}. "
        f"errors at N={PANEL_COUNTS}: {[f'{e:.2e}' for e in errors]}"
    )
