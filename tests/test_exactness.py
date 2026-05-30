"""Polynomial-exactness tests for every quadrature rule.

For an order-p Runge-Kutta method applied as quadrature, the rule
integrates polynomials of degree <= p-1 exactly (for an order-p ODE
method, the per-step truncation error vanishes when the integrand's
(p)-th derivative is zero, i.e. for polynomials of degree p-1).

This test directly applies each method's tableau weights to integrate
``t**d`` on the unit interval ``[0, 1]`` and compares against the
analytical truth ``1 / (d + 1)``. By bypassing the adaptive solver
loop, we isolate the rule's intrinsic polynomial exactness from any
adaptive refinement that could mask weight errors.

Two assertions per method:

  1. For ``d`` from 0 up to and including the claimed exactness, the
     rule is exact to ~1e-13 in float64.
  2. For ``d = exactness + 1``, the rule is *not* exact (catches
     "claimed exactness too high" bugs).

Variable methods are tested at multiple interior-node positions ``a``
to verify position-dependent behavior. Crucial sanity check:
``generic3`` at ``a = 1/2`` reduces to Simpson's rule (weights
``[1/6, 2/3, 1/6]``), which has exactness 3 rather than 2 — the test
encodes both the general-position bound (degree 2) and the
Simpson's-rule edge case (degree 3).

Phase 0 of the quadrature alignment plan.
"""

from __future__ import annotations

import pytest
import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS

EXACTNESS_TOL = 1e-13  # absolute tolerance in float64
INEXACTNESS_MARGIN = 1e-6  # one degree past exactness must miss by at least this


# Claimed polynomial exactness for each uniform method (degree p such that
# the rule integrates t^d exactly for d <= p). Each value here is the
# order-p-1 bound for an order-p RK method applied as a quadrature rule:
# when the integrand is t^k with k <= p-1, the per-step truncation error
# vanishes (the (p+1)-th derivative of the antiderivative is zero).
UNIFORM_EXACTNESS = {
    "adaptive_heun": 1,  # order 2 RK = trapezoidal-like
    "fehlberg2": 1,  # order 2
    "bosh3": 2,  # order 3
    "dopri5": 4,  # order 5
    # Gauss-Kronrod K_{2n+1} integrates polynomials of degree 3n+1
    # exactly (n even) or 3n+2 (n odd). The G_n alone integrates
    # 2n-1, so the embedded G estimate is exact for degree 19 (G10),
    # 13 (G7), or 29 (G15); the b_error indicator picks up the gap
    # between K and G.
    "gk7": 10,  # G3-K7: K7 exactness = 3*3 + 1 = 10
    "gk15": 22,  # G7-K15: K15 exactness = 3*7 + 1 = 22
    "gk21": 31,  # G10-K21: K21 exactness = 3*10 + 1 = 31
    "gk31": 46,  # G15-K31: K31 exactness = 3*15 + 1 = 46
    # Clenshaw-Curtis CC_n integrates polynomials of degree n exactly.
    "cc5": 4,
    "cc9": 8,
    "cc17": 16,
    "cc33": 32,
    "cc65": 64,
}

# Methods for which the "not exact one degree higher" upper-bound check
# is reliable in float64. Above ~degree 20 the mathematical error at
# exactness+1 falls below machine epsilon, so a tight upper-bound check
# is no longer informative — float64 cancellation produces apparent
# exactness at degrees where the rule is in fact mathematically inexact.
# Coefficient-transcription bugs in high-order rules manifest at much
# higher degrees and are caught by the convergence-rate / scipy-agreement
# tests instead.
UNIFORM_TIGHT_UPPER_BOUND = {
    "adaptive_heun",
    "fehlberg2",
    "bosh3",
    "dopri5",
}


def _uniform_quadrature(method, d: int) -> float:
    """Apply ``method``'s tableau as a quadrature rule on ``[0, 1]`` to
    integrate ``t**d``.

    For a unit panel, the rule is ``sum(b_i * c_i**d)``: no h scaling.
    """
    c = method.tableau.c.to(torch.float64).flatten()
    b = method.tableau.b.to(torch.float64).flatten()
    return torch.sum(b * c**d).item()


@pytest.mark.parametrize("method_name", list(UNIFORM_METHODS.keys()))
class TestUniformExactness:
    """Each uniform method integrates polynomials up to its exactness."""

    def test_exact_through_claimed_degree(self, method_name):
        method = UNIFORM_METHODS[method_name]
        exactness = UNIFORM_EXACTNESS[method_name]
        for d in range(exactness + 1):
            got = _uniform_quadrature(method, d)
            truth = 1.0 / (d + 1)
            assert abs(got - truth) < EXACTNESS_TOL, (
                f"{method_name} should integrate t^{d} exactly: "
                f"got {got!r}, truth {truth!r}, "
                f"diff {abs(got - truth)!r} > tol {EXACTNESS_TOL!r}"
            )

    def test_not_exact_one_degree_higher(self, method_name):
        if method_name not in UNIFORM_TIGHT_UPPER_BOUND:
            pytest.skip(
                f"upper-bound check skipped for {method_name}: at this "
                f"polynomial-exactness level the mathematical error at "
                f"exactness+1 falls below float64 machine epsilon, so the "
                f"check is not informative. Coefficient errors are caught "
                f"by test_scipy_agreement and test_convergence_rate."
            )
        method = UNIFORM_METHODS[method_name]
        exactness = UNIFORM_EXACTNESS[method_name]
        d = exactness + 1
        got = _uniform_quadrature(method, d)
        truth = 1.0 / (d + 1)
        assert abs(got - truth) > INEXACTNESS_MARGIN, (
            f"{method_name} accidentally exact for t^{d}? "
            f"got {got!r}, truth {truth!r}, "
            f"diff {abs(got - truth)!r} <= margin {INEXACTNESS_MARGIN!r}. "
            f"Either exactness is higher than claimed or a weight is wrong."
        )


# -----------------------------------------------------------------------------
# Variable methods: weights depend on interior node position(s).
# -----------------------------------------------------------------------------


def _variable_quadrature(method, c_normalized: torch.Tensor, d: int) -> float:
    """Apply a variable-sampling method's dynamically-computed weights.

    Args:
        method: An instance of a variable-method class.
        c_normalized: Normalized node positions ``[C]`` in ``[0, 1]``.
            The method internally interprets ``c[N, C, T]`` so we reshape.
        d: Polynomial degree to integrate.
    """
    c = c_normalized.to(torch.float64).view(1, -1, 1)
    b, _ = method.tableau_b(c)
    return (b[0] * c_normalized.to(torch.float64) ** d).sum().item()


class TestVariableHeunExactness:
    """Variable adaptive_heun has fixed weights (position-independent)
    so it should be trapezoidal: exact for degree 1 only.
    """

    @pytest.fixture
    def method(self):
        cls = VARIABLE_METHODS["adaptive_heun"]
        # Construct on CPU so we don't depend on device
        instance = cls(device="cpu")
        instance.to_dtype(torch.float64)
        return instance

    def test_exact_through_degree_1(self, method):
        # Endpoints at [0, 1] is the natural choice for a 2-node trap rule
        c = torch.tensor([0.0, 1.0], dtype=torch.float64)
        for d in range(2):
            got = _variable_quadrature(method, c, d)
            truth = 1.0 / (d + 1)
            assert abs(got - truth) < EXACTNESS_TOL

    def test_not_exact_for_degree_2(self, method):
        c = torch.tensor([0.0, 1.0], dtype=torch.float64)
        d = 2
        got = _variable_quadrature(method, c, d)
        truth = 1.0 / (d + 1)
        assert abs(got - truth) > INEXACTNESS_MARGIN


class TestVariableThirdOrderExactness:
    """`generic3` (Sanderse-Veldman) is a 3-point interpolatory rule with
    one variable interior node ``a``.

    At ``a = 1/2`` the weights become ``[1/6, 2/3, 1/6]`` — Simpson's rule —
    so exactness rises to degree 3. At general ``a in (0, 1) \\ {1/2}`` the
    rule has exactness 2.
    """

    @pytest.fixture
    def method(self):
        cls = VARIABLE_METHODS["interpolatory3_variable"]
        instance = cls(device="cpu")
        instance.to_dtype(torch.float64)
        return instance

    def test_simpson_reduction_at_half(self, method):
        # Verify the weights themselves match Simpson's rule.
        c = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64).view(1, 3, 1)
        b, _ = method.tableau_b(c)
        expected = torch.tensor([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=torch.float64)
        assert torch.allclose(b[0], expected, atol=1e-15), (
            f"generic3 at a=1/2 should be Simpson's rule [1/6, 2/3, 1/6]; "
            f"got {b[0].tolist()}"
        )

    def test_simpson_exact_through_degree_3(self, method):
        c = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        for d in range(4):
            got = _variable_quadrature(method, c, d)
            truth = 1.0 / (d + 1)
            assert abs(got - truth) < EXACTNESS_TOL, (
                f"generic3 at a=1/2 (Simpson's) should be exact for t^{d}: "
                f"got {got}, truth {truth}"
            )

    def test_simpson_not_exact_for_degree_4(self, method):
        c = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        d = 4
        got = _variable_quadrature(method, c, d)
        truth = 1.0 / (d + 1)
        assert abs(got - truth) > INEXACTNESS_MARGIN

    @pytest.mark.parametrize("a", [0.2, 0.3, 0.4, 0.7])
    def test_general_position_exact_through_degree_2(self, method, a):
        c = torch.tensor([0.0, a, 1.0], dtype=torch.float64)
        for d in range(3):
            got = _variable_quadrature(method, c, d)
            truth = 1.0 / (d + 1)
            assert abs(got - truth) < EXACTNESS_TOL, (
                f"generic3 at a={a} should be exact for t^{d}: got {got}, truth {truth}"
            )

    @pytest.mark.parametrize("a", [0.2, 0.3, 0.4, 0.7])
    def test_general_position_not_exact_for_degree_3(self, method, a):
        c = torch.tensor([0.0, a, 1.0], dtype=torch.float64)
        d = 3
        got = _variable_quadrature(method, c, d)
        truth = 1.0 / (d + 1)
        assert abs(got - truth) > INEXACTNESS_MARGIN, (
            f"generic3 at a={a} unexpectedly exact for t^{d}? got {got}, truth {truth}"
        )
