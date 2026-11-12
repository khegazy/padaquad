"""Tests for Butcher tableau coefficient validity."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch
from _helpers import UNIFORM_METHOD_NAMES

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS


@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
def test_uniform_tableau_b_sums_to_one(method_name):
    """The b weights of each uniform method must sum to 1 (consistency condition)."""
    method = UNIFORM_METHODS[method_name]
    b_sum = torch.sum(method.tableau.b)
    assert torch.abs(b_sum - 1.0) < 1e-7, (
        f"Uniform method '{method_name}': b weights sum to {b_sum.item()}, expected 1.0"
    )


@pytest.mark.parametrize(
    "method_name",
    [name for name, method in VARIABLE_METHODS.items() if method.order >= 3],
    ids=[name for name, method in VARIABLE_METHODS.items() if method.order >= 3],
)
def test_variable_tableau_b_sums_to_one(method_name):
    """Variable method b weights (computed from c) must sum to 1 for all valid c positions."""
    method = VARIABLE_METHODS[method_name]
    n_samples = 100

    # Create c tensors with the interior node ranging from 0.01 to 0.99
    c_tensor = torch.concatenate(
        [
            torch.zeros((n_samples, 1, 1)),
            torch.linspace(0.01, 0.99, n_samples)[:, None, None],
            torch.ones((n_samples, 1, 1)),
        ],
        dim=1,
    )
    b_tensor, _ = method.weights(c_tensor)  # [n_samples, 3, 1]
    b_sums = torch.sum(b_tensor[..., 0], dim=-1)

    assert torch.all(torch.abs(b_sums - 1.0) < 1e-5), (
        f"Variable method '{method_name}': some b weight sums deviate from 1.0. "
        f"Max deviation: {torch.max(torch.abs(b_sums - 1.0)).item():.2e}"
    )


# ---------------------------------------------------------------------------
# Independent checks of the stored weights: each computes the weights another
# way (numpy's Gauss-Legendre rule, a solve of the moment equations, exact
# fractions from the published tables) instead of repeating the library code.
# ---------------------------------------------------------------------------

GAUSS_POINTS = {"gk7": 3, "gk15": 7, "gk21": 10, "gk31": 15}


def _lower_weights(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Node fractions and weights ``b - b_error`` of the lower-degree method."""
    tableau = UNIFORM_METHODS[name].tableau
    return tableau.c.numpy(), (tableau.b - tableau.b_error).numpy()


@pytest.mark.parametrize("name", list(GAUSS_POINTS))
def test_gauss_kronrod_lower_method_is_gauss_legendre(name: str) -> None:
    """The error estimate subtracts the n-point Gauss-Legendre rule: its nodes
    and weights equal ``numpy.polynomial.legendre.leggauss(n)`` mapped to
    [0, 1], and every other node has weight 0."""
    c, w = _lower_weights(name)
    x, wx = np.polynomial.legendre.leggauss(GAUSS_POINTS[name])
    used = w != 0
    np.testing.assert_allclose(c[used], (x + 1) / 2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(w[used], wx / 2, rtol=0, atol=1e-15)


def _moment_weights(nodes: np.ndarray) -> np.ndarray:
    """Weights that integrate the Chebyshev polynomials T_0..T_(n-1) exactly on
    [-1, 1] at the given n distinct nodes (the interpolatory weights), from a
    linear solve of the moment equations. Integral of T_k over [-1, 1] is
    2 / (1 - k^2) for even k and 0 for odd k."""
    k = np.arange(len(nodes))
    moments = np.where(k % 2 == 0, 2.0 / (1.0 - k.astype(float) ** 2), 0.0)
    vander = np.polynomial.chebyshev.chebvander(nodes, len(nodes) - 1).T
    return np.linalg.solve(vander, moments)


@pytest.mark.parametrize("name", ["cc5", "cc9", "cc17", "cc33", "cc65"])
def test_clenshaw_curtis_weights_solve_moment_equations(name: str) -> None:
    """Clenshaw-Curtis weights (and the nested half rule subtracted in the error
    estimate) equal the interpolatory weights at Chebyshev nodes, found by
    solving the moment equations instead of the cosine-sum formula."""
    tableau = UNIFORM_METHODS[name].tableau
    x = 2 * tableau.c.numpy() - 1
    np.testing.assert_allclose(
        tableau.b.numpy(), _moment_weights(x) / 2, rtol=0, atol=1e-14
    )
    _, w = _lower_weights(name)
    half = x[::2]  # every other node: the nested rule with (n + 1) / 2 nodes
    np.testing.assert_allclose(w[::2], _moment_weights(half) / 2, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(w[1::2], 0.0)


@pytest.mark.parametrize("a", [0.1, 0.25, 0.3, 0.5, 0.7, 0.9])
def test_interpolatory3_weights_integrate_lagrange_basis(a: float) -> None:
    """interpolatory3_variable's weights at nodes 0, a, 1 are the integrals of
    the three Lagrange basis polynomials over [0, 1], and its error estimate
    subtracts the trapezoidal rule [1/2, 0, 1/2]."""
    nodes = np.array([0.0, a, 1.0])
    expected = []
    for i in range(3):
        others = np.delete(nodes, i)
        basis = np.poly1d(others, r=True) / np.prod(nodes[i] - others)
        expected.append(np.polyint(basis)(1.0) - np.polyint(basis)(0.0))
    method = VARIABLE_METHODS["interpolatory3_variable"]
    b, b_error = method.weights(torch.tensor(nodes, dtype=torch.float64)[None, :, None])
    np.testing.assert_allclose(b[0, :, 0].numpy(), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        (b - b_error)[0, :, 0].numpy(), [0.5, 0.0, 0.5], rtol=0, atol=1e-15
    )


# Lower-degree weights of the Runge-Kutta error estimates, as exact fractions
# from the published embedded pairs (see the comments in
# padaquad/methods/runge_kutta.py for the sources).
RK_LOWER_WEIGHTS = {
    # adaptive_heun subtracts the right-endpoint rectangle rule [0, 1], not
    # Euler's method [1, 0]; both give an estimate of size h |f(1) - f(0)| / 2,
    # with opposite signs.
    "adaptive_heun": [0, 1],
    "fehlberg2": [Fraction(1, 256), Fraction(255, 256), 0],
    "bosh3": [Fraction(7, 24), Fraction(1, 4), Fraction(1, 3), Fraction(1, 8)],
    "dopri5": [
        Fraction(1951, 21600),
        0,
        Fraction(22642, 50085),
        Fraction(451, 720),
        Fraction(-12231, 42400),
        Fraction(649, 6300),
        Fraction(1, 60),
    ],
}


@pytest.mark.parametrize("name", list(RK_LOWER_WEIGHTS))
def test_runge_kutta_error_weights_subtract_the_embedded_method(name: str) -> None:
    """``b - b_error`` equals the embedded lower-order weights to float64
    rounding."""
    _, w = _lower_weights(name)
    expected = np.array([float(v) for v in RK_LOWER_WEIGHTS[name]])
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-16)
