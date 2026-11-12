"""Polynomial-degree tests for every quadrature method.

A method's degree is the highest polynomial degree it integrates exactly. Each
method stores the degree its theory gives (``method.degree``); these tests
measure it from the stored nodes and weights with ``MethodClass.measure_degree``
and check, for every method:

1. it is exact through ``degree`` and not one degree further (both directions);
2. the lower-degree method inside its error estimate (weights ``b - b_error``)
   has degree ``error_exponent - 2``, the strength the solver's step-merging
   rule assumes;
3. in 60-digit arithmetic, the stored float64 tables are exact through
   ``degree`` up to the rounding of the stored values. This uses mpmath's own
   Legendre polynomials, so it also checks the float64 measurement.

The measurement integrates Legendre polynomials, not powers. A power
(2t - 1)^(d+1) is almost entirely made of lower-degree polynomials, so a
method's error on it is tiny (about 3e-17 for gk31, below float64 rounding) and
the power test reports false degrees; powers of t itself are worse. The Legendre errors one degree past each
method's degree are all at least 4e-6.

``examples/verification/polynomial_degree.py`` runs the same measurement for
users.
"""

from __future__ import annotations

import pytest
import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS

# Measured on 2026-11-12: the largest float64 error through a method's degree
# is 3.5e-16 and the smallest error one degree past it is 4.2e-6 (cc65).
MISS_MARGIN = 1e-8  # an error one degree past the degree must be at least this
# Measured in 60 digits on the stored float64 tables: at most 3.1e-16.
STORED_TABLE_TOL = 1e-14

# interpolatory3_variable: degree 2 for a general middle node, 3 at 1/2
# (Simpson's rule). Its default nodes put the middle node at 1/2.
OFF_CENTER = (0.25, 0.3, 0.7)


def _middle_node(a: float) -> torch.Tensor:
    return torch.tensor([0.0, a, 1.0], dtype=torch.float64)


@pytest.mark.parametrize("name", list(UNIFORM_METHODS))
def test_uniform_degree_both_directions(name: str) -> None:
    """Exact through ``degree``; not exact at ``degree + 1``."""
    method = UNIFORM_METHODS[name]
    measured = method.measure_degree()
    assert measured.degree == method.degree, (
        f"{name}: stored degree {method.degree}, measured {measured.degree}"
    )
    miss = abs(float(measured.errors[method.degree + 1]))
    assert miss >= MISS_MARGIN, f"{name}: error at degree + 1 is only {miss:.2e}"


@pytest.mark.parametrize("name", list(UNIFORM_METHODS))
def test_uniform_error_estimate_strength(name: str) -> None:
    """The lower-degree method in the error estimate has degree k - 2."""
    method = UNIFORM_METHODS[name]
    assert method.measure_degree().lower_degree == method.error_exponent - 2


@pytest.mark.parametrize("a", OFF_CENTER)
def test_interpolatory3_general_position(a: float) -> None:
    """Degree 2 and error-estimate degree 1 with the middle node off center."""
    method = VARIABLE_METHODS["interpolatory3_variable"]
    measured = method.measure_degree(_middle_node(a))
    assert measured.degree == method.degree == 2
    assert abs(float(measured.errors[3])) >= MISS_MARGIN
    assert measured.lower_degree == method.error_exponent - 2


def test_interpolatory3_center_is_simpson() -> None:
    """With the middle node at 1/2 the method is Simpson's rule: degree 3."""
    method = VARIABLE_METHODS["interpolatory3_variable"]
    assert method.measure_degree(_middle_node(0.5)).degree == method.degree + 1
    assert method.measure_degree().degree == method.degree + 1  # default nodes


def test_variable_adaptive_heun() -> None:
    """The variable registry's adaptive_heun is the trapezoidal rule."""
    method = VARIABLE_METHODS["adaptive_heun"]
    measured = method.measure_degree()
    assert measured.degree == method.degree == 1
    assert measured.lower_degree == method.error_exponent - 2


@pytest.mark.parametrize("name", list(UNIFORM_METHODS))
def test_stored_tables_in_60_digits(name: str) -> None:
    """The stored float64 tables are exact through ``degree`` to 1e-14 and miss
    at ``degree + 1``, evaluated in 60-digit arithmetic with mpmath's Legendre
    polynomials (independent of ``measure_degree``)."""
    mpmath = pytest.importorskip("mpmath")
    method = UNIFORM_METHODS[name]
    mp = mpmath.mp.clone()
    mp.dps = 60
    c = [float(v) for v in method.tableau.c]
    b = [float(v) for v in method.tableau.b]
    x = [2 * mp.mpf(v) - 1 for v in c]
    w = [mp.mpf(v) for v in b]
    for k in range(method.degree + 2):
        exact = 1 if k == 0 else 0
        error = abs(
            mp.fsum(wi * mp.legendre(k, xi) for xi, wi in zip(x, w, strict=True))
            - exact
        )
        if k <= method.degree:
            assert error <= STORED_TABLE_TOL, f"{name}: degree {k} error {error}"
        else:
            assert error >= MISS_MARGIN, f"{name}: degree {k} error only {error}"


def test_measure_degree_rejects_other_nodes_for_fixed_weights() -> None:
    """A method whose weights ignore the node positions can only be measured at
    its own node fractions."""
    method = UNIFORM_METHODS["cc5"]
    other = torch.tensor([0.0, 0.2, 0.5, 0.75, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="do not depend on the node positions"):
        method.measure_degree(other)


@pytest.mark.parametrize(
    "nodes",
    [
        torch.tensor([0.0, 1.0], dtype=torch.float64),  # wrong count
        torch.tensor([0.0, 0.7, 0.5], dtype=torch.float64),  # malformed
    ],
    ids=["wrong-count", "malformed"],
)
def test_measure_degree_rejects_bad_nodes(nodes: torch.Tensor) -> None:
    method = VARIABLE_METHODS["interpolatory3_variable"]
    with pytest.raises(ValueError, match="node"):
        method.measure_degree(nodes)


def test_measure_degree_rejects_weights_exact_everywhere() -> None:
    """Weights that are exact up to degree 2C cannot be right: raise."""
    method = UNIFORM_METHODS["gk7"]
    with pytest.raises(ValueError, match="no error above"):
        method.measure_degree(tol=1.0)
