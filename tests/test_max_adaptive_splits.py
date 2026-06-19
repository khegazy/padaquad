"""Tests for the ``max_adaptive_splits`` depth cap.

The adaptive loop splits any panel that fails the error tolerance into two
half-width sub-panels. ``max_adaptive_splits`` caps how many times a panel may
be split: a panel that has been split this many times is accepted even if it
still fails tolerance. ``None`` (default) means uncapped. It can be set at
construction or per integrate() call, with the call value taking priority.

Deterministic invariant exploited here: with an explicit initial mesh of ``K``
panels and a tolerance so tight that every (wide) panel fails, each panel is
split exactly ``n = max_adaptive_splits`` times, so the result has exactly
``K * 2**n`` panels. Panels at the cap are accepted unconditionally, so the
count is robust regardless of method order.

Every test runs under both ``take_gradient`` values: the cap lives in the
shared ``_adaptively_increase_mesh``, but the two modes reach it via different
evaluation paths (``_evaluate_f_on_full_nodes`` vs ``_evaluate_f_on_split_nodes``).
"""

from __future__ import annotations

import math

import pytest
import torch
from tests._helpers import (
    REMOVE_CUT,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
)

from padaquad import adaptive_quadrature, integrate, steps

# Tolerance so tight every wide panel fails -> panels always split until the cap.
# 1e-100 (not 1e-300) keeps the error ratio well clear of float64 overflow.
ATOL_FAIL = 1e-100
RTOL_FAIL = 1e-100

# Large batch so all panels evaluate together (deterministic counts in both
# take_gradient modes).
BIG_BATCH = 100_000

SOLVER_CASES = [
    (steps.ADAPTIVE_UNIFORM, "bosh3"),
    (steps.ADAPTIVE_UNIFORM, "gk21"),
    (steps.ADAPTIVE_VARIABLE, "adaptive_heun"),
]
SOLVER_IDS = ["uniform-bosh3", "uniform-gk21", "variable-adaptive_heun"]


def _mesh(n_panels: int) -> torch.Tensor:
    """Explicit uniform mesh of ``n_panels`` panels on [0, 1]. Shape [n+1, 1]."""
    return torch.linspace(0.0, 1.0, n_panels + 1, dtype=torch.float64).unsqueeze(-1)


def _hard_f(t: torch.Tensor) -> torch.Tensor:
    """A non-trivially-curved integrand so even wide panels fail the tolerance."""
    return torch.sin(30.0 * t)


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize(("sampling", "method"), SOLVER_CASES, ids=SOLVER_IDS)
@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_panel_count_equals_K_times_2_pow_n(sampling, method, n, take_gradient):
    """With a fail-everywhere tolerance, panels = K * 2**max_adaptive_splits."""
    K = 2
    solver = adaptive_quadrature(
        sampling_type=sampling,
        method=method,
        atol=ATOL_FAIL,
        rtol=RTOL_FAIL,
        remove_cut=REMOVE_CUT,
        max_adaptive_splits=n,
    )
    result = solver.integrate(
        f=_hard_f,
        mesh=_mesh(K),
        take_gradient=take_gradient,
        max_batch=BIG_BATCH,
    )
    assert result.h.shape[0] == K * 2**n


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_integrate_arg_takes_priority_over_construction(take_gradient):
    """A solver built with cap=5 but called with cap=1 behaves as cap=1."""
    K = 2
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="bosh3",
        atol=ATOL_FAIL,
        rtol=RTOL_FAIL,
        remove_cut=REMOVE_CUT,
        max_adaptive_splits=5,
    )
    result = solver.integrate(
        f=_hard_f,
        mesh=_mesh(K),
        take_gradient=take_gradient,
        max_batch=BIG_BATCH,
        max_adaptive_splits=1,
    )
    assert result.h.shape[0] == K * 2**1


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_construction_value_used_when_no_call_arg(take_gradient):
    """The construction cap is used when integrate() is called without one."""
    K = 2
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="bosh3",
        atol=ATOL_FAIL,
        rtol=RTOL_FAIL,
        remove_cut=REMOVE_CUT,
        max_adaptive_splits=1,
    )
    result = solver.integrate(
        f=_hard_f,
        mesh=_mesh(K),
        take_gradient=take_gradient,
        max_batch=BIG_BATCH,
    )
    assert result.h.shape[0] == K * 2**1


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_call_arg_used_when_no_construction_value(take_gradient):
    """The integrate() cap applies when construction left it as None."""
    K = 2
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="bosh3",
        atol=ATOL_FAIL,
        rtol=RTOL_FAIL,
        remove_cut=REMOVE_CUT,
    )
    result = solver.integrate(
        f=_hard_f,
        mesh=_mesh(K),
        take_gradient=take_gradient,
        max_batch=BIG_BATCH,
        max_adaptive_splits=2,
    )
    assert result.h.shape[0] == K * 2**2


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_uncapped_default_converges_correctly(take_gradient):
    """Default (None) is a no-op: a smooth integrand still converges accurately."""
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        remove_cut=REMOVE_CUT,
    )
    result = solver.integrate(
        f=torch.sin,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([math.pi], dtype=torch.float64),
        take_gradient=take_gradient,
    )
    assert abs(result.integral.item() - 2.0) < 1e-6


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_high_cap_does_not_interfere_via_free_function(take_gradient):
    """A cap that is never reached leaves the integral accurate.

    Also exercises the free ``integrate()`` function's max_adaptive_splits path.
    """
    result = integrate(
        f=torch.sin,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([math.pi], dtype=torch.float64),
        take_gradient=take_gradient,
        max_adaptive_splits=50,
    )
    assert abs(result.integral.item() - 2.0) < 1e-6


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_free_function_cap_panel_count(take_gradient):
    """The free integrate() honors the cap (deterministic panel count)."""
    K = 3
    result = integrate(
        f=_hard_f,
        method="bosh3",
        sampling="uniform",
        atol=ATOL_FAIL,
        rtol=RTOL_FAIL,
        mesh=_mesh(K),
        take_gradient=take_gradient,
        max_batch=BIG_BATCH,
        max_adaptive_splits=1,
    )
    assert result.h.shape[0] == K * 2**1
