"""Tests for the ``tracked_variables`` feature.

The integrand ``f`` may optionally return a 2-tuple
``(integrand, tracked_variables)`` where ``tracked_variables`` is a tuple of
tensors (or ``None``). Tracked variables are evaluated at every node but NOT
integrated; they are returned at the accepted nodes in
``result.tracked_variables``, aligned with ``result.nodes`` / ``result.y``.

These tests pin:
  * the backward-compatible contract (bare tensor -> ``None``),
  * per-node alignment and exact value preservation through the adaptive loop,
  * a single-tensor tracked output being wrapped into a 1-tuple,
  * multi-dimensional trailing shapes surviving the reshape,
  * non-float (integer) tracked variables across multiple record batches
    (exercising the dtype-aware sorted insert + sort), and
  * float32 / float64 handling.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests._helpers import (
    ATOL_MED,
    REMOVE_CUT,
    RTOL_MED,
    T_FINAL,
    T_INIT,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    make_uniform_solver,
)

from padaquad import adaptive_quadrature, steps

# (sampling_type, method) cases covering both uniform and variable solvers.
SOLVER_CASES = [
    (steps.ADAPTIVE_UNIFORM, "gk21"),
    (steps.ADAPTIVE_UNIFORM, "bosh3"),
    (steps.ADAPTIVE_VARIABLE, "adaptive_heun"),
]
SOLVER_IDS = ["uniform-gk21", "uniform-bosh3", "variable-adaptive_heun"]


def _make(sampling, method, atol=ATOL_MED, rtol=RTOL_MED):
    return adaptive_quadrature(
        sampling_type=sampling,
        method=method,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
    )


@pytest.mark.parametrize(("sampling", "method"), SOLVER_CASES, ids=SOLVER_IDS)
@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_basic_tracked_variables(sampling, method, take_gradient):
    """Two tracked variables are returned, per-node aligned and value-exact."""

    def f(t):
        return torch.sin(t), (t**2, torch.cos(t))

    solver = _make(sampling, method)
    result = solver.integrate(
        f=f,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=take_gradient,
    )

    tv = result.tracked_variables
    assert isinstance(tv, tuple)
    assert len(tv) == 2

    # tracked_variables are flattened to align with nodes/y on the leading
    # (P = total node) axis.
    P = result.nodes.shape[0]
    assert tv[0].shape[0] == P
    assert tv[1].shape[0] == P

    # Tracked values are evaluated at exactly the accepted nodes and never
    # recomputed/weighted, so they equal f's tracked outputs at result.nodes.
    assert torch.allclose(tv[0], result.nodes**2)
    assert torch.allclose(tv[1], torch.cos(result.nodes))
    assert torch.isfinite(tv[0]).all()

    # Tracked variables are diagnostic-only (detached).
    assert not tv[0].requires_grad
    assert not tv[1].requires_grad

    # The integral itself is unchanged: ∫_0^1 sin(t) dt = 1 - cos(1).
    expected = 1.0 - math.cos(1.0)
    assert abs(result.integral.item() - expected) < 1e-6


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_bare_tensor_has_no_tracked_variables(take_gradient):
    """The existing contract (f returns a bare tensor) yields None."""
    solver = make_uniform_solver("gk21", atol=ATOL_MED, rtol=RTOL_MED)
    result = solver.integrate(
        f=torch.sin,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=take_gradient,
    )
    assert result.tracked_variables is None


def test_single_tensor_tracked_is_wrapped():
    """A bare-tensor tracked output is normalized into a 1-tuple."""

    def f(t):
        return torch.sin(t), t**2

    solver = make_uniform_solver("gk21", atol=ATOL_MED, rtol=RTOL_MED)
    result = solver.integrate(
        f=f,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
    )
    assert isinstance(result.tracked_variables, tuple)
    assert len(result.tracked_variables) == 1
    assert torch.allclose(result.tracked_variables[0], result.nodes**2)


def test_multidim_trailing_shape_preserved():
    """A tracked variable with trailing shape [3] keeps that shape."""

    def f(t):
        tracked = torch.cat([t, t**2, t**3], dim=-1)  # [N*C, 3]
        return torch.sin(t), (tracked,)

    solver = make_uniform_solver("gk21", atol=ATOL_MED, rtol=RTOL_MED)
    result = solver.integrate(
        f=f,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
    )
    tv = result.tracked_variables[0]
    P = result.nodes.shape[0]
    assert tv.shape == (P, 3)
    expected = torch.cat([result.nodes, result.nodes**2, result.nodes**3], dim=-1)
    assert torch.allclose(tv, expected)


def test_non_float_tracked_across_multiple_batches():
    """Integer tracked variables survive multi-batch record insert + sort.

    A small ``max_batch`` forces the integration loop to process panels over
    several batches, exercising the sorted-insert / sort paths on a non-float
    (int64) tracked variable.
    """

    def f(t):
        labels = (t > 0.5).to(torch.int64)
        return torch.sin(t), (labels,)

    solver = make_uniform_solver("bosh3", atol=1e-8, rtol=1e-8)
    result = solver.integrate(
        f=f,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
        max_batch=solver.C * 2,
    )

    tv = result.tracked_variables[0]
    assert tv.dtype == torch.int64
    assert tv.shape[0] == result.nodes.shape[0]
    expected = (result.nodes > 0.5).to(torch.int64)
    assert torch.equal(tv, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tracked_variables_dtype(dtype):
    """Tracked variables follow the integration dtype (float32 / float64)."""

    def f(t):
        return torch.sin(t), (t**2,)

    solver = make_uniform_solver("gk21", atol=1e-5, rtol=1e-5)
    result = solver.integrate(
        f=f,
        mesh_init=torch.tensor([0.0], dtype=dtype),
        mesh_final=torch.tensor([1.0], dtype=dtype),
        take_gradient=False,
    )
    tv = result.tracked_variables[0]
    assert tv.dtype == dtype
    assert torch.allclose(tv, result.nodes**2)
