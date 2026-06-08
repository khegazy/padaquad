"""Investigative tests for the two error-indicator modes.

The parallel solver supports two error-indicator modes, controlled by
``use_absolute_error_ratio``:

  - **Absolute** (default): per-step ``error_ratio = |step_error| /
    (atol + rtol * |integral|)``. The reference is the *total* integral
    over the full domain. Every step uses the same denominator.

  - **Cumulative**: per-step ``error_ratio = |step_error| /
    (atol + rtol * |cumsum_to_step|)``. The reference is the running
    integral up to that step, mimicking traditional ODE error control
    where the error budget grows with the cumulative state magnitude.

The attached refactor plan flagged the cumulative mode as "unusual"
because it tightens (not loosens) the tolerance as the integral
shrinks: small ``cumsum`` early in the integration means a small
denominator, hence a *larger* error ratio for the same absolute step
error, hence a more easily-rejected step. Whether this is a
deliberate property of the controller or a transcription bug is
unclear without empirical investigation.

This file documents the empirical behavior of both modes in pinned
tests so any unintentional regression in either is caught. It does
not assert that one mode is correct — that is a Phase 1 follow-up
based on these findings.

Phase 0 of the quadrature alignment plan.
"""

from __future__ import annotations

import math

import torch
from tests._helpers import make_uniform_solver


def _make_synthetic_inputs(n: int, step_error: float, step_value: float, dtype):
    """Hand-crafted inputs to ``_compute_error_ratios``.

    Returns:
      ``mesh_quadratures`` of shape [N, 1] with every step contributing
      ``step_value`` (so cumsum grows linearly), and
      ``mesh_quadrature_errors`` of shape [N, 1] with every step contributing
      ``step_error`` (so per-step error is constant).
      ``integral`` is the total ``n * step_value``.
    """
    mesh_quadratures = torch.full((n, 1), step_value, dtype=dtype)
    mesh_quadrature_errors = torch.full((n, 1), step_error, dtype=dtype)
    integral = mesh_quadratures.sum(dim=0)
    return mesh_quadratures, mesh_quadrature_errors, integral


def test_absolute_mode_treats_steps_uniformly():
    """Pin the absolute-mode behavior. Constant per-step error should
    produce constant per-step error ratios because the denominator is
    the same total-integral value for every step.
    """
    solver = make_uniform_solver("dopri5", atol=1e-8, rtol=1e-8, device="cpu")
    dtype = solver.dtype
    n = 10
    step_error = 1e-9
    step_value = 0.1  # total integral = 1.0
    _mesh_quadratures, mesh_quadrature_errors, integral = _make_synthetic_inputs(
        n, step_error, step_value, dtype
    )

    solver.use_absolute_error_ratio = True
    error_ratios, _, _ = solver._compute_error_ratios(
        mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
    )

    # All ratios should be identical.
    spread = (error_ratios.max() - error_ratios.min()).item()
    assert spread < 1e-15, (
        f"absolute-mode error ratios should be identical for uniform input; "
        f"spread = {spread}"
    )

    # The numeric value: 1e-9 / max(1e-8, 1e-8 * 1.0) = 1e-9 / 1e-8 = 0.1
    expected = step_error / max(1e-8, 1e-8 * abs(integral.item()))
    assert math.isclose(error_ratios[0].item(), expected, rel_tol=1e-6), (
        f"absolute-mode: got {error_ratios[0].item()}, expected {expected}"
    )


def test_cumulative_mode_tightens_at_small_cumsum():
    """Pin the cumulative-mode behavior. The denominator
    ``max(atol, rtol * |cumsum|)`` grows once ``rtol*cumsum > atol``.
    So per-step error ratios DECREASE as the integration progresses —
    early steps (where atol dominates) share a flat tolerance, then
    ratios drop once rtol*cumsum overtakes atol.

    Use atol=5e-9, rtol=1e-8 so the crossover happens at cumsum=0.5
    (step 5 of 10): steps 1-5 are atol-clamped (flat ratios), steps
    6-10 are rtol-dominated (strictly decreasing ratios).
    """
    solver = make_uniform_solver("dopri5", atol=5e-9, rtol=1e-8, device="cpu")
    dtype = solver.dtype
    n = 10
    step_error = 1e-9
    step_value = 0.1
    mesh_quadratures, mesh_quadrature_errors, _integral = _make_synthetic_inputs(
        n, step_error, step_value, dtype
    )

    solver.use_absolute_error_ratio = False
    error_ratios, _, _ = solver._compute_error_ratios(
        mesh_quadrature_errors=mesh_quadrature_errors, mesh_quadratures=mesh_quadratures
    )

    # Cumsum grows: cumsum[0]=0.1, ..., cumsum[9]=1.0.
    # Per-step denominator: max(5e-9, 1e-8*cumsum).
    # Crossover at cumsum=0.5: steps 0-4 clamped at atol=5e-9 (flat ratios),
    # steps 5-9 rtol-dominated and strictly decreasing.
    # Step 0: max(5e-9, 1e-8*0.1) = 5e-9 → ratio = 1e-9 / 5e-9 = 0.2
    # Step 9: max(5e-9, 1e-8*1.0) = 1e-8 → ratio = 1e-9 / 1e-8 = 0.1
    diffs = error_ratios[1:] - error_ratios[:-1]
    assert torch.all(diffs <= 0), (
        f"cumulative-mode error ratios should be non-increasing as cumsum grows; "
        f"got error_ratios={error_ratios.flatten().tolist()}"
    )
    assert error_ratios[0].item() > error_ratios[-1].item(), (
        "cumulative-mode tightens tolerance at small cumsum (early steps)"
    )


def test_cumulative_mode_tolerance_is_per_step():
    """The cumulative-mode tolerance ``max(atol, rtol*|cumsum|)`` must be
    evaluated independently for each step, not as a single global max.

    Use atol=5e-9, rtol=1e-8 with 10 steps of 0.1 so that steps 0-4
    (cumsum 0.1-0.5) are atol-clamped and steps 5-9 (cumsum 0.6-1.0)
    are rtol-dominated. If the tolerance were computed globally (e.g.
    using the final cumsum for all steps), all ratios would equal
    step_error / 1e-8 = 0.1. The per-step computation produces a
    larger ratio (0.2) for the early atol-clamped steps.
    """
    atol = 5e-9
    rtol = 1e-8
    solver = make_uniform_solver("dopri5", atol=atol, rtol=rtol, device="cpu")
    dtype = solver.dtype
    n = 10
    step_error = 1e-9
    step_value = 0.1
    mesh_quadratures, mesh_quadrature_errors, _integral = _make_synthetic_inputs(
        n, step_error, step_value, dtype
    )

    solver.use_absolute_error_ratio = False
    error_ratios, _, _ = solver._compute_error_ratios(
        mesh_quadrature_errors=mesh_quadrature_errors, mesh_quadratures=mesh_quadratures
    )

    # Compute expected per-step tolerances manually.
    cumsums = torch.arange(1, n + 1, dtype=dtype) * step_value  # [0.1, 0.2, ..., 1.0]
    expected_tols = torch.maximum(
        torch.tensor(atol, dtype=dtype), torch.tensor(rtol, dtype=dtype) * cumsums
    )
    expected_ratios = torch.tensor(step_error, dtype=dtype) / expected_tols

    assert torch.allclose(error_ratios, expected_ratios, rtol=1e-6), (
        f"per-step ratios mismatch:\n  got      {error_ratios.tolist()}\n"
        f"  expected {expected_ratios.tolist()}"
    )
    # Sanity: early steps (atol-clamped) must differ from late steps (rtol-dominated).
    assert not torch.allclose(error_ratios[:5], error_ratios[5:]), (
        "early (atol-clamped) and late (rtol-dominated) ratios should differ; "
        "tolerance may have been applied globally rather than per step"
    )


def test_modes_agree_when_cumsum_equals_total_at_last_step():
    """At the LAST step, ``cumsum == total integral``, so the cumulative
    mode's denominator equals the absolute mode's denominator. Both
    modes should produce the same error ratio at that step.

    Pinning this anchors the relationship between the two modes.
    """
    solver = make_uniform_solver("dopri5", atol=1e-8, rtol=1e-8, device="cpu")
    dtype = solver.dtype
    n = 5
    step_error = 1e-9
    step_value = 0.1
    mesh_quadratures, mesh_quadrature_errors, integral = _make_synthetic_inputs(
        n, step_error, step_value, dtype
    )

    solver.use_absolute_error_ratio = True
    abs_ratios, _, _ = solver._compute_error_ratios(
        mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
    )
    solver.use_absolute_error_ratio = False
    cum_ratios, _, _ = solver._compute_error_ratios(
        mesh_quadrature_errors=mesh_quadrature_errors, mesh_quadratures=mesh_quadratures
    )

    # Last step: cumsum[-1] == integral, so denominators match exactly.
    assert math.isclose(abs_ratios[-1].item(), cum_ratios[-1].item(), rel_tol=1e-12), (
        f"absolute-last={abs_ratios[-1].item()}, "
        f"cumulative-last={cum_ratios[-1].item()}; should agree at last step."
    )
