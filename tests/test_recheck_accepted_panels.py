"""Tests for the accepted-panel re-check against the drifting tolerance.

In the default *absolute* error mode the accept/reject tolerance is
``atol + rtol * |reference_integral|`` with ``reference_integral`` the running
integral. The running integral drifts as more panels are evaluated, so a panel
accepted early -- against one denominator -- can later violate the *current*
tolerance once the denominator shrinks. The re-check re-examines the recorded
(already-accepted) panels every iteration and splits any that no longer pass.

The re-check runs only on the ``take_gradient=False`` path (no per-batch
backward to double-count, and that path computes the full-integral denominator).
``take_gradient=True`` therefore serves as the "re-check disabled" baseline.

A strongly *cancelling* integrand drives the effect: its partial integral peaks
mid-domain and the total is ~0, so the absolute denominator is large mid-run and
collapses toward ``atol`` at the end. ``conserve_memory=True`` + a small
``max_batch`` make the running integral accumulate batch by batch (the regime
where the denominator genuinely drifts within a single solve).
"""

from __future__ import annotations

import math

import torch
from _helpers import REMOVE_CUT

from padaquad import adaptive_quadrature, steps

ATOL = 1e-9
RTOL = 1e-6
# bosh3 has C=4 points/panel, so 16 evals ~ 4 panels per batch: small enough
# that the running integral accumulates over several batches.
MAX_BATCH = 16
METHOD = "bosh3"

T_INIT = torch.tensor([0.0], dtype=torch.float64)
T_FINAL = torch.tensor([1.0], dtype=torch.float64)


def _solver(**kwargs):
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=METHOD,
        atol=ATOL,
        rtol=RTOL,
        remove_cut=REMOVE_CUT,
        device="cpu",
        dtype=torch.float64,
        **kwargs,
    )


def _cancelling(t, *args):
    """f(t) = sin(2*pi*t): total integral 0, partial integral peaks at t=0.5.

    The absolute denominator (atol + rtol*|running integral|) is large mid-run
    and shrinks toward atol as the second half cancels the first.
    """
    return torch.sin(2.0 * math.pi * t)


def _run(solver, f, take_gradient, mesh_final=T_FINAL, **kwargs):
    return solver.integrate(
        f=f,
        mesh_init=T_INIT,
        mesh_final=mesh_final,
        take_gradient=take_gradient,
        max_batch=MAX_BATCH,
        conserve_memory=True,
        **kwargs,
    )


def _panel_ratios(solver, result):
    """Per-panel error ratio against the FINAL integral.

    Uses the solver's own error logic (including the round-off floor), so a
    ratio < 1 means the panel is within the final tolerance.
    """
    ratios, _, _ = solver._compute_error_ratios(
        mesh_quadrature_errors=result.mesh_quadrature_errors,
        mesh_quadratures=result.mesh_quadratures,
        integral=result.integral,
    )
    return ratios


def test_recheck_enforces_final_tolerance_on_all_panels():
    """With the re-check on, every recorded panel satisfies the final tolerance.

    This is the feature's contract: at convergence no accepted panel may violate
    the tolerance computed against the final integral.
    """
    solver = _solver()
    torch.manual_seed(0)
    result = _run(solver, _cancelling, take_gradient=False)
    ratios = _panel_ratios(solver, result)
    assert torch.all(ratios < 1.0 + 1e-9), (
        f"re-check should leave every panel within the final tolerance; "
        f"max ratio = {float(ratios.max())}"
    )


def test_without_recheck_some_panel_violates_final_tolerance():
    """take_gradient=True disables the re-check, so the bug is still present.

    With the denominator drifting down as the integral cancels, at least one
    early-accepted panel violates the tolerance computed against the final
    integral -- the situation the re-check exists to fix.
    """
    solver = _solver()
    torch.manual_seed(0)
    result = _run(solver, _cancelling, take_gradient=True)
    ratios = _panel_ratios(solver, result)
    assert torch.any(ratios >= 1.0), (
        f"expected at least one panel to violate the final tolerance without the "
        f"re-check; max ratio = {float(ratios.max())}"
    )


def test_integral_consistency_and_accuracy():
    """integral == sum(mesh_quadratures) exactly; value is correct and finite."""
    solver = _solver()
    torch.manual_seed(0)
    result = _run(solver, _cancelling, take_gradient=False)
    rebuilt = result.mesh_quadratures.sum(0)
    assert torch.allclose(result.integral, rebuilt, atol=1e-12, rtol=0.0), (
        f"integral {result.integral} != sum(mesh_quadratures) {rebuilt}"
    )
    # Analytic integral of sin(2*pi*t) over [0, 1] is 0.
    assert abs(float(result.integral.reshape(-1)[0])) < 1e-5
    assert torch.isfinite(result.integral_error).all()


def test_termination_near_cancellation_with_split_cap():
    """sin over [0, 2*pi] integrates to 0 (denominator collapses to atol).

    The re-check repeatedly re-splits as the denominator shrinks; with a finite
    max_adaptive_splits the solve must still terminate and stay finite/accurate.
    """
    solver = _solver(max_adaptive_splits=15)
    torch.manual_seed(0)
    result = _run(
        solver,
        lambda t, *a: torch.sin(t),
        take_gradient=False,
        mesh_final=torch.tensor([2.0 * math.pi], dtype=torch.float64),
        max_adaptive_splits=15,
    )
    assert torch.isfinite(result.integral).all()
    assert abs(float(result.integral.reshape(-1)[0])) < 1e-3


def test_recheck_skipped_with_fixed_reference_still_valid():
    """A fixed error_integral_reference pins the absolute denominator, so the
    re-check is skipped (no drift to correct). The normal accept decision then
    already judges every panel against that same fixed tolerance, so the result
    stays finite, accurate, and within tolerance."""
    solver = _solver()
    torch.manual_seed(0)
    result = _run(
        solver, _cancelling, take_gradient=False, error_integral_reference=0.0
    )
    ratios = _panel_ratios(solver, result)
    assert torch.all(ratios < 1.0 + 1e-9)
    assert torch.isfinite(result.integral).all()
    assert abs(float(result.integral.reshape(-1)[0])) < 1e-5
