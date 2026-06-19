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

import pytest
import torch
from _helpers import REMOVE_CUT, make_solver_for_unit_test
from scipy.integrate import quad

from padaquad import adaptive_quadrature, steps
from padaquad.results import IntegrationResult

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


def _all_panels_pass(solver, result):
    """Per-panel keep mask against the FINAL integral, scheme-agnostic.

    Uses the solver's own accept/reject logic (norm family OR failure_fraction,
    absolute OR cumulative), so ``keep.all()`` is the feature's contract: at
    convergence every recorded panel is accepted against the final tolerance.
    """
    ratios, _, per_dim = solver._compute_error_ratios(
        mesh_quadrature_errors=result.mesh_quadrature_errors,
        mesh_quadratures=result.mesh_quadratures,
        integral=result.integral,
    )
    keep, _ = solver._accept_reject_masks(ratios, per_dim)
    return keep


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


# ---------------------------------------------------------------------------
# A. Output dimensionality & error-norm schemes
# ---------------------------------------------------------------------------


def _vector_cancelling(t, *args):
    """[N, 2] cancelling integrand. Component 0: sin(2πt) (∫=0 over [0,1]).
    Component 1: exp(2t)·sin(6πt) (nonzero ∫, oscillating with growth). Both
    make the absolute denominator drift, exercising the re-check per component.
    """
    c0 = torch.sin(2.0 * math.pi * t)
    c1 = torch.exp(2.0 * t) * torch.sin(6.0 * math.pi * t)
    return torch.cat([c0, c1], dim=-1)


def _vector_truth():
    """Reference ∫ of _vector_cancelling over [0, 1] (scipy for component 1)."""
    i1, _ = quad(lambda t: math.exp(2.0 * t) * math.sin(6.0 * math.pi * t), 0.0, 1.0)
    return torch.tensor([0.0, i1], dtype=torch.float64)


def test_recheck_vector_integrand():
    """D>1 with the default L2 norm: shape, per-panel contract, accuracy."""
    solver = _solver()
    torch.manual_seed(0)
    result = _run(solver, _vector_cancelling, take_gradient=False)
    assert result.integral.shape == (2,)
    assert torch.all(torch.isfinite(result.integral))
    assert torch.all(_all_panels_pass(solver, result))
    assert torch.allclose(result.integral.cpu(), _vector_truth(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mesh_failure_tolerance", [0.0, 0.5])
def test_recheck_failure_fraction_mode(mesh_failure_tolerance):
    """failure_fraction scheme (per-component accept) with the re-check.

    Terminates, integral is finite/accurate, and every recorded panel passes
    the per-component criterion against the final integral.
    """
    solver = _solver(
        error_norm="failure_fraction",
        mesh_failure_tolerance=mesh_failure_tolerance,
    )
    torch.manual_seed(0)
    result = _run(solver, _vector_cancelling, take_gradient=False)
    assert result.integral.shape == (2,)
    assert torch.all(torch.isfinite(result.integral))
    assert torch.all(_all_panels_pass(solver, result))
    # Loose accuracy: with a 50% failure tolerance a component may be coarser.
    assert torch.allclose(result.integral.cpu(), _vector_truth(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("error_norm", ["max", "rms"])
def test_recheck_error_norm_max_rms(error_norm):
    """The re-check reuses _reduce_norm, so it must work for every norm."""
    solver = _solver(error_norm=error_norm)
    torch.manual_seed(0)
    result = _run(solver, _vector_cancelling, take_gradient=False)
    assert torch.all(torch.isfinite(result.integral))
    assert torch.all(_all_panels_pass(solver, result))
    assert torch.allclose(result.integral.cpu(), _vector_truth(), atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# B. Gradient correctness through the re-check
# ---------------------------------------------------------------------------


def _grad_integrand(t):
    """g(t) = exp(2t)·sin(6πt): oscillating with growth (drives the re-check),
    nonzero integral (so the gradient check is non-trivial)."""
    return torch.exp(2.0 * t) * torch.sin(6.0 * math.pi * t)


def test_recheck_preserves_gradient():
    """∇θ ∫ θ·g dt (through the re-check) equals ∫ g dt (autodiff consistency).

    Verifies the ``record["integral"] -= removed`` subtraction keeps the
    autograd graph correct after panels are removed and their children
    re-evaluated.
    """
    theta = torch.tensor([1.7], dtype=torch.float64, requires_grad=True)

    solver_a = _solver()
    torch.manual_seed(0)
    res_a = _run(solver_a, lambda t, *a: theta * _grad_integrand(t), take_gradient=False)
    res_a.integral.sum().backward()
    grad = float(theta.grad.reshape(-1)[0])

    solver_b = _solver()
    torch.manual_seed(0)
    res_b = _run(solver_b, lambda t, *a: _grad_integrand(t), take_gradient=False)
    integral_g = float(res_b.integral.reshape(-1)[0])

    assert abs(grad - integral_g) < 1e-5, (
        f"d/dtheta of integral(theta*g) = {grad}, but integral(g) = {integral_g}"
    )


# ---------------------------------------------------------------------------
# C. Auxiliary outputs survive panel removal
# ---------------------------------------------------------------------------


def test_recheck_filters_tracked_variables():
    """Tracked variables stay aligned with nodes after panels are removed."""

    def f_tracked(t, *args):
        return torch.sin(2.0 * math.pi * t), (t**2,)

    solver = _solver()
    torch.manual_seed(0)
    result = _run(solver, f_tracked, take_gradient=False)
    assert result.tracked_variables is not None
    assert len(result.tracked_variables) == 1
    tv = result.tracked_variables[0]
    assert tv.shape[0] == result.nodes.shape[0]
    assert torch.allclose(tv, result.nodes**2)


def test_recheck_with_y0_and_f_args():
    """y0 offset and f_args forwarding remain correct with the re-check active."""

    def f(t, scale, shift):
        return scale * torch.sin(2.0 * math.pi * t) + shift

    y0 = torch.tensor([3.0], dtype=torch.float64)
    solver = _solver()
    torch.manual_seed(0)
    result = _run(solver, f, take_gradient=False, f_args=(2.0, 0.0), y0=y0)
    # ∫ 2 sin(2πt) dt over [0, 1] = 0, so integral == y0.
    assert abs(float(result.integral.reshape(-1)[0]) - 3.0) < 1e-5
    assert torch.allclose(result.y0.cpu(), y0)


# ---------------------------------------------------------------------------
# D. Mode / control-flow edges
# ---------------------------------------------------------------------------


def test_recheck_cumulative_mode():
    """Cumulative-mode re-check: terminates, accurate, all panels within the
    final cumulative tolerance."""
    solver = _solver(use_absolute_error_ratio=False)
    torch.manual_seed(0)
    result = _run(solver, _cancelling, take_gradient=False)
    assert torch.all(torch.isfinite(result.integral))
    assert abs(float(result.integral.reshape(-1)[0])) < 1e-4
    assert torch.all(_all_panels_pass(solver, result))


def test_recheck_respects_split_cap():
    """Heavy cancellation with a small split cap still terminates with a finite,
    accurate integral and a finite optimal mesh."""
    solver = _solver(max_adaptive_splits=8)
    torch.manual_seed(0)
    result = _run(
        solver,
        lambda t, *a: torch.sin(t),
        take_gradient=False,
        mesh_final=torch.tensor([2.0 * math.pi], dtype=torch.float64),
        max_adaptive_splits=8,
    )
    assert torch.all(torch.isfinite(result.integral))
    assert abs(float(result.integral.reshape(-1)[0])) < 1e-2
    assert torch.all(torch.isfinite(result.mesh_optimal))


def test_recheck_actually_resplits_accepted_panels():
    """Directly prove the re-check removes & re-splits ALREADY-ACCEPTED panels.

    The other tests assert the end state (all panels within the final
    tolerance); this one instruments the live solver to confirm the *mechanism*
    engages. ``_remove_failed_record_panels`` is invoked only to drop recorded
    (already-accepted) panels, and ``(~record_keep_mask).sum()`` counts how many
    are removed (and thus fed to the splitter) each iteration.
    """
    solver = _solver()
    removed_total = 0
    orig_remove = solver._remove_failed_record_panels

    def counting_remove(record, record_keep_mask, loss_fxn):
        nonlocal removed_total
        removed_total += int((~record_keep_mask).sum())
        return orig_remove(record, record_keep_mask, loss_fxn)

    solver._remove_failed_record_panels = counting_remove
    torch.manual_seed(0)
    result = _run(solver, _cancelling, take_gradient=False)

    assert removed_total > 0, (
        "the re-check never removed/re-split an accepted panel; the feature did "
        "not engage on the cancelling integrand"
    )
    # The mechanism fired AND the end state is still within tolerance.
    assert torch.all(_all_panels_pass(solver, result))


def test_recheck_determinism():
    """Same seed => identical integral and panel count."""
    s1 = _solver()
    torch.manual_seed(0)
    r1 = _run(s1, _cancelling, take_gradient=False)
    s2 = _solver()
    torch.manual_seed(0)
    r2 = _run(s2, _cancelling, take_gradient=False)
    assert torch.equal(r1.integral, r2.integral)
    assert r1.h.shape[0] == r2.h.shape[0]


# ---------------------------------------------------------------------------
# E. Unit tests of _remove_failed_record_panels (record-emptying regression)
# ---------------------------------------------------------------------------


def _make_record(solver, n_panels=4):
    """Build an initialized record (via _record_results) with known per-step
    quadratures/errors and integral == sum(quadratures)."""
    boundaries = torch.linspace(0.0, 1.0, n_panels + 1, dtype=torch.float64)
    t_left, t_right = boundaries[:-1], boundaries[1:]
    C = 4
    c = torch.linspace(0, 1, C, dtype=torch.float64)
    nodes = t_left[:, None, None] + c[None, :, None] * (t_right - t_left)[:, None, None]
    h = (t_right - t_left).unsqueeze(-1)
    quad_vals = torch.arange(1, n_panels + 1, dtype=torch.float64).unsqueeze(-1) * 0.1
    err_vals = torch.arange(1, n_panels + 1, dtype=torch.float64).unsqueeze(-1) * 0.01
    results = IntegrationResult(
        integral=quad_vals.sum(0),
        integral_error=err_vals.sum(0),
        nodes=nodes,
        h=h,
        y=torch.ones(n_panels, C, 1, dtype=torch.float64),
        mesh_quadratures=quad_vals,
        mesh_quadrature_errors=err_vals,
        error_ratios=torch.full((n_panels,), 0.5, dtype=torch.float64),
        loss=quad_vals.sum(0),
    )
    mesh = torch.cat(
        [nodes[:, 0, :], torch.tensor([[1.0]], dtype=torch.float64)], dim=0
    )
    mi = solver._get_mesh_indices(mesh)
    record = solver._record_results({}, False, results, mi)
    return record, quad_vals, err_vals


def test_remove_failed_record_panels_subset():
    """Subset removal: exact integral rebuild, abs-error subtraction, loss, and
    per-step filtering."""
    solver = make_solver_for_unit_test()
    record, quad_vals, err_vals = _make_record(solver)
    orig_nodes = record["nodes"].clone()
    orig_err_total = record["integral_error"].clone()
    keep = torch.tensor([True, False, True, False])

    record = solver._remove_failed_record_panels(record, keep, solver._integral_loss)

    assert record["mesh_quadratures"].shape[0] == 2
    assert torch.allclose(record["integral"], quad_vals[keep].sum(0))
    # integral_error subtracts the removed panels' (abs) errors.
    assert torch.allclose(record["integral_error"], orig_err_total - err_vals[~keep].sum(0))
    # Default loss tracks the rebuilt integral (regression for the bound-method
    # identity bug: obj.m is obj.m is False).
    assert torch.allclose(record["loss"], record["integral"])
    assert torch.allclose(record["nodes"], orig_nodes[keep])
    assert record["error_ratios"].shape[0] == 2


def test_remove_all_then_record_does_not_crash():
    """Removing every recorded panel empties the per-step fields; the next
    _record_results must still merge a fresh batch without crashing (the
    record-emptying regression)."""
    solver = make_solver_for_unit_test()
    record, _, _ = _make_record(solver)
    keep = torch.zeros(4, dtype=torch.bool)

    record = solver._remove_failed_record_panels(record, keep, solver._integral_loss)
    assert record["mesh_quadratures"].shape[0] == 0
    assert record["nodes"].shape[0] == 0
    assert torch.allclose(record["integral"], torch.zeros_like(record["integral"]))

    # Fresh single-panel batch spanning [0, 1].
    C = 4
    c = torch.linspace(0, 1, C, dtype=torch.float64)
    fresh_nodes = (c.reshape(1, C, 1)).to(torch.float64)
    fresh = IntegrationResult(
        integral=torch.tensor([0.5], dtype=torch.float64),
        integral_error=torch.tensor([0.01], dtype=torch.float64),
        nodes=fresh_nodes,
        h=torch.tensor([[1.0]], dtype=torch.float64),
        y=torch.ones(1, C, 1, dtype=torch.float64),
        mesh_quadratures=torch.tensor([[0.5]], dtype=torch.float64),
        mesh_quadrature_errors=torch.tensor([[0.01]], dtype=torch.float64),
        error_ratios=torch.tensor([0.5], dtype=torch.float64),
        loss=torch.tensor([0.5], dtype=torch.float64),
    )
    mi_fresh = solver._get_mesh_indices(
        torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    )
    record = solver._record_results(record, False, fresh, mi_fresh)
    assert record["nodes"].shape[0] == 1
    assert torch.allclose(record["mesh_quadratures"], fresh.mesh_quadratures)
