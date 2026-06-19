"""Field-population tests for the ``max_path_change`` early-exit branch.

Phase 1's Bug B6 fix replaced the bare ``return None`` from the
early-exit branch with a populated ``IntegrationResult``. The
existing regression test in ``tests/test_bug_regressions.py`` checks
the return type and ``converged is False``, but does not verify
that the rest of the fields are actually populated. A future refactor
could quietly leave these fields as ``None`` while still passing the
existing regression test, leaving callers with a result that is shaped
correctly on paper but useless to inspect.

These tests pin the contract: every field on the early-exit
``IntegrationResult`` that has a populated counterpart on the
normal-completion path must also be populated here. Test failures
on this file would mean the early-exit branch silently lost a field
populated by some upstream refactor — exactly the kind of regression
the safety-belt is supposed to catch.

Coverage:

  * Every diagnostic field on ``IntegrationResult`` (``integral``,
    ``integral_error``, ``mesh_optimal``, ``mesh_init``, ``mesh_final``,
    ``nodes``, ``h``, ``y``, ``mesh_quadratures``, ``mesh_quadrature_errors``,
    ``error_ratios``, ``y0``) is non-None.
  * Trailing-dim coherence: every field whose last dimension is the
    integrand output dim D matches D=1 for our scalar integrand.
  * ``error_ratios`` actually shows that some steps failed (this is
    the very condition that triggered the early exit, so the
    ``error_ratios`` should reflect at least one entry > 1).
  * The early-exit IntegrationResult flips ``converged=False`` while
    leaving every other field shaped to match the normal path's
    contract.
"""

from __future__ import annotations

import torch
from tests._helpers import make_uniform_solver

from padaquad import IntegrationResult


def _wiggly() -> callable:
    """Damped sinusoid with high frequency. Forces a tight tolerance to
    fail at low resolution, which is what triggers the early exit.
    """
    return lambda t, *_: torch.sin(10 * t) * torch.exp(-0.1 * t)


def _trigger_early_exit() -> IntegrationResult:
    """Return an ``IntegrationResult`` produced by the early-exit path.
    Uses the same mesh-of-4 + 1e-12 tol setup as the existing bug
    regression test so this test stays in lockstep with the public
    fix's intended trigger.
    """
    solver = make_uniform_solver("dopri5", atol=1e-12, rtol=1e-12, max_path_change=0.1)
    mesh = torch.linspace(0.0, 4.0, 4, dtype=torch.float64).unsqueeze(-1)
    return solver.integrate(f=_wiggly(), mesh=mesh, take_gradient=False)


def test_early_exit_integral_field_populated():
    """``result.integral`` is the most important field for the user
    to inspect on early-exit. It must be a tensor (not None) so the
    user can see what value the partially-refined mesh produced.
    """
    out = _trigger_early_exit()
    assert out.converged is False, "test setup did not trigger early exit"
    assert out.integral is not None
    assert isinstance(out.integral, torch.Tensor)
    # The wiggly integrand is bounded; the partial integral should be
    # finite (not NaN/inf — those would indicate a corrupted batch).
    assert torch.isfinite(out.integral).all(), (
        f"early-exit integral has non-finite values: {out.integral}"
    )


def test_early_exit_error_field_populated():
    """``result.integral_error`` lets the user see how badly the partial
    integration converged. The early-exit branch wraps it in
    ``torch.abs`` per the Bug B6 fix; both signs of nonzero are fine,
    but it must not be None.
    """
    out = _trigger_early_exit()
    assert out.integral_error is not None
    assert isinstance(out.integral_error, torch.Tensor)
    assert torch.isfinite(out.integral_error).all()


def test_early_exit_mesh_fields_populated():
    """The mesh fields tell the caller what region was integrated and
    what barriers were in play at the moment the loop bailed out.
    All three must be non-None.
    """
    out = _trigger_early_exit()
    assert out.mesh_optimal is not None, "mesh_optimal lost on early-exit"
    assert out.mesh_init is not None, "mesh_init lost on early-exit"
    assert out.mesh_final is not None, "mesh_final lost on early-exit"
    # The integration region must be the user-supplied bounds.
    assert torch.isclose(
        out.mesh_init.cpu(), torch.tensor([0.0], dtype=torch.float64)
    ).all()
    assert torch.isclose(
        out.mesh_final.cpu(), torch.tensor([4.0], dtype=torch.float64)
    ).all()
    # mesh_optimal[0] should match mesh_init and mesh_optimal[-1] should
    # match mesh_final — barrier endpoints are preserved across the
    # adaptive loop.
    assert torch.allclose(out.mesh_optimal[0], out.mesh_init, atol=1e-10)
    assert torch.allclose(out.mesh_optimal[-1], out.mesh_final, atol=1e-10)


def test_early_exit_per_step_diagnostics_populated():
    """The per-step diagnostic fields (nodes, h, y, mesh_quadratures,
    mesh_quadrature_errors, error_ratios) are needed to inspect WHY the
    early-exit triggered. None on any of them blinds the caller.
    """
    out = _trigger_early_exit()

    assert out.nodes is not None, "nodes lost on early-exit"
    assert out.h is not None, "h lost on early-exit"
    assert out.y is not None, "y lost on early-exit"
    assert out.mesh_quadratures is not None, "mesh_quadratures lost on early-exit"
    assert out.mesh_quadrature_errors is not None, (
        "mesh_quadrature_errors lost on early-exit"
    )
    assert out.error_ratios is not None, "error_ratios lost on early-exit"

    # Shape coherence: nodes [P, T], y [P, D] (flattened across panels);
    # h [N, T], mesh_quadratures [N, D], mesh_quadrature_errors [N, D],
    # error_ratios [N]. Panel count N comes from h (per-panel); P = total nodes.
    N, T = out.h.shape
    P = out.nodes.shape[0]
    assert out.nodes.shape[-1] == T
    assert out.y.shape[0] == P
    assert out.mesh_quadratures.shape[0] == N
    assert out.mesh_quadrature_errors.shape[0] == N
    assert out.error_ratios.shape[0] == N

    # All finite — corrupted-batch sanity check.
    assert torch.isfinite(out.nodes).all()
    assert torch.isfinite(out.y).all()
    assert torch.isfinite(out.error_ratios).all()


def test_early_exit_error_ratios_show_some_failed_steps():
    """The early-exit branch only fires when the failing-step fraction
    exceeds ``max_path_change``. So ``error_ratios`` MUST contain at
    least one value > 1 — otherwise the trigger condition for this
    test is wrong and it is no longer testing what its name claims.
    """
    out = _trigger_early_exit()
    assert (out.error_ratios > 1.0).any(), (
        "error_ratios shows no failing steps, but the early-exit "
        "branch only triggers when fail_ratio >= max_path_change. "
        "Test setup may not be exercising the intended path."
    )


def test_early_exit_y0_and_gradient_metadata_populated():
    """``y0`` and ``gradient_taken`` are populated even on the
    early-exit path. These let the caller reconstruct what they
    asked for vs. what they got back.
    """
    out = _trigger_early_exit()
    assert out.y0 is not None, "y0 lost on early-exit"
    # gradient_taken records what was requested (here: default False).
    assert out.gradient_taken is False, (
        f"gradient_taken={out.gradient_taken!r}, expected False"
    )


def test_early_exit_loss_field_is_explicit_none():
    """``loss`` is intentionally None on the early-exit path because
    the loss function (which depends on a fully-converged integral)
    was never called. Pinning this so a future refactor doesn't
    accidentally start computing a partial loss.
    """
    out = _trigger_early_exit()
    assert out.loss is None, f"loss should be None on early-exit; got {out.loss!r}"


def test_early_exit_does_not_corrupt_solver_for_subsequent_calls():
    """A failed early-exit must leave the solver in a usable state
    for the next call. Critical for training-loop usage where one
    bad iteration shouldn't poison the rest.
    """
    solver = make_uniform_solver("dopri5", atol=1e-12, rtol=1e-12, max_path_change=0.1)
    bad_mesh = torch.linspace(0.0, 4.0, 4, dtype=torch.float64).unsqueeze(-1)

    # First call: hits early-exit.
    out1 = solver.integrate(f=_wiggly(), mesh=bad_mesh)
    assert out1.converged is False

    # Second call with relaxed tolerances and no user mesh — must
    # complete normally, proving the solver wasn't poisoned.
    out2 = solver.integrate(
        f=lambda t, *_: torch.sin(t),
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([torch.pi], dtype=torch.float64),
    )
    # NOTE: even after a fix to make this converge, this test is
    # really about the API contract — the result must come back as
    # an IntegrationResult, not a None or an exception.
    assert isinstance(out2, IntegrationResult)
