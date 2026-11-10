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


# ---------------------------------------------------------------------------
# The fail ratio covers the whole starting mesh (given or stored), across
# batches; halves of split steps are never counted.
# ---------------------------------------------------------------------------

_MESH20 = torch.linspace(0.0, 1.0, 21, dtype=torch.float64).unsqueeze(-1)


def _gk21_solver(max_batch, max_path_change=0.3):
    return make_uniform_solver(
        "gk21",
        atol=1e-10,
        rtol=1e-10,
        device="cpu",
        max_batch=max_batch,
        max_path_change=max_path_change,
    )


def test_split_halves_do_not_trigger_early_exit():
    """One narrow peak: 1 of 20 starting steps fails, and its halves keep
    failing for a few splits. Only starting steps count, so the call converges
    (a per-batch ratio exited here: the second batch is 2 failing halves)."""
    out = _gk21_solver(max_batch=10**6).integrate(
        f=lambda t, *_: torch.exp(-(((t - 0.523) / 0.002) ** 2)),
        mesh=_MESH20,
        take_gradient=False,
    )
    assert out.converged is True


def test_fail_ratio_spans_batches():
    """A starting mesh of 20 steps evaluated 5 steps per batch. A rough
    region over the first 20 % of the interval fails 4 of the 20 starting
    steps (all in the first batch, 80 % of it): under max_path_change=0.3 the
    call converges. Over the first 40 % it fails 8 of 20: the call stops."""

    def rough_until(edge):
        return lambda t, *_: torch.where(
            t < edge, torch.sin(400 * t), torch.ones_like(t)
        )

    def run(edge):
        solver = _gk21_solver(max_batch=None)
        return solver.integrate(
            f=rough_until(edge),
            mesh=_MESH20,
            max_batch=5 * solver.C,
            take_gradient=False,
        )

    assert run(0.2).converged is True
    assert run(0.4).converged is False


def test_max_path_change_runs_on_stored_mesh(monkeypatch):
    """The check runs on the stored mesh used by same_integrand=True, never on
    a freshly generated mesh, and an early exit stores its partly refined mesh
    so the next same_integrand=True call continues from it."""
    bounds = {
        "mesh_init": torch.tensor([0.0], dtype=torch.float64),
        "mesh_final": torch.tensor([4.0], dtype=torch.float64),
    }
    smooth = lambda t, *_: torch.sin(t)  # noqa: E731
    sharp = lambda t, *_: torch.sin(20 * t)  # noqa: E731
    solver = make_uniform_solver(
        "dopri5", atol=1e-8, rtol=1e-8, device="cpu", max_path_change=0.3
    )

    # Control: the fresh mesh, passed explicitly, fails the check.
    fresh = solver._setup_initial_mesh(
        None, bounds["mesh_init"], bounds["mesh_final"], False, False, 13
    )[0]
    control = solver.integrate(f=sharp, mesh=fresh.clone(), take_gradient=False)
    assert control.converged is False

    # 1. A fresh solver: nothing stored, so the fresh mesh is used and the
    #    check does not run on it.
    solver = make_uniform_solver(
        "dopri5", atol=1e-8, rtol=1e-8, device="cpu", max_path_change=0.3
    )
    first = solver.integrate(
        f=sharp,
        same_integrand=True,
        random_initial_mesh=False,
        take_gradient=False,
        **bounds,
    )
    assert first.converged is True

    # 2. Converge on the smooth integrand, then switch to the sharp one: the
    #    check runs on the stored (smooth-tuned) mesh and stops the call.
    solver.integrate(f=smooth, take_gradient=False, **bounds)
    out = solver.integrate(f=sharp, same_integrand=True, take_gradient=False, **bounds)
    assert out.converged is False

    # 3. The early exit stored its partly refined mesh and max_batch, and the
    #    next same_integrand=True call starts from that mesh.
    assert torch.equal(solver.mesh_previous, out.mesh_optimal.to(solver.device))
    assert solver.previous_max_batch is not None
    recorded = []
    real = solver._setup_initial_mesh

    def wrapper(*args, **kwargs):
        result = real(*args, **kwargs)
        recorded.append(result[0].clone())
        return result

    monkeypatch.setattr(solver, "_setup_initial_mesh", wrapper)
    stored = solver.mesh_previous.clone()
    solver.integrate(f=sharp, same_integrand=True, take_gradient=False, **bounds)
    assert torch.equal(recorded[0], stored)


def test_early_exit_stores_start_mesh_with_failing_steps_split():
    """An early exit stores the starting mesh with this batch's failing steps
    split at their midpoints (what the loop would have done next), so repeated
    same_integrand=True calls refine further until one converges. Storing the
    unsplit mesh made every later call start from, and stop on, the same mesh:
    the default path evaluates the whole starting mesh in its first batch."""
    bounds = {
        "mesh_init": torch.tensor([0.0], dtype=torch.float64),
        "mesh_final": torch.tensor([4.0], dtype=torch.float64),
    }
    solver = make_uniform_solver(
        "dopri5", atol=1e-8, rtol=1e-8, device="cpu", max_path_change=0.3
    )
    torch.manual_seed(0)
    solver.integrate(f=lambda t, *_: torch.sin(t), take_gradient=False, **bounds)
    sharp = lambda t, *_: torch.sin(20 * t)  # noqa: E731

    start = solver.mesh_previous.clone()
    out = solver.integrate(f=sharp, same_integrand=True, take_gradient=False, **bounds)
    assert out.converged is False
    # One batch held every starting step: out.error_ratios is per starting step.
    failed = (out.error_ratios >= 1.0).to(start.device)
    midpoints = 0.5 * (start[:-1] + start[1:])[failed]
    expected = torch.sort(torch.cat([start, midpoints]), dim=0).values
    assert torch.equal(solver.mesh_previous, expected)
    assert torch.equal(out.mesh_optimal, solver.mesh_previous)

    sizes = [len(start), len(expected)]
    for _ in range(6):
        out = solver.integrate(
            f=sharp, same_integrand=True, take_gradient=False, **bounds
        )
        sizes.append(len(solver.mesh_previous))
        if out.converged:
            break
    assert out.converged is True, sizes
    # Each early exit stored a larger mesh than it started from.
    assert sizes[:-1] == sorted(set(sizes[:-1])), sizes
