"""Tests for the split statistics and the final ``converged`` check.

``IntegrationResult`` reports how much adaptive refinement a call needed:

  - ``n_start_steps``: steps in the starting mesh (given, possibly cut to
    the bounds, or freshly generated);
  - ``n_start_steps_split``: starting steps split at least once;
  - ``percent_start_steps_split``: ``100 * n_start_steps_split / n_start_steps``;
  - ``max_split_depth``: the largest number of times any one step was halved.

``converged`` is ``True`` when every accepted step meets the tolerance judged
against the final integral (the reference ``mesh_optimal`` is built with), and
``False`` otherwise, including when a step's error is NaN/Inf.

The expected counts on the 20-step mesh below come from the integrands' shape:
the narrow peak sits in one step, and the rough region covers exactly 4 or 8 of
the 20 steps (``linspace`` puts barriers exactly at 0.2 and 0.4).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch
from tests._helpers import (
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    make_uniform_solver,
    needs_cuda,
)

from padaquad import UniformAdaptiveQuadrature, adaptive_quadrature, integrate, steps

_MESH20 = torch.linspace(0.0, 1.0, 21, dtype=torch.float64).unsqueeze(-1)
T_INIT = torch.tensor([0.0], dtype=torch.float64)
T_FINAL = torch.tensor([1.0], dtype=torch.float64)
BIG_BATCH = 10**6


def _gk21_solver(**kwargs):
    return make_uniform_solver("gk21", atol=1e-10, rtol=1e-10, device="cpu", **kwargs)


def _narrow_peak(t, *_):
    """A Gaussian of width 0.002 centred at 0.523, inside step [0.50, 0.55]."""
    return torch.exp(-(((t - 0.523) / 0.002) ** 2))


def _rough_until(edge):
    """sin(400 t) below ``edge`` (fails at width 0.05), constant above it."""
    return lambda t, *_: torch.where(t < edge, torch.sin(400 * t), torch.ones_like(t))


# ---------------------------------------------------------------------------
# Split statistics with known answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_narrow_peak_splits_one_starting_step(take_gradient):
    """Only the step holding the peak is split, three levels deep."""
    out, out_state = _gk21_solver(max_batch=BIG_BATCH).integrate(
        f=_narrow_peak, mesh=_MESH20, take_gradient=take_gradient
    )
    assert out.n_start_steps == 20
    assert out.n_start_steps_split == 1
    assert out.percent_start_steps_split == 5.0
    assert out.max_split_depth == 3
    assert out.converged is True


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize(("edge", "n_split"), [(0.2, 4), (0.4, 8)])
def test_rough_region_splits_the_steps_it_covers(edge, n_split, take_gradient):
    """The steps below ``edge`` are split once each; the rest are not. The
    integrand is evaluated in several calls of 5 steps (``max_batch``)."""
    solver = _gk21_solver(max_batch=BIG_BATCH)
    out, out_state = solver.integrate(
        f=_rough_until(edge),
        mesh=_MESH20,
        max_batch=5 * solver.C,
        take_gradient=take_gradient,
    )
    assert out.n_start_steps_split == n_split
    assert out.percent_start_steps_split == 100.0 * n_split / 20
    assert out.max_split_depth == 1
    assert out.converged is True


def test_smooth_integrand_on_fine_mesh_splits_nothing():
    out, out_state = _gk21_solver(max_batch=BIG_BATCH).integrate(
        f=lambda t, *_: torch.sin(t), mesh=_MESH20, take_gradient=False
    )
    assert out.n_start_steps == 20
    assert out.n_start_steps_split == 0
    assert out.percent_start_steps_split == 0.0
    assert out.max_split_depth == 0
    assert out.converged is True
    assert out.h.shape[0] == 20


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize(
    ("sampling", "method"),
    [("uniform", "bosh3"), ("uniform", "gk21"), ("variable", "adaptive_heun")],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_statistics_are_consistent_with_the_mesh(sampling, method, seed, take_gradient):
    """On fresh random meshes, for both samplings: the counts are in range,
    depth is 0 exactly when nothing was split, and each split starting step
    leaves at least two accepted steps."""
    torch.manual_seed(seed)
    out, out_state = integrate(
        f=lambda t: torch.exp(-(((t - 0.3) / 0.05) ** 2)),
        method=method,
        sampling=sampling,
        atol=1e-7,
        rtol=1e-7,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=take_gradient,
        max_batch=BIG_BATCH,
        device="cpu",
    )
    assert 0 <= out.n_start_steps_split <= out.n_start_steps
    assert (out.max_split_depth == 0) == (out.n_start_steps_split == 0)
    assert out.h.shape[0] >= out.n_start_steps + out.n_start_steps_split
    if out.max_split_depth == 0:
        assert out.h.shape[0] == out.n_start_steps


def test_statistics_count_the_cut_mesh():
    """Given an earlier mesh_optimal and a shorter interval, the starting mesh
    is that mesh cut to [0, 0.5] and padded with the new end point."""
    solver = _gk21_solver(max_batch=BIG_BATCH)
    f = lambda t, *_: torch.sin(3 * t)  # noqa: E731
    torch.manual_seed(0)
    cold, cold_state = solver.integrate(
        f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
    )
    stored = cold_state.mesh[:, 0].cpu()
    half = torch.tensor([0.5], dtype=torch.float64)
    out, out_state = solver.integrate(
        f=f, mesh=cold_state.mesh, mesh_final=half, take_gradient=False
    )
    n_inside = int(((stored > 0.0) & (stored < 0.5)).sum())
    assert out.n_start_steps == n_inside + 1
    assert out.converged is True


def test_per_batch_result_has_no_statistics_or_convergence():
    """The result handed to loss_fxn covers one batch: converged and the counts
    are not defined there and are None."""
    seen = []

    def loss_fxn(batch_result):
        seen.append(batch_result)
        return batch_result.integral.sum()

    out, out_state = _gk21_solver(max_batch=BIG_BATCH).integrate(
        f=_narrow_peak, mesh=_MESH20, take_gradient=False, loss_fxn=loss_fxn
    )
    assert out.converged is True
    assert seen
    for r in seen:
        assert r.converged is None
        assert r.n_start_steps is None
        assert r.n_start_steps_split is None
        assert r.max_split_depth is None
        assert r.percent_start_steps_split is None


# ---------------------------------------------------------------------------
# Removed max_path_change and the max_adaptive_splits type check
# ---------------------------------------------------------------------------


def test_stale_max_path_change_keyword_raises():
    with pytest.raises(TypeError, match="max_path_change"):
        adaptive_quadrature("uniform", method="gk21", max_path_change=0.1)


def test_positional_constructor_arguments_raise():
    """Constructor options are keyword-only, so the old positional slots
    (where max_path_change was the 4th) cannot be filled by position."""
    with pytest.raises(TypeError, match="positional argument"):
        UniformAdaptiveQuadrature(0.1, None, 0.9, 0.1, method="gk21", device="cpu")


def test_float_max_adaptive_splits_raises():
    """A float such as 0.1, meant for the removed max_path_change, must not
    act as a split cap."""
    with pytest.raises(TypeError, match="max_adaptive_splits"):
        UniformAdaptiveQuadrature(max_adaptive_splits=0.1, device="cpu")


@pytest.mark.parametrize(
    ("value", "error"), [(0.5, TypeError), (True, TypeError), (-1, ValueError)]
)
def test_invalid_per_call_max_adaptive_splits_raises(value, error):
    solver = _gk21_solver(max_batch=BIG_BATCH)
    with pytest.raises(error, match="max_adaptive_splits"):
        solver.integrate(
            f=_narrow_peak, mesh=_MESH20, take_gradient=False, max_adaptive_splits=value
        )


def test_numpy_integer_max_adaptive_splits_is_accepted():
    out, out_state = _gk21_solver(max_batch=BIG_BATCH).integrate(
        f=_narrow_peak,
        mesh=_MESH20,
        take_gradient=False,
        max_adaptive_splits=np.int64(1),
    )
    assert out.max_split_depth == 1


# ---------------------------------------------------------------------------
# A non-converged call, then warm starts (replaces the early-exit tests)
# ---------------------------------------------------------------------------


def test_capped_call_then_warm_starts_converge():
    """A cap of 1 stops refinement of sin(20 t) on [0, 4] short of the
    tolerance; repeated calls from the previous call's state,
    as the not-converged warning advises, stay on [0, 4], refine further and
    reach convergence, and an uncapped call on the same solver is accurate."""
    bounds = {
        "mesh_init": T_INIT,
        "mesh_final": torch.tensor([4.0], dtype=torch.float64),
    }
    sharp = lambda t, *_: torch.sin(20 * t)  # noqa: E731
    exact = (1.0 - math.cos(80.0)) / 20.0
    solver = make_uniform_solver(
        "dopri5", atol=1e-8, rtol=1e-8, device="cpu", max_batch=BIG_BATCH
    )
    torch.manual_seed(0)
    first, out_state = solver.integrate(
        f=sharp, take_gradient=False, max_adaptive_splits=1, **bounds
    )
    assert first.converged is False
    assert first.max_split_depth == 1

    results = [first]
    out = first
    for _ in range(6):
        out, out_state = solver.integrate(
            f=sharp,
            mesh=out_state.mesh,
            max_batch=out_state.max_batch,
            take_gradient=False,
            max_adaptive_splits=1,
        )
        assert torch.equal(out.mesh_init.cpu(), bounds["mesh_init"])
        assert torch.equal(out.mesh_final.cpu(), bounds["mesh_final"])
        results.append(out)
        if out.converged:
            break
    assert results[-1].converged is True, [r.n_start_steps for r in results]
    starts = [r.n_start_steps for r in results[1:]]
    assert starts == sorted(starts), starts

    uncapped, _ = solver.integrate(f=sharp, take_gradient=False, **bounds)
    assert uncapped.converged is True
    assert abs(float(uncapped.integral) - exact) < 1e-6


# ---------------------------------------------------------------------------
# converged in the different error modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("cap", [None, 0], ids=["uncapped", "cap0"])
@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "cumulative"])
def test_converged_matches_a_hand_computed_tolerance_check(
    absolute, take_gradient, cap, seed
):
    """An independent oracle for converged (D = 1, "2" norm): every step's
    |error| is below max(atol, rtol*|reference|, 50*eps*|step integral|), with
    reference the final integral without y0 (absolute mode) or the final
    running integral through the step (cumulative mode). Several batches, so
    the running integral drifts."""
    atol, rtol = 1e-12, 1e-6
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="dopri5",
        atol=atol,
        rtol=rtol,
        device="cpu",
        use_absolute_error_ratio=absolute,
    )
    torch.manual_seed(seed)
    out, out_state = solver.integrate(
        f=lambda t: torch.sin(2 * math.pi * t),
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        y0=torch.tensor([5.0], dtype=torch.float64),
        take_gradient=take_gradient,
        # The values leave the device with take_gradient=False (it raises with
        # take_gradient=True); converged must not depend on where they are.
        trade_speed_for_memory=not take_gradient,
        max_batch=8 * solver.C,
        max_adaptive_splits=cap,
    )
    q = out.mesh_quadratures[:, 0]
    err = out.mesh_quadrature_errors[:, 0]
    if absolute:
        reference = torch.abs(out.integral - out.y0)
    else:
        reference = torch.abs(torch.cumsum(q, 0))
    tol = torch.maximum(
        torch.clamp(rtol * reference, min=atol),
        50.0 * torch.finfo(torch.float64).eps * torch.abs(q),
    )
    assert out.converged is bool((err / tol < 1).all())


def test_y0_does_not_loosen_the_converged_tolerance():
    """The tolerance is on the integral of f, not on y0 + integral: a large y0
    would make rtol*|y0 + I| ~ 1 and pass every step."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-14, rtol=1e-6, device="cpu", max_batch=BIG_BATCH
    )
    out, out_state = solver.integrate(
        f=lambda t, *_: torch.sin(30 * t),
        mesh=torch.linspace(0.0, 1.0, 3, dtype=torch.float64).unsqueeze(-1),
        y0=torch.tensor([1e6], dtype=torch.float64),
        take_gradient=False,
        max_adaptive_splits=1,
    )
    assert out.converged is False


def test_cumulative_mode_recheck_uses_the_running_integral():
    """In cumulative mode the re-check of accepted steps judges each one against
    the running integral through it, including the current pass. Judging
    against the record alone (the previous behaviour) left this run with a
    failing step (converged=False)."""
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="dopri5",
        atol=1e-12,
        rtol=1e-6,
        device="cpu",
        use_absolute_error_ratio=False,
    )
    torch.manual_seed(1)
    out, out_state = solver.integrate(
        f=lambda t: torch.sin(2 * math.pi * t),
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
        trade_speed_for_memory=True,
        max_batch=16 * solver.C,
    )
    assert out.converged is True


@pytest.mark.parametrize(
    ("mesh_failure_tolerance", "converged"), [(0.0, False), (0.5, True)]
)
def test_failure_fraction_converged_follows_the_accept_rule(
    mesh_failure_tolerance, converged
):
    """D = 2: the first output fails everywhere (cap 0), the second is constant
    and exact. Half of each step's outputs fail, which the accept rule allows
    with mesh_failure_tolerance = 0.5 and rejects with 0.0."""
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="bosh3",
        atol=1e-12,
        rtol=1e-12,
        device="cpu",
        error_norm="failure_fraction",
        mesh_failure_tolerance=mesh_failure_tolerance,
        max_batch=BIG_BATCH,
    )
    out, out_state = solver.integrate(
        f=lambda t: torch.cat([torch.sin(30 * t), torch.ones_like(t)], dim=-1),
        mesh=torch.linspace(0.0, 1.0, 3, dtype=torch.float64).unsqueeze(-1),
        take_gradient=False,
        max_adaptive_splits=0,
    )
    assert out.integral.shape == (2,)
    assert out.converged is converged


@pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "cumulative"])
def test_failure_fraction_nan_component_is_not_converged(absolute):
    """A NaN in one output component of one step: the failure fraction itself
    stays finite (NaN > 1 is False), so only the per-element non-finite check
    makes converged False, also with mesh_failure_tolerance = 0.5."""
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="bosh3",
        atol=1e-9,
        rtol=1e-9,
        device="cpu",
        error_norm="failure_fraction",
        mesh_failure_tolerance=0.5,
        use_absolute_error_ratio=absolute,
        max_batch=BIG_BATCH,
    )
    out, out_state = solver.integrate(
        f=lambda t: torch.cat(
            [torch.ones_like(t), torch.where(t > 0.9, torch.nan, torch.ones_like(t))],
            dim=-1,
        ),
        mesh=torch.linspace(0.0, 1.0, 5, dtype=torch.float64).unsqueeze(-1),
        take_gradient=False,
        runtime_checks=False,
    )
    assert torch.isnan(out.integral[1])
    assert out.converged is False


def test_float32_smooth_integral_converges():
    out, out_state = integrate(
        f=torch.sin,
        method="gk21",
        atol=1e-5,
        rtol=1e-5,
        mesh_init=torch.tensor([0.0], dtype=torch.float32),
        mesh_final=torch.tensor([math.pi], dtype=torch.float32),
        take_gradient=False,
        max_batch=BIG_BATCH,
        device="cpu",
    )
    assert out.integral.dtype == torch.float32
    assert out.converged is True
    assert out.n_start_steps_split <= out.n_start_steps


def test_converged_uses_the_final_integral_not_the_given_reference():
    """sin(2 pi t) integrates to ~0. A reference of 100 makes the loop accept
    coarse steps, but against the final integral they fail: converged is False,
    in agreement with mesh_optimal, which splits them. enforce_convergence=False
    keeps the loop from refining them once the reference is dropped."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-9, rtol=1e-6, device="cpu", max_batch=BIG_BATCH
    )
    torch.manual_seed(0)
    out, out_state = solver.integrate(
        f=lambda t, *_: torch.sin(2 * math.pi * t),
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
        error_integral_reference=100.0,
        enforce_convergence=False,
    )
    assert out.converged is False
    assert len(out_state.mesh) - 1 > out.h.shape[0]


# ---------------------------------------------------------------------------
# The warning on a non-converged return
# ---------------------------------------------------------------------------


def _warning_messages(caplog):
    return [r.getMessage() for r in caplog.records if "did not converge" in r.message]


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_warning_reports_the_size_of_the_error(caplog, take_gradient):
    """D = 1, cap 0, two steps that both fail. The numbers in the message are
    recomputed here: the worst ratio |e| / max(atol, rtol*|I|), the summed
    |error estimate| and |I|. The take_gradient=False remedy is offered only
    when the call took gradients."""
    atol = rtol = 1e-12
    solver = make_uniform_solver(
        "bosh3", atol=atol, rtol=rtol, device="cpu", max_batch=BIG_BATCH
    )
    mesh = torch.linspace(0.0, 1.0, 3, dtype=torch.float64).unsqueeze(-1)
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out, out_state = solver.integrate(
            f=lambda t, *_: torch.sin(30 * t),
            mesh=mesh,
            take_gradient=take_gradient,
            max_adaptive_splits=0,
        )
    assert out.converged is False
    messages = _warning_messages(caplog)
    assert len(messages) == 1
    message = messages[0]
    abs_integral = float(torch.abs(out.integral))
    err = out.mesh_quadrature_errors[:, 0]
    worst = float(err.max()) / max(atol, rtol * abs_integral)
    assert "2 of 2 accepted steps (100.0%)" in message
    assert f"Largest error ratio among the finite failing steps: {worst:.3g}" in message
    assert f"summed |error estimate| is {float(err.sum()):.3g}," in message
    assert f"|integral| = {abs_integral:.3g} (" in message
    assert "raise max_adaptive_splits" in message
    assert ("take_gradient=False" in message) is take_gradient
    assert "state=state" in message


def test_warning_separates_failing_from_all_steps(caplog):
    """Two steps with atol = 0.01: the first fails (error ~0.018), the second
    passes (error ~0.0037). The failing sum counts the first step only; the
    total counts both."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-2, rtol=0.0, device="cpu", max_batch=BIG_BATCH
    )
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out, out_state = solver.integrate(
            f=lambda t, *_: torch.where(
                t < 0.5, torch.sin(30 * t), 0.1 * torch.sin(5 * t)
            ),
            mesh=torch.linspace(0.0, 1.0, 3, dtype=torch.float64).unsqueeze(-1),
            take_gradient=False,
            max_adaptive_splits=0,
        )
    err = out.mesh_quadrature_errors[:, 0]
    assert float(err[0]) > 1e-2 > float(err[1]) > 1e-3
    assert out.converged is False
    (message,) = _warning_messages(caplog)
    assert "1 of 2 accepted steps (50.0%)" in message
    assert (
        f"summed |error estimate| is {float(err[0]):.3g}, of "
        f"{float(err.sum()):.3g} summed over all finite steps"
    ) in message


def test_warning_reports_one_output_component(caplog):
    """D = 2 with very different scales: the error, total error and |integral|
    in the message all belong to component 0, the one whose failing error is
    largest relative to its integral (not the largest of each across
    components)."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-12, rtol=1e-12, device="cpu", max_batch=BIG_BATCH
    )
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out, out_state = solver.integrate(
            f=lambda t, *_: torch.cat(
                [torch.sin(30 * t), 1000.0 + 1e-3 * torch.sin(5 * t)], dim=-1
            ),
            mesh=torch.linspace(0.0, 1.0, 3, dtype=torch.float64).unsqueeze(-1),
            take_gradient=False,
            max_adaptive_splits=0,
        )
    assert out.converged is False
    (message,) = _warning_messages(caplog)
    err0 = float(out.mesh_quadrature_errors[:, 0].sum())
    assert "output component 0 of 2" in message
    assert f"summed |error estimate| is {err0:.3g}," in message
    assert f"|integral| = {float(torch.abs(out.integral[0])):.3g} (" in message


def test_warning_with_a_pinned_reference_suggests_the_reference(caplog):
    """With take_gradient=True and a fixed error_integral_reference the
    message points at both remedies: take_gradient=False turns
    enforce_convergence on, which drops the reference once every step is
    accepted, and a smaller reference."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-9, rtol=1e-6, device="cpu", max_batch=BIG_BATCH
    )
    torch.manual_seed(0)
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out, out_state = solver.integrate(
            f=lambda t, *_: torch.sin(2 * math.pi * t),
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=True,
            error_integral_reference=100.0,
        )
    assert out.converged is False
    (message,) = _warning_messages(caplog)
    assert "error_integral_reference" in message
    assert "take_gradient=False" in message
    assert "max_adaptive_splits" not in message


def test_no_warning_when_converged(caplog):
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out, out_state = _gk21_solver(max_batch=BIG_BATCH).integrate(
            f=_narrow_peak, mesh=_MESH20, take_gradient=False
        )
    assert out.converged is True
    assert not [r for r in caplog.records if "did not converge" in r.getMessage()]


# ---------------------------------------------------------------------------
# GPU: the final check runs before the record moves to result_device
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_converged_on_cuda_with_cpu_results(take_gradient):
    solver = make_uniform_solver("gk21", atol=1e-10, rtol=1e-10, device="cuda")
    out, out_state = solver.integrate(
        f=_narrow_peak,
        mesh=_MESH20,
        take_gradient=take_gradient,
        result_device="cpu",
    )
    assert out.converged is True
    assert out.n_start_steps_split == 1
    capped, _ = solver.integrate(
        f=lambda t, *_: torch.sin(400 * t),
        mesh=_MESH20,
        take_gradient=take_gradient,
        result_device="cpu",
        max_adaptive_splits=0,
    )
    assert capped.converged is False


# ---------------------------------------------------------------------------
# split_rule="accelerate": failing pieces with large error ratios split in four
# ---------------------------------------------------------------------------

_A = 0.3183098861837907
# Integrands that need many levels of refinement, with their integrals on [0, 1].
_HARD = {
    "lorentz": (
        lambda t: 1.0 / (1.0 + 1.0e6 * (t - 0.4) ** 2),
        (math.atan(600) + math.atan(400)) / 1000,
    ),
    "step": (lambda t: (t > _A).to(t.dtype), 1 - _A),
    "oscillatory": (lambda t: torch.sin(2.0e4 * t), (1 - math.cos(2.0e4)) / 2.0e4),
}


def _split_rule_run(split_rule, f, method="gk21", seed=0, atol=1e-9, **kwargs):
    """Integrate f on [0, 1] from a seeded random starting mesh; return the
    result, the state and the number of points of each call of f."""
    calls = []

    def counted(t):
        calls.append(t.shape[0])
        return f(t)

    kwargs.setdefault("max_batch", BIG_BATCH)
    torch.manual_seed(seed)
    out, out_state = integrate(
        counted,
        method,
        atol=atol,
        rtol=0.0,
        device="cpu",
        split_rule=split_rule,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        **kwargs,
    )
    return out, out_state, calls


@pytest.mark.parametrize("value", ["quarter", None, 4])
def test_invalid_split_rule_raises(value):
    if value is not None:
        with pytest.raises(ValueError, match="split_rule"):
            UniformAdaptiveQuadrature(split_rule=value, device="cpu")
    with pytest.raises(ValueError, match="split_rule"):
        _gk21_solver().integrate(
            f=_narrow_peak,
            mesh=_MESH20,
            take_gradient=False,
            split_rule="quarter" if value is None else value,
        )


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("name", list(_HARD))
@pytest.mark.parametrize("method", ["gk21", "gk15", "gk7", "cc17"])
def test_accelerate_meets_the_tolerance(method, name, take_gradient):
    """With the rule on, the integral is within 10 atol of the exact value and
    the call converges (gk7 accepts the step by accident when halving, so its
    error is checked only with the rule)."""
    f, exact = _HARD[name]
    out, _, _ = _split_rule_run(
        "accelerate", f, method=method, take_gradient=take_gradient
    )
    assert out.converged is True
    assert abs(float(out.integral) - exact) < 1e-8


def test_accelerate_saves_points_and_never_more_on_an_isolated_feature():
    """On sin(2e4 t) at atol 1e-11 the rule evaluates fewer points in fewer
    passes (seed 0, gk21: 66,759 points in 7 passes against 89,901 in 10); on
    a kink, whose failing pieces each leave one failing piece, it never acts
    and the run is the same bit for bit."""
    f, _ = _HARD["oscillatory"]
    (_, _, halve), (_, _, fast) = (
        _split_rule_run(rule, f, take_gradient=False, atol=1e-11)
        for rule in ("halve", "accelerate")
    )
    assert sum(fast) < 0.8 * sum(halve)
    assert len(fast) < len(halve)  # passes: one call of f each
    kink = lambda t: torch.abs(t - 1.0 / 3.0)  # noqa: E731
    (halve, s_halve, halve_calls), (fast, s_fast, fast_calls) = (
        _split_rule_run(rule, kink, take_gradient=False)
        for rule in ("halve", "accelerate")
    )
    assert fast_calls == halve_calls
    assert torch.equal(fast.integral, halve.integral)
    assert torch.equal(s_fast.mesh, s_halve.mesh)


@pytest.mark.parametrize("trade_speed_for_memory", [False, True])
@pytest.mark.parametrize(
    ("sampling", "method", "atol"),
    [("uniform", "gk21", 1e-11), ("variable", "interpolatory3_variable", 1e-7)],
)
def test_accelerate_is_the_same_for_any_max_batch(
    sampling, method, atol, trade_speed_for_memory
):
    """With take_gradient=False every pass is judged whole, so the four-way
    splits, the integral and the state's mesh do not depend on how the calls
    of f are cut (7 points: calls end inside subintervals), and f is
    evaluated at as many points either way. (gk21 splits in four on
    sin(3e3 t) at atol 1e-11; the low-degree variable method, whose 2**k is
    small, on the Lorentzian at 1e-7.)"""
    f = (lambda t: torch.sin(3.0e3 * t)) if method == "gk21" else _HARD["lorentz"][0]
    runs = [
        _split_rule_run(
            "accelerate",
            f,
            method=method,
            sampling=sampling,
            atol=atol,
            take_gradient=False,
            trade_speed_for_memory=trade_speed_for_memory,
            max_batch=max_batch,
        )
        for max_batch in (BIG_BATCH, 7)
    ]
    (out, out_state, calls), (small, small_state, small_calls) = runs
    assert torch.equal(small.integral, out.integral)
    assert torch.equal(small_state.mesh, out_state.mesh)
    assert sum(small_calls) == sum(calls)
    _, _, halve_calls = _split_rule_run(
        "halve", f, method=method, sampling=sampling, atol=atol, take_gradient=False
    )
    assert len(calls) < len(halve_calls)  # fewer passes: the rule acted


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("absolute", [True, False])
def test_accelerate_state_restarts_and_float32(absolute, dtype):
    """The state's mesh after four-way splits is a valid starting mesh: a
    second call from it converges, evaluates fewer points and gives the same
    integral within the tolerance; in cumulative mode and in float32 too."""
    f, exact = _HARD["oscillatory"]
    atol = 1e-9 if dtype == torch.float64 else 1e-4
    lo, hi = T_INIT.to(dtype), T_FINAL.to(dtype)
    solver = adaptive_quadrature(
        "uniform",
        "gk21",
        atol=atol,
        rtol=0.0,
        device="cpu",
        dtype=dtype,
        split_rule="accelerate",
        use_absolute_error_ratio=absolute,
        max_batch=BIG_BATCH,
    )
    points = []

    def counted(t):
        points.append(t.shape[0])
        return f(t)

    torch.manual_seed(0)
    first, state = solver.integrate(
        counted, mesh_init=lo, mesh_final=hi, take_gradient=False
    )
    n_first = sum(points)
    second, _ = solver.integrate(counted, state=state, take_gradient=False)
    assert first.converged
    assert second.converged
    assert torch.all(state.mesh[1:] > state.mesh[:-1])
    assert sum(points) - n_first < n_first
    for out in (first, second):
        assert abs(float(out.integral) - exact) < 10 * atol


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_accelerate_tests_no_piece_before_the_first_split(take_gradient):
    """Until a subinterval of the call is split, none is a piece of a split
    one, so the rule has nothing to test: a call that ends in its first pass
    runs the mesh update exactly as halving does (no four-way test, which cost
    about 30 GPU launches), and later updates test for four pieces."""
    tests = []
    solver = _gk21_solver(split_rule="accelerate", max_batch=BIG_BATCH)
    splittable = solver._splittable

    def spy(*args, halvings=1, **kwargs):
        tests.append(halvings)
        return splittable(*args, halvings=halvings, **kwargs)

    solver._splittable = spy
    smooth, _ = solver.integrate(
        f=lambda t: torch.exp(-t) * torch.sin(4 * t),
        mesh=_MESH20,
        take_gradient=take_gradient,
    )
    assert smooth.max_split_depth == 0
    assert 2 not in tests
    torch.manual_seed(0)
    solver.integrate(
        f=_HARD["lorentz"][0],
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=take_gradient,
    )
    assert 2 in tests


@pytest.mark.parametrize("method", ["gk15", "gk21", "cc33"])
def test_accelerate_spends_nothing_where_one_halving_suffices(method):
    """f = T_{k-1}(2t - 1), k the method's error_exponent, on one starting
    subinterval: every error estimate is c h^k exactly, so one halving divides
    every ratio by 2^k. Below 2^k one halving suffices and the rule halves;
    with the former fixed threshold 1e4 it split in four (gk21 at atol 3.2e-11:
    11 subintervals against 7). Over tolerances from 1e-6 to 1e-16 the rule
    never evaluates more points than halving, for the same integral."""
    k = adaptive_quadrature("uniform", method, device="cpu").method.error_exponent
    exact = 1 / (1 - (k - 1) ** 2)  # integral of T_n(2t - 1) on [0, 1], n even
    for atol in np.logspace(-6, -16, 11):
        points, errors = {}, {}
        for split_rule in ("halve", "accelerate"):
            calls = []

            def f(t, calls=calls):
                calls.append(t.shape[0])
                return torch.special.chebyshev_polynomial_t(2 * t - 1, k - 1)

            out, _ = integrate(
                f,
                method,
                atol=float(atol),
                rtol=0.0,
                device="cpu",
                split_rule=split_rule,
                mesh=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                take_gradient=False,
            )
            points[split_rule] = sum(calls)
            errors[split_rule] = abs(float(out.integral) - exact)
        assert points["accelerate"] <= points["halve"], atol
        assert errors["accelerate"] < max(10 * errors["halve"], 1e-15), atol


@pytest.mark.parametrize(
    ("ctor", "call"),
    [
        ({"split_rule": "accelerate", "error_norm": "failure_fraction"}, None),
        ({"error_norm": "failure_fraction"}, {"split_rule": "accelerate"}),
        ({"split_rule": "accelerate"}, {"error_norm": "failure_fraction"}),
    ],
    ids=["constructor", "call_split_rule", "call_error_norm"],
)
def test_accelerate_with_failure_fraction_raises(ctor, call):
    """split_rule="accelerate" splits by the error ratio, which
    error_norm="failure_fraction" does not have (it compares the fraction of
    failing output components): the combination raises before f is called,
    whether it comes from the constructor, the call, or both."""
    calls = []

    def f(t):
        calls.append(t.shape[0])
        return torch.sin(t)

    with pytest.raises(ValueError, match="failure_fraction") as raised:
        solver = UniformAdaptiveQuadrature(device="cpu", **ctor)
        solver.integrate(f=f, mesh=_MESH20, take_gradient=False, **(call or {}))
    assert 'split_rule="halve"' in str(raised.value)
    assert calls == []


def test_accelerate_with_a_norm_family_call_is_allowed():
    """A per-call norm-family error_norm makes the constructor's
    failure_fraction default usable with split_rule="accelerate"."""
    solver = UniformAdaptiveQuadrature(device="cpu", error_norm="failure_fraction")
    out, _ = solver.integrate(
        f=lambda t: torch.sin(t),
        mesh=_MESH20,
        take_gradient=False,
        split_rule="accelerate",
        error_norm="max",
    )
    assert out.converged


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_accelerate_bound_and_parameter_gradients(take_gradient):
    """Bounds that require grad get d/da = -f(a) and d/db = f(b) through the
    mesh points that four-way splits insert (with take_gradient=True the
    gradient collected on the mesh is carried across both halvings of an
    update), over several batches, and a parameter its exact gradient."""
    w = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.9], dtype=torch.float64, requires_grad=True)
    torch.manual_seed(0)
    out, _ = integrate(
        lambda t: w * torch.sin(2.0e4 * t),
        atol=1e-9,
        rtol=0.0,
        device="cpu",
        split_rule="accelerate",
        mesh_init=a,
        mesh_final=b,
        take_gradient=take_gradient,
        max_batch=500 * 21,
    )
    if not take_gradient:
        out.integral.sum().backward()
    assert math.isclose(float(a.grad), -2 * math.sin(2.0e3), rel_tol=1e-9)
    assert math.isclose(float(b.grad), 2 * math.sin(1.8e4), rel_tol=1e-9)
    exact = (math.cos(2.0e3) - math.cos(1.8e4)) / 2.0e4
    assert math.isclose(float(w.grad), exact, rel_tol=1e-6, abs_tol=1e-9)
