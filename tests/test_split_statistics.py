"""Tests for the split statistics and the final ``converged`` check.

``IntegrationResult`` reports how much adaptive refinement a call needed:

  - ``n_start_steps``: steps in the starting mesh (given, stored by
    ``same_f=True``, or freshly generated);
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
    out = _gk21_solver(max_batch=BIG_BATCH).integrate(
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
    out = solver.integrate(
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
    out = _gk21_solver(max_batch=BIG_BATCH).integrate(
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
    out = integrate(
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


def test_statistics_count_the_filtered_stored_mesh():
    """With same_f=True on a shorter interval, the starting mesh is the stored
    mesh filtered to [0, 0.5] and padded with the new end point."""
    solver = _gk21_solver(max_batch=BIG_BATCH)
    f = lambda t, *_: torch.sin(3 * t)  # noqa: E731
    torch.manual_seed(0)
    solver.integrate(f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False)
    stored = solver.mesh_previous[:, 0].cpu()
    half = torch.tensor([0.5], dtype=torch.float64)
    out = solver.integrate(
        f=f, same_f=True, mesh_init=T_INIT, mesh_final=half, take_gradient=False
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

    out = _gk21_solver(max_batch=BIG_BATCH).integrate(
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


def test_float_in_former_max_path_change_position_raises():
    """max_path_change was the 4th positional argument; that slot is now
    max_adaptive_splits, and a float there must not act as a cap."""
    with pytest.raises(TypeError, match="max_adaptive_splits"):
        UniformAdaptiveQuadrature(0.1, None, 0.9, 0.1, method="gk21", device="cpu")


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
    out = _gk21_solver(max_batch=BIG_BATCH).integrate(
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
    tolerance; repeated same_f=True calls refine from mesh_optimal and reach
    convergence, and an uncapped call on the same solver is accurate."""
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
    first = solver.integrate(
        f=sharp, take_gradient=False, max_adaptive_splits=1, **bounds
    )
    assert first.converged is False
    assert first.max_split_depth == 1

    results = [first]
    for _ in range(6):
        out = solver.integrate(
            f=sharp, same_f=True, take_gradient=False, max_adaptive_splits=1, **bounds
        )
        results.append(out)
        if out.converged:
            break
    assert results[-1].converged is True, [r.n_start_steps for r in results]
    starts = [r.n_start_steps for r in results[1:]]
    assert starts == sorted(starts), starts

    uncapped = solver.integrate(f=sharp, take_gradient=False, **bounds)
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
    out = solver.integrate(
        f=lambda t: torch.sin(2 * math.pi * t),
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        y0=torch.tensor([5.0], dtype=torch.float64),
        take_gradient=take_gradient,
        conserve_memory=True,
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
    out = solver.integrate(
        f=lambda t, *_: torch.sin(30 * t),
        mesh=torch.linspace(0.0, 1.0, 3, dtype=torch.float64).unsqueeze(-1),
        y0=torch.tensor([1e6], dtype=torch.float64),
        take_gradient=False,
        max_adaptive_splits=1,
    )
    assert out.converged is False


def test_cumulative_mode_recheck_uses_the_running_integral():
    """In cumulative mode the re-check of accepted steps judges each one against
    the running integral through it, including the current batch. Judging
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
    out = solver.integrate(
        f=lambda t: torch.sin(2 * math.pi * t),
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
        conserve_memory=True,
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
    out = solver.integrate(
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
    out = solver.integrate(
        f=lambda t: torch.cat(
            [torch.ones_like(t), torch.where(t > 0.9, torch.nan, torch.ones_like(t))],
            dim=-1,
        ),
        mesh=torch.linspace(0.0, 1.0, 5, dtype=torch.float64).unsqueeze(-1),
        take_gradient=False,
        error_on_nonfinite=False,
    )
    assert torch.isnan(out.integral[1])
    assert out.converged is False


def test_float32_smooth_integral_converges():
    out = integrate(
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
    in agreement with mesh_optimal, which splits them."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-9, rtol=1e-6, device="cpu", max_batch=BIG_BATCH
    )
    torch.manual_seed(0)
    out = solver.integrate(
        f=lambda t, *_: torch.sin(2 * math.pi * t),
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
        error_integral_reference=100.0,
    )
    assert out.converged is False
    assert len(out.mesh_optimal) - 1 > out.h.shape[0]


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
        out = solver.integrate(
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


def test_warning_separates_failing_from_all_steps(caplog):
    """Two steps with atol = 0.01: the first fails (error ~0.018), the second
    passes (error ~0.0037). The failing sum counts the first step only; the
    total counts both."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-2, rtol=0.0, device="cpu", max_batch=BIG_BATCH
    )
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out = solver.integrate(
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
        out = solver.integrate(
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
    """With a fixed error_integral_reference the re-check is off, so
    take_gradient=False is no remedy; the message points at the reference."""
    solver = make_uniform_solver(
        "bosh3", atol=1e-9, rtol=1e-6, device="cpu", max_batch=BIG_BATCH
    )
    torch.manual_seed(0)
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out = solver.integrate(
            f=lambda t, *_: torch.sin(2 * math.pi * t),
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=True,
            error_integral_reference=100.0,
        )
    assert out.converged is False
    (message,) = _warning_messages(caplog)
    assert "error_integral_reference" in message
    assert "take_gradient=False" not in message
    assert "max_adaptive_splits" not in message


def test_no_warning_when_converged(caplog):
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        out = _gk21_solver(max_batch=BIG_BATCH).integrate(
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
    out = solver.integrate(
        f=_narrow_peak,
        mesh=_MESH20,
        take_gradient=take_gradient,
        result_device="cpu",
    )
    assert out.converged is True
    assert out.n_start_steps_split == 1
    capped = solver.integrate(
        f=lambda t, *_: torch.sin(400 * t),
        mesh=_MESH20,
        take_gradient=take_gradient,
        result_device="cpu",
        max_adaptive_splits=0,
    )
    assert capped.converged is False
