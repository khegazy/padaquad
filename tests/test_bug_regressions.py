"""Regression tests for confirmed bugs in the parallel solver.

Tests in this file:

  - test_no_warm_start_path_correctness: anchors the no-warm-start path.
  - test_same_integrand_prefers_stored_mesh_and_max_batch,
    test_same_integrand_skips_memory_benchmark,
    test_stored_values_not_written_by_a_raising_call: the explicit
    ``same_integrand`` flag, which replaced matching integrands by ``id(f)``
    (that matching was bug B2's fix; ids are reused and closures get new ones).
  - test_warm_start_with_new_t_final_yields_correct_mesh: Bug B1, the
    warm-started mesh ends at the new ``mesh_final``.
  - test_max_path_change_returns_integral_output_not_none,
    test_normal_completion_has_converged_true: Bug B6.
  - test_float16_construction_raises: Bug B4.
  - NaN/Inf integrand tests, and the integrand test-call shape.
"""

from __future__ import annotations

import math
import threading
import warnings

import pytest
import torch
from tests._helpers import (
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    cached_max_batch,
    make_solver_for_unit_test,
    make_uniform_solver,
)

from padaquad import IntegrationResult, integrate

# -----------------------------------------------------------------------------
# Anchor: the no-warm-start path is correct as-is. Phase 1 must not regress it.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_no_warm_start_path_correctness(take_gradient):
    """A single fresh-solver call must produce an integral within the
    solver's reported error estimate. This anchors the no-warm-start
    path so Phase 1 fixes cannot accidentally regress it.
    """
    out = integrate(
        f=lambda t: torch.exp(-(t**2)),
        method="dopri5",
        atol=1e-8,
        rtol=1e-8,
        mesh_init=torch.tensor([-2.0], dtype=torch.float64),
        mesh_final=torch.tensor([2.0], dtype=torch.float64),
        take_gradient=take_gradient,
        max_batch=cached_max_batch(),
    )
    expected = math.sqrt(math.pi) * math.erf(2.0)
    actual = out.integral.item()
    # The solver reports an integral_error; the actual error should be
    # within an order of magnitude of that estimate. We pin a generous
    # bound here to anchor "this works" without being brittle to torch
    # version-level numerical differences.
    assert abs(actual - expected) < 1e-3, (
        f"got {actual}, expected {expected}, reported error {out.integral_error.item()}"
    )


# -----------------------------------------------------------------------------
# same_integrand: stored mesh and max_batch replace the caller's.
# -----------------------------------------------------------------------------


def _record_initial_mesh(solver, monkeypatch):
    """Wrap _setup_initial_mesh to record a copy of each call's starting mesh."""
    recorded = []
    real = solver._setup_initial_mesh

    def wrapper(*args, **kwargs):
        out = real(*args, **kwargs)
        recorded.append(out[0].cpu().clone())
        return out

    monkeypatch.setattr(solver, "_setup_initial_mesh", wrapper)
    return recorded


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_same_integrand_prefers_stored_mesh_and_max_batch(take_gradient, monkeypatch):
    """With ``same_integrand=True`` the stored mesh and max_batch of the last
    call replace the caller's; with nothing stored, or with
    ``same_integrand=False``, the caller's values are used."""
    solver = make_uniform_solver("dopri5", atol=1e-6, rtol=1e-6)
    recorded = _record_initial_mesh(solver, monkeypatch)
    user_mesh = torch.linspace(0.0, math.pi, 5, dtype=torch.float64).unsqueeze(-1)

    def call(same_integrand, max_batch):
        return solver.integrate(
            f=lambda t, *_: torch.sin(10 * t),
            mesh=user_mesh,
            same_integrand=same_integrand,
            max_batch=max_batch,
            take_gradient=take_gradient,
        )

    # 1. Nothing stored: the caller's mesh and max_batch, and no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        call(same_integrand=True, max_batch=300)
    assert torch.equal(recorded[-1], user_mesh)
    assert solver.previous_max_batch == 300

    # 2. Stored values replace the caller's.
    stored = solver.mesh_previous.cpu().clone()
    assert not torch.equal(stored, user_mesh)  # else step 2 proves nothing
    call(same_integrand=True, max_batch=50)
    assert torch.equal(recorded[-1], stored)
    assert solver.previous_max_batch == 300

    # 3. same_integrand=False: the caller's values again.
    call(same_integrand=False, max_batch=50)
    assert torch.equal(recorded[-1], user_mesh)
    assert solver.previous_max_batch == 50


def test_same_integrand_skips_memory_benchmark(monkeypatch):
    """The memory benchmark runs once for two ``same_integrand=True`` calls and
    on every call otherwise, even for the same f object (no id(f) matching)."""
    solver = make_solver_for_unit_test("dopri5")
    calls = []

    def fake_benchmark(*args, **kwargs):
        calls.append(1)
        solver.f_unit_mem_size = 1e-6  # GB per evaluation

    monkeypatch.setattr(solver, "_setup_memory_checks", fake_benchmark)
    f = lambda t, *_: torch.sin(t)  # noqa: E731
    bounds = {
        "mesh_init": torch.tensor([0.0], dtype=torch.float64),
        "mesh_final": torch.tensor([1.0], dtype=torch.float64),
    }
    for _ in range(2):
        solver.integrate(f=f, same_integrand=True, take_gradient=False, **bounds)
    assert len(calls) == 1
    for _ in range(2):
        solver.integrate(f=f, same_integrand=False, take_gradient=False, **bounds)
    assert len(calls) == 3


def test_stored_values_not_written_by_a_raising_call():
    """A call that raises leaves the stored mesh and max_batch of the last
    returning call untouched, so same_integrand never mixes two calls."""
    solver = make_uniform_solver("dopri5", atol=1e-6, rtol=1e-6)
    bounds = {
        "mesh_init": torch.tensor([0.0], dtype=torch.float64),
        "mesh_final": torch.tensor([math.pi], dtype=torch.float64),
    }
    solver.integrate(f=torch.sin, max_batch=300, take_gradient=False, **bounds)
    stored_mesh = solver.mesh_previous.clone()
    n_calls = []

    def fails_on_second_batch(t, *_):
        n_calls.append(1)
        if len(n_calls) > 2:  # 1st call: integrate()'s output check
            raise ValueError("boom")
        return torch.sin(10 * t)

    with pytest.raises(ValueError, match="boom"):
        solver.integrate(
            f=fails_on_second_batch, max_batch=50, take_gradient=False, **bounds
        )
    assert len(n_calls) > 2
    assert solver.previous_max_batch == 300
    assert torch.equal(solver.mesh_previous, stored_mesh)


# -----------------------------------------------------------------------------
# B1: mesh_init/mesh_final swap in the warm-start path: a stored mesh reused
# with a new mesh_final must end at the new mesh_final.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_warm_start_with_new_t_final_yields_correct_mesh(take_gradient):
    """Calling the solver a second time with ``same_integrand=True`` and a
    *different* ``mesh_final`` than the first call must still produce a
    monotone mesh that ends at the new ``mesh_final``. This exercises the
    warm-start path (Bug B1).
    """
    solver = make_uniform_solver("dopri5", atol=1e-6, rtol=1e-6)

    def f(t, *args):
        return torch.sin(t)

    mesh_init = torch.tensor([0.0], dtype=torch.float64)
    out_first = solver.integrate(
        f=f,
        mesh_init=mesh_init,
        mesh_final=torch.tensor([1.0], dtype=torch.float64),
        take_gradient=take_gradient,
    )
    expected_first = 1.0 - math.cos(1.0)
    assert abs(out_first.integral.item() - expected_first) < 1e-5

    out_second = solver.integrate(
        f=f,
        mesh_init=mesh_init,
        mesh_final=torch.tensor([1.5], dtype=torch.float64),
        same_integrand=True,
        take_gradient=take_gradient,
    )
    assert out_second is not None
    expected_second = 1.0 - math.cos(1.5)
    assert abs(out_second.integral.item() - expected_second) < 1e-5

    # Bug B1 fix: the warm-started mesh ends at the new mesh_final
    # (previously the buggy concatenation appended mesh_init here,
    # producing non-monotone barriers).
    assert abs(out_second.mesh_optimal[-1].item() - 1.5) < 1e-12

    # Mesh is monotone non-decreasing.
    diffs = out_second.mesh_optimal[1:, 0] - out_second.mesh_optimal[:-1, 0]
    assert torch.all(diffs >= 0), (
        f"warm-started mesh is not monotone: diffs.min()={diffs.min().item()}"
    )


# -----------------------------------------------------------------------------
# B6: max_path_change early exit must return an IntegrationResult with
# converged=False, not bare None (which violated the type contract).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_max_path_change_returns_integral_output_not_none(take_gradient):
    """When ``max_path_change`` triggers early exit on a user-provided
    mesh, the solver must return an ``IntegrationResult`` with
    ``converged=False``, not ``None``. Phase 1's Bug B6 fix.
    """
    # Provide a far-too-coarse mesh on a wiggly integrand so the solver
    # cannot meet a tight tolerance on most steps. max_path_change=0.1
    # means: exit if more than 10% of steps fail. With 4-point initial
    # mesh on a damped sine and 1e-12 atol, that threshold trips hard.
    solver = make_uniform_solver("dopri5", atol=1e-12, rtol=1e-12, max_path_change=0.1)
    t = torch.linspace(0.0, 4.0, 4, dtype=torch.float64).unsqueeze(-1)

    out = solver.integrate(
        f=lambda t, *_: torch.sin(10 * t) * torch.exp(-0.1 * t),
        mesh=t,
        take_gradient=take_gradient,
    )
    assert isinstance(out, IntegrationResult), (
        f"max_path_change early-exit returned {type(out).__name__}, "
        f"expected IntegrationResult. This is bug B6."
    )
    assert out.converged is False, (
        "early-exit IntegrationResult should have converged=False; "
        f"got converged={out.converged!r}"
    )


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_normal_completion_has_converged_true(take_gradient):
    """A normal integration call returns ``converged=True``. Pins the
    default value of the new field.
    """
    out = integrate(
        f=torch.sin,
        method="dopri5",
        atol=1e-6,
        rtol=1e-6,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([math.pi], dtype=torch.float64),
        take_gradient=take_gradient,
        max_batch=cached_max_batch(),
    )
    assert out.converged is True


# -----------------------------------------------------------------------------
# B4: float16 + adaptive should be refused.
# -----------------------------------------------------------------------------


def test_float16_construction_raises():
    """Constructing a parallel solver with dtype=float16 raises
    ``ValueError`` because float16's ~1e-3 precision floor cannot
    support adaptive error control to typical tolerances.

    Phase 1 fix (Bug B4): the guard lives in
    ``SolverBase._set_dtype``.
    """
    with pytest.raises(ValueError, match=r"float16|coarse"):
        make_uniform_solver("dopri5", atol=1e-5, rtol=1e-5, dtype=torch.float16)


# -----------------------------------------------------------------------------
# NaN/Inf integrand: a non-finite error ratio used to fall into neither
# keep_mask nor remove_mask in _adaptively_increase_mesh, so its mesh_trackers
# entry was never cleared and `while torch.any(mesh_trackers)` spun forever.
# Found by the popcornn project (a singular parameter-Jacobian integrand).
# Fix: the guard accepts non-finite panels (never hangs); by default the
# integrand is also checked at the source and a located ValueError is raised.
# -----------------------------------------------------------------------------


def _run_with_timeout(fn, seconds=20):
    """Run ``fn()`` on a daemon thread; fail the test if it doesn't finish.

    The guard guarantees termination, so a healthy build finishes far under the
    timeout. If the guard regresses (infinite loop), ``join`` times out and we
    convert the hang into a deterministic test failure instead of stalling the
    whole suite. ``pytest-timeout`` is not a project dependency, so this stdlib
    watchdog is used instead.
    """
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        pytest.fail(
            f"integrate() did not terminate within {seconds}s — the NaN/Inf "
            f"keep_mask guard in _adaptively_increase_mesh has regressed "
            f"(infinite loop)."
        )
    if "error" in box:
        raise box["error"]
    return box["value"]


def _nan_on_left_half(t, *args):
    """f(t) = NaN on [0, 0.5), t**2 on [0.5, 1].

    Region-based (not a single point) so that some quadrature node always
    lands in the NaN region regardless of the method's node placement,
    reproducing popcornn's singular-integrand case where f returns NaN/Inf at
    certain t.
    """
    return torch.where(t < 0.5, torch.full_like(t, float("nan")), t**2)


def _nan_on_right_half(t, *args):
    """f(t) = t**2 on [0, 0.5], NaN on (0.5, 1]. Mirror of _nan_on_left_half.

    Covers the NaN landing in the *last* panels. In the default absolute-error
    mode this is symmetric with the left case (a single NaN poisons the global
    tolerance, so every panel's ratio is NaN), but it differs under cumulative
    error mode where the denominator is a running cumsum -- see
    test_nonfinite_cumulative_error_mode_terminates.
    """
    return torch.where(t > 0.5, torch.full_like(t, float("nan")), t**2)


_NONFINITE_CASES = [("uniform", "gk21"), ("variable", "adaptive_heun")]
_NONFINITE_IDS = ["uniform_gk21", "variable_adaptive_heun"]

# NaN region position: left vs right half. The guard is position-agnostic
# (~torch.isfinite catches NaN anywhere in the error-ratio vector), so both
# must terminate / raise identically.
_NAN_SIDE_CASES = [_nan_on_left_half, _nan_on_right_half]
_NAN_SIDE_IDS = ["nan_left", "nan_right"]


@pytest.mark.parametrize("nan_f", _NAN_SIDE_CASES, ids=_NAN_SIDE_IDS)
@pytest.mark.parametrize(("sampling", "method"), _NONFINITE_CASES, ids=_NONFINITE_IDS)
@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_nonfinite_integrand_does_not_hang(nan_f, sampling, method, take_gradient):
    """With ``error_on_nonfinite=False`` a NaN-returning integrand must
    terminate (not hang) and return a NaN-containing result, whether the NaN
    falls in the first panels (nan_left) or the last panels (nan_right).

    This is the core regression: before the fix, the non-finite panel was in
    neither mask, so the main loop never exited.
    """
    out = _run_with_timeout(
        lambda: integrate(
            f=nan_f,
            method=method,
            sampling=sampling,
            atol=1e-6,
            rtol=1e-6,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([1.0], dtype=torch.float64),
            take_gradient=take_gradient,
            error_on_nonfinite=False,
            # Pin CPU so this NaN-logic regression is deterministic regardless
            # of whether a GPU is visible (NaN handling is device-agnostic).
            device="cpu",
            max_batch=cached_max_batch(),
        )
    )
    assert isinstance(out, IntegrationResult)
    # The non-finite contribution propagates into the integral (honest result).
    assert torch.isnan(out.integral).any()
    # The loop "converged" by accepting the non-finite panel rather than hanging.
    assert out.converged is True


@pytest.mark.parametrize("nan_f", _NAN_SIDE_CASES, ids=_NAN_SIDE_IDS)
@pytest.mark.parametrize(("sampling", "method"), _NONFINITE_CASES, ids=_NONFINITE_IDS)
@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_nonfinite_integrand_raises_by_default(nan_f, sampling, method, take_gradient):
    """By default (``error_on_nonfinite=True``) a NaN/Inf integrand raises a
    clear, located ``ValueError`` instead of hanging or silently returning NaN,
    for NaN in either the first or last panels.
    """
    with pytest.raises(ValueError, match=r"non-finite"):
        integrate(
            f=nan_f,
            method=method,
            sampling=sampling,
            atol=1e-6,
            rtol=1e-6,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([1.0], dtype=torch.float64),
            take_gradient=take_gradient,
            device="cpu",
            max_batch=cached_max_batch(),
        )


@pytest.mark.parametrize("nan_f", _NAN_SIDE_CASES, ids=_NAN_SIDE_IDS)
def test_nonfinite_cumulative_error_mode_terminates(nan_f):
    """Same guard, but under cumulative error mode (``use_absolute_error_ratio=
    False``), where the per-step tolerance denominator is a running cumsum.

    Unlike the default absolute mode (where one NaN poisons the global
    denominator so every panel's ratio is NaN regardless of position), here the
    NaN's position changes which panels get a finite vs non-finite ratio. This
    exercises the position-dependent path for both nan_left and nan_right; both
    must still terminate with a NaN-containing result.
    """
    out = _run_with_timeout(
        lambda: integrate(
            f=nan_f,
            method="gk21",
            sampling="uniform",
            atol=1e-6,
            rtol=1e-6,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([1.0], dtype=torch.float64),
            take_gradient=False,
            error_on_nonfinite=False,
            use_absolute_error_ratio=False,
            device="cpu",
            max_batch=cached_max_batch(),
        )
    )
    assert isinstance(out, IntegrationResult)
    assert torch.isnan(out.integral).any()
    assert out.converged is True


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_finite_integrand_unaffected_by_nonfinite_check(take_gradient):
    """The default-on finiteness check must not false-trigger on a smooth
    integrand: the integral is still correct and finite.
    """
    out = integrate(
        f=torch.sin,
        method="gk21",
        atol=1e-8,
        rtol=1e-8,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([math.pi], dtype=torch.float64),
        take_gradient=take_gradient,
        device="cpu",
        max_batch=cached_max_batch(),
    )
    assert torch.isfinite(out.integral).all()
    assert abs(out.integral.item() - 2.0) < 1e-6


def test_check_f_output_finite_helper():
    """Unit-test the finiteness helper directly across its branches.

    The helper reads ``self.error_on_nonfinite`` (set from the
    ``integrate(error_on_nonfinite=...)`` argument), so toggle it on the solver.
    """
    solver = make_uniform_solver("gk21", atol=1e-6, rtol=1e-6)
    nodes = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64)
    good = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    bad = torch.tensor([[1.0], [float("nan")], [3.0]], dtype=torch.float64)

    solver.error_on_nonfinite = True
    # Finite tensor: never raises.
    solver._check_f_output_finite(good, nodes)
    # Non-finite tensor with the flag on: raises and localizes the offending t.
    with pytest.raises(ValueError, match=r"non-finite") as excinfo:
        solver._check_f_output_finite(bad, nodes)
    assert "0.2" in str(excinfo.value), f"offending t not reported: {excinfo.value}"

    # Flag off: no raise (the guard keeps the run alive instead).
    solver.error_on_nonfinite = False
    solver._check_f_output_finite(bad, nodes)

    # Bare-scalar branch (e.g. f returning a Python number): finite passes,
    # non-finite raises.
    solver.error_on_nonfinite = True
    solver._check_f_output_finite(1.0, nodes)
    with pytest.raises(ValueError, match=r"non-finite"):
        solver._check_f_output_finite(float("inf"), nodes)


# -----------------------------------------------------------------------------
# The integrand test call receives one point of shape [1, T].
# -----------------------------------------------------------------------------


class _StopAfterFirstCall(Exception):
    """Raised by the recording integrand so integrate() stops after f's 1st call."""


@pytest.mark.parametrize("T", [1, 2])
def test_integrand_test_call_shape(T, monkeypatch):
    """integrate()'s first call to f (the output check) passes t shaped [1, T].

    The call used torch.tensor([[mesh_init]]), which only works when mesh_init
    holds one number: T=1 gave [1, 1], T=2 raised "only one element tensors
    can be converted to Python scalars". Through the public path T>1 is
    stopped earlier by the scalar `mesh_init < mesh_final` assert in
    _setup_integral_bounds (T>1 is not supported end to end yet), so for T=2
    that method is stubbed on this instance to reach the test call.
    """
    solver = make_solver_for_unit_test()
    if T > 1:
        monkeypatch.setattr(solver, "_setup_integral_bounds", lambda mesh, a, b: (a, b))
    shapes = []

    def f(t):
        shapes.append(tuple(t.shape))
        raise _StopAfterFirstCall

    with pytest.raises(_StopAfterFirstCall):
        solver.integrate(
            f,
            mesh_init=torch.zeros(T, dtype=torch.float64),
            mesh_final=torch.ones(T, dtype=torch.float64),
            max_batch=10000,
        )
    assert shapes == [(1, T)]
