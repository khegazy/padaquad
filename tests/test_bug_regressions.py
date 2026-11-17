"""Regression tests for confirmed bugs in the parallel solver.

Tests in this file:

  - test_no_warm_start_path_correctness: anchors the no-warm-start path.
  - test_reuse_mesh_and_max_batch and the tests after it: the solver keeps
    no warm-start state; a call reuses an earlier result's mesh and batch
    size by passing them (this replaced ``same_f``, which replaced matching
    integrands by ``id(f)``, bug B2's fix).
  - test_bounds_required_without_mesh and the tests after it: the interval
    rules (a mesh or both bounds; bounds inside a mesh cut it, outside raise).
  - test_normal_completion_has_converged_true: a normal call returns
    ``converged=True`` (Bug B6's early-exit path was removed with
    ``max_path_change``).
  - test_float16_construction_raises: Bug B4.
  - NaN/Inf integrand tests, and the integrand test-call shape.
"""

from __future__ import annotations

import math
import threading

import pytest
import torch
from tests._helpers import (
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    cached_max_batch,
    make_solver_for_unit_test,
    make_uniform_solver,
    needs_cuda,
)

from padaquad import IntegrationResult, adaptive_quadrature, integrate

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
# Reuse by passing: the solver keeps no warm-start state. A later call reuses an
# earlier result's mesh and batch size by passing mesh=r.mesh_optimal and
# max_batch=r.max_batch (this replaced same_f, mesh_previous and
# previous_max_batch).
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_reuse_mesh_and_max_batch(device, monkeypatch):
    """The first call benchmarks and returns max_batch; passing it and the
    optimal mesh to the next call skips the benchmark and starts from every
    point of that mesh."""
    solver = adaptive_quadrature("uniform", "gk21", device=device)
    calls = []

    def fake_benchmark(*args, **kwargs):
        calls.append(1)
        solver.f_mem_coeffs = (0.0, 1e-6, 0.0)  # GB: 1e-6 per evaluation

    monkeypatch.setattr(solver, "_setup_memory_checks", fake_benchmark)
    first = solver.integrate(
        f=torch.exp, mesh_init=_bound(2.0), mesh_final=_bound(3.0), take_gradient=False
    )
    assert len(calls) == 1
    assert isinstance(first.max_batch, int)

    second = solver.integrate(
        f=torch.exp,
        mesh=first.mesh_optimal,
        max_batch=first.max_batch,
        take_gradient=False,
    )
    assert len(calls) == 1
    assert second.n_start_steps == len(first.mesh_optimal) - 1
    exact = math.exp(3.0) - math.exp(2.0)
    assert math.isclose(float(second.integral), exact, rel_tol=1e-6)


def test_result_max_batch_is_the_batch_used():
    """result.max_batch is the max_batch the call used: the constructor's, or
    the call's own."""
    bounds = {"mesh_init": _bound(0.0), "mesh_final": _bound(1.0)}
    out = _solver().integrate(f=torch.exp, take_gradient=False, **bounds)
    assert out.max_batch == 10_000
    out = _solver().integrate(f=torch.exp, max_batch=500, take_gradient=False, **bounds)
    assert out.max_batch == 500


def test_solver_keeps_no_warm_start_state():
    """No stored mesh or batch size: a call after one on [2, 3] starts from a
    fresh mesh (13 points by default give 11 subintervals)."""
    solver = _solver()
    mesh = torch.linspace(2.0, 3.0, 9, dtype=torch.float64).unsqueeze(-1)
    solver.integrate(f=torch.exp, mesh=mesh, take_gradient=False)
    assert not hasattr(solver, "mesh_previous")
    assert not hasattr(solver, "previous_max_batch")
    out = solver.integrate(
        f=torch.exp, mesh_init=_bound(0.0), mesh_final=_bound(1.0), take_gradient=False
    )
    assert out.n_start_steps == 11


def test_same_f_is_gone():
    """same_f was removed; passing it raises TypeError."""
    with pytest.raises(TypeError, match="same_f"):
        _solver().integrate(
            f=torch.exp, same_f=True, mesh_init=_bound(0.0), mesh_final=_bound(1.0)
        )


# -----------------------------------------------------------------------------
# Interval rules: a call needs a mesh or both bounds; bounds inside a given mesh
# cut it; a bound outside it raises; a bound within rounding of a mesh end is
# that end. There are no constructor bounds and no stored warm-start mesh.
# -----------------------------------------------------------------------------


def _bound(x, dtype=torch.float64):
    """An integration bound of shape [1]."""
    return torch.tensor([x], dtype=dtype)


def _solver():
    """A CPU gk21 solver with a fixed max_batch (no memory benchmark)."""
    return adaptive_quadrature("uniform", "gk21", device="cpu", max_batch=10_000)


#: The mesh of the interval-rule tests: 9 points 0, 0.5, ..., 4 (8 subintervals).
_MESH = torch.linspace(0.0, 4.0, 9, dtype=torch.float64).unsqueeze(-1)


@pytest.mark.parametrize(
    ("lo", "hi"),
    [
        (0.0, 2.0),
        (0.0, math.nextafter(2.0, 3.0)),
        (0.5, 1.5),
    ],
    ids=["equal_to_the_ends", "within_rounding_of_an_end", "cutting"],
)
def test_bound_gradients_through_a_given_mesh(lo, hi):
    """Given bounds are the mesh's ends, so their gradients are those of any
    mesh end: for the integral of w e^t over [a, b], d/da = -w e^a and
    d/db = w e^b, also when a bound equals the end of a detached
    mesh_optimal (it used to be replaced by that end, and got no gradient)."""
    mesh = torch.linspace(0.0, 2.0, 9, dtype=torch.float64).unsqueeze(-1)
    w = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    a = _bound(lo).requires_grad_()
    b = _bound(hi).requires_grad_()
    out = _solver().integrate(
        f=lambda t: w * torch.exp(t),
        mesh=mesh,
        mesh_init=a,
        mesh_final=b,
        take_gradient=False,
    )
    out.integral.sum().backward()
    assert math.isclose(float(a.grad), -2 * math.exp(lo), rel_tol=1e-9)
    assert math.isclose(float(b.grad), 2 * math.exp(hi), rel_tol=1e-9)
    assert math.isclose(float(w.grad), math.exp(hi) - math.exp(lo), rel_tol=1e-9)


@pytest.mark.parametrize(
    "bounds",
    [
        {"mesh_init": torch.tensor(0.0), "mesh_final": torch.tensor(1.0)},
        {"mesh_init": torch.tensor([[0.0]]), "mesh_final": torch.tensor([[1.0]])},
        {"mesh_init": _bound(0.0), "mesh_final": torch.tensor([1.0, 1.0])},
        {"mesh_init": 0.0, "mesh_final": 1.0},
    ],
    ids=["zero_dim", "two_dim", "different_shapes", "python_floats"],
)
def test_bounds_without_mesh_must_have_shape_T(bounds):
    """Bounds are tensors of shape [T] on every call, with or without a mesh,
    so a loop whose first call has no mesh fails there, not one call later."""
    with pytest.raises(ValueError, match="shape"):
        _solver().integrate(f=torch.exp, take_gradient=False, **bounds)


@pytest.mark.parametrize(
    "bounds",
    [{}, {"mesh_init": _bound(0.0)}, {"mesh_final": _bound(1.0)}],
    ids=["none", "init_only", "final_only"],
)
def test_bounds_required_without_mesh(bounds):
    """Without a mesh both bounds are required: there are no default bounds."""
    with pytest.raises(ValueError, match="mesh_init and mesh_final"):
        _solver().integrate(f=torch.exp, take_gradient=False, **bounds)


def test_constructor_bounds_are_gone():
    """The constructor no longer takes bounds; each call gives its interval."""
    with pytest.raises(TypeError, match="mesh_init"):
        adaptive_quadrature("uniform", "gk21", device="cpu", mesh_init=_bound(0.0))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_bounds_inside_mesh_cut_it(dtype):
    """Bounds inside a given mesh cut it: the starting mesh is 1.2, 1.5, 2,
    2.5, 3, 3.3 (5 subintervals). The float64 mesh sets the dtype, so float32
    bounds are converted to float64."""
    lo, hi = _bound(1.2, dtype), _bound(3.3, dtype)
    out = _solver().integrate(
        f=torch.exp, mesh=_MESH, mesh_init=lo, mesh_final=hi, take_gradient=False
    )
    assert out.n_start_steps == 5
    assert torch.equal(out.mesh_init, lo.to(torch.float64))
    assert torch.equal(out.mesh_final, hi.to(torch.float64))
    exact = math.exp(float(hi)) - math.exp(float(lo))
    assert math.isclose(float(out.integral), exact, rel_tol=1e-6)


def test_one_bound_with_mesh_takes_the_other_from_mesh():
    """With a mesh, a missing bound is the mesh's end: [0, 3], 6 subintervals."""
    out = _solver().integrate(
        f=torch.exp, mesh=_MESH, mesh_final=_bound(3.0), take_gradient=False
    )
    assert torch.equal(out.mesh_init, _MESH[0])
    assert out.n_start_steps == 6


@pytest.mark.parametrize(
    "bounds",
    [{"mesh_init": _bound(-0.5)}, {"mesh_final": _bound(4.5)}],
    ids=["below", "above"],
)
def test_bound_outside_mesh_raises(bounds):
    """A bound outside the given mesh raises; extend the mesh to integrate further."""
    with pytest.raises(ValueError, match="outside the mesh"):
        _solver().integrate(f=torch.exp, mesh=_MESH, take_gradient=False, **bounds)


def test_bound_within_rounding_of_a_mesh_end_replaces_it():
    """A bound one float64 step past the mesh end replaces that end (no error,
    no near-zero-width subinterval): the interval is the bound's."""
    beyond = torch.nextafter(_MESH[-1], _MESH[-1] + 1)
    out = _solver().integrate(
        f=torch.exp, mesh=_MESH, mesh_final=beyond, take_gradient=False
    )
    assert torch.equal(out.mesh_final, beyond)
    assert out.n_start_steps == 8


#: An integrand whose integral is the interval's length.
_ONES = torch.ones_like


@pytest.mark.parametrize(
    ("mesh_final", "expected"),
    [(999.99995, 0.99995), (1000.00005, None)],
    ids=["inside_cuts", "outside_raises"],
)
def test_bound_near_a_large_mesh_end_is_not_snapped(mesh_final, expected):
    """At |t| ~ 1e3 a bound 5e-5 from the mesh end is a different interval: it
    cuts the mesh (inside) or raises (outside). A tolerance relative to the
    assertion rtol (1e-7 |t|) used to snap both to [999, 1000]."""
    mesh = torch.linspace(999.0, 1000.0, 9, dtype=torch.float64).unsqueeze(-1)
    call = {
        "f": _ONES,
        "mesh": mesh,
        "mesh_final": _bound(mesh_final),
        "take_gradient": False,
    }
    if expected is None:
        with pytest.raises(ValueError, match="outside the mesh"):
            _solver().integrate(**call)
        return
    out = _solver().integrate(**call)
    assert torch.equal(out.mesh_final, _bound(mesh_final))
    assert math.isclose(float(out.integral), expected, rel_tol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize(
    ("sampling", "method"),
    [("uniform", "gk21"), ("variable", "interpolatory3_variable")],
)
def test_bound_one_ulp_from_a_mesh_point_drops_that_point(sampling, method, dtype):
    """A bound one ULP above the interior point 3 would leave a one-ULP
    subinterval, for which variable sampling returned NaN; the point is
    dropped instead, so the starting mesh is 0, 0.5, ..., 2.5 and the bound."""
    mesh = torch.linspace(0.0, 4.0, 9, dtype=dtype).unsqueeze(-1)
    bound = torch.nextafter(mesh[6], mesh[7])
    solver = adaptive_quadrature(sampling, method, device="cpu", max_batch=10_000)
    out = solver.integrate(
        f=torch.exp, mesh=mesh, mesh_final=bound, take_gradient=False
    )
    assert out.n_start_steps == 6
    assert out.converged is True
    exact = math.exp(float(bound)) - 1.0
    assert math.isclose(float(out.integral), exact, rel_tol=1e-5)


def test_bound_near_zero_is_not_snapped():
    """The tolerance scales with the value compared, not with the mesh's
    largest end: a bound 1e-13 from the end 0 of a mesh reaching 1000 is a
    different interval (it was snapped to 0, where t^(-1/2) is infinite)."""
    mesh = torch.linspace(0.0, 1000.0, 11, dtype=torch.float64).unsqueeze(-1)
    out = _solver().integrate(
        f=_ONES, mesh=mesh, mesh_init=_bound(1e-13), take_gradient=False
    )
    assert torch.equal(out.mesh_init, _bound(1e-13))
    assert out.n_start_steps == 10
    assert math.isclose(float(out.integral), 1000.0 - 1e-13, rel_tol=1e-12)


def test_mesh_points_near_zero_are_kept():
    """A mesh alone is used as given: its point 1e-13 next to the end 0 is not
    dropped (5 subintervals)."""
    mesh = torch.tensor(
        [[0.0], [1e-13], [1e-6], [1.0], [500.0], [1000.0]], dtype=torch.float64
    )
    out = _solver().integrate(f=_ONES, mesh=mesh, take_gradient=False)
    assert out.n_start_steps == 5


def test_mesh_point_within_rounding_of_an_end_is_kept():
    """A mesh alone is used as given even where a cut would drop a point: the
    point one float64 step above the end 1 stays (2 subintervals). Only a bound
    that cuts the mesh drops the points within rounding of it."""
    one = _bound(1.0)
    mesh = torch.stack([one, torch.nextafter(one, one + 1), _bound(2.0)])
    out = _solver().integrate(f=_ONES, mesh=mesh, take_gradient=False)
    assert out.n_start_steps == 2


@pytest.mark.parametrize(
    "bound",
    [torch.tensor(2.0, dtype=torch.float64), torch.tensor([[2.0]]), 2.0],
    ids=["zero_dim", "two_dim", "python_float"],
)
def test_bound_of_wrong_shape_with_mesh_raises(bound):
    """With a mesh, a bound must be a tensor of shape [T]; other shapes raised
    an unrelated error from torch.concatenate."""
    with pytest.raises(ValueError, match="shape"):
        _solver().integrate(f=torch.exp, mesh=_MESH, mesh_final=bound)


def test_integrand_output_must_be_two_dimensional():
    """An integrand returning shape [N] instead of [N, D] raises ValueError
    (the check was an assert, which python -O removes)."""
    with pytest.raises(ValueError, match="shape"):
        _solver().integrate(
            f=lambda t: torch.exp(t)[:, 0], mesh=_MESH, take_gradient=False
        )


def test_float32_bound_at_a_float64_mesh_end_replaces_it():
    """float32(0.3) lies past the float64 mesh end 0.3 only by float32
    rounding (the coarser dtype sets the tolerance), so it raises no error and
    replaces that end."""
    mesh = torch.linspace(0.0, 0.3, 4, dtype=torch.float64).unsqueeze(-1)
    out = _solver().integrate(
        f=torch.exp,
        mesh=mesh,
        mesh_final=_bound(0.3, torch.float32),
        take_gradient=False,
    )
    assert torch.equal(out.mesh_final, _bound(0.3, torch.float32).to(torch.float64))
    assert out.n_start_steps == 3


def test_float32_bound_leaves_the_other_bounds_tolerance_unchanged():
    """Each bound's tolerance comes from its own dtype and the mesh's: a float32
    mesh_init must not snap the float64 mesh_final 999.9996, 4e-4 inside the
    end 1000 (far beyond float64 rounding), to that end."""
    mesh = torch.linspace(0.0, 1000.0, 11, dtype=torch.float64).unsqueeze(-1)
    out = _solver().integrate(
        f=_ONES,
        mesh=mesh,
        mesh_init=_bound(0.5, torch.float32),
        mesh_final=_bound(999.9996),
        take_gradient=False,
    )
    assert torch.equal(out.mesh_final, _bound(999.9996))
    assert math.isclose(float(out.integral), 999.4996, rel_tol=1e-12)


@pytest.mark.parametrize(
    "mesh",
    [
        torch.tensor([[1.0], [0.5], [0.0]], dtype=torch.float64),
        torch.tensor([[0.0], [0.5], [0.5], [1.0]], dtype=torch.float64),
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.linspace(0.0, 1.0, 3, dtype=torch.float64),
    ],
    ids=["reversed", "duplicate_point", "one_point", "one_dimensional"],
)
def test_invalid_mesh_raises(mesh):
    """A given mesh must be [M, T] with M >= 2 increasing points: ValueError
    (it was a bare AssertionError, which python -O removes)."""
    with pytest.raises(ValueError, match="mesh must be"):
        _solver().integrate(f=torch.exp, mesh=mesh, take_gradient=False)


def test_integrate_options_after_the_bounds_are_keyword_only():
    """Removing same_f (the 6th parameter) must not shift later positional
    arguments: everything after mesh_final is keyword-only."""
    with pytest.raises(TypeError, match="positional"):
        _solver().integrate(torch.exp, None, _MESH, None, None, True)


def test_solver_usable_after_a_raising_call():
    """A call whose integrand raises leaves the solver usable: the next call
    gives the exact integral."""

    def broken(t):
        if len(t) > 1:
            raise RuntimeError("boom")
        return torch.exp(t)

    solver = _solver()
    with pytest.raises(RuntimeError, match="boom"):
        solver.integrate(f=broken, mesh=_MESH, take_gradient=False)
    out = solver.integrate(f=torch.exp, mesh=_MESH, take_gradient=False)
    assert math.isclose(float(out.integral), math.exp(4.0) - 1.0, rel_tol=1e-6)


def test_bounds_equal_to_mesh_ends_leave_it_unchanged():
    """Bounds equal to the mesh's ends give the same call as the mesh alone."""
    alone = _solver().integrate(f=torch.exp, mesh=_MESH, take_gradient=False)
    with_bounds = _solver().integrate(
        f=torch.exp,
        mesh=_MESH,
        mesh_init=_MESH[0],
        mesh_final=_MESH[-1],
        take_gradient=False,
    )
    assert with_bounds.n_start_steps == alone.n_start_steps == 8
    assert torch.equal(with_bounds.integral, alone.integral)
    assert torch.equal(with_bounds.mesh_optimal, alone.mesh_optimal)


@pytest.mark.parametrize(
    "bounds",
    [
        {"mesh_init": _bound(1.0), "mesh_final": _bound(0.0)},
        {"mesh_init": _bound(1.0), "mesh_final": _bound(1.0)},
    ],
    ids=["reversed", "equal"],
)
def test_bounds_not_increasing_raise(bounds):
    """Bounds that are not increasing raise ValueError, which python -O
    keeps."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu", max_batch=10_000)
    with pytest.raises(ValueError, match="mesh_init < mesh_final"):
        solver.integrate(f=torch.exp, take_gradient=False, **bounds)


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("warm_bounds", [False, True], ids=["no_bounds", "bounds"])
def test_mesh_optimal_carries_no_autograd_graph(take_gradient, warm_bounds):
    """A bound that requires grad gets its gradient in its own call, but its
    autograd graph must not stay in mesh_optimal: a later call starting from it
    backpropagated into the earlier call's bound (hi.grad changed although
    hi was not passed), or, after a take_gradient=True call, raised "Trying
    to backward through the graph a second time"."""
    solver = adaptive_quadrature("uniform", "gk21", device="cpu", max_batch=10_000)
    hi = _bound(2.0).requires_grad_()
    first = solver.integrate(
        f=torch.exp, mesh_init=_bound(0.0), mesh_final=hi, take_gradient=take_gradient
    )
    if not take_gradient:
        first.integral.sum().backward()
    # d/dhi of the integral of e^t over [0, hi] is e^hi (the bound's gradient).
    assert math.isclose(float(hi.grad), math.exp(2.0), rel_tol=1e-6)
    assert not first.mesh_optimal.requires_grad
    grad_before = hi.grad.clone()

    bounds = (
        {"mesh_init": _bound(0.0), "mesh_final": _bound(2.0)} if warm_bounds else {}
    )
    warm = solver.integrate(
        f=torch.exp, mesh=first.mesh_optimal, take_gradient=True, **bounds
    )

    assert math.isclose(float(warm.integral), math.exp(2.0) - 1.0, rel_tol=1e-6)
    assert not warm.mesh_final.requires_grad
    assert torch.equal(hi.grad, grad_before)


# -----------------------------------------------------------------------------
# converged on a normal call
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
def test_normal_completion_has_converged_true(take_gradient):
    """A normal integration call returns ``converged=True``: every accepted step
    meets the tolerance against the final integral. Seeded, because the random
    initial mesh decides how close the worst step gets to the tolerance.
    """
    torch.manual_seed(0)
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
    # The loop ended by accepting the non-finite panel rather than hanging, but
    # a NaN error does not meet the tolerance, so the result is not converged.
    assert out.converged is False


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
    # Terminated, but a NaN error does not meet the tolerance.
    assert out.converged is False


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
    stopped earlier by the scalar `mesh_init < mesh_final` check in
    _setup_integral_bounds (T>1 is not supported end to end yet), so for T=2
    that method is stubbed on this instance to reach the test call.
    """
    solver = make_solver_for_unit_test()
    if T > 1:
        monkeypatch.setattr(
            solver, "_setup_integral_bounds", lambda mesh, a, b: (mesh, a, b)
        )
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
