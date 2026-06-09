"""Tests for the ``error_integral_reference`` argument of ``integrate()``.

In the default *absolute* error mode (``use_absolute_error_ratio=True``) a
panel's accept/reject tolerance is ``atol + rtol * |reference_integral|``. By
default the reference is the *running* integral, which is accumulated batch by
batch and is therefore incomplete during the early batches: it has not yet
picked up contributions from regions evaluated later. When much of the
integral's mass lives at later times (a growing ``exp(a*t)``), that early
denominator is too small and the adaptive controller over-refines the early
panels. ``error_integral_reference`` lets the caller supply a value close to the
true total so the rtol denominator is correct from the first batch.

The *direction* of the step-count change depends on where the integral's mass
sits in time: late-mass integrands benefit (fewer steps), early-mass integrands
barely change (the running integral is already near its final value early on).

Step count is read from ``result.nodes.shape[0]`` — the number of panels the
adaptive loop actually evaluated and accepted at convergence — not from
``mesh_optimal`` (the post-convergence prune-and-refine mesh, which is one step
removed from what the loop did and can mask the signal).

Small ``max_batch`` values are used so the running integral genuinely
accumulates over several batches (otherwise a single batch sees the whole
integral at once and the "incomplete early reference" effect never appears).
``atol`` is kept far below ``rtol * |I|`` so the behavior is rtol-dominated; if
``atol`` floored the denominator the reference would have no effect.
"""

from __future__ import annotations

import math

import torch
from _helpers import REMOVE_CUT

from padaquad import adaptive_quadrature, steps

# rtol-dominated regime: atol must stay well below rtol * |I|.
ATOL = 1e-12
RTOL = 1e-5

T_INIT = torch.tensor([0.0], dtype=torch.float64)
T_FINAL = torch.tensor([1.0], dtype=torch.float64)

# Small batch so the running integral accumulates over many batches, processed
# roughly left-to-right. This is what exposes the early-batch incompleteness:
# with a large batch the controller sees (nearly) the whole domain at once and
# the running integral is already near its final value the first time any panel
# is judged, so a reference has nothing to correct. bosh3 has C=4 quadrature
# points per panel, so max_batch=16 fits ~4 panels per batch.
MAX_BATCH = 16

# A low-order method keeps per-panel error near the accept/reject threshold, so
# the rtol denominator (which the reference controls) actually decides splits.
# A high-order rule like gk21 converges so far below tolerance on exp(a*t) that
# the denominator never changes any decision and the reference is invisible.
METHOD = "bosh3"

# Growth/decay rate. Large enough that exp(a*t) puts most of its mass at late t.
A = 8.0


def _make_solver(use_absolute_error_ratio: bool = True):
    """Low-order uniform solver pinned to CPU/float64 in absolute-error mode."""
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=METHOD,
        atol=ATOL,
        rtol=RTOL,
        remove_cut=REMOVE_CUT,
        device="cpu",
        dtype=torch.float64,
        use_absolute_error_ratio=use_absolute_error_ratio,
    )


def _growing(t, *args):
    """f(t) = exp(A*t): integral mass concentrated at late times. [N,1]."""
    return torch.exp(A * t)


def _decaying(t, *args):
    """f(t) = exp(-A*t): integral mass concentrated at early times. [N,1]."""
    return torch.exp(-A * t)


def _growing_integral() -> float:
    """Analytic ∫_0^1 exp(A*t) dt = (exp(A) - 1) / A."""
    return (math.exp(A) - 1.0) / A


def _decaying_integral() -> float:
    """Analytic ∫_0^1 exp(-A*t) dt = (1 - exp(-A)) / A."""
    return (1.0 - math.exp(-A)) / A


def _n_steps(result) -> int:
    """Number of accepted panels at convergence (the controller's actual work)."""
    return result.nodes.shape[0]


def _run(solver, f, **kwargs):
    """Integrate over [0, 1] with a forced small batch and no gradient."""
    return solver.integrate(
        f=f,
        mesh_init=T_INIT,
        mesh_final=T_FINAL,
        take_gradient=False,
        max_batch=MAX_BATCH,
        **kwargs,
    )


def _assert_accurate(result, true_integral, rtol=1e-3):
    """The integral should match the analytic value regardless of step count."""
    got = float(result.integral.reshape(-1)[0])
    assert math.isclose(got, true_integral, rel_tol=rtol), (
        f"integral {got} != analytic {true_integral}"
    )


def test_reference_reduces_steps_for_late_mass():
    """Late-mass integrand: a good reference cures early over-refinement.

    Without a reference, the early batches judge panels against a tiny running
    integral and over-refine. Supplying ≈ the true total enlarges the early
    denominator, so fewer panels are needed for the same accuracy.
    """
    true_I = _growing_integral()

    torch.manual_seed(0)
    baseline = _run(_make_solver(), _growing)

    torch.manual_seed(0)
    with_ref = _run(_make_solver(), _growing, error_integral_reference=true_I)

    assert _n_steps(with_ref) < _n_steps(baseline), (
        f"expected fewer steps with reference, got with_ref={_n_steps(with_ref)} "
        f"baseline={_n_steps(baseline)}"
    )
    _assert_accurate(baseline, true_I)
    _assert_accurate(with_ref, true_I)


def test_reference_little_effect_for_early_mass():
    """Early-mass integrand: a reference barely changes the step count.

    For exp(-A*t) the running integral is already near its final value by the
    time later panels are judged, so the reference has little to correct. We
    require it does no harm (<=) and stays close.
    """
    true_I = _decaying_integral()

    torch.manual_seed(0)
    baseline = _run(_make_solver(), _decaying)

    torch.manual_seed(0)
    with_ref = _run(_make_solver(), _decaying, error_integral_reference=true_I)

    n_base = _n_steps(baseline)
    n_ref = _n_steps(with_ref)
    assert n_ref <= n_base, (
        f"reference should not increase steps for early-mass integrand: "
        f"with_ref={n_ref} baseline={n_base}"
    )
    # "Close": within 20% of baseline (encodes the small-effect nuance).
    assert n_ref >= 0.8 * n_base, (
        f"reference unexpectedly changed early-mass step count a lot: "
        f"with_ref={n_ref} baseline={n_base}"
    )
    _assert_accurate(baseline, true_I)
    _assert_accurate(with_ref, true_I)


def test_too_small_reference_inflates_early_steps():
    """A too-small reference shrinks the denominator and over-refines early.

    Passing 1% of the true integral makes atol + rtol*|ref| tiny, so the
    controller splits aggressively. The extra panels concentrate in the
    early-time region (the first integration steps). Compared against a good
    reference, the too-small run has both more total panels and more panels
    whose left edge lies in the first half of the domain.
    """
    true_I = _growing_integral()
    midpoint = 0.5 * float(T_INIT + T_FINAL)

    torch.manual_seed(0)
    good = _run(_make_solver(), _growing, error_integral_reference=true_I)

    torch.manual_seed(0)
    too_small = _run(
        _make_solver(), _growing, error_integral_reference=0.01 * true_I
    )

    assert _n_steps(too_small) > _n_steps(good), (
        f"too-small reference should inflate step count: "
        f"too_small={_n_steps(too_small)} good={_n_steps(good)}"
    )

    # Left edge of each accepted panel: nodes[:, 0, 0].
    good_left = good.nodes[:, 0, 0]
    small_left = too_small.nodes[:, 0, 0]
    good_early = int((good_left < midpoint).sum())
    small_early = int((small_left < midpoint).sum())
    assert small_early > good_early, (
        f"extra steps should concentrate in the early region: "
        f"too_small_early={small_early} good_early={good_early}"
    )

    _assert_accurate(good, true_I)


def test_reference_accepts_scalar_and_tensor():
    """The coercion path accepts float / 0-d tensor / 1-element tensor.

    Exercises the just-fixed ``isinstance`` line: all three forms must run and
    produce the same integral and step count.
    """
    true_I = _growing_integral()
    refs = [
        true_I,  # Python float
        torch.tensor(true_I),  # 0-d tensor
        torch.tensor([true_I]),  # 1-element tensor
    ]

    results = []
    for ref in refs:
        torch.manual_seed(0)
        results.append(_run(_make_solver(), _growing, error_integral_reference=ref))

    n_steps = [_n_steps(r) for r in results]
    assert len(set(n_steps)) == 1, f"step counts differ across ref forms: {n_steps}"
    integrals = [float(r.integral.reshape(-1)[0]) for r in results]
    assert all(math.isclose(i, integrals[0], rel_tol=1e-12) for i in integrals), (
        f"integrals differ across ref forms: {integrals}"
    )
    for r in results:
        _assert_accurate(r, true_I)


def test_reference_no_effect_in_cumulative_mode():
    """In cumulative mode the reference is ignored (denominator is the cumsum).

    The argument targets absolute mode only; with
    ``use_absolute_error_ratio=False`` it must not change the step count.
    """
    true_I = _growing_integral()

    torch.manual_seed(0)
    baseline = _run(_make_solver(use_absolute_error_ratio=False), _growing)

    torch.manual_seed(0)
    with_ref = _run(
        _make_solver(use_absolute_error_ratio=False),
        _growing,
        error_integral_reference=true_I,
    )

    assert _n_steps(with_ref) == _n_steps(baseline), (
        f"reference must not affect cumulative mode: "
        f"with_ref={_n_steps(with_ref)} baseline={_n_steps(baseline)}"
    )
    _assert_accurate(baseline, true_I)
    _assert_accurate(with_ref, true_I)
