"""Adversarial tests of the batch size, the state and the backward restart.

Written by a bug hunt of the ``change_memory`` branch (report, deleted once
the bugs were fixed: ``docs/issues/2026-11-18-memory-state-bughunt.md`` in
commit ``d0bcbdc``). The first eight tests
pin the bugs it found (all fixed); the others pin behavior it checked and
found correct. Everything runs on the CPU
with simulated out-of-memory errors, except the tests marked ``needs_cuda``.
"""

from __future__ import annotations

import gc
import logging
import math
import weakref

import numpy as np
import pytest
import torch
from _helpers import needs_cuda

from padaquad import BackwardOutOfMemoryError, IntegrationState, adaptive_quadrature

A = torch.tensor([0.0], dtype=torch.float64)
EXACT = (1 - math.cos(40.0)) / 40 + 0.5  # integral of sin(40 t) + t on [0, 1]


def solver(method="gk21", **kwargs):
    return adaptive_quadrature(
        "uniform", method, device="cpu", atol=1e-10, rtol=1e-10, **kwargs
    )


def run(s, *args, **kwargs):
    """Call ``s.integrate`` and return ``(result, state)``."""
    return s.integrate(*args, **kwargs)


def even_mesh(n_subintervals: int) -> torch.Tensor:
    return torch.linspace(0, 1, n_subintervals + 1, dtype=torch.float64)[:, None]


def _fail_backward(grad):
    raise torch.OutOfMemoryError("simulated backward")


def _leaves(t: torch.Tensor) -> list[torch.Tensor]:
    """The leaf tensors t's autograd graph reaches (the AccumulateGrad nodes)."""
    found, stack, seen = [], [t.grad_fn], set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if getattr(node, "variable", None) is not None:
            found.append(node.variable)
        stack.extend(n for n, _ in node.next_functions)
    return found


# ---------------------------------------------------------------------------
# Bugs found by the hunt, fixed
# ---------------------------------------------------------------------------


def test_restart_releases_the_failed_batch():
    """After a backward out-of-memory error, padaquad restarts the integral by
    calling ``integrate`` again from inside the failed call. The tensors the
    failed batch saved for its backward pass (here ``u``, saved by the product
    ``w * u``) must be freed before the restarted call evaluates f: the error
    says memory ran out, and the restart is meant to fit in what is left. With
    the bug, the failed call's frame still references the batch's outputs, so
    every failed batch stays alive (one more per nested restart)."""
    w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    failed_refs: list[weakref.ref] = []
    alive_at_restart: list[int] = []
    pending = [False]

    def f(t):
        if pending[0]:
            pending[0] = False
            gc.collect()
            alive_at_restart.append(sum(r() is not None for r in failed_refs))
        u = torch.sin(40 * t)
        y = w * u
        if len(t) > 5000 and y.requires_grad:
            failed_refs.append(weakref.ref(u))

            def hook(grad):
                pending[0] = True
                raise torch.OutOfMemoryError("simulated backward")

            y.register_hook(hook)
        return y

    run(solver(), f, state=IntegrationState(even_mesh(500), 100, False))
    assert alive_at_restart, "no restart happened"
    assert alive_at_restart == [0] * len(alive_at_restart)


@pytest.mark.parametrize("leaf", ["made_in_f", "mesh_copy"])
def test_saved_gradients_do_not_keep_every_batch_alive(leaf):
    """With take_gradient=True, the dictionary of saved gradients holds every
    leaf tensor of every batch's graph for the whole call. Two cases:
    ``made_in_f``, an integrand that differentiates with respect to its input
    (a new leaf ``x = t.detach().requires_grad_()`` per batch, common in
    physics-informed losses), and ``mesh_copy``, padaquad's own detached copy
    of the mesh (``mesh_eval``) made per batch when a bound requires grad.
    Each batch's leaves should be freed once the batch is done (on ``main``
    at most the previous batch's is alive); with the bug all of them are,
    so a long run holds every batch's leaf and its gradient on the device."""
    w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    refs: list[weakref.ref] = []
    alive: list[int] = []

    def f(t):
        gc.collect()
        alive.append(sum(r() is not None for r in refs))
        if leaf == "made_in_f":
            x = t.detach().requires_grad_()
            refs.append(weakref.ref(x))
            return w * torch.sin(40 * x) + 0 * x
        refs.extend(weakref.ref(v) for v in _leaves(t))
        return torch.sin(40 * t) + t

    kwargs = {}
    if leaf == "mesh_copy":
        kwargs["mesh_final"] = torch.tensor([1.0], dtype=torch.float64).requires_grad_()
    run(solver(), f, mesh=even_mesh(200), max_batch=230, **kwargs)
    assert len(alive) > 10
    assert max(alive) <= 1, f"leaves of earlier batches alive: {alive}"


@pytest.mark.parametrize("through", ["mesh", "state"])
def test_restart_keeps_the_gradient_of_a_mesh_that_requires_grad(through):
    """A mesh that requires grad (no bounds given) receives the gradient of
    the integral through its end points: d/db = f(b), d/da = -f(a). After a
    backward out-of-memory error the restart starts from ``start_mesh``, a
    detached copy of the starting mesh, and the first attempt's ``mesh_grad``
    is dropped, so the caller's mesh gets no gradient at all (None), while
    an uninterrupted run gives f(1) = sin(40) + 1 at the last point."""

    def attempt(limit):
        mesh = even_mesh(500).clone().requires_grad_()

        def f(t):
            y = torch.sin(40 * t) + t
            if len(t) > limit and y.requires_grad:
                y = y * 1.0
                y.register_hook(_fail_backward)
            return y

        if through == "mesh":
            run(solver(), f, mesh=mesh, max_batch=11500)
        else:
            run(solver(), f, state=IntegrationState(mesh, 11500, True))
        return mesh.grad

    reference = attempt(math.inf)
    assert float(reference[-1, 0]) == pytest.approx(math.sin(40.0) + 1, rel=1e-8)
    restarted = attempt(5000)
    assert restarted is not None
    assert torch.allclose(restarted, reference, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("max_batch", [np.int64(5000)])
def test_result_state_after_a_numpy_or_float_max_batch(max_batch):
    """``integrate(max_batch=np.int64(5000))`` runs (as on ``main``), but the
    batch size is kept as given, and building the returned state raises
    ValueError ("max_batch must be an int"), so the hand-over of the state to
    the next call fails (also ``examples/pode/trainer.py``'s
    ``dataclasses.replace(state, ...)``). integrate now converts an integer
    (``operator.index``) and rejects a float up front
    (``test_invalid_max_batch_is_rejected_before_any_work``)."""
    _, state = run(
        solver(),
        lambda t: torch.sin(40 * t) + t,
        mesh_init=A,
        mesh_final=A + 1,
        take_gradient=False,
        max_batch=max_batch,
    )
    assert state.max_batch == 5000


def test_out_of_memory_retry_shrinks_the_call_that_failed():
    """A state of 100 points, not converged, and a pass of 5,750 points (250
    gk21 subintervals). The first call is 1,000 points; the planned size then
    grows to 10,000, but only 4,750 points are left, and that call fails
    (limit 4,500). The retry should be 0.75 x 4,750 = 3,562 points, which
    runs: one failed call. With the bug the retry is 0.75 x 10,000 = 7,500,
    so the same 4,750 points are evaluated again (and fail) until 0.75^k x
    10,000 drops below 4,750: three failed calls, and the warning says
    "Reducing max_batch from 10000 to 7500" for a call of 4,750 points."""
    sizes: list[int] = []

    def f(t):
        sizes.append(len(t))
        if len(t) > 4500:
            raise torch.OutOfMemoryError("simulated")
        return torch.sin(40 * t) + t

    result, _ = run(
        solver(),
        f,
        state=IntegrationState(even_mesh(250), 100, False),
        take_gradient=False,
    )
    assert math.isclose(float(result.integral), EXACT, abs_tol=1e-9)
    assert sum(n > 4500 for n in sizes) == 1, sizes[:8]


def test_small_unconverged_state_grows_to_one_subinterval_with_gradients():
    """A state whose batch size has not converged is allowed to grow. With
    ``take_gradient=True`` a call needs whole subintervals (gk21: 23 points),
    but the dispatcher takes ``min(n_waiting, 10 * 2) // 23 = 0``
    subintervals and ``_evaluate_f_on_full_nodes`` fails its assertion "Not
    enough free memory ...", although nothing ran out of memory and the batch
    could grow. (With ``take_gradient=False`` the same state works.)"""
    w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    run(
        solver(),
        lambda t: w * torch.sin(40 * t),
        state=IntegrationState(even_mesh(4), 2, False),
    )
    assert float(w.grad) == pytest.approx((1 - math.cos(40.0)) / 40, rel=1e-8)


def test_refused_restart_puts_back_gradients_it_can():
    """The backward pass of the third batch (one gk21 subinterval each, the
    given ``max_batch=23``) fails; a smaller batch cannot hold a subinterval,
    so padaquad raises ``BackwardOutOfMemoryError`` and asks the user to reset
    the gradients. The walk had saved every gradient, so it could put them
    back first: here ``w.grad`` held 1.0 from an earlier loss before the call,
    and resetting it as advised loses that 1.0. Expected: ``w.grad`` is 1.0
    again when the error is raised."""
    w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    w.grad = torch.tensor([1.0], dtype=torch.float64)
    calls = [0]

    def f(t):
        y = w * torch.sin(40 * t)
        if len(t) > 1:
            calls[0] += 1
            if calls[0] == 3:
                y.register_hook(_fail_backward)
        return y

    with pytest.raises(BackwardOutOfMemoryError, match="cannot hold one"):
        run(solver(), f, mesh=even_mesh(20), max_batch=23)
    assert float(w.grad) == 1.0


@needs_cuda
def test_backward_ratio_is_not_computed_from_a_stale_forward_peak(monkeypatch, caplog):
    """The check before a backward pass needs r F <= usable memory, where r is
    the previous batch's backward peak over ``_forward_bytes``. ``_forward_bytes``
    is written only when f's output requires grad. Call 1 on the solver
    leaves a large ``_forward_bytes`` (an integrand with a 4096-wide
    temporary). In call 2 the left batches' outputs need no gradient, but
    ``loss_fxn`` adds a parameter term, so their backward passes run and set r
    to (a few bytes) / (call 1's peak), about 5e-5 (measured). The first batch
    with t > 0.5 then needs about 4 bytes and passes the check with 1 kB
    usable, so its backward pass runs unchecked; only the batch after it, with
    a ratio measured on this call, is dropped. A fresh solver
    (``test_backward_check_control_on_a_fresh_solver``) drops the first one:
    one call of 2,300 points reaching t > 0.5, not two."""
    s = adaptive_quadrature("uniform", "gk21", device="cuda", atol=1e-10, rtol=1e-10)
    w = torch.tensor([1.3], dtype=torch.float64, device="cuda", requires_grad=True)
    # The value an earlier call left (the call's attributes are now put back
    # when it ends, so this also stands for a value left inside the call).
    s._forward_bytes = 2e7
    assert _run_mixed_call(s, w, monkeypatch, caplog) == 1


def _run_mixed_call(s, w, monkeypatch, caplog):
    """Batches of 100 subintervals left to right; f's output requires grad
    only for t > 0.5; the loss always requires grad (a parameter term).
    Returns the number of calls of 2,300 points that reach t > 0.5."""
    p = torch.tensor([0.5], dtype=torch.float64, device="cuda", requires_grad=True)
    calls = []

    def f(t):
        calls.append(len(t) == 2300 and bool((t > 0.5).any()))
        y = torch.sin(40 * t)
        if bool((t > 0.5).any()):
            y = y + w * torch.clamp(t - 0.5, min=0)
        return y

    monkeypatch.setattr(s, "_get_usable_memory", lambda tmu: 1e-6)  # 1 kB
    with caplog.at_level(logging.WARNING):
        run(
            s,
            f,
            mesh=even_mesh(500).cuda(),
            max_batch=2300,
            loss_fxn=lambda r: r.integral.sum() + 0 * p.sum(),
        )
    assert "predicted not to fit" in caplog.text
    return sum(calls)


# ---------------------------------------------------------------------------
# Checked and found correct (pass)
# ---------------------------------------------------------------------------


@needs_cuda
def test_backward_check_control_on_a_fresh_solver(monkeypatch, caplog):
    """Control for Bug 4: the same mixed call on a fresh solver (no stale
    ``_forward_bytes``) drops the first batch whose backward pass does not
    fit."""
    s = adaptive_quadrature("uniform", "gk21", device="cuda", atol=1e-10, rtol=1e-10)
    w = torch.tensor([1.3], dtype=torch.float64, device="cuda", requires_grad=True)
    assert _run_mixed_call(s, w, monkeypatch, caplog) == 1


def test_restart_keeps_every_other_argument():
    """A backward restart repeats the call with every argument: the
    constructor's integrand (``f=None``) and ``y0``, ``f_args``,
    ``loss_fxn``, ``error_integral_reference`` and a bound that requires grad.
    Integral, loss and both gradients equal an uninterrupted run's."""

    def attempt(limit):
        w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def f(t, c):
            y = w * torch.sin(40 * t) * c
            if len(t) > limit and y.requires_grad:
                y.register_hook(_fail_backward)
            return y

        result, _ = run(
            solver(f=f, y0=torch.tensor([2.0], dtype=torch.float64)),
            mesh_init=A,
            mesh_final=b,
            f_args=(torch.tensor(1.5, dtype=torch.float64),),
            loss_fxn=lambda r: 2 * r.integral.sum(),
            state=IntegrationState(None, 100, False),
            error_integral_reference=0.1,
        )
        return [
            float(result.integral),
            float(result.loss),
            float(w.grad),
            float(b.grad),
        ]

    assert attempt(3000) == pytest.approx(attempt(math.inf), rel=1e-10)


@pytest.mark.parametrize(
    ("kind", "method"), [("uniform", "gk21"), ("variable", "interpolatory3_variable")]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_test_calls_with_both_solvers_and_dtypes(kind, method, dtype):
    """The test calls run with the variable-sampling solver and in float32,
    and an f that returns tracked variables and takes f_args."""
    a = torch.tensor([0.0], dtype=dtype)

    def f(t, scale):
        return torch.cos(t) * scale, (2 * t,)

    s = adaptive_quadrature(kind, method, device="cpu", atol=1e-6, rtol=1e-6)
    result, state = run(
        s, f, mesh_init=a, mesh_final=a + 1, take_gradient=False, f_args=(2.0,)
    )
    assert float(result.integral) == pytest.approx(2 * math.sin(1.0), abs=1e-5)
    assert torch.equal(result.tracked_variables[0], 2 * result.nodes)
    assert state.max_batch >= 10


@pytest.mark.parametrize(
    ("limit", "start", "converged", "n", "method"),
    [
        (61, 7, False, 20, "gk21"),
        (500, 1, False, 120, "dopri5"),
        (777, 2900, True, 400, "cc9"),
        (3001, 13, False, 400, "gk21"),
        (math.inf, 3, False, 3, "cc9"),
    ],
)
def test_conserve_memory_growth_and_retries_keep_values_aligned(
    limit, start, converged, n, method
):
    """``conserve_memory=True`` with growth inside one pass and out-of-memory
    retries: the integral is right, and every returned value and tracked
    variable is the one f returned for its node."""

    def f(t):
        if len(t) > limit:
            raise torch.OutOfMemoryError("simulated")
        return torch.sin(40 * t) + t, (2 * t,)

    result, _ = run(
        solver(method),
        f,
        state=IntegrationState(even_mesh(n), start, converged),
        take_gradient=False,
        conserve_memory=True,
    )
    assert float(result.integral) == pytest.approx(EXACT, abs=1e-8)
    assert torch.equal(result.y, torch.sin(40 * result.nodes) + result.nodes)
    assert torch.equal(result.tracked_variables[0], 2 * result.nodes)
