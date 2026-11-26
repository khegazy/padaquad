"""Tests for the batch size of ``integrate`` and the state it returns.

Without ``max_batch`` or a ``state``, the first call of ``f`` is the first pass,
at most 10,000 points; no call of ``f`` comes before it, so every point ``f``
is called at belongs to the integration. While the batch size has not
converged, a call of ``f`` that needs more points tries 10 times as many; an
out-of-memory error shrinks it by 0.75 and makes it converged.
``result_state`` carries the mesh and the batch size to the next call. With
``take_gradient=True`` an out-of-memory error in a backward pass puts back the
gradients the call added (found by walking each batch's autograd graph) and
restarts the integral, or raises ``BackwardOutOfMemoryError`` when they cannot
all be found.

Every test runs on the CPU with simulated out-of-memory errors, except the
check before a backward pass, which needs the GPU's peak-memory counter.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import pickle

import pytest
import torch
from _helpers import needs_cuda

from padaquad import (
    BackwardOutOfMemoryError,
    IntegrationState,
    adaptive_quadrature,
)
from padaquad.quadrature.memory import _is_out_of_memory

A = torch.tensor([0.0], dtype=torch.float64)
B = torch.tensor([1.0], dtype=torch.float64)
EXACT = (1 - math.cos(40.0)) / 40 + 0.5  # integral of sin(40 t) + t on [0, 1]
D_W = (1 - math.cos(40.0)) / 40  # d/dw of the integral of w sin(40 t)
D_V = 0.005  # d/dv of the integral of v max(t - 0.9, 0)


class Recorder:
    """sin(40 t) + t that records each call's size and runs out of memory above
    ``limit`` points (``torch.OutOfMemoryError``, as on a GPU)."""

    def __init__(self, limit: float = math.inf):
        self.limit, self.sizes = limit, []

    def __call__(self, t):
        self.sizes.append(len(t))
        if len(t) > self.limit:
            raise torch.OutOfMemoryError(f"simulated, {len(t)} points")
        return torch.sin(40 * t) + t


def solver(method="gk21", **kwargs):
    return adaptive_quadrature(
        "uniform", method, device="cpu", atol=1e-10, rtol=1e-10, **kwargs
    )


def even_mesh(n_subintervals: int) -> torch.Tensor:
    return torch.linspace(0, 1, n_subintervals + 1, dtype=torch.float64)[:, None]


# ---------------------------------------------------------------------------
# The first call, growth, convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("take_gradient", [False, True])
@pytest.mark.parametrize("n_subintervals", [11, 48, 500])
def test_first_call_is_the_first_pass_up_to_10000_points(
    n_subintervals, take_gradient, caplog
):
    """No call of f comes before the first pass, and the first call is the
    whole first pass (gk21: 21 nodes per subinterval, so 231, 1,008 and 10,500
    points), at most 10,000 points (with take_gradient=True whole
    subintervals: 9,996); the rest of the pass follows in one call. No memory
    benchmark runs and none of its messages is logged."""
    f = Recorder()
    with caplog.at_level(logging.WARNING):
        result, result_state = solver().integrate(
            f, mesh=even_mesh(n_subintervals), take_gradient=take_gradient
        )
    first_pass = 21 * n_subintervals
    first_call = min(first_pass, 10_000 // 21 * 21 if take_gradient else 10_000)
    assert f.sizes[0] == first_call
    if first_call < first_pass:  # the rest of the first pass
        assert f.sizes[1] == first_pass - first_call
    assert result_state.max_batch >= first_call
    assert not result_state.max_batch_converged
    assert math.isclose(float(result.integral), EXACT, abs_tol=1e-9)
    assert "measuring how much memory" not in caplog.text


@pytest.mark.parametrize(
    ("take_gradient", "first_calls", "batch"),
    [
        # Each failed call shrinks by 0.75 (rounded half to even).
        (False, [231, 173, 130, 98, 74, 56], 56),
        # Whole subintervals of 21 points: 0.75 x 231 = 173 -> 8 subintervals.
        (True, [231, 168, 126, 84, 63, 42], 42),
    ],
)
def test_failed_first_call_shrinks_until_one_runs(take_gradient, first_calls, batch):
    """The first pass of the default mesh (231 points) runs out of memory
    (limit 60): it is evaluated again at 0.75 times the size of the call that
    failed until a call runs, which becomes the converged batch size; every
    later call is at most that size."""
    f = Recorder(limit=60)
    result, result_state = solver().integrate(
        f, mesh_init=A, mesh_final=B, take_gradient=take_gradient
    )
    assert f.sizes[: len(first_calls)] == first_calls
    assert max(f.sizes[len(first_calls) :]) <= batch
    assert (result_state.max_batch, result_state.max_batch_converged) == (batch, True)
    assert math.isclose(float(result.integral), EXACT, abs_tol=1e-9)


@pytest.mark.parametrize(("method", "limit"), [("gk21", 21), ("cc65", 70)])
def test_integrand_that_holds_one_subinterval_runs_with_take_gradient(method, limit):
    """With take_gradient=True an integrand that holds one subinterval (C
    points) but not two integrates: the failed first pass shrinks in whole
    subintervals down to one. The former test calls (10, 100, then 75, 56, ...
    points) could converge below C (18 for gk21 at a limit of 21, 56 for cc65
    at 70), and the first batch, of no whole subinterval, then raised
    AssertionError."""
    s = solver(method)
    f = Recorder(limit=limit)
    result, result_state = s.integrate(f, mesh_init=A, mesh_final=B, take_gradient=True)
    assert (result_state.max_batch, result_state.max_batch_converged) == (s.C, True)
    assert math.isclose(float(result.integral), EXACT, abs_tol=1e-9)


START_CASES = {
    "max_batch": {"mesh_init": A, "mesh_final": B, "max_batch": 10**6},
    "state": "state",
    "default_mesh": {"mesh_init": A, "mesh_final": B},
    **{f"mesh_{n}": {"mesh": even_mesh(n)} for n in (1, 3, 50, 500)},
}


@pytest.mark.parametrize("limit", [math.inf, 60])
@pytest.mark.parametrize("take_gradient", [False, True])
@pytest.mark.parametrize(
    ("sampling", "method"),
    [
        ("uniform", "gk21"),
        ("uniform", "bosh3"),
        ("uniform", "cc17"),
        ("variable", "interpolatory3_variable"),
    ],
)
@pytest.mark.parametrize("start", list(START_CASES))
def test_every_point_of_f_belongs_to_the_integration(
    start, sampling, method, take_gradient, limit
):
    """The points f is called at, in the calls that ran, are exactly C per
    subinterval the integration evaluated: no call of f is made for anything
    else, whatever the call is given (max_batch, a state, neither; small and
    large first passes), for both values of take_gradient, and also when f runs
    out of memory above 60 points (its failed calls are not counted). With
    neither max_batch nor a state the first call is the first pass, at most
    10,000 points (whole subintervals with take_gradient=True)."""
    s = adaptive_quadrature(sampling, method, device="cpu", atol=1e-8, rtol=1e-8)
    kwargs = START_CASES[start]
    if kwargs == "state":
        _, state = s.integrate(Recorder(), mesh_init=A, mesh_final=B)
        kwargs = {"state": state}
    evaluate = s._evaluate_f_on_mesh
    subintervals = [0]

    def spy(*args, **kw):
        out = evaluate(*args, **kw)
        subintervals[0] += len(out[0])
        return out

    s._evaluate_f_on_mesh = spy
    f = Recorder(limit)
    torch.manual_seed(0)
    result, _ = s.integrate(f, take_gradient=take_gradient, **kwargs)
    # bosh3's error estimate from a state's mesh is 4.5e-6 off at rtol 1e-8.
    assert math.isclose(float(result.integral), EXACT, abs_tol=1e-5)
    ran = [n for n in f.sizes if n <= limit]
    assert sum(ran) == s.C * subintervals[0]
    if math.isinf(limit) and start not in ("max_batch", "state"):
        first_pass = s.C * result.n_start_steps
        cap = 10_000 // s.C * s.C if take_gradient else 10_000
        assert f.sizes[0] == min(first_pass, cap)


@pytest.mark.parametrize("trade_speed_for_memory", [False, True])
def test_batch_grows_tenfold_per_call_until_out_of_memory(trade_speed_for_memory):
    """A state with batch size 100, not converged, and a pass of 11,508 points
    (548 gk21 subintervals). Calls of 1,000, 10,000, then the 508 left,
    whether the values stay on the device or not (``trade_speed_for_memory``); with a
    limit of 3,000 points the call of 10,000 fails and shrinks by 0.75 per
    retry until it fits, and the state converges. Both give the right
    integral."""
    state = IntegrationState(even_mesh(548), 100, False)
    f = Recorder()
    result, result_state = solver().integrate(
        f,
        state=state,
        take_gradient=False,
        trade_speed_for_memory=trade_speed_for_memory,
    )
    assert math.isclose(float(result.integral), EXACT, abs_tol=1e-9)
    assert f.sizes[:3] == [1000, 10000, 508]
    assert (result_state.max_batch, result_state.max_batch_converged) == (
        10000,
        False,
    )

    f = Recorder(limit=3000)
    result, result_state = solver().integrate(
        f,
        state=state,
        take_gradient=False,
        trade_speed_for_memory=trade_speed_for_memory,
    )
    assert math.isclose(float(result.integral), EXACT, abs_tol=1e-9)
    assert result_state.max_batch_converged
    assert result_state.max_batch <= 3000
    assert max(n for n in f.sizes if n <= 3000) == result_state.max_batch


def test_given_max_batch_is_fixed_and_converged():
    """A max_batch given to the call or to the constructor is used as it is
    (no growth) and returned as converged; given with a state,
    it replaces the state's."""
    f = Recorder()
    result, result_state = solver().integrate(
        f, mesh=even_mesh(46), max_batch=50, take_gradient=False
    )
    assert max(f.sizes) <= 50
    assert (result_state.max_batch, result_state.max_batch_converged) == (50, True)
    f = Recorder()
    result, result_state = solver(max_batch=40).integrate(
        f, mesh=even_mesh(46), take_gradient=False
    )
    assert max(f.sizes) <= 40
    assert result_state.max_batch_converged
    f = Recorder()
    state = IntegrationState(even_mesh(46), 5000, False)
    result, result_state = solver().integrate(
        f, state=state, max_batch=30, take_gradient=False
    )
    assert max(f.sizes) <= 30
    assert (result_state.max_batch, result_state.max_batch_converged) == (30, True)


def test_state_carries_mesh_and_batch_size_to_the_next_call():
    """The next call starts from the state's mesh and batch size: its first
    pass covers the mesh's subintervals. The state's
    batch size beats the constructor's max_batch."""
    first, first_state = solver().integrate(
        Recorder(), mesh_init=A, mesh_final=B, take_gradient=False
    )
    f = Recorder()
    second, _ = solver(max_batch=7).integrate(f, state=first_state, take_gradient=False)
    assert second.n_start_steps == len(first_state.mesh) - 1
    assert f.sizes[0] == min(21 * second.n_start_steps, 10 * first_state.max_batch)
    assert math.isclose(float(second.integral), EXACT, abs_tol=1e-9)


def test_state_batch_below_one_subinterval_with_take_gradient():
    """A take_gradient=False call may converge on a batch size below one
    subinterval's 21 points (here the given max_batch=10). A take_gradient=True
    call from its state raises ValueError before f is called, since its
    batches hold whole subintervals and PadaQuad does not exceed a batch size
    known to fit; a max_batch of 21 given with the state, or the same batch
    size not converged (it may grow), runs."""
    _, state = solver().integrate(
        Recorder(), mesh_init=A, mesh_final=B, take_gradient=False, max_batch=10
    )
    assert (state.max_batch, state.max_batch_converged) == (10, True)
    f = Recorder()
    with pytest.raises(
        ValueError, match="The state batch size 10 is below the 21 points"
    ):
        solver().integrate(f, state=state, take_gradient=True)
    assert f.sizes == []
    for kwargs in (
        {"state": state, "max_batch": 21},
        {"state": dataclasses.replace(state, max_batch_converged=False)},
    ):
        result, _ = solver().integrate(Recorder(), take_gradient=True, **kwargs)
        assert math.isclose(float(result.integral), EXACT, abs_tol=1e-9)


def test_state_and_mesh_together_raise():
    state = IntegrationState(even_mesh(4), 100, False)
    with pytest.raises(ValueError, match="mesh= or state="):
        solver().integrate(Recorder(), mesh=even_mesh(4), state=state)


def test_state_saves_loads_and_validates():
    """dataclasses.asdict gives a dictionary torch.load reads with
    weights_only=True; == compares identity instead of raising on the tensor
    field; invalid fields raise ValueError."""
    _, state = solver().integrate(Recorder(), mesh_init=A, mesh_final=B)
    buffer = io.BytesIO()
    torch.save(dataclasses.asdict(state), buffer)
    buffer.seek(0)
    loaded = IntegrationState(**torch.load(buffer, weights_only=True))
    assert torch.equal(loaded.mesh, state.mesh)
    assert (loaded.max_batch, loaded.max_batch_converged) == (
        state.max_batch,
        state.max_batch_converged,
    )
    assert state == state
    assert state != loaded
    for bad in ({"max_batch": 0}, {"max_batch": 2.0}, {"max_batch_converged": 1}):
        with pytest.raises(ValueError):
            dataclasses.replace(state, **bad)


# ---------------------------------------------------------------------------
# take_gradient=True: forward retry, backward out of memory, graph walk
# ---------------------------------------------------------------------------


class GradIntegrand:
    """w sin(40 t), plus v max(t - 0.9, 0) for a batch that reaches t > 0.9 (a
    parameter only some batches use). Its forward pass runs out of memory above
    ``forward_limit`` points; a hook makes the backward pass of a batch above
    ``backward_limit`` points run out of memory, before any gradient of this
    batch is written. ``function`` routes w through a custom autograd
    function that hides it from the autograd graph (it is not an input; the
    function adds its gradient itself), as activation checkpointing with
    ``use_reentrant=True`` does."""

    def __init__(
        self, forward_limit=math.inf, backward_limit=math.inf, function=False, fail=()
    ):
        self.w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
        self.v = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        self.forward_limit, self.backward_limit = forward_limit, backward_limit
        self.function, self.sizes = function, []
        # The backward passes of these batches (counted from 1) run out of
        # memory too, whatever their size.
        self.fail, self.batches = set(fail), 0

    def __call__(self, t):
        self.sizes.append(len(t))
        if len(t) > self.forward_limit:
            raise torch.OutOfMemoryError("simulated forward")
        if self.function:
            y = HiddenScale.apply(torch.sin(40 * t) + 0 * self.v, self)
        else:
            y = self.w * torch.sin(40 * t)
        if bool((t > 0.9).any()):
            y = y + self.v * torch.clamp(t - 0.9, min=0)
        self.batches += 1
        if y.requires_grad and (
            len(t) > self.backward_limit or self.batches in self.fail
        ):
            y.register_hook(self._fail)
        return y

    @staticmethod
    def _fail(grad):
        raise torch.OutOfMemoryError("simulated backward")


class HiddenScale(torch.autograd.Function):
    """x * owner.w, with w hidden: not an input, so the graph has no node for
    it; backward adds w's gradient itself. Shows up as a BackwardCFunction."""

    @staticmethod
    def forward(ctx, x, owner):
        ctx.owner = owner
        ctx.save_for_backward(x)
        return x * owner.w.detach()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        w = ctx.owner.w
        g = (grad * x).sum().reshape(1)
        w.grad = g if w.grad is None else w.grad + g
        return grad * w.detach(), None


def grads(f):
    return float(f.w.grad), float(f.v.grad)


GROWING = IntegrationState(even_mesh(500), 100, False)  # passes of 10,500 points


def test_forward_out_of_memory_is_retried_with_gradients():
    """take_gradient=True: f runs out of memory above 3,000 points; the batch
    is evaluated again smaller (no gradient was written yet), the state
    converges, and the gradients are right."""
    f = GradIntegrand(forward_limit=3000)
    result, result_state = solver().integrate(f, state=GROWING)
    assert result_state.max_batch_converged
    assert result_state.max_batch <= 3000
    assert grads(f) == pytest.approx((D_W, D_V), rel=1e-8)


@pytest.mark.parametrize("pass_parameters", [False, True])
def test_backward_out_of_memory_restarts_without_double_counting(
    pass_parameters, caplog
):
    """The backward pass of the batch of 9,513 points fails after the batch of
    987 already added its gradient. The gradients this call added are put
    back (found by the graph walk; v appears only in that later batch) and the
    integral restarts with batches 0.75 times smaller (whole subintervals of
    21 points: 9,513 fails, then 7,119 and 5,334, then 3,990 runs): the
    gradients equal one integral's, not more. A gradient held before the call is kept
    and added to."""
    f = GradIntegrand(backward_limit=5000)
    f.w.grad = torch.tensor([1.0], dtype=torch.float64)
    kwargs = {"parameters": [f.w, f.v]} if pass_parameters else {}
    with caplog.at_level(logging.WARNING):
        result, result_state = solver().integrate(f, state=GROWING, **kwargs)
    assert caplog.text.count("The backward pass of a batch of") == 3
    assert "restarted from its starting mesh" in caplog.text
    assert result_state.max_batch_converged
    assert result_state.max_batch <= 5000
    assert grads(f) == pytest.approx((1 + D_W, D_V), rel=1e-8)
    assert math.isclose(float(result.integral), 1.3 * D_W + 0.7 * D_V, abs_tol=1e-9)


def test_custom_autograd_function_needs_parameters():
    """A custom autograd function can hide parameters from the graph walk, so
    without parameters= a backward out-of-memory error raises
    BackwardOutOfMemoryError with a state to call again with; after resetting
    the gradients, that call gives the right ones. With parameters= the call
    restarts by itself."""
    f = GradIntegrand(backward_limit=5000, function=True)
    with pytest.raises(BackwardOutOfMemoryError, match="custom autograd") as info:
        solver().integrate(f, state=GROWING)
    state = info.value.state
    assert torch.equal(state.mesh, GROWING.mesh)
    assert state.max_batch_converged
    assert state.max_batch == 7135  # 0.75 x 9,513, the batch that failed
    f.w.grad = f.v.grad = None
    f.backward_limit = math.inf
    solver().integrate(f, state=state)
    assert grads(f) == pytest.approx((D_W, D_V), rel=1e-8)

    # An optimizer, and a generator (used up after one pass over it), which
    # every restart must still see. The second batch of the first two attempts
    # fails, so each attempt has added w's hidden gradient before it fails.
    for listed in (lambda f: torch.optim.SGD([f.w, f.v]), lambda f: iter([f.w, f.v])):
        f = GradIntegrand(function=True, fail={2, 4})
        solver().integrate(f, state=GROWING, parameters=listed(f))
        assert grads(f) == pytest.approx((D_W, D_V), rel=1e-8)


def test_post_accumulate_hook_raises_instead_of_restarting():
    """Work a post-accumulate-grad hook did cannot be undone."""
    f = GradIntegrand(backward_limit=5000)
    f.w.register_post_accumulate_grad_hook(lambda p: None)
    with pytest.raises(BackwardOutOfMemoryError, match="hook"):
        solver().integrate(f, state=GROWING)


def test_backward_error_is_not_a_retryable_out_of_memory_error():
    """padaquad's own out-of-memory handlers must not catch the new error (an
    outer integral would retry and keep the inner call's partial gradients),
    and it survives pickling with its state."""
    error = BackwardOutOfMemoryError("message")
    error.state = IntegrationState(None, 3, True)
    assert isinstance(error, torch.OutOfMemoryError)
    assert not _is_out_of_memory(error)
    copy = pickle.loads(pickle.dumps(error))
    assert str(copy) == "message"
    assert copy.state.max_batch == 3


@needs_cuda
def test_backward_check_shrinks_the_batch_on_a_gpu(monkeypatch, caplog):
    """On a GPU, when the usable memory after a forward pass is below its peak
    memory, the batch is dropped before any gradient is written and evaluated
    again smaller; the gradient is still right. A plain integral (no gradient
    needed) is never checked."""
    s = adaptive_quadrature("uniform", "gk21", device="cuda", atol=1e-10, rtol=1e-10)
    monkeypatch.setattr(s, "_get_usable_memory", lambda tmu: 1e-6)  # 1 kB
    w = torch.tensor([1.3], dtype=torch.float64, device="cuda", requires_grad=True)
    state = IntegrationState(even_mesh(500).cuda(), 1000, False)
    with caplog.at_level(logging.WARNING):
        result, result_state = s.integrate(lambda t: w * torch.sin(40 * t), state=state)
    assert "predicted not to fit" in caplog.text
    assert result_state.max_batch_converged
    assert result_state.max_batch < 1000
    assert float(w.grad) == pytest.approx(D_W, rel=1e-8)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        s.integrate(lambda t: torch.sin(40 * t), state=state)
    assert "predicted not to fit" not in caplog.text
