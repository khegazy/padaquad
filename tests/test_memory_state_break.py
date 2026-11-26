"""Break-and-speed review of the memory state, the growing batch size and the restart.

Written by the break-and-speed review of the ``change_memory`` branch (report,
deleted once the bugs were fixed:
``docs/issues/2026-11-18-memory-state-break-and-speed.md`` in commit
``59b7f11``). Tests marked
``xfail(strict=True)`` failed because of the bug named in their ``reason``
(all fixed now, the markers removed); the
others pin behavior the review checked and found correct. Everything runs on
the CPU with simulated out-of-memory errors, except the tests marked
``needs_cuda``, which cause real CUDA out-of-memory errors under a memory cap
(``torch.cuda.set_per_process_memory_fraction``).
"""

from __future__ import annotations

import gc
import math
import weakref

import pytest
import torch
from _helpers import needs_cuda

import padaquad.quadrature.base as base_module
import padaquad.quadrature.memory as memory_module
from padaquad import BackwardOutOfMemoryError, IntegrationState, adaptive_quadrature

A = torch.tensor([0.0], dtype=torch.float64)
B = torch.tensor([1.0], dtype=torch.float64)
# d/dw of the integral of w sin(40 t) on [0, 1]
D_W = (1 - math.cos(40.0)) / 40


def solver(device="cpu", **kwargs):
    return adaptive_quadrature(
        "uniform", "gk21", device=device, atol=1e-10, rtol=1e-10, **kwargs
    )


def even_mesh(n_subintervals: int, device="cpu") -> torch.Tensor:
    return torch.linspace(0, 1, n_subintervals + 1, dtype=torch.float64, device=device)[
        :, None
    ]


# ---------------------------------------------------------------------------
# Bugs
# ---------------------------------------------------------------------------


def test_saved_gradient_copies_are_freed_when_integrate_returns(monkeypatch):
    """With take_gradient=True, ``integrate`` copies the ``.grad`` of every
    tensor its backward passes write to, so it can put them back after an
    out-of-memory error. The copies are needed only during the call. When a
    parameter has a gradient before the call (gradient accumulation, or a loss
    term backpropagated first), the copy is as large as the parameter. Each
    entry of the dictionary holds a weak reference whose callback refers back
    to the dictionary, so the dictionary and every copy survive the call until
    the cyclic garbage collector runs (measured on an A100: one to three extra
    copies of a 64 MiB gradient held between calls)."""
    w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    w.grad = torch.ones_like(w)
    copies: list[weakref.ref] = []
    original = memory_module._save_grad

    def spy(leaf, saved):
        original(leaf, saved)
        copy = saved[id(leaf)][1]
        if copy is not None and all(r() is not copy for r in copies):
            copies.append(weakref.ref(copy))

    monkeypatch.setattr(memory_module, "_save_grad", spy)
    monkeypatch.setattr(base_module, "_save_grad", spy)
    gc.collect()
    gc.disable()
    try:
        solver().integrate(
            lambda t: w * torch.sin(40 * t), mesh=even_mesh(50), max_batch=230
        )
        alive = sum(r() is not None for r in copies)
    finally:
        gc.enable()
    assert copies, "no gradient was saved"
    assert alive == 0, f"{alive} gradient copies outlive the call"


def test_out_of_memory_in_loss_fxn_puts_back_the_gradients():
    """The batch size grows until f runs out of memory, so the memory left for
    what runs after f (the quadrature weights, ``loss_fxn``) is whatever f did
    not use. An out-of-memory error there propagates as a plain
    ``torch.OutOfMemoryError``: no retry, no restart, and the gradients of the
    batches before it stay in ``.grad`` (on main the batch was sized at 0.75 of
    the usable memory divided by 2.1 times f's memory, which left room). A
    caller that retries, for example with a smaller ``max_batch``, counts those
    batches twice. Expected: either the call recovers with the right gradient,
    or it raises with the gradients put back (as it does for an
    out-of-memory error in the backward pass). Real-GPU version:
    ``test_real_out_of_memory_in_loss_fxn_puts_back_the_gradients``."""
    w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    calls = [0]

    def loss_fxn(result):
        calls[0] += 1
        if calls[0] == 3:  # the third batch's loss does not fit
            raise torch.OutOfMemoryError("simulated out-of-memory error in loss_fxn")
        return result.integral.sum()

    try:
        solver().integrate(
            lambda t: w * torch.sin(40 * t),
            mesh=even_mesh(200),
            max_batch=230,
            loss_fxn=loss_fxn,
        )
    except torch.OutOfMemoryError:
        assert w.grad is None, f"partial gradient left in .grad: {w.grad}"
    else:
        assert float(w.grad) == pytest.approx(D_W, rel=1e-8)


def test_nested_integral_on_the_same_solver_keeps_the_outer_max_batch():
    """A double integral whose inner integral uses the same solver object. The
    outer call fixes ``max_batch=50``; the inner call (no ``max_batch``) sets
    the solver's ``_batch`` to its own growing size, and the outer call then
    calls f with up to 2,530 points and returns the inner call's batch size as
    its state. On main a given ``max_batch`` was a local variable of the call
    and was respected. Two threads sharing one solver hit the same problem."""
    s = solver()
    sizes: list[int] = []

    def outer(t):
        sizes.append(len(t))
        inner, _ = s.integrate(
            torch.cos, mesh_init=A, mesh_final=B, take_gradient=False
        )
        return torch.sin(40 * t) * inner.integral

    _, state = s.integrate(
        outer, mesh=even_mesh(200), max_batch=50, take_gradient=False
    )
    assert max(sizes) <= 50, f"outer calls of f up to {max(sizes)} points"
    assert (state.max_batch, state.max_batch_converged) == (50, True)


@pytest.mark.parametrize("max_batch", [-5, True, 2.7, "100"])
def test_invalid_max_batch_is_rejected_before_any_work(max_batch):
    """``integrate`` turns ``max_batch`` into an int with ``int()``
    (``base.py:868``), which accepts a bool, truncates a float and parses a
    string; a negative value fails an assertion deep in the evaluation, after
    the output check and the first call of f. ``IntegrationState`` rejects a
    bool and a non-int; ``integrate`` should apply the same rule at its
    boundary (``operator.index`` and a check for >= 1) before calling f."""
    calls = [0]

    def f(t):
        calls[0] += 1
        return torch.sin(t)

    with pytest.raises((TypeError, ValueError)):
        solver().integrate(
            f, mesh_init=A, mesh_final=B, max_batch=max_batch, take_gradient=False
        )
    assert calls[0] == 0


def _compiled_integrand():
    lin = torch.nn.Linear(1, 1, dtype=torch.float64)
    compiled = torch.compile(
        lambda t: torch.sin(40 * lin(t)), backend="aot_eager", dynamic=True
    )

    def f(t):
        y = compiled(t)
        if len(t) > 3000 and y.requires_grad:

            def fail(grad):
                raise torch.OutOfMemoryError("simulated backward")

            y.register_hook(fail)
        return y

    return f, lin


def test_compiled_integrand_needs_parameters_to_restart():
    """``torch.compile`` wraps the integrand in a custom autograd function,
    which can hide parameters from the graph walk, so without ``parameters=``
    a backward out-of-memory error raises ``BackwardOutOfMemoryError`` (as
    documented under ``parameters``) instead of restarting; the sibling test
    shows the restart with ``parameters=``. (The review proposed treating
    compiled functions as transparent; not done until a compiled region that
    hides a parameter is ruled out.)"""
    torch._dynamo.reset()
    f, _ = _compiled_integrand()
    with pytest.raises(BackwardOutOfMemoryError, match="custom autograd"):
        solver().integrate(f, state=IntegrationState(even_mesh(500), 100, False))


# ---------------------------------------------------------------------------
# Checked and found correct
# ---------------------------------------------------------------------------


def test_backward_that_never_fits_ends_in_an_error_after_few_restarts(caplog):
    """A backward pass that runs out of memory at every size: the restarts
    lower the batch size by 0.75 each time and stop with
    ``BackwardOutOfMemoryError`` once a batch cannot hold one subinterval, with
    the gradients put back. No endless restart."""
    w = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    w.grad = torch.full_like(w, 5.0)
    attempts = [0]

    def f(t):
        y = w * torch.sin(40 * t)

        def fail(grad):
            attempts[0] += 1
            raise torch.OutOfMemoryError("simulated backward")

        y.register_hook(fail)
        return y

    with pytest.raises(BackwardOutOfMemoryError, match="cannot hold one subinterval"):
        solver().integrate(f, mesh=even_mesh(100), max_batch=2300)
    # 2,289 -> 1,701 -> ... -> 21 points: about 15 attempts, each failing once.
    assert attempts[0] <= 20
    assert float(w.grad) == 5.0


def test_restart_of_a_compiled_integrand_with_parameters():
    """With ``parameters=`` the compiled integrand restarts and its gradient
    equals an uninterrupted run's."""
    torch._dynamo.reset()
    f, lin = _compiled_integrand()
    state = IntegrationState(even_mesh(500), 100, False)
    _, end = solver().integrate(f, state=state, parameters=lin.parameters())
    restarted = lin.weight.grad.clone()
    lin.weight.grad = lin.bias.grad = None
    solver().integrate(f, mesh=even_mesh(500), max_batch=end.max_batch)
    assert torch.allclose(restarted, lin.weight.grad, rtol=1e-12, atol=0)


def _cap_cuda_memory(gb: float) -> None:
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(gb * 2**30 / total, 0)


@pytest.fixture
def capped_cuda():
    """A 1 GiB cap on the caching allocator of cuda:0, removed afterwards. The
    backward check reads ``torch.cuda.mem_get_info``, which does not see the
    cap, so the check passes and the real error happens."""
    _cap_cuda_memory(1.0)
    yield
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    torch.cuda.empty_cache()


@needs_cuda
@pytest.mark.usefixtures("capped_cuda")
def test_real_backward_out_of_memory_restarts_to_the_reference_gradient():
    """A real CUDA out-of-memory error inside ``loss.backward()`` (a hook that
    needs 200 kB per point): PadaQuad puts back the gradients, restarts with a
    smaller batch, and the integral and gradient are bit for bit those of an
    uninterrupted run with the final batch size. The process can go on using
    the GPU afterwards."""
    w = torch.tensor([1.3], dtype=torch.float64, device="cuda", requires_grad=True)

    def f(t):
        y = w * torch.sin(40 * t) + t
        if y.requires_grad:
            y.register_hook(
                lambda g: (
                    g
                    + 0
                    * torch.ones(len(g), 25_000, dtype=g.dtype, device=g.device).sum(
                        1, keepdim=True
                    )
                )
            )
        return y

    state = IntegrationState(even_mesh(1000, "cuda"), 230, False)
    result, end = solver("cuda").integrate(f, state=state)
    restarted = w.grad.clone()
    assert end.max_batch_converged
    assert end.max_batch < 20_000
    w.grad = None
    reference, _ = solver("cuda").integrate(
        f, mesh=even_mesh(1000, "cuda"), max_batch=end.max_batch
    )
    assert torch.equal(restarted, w.grad)
    assert torch.equal(result.integral, reference.integral)


@needs_cuda
@pytest.mark.usefixtures("capped_cuda")
@pytest.mark.parametrize("take_gradient", [False, True])
def test_real_out_of_memory_in_f_is_retried(take_gradient):
    """A real CUDA out-of-memory error inside f (200 kB of temporaries per
    point) is retried smaller. Without gradients the integral is bit for bit
    that of a run with the final batch size; with gradients the earlier
    batches had other sizes, so the gradient agrees to rounding."""
    w = torch.tensor([1.3], dtype=torch.float64, device="cuda", requires_grad=True)

    def f(t):
        big = torch.ones(len(t), 25_000, dtype=t.dtype, device=t.device)
        return w * torch.sin(40 * t) + t + 0 * big.sum(1, keepdim=True)

    state = IntegrationState(even_mesh(2000, "cuda"), 230, False)
    result, end = solver("cuda").integrate(f, state=state, take_gradient=take_gradient)
    assert end.max_batch_converged
    assert end.max_batch < 46_000
    grad = None if w.grad is None else w.grad.clone()
    w.grad = None
    reference, _ = solver("cuda").integrate(
        f,
        mesh=even_mesh(2000, "cuda"),
        max_batch=end.max_batch,
        take_gradient=take_gradient,
    )
    if take_gradient:
        assert torch.allclose(grad, w.grad, rtol=1e-12, atol=0)
        assert torch.allclose(result.integral, reference.integral, rtol=1e-12, atol=0)
    else:
        assert torch.equal(result.integral, reference.integral)


@needs_cuda
@pytest.mark.usefixtures("capped_cuda")
def test_real_out_of_memory_in_loss_fxn_puts_back_the_gradients():
    """The GPU version of ``test_out_of_memory_in_loss_fxn_puts_back_the_gradients``:
    f is cheap, and ``loss_fxn`` penalizes every integrand value through a
    [points, 4096] temporary. The batch grows from 2,289 to 22,890 points (whole
    gk21 subintervals of 21 points), and the loss of the 22,890-point batch (a
    715 MiB temporary) does not fit; the
    CUDA error propagates with the first batch's gradient in ``w.grad``."""
    w = torch.tensor([1.3], dtype=torch.float64, device="cuda", requires_grad=True)

    def loss_fxn(result):
        ones = torch.ones(1, 4096, dtype=result.y.dtype, device=result.y.device)
        penalty = (result.y.reshape(-1, 1) * ones).pow(2).mean()
        return result.integral.sum() + 0 * penalty

    try:
        solver("cuda").integrate(
            lambda t: w * torch.sin(40 * t),
            state=IntegrationState(even_mesh(10_000, "cuda"), 230, False),
            loss_fxn=loss_fxn,
        )
    except torch.OutOfMemoryError:
        assert w.grad is None, f"partial gradient left in .grad: {w.grad}"
    else:
        assert float(w.grad) == pytest.approx(D_W, rel=1e-6)
