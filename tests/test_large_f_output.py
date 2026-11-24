"""Tests for ``integrate(large_f_output=True)``.

With ``take_gradient=False`` every pass evaluates all waiting subintervals, in
calls of at most ``max_batch`` points, and integrates each call's completed
subintervals right away. ``large_f_output`` changes only where their nodes,
integrand values and tracked variables go: they stay on the solver's device by
default and move to ``result_device`` with the option. So the option must
change nothing else: not the mesh, not the calls of ``f``, not the integral
(bit for bit), whatever ``max_batch`` is. It raises with ``take_gradient=True``
and with anything that would give the values an autograd graph.

Before the fix the option (then ``conserve_memory``) judged each batch alone
against the running integral of the batches before it: ``exp(30 t)`` needed
1126 integrand points instead of 298, and the count depended on ``max_batch``.
"""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import needs_cuda

from padaquad import adaptive_quadrature, integrate

BOUNDS = {
    "mesh_init": torch.tensor([0.0], dtype=torch.float64),
    "mesh_final": torch.tensor([1.0], dtype=torch.float64),
}
DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]
# The calls of f end inside a subinterval for all of these with gk7 (C = 9):
# below C (5, 7), and not multiples of C (18 is, 23 and 100 are not).
MAX_BATCHES = [5, 7, 18, 23, 100]


class Counter:
    """Wrap an integrand and count the points it is called on."""

    def __init__(self, f):
        self.f = f
        self.n_points = 0

    def __call__(self, t, *args):
        self.n_points += t.shape[0]
        return self.f(t, *args)


def _exp30(t, *args):
    return torch.exp(30.0 * t)


def _run(large, method="gk7", sampling="uniform", f=_exp30, device="cpu", **kwargs):
    """Integrate f over [0, 1] (seed 0, take_gradient=False); return the result
    and the number of points f was called on."""
    options = {"atol": 1e-14, "rtol": 1e-8, "device": device} | {
        k: kwargs.pop(k) for k in ("atol", "rtol", "error_norm") if k in kwargs
    }
    solver = adaptive_quadrature(sampling, method=method, **options)
    counter = Counter(f)
    torch.manual_seed(0)
    result, _ = solver.integrate(
        counter,
        **BOUNDS,
        take_gradient=False,
        large_f_output=large,
        **kwargs,
    )
    return result, counter.n_points


def _assert_same_refinement(off, on):
    """Same subintervals, nodes, values and contributions, bit for bit."""
    for field in ("h", "nodes", "y", "mesh_quadratures", "integral", "integral_error"):
        a, b = getattr(off, field), getattr(on, field)
        assert torch.equal(a.cpu(), b.cpu()), f"{field} differs"
    assert off.converged == on.converged


# ---------------------------------------------------------------------------
# (a) The same refinement with and without the option, for every max_batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_batch", MAX_BATCHES)
def test_same_refinement_as_default(device, max_batch):
    """exp(30 t), gk7, rtol 1e-8: the same number of points with and without
    the option, for every max_batch, and the same result bit for bit. (The
    issue's run, with float64 as torch's default dtype for the random starting
    mesh, needed 298 points without the option and 1126 with it.)"""
    off, n_off = _run(False, device=device, max_batch=max_batch)
    on, n_on = _run(True, device=device, max_batch=max_batch)
    assert n_off == n_on
    _assert_same_refinement(off, on)
    exact = (math.exp(30.0) - 1.0) / 30.0
    assert abs(float(on.integral) - exact) < 1e-8 * exact


@pytest.mark.parametrize("device", DEVICES)
def test_refinement_does_not_depend_on_max_batch(device):
    """An elementwise integrand gives the same mesh and integral for every
    max_batch, with the option on."""
    reference, n_reference = _run(True, device=device, max_batch=MAX_BATCHES[-1])
    for max_batch in MAX_BATCHES[:-1]:
        result, n_points = _run(True, device=device, max_batch=max_batch)
        assert n_points == n_reference
        _assert_same_refinement(reference, result)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_batch", [2, 7, 50])
def test_variable_sampling_same_refinement(device, max_batch):
    """interpolatory3_variable (weights from each subinterval's own nodes): the
    option changes nothing, bit for bit, on the CPU and on CUDA."""
    off, n_off = _run(
        False,
        method="interpolatory3_variable",
        sampling="variable",
        device=device,
        max_batch=max_batch,
        rtol=1e-6,
    )
    on, n_on = _run(
        True,
        method="interpolatory3_variable",
        sampling="variable",
        device=device,
        max_batch=max_batch,
        rtol=1e-6,
    )
    assert n_off == n_on
    _assert_same_refinement(off, on)


@pytest.mark.parametrize("device", DEVICES)
def test_matrix_product_integrand_same_refinement(device):
    """An integrand whose values depend on how many points a call holds (a
    matrix product) still gives the same result with and without the option
    at the same max_batch: the calls of f are the same."""
    gen = torch.Generator().manual_seed(1)
    weights = torch.randn(1, 16, generator=gen, dtype=torch.float64).to(device)
    out = torch.randn(16, 3, generator=gen, dtype=torch.float64).to(device)

    def mlp(t, *args):
        return torch.tanh(t @ weights) @ out

    off, n_off = _run(False, f=mlp, device=device, max_batch=37, rtol=1e-6)
    on, n_on = _run(True, f=mlp, device=device, max_batch=37, rtol=1e-6)
    assert n_off == n_on
    _assert_same_refinement(off, on)


# ---------------------------------------------------------------------------
# (b) Where the values are, and how much the device holds
# ---------------------------------------------------------------------------


def test_each_call_is_integrated_alone():
    """Each call of f is integrated as soon as it returns: _calculate_integral
    receives at most the subintervals one call can complete (a call of
    max_batch points plus fewer than C points left from the previous one)."""
    max_batch = 23
    solver = adaptive_quadrature("uniform", method="gk7", device="cpu", rtol=1e-8)
    sizes = []
    calculate = solver._calculate_integral

    def spy(t, y, y0=None):
        sizes.append(len(t))
        return calculate(t, y, y0)

    solver._calculate_integral = spy
    torch.manual_seed(0)
    solver.integrate(
        _exp30, **BOUNDS, take_gradient=False, large_f_output=True, max_batch=max_batch
    )
    assert sizes
    assert max(sizes) <= (max_batch + solver.C - 1) // solver.C


@needs_cuda
def test_values_go_to_result_device():
    """On a CUDA solver with result_device="cpu", every result loss_fxn
    receives (per batch, and for the subintervals the re-check removes) has its
    nodes, values and tracked variables on the CPU and the rest on the GPU;
    without the option everything is on the GPU."""
    for large, values_device in ((True, "cpu"), (False, "cuda")):
        seen = []

        def loss_fxn(r):
            seen.append(
                (
                    r.nodes.device.type,
                    r.y.device.type,
                    r.tracked_variables[0].device.type,
                    r.mesh_quadratures.device.type,
                    r.h.device.type,
                )
            )
            return r.integral.sum()

        def f(t):
            value = (
                1.0
                + 0.5 * torch.sin(4.0 * math.pi * t)
                - 100.0 / math.sqrt(math.pi) * torch.exp(-1e4 * (t - 0.37) ** 2)
            )
            return value, (2 * t,)

        solver = adaptive_quadrature(
            "uniform", method="bosh3", device="cuda", atol=1e-9, rtol=1e-6
        )
        removed = []
        remove = solver._remove_failed_record_panels

        def spy(record, keep, *args):
            removed.append(int((~keep).sum()))
            return remove(record, keep, *args)

        solver._remove_failed_record_panels = spy
        torch.manual_seed(0)
        solver.integrate(
            f,
            **{k: v.cuda() for k, v in BOUNDS.items()},
            take_gradient=False,
            large_f_output=large,
            max_batch=16,
            loss_fxn=loss_fxn,
            result_device="cpu",
        )
        assert sum(removed) > 0, "no removal: the removal result was not checked"
        assert set(seen) == {(values_device,) * 3 + ("cuda", "cuda")}


@needs_cuda
def test_device_peak_memory_is_lower():
    """gk21, 4096 outputs, 1000 subintervals of a fixed mesh, calls of 230
    points: the peak on the GPU is at least 2 times lower with the option
    (measured 2.75 for 1e4 and 1e5 outputs: the values of the whole pass leave
    the device, while the contributions and error estimates of every
    subinterval stay)."""
    freqs = torch.linspace(1.0, 2.0, 4096, dtype=torch.float64, device="cuda")

    def f(t):
        return torch.sin(t * freqs)

    mesh = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64, device="cuda")[:, None]
    peaks = {}
    for large in (False, True):
        solver = adaptive_quadrature("uniform", method="gk21", device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result, _ = solver.integrate(
            f,
            mesh=mesh,
            take_gradient=False,
            max_batch=230,
            max_adaptive_splits=0,
            large_f_output=large,
        )
        torch.cuda.synchronize()
        peaks[large] = torch.cuda.max_memory_allocated() - base
        del result
    assert peaks[True] < peaks[False] / 2, peaks


# ---------------------------------------------------------------------------
# (c) What the option refuses
# ---------------------------------------------------------------------------


def _refused(**kwargs):
    """Call integrate with large_f_output=True; return the ValueError and the
    number of calls of f."""
    calls = []
    f = kwargs.pop("f", None)

    def counted(t, *args):
        calls.append(len(t))
        return f(t) if f is not None else torch.sin(t)

    options = {"take_gradient": False, **BOUNDS, **kwargs}
    with pytest.raises(ValueError, match="large_f_output") as info:
        integrate(counted, "gk21", device="cpu", large_f_output=True, **options)
    assert "gradient_as_output" in str(info.value)
    return str(info.value), calls


def test_take_gradient_raises_before_any_call_of_f():
    message, calls = _refused(take_gradient=True)
    assert "take_gradient=True" in message
    assert calls == []


@pytest.mark.parametrize("bound", ["mesh_init", "mesh_final"])
def test_bound_that_requires_grad_raises(bound):
    """Either bound, before any call of f (the lower bound alone used to reach
    f in the shape check, the upper one did not)."""
    value = BOUNDS[bound].clone().requires_grad_()
    message, calls = _refused(**{bound: value})
    assert "bound" in message
    assert calls == []


def test_mesh_that_requires_grad_raises():
    mesh = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)[:, None].requires_grad_()
    message, calls = _refused(mesh=mesh, mesh_init=None, mesh_final=None)
    assert "bound or mesh" in message
    assert calls == []


def test_output_that_requires_grad_raises():
    """f's own parameters: caught at the one-point shape check, before the
    integration's calls of f."""
    theta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    message, calls = _refused(f=lambda t: theta * torch.sin(t))
    assert "torch.func.grad" in message
    assert calls == [1]


# ---------------------------------------------------------------------------
# (d) result_device=None keeps everything on the solver's device
# ---------------------------------------------------------------------------


def test_result_device_none_runs():
    on, _ = _run(True, max_batch=18, result_device=None)
    cpu, _ = _run(True, max_batch=18, result_device="cpu")
    assert on.y.device == on.integral.device
    _assert_same_refinement(on, cpu)


# ---------------------------------------------------------------------------
# (e) Every feature gives the same result with and without the option
# ---------------------------------------------------------------------------


def _peak(t, *args):
    """A smooth part cancelled by a narrow peak (integral 0): drives re-check
    removals and the drop of a fixed reference."""
    return (
        1.0
        + 0.5 * torch.sin(4.0 * math.pi * t)
        - 100.0 / math.sqrt(math.pi) * torch.exp(-1e4 * (t - 0.37) ** 2)
    )


def _nan_tail(t, *args):
    return torch.where(t > 0.9, torch.nan, torch.sin(5.0 * t))


def _three_outputs(t, *args):
    return torch.cat([torch.sin(7 * t), torch.exp(-3 * t) * t, torch.cos(11 * t)], -1)


def _tracked(t, *args):
    return torch.sin(5.0 * t), (t**2, torch.cos(t))


FEATURES = {
    "recheck": {"f": _peak, "method": "bosh3", "atol": 1e-9, "rtol": 1e-6},
    "no_recheck": {
        "f": _peak,
        "method": "bosh3",
        "atol": 1e-9,
        "rtol": 1e-6,
        "enforce_convergence": False,
    },
    "fixed_reference": {
        "f": _peak,
        "method": "bosh3",
        "atol": 1e-9,
        "rtol": 1e-6,
        "error_integral_reference": 1.0,
    },
    "split_cap": {"f": _peak, "method": "bosh3", "max_adaptive_splits": 3},
    "nonfinite": {"f": _nan_tail, "error_on_nonfinite": False},
    "tracked": {"f": _tracked},
    "three_outputs": {"f": _three_outputs},
    "failure_fraction": {"f": _three_outputs, "error_norm": "failure_fraction"},
    "custom_loss": {"f": _peak, "loss_fxn": lambda r: 2.0 * r.integral.sum()},
}


@pytest.mark.parametrize("max_batch", [7, 50])
@pytest.mark.parametrize("feature", list(FEATURES))
def test_features_unchanged(feature, max_batch):
    kwargs = dict(FEATURES[feature])
    off, n_off = _run(False, max_batch=max_batch, **dict(kwargs))
    on, n_on = _run(True, max_batch=max_batch, **dict(kwargs))
    assert n_off == n_on
    for field in ("h", "nodes", "mesh_quadratures", "integral_error"):
        assert torch.equal(
            torch.nan_to_num(getattr(off, field), nan=7.0),
            torch.nan_to_num(getattr(on, field), nan=7.0),
        ), field
    assert torch.equal(
        torch.nan_to_num(off.integral, nan=7.0), torch.nan_to_num(on.integral, nan=7.0)
    )
    if feature == "tracked":
        for a, b in zip(off.tracked_variables, on.tracked_variables, strict=True):
            assert torch.equal(a, b)
    if feature == "custom_loss":
        assert torch.equal(off.loss, on.loss)
