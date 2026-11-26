"""Tests for ``trade_speed_for_memory``, ``evaluation_buffer`` and ``record_evaluations``.

With ``take_gradient=False`` every pass evaluates all waiting subintervals, in
calls of at most ``max_batch`` points. The three options change only when the
values of whole subintervals are integrated (once per pass by default; with
``trade_speed_for_memory=True`` whenever ``evaluation_buffer`` bytes wait,
after every call without it), where they go (the solver's device until the
accept-or-split decision by default, ``result_device`` right away with
``trade_speed_for_memory=True``), and whether they are kept at all
(``record_evaluations``). So they must change nothing else: not the mesh, not
the calls of ``f``, not the integral (bit for bit), whatever ``max_batch`` is.

Before 2.0.0, the option ``conserve_memory`` judged each batch alone against
the running integral of the batches before it: ``exp(30 t)`` needed 1126
integrand points instead of 298, and the count depended on ``max_batch``.
"""

from __future__ import annotations

import logging
import math
import sys

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
# The settings compared with the default. A buffer of 1,000 bytes holds a few
# gk7 subintervals of one float64 output (72 bytes each); 1e9 holds every pass.
SETTINGS = {
    "trade": {"trade_speed_for_memory": True},
    "trade_small_buffer": {"trade_speed_for_memory": True, "evaluation_buffer": 1000},
    "trade_large_buffer": {"trade_speed_for_memory": True, "evaluation_buffer": 10**9},
    "no_record": {"record_evaluations": False},
    "trade_no_record": {"trade_speed_for_memory": True, "record_evaluations": False},
}


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


def _run(method="gk7", sampling="uniform", f=_exp30, device="cpu", **kwargs):
    """Integrate f over [0, 1] (seed 0, take_gradient=False by default); return
    the result, the state and the number of points f was called on."""
    options = {"atol": 1e-14, "rtol": 1e-8, "device": device} | {
        k: kwargs.pop(k)
        for k in ("atol", "rtol", "error_norm", "merge_rule")
        if k in kwargs
    }
    solver = adaptive_quadrature(sampling, method=method, **options)
    counter = Counter(f)
    torch.manual_seed(0)
    result, state = solver.integrate(
        counter, **BOUNDS, **({"take_gradient": False} | kwargs)
    )
    return result, state, counter.n_points


def _assert_same_refinement(default, other, values=True):
    """Same subintervals, nodes, contributions and integral, bit for bit (NaN
    equal to NaN); with ``values``, the same integrand and tracked values."""
    fields = ["h", "nodes", "mesh_quadratures", "integral", "integral_error"]
    for field in fields + (["y"] if values else []):
        a, b = getattr(default, field).cpu(), getattr(other, field).cpu()
        assert torch.equal(
            torch.nan_to_num(a, nan=7.0), torch.nan_to_num(b, nan=7.0)
        ), f"{field} differs"
    if values and default.tracked_variables is not None:
        for a, b in zip(
            default.tracked_variables, other.tracked_variables, strict=True
        ):
            assert torch.equal(a.cpu(), b.cpu())
    assert default.converged == other.converged


def _compare(setting, **kwargs):
    """Run the default and ``setting``; check the calls of f and the result."""
    default, default_state, n_default = _run(**kwargs)
    other, other_state, n_other = _run(**kwargs, **SETTINGS[setting])
    assert n_default == n_other
    recorded = SETTINGS[setting].get("record_evaluations", True)
    _assert_same_refinement(default, other, values=recorded)
    if recorded:
        assert torch.equal(default_state.mesh, other_state.mesh)
    else:
        assert other.y is None
        assert other.tracked_variables is None
    return other


# ---------------------------------------------------------------------------
# (a) The same refinement with every setting, for every max_batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_batch", MAX_BATCHES)
@pytest.mark.parametrize("setting", list(SETTINGS))
def test_same_refinement_as_default(setting, max_batch, device):
    """exp(30 t), gk7, rtol 1e-8: the same points, mesh and integral, bit for
    bit. (The issue's run, with float64 as torch's default dtype for the random
    starting mesh, needed 298 points by default and 1126 with conserve_memory.)"""
    result = _compare(setting, device=device, max_batch=max_batch)
    exact = (math.exp(30.0) - 1.0) / 30.0
    assert abs(float(result.integral) - exact) < 1e-8 * exact


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("setting", ["default", *SETTINGS])
def test_refinement_does_not_depend_on_max_batch(setting, device):
    """An elementwise integrand gives the same mesh and integral for every
    max_batch."""
    options = SETTINGS.get(setting, {})
    reference, _, n_reference = _run(
        device=device, max_batch=MAX_BATCHES[-1], **options
    )
    for max_batch in MAX_BATCHES[:-1]:
        result, _, n_points = _run(device=device, max_batch=max_batch, **options)
        assert n_points == n_reference
        _assert_same_refinement(
            reference, result, values=options.get("record_evaluations", True)
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_batch", [2, 7, 50])
@pytest.mark.parametrize("setting", list(SETTINGS))
def test_variable_sampling_same_refinement(setting, max_batch, device):
    """interpolatory3_variable (weights from each subinterval's own nodes): the
    setting changes nothing, bit for bit, on the CPU and on CUDA."""
    _compare(
        setting,
        method="interpolatory3_variable",
        sampling="variable",
        device=device,
        max_batch=max_batch,
        rtol=1e-6,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("setting", list(SETTINGS))
def test_matrix_product_integrand_same_refinement(setting, device):
    """An integrand whose values depend on how many points a call holds (a
    matrix product) still gives the same result with every setting at the same
    max_batch: the calls of f are the same."""
    gen = torch.Generator().manual_seed(1)
    weights = torch.randn(1, 16, generator=gen, dtype=torch.float64).to(device)
    out = torch.randn(16, 3, generator=gen, dtype=torch.float64).to(device)

    def mlp(t, *args):
        return torch.tanh(t @ weights) @ out

    _compare(setting, f=mlp, device=device, max_batch=37, rtol=1e-6)


# ---------------------------------------------------------------------------
# (b) When the values are integrated, where they go, how much the device holds
# ---------------------------------------------------------------------------


def _integration_sizes(max_batch, **options):
    """The number of subintervals of every _calculate_integral call, and of
    every pass (the subintervals each _evaluate_f_on_split_nodes call gets)."""
    solver = adaptive_quadrature("uniform", method="gk7", device="cpu", rtol=1e-8)
    sizes, passes = [], []
    calculate = solver._calculate_integral
    evaluate = solver._evaluate_f_on_split_nodes

    def spy_calculate(t, y, mesh_left, mesh_right, y0=None):
        sizes.append(len(t))
        return calculate(t, y, mesh_left, mesh_right, y0)

    def spy_evaluate(f, f_args, mesh, step_idxs, *args):
        passes.append(len(step_idxs))
        return evaluate(f, f_args, mesh, step_idxs, *args)

    solver._calculate_integral = spy_calculate
    solver._evaluate_f_on_split_nodes = spy_evaluate
    torch.manual_seed(0)
    solver.integrate(
        _exp30, **BOUNDS, take_gradient=False, max_batch=max_batch, **options
    )
    assert sizes
    return sizes, passes, solver.C


def test_default_integrates_each_pass_at_once():
    """Without trade_speed_for_memory, _calculate_integral runs once per pass, on
    all its subintervals."""
    sizes, passes, _ = _integration_sizes(23)
    assert sizes == passes


def test_trade_integrates_each_call_alone():
    """trade_speed_for_memory without a buffer: _calculate_integral receives at
    most the subintervals one call can complete (a call of max_batch points plus
    fewer than C points left from the previous one)."""
    max_batch = 23
    sizes, passes, C = _integration_sizes(max_batch, trade_speed_for_memory=True)
    assert max(sizes) <= (max_batch + C - 1) // C
    assert len(sizes) > len(passes)


def test_buffer_bounds_what_waits():
    """With a buffer, the values integrated together are below the buffer plus
    one call's subintervals (they are integrated as soon as they reach it), and
    fewer integrations run than without a buffer."""
    max_batch, buffer = 23, 1000
    per_subinterval = 9 * 8  # gk7: C = 9 float64 values of one output
    sizes, _, C = _integration_sizes(
        max_batch, trade_speed_for_memory=True, evaluation_buffer=buffer
    )
    unbuffered, _, _ = _integration_sizes(max_batch, trade_speed_for_memory=True)
    assert (
        max(sizes) * per_subinterval
        < buffer + (max_batch + C - 1) // C * per_subinterval
    )
    assert len(sizes) < len(unbuffered)
    assert sum(sizes) == sum(unbuffered)


def _value_devices(trade, record=True):
    """Devices of the fields of every result loss_fxn receives (per batch, and
    for the subintervals the re-check removes), CUDA solver, result_device cpu."""
    seen = []

    def loss_fxn(r):
        seen.append(
            (
                r.y.device.type if r.y is not None else None,
                r.tracked_variables[0].device.type if r.tracked_variables else None,
                r.nodes.device.type,
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
    result, state = solver.integrate(
        f,
        **{k: v.cuda() for k, v in BOUNDS.items()},
        take_gradient=False,
        trade_speed_for_memory=trade,
        record_evaluations=record,
        max_batch=16,
        loss_fxn=loss_fxn,
        result_device="cpu",
    )
    assert sum(removed) > 0, "no removal: the removal result was not checked"
    return set(seen), result, state


@needs_cuda
@pytest.mark.parametrize(
    ("trade", "record", "values"),
    [(True, True, "cpu"), (False, True, "cuda"), (False, False, None)],
)
def test_value_devices(trade, record, values):
    """The values (and tracked variables) are on result_device with
    trade_speed_for_memory, on the solver's device without it, and absent
    without record_evaluations; the rest is on the solver's device. The result
    itself is on result_device, apart from the integral and the state."""
    seen, result, state = _value_devices(trade, record)
    assert seen == {(values, values, "cuda", "cuda", "cuda")}
    assert result.nodes.device.type == "cpu"
    assert result.h.device.type == "cpu"
    assert result.integral.device.type == "cuda"
    assert state.mesh.device.type == "cuda"


@needs_cuda
@pytest.mark.parametrize("record", [True, False])
def test_default_holds_a_pass_once(record):
    """Without trade_speed_for_memory, the GPU holds a pass's values about once:
    gk21, 4096 float64 outputs, a first pass of 1000 subintervals in calls of
    230 points, of which the 500 on [0.5, 1] are split (a frequency up to
    40,000 there). Each call is written into one array for the pass, and the
    accepted rows are moved within it, so the peak is below 1.8 times the
    pass's values (measured 1.55; 2.43 with a list of calls joined at the end
    and the accepted rows copied). The rest is the per-subinterval
    contributions and error estimates of the 4096 outputs. The same with
    record_evaluations=False, where the values are dropped after each pass."""
    freqs = torch.linspace(1.0, 2.0, 4096, dtype=torch.float64, device="cuda")

    def f(t):
        return torch.sin(t * freqs * (1 + 20000 * (t > 0.5)))

    mesh = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64, device="cuda")[:, None]
    solver = adaptive_quadrature(
        "uniform", method="gk21", device="cuda", atol=1e-10, rtol=1e-10
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result, _ = solver.integrate(
        f,
        mesh=mesh,
        take_gradient=False,
        max_batch=230,
        max_adaptive_splits=1,
        record_evaluations=record,
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert len(result.h) == 1500  # 500 of the 1000 were split once
    assert peak < 1.8 * 1000 * solver.C * 4096 * 8, peak


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("record", [True, False])
def test_one_call_pass_keeps_f_output_intact(device, record):
    """When one call covers a pass (max_batch above every pass's points), its
    output is kept as the pass's values without a copy; the accepted rows are
    then copied from it, so the tensors f returned are never overwritten, and
    the result equals a run whose calls are cut (max_batch 23)."""
    returned, snapshots = [], []

    def f(t):
        out = torch.exp(30.0 * t)
        returned.append(out)
        snapshots.append(out.clone())
        return out

    torch.manual_seed(0)
    result, _, _ = _run(f=f, device=device, max_batch=10**6, record_evaluations=record)
    assert len(returned) > 2  # passes with splits
    for out, snap in zip(returned, snapshots, strict=True):
        assert torch.equal(out, snap)
    torch.manual_seed(0)
    cut, _, _ = _run(device=device, max_batch=23, record_evaluations=record)
    assert torch.equal(result.integral, cut.integral)
    assert torch.equal(result.h, cut.h)


def test_keep_rows_in_place(monkeypatch):
    """_keep_rows with in_place moves the kept rows to the front of x, in
    blocks (here of 3 rows, so the last block is shorter), and returns a view
    of x equal to indexing; without in_place, or for an x that requires grad
    or is a view, it indexes and leaves x unchanged."""
    import padaquad.quadrature.base as qbase

    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 4, 3, generator=g)
    keep = torch.rand(20, generator=g) < 0.6
    expected = x[keep].clone()
    monkeypatch.setattr(qbase, "_BLOCK_BYTES", 3 * 4 * 3 * x.element_size())
    copied = qbase._keep_rows(x, keep, in_place=False)
    assert torch.equal(copied, expected)
    assert copied.data_ptr() != x.data_ptr()
    grad_x = x.clone().requires_grad_()
    assert torch.equal(qbase._keep_rows(grad_x, keep, in_place=True), expected)
    assert torch.equal(grad_x, x)
    view = x.clone()[:]
    assert torch.equal(qbase._keep_rows(view, keep, in_place=True), expected)
    assert torch.equal(view, x)  # a view is copied from, not overwritten
    kept = qbase._keep_rows(x, keep, in_place=True)
    assert torch.equal(kept, expected)
    assert kept.data_ptr() == x.data_ptr()
    assert len(qbase._keep_rows(x, torch.zeros(20, dtype=torch.bool), True)) == 0


def test_pass_array_out_of_memory_names_the_remedies(monkeypatch):
    """An out-of-memory error at the allocation of a pass's array (default
    setting, several calls per pass) is raised as torch.OutOfMemoryError
    naming trade_speed_for_memory and expandable_segments; another error at
    that allocation passes through unchanged."""
    new_empty = torch.Tensor.new_empty

    def failing(error):
        def new_empty_in_split(self, *args, **kwargs):
            # Only the allocations made by the split evaluation fail.
            if sys._getframe(1).f_code.co_name == "_evaluate_f_on_split_nodes":
                raise error
            return new_empty(self, *args, **kwargs)

        return new_empty_in_split

    monkeypatch.setattr(
        torch.Tensor,
        "new_empty",
        failing(RuntimeError("DefaultCPUAllocator: not enough memory")),
    )
    with pytest.raises(torch.OutOfMemoryError) as info:
        _run(max_batch=5)
    message = str(info.value)
    assert "trade_speed_for_memory=True" in message
    assert "expandable_segments:True" in message
    assert "#ug-pass-array-oom" in message
    monkeypatch.setattr(torch.Tensor, "new_empty", failing(RuntimeError("other")))
    with pytest.raises(RuntimeError, match=r"^other$"):
        _run(max_batch=5)


@needs_cuda
def test_device_peak_memory_is_lower():
    """gk21, 4096 outputs, 1000 subintervals of a fixed mesh, calls of 230
    points: the peak on the GPU is at least 2 times lower with
    trade_speed_for_memory (the values of the whole pass leave the device, while
    the contributions and error estimates of every subinterval stay)."""
    freqs = torch.linspace(1.0, 2.0, 4096, dtype=torch.float64, device="cuda")

    def f(t):
        return torch.sin(t * freqs)

    mesh = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64, device="cuda")[:, None]
    peaks = {}
    for trade in (False, True):
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
            trade_speed_for_memory=trade,
        )
        torch.cuda.synchronize()
        peaks[trade] = torch.cuda.max_memory_allocated() - base
        del result
    assert peaks[True] < peaks[False] / 2, peaks


# ---------------------------------------------------------------------------
# (c) What is refused, before any call of f
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"trade_speed_for_memory": True, "take_gradient": True}, "take_gradient=True"),
        ({"evaluation_buffer": 1000}, "only with trade_speed_for_memory=True"),
        ({"trade_speed_for_memory": True, "evaluation_buffer": 0}, "int >= 1"),
        ({"trade_speed_for_memory": True, "evaluation_buffer": -5}, "int >= 1"),
        ({"trade_speed_for_memory": True, "evaluation_buffer": True}, "int >= 1"),
    ],
)
def test_refused_before_any_call_of_f(options, match):
    calls = []

    def f(t):
        calls.append(len(t))
        return torch.sin(t)

    with pytest.raises(ValueError, match=match):
        integrate(
            f, "gk21", device="cpu", **({"take_gradient": False, **BOUNDS} | options)
        )
    assert calls == []


def test_buffer_that_is_not_an_integer_raises():
    with pytest.raises(TypeError):
        integrate(
            torch.sin,
            "gk21",
            device="cpu",
            take_gradient=False,
            trade_speed_for_memory=True,
            evaluation_buffer=1.5e6,
            **BOUNDS,
        )


# ---------------------------------------------------------------------------
# (d) Warnings, on the first call of a series only (no state)
# ---------------------------------------------------------------------------

THETA = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
WARNINGS = {
    "no_buffer": ({"trade_speed_for_memory": True}, torch.sin, "evaluation_buffer"),
    "requires_grad": (
        {"trade_speed_for_memory": True, "evaluation_buffer": 10**6},
        lambda t: THETA * torch.sin(t),
        "gradient_as_output",
    ),
    "no_record": ({"record_evaluations": False}, torch.sin, "merge_rule='power_law'"),
}


@pytest.mark.parametrize("name", list(WARNINGS))
def test_warning_on_first_call_only(name, caplog):
    options, f, text = WARNINGS[name]
    solver = adaptive_quadrature("uniform", method="gk21", device="cpu")
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        _, state = solver.integrate(f, **BOUNDS, take_gradient=False, **options)
    first = [r.getMessage() for r in caplog.records if text in r.getMessage()]
    assert len(first) == 1, [r.getMessage() for r in caplog.records]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        solver.integrate(f, state=state, take_gradient=False, **options)
    assert not [r for r in caplog.records if text in r.getMessage()]


@pytest.mark.parametrize(
    "options",
    [
        {},
        # The values stay on the solver's device: no warning (the author's choice).
        {
            "trade_speed_for_memory": True,
            "evaluation_buffer": 10**6,
            "result_device": None,
        },
    ],
    ids=["default", "trade_same_device"],
)
def test_no_warning(options, caplog):
    solver = adaptive_quadrature("uniform", method="gk21", device="cpu")
    with caplog.at_level(logging.WARNING, logger="padaquad.quadrature.base"):
        solver.integrate(torch.sin, **BOUNDS, take_gradient=False, **options)
    assert not caplog.records


# ---------------------------------------------------------------------------
# (e) record_evaluations=False
# ---------------------------------------------------------------------------


def test_no_record_state_mesh_is_power_law():
    """Without values the state's mesh is the one merge_rule="power_law" builds
    from the same run."""
    no_record, no_record_state, _ = _run(
        f=_peak,
        method="bosh3",
        atol=1e-9,
        rtol=1e-6,
        max_batch=50,
        record_evaluations=False,
    )
    power_law, power_law_state, _ = _run(
        f=_peak,
        method="bosh3",
        atol=1e-9,
        rtol=1e-6,
        max_batch=50,
        merge_rule="power_law",
    )
    _assert_same_refinement(power_law, no_record, values=False)
    assert torch.equal(no_record_state.mesh, power_law_state.mesh)


def test_no_record_loss_receives_no_values():
    seen = []

    def loss_fxn(r):
        seen.append((r.y, r.tracked_variables))
        return r.integral.sum()

    _run(f=_tracked, max_batch=50, record_evaluations=False, loss_fxn=loss_fxn)
    assert seen and set(seen) == {(None, None)}


def test_no_record_with_take_gradient():
    """record_evaluations=False with per-batch backward: the same integral and
    gradient, no values."""
    results = {}
    for record in (True, False):
        theta = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        result, _, _ = _run(
            f=lambda t, *a: torch.sin(theta * t),
            take_gradient=True,
            max_batch=50,
            record_evaluations=record,
        )
        results[record] = (result, theta.grad)
    _assert_same_refinement(results[True][0], results[False][0], values=False)
    assert results[False][0].y is None
    assert torch.equal(results[True][1], results[False][1])


# ---------------------------------------------------------------------------
# (f) Every feature gives the same result with every setting
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
    "nonfinite": {"f": _nan_tail, "runtime_checks": False},
    "tracked": {"f": _tracked},
    "three_outputs": {"f": _three_outputs},
    "failure_fraction": {"f": _three_outputs, "error_norm": "failure_fraction"},
    "custom_loss": {"f": _peak, "loss_fxn": lambda r: 2.0 * r.integral.sum()},
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_batch", [7, 50])
@pytest.mark.parametrize("setting", list(SETTINGS))
@pytest.mark.parametrize("feature", list(FEATURES))
def test_features_unchanged(feature, setting, max_batch, device):
    """On the CPU result_device is the solver's device, so nothing moves there;
    the CUDA cases are the ones where the values leave the solver's device."""
    other = _compare(setting, max_batch=max_batch, device=device, **FEATURES[feature])
    if feature == "custom_loss":
        default, _, _ = _run(max_batch=max_batch, device=device, **FEATURES[feature])
        assert torch.equal(default.loss, other.loss)
