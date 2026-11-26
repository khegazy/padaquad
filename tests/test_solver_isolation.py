"""Cross-solver isolation tests.

Both registries hold float64 CPU methods that are never modified. A solver
uses the method converted to its device and dtype by ``MethodClass.to``: a copy
built once per process and shared by every solver of that method, device and
dtype (it replaced a deep copy per solver, ``MethodClass.clone``, and the
in-place ``to_dtype``/``to_device``, whose use on a shared instance once
degraded ``UNIFORM_METHODS["dopri5"]`` to float32 for the rest of a session).

These tests pin what sharing must keep:

  - **Dtype mixing**: solvers of one method at different dtypes, built and
    used in any order, keep their own tables.
  - **Registry untouched**: building solvers at every dtype leaves the
    registry entry at its float64 values, and no converted method shares
    memory with it.
  - **One object per key**: ``to`` returns the same object for the same
    method, device and dtype, another for another device or dtype.
  - **No in-place writes**: no integration writes into a method's tensors
    (their version counters do not move), since a write would reach every
    solver sharing the method. ``_Tableau`` is frozen.
  - **Replaced registry entries**: a solver keeps the method it was built
    with when the registry entry is replaced, also when it runs the optimal
    mesh on another device.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from tests._helpers import DEVICE, make_uniform_solver, needs_cuda

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS, adaptive_quadrature
from padaquad.base import steps
from padaquad.methods import UniformMethodClass, _get_method, _Tableau


def _b_values(solver):
    return solver.method.tableau.b.detach().cpu().tolist()


def _method_tensors(method):
    """The tensors of a method: its tableau's, or the node fractions of a variable one."""
    if isinstance(method, UniformMethodClass):
        return [method.tableau.c, method.tableau.b, method.tableau.b_error]
    return [method.c]


def test_sequential_dtype_mixing_does_not_corrupt_float64_solver():
    """A float32 solver constructed AFTER a float64 solver must not
    degrade the float64 solver's tableau precision."""
    solver_a = make_uniform_solver("dopri5", atol=1e-8, rtol=1e-8)
    a_b_before = _b_values(solver_a)
    a_dtype_before = solver_a.method.tableau.b.dtype

    solver_b = make_uniform_solver("dopri5", atol=1e-5, rtol=1e-5, dtype=torch.float32)
    assert solver_b.method.tableau.b.dtype == torch.float32

    # solver_a unchanged.
    assert solver_a.method.tableau.b.dtype == a_dtype_before == torch.float64
    assert _b_values(solver_a) == a_b_before


def test_concurrent_solvers_at_different_dtypes_remain_independent():
    """Two solvers constructed in any order can be used in any order
    without dtype drift between them."""
    solver_a = make_uniform_solver("bosh3", atol=1e-8, rtol=1e-8)
    solver_b = make_uniform_solver("bosh3", atol=1e-5, rtol=1e-5, dtype=torch.float32)

    # Run a round of integrations on each, alternating.
    a_b_snap = _b_values(solver_a)
    b_b_snap = _b_values(solver_b)

    for _ in range(3):
        solver_a.integrate(
            f=torch.sin,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([1.0], dtype=torch.float64),
        )
        solver_b.integrate(
            f=torch.sin,
            mesh_init=torch.tensor([0.0], dtype=torch.float32),
            mesh_final=torch.tensor([1.0], dtype=torch.float32),
        )

    assert solver_a.method.tableau.b.dtype == torch.float64
    assert solver_b.method.tableau.b.dtype == torch.float32
    assert _b_values(solver_a) == a_b_snap
    assert _b_values(solver_b) == b_b_snap


def test_global_singleton_never_mutated_by_solver_construction():
    """The canonical UNIFORM_METHODS["dopri5"] must remain bit-stable
    in both dtype AND values regardless of how many solvers we
    construct at any dtype.
    """
    canonical = UNIFORM_METHODS["dopri5"]
    snap_b = canonical.tableau.b.clone()
    snap_c = canonical.tableau.c.clone()
    snap_b_error = canonical.tableau.b_error.clone()
    snap_dtype = canonical.tableau.b.dtype

    for dtype in (torch.float64, torch.float32, torch.float64, torch.float32):
        _get_method(steps.ADAPTIVE_UNIFORM, "dopri5", "cpu", dtype)

    assert canonical.tableau.b.dtype == snap_dtype == torch.float64
    assert torch.equal(canonical.tableau.b, snap_b)
    assert torch.equal(canonical.tableau.c, snap_c)
    assert torch.equal(canonical.tableau.b_error, snap_b_error)


@pytest.mark.parametrize(
    "source",
    [UNIFORM_METHODS["dopri5"], VARIABLE_METHODS["interpolatory3_variable"]],
    ids=["uniform", "variable"],
)
def test_to_returns_one_shared_copy_per_device_and_dtype(source):
    """``to`` returns the same object for the same method, device and dtype
    (however the device is spelled), another one for another dtype; each is a
    copy that shares no memory with its source, also at the source's own
    device and dtype, and the source is unchanged."""
    before = [t.clone() for t in _method_tensors(source)]
    same64 = source.to("cpu", torch.float64)
    assert same64 is source.to(torch.device("cpu"), torch.float64)
    as32 = source.to("cpu", torch.float32)
    assert as32 is not same64
    assert as32 is source.to("cpu", torch.float32)
    for converted, dtype in ((same64, torch.float64), (as32, torch.float32)):
        assert converted is not source
        assert type(converted) is type(source)
        assert converted.degree == source.degree
        assert converted.error_exponent == source.error_exponent
        for t, s in zip(
            _method_tensors(converted), _method_tensors(source), strict=True
        ):
            assert t.dtype == dtype
            assert t.data_ptr() != s.data_ptr()
            assert torch.equal(t, s.to(dtype))
    for t, b in zip(_method_tensors(source), before, strict=True):
        assert t.dtype == torch.float64
        assert torch.equal(t, b)
    if not isinstance(source, UniformMethodClass):
        # A module-level function that holds no state: shared on purpose.
        assert as32.weight_function is source.weight_function


def test_tableau_is_frozen():
    """A shared method's tableau cannot be reassigned."""
    tableau = UNIFORM_METHODS["bosh3"].to("cpu", torch.float32).tableau
    with pytest.raises(dataclasses.FrozenInstanceError):
        tableau.b = torch.zeros_like(tableau.b)


@pytest.mark.parametrize("sampling", ["uniform", "variable"])
def test_integrations_do_not_write_into_method_tensors(sampling):
    """No integration writes into the tensors of the solver's method or of the
    registry entry: their version counters, which every in-place operation
    raises, do not move. Covers both solvers, take_gradient True and False,
    result_device on the solver's device and on the CPU (on a CUDA solver,
    the method converted for the optimal mesh), the merge prediction and
    record_evaluations=False."""
    solver = adaptive_quadrature(sampling, atol=1e-9, rtol=1e-7, device=DEVICE)
    registry = UNIFORM_METHODS if sampling == "uniform" else VARIABLE_METHODS
    tensors = _method_tensors(solver.method)
    tensors += _method_tensors(registry[solver.method_name])
    tensors += _method_tensors(solver.method.to("cpu", torch.float64))
    versions = [t._version for t in tensors]

    def f(t):
        return torch.cat([torch.sin(9 * t), torch.exp(-t) * t], dim=-1)

    a, b = (torch.tensor([x], dtype=torch.float64) for x in (0.0, 1.0))
    for options in (
        {"take_gradient": False},
        {"take_gradient": True},
        {"take_gradient": False, "result_device": DEVICE},
        {"take_gradient": False, "result_device": "cpu"},
        {"take_gradient": False, "record_evaluations": False},
    ):
        _, state = solver.integrate(
            f, mesh_init=a, mesh_final=b, max_batch=500, **options
        )
        solver.integrate(f, state=state, **options)
    assert [t._version for t in tensors] == versions


def test_replaced_registry_entry_reaches_new_solvers_only():
    """A solver keeps the method it was built with when the registry entry
    under its name is replaced; a new solver gets the new entry."""
    f64 = torch.float64

    def rule(b):
        return UniformMethodClass(
            degree=1,
            error_exponent=2,
            tableau=_Tableau(
                c=torch.tensor([0.0, 1.0], dtype=f64),
                b=torch.tensor(b, dtype=f64),
                b_error=torch.tensor([0.5, -0.5], dtype=f64),
            ),
        )

    UNIFORM_METHODS["isolation_rule"] = rule([0.5, 0.5])
    try:
        old = adaptive_quadrature("uniform", "isolation_rule", device="cpu")
        UNIFORM_METHODS["isolation_rule"] = rule([0.25, 0.75])
        new = adaptive_quadrature("uniform", "isolation_rule", device="cpu")
        assert old.method.tableau.b.tolist() == [0.5, 0.5]
        assert new.method.tableau.b.tolist() == [0.25, 0.75]
    finally:
        del UNIFORM_METHODS["isolation_rule"]


@needs_cuda
def test_optimal_mesh_on_another_device_keeps_the_solvers_method():
    """With result_device="cpu" on a CUDA solver, the optimal mesh runs with
    the method converted to the CPU (``_integration_on_device``); afterwards
    the solver holds the same CUDA method object as before, which is still
    what ``_get_method`` returns, and the CPU copy is the one ``to`` gives."""
    solver = adaptive_quadrature("uniform", "gk21", device="cuda")
    method = solver.method
    seen = []
    get_optimal_mesh = solver._get_optimal_mesh

    def spy(*args, **kwargs):
        seen.append((solver.device, solver.method))
        return get_optimal_mesh(*args, **kwargs)

    solver._get_optimal_mesh = spy
    a, b = (torch.tensor([x], dtype=torch.float64) for x in (0.0, 1.0))
    solver.integrate(torch.sin, mesh_init=a, mesh_final=b, result_device="cpu")
    assert seen == [(torch.device("cpu"), method.to("cpu", torch.float64))]
    assert solver.method is method
    assert solver.method.tableau.b.device == solver.device
    assert method is _get_method(
        steps.ADAPTIVE_UNIFORM, "gk21", solver.device, torch.float64
    )
    assert method is UNIFORM_METHODS["gk21"].to("cuda", torch.float64)
