"""Method registry consistency tests.

Phase 5 of the quadrature alignment plan split ``methods.py`` into a
``methods/`` subpackage with one file per family (runge_kutta,
gauss_kronrod, clenshaw_curtis, interpolatory). The aggregated
``UNIFORM_METHODS`` and ``VARIABLE_METHODS`` registries are now built
in ``methods/__init__.py`` from per-family registries via dict-merge.

Risks introduced by the split:

  * A family registry could lose an entry without the aggregated
    registry's contents being checked.
  * Two families could collide on the same method name (last write wins
    silently — a method could move families with no error).
  * A method could be registered but never reachable through the public
    factory ``adaptive_quadrature(...)``.
  * A rule could be registered as an instance of a bespoke subclass
    instead of one of the two generic classes (``UniformMethodClass``,
    ``VariableMethodClass``), so it would need handling of its own.

This file pins all four invariants, plus ``_get_method``'s copy isolation
and the ``MethodClass`` interface for every registered rule. It is a lightweight registry-shape
sanity net for any future family addition.
"""

from __future__ import annotations

import pytest
import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS, adaptive_quadrature
from padaquad.base import steps
from padaquad.methods import (
    CC_METHODS,
    GK_METHODS,
    INTERPOLATORY_METHODS,
    RK_METHODS,
    MethodClass,
    UniformMethodClass,
    VariableMethodClass,
    _get_method,
)
from padaquad.methods._base import _Tableau


def test_uniform_registry_is_disjoint_union_of_family_registries():
    """``UNIFORM_METHODS`` keys = RK + GK + CC, with no name collisions
    across families. A collision would be silently swallowed by the
    dict-merge in ``methods/__init__.py``.
    """
    rk_keys = set(RK_METHODS.keys())
    gk_keys = set(GK_METHODS.keys())
    cc_keys = set(CC_METHODS.keys())

    # No cross-family name collisions.
    assert not (rk_keys & gk_keys), f"RK/GK collision: {rk_keys & gk_keys}"
    assert not (rk_keys & cc_keys), f"RK/CC collision: {rk_keys & cc_keys}"
    assert not (gk_keys & cc_keys), f"GK/CC collision: {gk_keys & cc_keys}"

    expected_uniform_keys = rk_keys | gk_keys | cc_keys
    assert set(UNIFORM_METHODS.keys()) == expected_uniform_keys, (
        f"UNIFORM_METHODS = {sorted(UNIFORM_METHODS.keys())}, "
        f"expected union = {sorted(expected_uniform_keys)}"
    )


def test_variable_registry_matches_interpolatory_family():
    """Currently the only variable-sampling family is INTERPOLATORY."""
    assert set(VARIABLE_METHODS.keys()) == set(INTERPOLATORY_METHODS.keys())


def test_uniform_and_variable_registries_have_well_known_methods():
    """Pin specific method names so an accidental delete of a method
    fails loudly. These are part of the documented API surface.
    """
    # Runge-Kutta
    for name in ("adaptive_heun", "fehlberg2", "bosh3", "dopri5"):
        assert name in UNIFORM_METHODS, f"missing RK method: {name}"

    # Gauss-Kronrod
    for name in ("gk15", "gk21", "gk31"):
        assert name in UNIFORM_METHODS, f"missing GK method: {name}"

    # Clenshaw-Curtis
    for name in ("cc17", "cc33", "cc65"):
        assert name in UNIFORM_METHODS, f"missing CC method: {name}"

    # Variable family
    assert "interpolatory3_variable" in VARIABLE_METHODS
    assert "adaptive_heun" in VARIABLE_METHODS  # also registered as variable


def test_every_uniform_method_has_well_formed_tableau():
    """Singleton ``UniformMethodClass`` entries must have a populated
    ``_Tableau`` with 1-D c/b/b_error tensors of matching length.
    """
    for name, method in UNIFORM_METHODS.items():
        assert isinstance(method, UniformMethodClass), (
            f"{name}: expected UniformMethodClass, got {type(method).__name__}"
        )
        assert isinstance(method.tableau, _Tableau), (
            f"{name}: tableau is not _Tableau ({type(method.tableau).__name__})"
        )
        c = method.tableau.c
        b = method.tableau.b
        b_error = method.tableau.b_error
        # All canonical singletons start in float64 (per Phase 4).
        assert c.dtype == torch.float64, f"{name}: c dtype={c.dtype}"
        assert b.dtype == torch.float64, f"{name}: b dtype={b.dtype}"
        assert b_error.dtype == torch.float64, f"{name}: b_error dtype={b_error.dtype}"
        # One stored shape for every method: c, b, b_error are all [C].
        assert c.ndim == b.ndim == b_error.ndim == 1, (
            f"{name}: expected 1-D tables, got c={tuple(c.shape)} "
            f"b={tuple(b.shape)} b_error={tuple(b_error.shape)}"
        )
        assert c.shape == b.shape == b_error.shape, (
            f"{name}: shape mismatch c={tuple(c.shape)} b={tuple(b.shape)} "
            f"b_error={tuple(b_error.shape)}"
        )
        # degree is an int >= 0 and error_exponent an int >= 2 (the values
        # themselves are measured from the tables in tests/test_exactness.py).
        assert type(method.degree) is int, f"{name}: degree={method.degree!r}"
        assert method.degree >= 0, f"{name}: degree={method.degree!r}, expected >= 0"
        assert type(method.error_exponent) is int, (
            f"{name}: error_exponent={method.error_exponent!r}"
        )
        assert method.error_exponent >= 2, (
            f"{name}: error_exponent={method.error_exponent!r}, expected >= 2"
        )


def _registered() -> list[tuple[steps, str]]:
    """(sampling type, name) for every method in both registries."""
    return [(steps.ADAPTIVE_UNIFORM, name) for name in UNIFORM_METHODS] + [
        (steps.ADAPTIVE_VARIABLE, name) for name in VARIABLE_METHODS
    ]


def _registry(sampling_type: steps) -> dict:
    return (
        UNIFORM_METHODS if sampling_type == steps.ADAPTIVE_UNIFORM else VARIABLE_METHODS
    )


def _tensors(method) -> list[torch.Tensor]:
    """Every tensor attribute of a method, including those of its tableau."""
    found = [v for v in vars(method).values() if torch.is_tensor(v)]
    if isinstance(method, UniformMethodClass):
        found += [method.tableau.c, method.tableau.b, method.tableau.b_error]
    return found


@pytest.mark.parametrize(("sampling_type", "name"), _registered())
def test_get_method_returns_a_copy_of_the_registry_entry(sampling_type, name):
    """``_get_method`` returns a copy of the registry instance that shares no
    tensor storage with it, at both dtypes, and leaves the registry instance
    at float64; the copy is the one ``MethodClass.to`` returns for that key.
    """
    canonical = _registry(sampling_type)[name]
    for dtype in (torch.float64, torch.float32):
        instance = _get_method(sampling_type, name, "cpu", dtype)
        assert isinstance(instance, MethodClass)
        assert instance is not canonical, (
            f"{name}: _get_method returned the registry instance, not a copy"
        )
        assert instance is canonical.to("cpu", dtype)
        canonical_ptrs = {t.data_ptr() for t in _tensors(canonical)}
        assert not canonical_ptrs & {t.data_ptr() for t in _tensors(instance)}, (
            f"{name}: a tensor aliases the registry instance"
        )
        assert all(t.dtype == dtype for t in _tensors(instance))
        assert all(t.dtype == torch.float64 for t in _tensors(canonical))


@pytest.mark.parametrize(("sampling_type", "name"), _registered())
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_method_interface(sampling_type, name, dtype):
    """Every registered rule implements the MethodClass interface the
    solvers use: ``node_fractions`` non-decreasing in [0, 1] with ``n_nodes``
    entries (from 0 to 1 for a variable method, whose weights rescale by the
    end nodes), and ``weights`` of shape exactly [N, C, 1] whose b sums to 1
    over the nodes of every panel.
    """
    method = _get_method(sampling_type, name, "cpu", dtype)
    c = method.node_fractions()
    assert c.ndim == 1
    assert len(c) == method.n_nodes
    assert torch.all((c >= 0) & (c <= 1))
    if sampling_type == steps.ADAPTIVE_VARIABLE:
        assert float(c[0]) == 0.0
        assert float(c[-1]) == 1.0
    assert torch.all(c[1:] >= c[:-1])
    # N = 4 panels of different widths, nodes at the rule's own fractions.
    left = torch.tensor([0.0, 1.0, 1.5, 4.0], dtype=dtype).unsqueeze(-1)
    width = torch.tensor([1.0, 0.5, 2.5, 0.25], dtype=dtype).unsqueeze(-1)
    nodes = (left + c.unsqueeze(0) * width).unsqueeze(-1)  # [4, C, 1]
    b, b_error = method.weights(nodes)
    assert b.shape == (4, method.n_nodes, 1)
    assert b_error.shape == (4, method.n_nodes, 1)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert torch.allclose(b.sum(dim=1), torch.ones(4, 1, dtype=dtype), atol=tol)


def test_factory_can_construct_a_solver_for_every_registered_method():
    """The ``adaptive_quadrature(...)`` factory must accept every name
    in both registries. This catches the case where a method is
    registered but the factory's name dispatch is missing it.
    """
    for name in UNIFORM_METHODS:
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_UNIFORM,
            method=name,
            atol=1e-6,
            rtol=1e-6,
        )
        assert solver is not None, f"factory failed for uniform: {name}"

    for name in VARIABLE_METHODS:
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_VARIABLE,
            method=name,
            atol=1e-6,
            rtol=1e-6,
        )
        assert solver is not None, f"factory failed for variable: {name}"


@pytest.mark.parametrize(("sampling_type", "name"), _registered())
def test_every_method_integrates_constants_and_lines_on_an_uneven_mesh(
    sampling_type, name
):
    """Each subinterval's width comes from its mesh points, not from its first
    and last node, so a method whose nodes are interior (Gauss-Kronrod) gets
    the width right. This catches the regression that originally produced
    ``gk21(t) = 0.4978`` instead of ``0.5`` for ``f(t) = t`` (the width was the
    distance between the outermost nodes)."""
    solver = adaptive_quadrature(sampling_type, name, device="cpu")
    mesh = torch.tensor([[0.0], [0.3], [1.0], [2.5]], dtype=torch.float64)
    nodes = solver._compute_nodes(mesh[:-1], mesh[1:])
    for y, exact in ((torch.ones_like(nodes), 2.5), (nodes, 2.5**2 / 2)):
        out = solver._calculate_integral(nodes, y, mesh[:-1], mesh[1:])
        assert abs(float(out.integral) - exact) < 1e-13, (name, float(out.integral))
        assert torch.equal(out.h, mesh[1:] - mesh[:-1])
        assert torch.equal(out.mesh_left, mesh[:-1])


def _gauss2() -> UniformMethodClass:
    """The 2-point Gauss-Legendre rule on [0, 1], a method with no node at
    either end: nodes 1/2 -+ sqrt(3)/6, weights 1/2 each, degree 3 (the roots
    +-1/sqrt(3) of the Legendre polynomial (3x^2 - 1)/2 and weights 1 on
    [-1, 1], mapped to [0, 1]; measure_degree confirms it below). The error
    weights subtract the rectangle rule on the first node (degree 0)."""
    f64 = torch.float64
    half_gap = 3**0.5 / 6
    return UniformMethodClass(
        degree=3,
        error_exponent=2,
        tableau=_Tableau(
            c=torch.tensor([0.5 - half_gap, 0.5 + half_gap], dtype=f64),
            b=torch.tensor([0.5, 0.5], dtype=f64),
            b_error=torch.tensor([0.5 - 1.0, 0.5], dtype=f64),
        ),
    )


@pytest.mark.parametrize("take_gradient", [False, True])
def test_a_method_without_end_nodes_integrates(monkeypatch, take_gradient):
    """A uniform method whose nodes include neither end runs end to end: the
    integral of a cubic is exact, f is never called at a mesh point (start-up
    calls included, which use the first node of the interval), and the result
    lists every evaluated point once, N * C rows, with the values f returned
    there."""
    method = _gauss2()
    measured = method.measure_degree()
    assert (measured.degree, measured.lower_degree) == (3, 0)
    monkeypatch.setitem(UNIFORM_METHODS, "gauss2", method)
    calls = []

    def f(t):
        calls.append(t.detach().clone())
        return t**3 + 1.0

    torch.manual_seed(0)
    solver = adaptive_quadrature("uniform", "gauss2", atol=1e-8, rtol=0, device="cpu")
    bounds = (
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    result, state = solver.integrate(
        f, mesh_init=bounds[0], mesh_final=bounds[1], take_gradient=take_gradient
    )
    assert result.converged
    assert abs(float(result.integral) - 6.0) < 1e-12
    n = len(result.h)
    assert result.nodes.shape == (n * 2, 1)
    assert torch.equal(result.y, result.nodes**3 + 1.0)
    evaluated = torch.cat(calls)
    assert not torch.isin(evaluated, bounds[0]).any()
    assert not torch.isin(evaluated, bounds[1]).any()
    left = bounds[0] + torch.cumsum(result.h, 0) - result.h
    assert not torch.isin(result.nodes, left).any()


def test_uniform_methods_b_weights_sum_to_one():
    """A weighted-sum quadrature rule must have ``sum(b) == 1`` for
    integration over ``[0, 1]``. This is the most basic correctness
    check on the tableau's ``b`` row and is independent of polynomial
    exactness or method order.
    """
    for name, method in UNIFORM_METHODS.items():
        b_sum = float(method.tableau.b.sum())
        assert abs(b_sum - 1.0) < 1e-12, (
            f"{name}: sum(b) = {b_sum}, expected 1.0 "
            f"(broken weights produce wrong magnitude regardless of order)"
        )


def test_only_adaptive_heun_is_in_both_registries():
    """``adaptive_heun`` is the only method that legitimately appears
    in both registries, and it is one object: its two nodes are the panel
    ends, so its weights cannot depend on node positions and the same rule
    serves both sampling modes. Any other name collision is a bug.
    """
    overlap = set(UNIFORM_METHODS.keys()) & set(VARIABLE_METHODS.keys())
    assert overlap == {"adaptive_heun"}, (
        f"unexpected uniform/variable name collision: {overlap}"
    )
    assert VARIABLE_METHODS["adaptive_heun"] is UNIFORM_METHODS["adaptive_heun"]


@pytest.mark.parametrize(("sampling_type", "name"), _registered())
def test_every_rule_is_an_instance_of_a_generic_class(sampling_type, name):
    """Every registered rule is an instance of exactly one of the two generic
    classes, never of a bespoke subclass: a new rule is data (a ``_Tableau``,
    or node fractions and a weight function), not a new class."""
    method = _registry(sampling_type)[name]
    assert type(method) in (UniformMethodClass, VariableMethodClass), (
        f"{name}: registered as {type(method).__name__}"
    )
