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
  * The ``_get_method`` factory's clone path (uniform) and per-instance
    construction path (variable) could fall out of sync with the
    family-style each method belongs to.

This file pins all four invariants. It is a lightweight registry-shape
sanity net for any future family addition.
"""

from __future__ import annotations

import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS, adaptive_quadrature
from padaquad.base import steps
from padaquad.methods import (
    CC_METHODS,
    GK_METHODS,
    INTERPOLATORY_METHODS,
    RK_METHODS,
    MethodClass,
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
    """Singleton ``MethodClass`` entries must have a populated
    ``_Tableau`` with c/b/b_error tensors of matching length.
    """
    for name, method in UNIFORM_METHODS.items():
        assert isinstance(method, MethodClass), (
            f"{name}: expected MethodClass, got {type(method).__name__}"
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
        # b and b_error must broadcast against C; shape compatibility
        # check: their last dim equals len(c).
        assert c.shape[-1] == b.shape[-1] == b_error.shape[-1], (
            f"{name}: shape mismatch c={tuple(c.shape)} b={tuple(b.shape)} "
            f"b_error={tuple(b_error.shape)}"
        )
        # order is a positive int.
        assert isinstance(method.order, int), (
            f"{name}: order has type {type(method.order).__name__}, expected int"
        )
        assert method.order >= 1, f"{name}: order={method.order!r}, expected >= 1"


def test_get_method_returns_clone_for_every_uniform_method():
    """Every registered uniform method must be retrievable via
    ``_get_method`` — and the returned instance must NOT alias the
    canonical singleton (clone isolation guarantee).
    """
    for name, canonical in UNIFORM_METHODS.items():
        instance = _get_method(steps.ADAPTIVE_UNIFORM, name, "cpu", torch.float64)
        assert instance is not canonical, (
            f"{name}: _get_method returned the singleton, not a clone"
        )
        assert instance.tableau is not canonical.tableau, (
            f"{name}: tableau aliases the canonical singleton"
        )
        # Tensor identity: distinct storage.
        assert instance.tableau.b.data_ptr() != canonical.tableau.b.data_ptr(), (
            f"{name}: b tensor aliases the singleton"
        )


def test_get_method_returns_fresh_instance_for_every_variable_method():
    """Variable methods are constructed fresh per-solver (they manage
    their own per-instance state). Verify each one constructs OK at
    both supported dtypes.
    """
    for name in VARIABLE_METHODS:
        for dtype in (torch.float64, torch.float32):
            instance = _get_method(steps.ADAPTIVE_VARIABLE, name, "cpu", dtype)
            # Variable subclasses expose tableau_b(c) — this is the
            # variable-method calling convention. Just verify it's not
            # accidentally a uniform MethodClass.
            assert hasattr(instance, "tableau_b"), (
                f"{name}: variable method missing tableau_b method"
            )


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


def test_uniform_methods_have_canonical_first_and_last_nodes():
    """The solver framework computes step length as
    ``h = t[:, -1] - t[:, 0]`` assuming ``c[0] == 0`` and ``c[-1] == 1``.
    Methods whose intrinsic nodes are interior (e.g. Gauss-Kronrod) MUST
    pad with zero-weight endpoint nodes to satisfy this contract;
    otherwise the integrand is rescaled to the wrong interval.

    This test catches the regression that originally produced
    ``gk21(t) = 0.4978`` instead of ``0.5`` for ``f(t) = t``.
    """
    for name, method in UNIFORM_METHODS.items():
        c = method.tableau.c
        # Some methods have c shape [C], others [1, C]. Flatten to
        # 1-D for the boundary check.
        c_flat = c.flatten()
        assert float(c_flat[0]) == 0.0, (
            f"{name}: c[0] = {float(c_flat[0])}, expected 0.0 "
            f"(solver assumes step starts at c[0])"
        )
        assert float(c_flat[-1]) == 1.0, (
            f"{name}: c[-1] = {float(c_flat[-1])}, expected 1.0 "
            f"(solver assumes step ends at c[-1])"
        )


def test_uniform_methods_b_weights_sum_to_one():
    """A weighted-sum quadrature rule must have ``sum(b) == 1`` for
    integration over ``[0, 1]``. This is the most basic correctness
    check on the tableau's ``b`` row and is independent of polynomial
    exactness or method order.
    """
    for name, method in UNIFORM_METHODS.items():
        b = method.tableau.b.flatten()
        b_sum = float(b.sum())
        assert abs(b_sum - 1.0) < 1e-12, (
            f"{name}: sum(b) = {b_sum}, expected 1.0 "
            f"(broken weights produce wrong magnitude regardless of order)"
        )


def test_no_method_name_appears_in_both_registries_with_different_class_kinds():
    """``adaptive_heun`` is the only method that legitimately appears
    in both registries (same name, different family). Any other
    name collision is a bug.

    The dual registration is intentional because adaptive_heun is the
    second-order rule that admits both fixed and variable nodes — the
    sampling_type chooses which version is used. Verify this is the
    only such case.
    """
    overlap = set(UNIFORM_METHODS.keys()) & set(VARIABLE_METHODS.keys())
    assert overlap == {"adaptive_heun"}, (
        f"unexpected uniform/variable name collision: {overlap}"
    )
