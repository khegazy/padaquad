"""Unit tests for SolverBase methods: _set_dtype, set_dtype_by_input, _check_variables, _integral_loss."""

from __future__ import annotations

import pytest
import torch
from _helpers import make_solver_for_unit_test

from padaquad import UNIFORM_METHODS
from padaquad.results import IntegrationResult


def _save_tableau(method_name):
    """Save a copy of a uniform method's tableau tensors (they are singletons)."""
    tab = UNIFORM_METHODS[method_name].tableau
    return {
        "c": tab.c.clone(),
        "b": tab.b.clone(),
        "b_error": tab.b_error.clone(),
    }


def _restore_tableau(method_name, saved):
    """Restore a uniform method's tableau tensors from saved copies."""
    tab = UNIFORM_METHODS[method_name].tableau
    tab.c = saved["c"]
    tab.b = saved["b"]
    tab.b_error = saved["b_error"]


# ---------------------------------------------------------------------------
# _set_dtype
# ---------------------------------------------------------------------------


class TestSetDtype:
    """Tests for SolverBase._set_dtype: dtype conversion and assertion tolerances."""

    def test_float64_tolerances(self):
        """float64 sets tight assertion tolerances."""
        solver = make_solver_for_unit_test()
        solver._set_dtype(torch.float64)
        assert solver.atol_assert == 1e-15
        assert solver.rtol_assert == 1e-7

    def test_float32_tolerances(self):
        """float32 sets medium assertion tolerances."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver._set_dtype(torch.float32)
            assert solver.atol_assert == 1e-7
            assert solver.rtol_assert == 1e-5
        finally:
            _restore_tableau("bosh3", saved)

    def test_float16_raises(self):
        """float16 is refused at _set_dtype time (Bug B4 fix). Its ~1e-3
        precision floor exceeds typical adaptive tolerances; the
        previous "loose-tolerance" path silently produced wrong results.
        """
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            with pytest.raises(ValueError, match=r"float16|coarse"):
                solver._set_dtype(torch.float16)
        finally:
            _restore_tableau("bosh3", saved)

    def test_invalid_dtype_raises(self):
        """Non-float dtype raises ValueError."""
        solver = make_solver_for_unit_test()
        try:
            solver._set_dtype(torch.int32)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_noop_same_dtype(self):
        """Calling with the same dtype is a no-op."""
        solver = make_solver_for_unit_test()
        solver._set_dtype(torch.float64)
        solver._set_dtype(torch.float64)
        assert solver.dtype == torch.float64

    def test_converts_y0(self):
        """y0 dtype matches after conversion."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver._set_dtype(torch.float32)
            assert solver.y0.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)

    def test_converts_t_init_t_final(self):
        """mesh_init and mesh_final dtype match after conversion."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver._set_dtype(torch.float32)
            assert solver.mesh_init.dtype == torch.float32
            assert solver.mesh_final.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)

    def test_converts_cached_barriers(self):
        """mesh_previous is converted if set."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        solver.mesh_previous = torch.linspace(0, 1, 5, dtype=torch.float64).unsqueeze(
            -1
        )
        try:
            solver._set_dtype(torch.float32)
            assert solver.mesh_previous.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)


# ---------------------------------------------------------------------------
# set_dtype_by_input
# ---------------------------------------------------------------------------


class TestSetDtypeByInput:
    """Tests for SolverBase.set_dtype_by_input: infer dtype from tensors."""

    def test_infers_from_t(self):
        """Dtype inferred from t when provided."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver.set_dtype_by_input(mesh=torch.tensor([0.0], dtype=torch.float32))
            assert solver.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)

    def test_infers_from_t_init(self):
        """Dtype inferred from mesh_init when t is None."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver.set_dtype_by_input(
                mesh=None, mesh_init=torch.tensor([0.0], dtype=torch.float32)
            )
            assert solver.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)

    def test_infers_from_t_final(self):
        """Dtype inferred from mesh_final when t and mesh_init are None."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver.set_dtype_by_input(
                mesh=None,
                mesh_init=None,
                mesh_final=torch.tensor([1.0], dtype=torch.float32),
            )
            assert solver.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)

    def test_all_none_noop(self):
        """All None leaves dtype unchanged."""
        solver = make_solver_for_unit_test()
        solver.set_dtype_by_input(mesh=None, mesh_init=None, mesh_final=None)
        assert solver.dtype == torch.float64

    def test_priority_t_over_t_init(self):
        """t takes priority over mesh_init."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver.set_dtype_by_input(
                mesh=torch.tensor([0.0], dtype=torch.float32),
                mesh_init=torch.tensor([0.0], dtype=torch.float64),
            )
            assert solver.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)


# ---------------------------------------------------------------------------
# _check_variables
# ---------------------------------------------------------------------------


class TestCheckVariables:
    """Tests for SolverBase._check_variables: default filling and type conversion."""

    def test_fills_defaults(self):
        """All None args are replaced with stored defaults."""
        solver = make_solver_for_unit_test()
        _f, mesh_init, mesh_final, y0 = solver._check_variables()
        assert mesh_init is not None
        assert mesh_final is not None
        assert y0 is not None
        assert torch.equal(mesh_init, solver.mesh_init)
        assert torch.equal(mesh_final, solver.mesh_final)
        assert torch.equal(y0, solver.y0)

    def test_overrides_with_args(self):
        """Explicit args override stored defaults."""
        solver = make_solver_for_unit_test()
        custom_t_init = torch.tensor([2.0], dtype=torch.float64)
        custom_t_final = torch.tensor([5.0], dtype=torch.float64)
        custom_y0 = torch.tensor([10.0], dtype=torch.float64)

        _, mesh_init, mesh_final, y0 = solver._check_variables(
            mesh_init=custom_t_init, mesh_final=custom_t_final, y0=custom_y0
        )
        assert torch.equal(mesh_init, custom_t_init)
        assert torch.equal(mesh_final, custom_t_final)
        assert torch.equal(y0, custom_y0)

    def test_converts_dtype(self):
        """Tensors are converted to solver's dtype."""
        solver = make_solver_for_unit_test()
        t_init_f32 = torch.tensor([0.0], dtype=torch.float32)
        _, mesh_init, _, _ = solver._check_variables(mesh_init=t_init_f32)
        assert mesh_init.dtype == torch.float64


# ---------------------------------------------------------------------------
# _integral_loss
# ---------------------------------------------------------------------------


class TestIntegralLoss:
    """Tests for SolverBase._integral_loss: default loss returns integral."""

    def test_returns_integral(self):
        """Default loss is the integral value itself."""
        solver = make_solver_for_unit_test()
        output = IntegrationResult(integral=torch.tensor([5.0]))
        result = solver._integral_loss(output)
        assert torch.equal(result, torch.tensor([5.0]))

    def test_multidim(self):
        """Works with multi-dimensional integral."""
        solver = make_solver_for_unit_test()
        output = IntegrationResult(integral=torch.tensor([1.0, 2.0]))
        result = solver._integral_loss(output)
        assert torch.equal(result, torch.tensor([1.0, 2.0]))
