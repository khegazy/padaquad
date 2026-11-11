"""Unit tests for SolverBase methods: _set_dtype, set_dtype_by_input, _check_variables, _integral_loss, device selection."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    T_FINAL,
    T_INIT,
    constant_integrand,
    make_solver_for_unit_test,
    make_uniform_solver,
    needs_cuda,
)

from padaquad import UNIFORM_METHODS, integrand_dict
from padaquad.distributed import _resolve_device
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
            assert solver.init_y0.dtype == torch.float32
        finally:
            _restore_tableau("bosh3", saved)

    def test_converts_t_init_t_final(self):
        """mesh_init and mesh_final dtype match after conversion."""
        saved = _save_tableau("bosh3")
        solver = make_solver_for_unit_test()
        try:
            solver._set_dtype(torch.float32)
            assert solver.init_mesh_init.dtype == torch.float32
            assert solver.init_mesh_final.dtype == torch.float32
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

    def test_integrate_infers_dtype_from_mesh_final_only(self):
        """integrate() given only mesh_final adopts mesh_final's dtype.

        Goes through integrate(), not set_dtype_by_input directly: integrate()
        used to pass mesh_final=mesh_init, so a float32 mesh_final on a float64
        solver was ignored and the integral ran in float64.
        """
        solver = make_solver_for_unit_test()  # float64
        result = solver.integrate(
            constant_integrand,
            mesh_final=torch.tensor([2.0], dtype=torch.float32),
            max_batch=10000,
        )
        assert solver.dtype == torch.float32
        assert result.integral.dtype == torch.float32
        assert torch.allclose(result.integral, torch.tensor([2.0]))


# ---------------------------------------------------------------------------
# _check_variables
# ---------------------------------------------------------------------------


class TestCheckVariables:
    """Tests for SolverBase._check_variables: default filling and type conversion."""

    def test_fills_defaults(self):
        """All None tensor args are replaced with stored defaults."""
        solver = make_solver_for_unit_test()
        _f, mesh_init, mesh_final, y0 = solver._check_variables(f=torch.sin)
        assert mesh_init is not None
        assert mesh_final is not None
        assert y0 is not None
        assert torch.equal(mesh_init, solver.init_mesh_init)
        assert torch.equal(mesh_final, solver.init_mesh_final)
        assert torch.equal(y0, solver.init_y0)

    def test_overrides_with_args(self):
        """Explicit args override stored defaults."""
        solver = make_solver_for_unit_test()
        custom_t_init = torch.tensor([2.0], dtype=torch.float64)
        custom_t_final = torch.tensor([5.0], dtype=torch.float64)
        custom_y0 = torch.tensor([10.0], dtype=torch.float64)

        _, mesh_init, mesh_final, y0 = solver._check_variables(
            f=torch.sin,
            mesh_init=custom_t_init,
            mesh_final=custom_t_final,
            y0=custom_y0,
        )
        assert torch.equal(mesh_init, custom_t_init)
        assert torch.equal(mesh_final, custom_t_final)
        assert torch.equal(y0, custom_y0)

    def test_converts_dtype(self):
        """Tensors are converted to solver's dtype."""
        solver = make_solver_for_unit_test()
        t_init_f32 = torch.tensor([0.0], dtype=torch.float32)
        _, mesh_init, _, _ = solver._check_variables(f=torch.sin, mesh_init=t_init_f32)
        assert mesh_init.dtype == torch.float64

    def test_missing_f_raises(self):
        """No integrand here or at construction raises ValueError."""
        solver = make_solver_for_unit_test()
        assert solver.init_f is None
        with pytest.raises(ValueError, match="No integrand given"):
            solver._check_variables()

    def test_falls_back_to_constructor_f(self):
        """f=None uses the integrand given at construction."""
        solver = make_solver_for_unit_test()
        solver.init_f = torch.sin
        f, _, _, _ = solver._check_variables()
        assert f is torch.sin


# ---------------------------------------------------------------------------
# _integral_loss
# ---------------------------------------------------------------------------


def _result_with_integral(integral: torch.Tensor) -> IntegrationResult:
    """IntegrationResult whose only meaningful field is ``integral``."""
    empty = torch.zeros(0)
    return IntegrationResult(
        integral=integral,
        integral_error=torch.zeros_like(integral),
        mesh_init=torch.zeros(1),
        mesh_final=torch.ones(1),
        nodes=empty,
        h=empty,
        y=empty,
        mesh_quadratures=empty,
        mesh_quadrature_errors=empty,
        error_ratios=empty,
        gradient_taken=False,
        y0=torch.zeros_like(integral),
    )


class TestIntegralLoss:
    """Tests for SolverBase._integral_loss: default loss returns integral."""

    def test_returns_integral(self):
        """Default loss is the integral value itself."""
        solver = make_solver_for_unit_test()
        output = _result_with_integral(torch.tensor([5.0]))
        result = solver._integral_loss(output)
        assert torch.equal(result, torch.tensor([5.0]))

    def test_multidim(self):
        """Works with multi-dimensional integral."""
        solver = make_solver_for_unit_test()
        output = _result_with_integral(torch.tensor([1.0, 2.0]))
        result = solver._integral_loss(output)
        assert torch.equal(result, torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

CPU, CUDA0, CUDA1 = map(torch.device, ("cpu", "cuda:0", "cuda:1"))
needs_2_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")


class TestDeviceSelection:
    """Tests for _resolve_device and the solver's resolved ``device``."""

    @pytest.mark.parametrize(
        ("device", "index_default", "distributed", "available", "expected"),
        [
            (None, 0, False, True, CUDA0),
            (None, 0, False, False, CPU),
            ("cuda", 1, False, True, CUDA1),
            ("cuda:1", 0, False, True, CUDA1),
            (CUDA1, 0, False, True, CUDA1),
            ("cuda:1", 1, True, True, CUDA1),
            ("cpu", 0, False, True, CPU),
            ("cpu:0", 0, False, True, CPU),
            ("cuda:2", 0, False, True, ValueError),  # index not visible
            ("meta", 0, False, True, ValueError),  # unsupported type
            ("cuda:1", 0, True, True, ValueError),  # conflicts with rank's cuda:0
            ("cuda", 0, False, False, ValueError),  # CUDA unavailable
        ],
    )
    def test_resolve_device(
        self, monkeypatch, device, index_default, distributed, available, expected
    ):
        """Every spelling resolves to bare cpu or an explicitly indexed cuda, and
        invalid requests raise ValueError instead of failing later."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        if expected is ValueError:
            with pytest.raises(ValueError):
                _resolve_device(device, index_default, distributed)
        else:
            assert _resolve_device(device, index_default, distributed) == expected

    @pytest.mark.parametrize(
        ("device", "result_device", "expected", "expected_result"),
        [
            ("cpu", "cpu", CPU, CPU),
            (CPU, None, CPU, CPU),  # None keeps results on the solver's device
            (None, "cpu", CUDA0 if torch.cuda.is_available() else CPU, CPU),
            pytest.param("cpu", "cuda", CPU, CUDA0, marks=needs_cuda),
            # Index-less "cuda" resolves to the solver's GPU, not cuda:0.
            pytest.param("cuda:1", "cuda", CUDA1, CUDA1, marks=needs_2_gpus),
            pytest.param(CUDA1, "cuda:0", CUDA1, CUDA0, marks=needs_2_gpus),
        ],
    )
    def test_solver_device(self, device, result_device, expected, expected_result):
        """The solver and its results land on exactly the requested devices,
        nothing is allocated on an unrequested GPU 0, and the global default /
        current CUDA device are unchanged."""
        default = torch.get_default_device()
        # current_device() also initializes CUDA, which the stats reset requires.
        current = torch.cuda.current_device() if torch.cuda.is_available() else None
        check_gpu0 = current is not None and CUDA0 not in (expected, expected_result)
        if check_gpu0:
            torch.cuda.reset_accumulated_memory_stats(0)
        f, solution, _ = integrand_dict["t_squared"]

        def f_on_device(t):
            assert t.device == expected
            return f(t)

        solver = make_uniform_solver("gk21", device=device)
        result = solver.integrate(
            f_on_device,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            result_device=result_device,
        )

        assert solver.device == expected
        assert (solver.get_cuda_memory_usage() == -1) == (expected.type == "cpu")
        assert result.nodes.device == expected_result
        assert torch.allclose(result.integral.cpu(), solution(T_INIT, T_FINAL))
        assert torch.get_default_device() == default
        if current is not None:
            assert torch.cuda.current_device() == current
        if check_gpu0:
            assert torch.cuda.memory_stats(0)["allocated_bytes.all.allocated"] == 0
