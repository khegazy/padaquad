"""Smoke test of the public API.

Phase 0 of the quadrature alignment plan. This test imports every
documented public symbol and exercises each at its simplest signature.
Its job is to catch accidental rename omissions during the Phase 3
public-API rename pass: if any export is dropped or any documented
field disappears, this file will fail.

This test pins the *current* (ODE-framed) public API. After Phase 3
the new (quadrature-framed) public API will replace these assertions
in this same file.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests._helpers import TAKE_GRADIENT_IDS, TAKE_GRADIENT_VALUES

import padaquad
from padaquad import (
    UNIFORM_METHODS,
    VARIABLE_METHODS,
    IntegrationResult,
    UniformAdaptiveQuadrature,
    VariableAdaptiveQuadrature,
    adaptive_quadrature,
    integrand_dict,
    integrate,
    steps,
    wolf_schlegel,
)


class TestPublicExports:
    """Every documented public symbol exists at the package level."""

    def test_classes_exist(self):
        assert UniformAdaptiveQuadrature is not None
        assert VariableAdaptiveQuadrature is not None
        assert IntegrationResult is not None

    def test_functions_exist(self):
        assert callable(integrate)
        assert callable(adaptive_quadrature)

    def test_data_exports_exist(self):
        assert isinstance(UNIFORM_METHODS, dict)
        assert len(UNIFORM_METHODS) > 0
        assert isinstance(VARIABLE_METHODS, dict)
        assert len(VARIABLE_METHODS) > 0
        assert isinstance(integrand_dict, dict)
        assert len(integrand_dict) > 0
        assert callable(wolf_schlegel)

    def test_steps_enum(self):
        assert hasattr(steps, "ADAPTIVE_UNIFORM")
        assert hasattr(steps, "ADAPTIVE_VARIABLE")
        assert hasattr(steps, "FIXED")

    def test_module_has_dunder_attributes(self):
        assert hasattr(padaquad, "__name__")


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
class TestOdePathIntegralEntryPoint:
    """The free function `integrate` works for the simplest case."""

    def test_simple_sine_integral(self, take_gradient):
        result = integrate(
            f=torch.sin,
            method="dopri5",
            atol=1e-8,
            rtol=1e-6,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([math.pi], dtype=torch.float64),
            take_gradient=take_gradient,
        )
        assert isinstance(result, IntegrationResult)
        # int_0^pi sin(t) dt = 2
        assert abs(result.integral.item() - 2.0) < 1e-6

    def test_constant_integral(self, take_gradient):
        # int_0^1 1 dt = 1
        result = integrate(
            f=torch.ones_like,
            method="bosh3",
            atol=1e-10,
            rtol=1e-10,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([1.0], dtype=torch.float64),
            take_gradient=take_gradient,
        )
        assert abs(result.integral.item() - 1.0) < 1e-9


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
class TestIntegrationResultFields:
    """The IntegrationResult dataclass exposes the documented fields.

    Phase 3 renames: value/error/mesh_*/nodes replace the old
    ODE-flavored names integral/integral_error/t_optimal/t/mesh_init/mesh_final.
    """

    def _run(self, take_gradient):
        return integrate(
            f=torch.sin,
            method="dopri5",
            atol=1e-8,
            rtol=1e-6,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([math.pi], dtype=torch.float64),
            take_gradient=take_gradient,
        )

    def test_documented_fields_present(self, take_gradient):
        r = self._run(take_gradient)
        for name in (
            "integral",
            "integral_error",
            "mesh_optimal",
            "mesh_init",
            "mesh_final",
            "nodes",
            "h",
            "y",
            "tracked_variables",
            "mesh_quadratures",
            "mesh_quadrature_errors",
            "error_ratios",
            "loss",
            "gradient_taken",
            "y0",
            "converged",
        ):
            assert hasattr(r, name), f"IntegrationResult missing field {name!r}"

    def test_tracked_variables_none_by_default(self, take_gradient):
        """A bare-tensor integrand leaves tracked_variables as None."""
        r = self._run(take_gradient)
        assert r.tracked_variables is None

    def test_tracked_variables_populated_from_tuple_output(self, take_gradient):
        """An integrand returning (integrand, (tracked,)) populates the field."""

        def f(t):
            return torch.sin(t), (t**2,)

        r = integrate(
            f=f,
            method="dopri5",
            atol=1e-8,
            rtol=1e-6,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([math.pi], dtype=torch.float64),
            take_gradient=take_gradient,
        )
        assert isinstance(r.tracked_variables, tuple)
        assert len(r.tracked_variables) == 1
        # tracked_variables are flattened to align with nodes on the leading axis.
        assert r.tracked_variables[0].shape[0] == r.nodes.shape[0]


class TestSolverFactory:
    """The factory function dispatches correctly to uniform/variable solvers."""

    def test_uniform_dispatch(self):
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_UNIFORM,
            method="dopri5",
            atol=1e-6,
            rtol=1e-6,
        )
        assert isinstance(solver, UniformAdaptiveQuadrature)

    def test_variable_dispatch(self):
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_VARIABLE,
            method="interpolatory3_variable",
            atol=1e-6,
            rtol=1e-6,
        )
        assert isinstance(solver, VariableAdaptiveQuadrature)


class TestMethodRegistries:
    """The method registries contain the documented entries."""

    def test_uniform_methods_present(self):
        for name in ("adaptive_heun", "fehlberg2", "bosh3", "dopri5"):
            assert name in UNIFORM_METHODS, f"missing uniform method {name!r}"

    def test_variable_methods_present(self):
        for name in ("adaptive_heun", "interpolatory3_variable"):
            assert name in VARIABLE_METHODS, f"missing variable method {name!r}"
