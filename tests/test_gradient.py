"""Unit tests for gradient/backpropagation functionality."""

from __future__ import annotations

import torch
from _helpers import (
    ATOL_LOOSE,
    DEVICE,
    RTOL_LOOSE,
    SEED,
    ScaledIntegrand,
    make_solver_for_unit_test,
    make_uniform_solver,
    make_variable_solver_for_unit_test,
)

from padaquad.runge_kutta import _RK_integral

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestGradientFlowRKIntegral
# ---------------------------------------------------------------------------


class TestGradientFlowRKIntegral:
    """Tests for gradient flow through _RK_integral()."""

    def test_gradient_flows_to_y(self):
        """Gradients propagate from integral back to y."""
        t = torch.tensor([[[0.0], [1.0]]])
        y = torch.tensor([[[1.0], [1.0]]], requires_grad=True)
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([0.0])

        integral, _rk_steps, _h = _RK_integral(t, y, b, y0)
        integral.sum().backward()

        assert y.grad is not None
        assert y.grad.abs().sum() > 0

    def test_gradient_flows_to_y0(self):
        """d(integral)/d(y0) = 1.0 since integral = y0 + sum(...)."""
        t = torch.tensor([[[0.0], [1.0]]])
        y = torch.tensor([[[1.0], [1.0]]])
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([0.0], requires_grad=True)

        integral, _rk_steps, _h = _RK_integral(t, y, b, y0)
        integral.sum().backward()

        assert y0.grad is not None
        assert torch.allclose(y0.grad, torch.tensor([1.0]))

    def test_gradient_correct_value(self):
        """b=[0.5,0.5], y=[[a,a]], h=1 → d(integral)/da = 1.0."""
        a = torch.tensor([1.0], requires_grad=True)
        t = torch.tensor([[[0.0], [1.0]]])
        y = a.unsqueeze(0).unsqueeze(0).expand(1, 2, 1)
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([0.0])

        integral, _rk_steps, _h = _RK_integral(t, y, b, y0)
        # integral = y0 + h * (b0*a + b1*a) = 0 + 1*(0.5a + 0.5a) = a
        integral.sum().backward()

        assert a.grad is not None
        assert torch.allclose(a.grad, torch.tensor([1.0]))

    def test_no_gradient_when_detached(self):
        """y without requires_grad produces no gradient."""
        t = torch.tensor([[[0.0], [1.0]]])
        y = torch.tensor([[[1.0], [1.0]]])
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([0.0])

        _integral, _rk_steps, _h = _RK_integral(t, y, b, y0)

        assert y.grad is None


# ---------------------------------------------------------------------------
# TestGradientFlowThroughIntegrate
# ---------------------------------------------------------------------------


class TestGradientFlowThroughIntegrate:
    """Tests for gradient flow through the full integrate() method."""

    def test_gradient_flows_to_params(self):
        """take_gradient=False, manual backward() → param grad nonzero."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0).to(DEVICE)
        solver = make_uniform_solver(
            "bosh3", atol=ATOL_LOOSE, rtol=RTOL_LOOSE, f=integrand, device=DEVICE
        )
        result = solver.integrate(
            mesh_init=torch.tensor([0], dtype=torch.float64),
            mesh_final=torch.tensor([1], dtype=torch.float64),
            take_gradient=False,
        )

        result.integral.sum().backward()
        assert integrand.scale.grad is not None
        assert integrand.scale.grad.abs().item() > 0

    def test_take_gradient_flag(self):
        """take_gradient=True → gradient_taken is True and params have grads."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0).to(DEVICE)
        solver = make_uniform_solver(
            "bosh3", atol=ATOL_LOOSE, rtol=RTOL_LOOSE, f=integrand, device=DEVICE
        )
        result = solver.integrate(
            mesh_init=torch.tensor([0], dtype=torch.float64),
            mesh_final=torch.tensor([1], dtype=torch.float64),
            take_gradient=True,
        )

        assert result.gradient_taken is True
        # Internal loss.backward() was called, so params should have grads
        assert integrand.scale.grad is not None

    def test_gradient_taken_false(self):
        """take_gradient=False → gradient_taken is False."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0).to(DEVICE)
        solver = make_uniform_solver(
            "bosh3", atol=ATOL_LOOSE, rtol=RTOL_LOOSE, f=integrand, device=DEVICE
        )
        result = solver.integrate(
            mesh_init=torch.tensor([0], dtype=torch.float64),
            mesh_final=torch.tensor([1], dtype=torch.float64),
            take_gradient=False,
        )

        assert result.gradient_taken is False


# ---------------------------------------------------------------------------
# TestErrorDetaching
# ---------------------------------------------------------------------------


class TestErrorDetaching:
    """Tests for error estimate detaching in uniform vs variable solvers."""

    def test_uniform_error_detached(self):
        """Uniform solver: integral_error and mesh_quadrature_errors are detached."""
        solver = make_solver_for_unit_test("bosh3")
        C = len(solver.method.tableau.c)
        t = torch.linspace(0, 1, C, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
        y = torch.ones(1, C, 1, dtype=torch.float64, requires_grad=True)
        y0 = torch.tensor([0.0], dtype=torch.float64)

        method_output = solver._calculate_integral(t, y, y0)

        assert method_output.integral_error.grad_fn is None
        assert method_output.mesh_quadrature_errors.grad_fn is None

    def test_uniform_integral_keeps_grad(self):
        """Uniform solver: primary integral is part of the computation graph."""
        solver = make_solver_for_unit_test("bosh3")
        C = len(solver.method.tableau.c)
        t = torch.linspace(0, 1, C, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
        y = torch.ones(1, C, 1, dtype=torch.float64, requires_grad=True)
        y0 = torch.tensor([0.0], dtype=torch.float64)

        method_output = solver._calculate_integral(t, y, y0)

        assert method_output.integral.grad_fn is not None
        assert method_output.mesh_quadratures.grad_fn is not None

    def test_variable_error_keeps_grad(self):
        """Variable solver: error estimates are NOT detached."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        C = len(solver.method.tableau.c)
        t = torch.linspace(0, 1, C, dtype=torch.float64).unsqueeze(0).unsqueeze(-1)
        y = torch.ones(1, C, 1, dtype=torch.float64, requires_grad=True)
        y0 = torch.tensor([0.0], dtype=torch.float64)

        method_output = solver._calculate_integral(t, y, y0)

        assert method_output.integral_error.grad_fn is not None
        assert method_output.mesh_quadrature_errors.grad_fn is not None


# ---------------------------------------------------------------------------
# TestCustomLoss
# ---------------------------------------------------------------------------


class TestCustomLoss:
    """Tests for the loss_fxn parameter in integrate()."""

    def test_default_loss_equals_integral(self):
        """No loss_fxn → result.loss equals the integral."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0).to(DEVICE)
        integrand.eval()
        solver = make_uniform_solver(
            "bosh3", atol=ATOL_LOOSE, rtol=RTOL_LOOSE, f=integrand, device=DEVICE
        )
        result = solver.integrate(
            mesh_init=torch.tensor([0], dtype=torch.float64),
            mesh_final=torch.tensor([1], dtype=torch.float64),
        )

        assert torch.allclose(result.loss.cpu(), result.integral.cpu())

    def test_custom_loss_applied(self):
        """Custom loss_fxn is used instead of the default."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0).to(DEVICE)
        integrand.eval()
        solver = make_uniform_solver(
            "bosh3", atol=ATOL_LOOSE, rtol=RTOL_LOOSE, f=integrand, device=DEVICE
        )

        def double_loss(output):
            return output.integral * 2

        result = solver.integrate(
            mesh_init=torch.tensor([0], dtype=torch.float64),
            mesh_final=torch.tensor([1], dtype=torch.float64),
            loss_fxn=double_loss,
        )

        # loss should be approximately 2 * integral
        expected = result.integral.cpu() * 2
        assert torch.allclose(result.loss.cpu(), expected, rtol=1e-4)

    def test_custom_loss_gradient(self):
        """take_gradient=True with custom loss → params have grads."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0).to(DEVICE)
        solver = make_uniform_solver(
            "bosh3", atol=ATOL_LOOSE, rtol=RTOL_LOOSE, f=integrand, device=DEVICE
        )

        def square_loss(output):
            return (output.integral**2).sum()

        result = solver.integrate(
            mesh_init=torch.tensor([0], dtype=torch.float64),
            mesh_final=torch.tensor([1], dtype=torch.float64),
            take_gradient=True,
            loss_fxn=square_loss,
        )

        assert result.gradient_taken is True
        assert integrand.scale.grad is not None
        assert integrand.scale.grad.abs().item() > 0
