"""Integration tests for gradient-based optimization through the solver."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    ATOL_LOOSE,
    REMOVE_CUT,
    RTOL_LOOSE,
    SEED,
    ScaledIntegrand,
    cached_max_batch,
)
from torch import nn

from padaquad import adaptive_quadrature, steps
from padaquad.examples import _WS_MIN_FINAL, _WS_MIN_INIT

GRADIENT_METHODS = ["bosh3", "dopri5"]
T_INIT = torch.tensor([0], dtype=torch.float64)
T_FINAL = torch.tensor([1], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ExpIntegrand(nn.Module):
    """f(t) = scale * (exp(-2t) + 3*exp(-3t)), with learnable scale."""

    __name__ = "_ExpIntegrand"

    def __init__(self, scale=1.0):
        """Initialize with a learnable scale parameter."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale], dtype=torch.float64))

    def forward(self, t, *args):
        """Evaluate scale * (exp(-2t) + 3*exp(-3t)), returning shape [N, 1]."""
        while len(t.shape) < 2:
            t = t.unsqueeze(0)
        return self.scale * (torch.exp(-2 * t) + 3 * torch.exp(-3 * t))


def _exp_analytical_integral(mesh_init=0.0, mesh_final=1.0):
    """Analytical integral of exp(-2t) + 3*exp(-3t) from mesh_init to mesh_final."""
    # Antiderivative: -exp(-2t)/2 - exp(-3t)
    F_final = -torch.exp(
        torch.tensor(-2.0 * mesh_final, dtype=torch.float64)
    ) / 2 - torch.exp(torch.tensor(-3.0 * mesh_final, dtype=torch.float64))
    F_init = -torch.exp(
        torch.tensor(-2.0 * mesh_init, dtype=torch.float64)
    ) / 2 - torch.exp(torch.tensor(-3.0 * mesh_init, dtype=torch.float64))
    return F_final - F_init


class _PathIntegrand(nn.Module):
    """Parameterized path over the Wolf-Schlegel potential.

    Uses radial basis functions to deform a linear path between WS minima.
    The offsets parameter controls deformations perpendicular to the base path.
    """

    __name__ = "_PathIntegrand"

    def __init__(self, n_control=5):
        """Initialize with n_control learnable 2D offset points."""
        super().__init__()
        self.offsets = nn.Parameter(
            torch.randn(n_control, 2, dtype=torch.float64) * 0.1
        )
        # RBF centers evenly spaced in [0, 1]
        self.register_buffer(
            "centers", torch.linspace(0, 1, n_control, dtype=torch.float64)
        )
        self.register_buffer("ws_init", _WS_MIN_INIT.double())
        self.register_buffer("ws_delta", (_WS_MIN_FINAL - _WS_MIN_INIT).double())

    def forward(self, t, *args):
        """Evaluate the WS potential along the parameterized path, returning [N, 1]."""
        while len(t.shape) < 2:
            t = t.unsqueeze(0)
        # Base linear path: [N, 2]
        base = self.ws_init + t * self.ws_delta
        # RBF deformation: Gaussian basis functions centered at self.centers
        # t shape: [N, 1], centers shape: [K] → dists: [N, K]
        dists = (t - self.centers.unsqueeze(0)) ** 2
        weights = torch.exp(-10.0 * dists)  # [N, K]
        # Deformation: [N, K] @ [K, 2] → [N, 2]
        deformation = weights @ self.offsets
        path = base + deformation
        x = path[:, 0:1]
        y = path[:, 1:2]
        return 10 * (x**4 + y**4 - 2 * x**2 - 4 * y**2 + x * y + 0.2 * x + 0.1 * y)


class _DerivativeNet(nn.Module):
    """Small MLP that approximates f(t). Returns [N, 1]."""

    __name__ = "_DerivativeNet"

    def __init__(self, hidden=32):
        """Initialize a 2-layer tanh MLP with given hidden size."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )

    def forward(self, t, *args):
        """Pass t through the MLP, returning shape [N, 1]."""
        while len(t.shape) < 2:
            t = t.unsqueeze(0)
        return self.net(t)


def _make_solver(method_name, f):
    """Create a solver with loose tolerances for gradient tests.

    Pinned to CPU: these gradient-correctness tests use nn.Module integrands
    whose parameters live on CPU, so the solver must be on CPU too (otherwise it
    auto-selects CUDA on a GPU machine and f(cuda_nodes) clashes with the CPU
    parameters). Gradient flow is device-agnostic, so CPU is appropriate.
    """
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=ATOL_LOOSE,
        rtol=RTOL_LOOSE,
        remove_cut=REMOVE_CUT,
        f=f,
        device="cpu",
        max_batch=cached_max_batch(),
    )


# ---------------------------------------------------------------------------
# TestQuadratureWithGradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", GRADIENT_METHODS)
class TestQuadratureWithGradients:
    """Verify analytical accuracy and gradient flow."""

    def test_exp_integral_accuracy(self, method_name):
        """Numerical quadrature of exp(-2t) + 3*exp(-3t) matches analytical."""
        torch.manual_seed(SEED)
        integrand = _ExpIntegrand(scale=1.0)
        integrand.eval()
        solver = _make_solver(method_name, integrand)

        result = solver.integrate(mesh_init=T_INIT, mesh_final=T_FINAL)
        analytical = _exp_analytical_integral(0.0, 1.0)

        assert torch.allclose(result.integral, analytical, atol=1e-3), (
            f"Expected {analytical.item():.6f}, got {result.integral.item():.6f}"
        )

    def test_exp_integral_gradient_to_params(self, method_name):
        """Gradient of integral w.r.t. scale is nonzero and positive."""
        torch.manual_seed(SEED)
        integrand = _ExpIntegrand(scale=1.0)
        integrand.eval()  # Prevent auto-backward; we do manual backward
        solver = _make_solver(method_name, integrand)

        result = solver.integrate(
            mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
        result.integral.sum().backward()

        assert integrand.scale.grad is not None
        assert integrand.scale.grad.item() > 0, (
            f"Expected positive grad, got {integrand.scale.grad.item():.6f}"
        )


# ---------------------------------------------------------------------------
# TestPathOptimization
# ---------------------------------------------------------------------------


class TestPathOptimization:
    """Optimize a parameterized path over the Wolf-Schlegel potential."""

    def test_gradients_nonzero(self):
        """After first integration, offsets.grad is nonzero."""
        torch.manual_seed(SEED)
        path_integrand = _PathIntegrand(n_control=5)
        path_integrand.eval()
        solver = _make_solver("bosh3", path_integrand)

        result = solver.integrate(
            mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
        result.integral.sum().backward()

        assert path_integrand.offsets.grad is not None
        assert path_integrand.offsets.grad.abs().sum() > 0

    def test_loss_decreases(self):
        """10 Adam steps decrease the path integral loss."""
        torch.manual_seed(SEED)
        path_integrand = _PathIntegrand(n_control=5)
        path_integrand.eval()
        optimizer = torch.optim.Adam(path_integrand.parameters(), lr=1e-2)

        # Initial loss
        solver = _make_solver("bosh3", path_integrand)
        result = solver.integrate(
            mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
        initial_loss = result.integral.sum().item()

        # Optimization loop
        for _ in range(10):
            optimizer.zero_grad()
            solver = _make_solver("bosh3", path_integrand)
            result = solver.integrate(
                mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
            )
            loss = result.integral.sum()
            loss.backward()
            optimizer.step()

        final_loss = result.integral.sum().item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"
        )


# ---------------------------------------------------------------------------
# TestQuadratureWithInitialValue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", GRADIENT_METHODS)
class TestQuadratureWithInitialValue:
    """Verify y0 + integral(f(t)dt) matches the analytical value."""

    def test_integral_with_y0_accuracy(self, method_name):
        """y0 + integral matches the analytical value for the given integrand."""
        torch.manual_seed(SEED)
        integrand = _ExpIntegrand(scale=1.0)
        integrand.eval()
        solver = _make_solver(method_name, integrand)

        result = solver.integrate(mesh_init=T_INIT, mesh_final=T_FINAL)

        # The solver computes ∫f(t)dt; add y0 externally
        y0 = torch.tensor([4.0], dtype=torch.float64)
        y_T = y0 + result.integral
        analytical = y0 + _exp_analytical_integral(0.0, 1.0)
        assert torch.allclose(y_T, analytical, atol=1e-3), (
            f"Expected {analytical.item():.6f}, got {y_T.item():.6f}"
        )

    def test_neural_integrand(self, method_name):
        """Train MLP to approximate f(t), integrate, verify accuracy."""
        torch.manual_seed(SEED)

        # Pre-train the MLP to approximate exp(-2t) + 3*exp(-3t)
        net = _DerivativeNet(hidden=64)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

        t_train = torch.linspace(0, 1, 100, dtype=torch.float64).unsqueeze(-1)
        y_target = torch.exp(-2 * t_train) + 3 * torch.exp(-3 * t_train)

        for _ in range(1000):
            optimizer.zero_grad()
            y_pred = net(t_train)
            loss = torch.nn.functional.mse_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()

        # Verify pre-training worked
        with torch.no_grad():
            y_pred = net(t_train)
            pretrain_error = (y_pred - y_target).abs().max()
        assert pretrain_error < 0.05, (
            f"Pre-training failed: max_error={pretrain_error:.4f}"
        )

        # Now integrate using the trained network
        net.eval()
        solver = _make_solver(method_name, net)
        result = solver.integrate(mesh_init=T_INIT, mesh_final=T_FINAL)

        analytical = _exp_analytical_integral(0.0, 1.0)
        # Looser tolerance since the MLP is an approximation
        assert torch.allclose(result.integral, analytical, atol=0.1), (
            f"Expected ~ {analytical.item():.4f}, got {result.integral.item():.4f}"
        )


# ---------------------------------------------------------------------------
# TestTrainingLoopPattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method_name", GRADIENT_METHODS)
class TestTrainingLoopPattern:
    """Tests for end-to-end training loop patterns."""

    def test_manual_backward(self, method_name):
        """take_gradient=False: manual backward() on result.integral → param grads exist."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0)
        solver = _make_solver(method_name, integrand)

        result = solver.integrate(
            mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
        )
        result.integral.sum().backward()

        assert integrand.scale.grad is not None
        assert integrand.scale.grad.abs().item() > 0

    def test_optimizer_step_reduces_loss(self, method_name):
        """10 Adam steps with ScaledIntegrand → loss decreases."""
        torch.manual_seed(SEED)
        integrand = ScaledIntegrand(scale=2.0)
        integrand.eval()
        optimizer = torch.optim.Adam(integrand.parameters(), lr=0.01)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            solver = _make_solver(method_name, integrand)
            result = solver.integrate(
                mesh_init=T_INIT,
                mesh_final=T_FINAL,
                take_gradient=False,
            )
            # Minimize |integral|^2 (drive integral toward 0)
            loss = (result.integral**2).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
