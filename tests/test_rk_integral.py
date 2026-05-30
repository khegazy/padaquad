"""Unit tests for the core _RK_integral function."""

from __future__ import annotations

import pytest
import torch

from padaquad import UNIFORM_METHODS
from padaquad.runge_kutta import _RK_integral


class TestRKIntegralBasic:
    """Test _RK_integral with hand-computed inputs."""

    def test_single_step_constant(self):
        """Constant integrand y=1 over [0,1] with trapezoidal weights."""
        t = torch.tensor([[[0.0], [1.0]]])  # [1, 2, 1]
        y = torch.tensor([[[1.0], [1.0]]])  # [1, 2, 1]
        b = torch.tensor([[[0.5], [0.5]]])  # [1, 2, 1]
        y0 = torch.tensor([0.0])

        integral, rk_steps, h = _RK_integral(t, y, b, y0)

        assert torch.allclose(integral, torch.tensor([1.0]))
        assert torch.allclose(rk_steps, torch.tensor([[1.0]]))
        assert torch.allclose(h, torch.tensor([[1.0]]))

    def test_single_step_linear_fehlberg2(self):
        """Linear integrand y=t over [0,1] with fehlberg2 weights gives exact 0.5."""
        c = UNIFORM_METHODS["fehlberg2"].tableau.c  # [0, 0.5, 1]
        b_vals = UNIFORM_METHODS["fehlberg2"].tableau.b  # [1/512, 255/256, 1/512]
        t = c.unsqueeze(0).unsqueeze(-1)  # [1, 3, 1]
        y = c.unsqueeze(0).unsqueeze(-1)  # y = t
        b = b_vals.unsqueeze(0).unsqueeze(-1)  # [1, 3, 1]
        y0 = torch.tensor([0.0], dtype=torch.float64)

        integral, _, _ = _RK_integral(t, y, b, y0)

        assert torch.allclose(integral, torch.tensor([0.5], dtype=torch.float64))

    def test_multiple_steps(self):
        """Two half-steps of constant y=1 sum to 1.0."""
        t = torch.tensor(
            [
                [[0.0], [0.5]],
                [[0.5], [1.0]],
            ]
        )  # [2, 2, 1]
        y = torch.ones(2, 2, 1)
        b = torch.tensor([[[0.5], [0.5]]])  # [1, 2, 1] broadcast over N
        y0 = torch.tensor([0.0])

        integral, rk_steps, h = _RK_integral(t, y, b, y0)

        assert torch.allclose(integral, torch.tensor([1.0]))
        assert torch.allclose(rk_steps, torch.tensor([[0.5], [0.5]]))
        assert torch.allclose(h, torch.tensor([[0.5], [0.5]]))

    def test_nonzero_y0(self):
        """y0 is added to the integral result."""
        t = torch.tensor([[[0.0], [1.0]]])
        y = torch.tensor([[[1.0], [1.0]]])
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([5.0])

        integral, _, _ = _RK_integral(t, y, b, y0)

        assert torch.allclose(integral, torch.tensor([6.0]))

    def test_multidim_output(self):
        """D=2: each output dimension integrated independently."""
        t = torch.tensor([[[0.0], [1.0]]])  # [1, 2, 1]
        y = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])  # [1, 2, 2]
        b = torch.tensor([[[0.5], [0.5]]])  # [1, 2, 1]
        y0 = torch.tensor([0.0, 0.0])

        integral, rk_steps, _h = _RK_integral(t, y, b, y0)

        assert integral.shape == (2,)
        assert torch.allclose(integral, torch.tensor([1.0, 2.0]))
        assert rk_steps.shape == (1, 2)

    def test_zero_width_step(self):
        """Step with h=0 contributes nothing to the integral."""
        t = torch.tensor([[[0.5], [0.5]]])  # h = 0
        y = torch.tensor([[[100.0], [100.0]]])  # large y, but h=0
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([3.0])

        integral, _rk_steps, h = _RK_integral(t, y, b, y0)

        assert torch.allclose(integral, torch.tensor([3.0]))  # y0 only
        assert torch.allclose(h, torch.tensor([[0.0]]))

    def test_variable_b_per_step(self):
        """Different b weights per step (simulating variable solver)."""
        t = torch.tensor(
            [
                [[0.0], [0.5], [1.0]],
                [[1.0], [1.5], [2.0]],
            ]
        )  # [2, 3, 1]
        y = torch.ones(2, 3, 1)  # constant y=1
        # Different weights per step
        b = torch.tensor(
            [
                [[0.25], [0.5], [0.25]],
                [[1.0 / 6], [4.0 / 6], [1.0 / 6]],
            ]
        )  # [2, 3, 1]
        y0 = torch.tensor([0.0])

        integral, rk_steps, _ = _RK_integral(t, y, b, y0)

        # Step 0: h=1, sum(b*y) = 0.25+0.5+0.25 = 1.0, contribution = 1.0
        # Step 1: h=1, sum(b*y) = 1/6+4/6+1/6 = 1.0, contribution = 1.0
        assert torch.allclose(integral, torch.tensor([2.0]))
        assert torch.allclose(rk_steps[0], torch.tensor([1.0]))
        assert torch.allclose(rk_steps[1], torch.tensor([1.0]))


class TestRKIntegralShapes:
    """Test output shapes for various input dimensions."""

    @pytest.mark.parametrize("N", [1, 5, 10])
    @pytest.mark.parametrize("D", [1, 3])
    def test_output_shapes(self, N, D):
        """Output shapes match [D], [N,D], [N,T] for arbitrary N and D."""
        C, T = 4, 1
        t = torch.linspace(0, 1, N + 1).unfold(0, 2, 1).unsqueeze(-1)  # [N, 2, 1]
        # Expand to C quadrature points per step
        t_left = t[:, 0, :]  # [N, 1]
        t_right = t[:, 1, :]  # [N, 1]
        c_vals = torch.linspace(0, 1, C)
        t_full = t_left.unsqueeze(1) + c_vals.unsqueeze(0).unsqueeze(-1) * (
            t_right - t_left
        ).unsqueeze(1)  # [N, C, T]

        y = torch.ones(N, C, D)
        b = torch.ones(1, C, 1) / C
        y0 = torch.zeros(D)

        integral, rk_steps, h = _RK_integral(t_full, y, b, y0)

        assert integral.shape == (D,)
        assert rk_steps.shape == (N, D)
        assert h.shape == (N, T)
