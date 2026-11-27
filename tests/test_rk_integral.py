"""Unit tests for the core _RK_integral function."""

from __future__ import annotations

import pytest
import torch

from padaquad import UNIFORM_METHODS
import padaquad.quadrature.base as qbase
from padaquad.quadrature.base import _RK_integral


class TestRKIntegralBasic:
    """Test _RK_integral with hand-computed inputs."""

    def test_single_step_constant(self):
        """Constant integrand y=1 over [0,1] with trapezoidal weights."""
        h = torch.tensor([[1.0]])  # [1, 1]
        y = torch.tensor([[[1.0], [1.0]]])  # [1, 2, 1]
        b = torch.tensor([[[0.5], [0.5]]])  # [1, 2, 1]
        y0 = torch.tensor([0.0])

        integral, rk_steps = _RK_integral(h, y, b, y0)

        assert torch.allclose(integral, torch.tensor([1.0]))
        assert torch.allclose(rk_steps, torch.tensor([[1.0]]))

    def test_single_step_linear_fehlberg2(self):
        """Linear integrand y=t over [0,1] with fehlberg2 weights gives exact 0.5."""
        c = UNIFORM_METHODS["fehlberg2"].tableau.c  # [0, 0.5, 1]
        b_vals = UNIFORM_METHODS["fehlberg2"].tableau.b  # [1/512, 255/256, 1/512]
        h = torch.ones(1, 1, dtype=torch.float64)
        y = c.unsqueeze(0).unsqueeze(-1)  # y = t
        b = b_vals.unsqueeze(0).unsqueeze(-1)  # [1, 3, 1]
        y0 = torch.tensor([0.0], dtype=torch.float64)

        integral, _ = _RK_integral(h, y, b, y0)

        assert torch.allclose(integral, torch.tensor([0.5], dtype=torch.float64))

    def test_multiple_steps(self):
        """Two half-steps of constant y=1 sum to 1.0."""
        h = torch.tensor([[0.5], [0.5]])  # [2, 1]
        y = torch.ones(2, 2, 1)
        b = torch.tensor([[[0.5], [0.5]]])  # [1, 2, 1] broadcast over N
        y0 = torch.tensor([0.0])

        integral, rk_steps = _RK_integral(h, y, b, y0)

        assert torch.allclose(integral, torch.tensor([1.0]))
        assert torch.allclose(rk_steps, torch.tensor([[0.5], [0.5]]))

    def test_nonzero_y0(self):
        """y0 is added to the integral result."""
        h = torch.tensor([[1.0]])
        y = torch.tensor([[[1.0], [1.0]]])
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([5.0])

        integral, _ = _RK_integral(h, y, b, y0)

        assert torch.allclose(integral, torch.tensor([6.0]))

    def test_multidim_output(self):
        """D=2: each output dimension integrated independently."""
        h = torch.tensor([[1.0]])  # [1, 1]
        y = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])  # [1, 2, 2]
        b = torch.tensor([[[0.5], [0.5]]])  # [1, 2, 1]
        y0 = torch.tensor([0.0, 0.0])

        integral, rk_steps = _RK_integral(h, y, b, y0)

        assert integral.shape == (2,)
        assert torch.allclose(integral, torch.tensor([1.0, 2.0]))
        assert rk_steps.shape == (1, 2)

    def test_zero_width_step(self):
        """Step with h=0 contributes nothing to the integral."""
        h = torch.tensor([[0.0]])
        y = torch.tensor([[[100.0], [100.0]]])  # large y, but h=0
        b = torch.tensor([[[0.5], [0.5]]])
        y0 = torch.tensor([3.0])

        integral, rk_steps = _RK_integral(h, y, b, y0)

        assert torch.allclose(integral, torch.tensor([3.0]))  # y0 only
        assert torch.allclose(rk_steps, torch.tensor([[0.0]]))

    def test_variable_b_per_step(self):
        """Different b weights per step (simulating variable solver)."""
        h = torch.tensor([[1.0], [1.0]])  # [2, 1]
        y = torch.ones(2, 3, 1)  # constant y=1
        # Different weights per step
        b = torch.tensor(
            [
                [[0.25], [0.5], [0.25]],
                [[1.0 / 6], [4.0 / 6], [1.0 / 6]],
            ]
        )  # [2, 3, 1]
        y0 = torch.tensor([0.0])

        integral, rk_steps = _RK_integral(h, y, b, y0)

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
        """Output shapes match [D] and [N, D] for arbitrary N and D."""
        C, T = 4, 1
        h = torch.full((N, T), 1.0 / N)
        y = torch.ones(N, C, D)
        b = torch.ones(1, C, 1) / C
        y0 = torch.zeros(D)

        integral, rk_steps = _RK_integral(h, y, b, y0)

        assert integral.shape == (D,)
        assert rk_steps.shape == (N, D)
        assert torch.allclose(integral, torch.ones(D))


@pytest.mark.parametrize("N", [0, 1, 7, 50])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_blocks_give_the_same_bits(monkeypatch, N, dtype):
    """The weighted sum formed in blocks of rows (here 3 subintervals of
    C = 5 nodes and D = 4 outputs, by lowering _BLOCK_BYTES) equals one
    sum over all rows bit for bit, with per-subinterval and expanded weights;
    N = 0 gives empty contributions and the integral y0."""
    g = torch.Generator().manual_seed(N)
    C, D = 5, 4
    h = torch.rand(N, 1, generator=g, dtype=dtype)
    y = torch.randn(N, C, D, generator=g, dtype=dtype)
    y0 = torch.randn(D, generator=g, dtype=dtype)
    for b in (
        torch.rand(N, C, 1, generator=g, dtype=dtype),
        torch.rand(C, 1, generator=g, dtype=dtype).expand(N, -1, -1),
    ):
        one = _RK_integral(h, y, b, y0)
        monkeypatch.setattr(qbase, "_BLOCK_BYTES", 3 * C * D * y.element_size())
        blocks = _RK_integral(h, y, b, y0)
        monkeypatch.undo()
        for x, z in zip(one, blocks, strict=True):
            assert torch.equal(x, z)
        assert blocks[1].shape == (N, D)
        assert torch.equal(one[1], h * torch.sum(b * y, dim=1))
