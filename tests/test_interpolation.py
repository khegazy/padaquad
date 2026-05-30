"""Unit tests for _compute_nodes (quadrature point placement)."""

from __future__ import annotations

import pytest
import torch
from _helpers import UNIFORM_METHOD_NAMES, make_solver_for_unit_test

from padaquad import UNIFORM_METHODS


@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
class TestComputeNodes:
    """Tests for uniform solver _compute_nodes."""

    def _make_solver(self, method_name):
        return make_solver_for_unit_test(method_name=method_name)

    def test_endpoints(self, method_name):
        """First point equals mesh_left, last point equals mesh_right."""
        solver = self._make_solver(method_name)
        mesh_left = torch.tensor([[0.0]], dtype=torch.float64)
        mesh_right = torch.tensor([[1.0]], dtype=torch.float64)

        result = solver._compute_nodes(mesh_left, mesh_right)

        assert torch.allclose(result[:, 0, :], mesh_left)
        assert torch.allclose(result[:, -1, :], mesh_right)

    def test_shape_single_step(self, method_name):
        """Output shape is [1, C, 1] for a single step."""
        solver = self._make_solver(method_name)
        C = len(UNIFORM_METHODS[method_name].tableau.c)
        mesh_left = torch.tensor([[0.0]], dtype=torch.float64)
        mesh_right = torch.tensor([[1.0]], dtype=torch.float64)

        result = solver._compute_nodes(mesh_left, mesh_right)

        assert result.shape == (1, C, 1)

    def test_shape_multiple_steps(self, method_name):
        """Output shape is [N, C, 1] for N steps."""
        solver = self._make_solver(method_name)
        C = len(UNIFORM_METHODS[method_name].tableau.c)
        N = 5
        mesh_left = torch.linspace(0, 0.8, N, dtype=torch.float64).unsqueeze(-1)
        mesh_right = torch.linspace(0.2, 1.0, N, dtype=torch.float64).unsqueeze(-1)

        result = solver._compute_nodes(mesh_left, mesh_right)

        assert result.shape == (N, C, 1)

    def test_monotonic_within_step(self, method_name):
        """Quadrature points within each step are non-decreasing."""
        solver = self._make_solver(method_name)
        N = 3
        boundaries = torch.linspace(0, 1, N + 1, dtype=torch.float64)
        mesh_left = boundaries[:-1].unsqueeze(-1)
        mesh_right = boundaries[1:].unsqueeze(-1)

        result = solver._compute_nodes(mesh_left, mesh_right)

        for i in range(N):
            diffs = result[i, 1:, 0] - result[i, :-1, 0]
            assert torch.all(diffs >= -1e-15), (
                f"Step {i}: non-monotonic points {result[i, :, 0]}"
            )

    def test_matches_tableau_c_on_unit_step(self, method_name):
        """On a unit step [0,1], points equal the tableau c values."""
        solver = self._make_solver(method_name)
        c = UNIFORM_METHODS[method_name].tableau.c
        mesh_left = torch.tensor([[0.0]], dtype=torch.float64)
        mesh_right = torch.tensor([[1.0]], dtype=torch.float64)

        result = solver._compute_nodes(mesh_left, mesh_right)

        assert torch.allclose(result[0, :, 0], c.to(torch.float64))

    def test_scaled_step(self, method_name):
        """On [2, 4], points equal 2 + c * 2."""
        solver = self._make_solver(method_name)
        c = UNIFORM_METHODS[method_name].tableau.c.to(torch.float64)
        mesh_left = torch.tensor([[2.0]], dtype=torch.float64)
        mesh_right = torch.tensor([[4.0]], dtype=torch.float64)

        result = solver._compute_nodes(mesh_left, mesh_right)

        expected = 2.0 + c * 2.0
        assert torch.allclose(result[0, :, 0], expected)

    def test_multiple_steps_independent(self, method_name):
        """Each step is interpolated independently from others."""
        solver = self._make_solver(method_name)
        c = UNIFORM_METHODS[method_name].tableau.c.to(torch.float64)
        mesh_left = torch.tensor([[0.0], [1.0], [5.0]], dtype=torch.float64)
        mesh_right = torch.tensor([[1.0], [3.0], [10.0]], dtype=torch.float64)

        result = solver._compute_nodes(mesh_left, mesh_right)

        # Step 0: [0, 1], step 1: [1, 3], step 2: [5, 10]
        assert torch.allclose(result[0, :, 0], 0.0 + c * 1.0)
        assert torch.allclose(result[1, :, 0], 1.0 + c * 2.0)
        assert torch.allclose(result[2, :, 0], 5.0 + c * 5.0)

    def test_tiny_step(self, method_name):
        """Very small step width produces no NaN/Inf."""
        solver = self._make_solver(method_name)
        mesh_left = torch.tensor([[0.0]], dtype=torch.float64)
        mesh_right = torch.tensor([[1e-15]], dtype=torch.float64)

        result = solver._compute_nodes(mesh_left, mesh_right)

        assert torch.isfinite(result).all()
