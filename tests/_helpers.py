"""Shared constants and helpers for the test suite."""

from __future__ import annotations

import torch
from torch import nn

from padaquad import (
    UNIFORM_METHODS,
    VARIABLE_METHODS,
    adaptive_quadrature,
    integrand_dict,
    steps,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 2025

# Tight tolerances for accuracy tests
ATOL_TIGHT = 1e-12
RTOL_TIGHT = 1e-10

# Loose tolerances for adaptivity / dtype tests
ATOL_LOOSE = 1e-5
RTOL_LOOSE = 1e-5

# Medium tolerances
ATOL_MED = 1e-9
RTOL_MED = 1e-7

T_INIT = torch.tensor([0], dtype=torch.float64)
T_FINAL = torch.tensor([1], dtype=torch.float64)

REMOVE_CUT = 0.1

# ---------------------------------------------------------------------------
# Parametrize helpers — usable as @pytest.mark.parametrize values
# ---------------------------------------------------------------------------

UNIFORM_METHOD_NAMES = list(UNIFORM_METHODS.keys())
VARIABLE_METHOD_NAMES = list(VARIABLE_METHODS.keys())
INTEGRAND_NAMES = list(integrand_dict.keys())

# ``take_gradient`` is a parameter of ``.integrate()`` (not the solver
# constructor). The two modes share numerics in the current implementation
# but diverge in memory/backward behavior. Tests that exercise the
# integration loop are parametrized over both so the upcoming
# ``take_gradient`` code-path split can be validated independently.
#
# The parametrize IDs ``take_grad_True`` / ``take_grad_False`` are chosen
# to be unique substrings that do NOT appear anywhere else in the test
# suite. This lets ``pytest -k "take_grad_True"`` cleanly select only
# the gradient-bearing parametrized tests without sweeping in the
# dedicated gradient test files (test_gradient.py,
# test_gradient_integration.py, test_autodiff_consistency.py), whose
# filenames contain ``grad`` and would otherwise be matched.
TAKE_GRADIENT_VALUES = [True, False]
TAKE_GRADIENT_IDS = ["take_grad_True", "take_grad_False"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_uniform_solver(method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT, **kwargs):
    """Create a parallel uniform-sampling RK solver."""
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
        **kwargs,
    )


def make_solver_for_unit_test(
    method_name="bosh3", atol=1e-6, rtol=1e-6, dtype=torch.float64, **kwargs
):
    """Create a minimal solver for testing internal methods (no f needed).

    Pinned to CPU: these unit tests call internal methods directly with
    hand-built CPU tensors, so the solver device must match (otherwise the
    solver auto-selects CUDA on a GPU machine and the internal device-side
    allocations clash with the CPU inputs).

    ``dtype`` and any extra ``kwargs`` (e.g. ``error_norm``,
    ``mesh_failure_tolerance``, ``use_absolute_error_ratio``) are forwarded to
    the solver constructor so internal-method tests can pin them directly.
    """
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
        device="cpu",
        dtype=dtype,
        **kwargs,
    )


def make_variable_solver_for_unit_test(
    method_name="adaptive_heun", atol=1e-6, rtol=1e-6
):
    """Create a minimal variable solver for testing internal methods.

    Pinned to CPU for the same reason as ``make_solver_for_unit_test``.
    """
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_VARIABLE,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
        device="cpu",
    )


def constant_integrand(t, *args):
    """f(t) = 1 for all t. Returns shape [N, 1]."""
    if len(t.shape) == 1:
        return torch.ones(1, dtype=t.dtype, device=t.device)
    return torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)


def assert_time_ordering(integral_output):
    """Assert that all time points in the output are non-decreasing.

    Allows ULP-level float rounding at panel boundaries: the last
    node of panel i (at c=1.0, evaluating to barrier_i + h_i) and the
    first node of panel i+1 (at c=0.0, evaluating to barrier_{i+1})
    are mathematically equal but can differ by 1 ULP when the
    subtraction h_i = barrier_{i+1} - barrier_i isn't exactly
    representable. The tolerance scales with the integration domain.
    """
    t_flat = torch.flatten(integral_output.nodes, start_dim=0, end_dim=1)
    eps = torch.finfo(t_flat.dtype).eps
    scale = t_flat.abs().max().clamp_min(1.0)
    tol = 8 * eps * scale  # generous bound for accumulated rounding
    assert torch.all(t_flat[1:] - t_flat[:-1] >= -tol), (
        "Time points are not non-decreasing"
    )


def assert_optimal_mesh_ordering(integral_output):
    """Assert that the optimal mesh time points are non-decreasing."""
    t_optimal_flat = torch.flatten(integral_output.mesh_optimal, start_dim=0, end_dim=1)
    assert torch.all(t_optimal_flat[1:] - t_optimal_flat[:-1] >= 0), (
        "Optimal mesh time points are not non-decreasing"
    )


def assert_step_continuity(integral_output):
    """Assert that consecutive steps share boundary points (end of step i == start of step i+1)."""
    assert torch.allclose(
        integral_output.nodes[1:, 0, :], integral_output.nodes[:-1, -1, :]
    ), "Consecutive steps do not share boundary points"


# ---------------------------------------------------------------------------
# Parameterized nn.Module integrands for gradient tests
# ---------------------------------------------------------------------------


class ScaledIntegrand(nn.Module):
    """f(t) = scale * t^2, with learnable scale. Returns [N, 1]."""

    __name__ = "ScaledIntegrand"

    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale], dtype=torch.float64))

    def forward(self, t, *args):
        """Evaluate scale * t^2, returning shape [N, 1]."""
        while len(t.shape) < 2:
            t = t.unsqueeze(0)
        return self.scale * t**2
