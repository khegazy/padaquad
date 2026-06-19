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


# ---------------------------------------------------------------------------
# Cached max_batch — run the slow memory benchmark at most once
# ---------------------------------------------------------------------------
#
# Building a solver with ``max_batch=None`` makes the first ``.integrate()``
# call benchmark the integrand's per-evaluation memory footprint
# (``_setup_memory_checks``), which is slow. The solver only skips that
# benchmark when the *same* integrand is reused (``id(f)`` match) or an explicit
# ``max_batch`` is supplied. The test suite reuses neither — every
# (method × integrand × ...) cell builds a fresh solver on a fresh integrand —
# so the benchmark runs hundreds of times even though the answer is always the
# same: all the analytic integrands share one per-evaluation footprint.
#
# We therefore benchmark a single representative integrand exactly once (per
# device) and pass the resulting ``max_batch`` to every solver. This is
# behavior-preserving:
#   * ``take_gradient=False`` evaluates every pending panel in one logical pass,
#     so ``max_batch`` only controls memory chunking and never the result (see
#     test_evaluate_f_on_mesh::test_integrate_result_invariant_across_max_batch).
#   * ``take_gradient=True`` reproduces the unbatched result as long as the
#     budget stays above the (tiny) total node count of these integrals, which
#     the benchmarked value does by a wide margin — exactly as the per-test
#     benchmark already did.

_MAX_BATCH_CACHE: dict = {}


def cached_max_batch(device="cpu"):
    """Return a ``max_batch`` for the analytic integrands, benchmarking once.

    The memory footprint of every analytic integrand is the same, so the slow
    ``_setup_memory_checks`` benchmark only needs to run for the first solver on
    a given device; the resulting budget is cached and reused by every later
    solver, skipping the benchmark entirely. Solvers that want their own value
    can still pass an explicit ``max_batch`` (it takes precedence — see
    ``make_uniform_solver``).
    """
    key = str(device)
    if key not in _MAX_BATCH_CACHE:
        f = integrand_dict["damped_sine"][0]
        probe = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_UNIFORM,
            method="bosh3",
            atol=ATOL_LOOSE,
            rtol=RTOL_LOOSE,
            remove_cut=REMOVE_CUT,
            device=device,
        )
        # One real run benchmarks the footprint (take_gradient=False); the
        # resulting budget is large enough to keep every test integral
        # single-batch in both modes (see note above).
        probe.integrate(f, mesh_init=T_INIT, mesh_final=T_FINAL)
        _MAX_BATCH_CACHE[key] = probe.get_max_batch()
    return _MAX_BATCH_CACHE[key]

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


def make_uniform_solver(
    method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT, device="cpu", **kwargs
):
    """Create a parallel uniform-sampling RK solver.

    Pinned to CPU by default: the suite is CPU-designed, and on a GPU machine
    the solver otherwise auto-selects CUDA, where the per-eval memory budget can
    fail the pre-check when the (often shared) GPU is short on free memory.
    Pass ``device=`` to override.

    ``max_batch`` defaults to the cached benchmark value (``cached_max_batch``)
    so the slow per-solver memory benchmark runs at most once; pass an explicit
    ``max_batch`` to override.
    """
    if "max_batch" not in kwargs:
        kwargs["max_batch"] = cached_max_batch(device)
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
        device=device,
        **kwargs,
    )


def make_variable_solver(
    method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT, device="cpu", **kwargs
):
    """Create a parallel variable-sampling solver.

    Pinned to CPU by default for the same reason as ``make_uniform_solver``, and
    likewise defaults ``max_batch`` to the cached benchmark value so the slow
    memory benchmark runs at most once.
    """
    if "max_batch" not in kwargs:
        kwargs["max_batch"] = cached_max_batch(device)
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_VARIABLE,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
        device=device,
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

    ``nodes`` is the flattened ascending node sequence ([P, T]), so a small
    tolerance still guards against ULP-level float rounding accumulated along
    the sequence. The tolerance scales with the integration domain.
    """
    t_flat = integral_output.nodes  # already flattened across panels: [P, T]
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
    """Assert consecutive panels join continuously in the flattened output.

    Flattening drops the duplicated end-of-panel-i / start-of-panel-(i+1)
    boundary, so the panels form one continuous traversal. We verify the node
    sequence runs in order (non-decreasing) and spans the full integration
    domain end to end (mesh_init .. mesh_final), so no leading/trailing panel is
    missing and panels join without a backward jump. Equal adjacent nodes are
    allowed: methods like dopri5 repeat a node within a panel (c = 1 twice).
    """
    nodes = integral_output.nodes  # flattened across panels: [P, T]
    eps = torch.finfo(nodes.dtype).eps
    tol = 8 * eps * float(nodes.abs().max().clamp_min(1.0))
    assert torch.all(nodes[1:] - nodes[:-1] >= -tol), (
        "Flattened nodes are not ordered across panel boundaries"
    )
    # mesh_init/mesh_final stay on the integration device while nodes may be
    # offloaded to result_device, so compare on the nodes' device.
    mesh_init = integral_output.mesh_init.to(nodes.device)
    mesh_final = integral_output.mesh_final.to(nodes.device)
    assert torch.allclose(nodes[0], mesh_init, atol=tol, rtol=0), (
        "Flattened nodes do not start at mesh_init"
    )
    assert torch.allclose(nodes[-1], mesh_final, atol=tol, rtol=0), (
        "Flattened nodes do not end at mesh_final"
    )


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
