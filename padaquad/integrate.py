"""
High-level public API for adaptive numerical quadrature.

Provides ``integrate()``, the main entry point for users who want
to compute a definite integral without manually constructing solver objects.
This function creates the appropriate solver, runs the integration, and
returns the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import steps
from .runge_kutta import adaptive_quadrature

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from .results import IntegrationResult


def integrate(
    f: Callable,
    method: str = "gk21",
    sampling: str = "uniform",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mesh: torch.Tensor | None = None,
    mesh_init: torch.Tensor | None = None,
    mesh_final: torch.Tensor | None = None,
    y0: torch.Tensor | None = None,
    remove_cut: float = 0.1,
    total_mem_usage: float = 0.9,
    use_absolute_error_ratio: bool = True,
    take_gradient: bool = True,
    max_adaptive_splits: int | None = None,
    device: str | None = None,
    **kwargs,
) -> IntegrationResult:
    """
    Compute the definite integral of f from mesh_init to mesh_final.

    This is the main public API. It creates a parallel adaptive quadrature
    solver, runs the integration in batched evaluations on GPU/CPU, and
    returns the result.

    The integrand f depends only on time t (not on accumulated state
    y). This independence is what enables batched parallel evaluation.

    Example::

        result = integrate(
            f=lambda t: torch.sin(t),
            method='gk21',
            atol=1e-8,
            rtol=1e-6,
            mesh_init=torch.tensor([0.0]),
            mesh_final=torch.tensor([3.14159]),
        )
        print(result.integral)  # Should be close to 2.0

    Args:
        f: The integrand function. Takes time points of shape [N, T]
            and returns evaluations of shape [N, D]. Optionally, f may
            instead return a 2-tuple ``(integrand, tracked_variables)``
            where ``tracked_variables`` is a tuple of tensors (or None).
            Tracked variables are evaluated at every node but NOT
            integrated; they are returned at the accepted nodes in
            ``result.tracked_variables``.
        method: Name of the quadrature method. Available:
            uniform sampling: 'adaptive_heun', 'fehlberg2', 'bosh3',
              'dopri5', 'gk15', 'gk21', 'gk31', 'cc17', 'cc33', 'cc65';
            variable sampling: 'adaptive_heun', 'interpolatory3_variable'.
        sampling: 'uniform' uses fixed quadrature point positions within
            each step; 'variable' computes weights dynamically based on
            actual point positions.
        atol: Absolute error tolerance. A step is accepted when its
            estimated error < atol + rtol * |integral_value|.
        rtol: Relative error tolerance. See atol.
        t: Optional initial time mesh (step barrier positions). If None,
            the solver generates an initial mesh automatically. Shape:
            [N, T] for step barriers.
        mesh_init: Lower integration bound. Shape: [T].
        mesh_final: Upper integration bound. Shape: [T].
        y0: Initial value of the integral accumulator. Shape: [D].
        remove_cut: Threshold for merging consecutive steps whose
            combined error ratio is below this value. Must be < 1. Lower
            values are more conservative (keep more steps).
        total_mem_usage: Fraction of total GPU/CPU memory the solver is
            allowed to use (0 < value <= 1). The solver dynamically
            adjusts batch size to stay within this budget.
        use_absolute_error_ratio: If True (default), error ratios use the
            total integral value as the reference. If False, uses the
            cumulative integral up to each step (more like traditional
            ODE error control).
        max_adaptive_splits: Maximum number of times a panel may be split
            during adaptive refinement. A panel split this many times is
            accepted even if it still fails the error tolerance, rather than
            being split further. If None (default), refinement is uncapped.
        device: Device to run on (e.g. 'cuda', 'cpu'). If None,
            auto-detects.
        **kwargs: Additional keyword arguments forwarded to the solver
            constructor (e.g. max_batch, max_path_change, error_calc_idx).

    Returns:
        IntegrationResult containing the computed integral, error estimates,
        time mesh, and optimization diagnostics.

    Raises:
        ValueError: If sampling is not 'uniform' or 'variable'.

    Note:
        If t is None, the solver creates an initial mesh of
        ~sqrt(N_init_steps) barriers with random sub-divisions in
        [mesh_init, mesh_final]. Steps are adaptively added (split) or removed
        (merged) based on error estimates until all steps meet tolerance.
    """
    # Select the sampling strategy for the parallel solver
    if sampling.lower() == "uniform":
        sampling_type = steps.ADAPTIVE_UNIFORM
    elif sampling.lower() == "variable":
        sampling_type = steps.ADAPTIVE_VARIABLE
    else:
        raise ValueError(
            f"Sampling method must be either 'uniform' or 'variable', "
            f"instead got {sampling}"
        )

    # Create the parallel RK solver and run integration
    integrator = adaptive_quadrature(
        sampling_type=sampling_type,
        method=method,
        f=f,
        atol=atol,
        rtol=rtol,
        remove_cut=remove_cut,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        use_absolute_error_ratio=use_absolute_error_ratio,
        device=device,
        **kwargs,
    )

    return integrator.integrate(
        y0=y0,
        mesh=mesh,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        total_mem_usage=total_mem_usage,
        take_gradient=take_gradient,
        max_adaptive_splits=max_adaptive_splits,
    )
