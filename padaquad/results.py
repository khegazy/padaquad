"""
Result dataclasses returned by the integration pipeline.

``IntegrationResult`` is the user-facing return value of
``integrate()`` and ``AdaptiveQuadrature.integrate()``. It carries the
computed integral, the estimated error, the optimized mesh (for
warm-start reuse), and per-step diagnostics.

``MethodOutput`` is an internal dataclass produced by a concrete
solver's ``_calculate_integral()`` method. It carries the same kind of
data but for a single batch of accepted steps rather than for the
whole integration run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class IntegrationResult:
    """
    Complete output of a numerical integration run.

    Returned by all solver ``integrate()`` methods. Contains the computed
    integral value along with diagnostics about step sizes, errors, and
    the optimized mesh for potential warm-start reuse.

    Attributes:
        integral: The computed integral value. Shape: [D].
        integral_error: Estimated total error (sum of per-step errors)
            using the embedded lower-order method. Shape: [D].
        mesh_optimal: Optimized step-barrier mesh for warm-start reuse
            in subsequent integration calls with the same integrand.
            Shape: [M, T].
        mesh_init: Lower integration bound used. Shape: [T].
        mesh_final: Upper integration bound used. Shape: [T].
        nodes: Per-step quadrature node positions, organized by
            integration step. Shape: [N, C, T].
        h: Step sizes (right - left) for each integration step.
            Shape: [N, T].
        y: Integrand evaluations at each node. Shape: [N, C, D].
        tracked_variables: Optional extra quantities emitted by the
            integrand alongside the integrand value. These are evaluated at
            every node but NOT integrated; they are returned at the accepted
            nodes. A tuple of detached tensors, each shaped [N, C, *var_dims]
            and aligned with ``nodes``/``y``. ``None`` when the integrand
            returns only the integrand value (no tracked variables).
        mesh_quadratures: Weighted contribution of each step to the total
            integral (h * sum(b_i * y_i) per step). Shape: [N, D].
        mesh_quadrature_errors: Per-step error estimates from the difference
            between the primary and embedded methods. Shape: [N, D].
        error_ratios: Per-step error ratios (error / tolerance). Values
            > 1 indicate the step did not meet accuracy requirements.
            Shape: [N].
        loss: Loss value computed by the loss function (defaults to
            the integral itself). Shape: scalar or [D].
        gradient_taken: Whether per-batch backpropagation was performed
            during this integration run.
        y0: Initial integral accumulator value used. Shape: [D].
        converged: Whether the adaptive refinement met the tolerance
            criterion. ``True`` for normal completion, ``False`` if the
            integrator hit ``max_path_change`` and exited with a
            partially-refined mesh.
    """

    integral: torch.Tensor
    integral_error: torch.Tensor = None
    mesh_optimal: torch.Tensor = None
    mesh_init: torch.Tensor = None
    mesh_final: torch.Tensor = None
    nodes: torch.Tensor = None
    h: torch.Tensor = None
    y: torch.Tensor = None
    tracked_variables: tuple[torch.Tensor, ...] | None = None
    mesh_quadratures: torch.Tensor = None
    mesh_quadrature_errors: torch.Tensor = None
    error_ratios: torch.Tensor = None
    loss: torch.Tensor = None
    gradient_taken: bool = None
    y0: torch.Tensor = None
    converged: bool = True


@dataclass
class MethodOutput:
    """
    Output from a single batch of RK integration steps.

    Produced by ``_calculate_integral()`` in concrete solver subclasses.
    Contains both the primary integral estimate and the embedded error
    estimate, broken down per step. Field names mirror
    ``IntegrationResult`` for consistency between the internal
    batch-level output and the user-facing run-level output.

    Attributes:
        integral: Total integral contribution from this batch of steps
            (sum of mesh_quadratures). Shape: [D].
        integral_error: Total error estimate from this batch (sum of
            mesh_quadrature_errors). Shape: [D].
        mesh_quadratures: Per-step integral contributions. Each entry is
            h_i * sum(b_j * y_ij) for step i. Shape: [N, D].
        mesh_quadrature_errors: Per-step error estimates from the difference between
            order-p and order-(p-1) methods. Shape: [N, D].
        h: Step sizes (mesh_right - mesh_left) for each step. Shape: [N, T].
    """

    integral: torch.Tensor
    integral_error: torch.Tensor
    mesh_quadratures: torch.Tensor
    mesh_quadrature_errors: torch.Tensor
    h: torch.Tensor
