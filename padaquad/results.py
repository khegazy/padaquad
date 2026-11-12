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


@dataclass(kw_only=True)
class IntegrationResult:
    """
    Complete output of a numerical integration run.

    Returned by all solver ``integrate()`` methods. Contains the computed
    integral value along with diagnostics about step sizes, errors, and
    the optimized mesh for potential warm-start reuse.

    Construct it with keyword arguments only (``kw_only=True``). Every field is
    required except ``mesh_optimal``, ``tracked_variables``, ``loss``,
    ``converged``, ``n_start_steps``, ``n_start_steps_split`` and
    ``max_split_depth``, which may be ``None`` as described below.
    ``gradient_taken`` and ``converged`` are ``bool``, the three split counts
    are ``int``, ``tracked_variables`` is a tuple of tensors, and every other
    field is a tensor.

    ``integrate()`` also builds one result per batch and passes it to
    ``loss_fxn``. That result covers only the batch's accepted steps:
    ``integral`` is their contribution without ``y0``, ``y0`` is zero,
    ``mesh_optimal``, ``loss``, ``converged`` and the three split counts are
    ``None``, and ``nodes``, ``y`` and ``tracked_variables`` keep one row per
    step (shapes [N, C, T], [N, C, D] and [N, C, *var_dims]) instead of the
    flattened shapes below.

    Attributes
    ----------
    integral : torch.Tensor
        The computed integral value. Shape: [D].
    integral_error : torch.Tensor
        Estimated total error (sum of per-step errors)
        using the embedded lower-order method. Shape: [D].
    mesh_optimal : torch.Tensor | None
        Optimized step-barrier mesh for warm-start reuse
        in subsequent integration calls with the same integrand (pass
        ``same_f=True``).
        Shape: [M, T]. ``None`` in the per-batch result passed to
        ``loss_fxn`` during integration (no optimal mesh exists yet);
        always a tensor on a result returned by ``integrate()``.
    mesh_init : torch.Tensor
        Lower integration bound used. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound used. Shape: [T].
    nodes : torch.Tensor
        Quadrature node positions flattened across panels into a
        single ascending sequence. Consecutive panels share one boundary
        point, which is deduplicated, so for N panels of C nodes each the
        length is ``C + (N - 1) * (C - 1)``. Shape: [P, T].
    h : torch.Tensor
        Step sizes (right - left) for each integration step.
        Shape: [N, T].
    y : torch.Tensor
        Integrand evaluations at each node, flattened to match ``nodes``.
        Shape: [P, D].
    tracked_variables : tuple[torch.Tensor, ...] | None
        Optional extra quantities emitted by the
        integrand alongside the integrand value. These are evaluated at
        every node but NOT integrated; they are returned at the accepted
        nodes. A tuple of detached tensors, each shaped [P, *var_dims]
        and flattened to align with ``nodes``/``y``. ``None`` when the
        integrand returns only the integrand value (no tracked variables).
    mesh_quadratures : torch.Tensor
        Weighted contribution of each step to the total
        integral (h * sum(b_i * y_i) per step). Shape: [N, D].
    mesh_quadrature_errors : torch.Tensor
        Per-step error estimates from the difference
        between the primary and embedded methods. Shape: [N, D].
    error_ratios : torch.Tensor
        Per-step error ratios (error / tolerance) at the time
        each step was accepted, judged against the running integral of
        that moment. Values >= 1 mean the step did not meet the tolerance
        then. ``converged`` judges the steps again against the final
        integral, so it can be ``False`` while every entry here is < 1.
        Shape: [N].
    loss : torch.Tensor | None
        Loss value computed by the loss function (defaults to
        the integral itself). Shape: scalar or [D]. ``None`` only in the
        per-batch result before ``loss_fxn`` is applied.
    gradient_taken : bool
        Whether per-batch backpropagation was performed
        during this integration run.
    y0 : torch.Tensor
        Initial integral accumulator value used. Shape: [D].
    converged : bool | None
        Whether every accepted step meets the error tolerance when
        judged against the final integral (in cumulative mode, the final
        running integral through the step; the reference ``mesh_optimal``
        is built with). Usually ``False`` when ``max_adaptive_splits``
        stopped the refinement of a failing step; ``False`` when a step's
        error ratio is NaN/Inf, or when a step accepted against
        the running integral fails against the final one (with
        ``take_gradient=True`` accepted steps are never re-checked). A
        warning with the size of the error is logged when ``False``.
        ``None`` in the per-batch result.
    n_start_steps : int | None
        Number of steps in the starting mesh (the caller's
        ``mesh``, the mesh stored by ``same_f=True``, or a freshly
        generated one). ``None`` in the per-batch result.
    n_start_steps_split : int | None
        Number of starting-mesh steps that were split at
        least once during integration, including splits made by the
        re-check of accepted steps. Splits made afterwards to build
        ``mesh_optimal`` are not counted, so with
        ``max_adaptive_splits=0`` this is 0 even when steps failed; read
        ``converged`` then. ``None`` in the per-batch result.
    max_split_depth : int | None
        Largest number of times any one step was halved
        (0 when no step was split; at most ``max_adaptive_splits``).
        ``None`` in the per-batch result.
    """

    integral: torch.Tensor
    integral_error: torch.Tensor
    mesh_init: torch.Tensor
    mesh_final: torch.Tensor
    nodes: torch.Tensor
    h: torch.Tensor
    y: torch.Tensor
    mesh_quadratures: torch.Tensor
    mesh_quadrature_errors: torch.Tensor
    error_ratios: torch.Tensor
    gradient_taken: bool
    y0: torch.Tensor
    mesh_optimal: torch.Tensor | None = None
    tracked_variables: tuple[torch.Tensor, ...] | None = None
    loss: torch.Tensor | None = None
    converged: bool | None = None
    n_start_steps: int | None = None
    n_start_steps_split: int | None = None
    max_split_depth: int | None = None

    @property
    def percent_start_steps_split(self) -> float | None:
        """
        Percentage of starting-mesh steps that were split at least once.

        ``100 * n_start_steps_split / n_start_steps``; ``None`` in the
        per-batch result, where the counts are ``None``.
        """
        if self.n_start_steps is None or self.n_start_steps_split is None:
            return None
        return 100.0 * self.n_start_steps_split / self.n_start_steps


@dataclass
class MethodOutput:
    """
    Output from a single batch of RK integration steps.

    Produced by ``AdaptiveQuadrature._calculate_integral()``.
    Contains both the primary integral estimate and the embedded error
    estimate, broken down per step. Field names mirror
    ``IntegrationResult`` for consistency between the internal
    batch-level output and the user-facing run-level output.

    Attributes
    ----------
    integral : torch.Tensor
        Total integral contribution from this batch of steps
        (sum of mesh_quadratures). Shape: [D].
    integral_error : torch.Tensor
        Total error estimate from this batch (sum of
        mesh_quadrature_errors). Shape: [D].
    mesh_quadratures : torch.Tensor
        Per-step integral contributions. Each entry is
        h_i * sum(b_j * y_ij) for step i. Shape: [N, D].
    mesh_quadrature_errors : torch.Tensor
        Per-step error estimates from the difference between
        order-p and order-(p-1) methods. Shape: [N, D].
    h : torch.Tensor
        Step sizes (mesh_right - mesh_left) for each step. Shape: [N, T].
    """

    integral: torch.Tensor
    integral_error: torch.Tensor
    mesh_quadratures: torch.Tensor
    mesh_quadrature_errors: torch.Tensor
    h: torch.Tensor
