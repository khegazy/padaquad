"""
Runge-Kutta integration solver implementations.

Provides concrete RK solver classes that implement the ``_calculate_integral()``
method from the parallel solver base classes. The core computation is the
RK weighted sum: integral = y0 + sum_over_steps(h * sum(b_i * f(t_i))).

Two solver variants:

- ``UniformAdaptiveQuadrature``: Uses fixed tableau b weights
  (same weights for every step). Faster since weights are precomputed.
- ``VariableAdaptiveQuadrature``: Computes tableau b weights
  dynamically from the actual quadrature point positions within each step.
  Necessary when mesh refinement places points at non-standard positions.

Both compute two estimates per step: the primary integral (using b weights)
and an error estimate (using b_error weights from the embedded lower-order
method). The error estimate drives the adaptive step refinement.
"""

from __future__ import annotations

import logging

import torch

from .base import get_sampling_type, steps
from .quadrature import (
    _UniformAdaptiveQuadratureBase,
    _VariableAdaptiveQuadratureBase,
)
from .results import MethodOutput

logger = logging.getLogger(__name__)


def _RK_integral(
    t: torch.Tensor,
    y: torch.Tensor,
    tableau_b: torch.Tensor,
    y0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Runge-Kutta weighted sum for a batch of integration steps.

    This is the core numerical operation: for each step, multiply the integrand
    evaluations by the tableau b weights, sum over the C quadrature points
    within the step, and multiply by the step size h. Then sum all step
    contributions to get the total integral.

    The formula per step i is:
        RK_step_i = h_i * sum_j(b_j * y_ij)

    And the total integral is:
        integral = y0 + sum_i(RK_step_i)

    Args:
        t: Time points for each step. The step size h is computed as
            t[:,-1] - t[:,0] (last minus first point in each step).
            Shape: [N, C, T].
        y: Integrand evaluations at the time points in t.
            Shape: [N, C, D].
        tableau_b: Weights for combining evaluations within each step.
            For uniform methods these are the same for all steps [1, C, 1].
            For variable methods these differ per step [N, C, 1].
        y0: Starting accumulator value. Shape: [D].

    Returns:
        Tuple of:
            - integral: Total integral value (y0 + sum of all step contributions).
              Shape: [D].
            - RK_steps: Per-step contributions to the integral. Each entry is
              h_i * sum(b_j * y_ij). Shape: [N, D].
            - h: Step sizes (t_last - t_first) for each step. Shape: [N, T].
    """
    # Step size: difference between last and first time point in each step
    h = t[:, -1] - t[:, 0]
    logger.debug("H shape=%s values=%s", h.shape, h)

    # Weighted sum of integrand evaluations within each step, scaled by step size.
    # tableau_b*y: weight each evaluation, dim=1 sums over C quadrature points
    RK_steps = h * torch.sum(tableau_b * y, dim=1)
    logger.debug("RK_steps shape=%s values=%s", RK_steps.shape, RK_steps)
    # Sum all step contributions to get the total integral
    integral = y0 + torch.sum(RK_steps, dim=0)
    return integral, RK_steps, h


class UniformAdaptiveQuadrature(_UniformAdaptiveQuadratureBase):
    """
    Parallel adaptive solver using Runge-Kutta with fixed (uniform) tableau weights.

    Implements ``_calculate_integral()`` using the RK weighted sum with constant
    b weights from the method's Butcher tableau. Since the b weights don't change
    between steps, they are looked up once and broadcast across all steps in the batch.

    The error estimate is computed by running the same RK sum with b_error weights
    (the difference between the primary and embedded lower-order methods).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the uniform RK solver."""
        super().__init__(*args, **kwargs)

    def _calculate_integral(
        self, t: torch.Tensor, y: torch.Tensor, y0: torch.Tensor
    ) -> MethodOutput:
        """
        Compute RK integral and error estimate for a batch of steps.

        Runs the RK weighted sum twice: once with the primary b weights to get
        the integral, and once with b_error weights to get the error estimate.
        Error estimates are detached from the computation graph since they are
        only used for step-size decisions, not for gradient computation.

        Args:
            t: Time points for each step with C quadrature points per step.
                Shape: [N, C, T].
            y: Integrand evaluations at the time points.
                Shape: [N, C, D].
            y0: Starting accumulator value. Shape: [D].

        Returns:
            MethodOutput containing the integral, error estimates, per-step
            contributions, per-step errors, and step sizes.
        """
        tableau_b, tableau_b_error = self._get_tableau_b(t)
        # Primary integral estimate using order-p weights
        integral, RK_steps, h = _RK_integral(t, y, tableau_b, y0=y0)
        # Error estimate using (order-p minus order-(p-1)) weights
        integral_error, step_errors, _ = _RK_integral(t, y, tableau_b_error, y0=y0)
        return MethodOutput(
            integral=integral,
            integral_error=integral_error.detach(),
            mesh_quadratures=RK_steps,
            mesh_quadrature_errors=step_errors.detach(),
            h=h,
        )

    def _get_tableau_b(self, _t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the fixed tableau b and b_error weights.

        For uniform methods, the weights are constant (independent of the
        actual time points). An extra dimension is added for broadcasting
        with the D (output) dimension.

        Args:
            t: Time points (unused for uniform methods, but accepted for
                interface compatibility). Shape: [N, C, T].

        Returns:
            Tuple of (b, b_error) with shape [C, 1] or [1, C, 1], ready
            for broadcasting with y of shape [N, C, D].
        """
        return self.method.tableau.b.unsqueeze(
            -1
        ), self.method.tableau.b_error.unsqueeze(-1)

    def _get_num_tableau_c(self) -> int:
        """
        Return C, the number of quadrature points per integration step.

        This equals the number of tableau c values (node positions) and
        determines how many integrand evaluations are needed per step.

        Returns:
            Number of quadrature points per step.
        """
        return len(self.method.tableau.c)


class VariableAdaptiveQuadrature(_VariableAdaptiveQuadratureBase):
    """
    Parallel adaptive solver using Runge-Kutta with dynamic (variable) tableau weights.

    Implements ``_calculate_integral()`` using the RK weighted sum where b weights
    are recomputed for each step based on the actual positions of the quadrature
    points. This is necessary when adaptive refinement places points at non-standard
    positions (not at the fixed fractions specified by a uniform tableau).

    The b weights are computed by normalizing the time points within each step
    to [0, 1] and passing them to the method's ``tableau_b()`` function.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the variable RK solver."""
        super().__init__(*args, **kwargs)

    def _calculate_integral(
        self, t: torch.Tensor, y: torch.Tensor, y0: torch.Tensor
    ) -> MethodOutput:
        """
        Compute RK integral and error estimate with dynamically computed weights.

        Same structure as the uniform version, but b weights are computed from
        the actual quadrature positions rather than being constant.

        Args:
            t: Time points for each step with C quadrature points per step.
                Shape: [N, C, T].
            y: Integrand evaluations at the time points.
                Shape: [N, C, D].
            y0: Starting accumulator value. Shape: [D].

        Returns:
            MethodOutput containing the integral, error estimates, per-step
            contributions, per-step errors, and step sizes.
        """
        tableau_b, tableau_b_error = self._get_tableau_b(t)
        # Primary integral estimate using dynamically computed order-p weights
        integral, RK_steps, h = _RK_integral(t, y, tableau_b, y0=y0)
        # Error estimate using dynamically computed error weights
        integral_error, step_errors, _ = _RK_integral(t, y, tableau_b_error, y0=y0)
        return MethodOutput(
            integral=integral,
            integral_error=integral_error,
            mesh_quadratures=RK_steps,
            mesh_quadrature_errors=step_errors,
            h=h,
        )

    def _get_tableau_b(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute tableau b weights from the actual quadrature point positions.

        Normalizes the time points within each step to the range [0, 1],
        then passes them to the method's ``tableau_b()`` function which
        computes weights as a function of the normalized positions.

        Args:
            t: Time points for each step. Shape: [N, C, T].

        Returns:
            Tuple of (b, b_error) with shape [N, C, 1], where the weights
            vary per step based on the actual point positions.
        """
        # Normalize time points to [0, 1] within each step:
        # shift so first point is 0, then divide by step size
        norm_dt = t - t[:, 0, None]
        norm_dt = norm_dt / norm_dt[:, -1, None]
        # Compute weights from normalized positions
        b, b_error = self.method.tableau_b(norm_dt)
        return b.unsqueeze(-1), b_error.unsqueeze(-1)

    def _get_num_tableau_c(self) -> int:
        """
        Return C, the number of quadrature points per integration step.

        Returns:
            Number of quadrature points per step.
        """
        return self.method.n_tableau_c


def adaptive_quadrature(
    sampling_type: str | steps, *args, **kwargs
) -> UniformAdaptiveQuadrature | VariableAdaptiveQuadrature:
    """
    Factory function to create the appropriate parallel RK solver.

    Selects between the uniform-sampling and variable-sampling solver
    based on the requested sampling type. All additional arguments are
    forwarded to the solver constructor.

    Args:
        sampling_type: Either a ``steps`` enum value or a string name
            ('uniform', 'adaptive_uniform', 'variable', 'adaptive_variable').
        *args: Positional arguments forwarded to the solver constructor.
        **kwargs: Keyword arguments forwarded to the solver constructor
            (e.g., method, atol, rtol, f, etc.).

    Returns:
        An initialized parallel RK solver instance.

    Raises:
        ValueError: If sampling_type is not a recognized uniform or variable type.
    """
    if isinstance(sampling_type, str):
        sampling_type = get_sampling_type(sampling_type)
    if sampling_type == steps.ADAPTIVE_UNIFORM:
        return UniformAdaptiveQuadrature(*args, **kwargs)
    elif sampling_type == steps.ADAPTIVE_VARIABLE:
        return VariableAdaptiveQuadrature(*args, **kwargs)
    else:
        raise ValueError
