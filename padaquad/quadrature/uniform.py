"""
Uniform-sampling adaptive quadrature: ``UniformAdaptiveQuadrature``.

In uniform sampling, quadrature points within each step are placed at
fixed fractional positions defined by the method's tableau.c values
(e.g. dopri5 always uses ``c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]``). The
b weights are also constant.

When the adaptive controller splits a step, the new sub-steps get
fresh evaluations at the standard c positions; the old evaluations
are discarded. When merging two steps, a new combined step is built
spanning ``[A.start, B.end]`` and re-evaluated at the standard c
positions.

This class holds the uniform-specific overrides of the abstract methods
declared in ``AdaptiveQuadrature``; everything else, including the
integral computation, is shared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange

from padaquad.base import steps

from .base import AdaptiveQuadrature

if TYPE_CHECKING:
    from collections.abc import Callable


class UniformAdaptiveQuadrature(AdaptiveQuadrature):
    """
    Parallel adaptive solver for rules with fixed node fractions and weights.

    In uniform sampling, quadrature points within each step are placed at
    fixed fractional positions defined by the tableau's c values. For example,
    dopri5 always evaluates at c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1] of each
    step's width. The b weights are also fixed constants.

    This is the simpler and more common solver variant. When a step is split,
    the new sub-steps get fresh evaluations at the standard c positions.

    Supported methods: every name in ``UNIFORM_METHODS`` (Runge-Kutta,
    Gauss-Kronrod and Clenshaw-Curtis rules).
    """

    sampling_type = steps.ADAPTIVE_UNIFORM

    def _evaluate_adaptive_nodes(
        self,
        f: Callable[..., Any],
        idxs_add: torch.Tensor,
        _y: torch.Tensor,
        nodes: torch.Tensor,
        f_args: tuple[Any, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split failed steps at their midpoints and evaluate the integrand.

        Each step that failed the error check is divided into two sub-steps
        at its midpoint. Fresh quadrature points are placed in each sub-step
        using the tableau's c values, and the integrand is evaluated at all
        new points.

        Args:
            f: The integrand function f(t). Takes [N, T], returns [N, D].
            idxs_add: Indices of steps that need splitting. Shape: [R].
            _y: Current integrand evaluations (unused here, present for
                interface compatibility with the variable solver). Shape: [N, C, D].
            nodes: Current quadrature point positions. Shape: [N, C, T].
            f_args: Extra arguments passed to f.

        Returns:
            Tuple of (y_add, nodes_new) where:
                - y_add: Integrand values at new quadrature points.
                    Shape: [2*R, C, D] (two sub-steps per split step).
                - nodes_new: New quadrature point positions.
                    Shape: [2*R, C, T].
        """
        T = nodes.shape[-1]
        # Compute the midpoint of each failed step
        mesh_mid = (nodes[idxs_add, -1] + nodes[idxs_add, 0]) / 2.0
        # Build left and right barriers for the two new sub-steps
        mesh_left = torch.concatenate(
            [nodes[idxs_add, None, 0], mesh_mid[:, None]], dim=1
        )
        mesh_right = torch.concatenate(
            [mesh_mid[:, None], nodes[idxs_add, None, -1]], dim=1
        )
        # Place quadrature points in each sub-step and evaluate
        nodes_new = self._compute_nodes(mesh_left.view(-1, T), mesh_right.view(-1, T))
        y_add = f(nodes_new.view(-1, T), *f_args)
        y_add = rearrange(y_add, "(N C) D -> N C D", C=self.C)
        return y_add, nodes_new

    def _merge_excess_nodes(
        self,
        nodes: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        mesh_quadrature_errors: torch.Tensor,
        remove_idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge pairs of consecutive over-resolved steps into single larger steps.

        For uniform sampling, merging replaces two consecutive steps [A, B]
        with a single step spanning [A.start, B.end]. New quadrature points
        are placed at the standard tableau c positions within the merged step.
        The integral contributions and errors of both steps are summed.

        Args:
            nodes: Quadrature point positions for all steps. Shape: [N, C, T].
            mesh_quadratures: Integral contribution of each step. Shape: [N, D].
            mesh_quadrature_errors: Error estimate of each step. Shape: [N, D].
            remove_idxs: Indices of the first step in each pair to merge.
                The pair (remove_idxs[i], remove_idxs[i]+1) is merged.
                Shape: [R].

        Returns:
            Tuple of (nodes_pruned, mesh_quadratures_pruned, mesh_quadrature_errors_pruned)
            with the merged steps replacing the original pairs. Each has
            N-R entries along the first dimension.
        """
        if len(remove_idxs) == 0 or len(nodes) == 1:
            return nodes, mesh_quadratures, mesh_quadrature_errors

        # Create merged steps spanning from the left edge of the first step
        # to the right edge of the second step, with new c-based quadrature points
        nodes_replace = self._compute_nodes(
            nodes[remove_idxs, 0], nodes[remove_idxs + 1, -1]
        )

        # Sum integral contributions and errors of the merged pair
        mesh_quadratures_replace = (
            mesh_quadratures[remove_idxs] + mesh_quadratures[remove_idxs + 1]
        )
        mesh_quadrature_errors_replace = (
            mesh_quadrature_errors[remove_idxs]
            + mesh_quadrature_errors[remove_idxs + 1]
        )

        # Remove the first step of each pair from the arrays
        remove_mask = torch.ones(len(nodes), device=self.device, dtype=torch.bool)
        remove_mask[remove_idxs] = False
        nodes_pruned = nodes[remove_mask]
        mesh_quadratures_pruned = mesh_quadratures[remove_mask]
        mesh_quadrature_errors_pruned = mesh_quadrature_errors[remove_mask]

        # Place the merged step data at the position of the second (kept) step,
        # adjusting indices to account for earlier removals shifting positions
        remove_idxs_shifted = remove_idxs - torch.arange(
            len(remove_idxs), device=self.device
        )
        nodes_pruned[remove_idxs_shifted] = nodes_replace
        mesh_quadratures_pruned[remove_idxs_shifted] = mesh_quadratures_replace
        mesh_quadrature_errors_pruned[remove_idxs_shifted] = (
            mesh_quadrature_errors_replace
        )

        # Verify time ordering is preserved after merging
        nodes_pruned_flat = torch.flatten(nodes_pruned, start_dim=0, end_dim=1)
        assert torch.all(
            nodes_pruned_flat[1:] - nodes_pruned_flat[:-1] + self.atol_assert >= 0
        )
        nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=1)
        assert torch.all(nodes_flat[1:] - nodes_flat[:-1] + self.atol_assert >= 0)

        return nodes_pruned, mesh_quadratures_pruned, mesh_quadrature_errors_pruned
