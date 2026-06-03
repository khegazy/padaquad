"""
Variable-sampling adaptive quadrature: ``_VariableAdaptiveQuadratureBase``.

In variable sampling, quadrature points within each step can sit at
arbitrary positions; the b weights are recomputed dynamically from the
actual node positions via the method's ``tableau_b(c)`` callable.

This flexibility is what enables the variable solver's main
optimization: when an existing step is split into two sub-steps, the
old evaluation points end up at non-standard fractional positions
within the new sub-steps. The variable formula re-weights them
correctly so the old f-evaluations are reused, avoiding the redundant
recomputation that the uniform solver pays.

This intermediate base class holds the variable-specific overrides of
the abstract methods declared in ``AdaptiveQuadrature``. Concrete
solvers subclass this with a specific RK family for the
``_calculate_integral`` implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange

from padaquad.methods import VARIABLE_METHODS

from .base import AdaptiveQuadrature

if TYPE_CHECKING:
    from collections.abc import Callable


class _VariableAdaptiveQuadratureBase(AdaptiveQuadrature):
    """
    Parallel solver using variable-sampling Runge-Kutta methods.

    In variable sampling, quadrature points within each step can be at
    arbitrary positions (not fixed by the tableau). The b weights are
    computed dynamically based on the actual positions of the points via
    the method's ``tableau_b(c)`` function.

    This flexibility is especially useful during adaptive refinement: when
    a step is split, the existing evaluation points from the original step
    can be reused in the sub-steps (they just end up at different fractional
    positions within the new, smaller steps). This avoids redundant function
    evaluations.

    When merging two steps, the combined points are subsampled at evenly
    spaced indices to fit C points, and the b weights are recomputed for
    their new fractional positions.

    Supported methods: 'adaptive_heun', 'interpolatory3_variable'.

    Attributes:
        method: The variable method instance with a ``tableau_b(c)`` method.
        order: Convergence order of the RK method.
        C: Number of quadrature points per step.
        Cm1: C - 1, used for indexing.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the variable solver and set up its RK method."""
        super().__init__(*args, **kwargs)
        assert self.method_name in VARIABLE_METHODS, (
            f"Cannot find method '{self.method_name}' in supported methods: {list(VARIABLE_METHODS.keys())}"
        )
        self.method = VARIABLE_METHODS[self.method_name]()
        self.method.to_dtype(self.dtype)
        # Move the method's tensors onto the solver device. The uniform solver
        # does this via _get_method(..., self.device, ...); the variable solver
        # built its method without a device, so its tensors (e.g. b_delta) would
        # otherwise stay on CPU and clash with device-side nodes during
        # tableau_b(c). DistributedEnvironment owns the device; the method is
        # moved onto it.
        self.method.to_device(self.device)
        self.order = self.method.order
        self.C = self.method.n_tableau_c
        self.Cm1 = self.C - 1

    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """Convert variable method tensors to the new dtype."""
        if hasattr(self, "method"):
            self.method.to_dtype(dtype)

    def _compute_nodes(
        self, mesh_left: torch.Tensor, mesh_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Place initial quadrature points uniformly within each step.

        Variable-sampling methods compute their tableau b weights
        dynamically from the actual node positions, so any internally
        consistent placement works as the *initial* layout. We use
        evenly-spaced fractions ``[0, 1/(C-1), ..., 1]`` so that the
        first call lays out C points spanning each panel from end to
        end. Subsequent splits and merges (handled by
        ``_evaluate_adaptive_nodes`` and ``_merge_excess_nodes``) reuse and
        rearrange these points.

        At ``a = 1/2`` for the 3-point ``interpolatory3_variable``
        method this layout matches Simpson's rule exactly — see
        tests/test_exactness.py for the reduction check.

        Args:
            mesh_left: Left barrier of each step. Shape: [N, T].
            mesh_right: Right barrier of each step. Shape: [N, T].

        Returns:
            Quadrature point positions. Shape: [N, C, T].
        """
        dt = (mesh_right - mesh_left).unsqueeze(1)
        # Uniformly-spaced fractions of [0, 1] across C points.
        fractions = torch.linspace(
            0.0, 1.0, self.C, dtype=self.dtype, device=self.device
        ).view(self.C, 1)
        return mesh_left.unsqueeze(1) + fractions * dt

    def _evaluate_adaptive_nodes(
        self,
        f: Callable,
        idxs_add: torch.Tensor,
        y: torch.Tensor,
        nodes: torch.Tensor,
        f_args: tuple = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split failed steps and reuse existing evaluations where possible.

        Unlike the uniform solver which discards old evaluations, the variable
        solver can reuse them. New evaluation points are placed at the midpoints
        between each pair of consecutive existing points. The original and new
        points are then interleaved and reshaped into two sub-steps of C points
        each.

        For a step with C=3 points [p0, p1, p2], this creates midpoints
        [m01, m12] and interleaves them: [p0, m01, p1, m12, p2, p2]. The
        duplicate at position C acts as the shared boundary point. This is
        then reshaped into two sub-steps: [p0, m01, p1] and [m12, p2, p2].

        Args:
            f: The integrand function f(t). Takes [N, T], returns [N, D].
            idxs_add: Indices of steps that need splitting. Shape: [R].
            y: Current integrand evaluations, reused in sub-steps. Shape: [N, C, D].
            nodes: Current quadrature point positions. Shape: [N, C, T].
            f_args: Extra arguments passed to f.

        Returns:
            Tuple of (y_add_combined, nodes_new) where:
                - y_add_combined: Integrand values for the new sub-steps,
                    interleaving old and new evaluations. Shape: [2*R, C, D].
                - nodes_new: Quadrature point positions for the new sub-steps.
                    Shape: [2*R, C, T].
        """
        # Compute midpoints between each pair of consecutive quadrature points
        nodes_mid = (nodes[idxs_add, 1:] + nodes[idxs_add, :-1]) / 2  # [n_add, C-1, 1]
        # Evaluate the integrand at the new midpoints
        y_add = f(nodes_mid.view(-1, nodes.shape[-1]), *f_args)
        y_add = rearrange(y_add, "(N C) D -> N C D", N=len(idxs_add))
        D = y_add.shape[-1]

        # Build index arrays to interleave old points (even positions) and
        # new midpoints (odd positions) into a 2*C-length array
        select_prev_idxs = torch.arange(self.C, device=self.device) * 2
        select_prev_idxs[select_prev_idxs >= self.C] += 1
        select_add_idxs = torch.arange(self.Cm1, device=self.device) * 2 + 1
        select_add_idxs[select_add_idxs >= self.C] += 1

        # Interleave old and new time points into a combined array
        nodes_new = torch.nan * torch.ones(
            (len(idxs_add), (self.C) * 2, D), dtype=self.dtype, device=self.device
        )
        nodes_new[:, select_prev_idxs] = nodes[idxs_add]
        nodes_new[:, select_add_idxs] = nodes_mid
        # Duplicate the boundary point so both sub-steps share it
        nodes_new[:, self.C] = nodes_new[:, self.C - 1]
        # Reshape from [R, 2*C] into [2*R, C] (two sub-steps per original step)
        nodes_new = torch.reshape(nodes_new, (len(idxs_add) * 2, self.C, D))

        # Interleave old and new integrand values the same way
        y_add_combined = torch.nan * torch.ones(
            (len(idxs_add), self.C * 2, D), dtype=self.dtype, device=self.device
        )
        y_add_combined[:, select_prev_idxs] = y[idxs_add]
        y_add_combined[:, select_add_idxs] = y_add
        y_add_combined[:, self.C] = y_add_combined[:, self.C - 1]
        y_add_combined = torch.reshape(y_add_combined, (len(idxs_add) * 2, self.C, D))

        return y_add_combined, nodes_new

    def _merge_excess_nodes(
        self,
        nodes: torch.Tensor,
        mesh_quadratures: torch.Tensor,
        mesh_quadrature_errors: torch.Tensor,
        remove_idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge pairs of consecutive over-resolved steps into single larger steps.

        For variable sampling, merging concatenates the quadrature points from
        both steps (C + C-1 = 2C-1 points, since the shared boundary is not
        duplicated from the second step), then subsamples every other point to
        get back to C points. Because the points are now at non-standard
        fractional positions within the merged step, the variable method's
        ``tableau_b(c)`` will recompute appropriate weights on the next evaluation.

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
        nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=1)
        assert torch.all(nodes_flat[1:] - nodes_flat[:-1] + self.atol_assert >= 0)

        # Concatenate points from both steps (skip first point of second step
        # since it equals the last point of the first step), giving 2C-1 points
        combined_steps = torch.concatenate(
            [nodes[remove_idxs, :], nodes[remove_idxs + 1, 1:]], dim=1
        )
        mesh_quadratures_replace = (
            mesh_quadratures[remove_idxs] + mesh_quadratures[remove_idxs + 1]
        )
        mesh_quadrature_errors_replace = (
            mesh_quadrature_errors[remove_idxs]
            + mesh_quadrature_errors[remove_idxs + 1]
        )
        # Subsample every other point to reduce 2C-1 back to C points
        keep_idxs = torch.arange(self.C, dtype=torch.long, device=self.device) * 2

        # Remove the first step of each pair from the arrays
        remove_mask = torch.ones(len(nodes), dtype=torch.bool, device=self.device)
        remove_mask[remove_idxs] = False
        nodes_pruned = nodes[remove_mask]
        mesh_quadratures_pruned = mesh_quadratures[remove_mask]
        mesh_quadrature_errors_pruned = mesh_quadrature_errors[remove_mask]

        # Place the merged step data at the position of the second (kept) step,
        # adjusting indices to account for earlier removals shifting positions
        update_idxs = remove_idxs - torch.arange(len(remove_idxs), device=self.device)
        nodes_pruned[update_idxs] = combined_steps[:, keep_idxs]
        mesh_quadratures_pruned[update_idxs] = mesh_quadratures_replace
        mesh_quadrature_errors_pruned[update_idxs] = mesh_quadrature_errors_replace

        # Verify time ordering and step continuity after merging
        nodes_pruned_flat = torch.flatten(nodes_pruned, start_dim=0, end_dim=1)
        assert torch.all(
            nodes_pruned_flat[1:] - nodes_pruned_flat[:-1] + self.atol_assert >= 0
        )
        assert torch.allclose(
            nodes_pruned[:-1, -1, :],
            nodes_pruned[1:, 0, :],
            atol=self.atol_assert,
            rtol=self.rtol_assert,
        )

        return nodes_pruned, mesh_quadratures_pruned, mesh_quadrature_errors_pruned
