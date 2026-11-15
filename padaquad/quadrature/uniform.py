"""
Uniform-sampling adaptive quadrature: ``UniformAdaptiveQuadrature``.

In uniform sampling, the nodes of every subinterval sit at the method's fixed
node fractions, the tableau's ``c`` values (dopri5 always uses
``c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]``), and the weights ``b`` are constants.

When a subinterval is split, both halves are evaluated at fresh nodes at the
``c`` positions; the old integrand values are discarded. When the optimal mesh
merges two neighboring subintervals A and B, the merged subinterval spans
``[A.start, B.end]`` with nodes at the ``c`` positions; it is not evaluated, and
its contribution and error are the sums of A's and B's.

This class holds the uniform-specific overrides of the abstract methods
declared in ``AdaptiveQuadrature``; everything else, including node placement
and the integral computation, is shared.
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
    Adaptive quadrature solver for methods with fixed node fractions and weights.

    The nodes of every subinterval sit at the method's fixed node fractions,
    so the weights are fixed numbers. For example, dopri5 always evaluates the
    integrand at the fractions ``[0, 1/5, 3/10, 4/5, 8/9, 1, 1]`` of each
    subinterval's width. When a subinterval fails the tolerance, it is split in
    half and the integrand is evaluated at fresh nodes in both halves.

    The method is one of ``UNIFORM_METHODS``: the Runge-Kutta, Gauss-Kronrod
    and Clenshaw-Curtis methods, with ``gk21`` the default. The constructor
    takes the options of ``AdaptiveQuadrature``, all as keyword arguments.

    Examples
    --------
    Integrate ``cos(t)`` from 0 to 1; the exact value is ``sin(1)``:

    >>> import torch
    >>> from padaquad import UniformAdaptiveQuadrature
    >>> _ = torch.manual_seed(0)
    >>> solver = UniformAdaptiveQuadrature(
    ...     method="gk21", atol=1e-10, rtol=1e-10, device="cpu", max_batch=10_000
    ... )
    >>> result = solver.integrate(
    ...     torch.cos,
    ...     mesh_init=torch.tensor([0.0], dtype=torch.float64),
    ...     mesh_final=torch.tensor([1.0], dtype=torch.float64),
    ... )
    >>> print(round(float(result.integral), 6))
    0.841471
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
        Split failed subintervals at their midpoints and evaluate the integrand.

        Each subinterval that failed the error check is divided into two halves
        at its midpoint (the mean of its first and last node, which are its
        endpoints). Fresh nodes are placed in each half at the tableau's ``c``
        values, and the integrand is evaluated at all 2C new nodes.

        ``integrate`` does not call this method: its loop inserts the midpoint
        into the mesh and evaluates the new subintervals through
        ``_compute_nodes``, which gives the same nodes. Only the tests call it.

        Parameters
        ----------
        f : Callable[..., Any]
            The integrand f(t). Takes [N, T], returns [N, D].
        idxs_add : torch.Tensor
            Indices of the subintervals to split. Shape: [R].
        _y : torch.Tensor
            Current integrand values (unused here, present for interface
            compatibility with the variable solver). Shape: [N, C, D].
        nodes : torch.Tensor
            Current node positions. Shape: [N, C, T].
        f_args : tuple[Any, ...]
            Extra arguments passed to f.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (y_add, nodes_new) where:
                - y_add: Integrand values at the new nodes.
                    Shape: [2*R, C, D] (two halves per split subinterval).
                - nodes_new: The new node positions.
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
        Merge pairs of neighboring over-resolved subintervals into one each.

        For uniform sampling, merging replaces two neighboring subintervals
        A and B with one subinterval spanning ``[A.start, B.end]``, with new
        nodes at the tableau's ``c`` positions. The merged subinterval is not
        evaluated: its contribution and error are the sums of A's and B's.

        Parameters
        ----------
        nodes : torch.Tensor
            Node positions of all subintervals. Shape: [N, C, T].
        mesh_quadratures : torch.Tensor
            Integral contribution of each subinterval. Shape: [N, D].
        mesh_quadrature_errors : torch.Tensor
            Error estimate of each subinterval. Shape: [N, D].
        remove_idxs : torch.Tensor
            Index of the first subinterval of each pair to merge.
            The pair (remove_idxs[i], remove_idxs[i]+1) is merged.
            Shape: [R].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (nodes_pruned, mesh_quadratures_pruned, mesh_quadrature_errors_pruned)
            with the merged subintervals replacing the original pairs. Each has
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
