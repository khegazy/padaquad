"""
Variable-sampling adaptive quadrature: ``VariableAdaptiveQuadrature``.

In variable sampling, the weights ``b`` of each subinterval are computed from
the actual node positions by the method's ``weights(nodes)``, so nodes need not
sit at fixed fractions of the subinterval.

That would let a split reuse the integrand values already computed: the old
nodes keep their positions in the two halves and are re-weighted. This class's
split method implements that reuse, but ``AdaptiveQuadrature.integrate`` does
not call it. The integration loop places the nodes of every subinterval, split
or not, at the method's node fractions and evaluates the integrand there, as
for uniform sampling.

Everything else, including node placement, the integral computation and the
merging of subintervals for the optimal mesh, is shared with the uniform
solver in ``AdaptiveQuadrature``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange

from padaquad.base import steps

from .base import AdaptiveQuadrature

if TYPE_CHECKING:
    from collections.abc import Callable


class VariableAdaptiveQuadrature(AdaptiveQuadrature):
    """
    Adaptive quadrature solver for methods whose weights depend on node positions.

    The method computes the weights of each subinterval from where its nodes
    actually are, instead of using fixed numbers. The integration itself runs
    as in ``UniformAdaptiveQuadrature``: the nodes of every subinterval are
    placed at the method's node fractions, and when a subinterval fails the
    tolerance, it is split in half and the integrand is evaluated at fresh
    nodes in both halves.

    The method is one of ``VARIABLE_METHODS``: 'adaptive_heun' or
    'interpolatory3_variable'. The constructor takes the options of
    ``AdaptiveQuadrature``, all as keyword arguments.

    Examples
    --------
    Integrate ``exp(t)`` from 0 to 1; the exact value is ``e - 1``:

    >>> import torch
    >>> from padaquad import VariableAdaptiveQuadrature
    >>> _ = torch.manual_seed(0)
    >>> solver = VariableAdaptiveQuadrature(
    ...     method="interpolatory3_variable",
    ...     atol=1e-8,
    ...     rtol=1e-8,
    ...     device="cpu",
    ...     max_batch=10_000,
    ... )
    >>> result, state = solver.integrate(
    ...     torch.exp,
    ...     mesh_init=torch.tensor([0.0], dtype=torch.float64),
    ...     mesh_final=torch.tensor([1.0], dtype=torch.float64),
    ... )
    >>> print(round(float(result.integral), 6))
    1.718282
    """

    sampling_type = steps.ADAPTIVE_VARIABLE

    def _evaluate_adaptive_nodes(
        self,
        f: Callable[..., Any],
        idxs_add: torch.Tensor,
        y: torch.Tensor,
        nodes: torch.Tensor,
        f_args: tuple[Any, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split failed subintervals, reusing their existing integrand values.

        Unlike the uniform solver, which discards old values, this keeps them.
        New nodes are placed midway between each pair of neighboring old
        nodes, so only C - 1 new integrand values are computed per split. The
        old and new nodes (2C - 1 in all) are interleaved, the middle one is
        duplicated as the shared endpoint, and the 2C entries are reshaped
        into two halves of C nodes each.

        For a subinterval with C=3 nodes [p0, p1, p2], this creates midpoints
        [m01, m12] and the array [p0, m01, p1, p1, m12, p2], reshaped into the
        halves [p0, m01, p1] and [p1, m12, p2]. The split point is the middle
        node, which is the subinterval's midpoint when the node fractions are
        symmetric about 1/2.

        ``integrate`` does not call this method: its loop evaluates both halves
        of a split subinterval at fresh nodes from ``_compute_nodes``. Only the
        unit tests call it. When D differs from T, ``nodes_new`` comes out with
        shape [2*R, C, D], because it is allocated with last dimension D.

        Parameters
        ----------
        f : Callable[..., Any]
            The integrand f(t). Takes [N, T], returns [N, D].
        idxs_add : torch.Tensor
            Indices of the subintervals to split. Shape: [R].
        y : torch.Tensor
            Current integrand values, reused in the halves. Shape: [N, C, D].
        nodes : torch.Tensor
            Current node positions. Shape: [N, C, T].
        f_args : tuple[Any, ...]
            Extra arguments passed to f.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (y_add_combined, nodes_new) where:
                - y_add_combined: Integrand values for the new halves,
                    interleaving old and new values. Shape: [2*R, C, D].
                - nodes_new: Node positions of the new halves.
                    Shape: [2*R, C, T] when D = T (see above).
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
        select_add_idxs = torch.arange(self.C - 1, device=self.device) * 2 + 1
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
