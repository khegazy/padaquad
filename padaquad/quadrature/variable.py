"""
Variable-sampling adaptive quadrature: ``VariableAdaptiveQuadrature``.

In variable sampling, the weights ``b`` of each subinterval are computed from
the actual node positions by the method's ``weights(nodes)``, so nodes need not
sit at fixed fractions of the subinterval. The integration loop nevertheless
places the nodes of every subinterval, split or not, at the method's node
fractions and evaluates the integrand there, as for uniform sampling: a split
subinterval's values are not reused (see
``docs/issues/variable-sampling-reuse.md``).

Everything else, including node placement, the integral computation and the
merging of subintervals for the optimal mesh, is shared with the uniform
solver in ``AdaptiveQuadrature``.
"""

from __future__ import annotations

from padaquad.base import steps

from .base import AdaptiveQuadrature


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
