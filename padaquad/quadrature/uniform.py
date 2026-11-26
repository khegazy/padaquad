"""
Uniform-sampling adaptive quadrature: ``UniformAdaptiveQuadrature``.

In uniform sampling, the nodes of every subinterval sit at the method's fixed
node fractions, the tableau's ``c`` values (dopri5 always uses
``c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]``), and the weights ``b`` are constants.

When a subinterval is split, both halves are evaluated at fresh nodes at the
``c`` positions; the old integrand values are discarded. When the optimal mesh
merges two neighboring subintervals, their shared mesh point is removed; the
merged subinterval is not evaluated, and its contribution and error are the
sums of the two.

Everything, including node placement, the integral computation and the
merging, is shared with the variable solver in ``AdaptiveQuadrature``; this
class only selects the uniform method registry.
"""

from __future__ import annotations

from padaquad.base import steps

from .base import AdaptiveQuadrature


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
    >>> result, state = solver.integrate(
    ...     torch.cos,
    ...     mesh_init=torch.tensor([0.0], dtype=torch.float64),
    ...     mesh_final=torch.tensor([1.0], dtype=torch.float64),
    ... )
    >>> print(round(float(result.integral), 6))
    0.841471
    """

    sampling_type = steps.ADAPTIVE_UNIFORM
