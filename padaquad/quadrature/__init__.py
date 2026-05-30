"""
Parallel adaptive numerical quadrature.

The library's core: a parallel adaptive-stepsize quadrature engine
that evaluates many integration steps simultaneously in a batch.
Unlike sequential ODE-style integrators (where each step depends on
the previous result), this solver assumes the integrand ``f(t)``
depends only on time, so step contributions are independent and can
be batched on GPU.

Submodules:

  - ``base``: ``AdaptiveQuadrature``, the abstract base class with
    the main integration loop, error computation, mesh refinement,
    and memory management. Subclasses implement
    ``_calculate_integral``, ``_compute_nodes``,
    ``_evaluate_adaptive_nodes``, and ``_merge_excess_nodes``.

  - ``uniform``: ``_UniformAdaptiveQuadratureBase`` — overrides for
    methods with fixed tableau ``c`` (RK, Gauss-Kronrod,
    Clenshaw-Curtis).

  - ``variable``: ``_VariableAdaptiveQuadratureBase`` — overrides
    for methods with arbitrary node positions where weights are
    recomputed per-step from the actual ``c``.

The concrete classes that users instantiate (``UniformAdaptiveQuadrature``
and ``VariableAdaptiveQuadrature``) live in ``padaquad.runge_kutta``
together with the RK-specific ``_calculate_integral`` implementation.
"""

from __future__ import annotations

from .base import AdaptiveQuadrature
from .uniform import _UniformAdaptiveQuadratureBase
from .variable import _VariableAdaptiveQuadratureBase

__all__ = [
    "AdaptiveQuadrature",
    "_UniformAdaptiveQuadratureBase",
    "_VariableAdaptiveQuadratureBase",
]
