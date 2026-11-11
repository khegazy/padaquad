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
    node placement and the integral computation, which reach the
    quadrature rule only through its ``MethodClass`` interface.
    Subclasses set ``sampling_type`` and implement
    ``_evaluate_adaptive_nodes`` and ``_merge_excess_nodes``.

  - ``uniform``: ``UniformAdaptiveQuadrature`` — the solver for rules with
    fixed node fractions (RK, Gauss-Kronrod, Clenshaw-Curtis).

  - ``variable``: ``VariableAdaptiveQuadrature`` — the solver for rules
    whose weights are recomputed per step from the actual nodes.

  - ``memory``: the ``_MemoryManagement`` mixin (memory benchmark, batch
    budget, out-of-memory recognition).
"""

from __future__ import annotations

from .base import AdaptiveQuadrature
from .uniform import UniformAdaptiveQuadrature
from .variable import VariableAdaptiveQuadrature

__all__ = [
    "AdaptiveQuadrature",
    "UniformAdaptiveQuadrature",
    "VariableAdaptiveQuadrature",
]
