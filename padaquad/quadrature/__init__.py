"""
Parallel adaptive numerical quadrature.

The library's core: an adaptive quadrature engine that evaluates many
subintervals at once, in batches. Unlike an ODE solver, where each step
depends on the result of the previous one, it assumes the integrand ``f(t)``
depends only on the integration variable t, so the subintervals are
independent and can be evaluated in parallel, for example on a GPU.

Submodules:

  - ``base``: ``AdaptiveQuadrature``, the abstract base class with
    the main integration loop, error computation, mesh refinement,
    node placement and the integral computation, which reach the
    quadrature method only through its ``MethodClass`` interface.
    Subclasses set ``sampling_type`` and implement
    ``_evaluate_adaptive_nodes`` and ``_merge_excess_nodes``.

  - ``uniform``: ``UniformAdaptiveQuadrature``, the solver for methods with
    fixed node fractions and weights (Runge-Kutta, Gauss-Kronrod,
    Clenshaw-Curtis).

  - ``variable``: ``VariableAdaptiveQuadrature``, the solver for methods
    whose weights are computed from the actual node positions.

  - ``memory``: the ``_MemoryManagement`` mixin (memory benchmark, batch
    size, out-of-memory recognition).
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
