"""
Adaptive numerical quadrature in PyTorch.

padaquad computes definite integrals of a known integrand over the integration
variable t by evaluating many subintervals in parallel batches, with autograd
through the integration.

Public names: ``integrate`` (one-shot quadrature), ``adaptive_quadrature``
(solver factory for repeated calls), the solvers ``UniformAdaptiveQuadrature``
and ``VariableAdaptiveQuadrature``, the result type ``IntegrationResult`` and the
``IntegrationState`` it hands to the next call, the error
``BackwardOutOfMemoryError``, the
method registries ``UNIFORM_METHODS`` and ``VARIABLE_METHODS``, the test
integrands ``integrand_dict``, ``hard_integrand_dict`` and ``wolf_schlegel``, and the sampling-mode enum
``steps``.
"""

from __future__ import annotations

from ._frontend import adaptive_quadrature, integrate
from .base import steps
from .examples import hard_integrand_dict, integrand_dict, wolf_schlegel
from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .quadrature import UniformAdaptiveQuadrature, VariableAdaptiveQuadrature
from .quadrature.memory import BackwardOutOfMemoryError
from .results import IntegrationResult, IntegrationState

__all__ = [
    "UNIFORM_METHODS",
    "VARIABLE_METHODS",
    "BackwardOutOfMemoryError",
    "IntegrationResult",
    "IntegrationState",
    "UniformAdaptiveQuadrature",
    "VariableAdaptiveQuadrature",
    "adaptive_quadrature",
    "hard_integrand_dict",
    "integrand_dict",
    "integrate",
    "steps",
    "wolf_schlegel",
]
