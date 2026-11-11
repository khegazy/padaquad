from __future__ import annotations

from .base import steps
from .examples import integrand_dict, wolf_schlegel
from .integrate import adaptive_quadrature, integrate
from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .quadrature import UniformAdaptiveQuadrature, VariableAdaptiveQuadrature
from .results import IntegrationResult

__all__ = [
    "UNIFORM_METHODS",
    "VARIABLE_METHODS",
    "IntegrationResult",
    "UniformAdaptiveQuadrature",
    "VariableAdaptiveQuadrature",
    "adaptive_quadrature",
    "integrand_dict",
    "integrate",
    "steps",
    "wolf_schlegel",
]
