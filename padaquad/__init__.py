from __future__ import annotations

from .base import steps
from .examples import integrand_dict, wolf_schlegel
from .integrate import integrate
from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .results import IntegrationResult
from .runge_kutta import (
    UniformAdaptiveQuadrature,
    VariableAdaptiveQuadrature,
    adaptive_quadrature,
)
