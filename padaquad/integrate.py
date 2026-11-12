"""
High-level public API for adaptive numerical quadrature.

Provides ``integrate()``, the main entry point for users who want
to compute a definite integral without manually constructing solver objects,
and ``adaptive_quadrature()``, the factory that builds the solver for a
sampling type (``integrate()`` uses it).
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from .base import DEFAULT_METHOD, DEFAULT_SAMPLING_TYPE, get_sampling_type, steps
from .quadrature import (
    AdaptiveQuadrature,
    UniformAdaptiveQuadrature,
    VariableAdaptiveQuadrature,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .results import IntegrationResult


def adaptive_quadrature(
    sampling_type: str | steps = DEFAULT_SAMPLING_TYPE,
    method: str = DEFAULT_METHOD,
    **options: Any,
) -> UniformAdaptiveQuadrature | VariableAdaptiveQuadrature:
    """
    Build the adaptive solver for a sampling type.

    Selects between the uniform-sampling and the variable-sampling solver.
    ``sampling_type`` and ``method`` may be passed by position; every other
    option is a keyword argument of the solver constructor
    (``AdaptiveQuadrature.__init__``).

    Args:
        sampling_type: A ``steps`` value or a name ('uniform',
            'adaptive_uniform', 'variable', 'adaptive_variable'). Default:
            ``DEFAULT_SAMPLING_TYPE`` ('uniform').
        method: Name of the quadrature method. Default: ``DEFAULT_METHOD``
            ('gk21').
        **options: Keyword options of the solver constructor (atol, rtol, f,
            mesh_init, max_batch, device, ...).

    Returns:
        The solver.

    Raises:
        ValueError: If sampling_type is not a recognized adaptive sampling type.
        TypeError: If an option is not an option of the solver.
    """
    if isinstance(sampling_type, str):
        sampling_type = get_sampling_type(sampling_type)
    if sampling_type == steps.ADAPTIVE_UNIFORM:
        return UniformAdaptiveQuadrature(method=method, **options)
    elif sampling_type == steps.ADAPTIVE_VARIABLE:
        return VariableAdaptiveQuadrature(method=method, **options)
    else:
        raise ValueError(
            f"adaptive_quadrature supports sampling_type {steps.ADAPTIVE_UNIFORM} "
            f"or {steps.ADAPTIVE_VARIABLE}, got {sampling_type}."
        )


# Options of the solver's ``integrate`` call, read from its signature. In the
# one-shot ``integrate()`` below, these go to the call and every other keyword
# goes to the constructor, so no option or default is repeated here.
_CALL_OPTIONS = frozenset(
    inspect.signature(AdaptiveQuadrature.integrate).parameters
) - {"self"}


def integrate(
    f: Callable[..., Any],
    method: str = DEFAULT_METHOD,
    sampling: str | steps = DEFAULT_SAMPLING_TYPE,
    **options: Any,
) -> IntegrationResult:
    """
    Compute the definite integral of f from mesh_init to mesh_final.

    Builds the solver with ``adaptive_quadrature(sampling, method, ...)`` and
    calls its ``integrate`` once. Each keyword in ``options`` goes to the
    solver's ``integrate`` call if it is one of that call's arguments (for
    example mesh_init, mesh_final, mesh, y0, take_gradient, f_args, same_f,
    max_batch, result_device), and to the solver constructor otherwise (for
    example atol, rtol, remove_cut, use_absolute_error_ratio, device). The
    defaults are those of ``AdaptiveQuadrature.__init__`` and
    ``AdaptiveQuadrature.integrate``.

    The integrand f depends only on t, not on an accumulated state y; this
    is what lets many points be evaluated in one batch.

    Example::

        result = integrate(
            torch.sin,
            atol=1e-8,
            rtol=1e-8,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([3.14159], dtype=torch.float64),
            device="cpu",
            max_batch=10_000,
        )
        print(round(float(result.integral), 6))  # 2.0

    Args:
        f: The integrand. Takes points of shape [N, T] and returns values of
            shape [N, D], or a 2-tuple ``(values, tracked_variables)`` (see
            ``AdaptiveQuadrature.integrate``).
        method: Name of the quadrature method, a key of ``UNIFORM_METHODS``
            (uniform sampling) or ``VARIABLE_METHODS`` (variable sampling).
            Default: ``DEFAULT_METHOD`` ('gk21').
        sampling: 'uniform' (fixed node positions) or 'variable' (weights
            computed from the actual node positions), or any sampling type
            ``adaptive_quadrature`` accepts. Default: 'uniform'.
        **options: Keyword options of ``AdaptiveQuadrature.integrate`` and of
            ``AdaptiveQuadrature.__init__``.

    Returns:
        IntegrationResult with the integral, its error estimate, the mesh and
        diagnostics.

    Raises:
        ValueError: If sampling is not a supported sampling type, or method is
            not a method of that sampling type.
        TypeError: If an option is neither a call nor a constructor option.
    """
    call = {name: value for name, value in options.items() if name in _CALL_OPTIONS}
    construct = {
        name: value for name, value in options.items() if name not in _CALL_OPTIONS
    }
    solver = adaptive_quadrature(sampling, method, **construct)
    return solver.integrate(f, **call)
