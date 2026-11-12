"""
One-call integration and the solver factory.

``integrate()`` computes a definite integral in one call, without building a
solver object. ``adaptive_quadrature()`` builds the solver for a sampling type;
keep the solver to integrate several times, for example to warm-start each call
from the mesh of the previous one.
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
    Build the solver for a sampling type and a method.

    Returns a ``UniformAdaptiveQuadrature`` for uniform sampling and a
    ``VariableAdaptiveQuadrature`` for variable sampling. ``sampling_type`` and
    ``method`` may be passed by position; every other option is a keyword
    argument of the solver's constructor.

    Parameters
    ----------
    sampling_type : str or steps
        ``"uniform"`` (fixed node fractions) or ``"variable"`` (weights
        computed from the node positions); ``"adaptive_uniform"``,
        ``"adaptive_variable"`` and the ``steps`` values are also accepted.
    method : str
        Name of the method, a key of ``UNIFORM_METHODS`` (uniform sampling) or
        ``VARIABLE_METHODS`` (variable sampling). Default: ``"gk21"``.
    **options : Any
        Keyword arguments of the solver's constructor (for example ``atol``,
        ``rtol``, ``f``, ``mesh_init``, ``mesh_final``, ``max_batch``,
        ``device``).

    Returns
    -------
    UniformAdaptiveQuadrature or VariableAdaptiveQuadrature
        The solver.

    Raises
    ------
    ValueError
        If ``sampling_type`` is not ``"uniform"`` or ``"variable"`` (or one of
        their other names), or ``method`` is not a method of that sampling type.
    TypeError
        If an option is not an argument of the solver's constructor.

    Examples
    --------
    Build the solver once and call it twice on the same integrand. With
    ``same_f=True`` the second call starts from the mesh the first call ended
    with and reuses its ``max_batch``.

    >>> import torch
    >>> from padaquad import adaptive_quadrature
    >>> _ = torch.manual_seed(0)  # the starting mesh is random
    >>> solver = adaptive_quadrature(
    ...     "variable", "interpolatory3_variable", atol=1e-8, rtol=1e-8, device="cpu"
    ... )
    >>> bounds = {
    ...     "mesh_init": torch.tensor([0.0], dtype=torch.float64),
    ...     "mesh_final": torch.tensor([1.0], dtype=torch.float64),
    ... }
    >>> first = solver.integrate(lambda t: t**2, max_batch=10_000, **bounds)
    >>> second = solver.integrate(lambda t: t**2, same_f=True, **bounds)
    >>> print(round(float(first.integral), 8), round(float(second.integral), 8))
    0.33333333 0.33333333
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
    Compute the integral of ``f`` from ``mesh_init`` to ``mesh_final`` in one call.

    Builds the solver with ``adaptive_quadrature(sampling, method, ...)`` and
    calls its ``integrate`` method once. Each keyword in ``options`` goes to
    that ``integrate`` call if it is one of the call's arguments (for example
    ``mesh_init``, ``mesh_final``, ``mesh``, ``y0``, ``f_args``,
    ``take_gradient``, ``max_batch``, ``result_device``), and to the solver's
    constructor otherwise (for example ``atol``, ``rtol``, ``remove_cut``,
    ``use_absolute_error_ratio``, ``device``). Defaults are those of the
    solver class. A one-call solver has no previous mesh, so
    ``same_f=True`` has no effect here; use ``adaptive_quadrature`` for
    repeated calls.

    The integrand depends only on the integration variable ``t`` (and on
    ``f_args``), never on the running value of the integral, so the nodes of
    many subintervals can be evaluated in one batch.

    Parameters
    ----------
    f : Callable
        The integrand. Takes points of shape ``[N, T]`` and returns values of
        shape ``[N, D]``, or a pair ``(values, tracked_variables)``; see
        :ref:`ug-shapes` and :ref:`ug-tracked-variables`.
    method : str
        Name of the method, a key of ``UNIFORM_METHODS`` (uniform sampling) or
        ``VARIABLE_METHODS`` (variable sampling). Default: ``"gk21"``.
    sampling : str or steps
        ``"uniform"`` (fixed node fractions, the default) or ``"variable"``
        (weights computed from the node positions), or any other sampling
        type ``adaptive_quadrature`` accepts.
    **options : Any
        Keyword arguments of the solver's ``integrate`` method and of its
        constructor.

    Returns
    -------
    IntegrationResult
        The integral, its error estimate, the mesh and convergence
        diagnostics.

    Raises
    ------
    ValueError
        If ``sampling`` is not a supported sampling type, or ``method`` is not
        a method of that sampling type.
    TypeError
        If an option is neither an argument of the ``integrate`` call nor of
        the constructor.

    Examples
    --------
    The integral of ``sin(t)`` from 0 to ``pi`` is 2.

    >>> import torch
    >>> from padaquad.integrate import integrate
    >>> _ = torch.manual_seed(0)  # the starting mesh is random
    >>> result = integrate(
    ...     torch.sin,
    ...     atol=1e-10,
    ...     rtol=1e-10,
    ...     mesh_init=torch.tensor([0.0], dtype=torch.float64),
    ...     mesh_final=torch.tensor([torch.pi], dtype=torch.float64),
    ...     device="cpu",
    ...     max_batch=10_000,
    ... )
    >>> print(round(float(result.integral), 8), result.converged)
    2.0 True
    """
    call = {name: value for name, value in options.items() if name in _CALL_OPTIONS}
    construct = {
        name: value for name, value in options.items() if name not in _CALL_OPTIONS
    }
    solver = adaptive_quadrature(sampling, method, **construct)
    return solver.integrate(f, **call)
