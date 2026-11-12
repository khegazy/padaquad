"""
Test integrands, most with an exact integral.

``t``, ``t_squared``, ``sine_squared``, ``exp`` and ``damped_sine`` each have a
``*_solution`` function that returns the exact integral from ``mesh_init`` to
``mesh_final``. ``identity`` (with ``identity_solution``) and
``wolf_schlegel`` (no exact integral) are not in ``integrand_dict``.

``integrand_dict`` maps each name to a tuple ``(integrand, solution,
error_cutoff)``, where ``error_cutoff`` is the largest relative error
``|computed - exact| / |exact|`` the tests accept.
"""

from __future__ import annotations

import torch


def identity(_t: torch.Tensor, _y: torch.Tensor | None = None) -> int:
    """
    Constant integrand ``f(t) = 1``.

    Exact integral: ``mesh_final - mesh_init``.

    Parameters
    ----------
    _t : torch.Tensor
        Points of the integration variable t (unused). Shape: [N, T].
    _y : torch.Tensor or None
        Unused.

    Returns
    -------
    int
        The constant value 1.
    """
    return 1


def identity_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor
) -> torch.Tensor:
    """
    Exact integral of 1 from mesh_init to mesh_final.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [T].

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """
    return mesh_final - mesh_init


def t(t: torch.Tensor, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Linear integrand ``f(t) = t``.

    Exact integral: ``(mesh_final**2 - mesh_init**2) / 2``.

    Parameters
    ----------
    t : torch.Tensor
        Points of the integration variable t. Shape: [N, T].
    _y : torch.Tensor or None
        Unused.

    Returns
    -------
    torch.Tensor
        The input t unchanged. Shape: [N, T].
    """
    return t


def t_solution(mesh_init: torch.Tensor, mesh_final: torch.Tensor) -> torch.Tensor:
    """
    Exact integral of t from mesh_init to mesh_final.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [T].

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """
    return 0.5 * (mesh_final**2 - mesh_init**2)


def t_squared(t: torch.Tensor, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Quadratic integrand ``f(t) = t**2``.

    Exact integral: ``(mesh_final**3 - mesh_init**3) / 3``.

    Parameters
    ----------
    t : torch.Tensor
        Points of the integration variable t. Shape: [N, T].
    _y : torch.Tensor or None
        Unused.

    Returns
    -------
    torch.Tensor
        t squared. Shape: [N, T].
    """
    return t**2


def t_squared_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor
) -> torch.Tensor:
    """
    Exact integral of t^2 from mesh_init to mesh_final.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [T].

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """
    return (mesh_final**3 - mesh_init**3) / 3.0


def sine_squared(
    t: torch.Tensor, w: float = 3.7, _y: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Oscillating integrand ``f(t) = sin(2*pi*w*t)**2``.

    A larger ``w`` gives more oscillations per unit of t, so the adaptive mesh
    needs narrower subintervals.

    Parameters
    ----------
    t : torch.Tensor
        Points of the integration variable t. Shape: [N, T].
    w : float
        Frequency: number of periods of ``sin(2*pi*w*t)`` per unit of t.
    _y : torch.Tensor or None
        Unused.

    Returns
    -------
    torch.Tensor
        ``sin(2*pi*w*t)**2`` at each point. Shape: [N, T].
    """
    return torch.sin(t * w * 2 * torch.pi) ** 2


def sine_squared_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, w: float = 3.7
) -> torch.Tensor:
    """
    Exact integral of ``sin(2*pi*w*t)**2`` from mesh_init to mesh_final.

    With ``sin(x)**2 = (1 - cos(2x)) / 2`` and ``k = 4*pi*w``, the integral is
    ``(b - a) / 2 - (sin(k*b) - sin(k*a)) / (2k)`` for bounds a and b.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [T].
    w : float
        Frequency (must match the integrand).

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """
    _w = 4 * torch.pi * w
    return (mesh_final - mesh_init) / 2.0 - (
        torch.sin(torch.tensor([_w * mesh_final]))
        - torch.sin(torch.tensor([_w * mesh_init]))
    ) / (2 * _w)


def exp(t: torch.Tensor, a: float = 5, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Exponential integrand ``f(t) = exp(a*t)``.

    The integrand grows by a factor ``exp(a)`` per unit of t, so its size
    varies strongly over a long domain.

    Parameters
    ----------
    t : torch.Tensor
        Points of the integration variable t. Shape: [N, T].
    a : float
        Growth rate.
    _y : torch.Tensor or None
        Unused.

    Returns
    -------
    torch.Tensor
        ``exp(a*t)`` at each point. Shape: [N, T].
    """
    return torch.exp(a * t)


def exp_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, a: float = 5
) -> torch.Tensor:
    """
    Exact integral of ``exp(a*t)`` from mesh_init to mesh_final.

    For bounds a0 and a1 it is ``(exp(a*a1) - exp(a*a0)) / a``.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [T].
    a : float
        Growth rate (must match the integrand).

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """
    return (
        torch.exp(torch.tensor([mesh_final * a]))
        - torch.exp(torch.tensor([mesh_init * a]))
    ) / a


def damped_sine(
    t: torch.Tensor, w: float = 3.7, a: float = 5, _y: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Damped oscillating integrand ``f(t) = exp(-a*t) * sin(2*pi*w*t)``.

    It changes sign twice per period and its amplitude decays, so positive
    and negative parts of the integral partly cancel.

    Parameters
    ----------
    t : torch.Tensor
        Points of the integration variable t. Shape: [N, T].
    w : float
        Frequency: number of periods of ``sin(2*pi*w*t)`` per unit of t.
    a : float
        Decay rate of the amplitude ``exp(-a*t)``.
    _y : torch.Tensor or None
        Unused.

    Returns
    -------
    torch.Tensor
        ``exp(-a*t) * sin(2*pi*w*t)`` at each point. Shape: [N, T].
    """
    return torch.exp(-a * t) * torch.sin(w * t * 2 * torch.pi)


def damped_sine_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, w: float = 3.7, a: float = 5
) -> torch.Tensor:
    """
    Exact integral of ``exp(-a*t) * sin(2*pi*w*t)`` from mesh_init to mesh_final.

    With ``k = 2*pi*w``, an antiderivative is
    ``F(t) = -exp(-a*t) * (a*sin(k*t) + k*cos(k*t)) / (a**2 + k**2)``
    (derivation: differentiating F gives back the integrand), and the integral
    is ``F(mesh_final) - F(mesh_init)``.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [T].
    w : float
        Frequency (must match the integrand).
    a : float
        Decay rate (must match the integrand).

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """
    _w = 2 * torch.pi * w

    def numerator(t: torch.Tensor, _w: float, a: float) -> torch.Tensor:
        """Evaluate exp(-a t) (a sin(_w t) + _w cos(_w t)), the antiderivative's numerator."""
        t = torch.tensor([t])
        return torch.exp(-a * t) * (a * torch.sin(_w * t) + _w * torch.cos(_w * t))

    return (
        -1
        * (numerator(mesh_final, _w, a) - numerator(mesh_init, _w, a))
        / (a**2 + _w**2)
    )


# ---------------------------------------------------------------------------
# A two-dimensional quartic potential evaluated along a straight path
# ---------------------------------------------------------------------------

_WS_MIN_INIT = torch.tensor([1.133, -1.486])
_WS_MIN_FINAL = torch.tensor([-1.166, 1.477])


def wolf_schlegel(t: torch.Tensor, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Quartic potential of two coordinates, evaluated along a straight path.

    The path runs from :math:`(x, y) = (1.133, -1.486)` at t = 0 to
    :math:`(-1.166, 1.477)` at t = 1, two points within 0.001 of local minima
    of the potential

    .. math::

        V(x, y) = 10\\,(x^4 + y^4 - 2x^2 - 4y^2 + xy + 0.2x + 0.1y).

    The integral over t from 0 to 1 is the average of V along the path. The
    integrand has one output (D = 1). No exact integral is provided.

    Parameters
    ----------
    t : torch.Tensor
        Points of the integration variable t, from 0 to 1. Shape: [N, 1].
    _y : torch.Tensor or None
        Unused.

    Returns
    -------
    torch.Tensor
        V at the path point of each t. Shape: [N, 1].

    Examples
    --------
    >>> import torch
    >>> from padaquad.examples import wolf_schlegel
    >>> from padaquad.integrate import integrate
    >>> _ = torch.manual_seed(0)  # the starting mesh is random
    >>> result = integrate(
    ...     wolf_schlegel,
    ...     atol=1e-9,
    ...     rtol=1e-9,
    ...     mesh_init=torch.tensor([0.0], dtype=torch.float64),
    ...     mesh_final=torch.tensor([1.0], dtype=torch.float64),
    ...     device="cpu",
    ...     max_batch=10_000,
    ... )
    >>> print(round(float(result.integral), 6))
    -30.658508
    """
    while len(t.shape) < 2:
        t = t.unsqueeze(0)
    interpolate = _WS_MIN_INIT.to(t.device) + t * (_WS_MIN_FINAL - _WS_MIN_INIT).to(
        t.device
    )
    x = interpolate[:, 0].unsqueeze(-1)
    y = interpolate[:, 1].unsqueeze(-1)
    return 10 * (x**4 + y**4 - 2 * x**2 - 4 * y**2 + x * y + 0.2 * x + 0.1 * y)


# Registry of test integrands: name -> (integrand, exact-integral function, error_cutoff).
# error_cutoff is the largest relative error |computed - exact| / |exact| the tests accept.
integrand_dict = {
    "t": (t, t_solution, 1e-7),
    "t_squared": (t_squared, t_squared_solution, 1e-6),
    "sine_squared": (sine_squared, sine_squared_solution, 1e-6),
    "exp": (exp, exp_solution, 1e-6),
    "damped_sine": (damped_sine, damped_sine_solution, 1e-6),
}
