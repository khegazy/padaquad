"""
Example integrand functions with known analytical solutions for testing.

Each function computes f(t) and has a corresponding ``*_solution`` function that
returns the exact value of the integral from mesh_init to mesh_final. These are used
by the test suite to verify numerical integration accuracy.

The ``integrand_dict`` maps names to ``(integrand, solution, error_cutoff)`` tuples,
where ``error_cutoff`` is the maximum acceptable relative error for tests.
"""

from __future__ import annotations

import torch


def identity(_t: torch.Tensor, _y: torch.Tensor | None = None) -> int:
    """
    Constant integrand f(t) = 1.

    Analytical integral: mesh_final - mesh_init.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        y: Unused. Present for interface compatibility.

    Returns:
        The constant value 1.
    """
    return 1


def identity_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor
) -> torch.Tensor:
    """
    Analytical solution for the integral of 1 from mesh_init to mesh_final.

    Args:
        mesh_init: Lower integration bound. Shape: [T].
        mesh_final: Upper integration bound. Shape: [T].

    Returns:
        Exact integral value. Shape: [T].
    """
    return mesh_final - mesh_init


def t(t: torch.Tensor, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Linear integrand f(t) = t.

    Analytical integral: (mesh_final^2 - mesh_init^2) / 2.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        y: Unused. Present for interface compatibility.

    Returns:
        The input t unchanged. Shape: [N, T].
    """
    return t


def t_solution(mesh_init: torch.Tensor, mesh_final: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution for the integral of t from mesh_init to mesh_final.

    Args:
        mesh_init: Lower integration bound. Shape: [T].
        mesh_final: Upper integration bound. Shape: [T].

    Returns:
        Exact integral value. Shape: [T].
    """
    return 0.5 * (mesh_final**2 - mesh_init**2)


def t_squared(t: torch.Tensor, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Quadratic integrand f(t) = t^2.

    Analytical integral: (mesh_final^3 - mesh_init^3) / 3.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        y: Unused. Present for interface compatibility.

    Returns:
        t squared. Shape: [N, T].
    """
    return t**2


def t_squared_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor
) -> torch.Tensor:
    """
    Analytical solution for the integral of t^2 from mesh_init to mesh_final.

    Args:
        mesh_init: Lower integration bound. Shape: [T].
        mesh_final: Upper integration bound. Shape: [T].

    Returns:
        Exact integral value. Shape: [T].
    """
    return (mesh_final**3 - mesh_init**3) / 3.0


def sine_squared(
    t: torch.Tensor, w: float = 3.7, _y: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Oscillatory integrand f(t) = sin^2(2*pi*w*t).

    Tests the integrator's ability to handle rapid oscillations. Higher w
    values create more oscillations, requiring finer step sizes.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        w: Frequency parameter controlling oscillation rate.
        y: Unused. Present for interface compatibility.

    Returns:
        sin^2(2*pi*w*t) evaluated at each time point. Shape: [N, T].
    """
    return torch.sin(t * w * 2 * torch.pi) ** 2


def sine_squared_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, w: float = 3.7
) -> torch.Tensor:
    """
    Analytical solution for the integral of sin^2(2*pi*w*t) from mesh_init to mesh_final.

    Uses the identity sin^2(x) = (1 - cos(2x))/2 to derive the closed form.

    Args:
        mesh_init: Lower integration bound. Shape: [T].
        mesh_final: Upper integration bound. Shape: [T].
        w: Frequency parameter (must match the integrand).

    Returns:
        Exact integral value. Shape: [T].
    """
    _w = 4 * torch.pi * w
    return (mesh_final - mesh_init) / 2.0 - (
        torch.sin(torch.tensor([_w * mesh_final]))
        - torch.sin(torch.tensor([_w * mesh_init]))
    ) / (2 * _w)


def exp(t: torch.Tensor, a: float = 5, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Exponential integrand f(t) = exp(a*t).

    Tests integration of rapidly growing functions. The growth rate ``a``
    makes this increasingly difficult for larger integration domains.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        a: Growth rate parameter.
        y: Unused. Present for interface compatibility.

    Returns:
        exp(a*t) evaluated at each time point. Shape: [N, T].
    """
    return torch.exp(a * t)


def exp_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, a: float = 5
) -> torch.Tensor:
    """
    Analytical solution for the integral of exp(a*t) from mesh_init to mesh_final.

    Args:
        mesh_init: Lower integration bound. Shape: [T].
        mesh_final: Upper integration bound. Shape: [T].
        a: Growth rate parameter (must match the integrand).

    Returns:
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
    Damped oscillatory integrand f(t) = exp(-a*t) * sin(2*pi*w*t).

    Combines exponential decay with oscillation. This is one of the more
    challenging test cases as it requires the integrator to handle both
    rapid sign changes and varying amplitude.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        w: Frequency parameter controlling oscillation rate.
        a: Damping rate parameter controlling exponential decay.
        y: Unused. Present for interface compatibility.

    Returns:
        exp(-a*t)*sin(2*pi*w*t) evaluated at each time point. Shape: [N, T].
    """
    return torch.exp(-a * t) * torch.sin(w * t * 2 * torch.pi)


def damped_sine_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, w: float = 3.7, a: float = 5
) -> torch.Tensor:
    """
    Analytical solution for the integral of exp(-a*t)*sin(2*pi*w*t) from mesh_init to mesh_final.

    Derived via integration by parts (or Laplace transform tables).

    Args:
        mesh_init: Lower integration bound. Shape: [T].
        mesh_final: Upper integration bound. Shape: [T].
        w: Frequency parameter (must match the integrand).
        a: Damping rate parameter (must match the integrand).

    Returns:
        Exact integral value. Shape: [T].
    """
    _w = 2 * torch.pi * w

    def numerator(t, _w, a):
        """Evaluate the antiderivative numerator exp(-at)(a*sin(wt) + w*cos(wt))."""
        t = torch.tensor([t])
        return torch.exp(-a * t) * (a * torch.sin(_w * t) + _w * torch.cos(_w * t))

    return (
        -1
        * (numerator(mesh_final, _w, a) - numerator(mesh_init, _w, a))
        / (a**2 + _w**2)
    )


# ---------------------------------------------------------------------------
# Wolf-Schlegel 2D potential energy surface along a linear interpolation path
# ---------------------------------------------------------------------------

_WS_MIN_INIT = torch.tensor([1.133, -1.486])
_WS_MIN_FINAL = torch.tensor([-1.166, 1.477])


def wolf_schlegel(t: torch.Tensor, _y: torch.Tensor | None = None) -> torch.Tensor:
    """
    Wolf-Schlegel 2D potential evaluated along a linear path between two minima.

    The path linearly interpolates between (1.133, -1.486) and (-1.166, 1.477)
    as t goes from 0 to 1. Returns the potential energy:
    V(x,y) = 10*(x^4 + y^4 - 2x^2 - 4y^2 + xy + 0.2x + 0.1y).

    This is a multi-dimensional output integrand (returns a scalar per time point)
    useful for testing integration of chemistry potential energy surfaces.

    Args:
        t: Time points in [0, 1]. Shape: [N, T].
        _y: Unused. Present for interface compatibility.

    Returns:
        Potential energy at each time point. Shape: [N, 1].
    """
    while len(t.shape) < 2:
        t = t.unsqueeze(0)
    interpolate = _WS_MIN_INIT.to(t.device) + t * (_WS_MIN_FINAL - _WS_MIN_INIT).to(
        t.device
    )
    x = interpolate[:, 0].unsqueeze(-1)
    y = interpolate[:, 1].unsqueeze(-1)
    return 10 * (x**4 + y**4 - 2 * x**2 - 4 * y**2 + x * y + 0.2 * x + 0.1 * y)


# Registry of test integrands: maps name -> (integrand_fn, analytical_solution_fn, error_cutoff).
# error_cutoff is the maximum acceptable relative error |computed - exact| / |exact|.
integrand_dict = {
    "t": (t, t_solution, 1e-7),
    "t_squared": (t_squared, t_squared_solution, 1e-6),
    "sine_squared": (sine_squared, sine_squared_solution, 1e-6),
    "exp": (exp, exp_solution, 1e-6),
    "damped_sine": (damped_sine, damped_sine_solution, 1e-6),
}
