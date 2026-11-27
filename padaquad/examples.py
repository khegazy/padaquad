"""
Test integrands, most with an exact integral.

``t``, ``t_squared``, ``sine_squared``, ``exp`` and ``damped_sine`` each have a
``*_solution`` function that returns the exact integral from ``mesh_init`` to
``mesh_final``. ``identity`` (with ``identity_solution``) and
``wolf_schlegel`` (no exact integral) are not in ``integrand_dict``.

``integrand_dict`` maps each name to a tuple ``(integrand, solution,
error_cutoff)``, where ``error_cutoff`` is the largest relative error
``|computed - exact| / |exact|`` the tests accept. ``hard_integrand_dict`` has
the same form for harder integrands: many oscillations (of one frequency, or
of a frequency that grows), a sharp peak, a spectrum of 200 sharp peaks, an
infinite derivative, and 1,000 outputs at once.

The integrands in both dictionaries take the array module as the keyword
argument ``xp`` (``torch`` by default; also ``numpy`` or ``jax.numpy``), so
one definition serves PadaQuad and libraries that use NumPy or JAX arrays.
The ``*_solution`` functions take torch tensors.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from types import ModuleType

# A torch tensor or a NumPy array; JAX arrays also work at run time.
Array = TypeVar("Array", torch.Tensor, npt.NDArray[np.floating[Any]])


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


def t(
    t: Array,
    _y: torch.Tensor | None = None,
    *,
    xp: ModuleType = torch,  # noqa: ARG001 (every integrand takes xp)
) -> Array:
    """
    Linear integrand ``f(t) = t``.

    Exact integral: ``(mesh_final**2 - mesh_init**2) / 2``.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, T].
    _y : torch.Tensor or None
        Unused.
    xp : module
        The array module of t (unused here; accepted like every integrand).

    Returns
    -------
    torch.Tensor or numpy.ndarray
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


def t_squared(
    t: Array,
    _y: torch.Tensor | None = None,
    *,
    xp: ModuleType = torch,  # noqa: ARG001 (every integrand takes xp)
) -> Array:
    """
    Quadratic integrand ``f(t) = t**2``.

    Exact integral: ``(mesh_final**3 - mesh_init**3) / 3``.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, T].
    _y : torch.Tensor or None
        Unused.
    xp : module
        The array module of t (unused here; accepted like every integrand).

    Returns
    -------
    torch.Tensor or numpy.ndarray
        The square of t. Shape: [N, T].
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
    t: Array,
    w: float = 3.7,
    _y: torch.Tensor | None = None,
    *,
    xp: ModuleType = torch,
) -> Array:
    """
    Oscillating integrand ``f(t) = sin(2*pi*w*t)**2``.

    A larger ``w`` gives more oscillations per unit of t, so the adaptive mesh
    needs narrower subintervals.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, T].
    w : float
        Frequency: number of periods of ``sin(2*pi*w*t)`` per unit of t.
    _y : torch.Tensor or None
        Unused.
    xp : module
        The array module of t: ``torch`` (default), ``numpy`` or ``jax.numpy``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        ``sin(2*pi*w*t)**2`` at each point. Shape: [N, T].
    """
    return xp.sin(t * w * 2 * xp.pi) ** 2


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


def exp(
    t: Array,
    a: float = 5,
    _y: torch.Tensor | None = None,
    *,
    xp: ModuleType = torch,
) -> Array:
    """
    Exponential integrand ``f(t) = exp(a*t)``.

    The integrand grows by a factor ``exp(a)`` per unit of t, so its size
    varies strongly over a long domain.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, T].
    a : float
        Growth rate.
    _y : torch.Tensor or None
        Unused.
    xp : module
        The array module of t: ``torch`` (default), ``numpy`` or ``jax.numpy``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        ``exp(a*t)`` at each point. Shape: [N, T].
    """
    return xp.exp(a * t)


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
    t: Array,
    w: float = 3.7,
    a: float = 5,
    _y: torch.Tensor | None = None,
    *,
    xp: ModuleType = torch,
) -> Array:
    """
    Damped oscillating integrand ``f(t) = exp(-a*t) * sin(2*pi*w*t)``.

    It changes sign twice per period and its amplitude decays, so positive
    and negative parts of the integral partly cancel.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, T].
    w : float
        Frequency: number of periods of ``sin(2*pi*w*t)`` per unit of t.
    a : float
        Decay rate of the amplitude ``exp(-a*t)``.
    _y : torch.Tensor or None
        Unused.
    xp : module
        The array module of t: ``torch`` (default), ``numpy`` or ``jax.numpy``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        ``exp(-a*t) * sin(2*pi*w*t)`` at each point. Shape: [N, T].
    """
    return xp.exp(-a * t) * xp.sin(w * t * 2 * xp.pi)


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
        """
        Evaluate exp(-a t) (a sin(_w t) + _w cos(_w t)), the antiderivative's numerator.

        Parameters
        ----------
        t : torch.Tensor
            A bound of the integral, with one element (T = 1).
        _w : float
            Angular frequency, 2 pi w.
        a : float
            Decay rate.

        Returns
        -------
        torch.Tensor
            The numerator at t. Shape: [1].
        """
        t = torch.tensor([t])
        return torch.exp(-a * t) * (a * torch.sin(_w * t) + _w * torch.cos(_w * t))

    return (
        -1
        * (numerator(mesh_final, _w, a) - numerator(mesh_init, _w, a))
        / (a**2 + _w**2)
    )


def sharp_peak(
    t: Array,
    c: float | npt.ArrayLike = 0.37,
    eps: float = 1e-3,
    *,
    xp: ModuleType = torch,
) -> Array:
    """
    Lorentzian peaks ``f(t) = sum_j 1 / (1 + ((t - c_j) / eps)**2)`` of width ``eps``.

    Away from the peak positions ``c_j`` the integrand is nearly 0, so the
    subintervals have to be refined only in regions about ``eps`` wide around
    each of them.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, 1].
    c : float or array_like
        Position of the peak, or a 1-D sequence of positions (one peak each).
    eps : float
        Half width of each peak at half its height.
    xp : module
        The array module of t: ``torch`` (default), ``numpy`` or ``jax.numpy``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        The sum of the peaks at each point. Shape: [N, 1].
    """
    if isinstance(t, torch.Tensor):
        centers = torch.as_tensor(c, dtype=t.dtype, device=t.device)
    else:
        centers = xp.asarray(c, dtype=t.dtype)
    return (1 / (1 + ((t - centers) / eps) ** 2)).sum(-1)[:, None]


def sharp_peak_solution(
    mesh_init: torch.Tensor,
    mesh_final: torch.Tensor,
    c: float | npt.ArrayLike = 0.37,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    Exact integral of ``sharp_peak`` from mesh_init to mesh_final.

    An antiderivative of one peak is ``eps * arctan((t - c) / eps)``; the
    integral is the sum over the peaks.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [1].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [1].
    c : float or array_like
        Position or positions of the peaks (must match the integrand).
    eps : float
        Half width of the peaks (must match the integrand).

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [1].
    """
    centers = torch.as_tensor(c, dtype=mesh_init.dtype, device=mesh_init.device)
    return (
        eps
        * (
            torch.arctan((mesh_final - centers) / eps)
            - torch.arctan((mesh_init - centers) / eps)
        )
    ).sum(-1, keepdim=True)


def chirp(t: Array, w: float = 1e4, *, xp: ModuleType = torch) -> Array:
    """
    Chirp ``f(t) = 2*t*sin(w*t**2)``, whose frequency grows along t.

    The phase ``w*t**2`` goes through ``w / (2*pi)`` periods on [0, 1] (about
    1,600 for the default ``w``), each shorter than the one before, so the
    subintervals have to shrink steadily from left to right.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, T].
    w : float
        Angular frequency of the phase ``w*t**2``.
    xp : module
        The array module of t: ``torch`` (default), ``numpy`` or ``jax.numpy``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        ``2*t*sin(w*t**2)`` at each point. Shape: [N, T].
    """
    return 2 * t * xp.sin(w * t**2)


def chirp_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, w: float = 1e4
) -> torch.Tensor:
    """
    Exact integral of ``chirp`` from mesh_init to mesh_final.

    An antiderivative is ``-cos(w*t**2) / w``, so the integral from a to b is
    ``(cos(w*a**2) - cos(w*b**2)) / w``, written as
    ``2 * sin(w*(a**2 + b**2)/2) * sin(w*(b**2 - a**2)/2) / w`` to avoid the
    cancellation of the two cosines (as in ``many_outputs_solution``).

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound a. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound b. Shape: [T].
    w : float
        Angular frequency (must match the integrand).

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """
    a2, b2 = mesh_init**2, mesh_final**2
    return 2 * torch.sin(w * (a2 + b2) / 2) * torch.sin(w * (b2 - a2) / 2) / w


def kink(t: Array, c: float = 1 / 3, *, xp: ModuleType = torch) -> Array:
    """
    Integrand ``f(t) = sqrt(|t - c|)``, whose derivative is infinite at ``c``.

    The integrand is finite everywhere, so every method can evaluate it at
    any point, but no polynomial approximates it well near ``c``.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, T].
    c : float
        Where the derivative is infinite.
    xp : module
        The array module of t: ``torch`` (default), ``numpy`` or ``jax.numpy``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        ``sqrt(|t - c|)`` at each point. Shape: [N, T].
    """
    return xp.sqrt(xp.abs(t - c))


def kink_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, c: float = 1 / 3
) -> torch.Tensor:
    """
    Exact integral of ``kink`` from mesh_init to mesh_final.

    An antiderivative is ``(2/3) * sign(t - c) * |t - c|**1.5``.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound. Shape: [T].
    mesh_final : torch.Tensor
        Upper integration bound. Shape: [T].
    c : float
        Where the derivative is infinite (must match the integrand).

    Returns
    -------
    torch.Tensor
        Exact integral value. Shape: [T].
    """

    def antiderivative(x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate ``(2/3) * sign(x - c) * |x - c|**1.5``.

        Parameters
        ----------
        x : torch.Tensor
            A bound of the integral. Shape: [T].

        Returns
        -------
        torch.Tensor
            The antiderivative at x. Shape: [T].
        """
        return 2 / 3 * torch.sign(x - c) * torch.abs(x - c) ** 1.5

    return antiderivative(mesh_final) - antiderivative(mesh_init)


def many_outputs(t: Array, n: int = 1000, *, xp: ModuleType = torch) -> Array:
    """
    Vector-valued integrand ``f_k(t) = sin(k*t)`` for ``k = 1, ..., n``.

    Each of the n outputs is one integral; the highest frequencies set the
    size of the subintervals for all of them.

    Parameters
    ----------
    t : torch.Tensor or numpy.ndarray
        Points of the integration variable t. Shape: [N, 1].
    n : int
        Number of outputs.
    xp : module
        The array module of t: ``torch`` (default), ``numpy`` or ``jax.numpy``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        ``sin(k*t)`` for every k at each point. Shape: [N, n].
    """
    k = xp.arange(1, n + 1, dtype=t.dtype)
    if xp is torch:
        k = k.to(t.device)  # torch.arange builds on the CPU by default
    return xp.sin(t * k)


def many_outputs_solution(
    mesh_init: torch.Tensor, mesh_final: torch.Tensor, n: int = 1000
) -> torch.Tensor:
    """
    Exact integrals of ``many_outputs`` from mesh_init to mesh_final.

    The integral of ``sin(k*t)`` is ``(cos(k*a) - cos(k*b)) / k``, written as
    ``2 * sin(k*(a + b)/2) * sin(k*(b - a)/2) / k``: the difference of two
    cosines loses about 9 digits to cancellation where it is small (k = 710
    on [0, 1]); the product does not.

    Parameters
    ----------
    mesh_init : torch.Tensor
        Lower integration bound a. Shape: [1].
    mesh_final : torch.Tensor
        Upper integration bound b. Shape: [1].
    n : int
        Number of outputs (must match the integrand).

    Returns
    -------
    torch.Tensor
        Exact integral of each output. Shape: [n].
    """
    k = torch.arange(1, n + 1, dtype=mesh_init.dtype, device=mesh_init.device)
    return (
        2
        * torch.sin(k * (mesh_init + mesh_final) / 2)
        * torch.sin(k * (mesh_final - mesh_init) / 2)
        / k
    )


# ---------------------------------------------------------------------------
# A two-dimensional quartic potential evaluated along a straight path
# ---------------------------------------------------------------------------

_WS_MIN_INIT = torch.tensor([1.133, -1.486])
_WS_MIN_FINAL = torch.tensor([-1.166, 1.477])


def wolf_schlegel(t: torch.Tensor, _y: torch.Tensor | None = None) -> torch.Tensor:
    r"""
    Quartic potential of two coordinates, evaluated along a straight path.

    The path runs from :math:`(x, y) = (1.133, -1.486)` at t = 0 to
    :math:`(-1.166, 1.477)` at t = 1, two points within 0.001 of local minima
    of the potential

    .. math::

        V(x, y) = 10\,(x^4 + y^4 - 2x^2 - 4y^2 + xy + 0.2x + 0.1y).

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
    >>> from padaquad import integrate
    >>> _ = torch.manual_seed(0)  # the starting mesh is random
    >>> result, state = integrate(
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

# The 200 peak positions of "spectrum": the golden-ratio sequence
# frac(0.5 + j*g), g = (sqrt(5) - 1)/2, spreads them evenly over [0, 1]
# without a pattern a regular starting mesh could line up with.
_SPECTRUM_CENTERS = (0.5 + (5**0.5 - 1) / 2 * np.arange(200)) % 1.0

# Harder integrands, in the same form. "oscillatory" is damped_sine with 100
# periods and a decay rate of 1: exp(-t) * sin(200*pi*t). "spectrum" is
# sharp_peak with 200 peaks of width 1e-4.
hard_integrand_dict = {
    "oscillatory": (
        partial(damped_sine, w=100, a=1),
        partial(damped_sine_solution, w=100, a=1),
        1e-6,
    ),
    "chirp": (chirp, chirp_solution, 1e-6),
    "sharp_peak": (sharp_peak, sharp_peak_solution, 1e-6),
    "spectrum": (
        partial(sharp_peak, c=_SPECTRUM_CENTERS, eps=1e-4),
        partial(sharp_peak_solution, c=_SPECTRUM_CENTERS, eps=1e-4),
        1e-6,
    ),
    "kink": (kink, kink_solution, 1e-6),
    "many_outputs": (many_outputs, many_outputs_solution, 1e-6),
}
