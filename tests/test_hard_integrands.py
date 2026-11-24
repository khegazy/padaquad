"""The test integrands in every array module, and their exact integrals.

Each integrand of ``integrand_dict`` and ``hard_integrand_dict`` is written
once for torch, NumPy and JAX (the ``xp`` argument): the forms must agree.
Each exact-solution function of ``hard_integrand_dict`` is checked against
``scipy.integrate.quad`` on two intervals, which guards its derivation, and
padaquad must reach the benchmark's tolerance on each harder integrand.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate
import torch

from padaquad import hard_integrand_dict, integrand_dict, integrate

ALL = {**integrand_dict, **hard_integrand_dict}
F64 = torch.float64
POINTS = torch.linspace(0.0, 1.7, 1000, dtype=F64).reshape(-1, 1)
# Points where an integrand is not smooth or has a narrow peak, for
# scipy.integrate.quad (the spectrum's peak positions, from its partial).
BREAKS = {
    "sharp_peak": [0.37],
    "spectrum": list(hard_integrand_dict["spectrum"][0].keywords["c"]),
    "kink": [1 / 3],
}
# Output components of many_outputs checked against quad: the first, one
# where (1 - cos k)/k cancels badly on [0, 1] (k = 710), and the last.
COMPONENTS = {"many_outputs": [0, 709, 999]}


@pytest.mark.parametrize("name", ALL)
def test_numpy_form_matches_torch(name: str) -> None:
    f = ALL[name][0]
    np.testing.assert_allclose(
        f(POINTS.numpy(), xp=np), f(POINTS).numpy(), rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize("name", ALL)
def test_jax_form_matches_torch(name: str) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    with jax.enable_x64(True):
        values = np.asarray(ALL[name][0](jnp.asarray(POINTS.numpy()), xp=jnp))
    np.testing.assert_allclose(
        values, ALL[name][0](POINTS).numpy(), rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.2, 1.7)])
@pytest.mark.parametrize("name", hard_integrand_dict)
def test_exact_solution_matches_quad(name: str, bounds: tuple[float, float]) -> None:
    f, solution, _ = hard_integrand_dict[name]
    a, b = bounds
    exact = solution(torch.tensor([a], dtype=F64), torch.tensor([b], dtype=F64))
    exact = exact.reshape(-1)
    for d in COMPONENTS.get(name, [0]):
        reference, error = scipy.integrate.quad(
            lambda s, d=d: float(f(np.array([[s]]), xp=np)[0, d]),
            a,
            b,
            points=[p for p in BREAKS.get(name, []) if a < p < b] or None,
            epsabs=1e-14,
            epsrel=1e-13,
            limit=10_000,
        )
        assert error < 1e-12
        assert float(exact[d]) == pytest.approx(reference, rel=1e-10, abs=1e-13)


@pytest.mark.parametrize("name", hard_integrand_dict)
def test_padaquad_reaches_the_benchmark_tolerance(name: str) -> None:
    f, solution, _ = hard_integrand_dict[name]
    a = torch.tensor([0.0], dtype=F64)
    b = torch.tensor([1.0], dtype=F64)
    torch.manual_seed(0)
    result, _ = integrate(
        f,
        "gk21",
        atol=1e-8,
        rtol=0.0,
        error_norm="max",
        mesh_init=a,
        mesh_final=b,
        device="cpu",
        max_batch=100_000,
        take_gradient=False,
    )
    assert result.converged
    error = (result.integral.reshape(-1) - solution(a, b).reshape(-1)).abs().max()
    assert float(error) < 1e-6
