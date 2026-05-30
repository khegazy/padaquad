"""Tests using the Wolf-Schlegel potential energy surface.

The Wolf-Schlegel 2D potential is integrated along a linear path between
two minima as a 1D quadrature problem in t. The reference is
``scipy.integrate.quad`` (QUADPACK), used here as ground truth.

Phase 3 of the quadrature alignment plan: this test was previously
``parallel-vs-serial`` (where serial used ``torchdiffeq.odeint``). With
the serial path removed, the natural cross-validation is against
scipy.integrate.quad — a more rigorous reference than another
torch-based solver.
"""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    ATOL_LOOSE,
    REMOVE_CUT,
    RTOL_LOOSE,
    SEED,
    T_FINAL,
    T_INIT,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    UNIFORM_METHOD_NAMES,
)
from scipy import integrate as scipy_integrate

from padaquad import (
    adaptive_quadrature,
    steps,
    wolf_schlegel,
)


def _wolf_schlegel_scalar(t_value: float) -> float:
    """scipy-friendly Wolf-Schlegel evaluator: scalar in, scalar out."""
    t_tensor = torch.tensor([[t_value]], dtype=torch.float64)
    return wolf_schlegel(t_tensor).item()


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
def test_wolf_schlegel_parallel_vs_scipy(method_name, take_gradient):
    """Parallel integration of Wolf-Schlegel matches scipy.integrate.quad."""
    torch.manual_seed(SEED)
    parallel_solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=ATOL_LOOSE,
        rtol=RTOL_LOOSE,
        remove_cut=REMOVE_CUT,
        f=wolf_schlegel,
    )

    parallel_output = parallel_solver.integrate(
        mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=take_gradient
    )

    scipy_value, scipy_err = scipy_integrate.quad(
        _wolf_schlegel_scalar,
        float(T_INIT.item()),
        float(T_FINAL.item()),
        epsabs=ATOL_LOOSE,
        epsrel=RTOL_LOOSE,
    )

    diff = abs(parallel_output.integral.item() - scipy_value)
    # Generous bound — Wolf-Schlegel is wiggly, low-order methods at
    # ATOL_LOOSE = 1e-5 deliver a few-decimal-place result.
    bound = max(scipy_err, 1e-3 * abs(scipy_value))
    assert diff < bound, (
        f"{method_name}: parallel={parallel_output.integral.item():.6f}, "
        f"scipy={scipy_value:.6f}, diff={diff:.2e}, bound={bound:.2e}"
    )
