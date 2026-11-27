"""
A chirp
=======

``f(t) = 2 t sin(10^4 t^2)`` on [0, 1] goes through about 1,600 periods, each
shorter than the one before: its frequency grows in proportion to t. The
subintervals must shrink steadily from left to right, and several hundred of
them are needed. PadaQuad evaluates the nodes of all the subintervals waiting
to be checked in one call of the integrand.

Run with::

    python examples/hard_integrals/chirp.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from padaquad import hard_integrand_dict, integrate

f, solution, _ = hard_integrand_dict["chirp"]
a = torch.tensor([0.0], dtype=torch.float64)
b = torch.tensor([1.0], dtype=torch.float64)
exact = float(solution(a, b).reshape(-1)[0])

# %%
# Integrate with ``gk21`` (the default method) and ``gk31``, whose higher
# polynomial degree suits a smooth integrand that oscillates. The tolerance is
# the one of the Performance page: ``atol = 1e-8``, ``rtol = 0``.
for method in ("gk21", "gk31"):
    torch.manual_seed(0)  # the starting mesh is random
    result, _ = integrate(
        f,
        method,
        atol=1e-8,
        rtol=0.0,
        mesh_init=a,
        mesh_final=b,
        device="cpu",
        max_batch=100_000,
        take_gradient=False,
    )
    error = abs(float(result.integral.reshape(-1)[0]) - exact)
    print(
        f"{method}: integral {float(result.integral.reshape(-1)[0]):.12e} "
        f"(exact {exact:.12e}, error {error:.1e}), converged {result.converged}, "
        f"{len(result.h)} subintervals, split at most {result.max_split_depth} times"
    )
    assert result.converged
    assert error < 1e-6

# %%
# The integrand and the width of every subinterval of the last run: the
# widths shrink as the periods do.
t = torch.linspace(0.0, 1.0, 40001, dtype=torch.float64).reshape(-1, 1)
edges = torch.cat([a, a + torch.cumsum(result.h[:, 0], 0)])  # subinterval ends
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
top.plot(t[:, 0], f(t)[:, 0], lw=0.2)
top.set_ylabel("f(t)")
bottom.stairs(result.h[:, 0], edges, baseline=None)
bottom.set(yscale="log", xlabel="t", ylabel="subinterval width")
fig.tight_layout()
plt.show()
