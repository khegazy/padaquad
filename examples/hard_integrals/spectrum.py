"""
A spectrum of 200 sharp peaks
=============================

``f(t) = sum_j 1 / (1 + ((t - c_j) / 10^-4)^2)`` on [0, 1] has 200 peaks of
width 10^-4, like the lines of a spectrum, at positions ``c_j`` spread over
the interval. Each peak needs its own short subintervals and the stretches
between them need few, so the mesh needs over a thousand subintervals of very
different widths. Adaptive splitting finds every peak without being told where
they are.

Run with::

    python examples/hard_integrals/spectrum.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from padaquad import hard_integrand_dict, integrate

f, solution, _ = hard_integrand_dict["spectrum"]
a = torch.tensor([0.0], dtype=torch.float64)
b = torch.tensor([1.0], dtype=torch.float64)
exact = float(solution(a, b).reshape(-1)[0])

# %%
# Integrate with ``gk21`` (the default method) and ``gk7``: with 7 nodes per
# subinterval, ``gk7`` spends fewer evaluations on the many subintervals that
# are split only to locate the peaks. The tolerance is the one of the
# Performance page: ``atol = 1e-8``, ``rtol = 0``.
for method in ("gk21", "gk7"):
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
        f"{method}: integral {float(result.integral.reshape(-1)[0]):.12f} "
        f"(exact {exact:.12f}, error {error:.1e}), converged {result.converged}, "
        f"{len(result.h)} subintervals, split at most {result.max_split_depth} times"
    )
    assert result.converged
    assert error < 1e-6

# %%
# The integrand and the width of every subinterval of the last run: the
# widths drop by orders of magnitude at every peak.
t = torch.linspace(0.0, 1.0, 200001, dtype=torch.float64).reshape(-1, 1)
edges = torch.cat([a, a + torch.cumsum(result.h[:, 0], 0)])  # subinterval ends
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
top.plot(t[:, 0], f(t)[:, 0], lw=0.5)
top.set_ylabel("f(t)")
bottom.stairs(result.h[:, 0], edges, baseline=None)
bottom.set(yscale="log", xlabel="t", ylabel="subinterval width")
fig.tight_layout()
plt.show()
