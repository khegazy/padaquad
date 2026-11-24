"""
A thousand integrals at once
============================

``f(t) = (sin(t), sin(2t), ..., sin(1000 t))`` on [0, 1] has 1,000 outputs,
so one integration computes 1,000 integrals. Every subinterval must be short
enough for the fastest output, ``sin(1000 t)``. With ``error_norm="max"`` a
subinterval is accepted only when the error of every output is below the
tolerance.

Run with::

    python examples/hard_integrals/many_outputs.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from padaquad import hard_integrand_dict, integrate

f, solution, _ = hard_integrand_dict["many_outputs"]
a = torch.tensor([0.0], dtype=torch.float64)
b = torch.tensor([1.0], dtype=torch.float64)
exact = solution(a, b)  # one integral per output, shape [1000]

# %%
# Integrate with ``gk21`` (the default method) and ``gk31``, whose higher
# polynomial degree suits these smooth outputs. The tolerance is the one of
# the Performance page: ``atol = 1e-8``, ``rtol = 0``, for every output.
for method in ("gk21", "gk31"):
    torch.manual_seed(0)  # the starting mesh is random
    result, _ = integrate(
        f,
        method,
        atol=1e-8,
        rtol=0.0,
        error_norm="max",
        mesh_init=a,
        mesh_final=b,
        device="cpu",
        max_batch=100_000,
        take_gradient=False,
    )
    errors = (result.integral - exact).abs()
    print(
        f"{method}: largest error of the {len(exact)} integrals {float(errors.max()):.1e}, "
        f"converged {result.converged}, {len(result.h)} subintervals"
    )
    assert result.converged
    assert float(errors.max()) < 1e-6

# %%
# The error of each of the 1,000 integrals of the last run, and the width of
# every subinterval: the widths are nearly equal, set by ``sin(1000 t)``.
k = torch.arange(1, len(exact) + 1)
left = a + torch.cumsum(result.h[:, 0], 0) - result.h[:, 0]
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5))
top.semilogy(k, errors.clamp_min(1e-18))
top.set(xlabel="output k (the integral of sin(k t))", ylabel="absolute error")
bottom.step(left, result.h[:, 0], where="post")
bottom.set(yscale="log", xlabel="t", ylabel="subinterval width")
fig.tight_layout()
plt.show()
