"""
Warm starts from the optimal mesh
=================================

This script checks the merge rules that build ``mesh_optimal``, the mesh a later
call starts from when it is passed as ``mesh=``. After an integration, pairs of
neighbouring subintervals are merged when the merged subinterval is predicted to
pass its tolerance with a margin. With ``merge_rule="predicted"`` (the default)
the merged subinterval's integral and error estimate are computed from the
integrand values already stored for the two neighbours, by interpolation; with
``merge_rule="power_law"`` (padaquad 1.3.0 and 1.4.0) the neighbours' summed
error is multiplied by ``2**(error_exponent - 1)``. See the user guide, "The
optimal mesh and the merge rule".

What a warm start must not do is make the result less accurate than the call it
started from. Each chain below is one call from an explicit starting mesh (the
cold call) followed by four calls each starting from the previous call's
``mesh_optimal`` and ``max_batch`` (the warm calls). A warm call
passes when its error is at most the largest of

- the tolerance ``max(atol, rtol * |I|)``;
- 1.05 times the cold call's error (a cold call already outside the tolerance
  is a limit of the error estimate, not of merging);
- the sum of the rounding floors ``50 * eps * |Q_i|`` of its subintervals,
  the accuracy padaquad accepts when the tolerance is below what the dtype can
  resolve.

The script also checks that ``"predicted"`` costs no more integrand evaluations
than ``"power_law"`` on a smooth integrand, and prints the evaluations of both
for every chain.

Run it with::

    python examples/verification/merge_prediction.py

It exits with status 1 if a check fails, except for the known limitation of
``dopri5`` explained below, which is reported as ``[KNOWN]``.

Known limitation: no merge rule can see what the stored values miss
-------------------------------------------------------------------

Every merge rule decides from values the integration already stored. A feature
that lies between the stored nodes is invisible to all of them, and the next
call evaluates the merged subinterval at new nodes. For Gauss-Kronrod and
Clenshaw-Curtis methods the merged subinterval's error estimate then catches the
feature and the subinterval is split again. ``dopri5``'s error weights are much
smaller than its integration weights at some nodes, so its estimate can accept
the merged subinterval with a larger error, under every merge rule.
"""

# %%
# Warm-start chains in float64
# ----------------------------
#
# Every method except ``adaptive_heun`` and ``fehlberg2`` (their meshes are
# large at this tolerance), on three integrands with known integrals: a smooth
# one, a narrow peak and a kink. Every chain starts from 16 equal subintervals,
# so the output is the same on every machine.

from __future__ import annotations

import math

import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS, adaptive_quadrature

THIRD = 1.0 / 3.0
INTEGRANDS = {
    "exp": (torch.exp, math.e - 1.0),
    "peak": (
        lambda t: 1.0 / (1.0 + 625.0 * (t - 0.5) ** 2),
        2.0 * math.atan(12.5) / 25.0,
    ),
    "kink": (lambda t: torch.abs(t - THIRD), (THIRD**2 + (1.0 - THIRD) ** 2) / 2.0),
}
KNOWN_LIMITATIONS = {"dopri5"}
N_WARM = 4

failures = 0
known = 0


def chain(sampling, name, f, exact, mesh, dtype, merge_rule, atol, rtol):
    """Run a cold call from ``mesh`` and N_WARM warm calls, each from the
    previous call's ``mesh_optimal`` and ``max_batch``.

    Returns the warm calls' integrand evaluations, their starting subintervals
    that had to be split again, the worst warm error divided by its bound, and
    whether every warm call is finite and converged.
    """
    solver = adaptive_quadrature(
        sampling,
        name,
        atol=atol,
        rtol=rtol,
        dtype=dtype,
        device="cpu",
        max_batch=100_000,
        merge_rule=merge_rule,
    )
    count = [0]

    def counted(t):
        count[0] += t.shape[0]
        return f(t)

    eps = torch.finfo(dtype).eps
    tol = max(atol, rtol * abs(exact))
    cold = solver.integrate(f=counted, mesh=mesh, take_gradient=False)
    cold_error = abs(float(cold.integral) - exact)
    count[0] = 0
    splits, worst, healthy = 0, 0.0, True
    warm = cold
    for _ in range(N_WARM):
        warm = solver.integrate(
            f=counted,
            mesh=warm.mesh_optimal,
            max_batch=warm.max_batch,
            take_gradient=False,
        )
        floor = float((50 * eps * warm.mesh_quadratures.abs()).sum())
        bound = max(tol, 1.05 * cold_error, floor)
        worst = max(worst, abs(float(warm.integral) - exact) / bound)
        splits += warm.n_start_steps_split
        healthy &= bool(torch.isfinite(warm.integral).all()) and warm.converged
    return count[0], splits, worst, healthy


runs = [
    ("uniform", n) for n in UNIFORM_METHODS if n not in ("adaptive_heun", "fehlberg2")
]
runs += [("variable", n) for n in VARIABLE_METHODS if n not in UNIFORM_METHODS]
mesh = torch.linspace(0.0, 1.0, 17, dtype=torch.float64).unsqueeze(-1)
print(
    f"{'method':24s} {'integrand':9s} {'warm evaluations':>17s} {'(power_law)':>12s} "
    f"{'re-split':>9s} {'worst error/bound':>18s}"
)
for sampling, name in runs:
    for fname, (f, exact) in INTEGRANDS.items():
        evals, splits, worst, healthy = chain(
            sampling, name, f, exact, mesh, torch.float64, "predicted", 1e-6, 1e-5
        )
        evals_old, *_ = chain(
            sampling, name, f, exact, mesh, torch.float64, "power_law", 1e-6, 1e-5
        )
        print(
            f"{name:24s} {fname:9s} {evals:17d} {evals_old:12d} {splits:9d} {worst:18.2e}"
        )
        if worst > 1.0 or not healthy:
            status = "KNOWN" if name in KNOWN_LIMITATIONS else "FAIL"
            known += status == "KNOWN"
            failures += status == "FAIL"
            print(f"  [{status}] {name} on {fname}: a warm call is outside its bound")
        if fname == "exp" and evals > evals_old:
            failures += 1
            print(f"  [FAIL] {name} on exp: predicted used more evaluations")

# %%
# Rounding-level tolerances
# -------------------------
#
# At rtol = 1e-14 the tolerance of sin(200 t), whose integral is about 200 times
# smaller than the sum of the subintervals' absolute contributions, is below what
# float64 can certify, and sin(2 pi t) integrates to 0, so only the rounding
# floors set its tolerance. Every chain starts from 64 equal subintervals.

mesh = torch.linspace(0.0, 1.0, 65, dtype=torch.float64).unsqueeze(-1)
rounding = {
    "sin(200 t)": (lambda t: torch.sin(200.0 * t), (1.0 - math.cos(200.0)) / 200.0),
    "sin(2 pi t)": (lambda t: torch.sin(2.0 * math.pi * t), 0.0),
}
for name in ("gk21", "cc33", "cc65"):
    for fname, (f, exact) in rounding.items():
        evals, splits, worst, healthy = chain(
            "uniform", name, f, exact, mesh, torch.float64, "predicted", 0.0, 1e-14
        )
        print(
            f"{name:6s} rtol 1e-14 {fname:12s} warm evaluations {evals:6d}, "
            f"re-split {splits}, worst error/bound {worst:.2e}"
        )
        if worst > 1.0 or not healthy:
            failures += 1
            print(f"  [FAIL] {name} on {fname}: a warm call is outside its bound")

# %%
# float32
# -------
#
# In float32 the error estimates of the high-degree methods sit at the rounding
# level, where ``"power_law"`` merges nothing; ``"predicted"`` must stay finite
# and within the bound.

mesh = torch.linspace(0.0, 1.0, 17, dtype=torch.float32).unsqueeze(-1)
for name in ("gk21", "cc33"):
    for fname in ("exp", "kink"):
        f, exact = INTEGRANDS[fname]
        evals, splits, worst, healthy = chain(
            "uniform", name, f, exact, mesh, torch.float32, "predicted", 1e-5, 1e-5
        )
        evals_old, *_ = chain(
            "uniform", name, f, exact, mesh, torch.float32, "power_law", 1e-5, 1e-5
        )
        print(
            f"{name:6s} float32 {fname:5s} warm evaluations {evals:5d} "
            f"(power_law {evals_old:5d}), re-split {splits}, worst error/bound {worst:.2e}"
        )
        if worst > 1.0 or not healthy:
            failures += 1
            print(
                f"  [FAIL] {name} on {fname} in float32: a warm call is outside its bound"
            )

# %%
# Result
# ------

if known:
    print(f"[KNOWN] {known} chain(s) of a method with a known limitation exceeded")
    print("        the bound (see the explanation at the top of this script).")
if failures:
    print(f"[FAIL] {failures} check(s) failed.")
    raise SystemExit(1)
print("Every warm start stayed within its bound.")
