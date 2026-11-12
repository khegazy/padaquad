"""
Polynomial degree of every method
=================================

The *degree* of a quadrature method is the highest polynomial degree it
integrates exactly. It is the main accuracy property of a method: a method of
degree d makes an error on one subinterval of width h that falls like
h^(d + 2) for a smooth integrand, so a higher degree means fewer subintervals
for the same accuracy.

This script measures the degree of every method in padaquad from its stored
nodes and weights, and checks it against the degree the method claims
(``method.degree``). It also checks each method's *error estimate*. padaquad
estimates the error on a subinterval as the difference between the method and
a lower-degree method that shares its nodes; the solver assumes that this
lower-degree method has degree ``error_exponent - 2``, and the script checks
that too.

Run it with::

    python examples/verification/polynomial_degree.py

It prints one row per method and exits with status 1 if any check fails. The
same checks run in the test suite (``tests/test_exactness.py``).
"""

# %%
# Part 1: measure every method
# ----------------------------
#
# ``measure_degree`` integrates the Legendre polynomials P_0, P_1, P_2, ... over
# one subinterval and compares each result with the exact value. The degree is
# the last polynomial integrated without error (to 1e-12). "Miss" is the error
# on the first polynomial the method gets wrong; it is far above rounding
# (about 1e-16), so the measurement is unambiguous.
#
# ``interpolatory3_variable`` computes its weights from the actual node
# positions. With the middle node exactly at the center it becomes Simpson's
# rule, which has one degree more, so it is measured with the middle node at
# 0.3, which shows its general degree.

from __future__ import annotations

import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS

GENERAL_POSITION = {
    "interpolatory3_variable": torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
}

methods = dict(UNIFORM_METHODS)
methods.update({f"{name} (variable)": m for name, m in VARIABLE_METHODS.items()})

failures = 0
print(
    f"{'method':34s} {'nodes':>5s} {'claimed':>7s} {'measured':>8s} {'miss':>9s}   ",
    end="",
)
print(f"{'estimate: expected':>18s} {'measured':>8s}")
for label, method in methods.items():
    nodes = GENERAL_POSITION.get(label.removesuffix(" (variable)"))
    measured = method.measure_degree(nodes)
    miss = float(measured.errors[measured.degree + 1])
    ok = (
        measured.degree == method.degree
        and measured.lower_degree == method.error_exponent - 2
    )
    failures += not ok
    print(
        f"{label:34s} {method.n_nodes:5d} {method.degree:7d} {measured.degree:8d} "
        f"{miss:9.1e}   {method.error_exponent - 2:18d} {measured.lower_degree:8d}  "
        f"[{'OK ' if ok else 'FAIL'}]"
    )

# %%
# Part 2: why the test uses Legendre polynomials and not powers of t
# ------------------------------------------------------------------
#
# A simpler test would integrate the powers t^0, t^1, t^2, ... instead. That
# test fails for the high-degree methods, and this part shows why.
#
# A high power of t looks almost exactly like a combination of lower powers,
# which the method integrates perfectly. Only a small fraction of t^k is new at
# degree k: the fraction (k!)^2 2^k / (2k)!, about 4e-14 for k = 48. So the
# method's error on t^(degree + 1) is its Legendre miss times that fraction.
# When that product is below float64 rounding (about 1e-16), the power test
# cannot see the miss and reports a degree that is too high.
#
# The columns: the true degree; the degree a power test reports (tolerance
# 1e-12, searching up to 2 x nodes, the most any method can reach); the
# Legendre miss; and the power miss it implies, which is what the power test
# would have to detect.

print()
print(
    f"{'method':14s} {'degree':>6s} {'power test says':>15s} {'Legendre miss':>13s} ",
    end="",
)
print(f"{'power miss':>11s}")
for name, method in UNIFORM_METHODS.items():
    c = method.node_fractions()
    b = method.weights(c[None, :, None])[0][0, :, 0]
    x = 2 * c - 1
    power_degree = None
    for k in range(2 * len(c) + 1):
        exact = 1.0 / (k + 1) if k % 2 == 0 else 0.0  # integral of x^k over [0, 1]
        if abs(float(b @ x**k) - exact) > 1e-12:
            power_degree = k - 1
            break
    measured = method.measure_degree()
    k = measured.degree + 1
    share = torch.exp(  # (k!)^2 2^k / (2k)!, in logs to avoid overflow
        2 * torch.lgamma(torch.tensor(k + 1.0))
        + k * torch.log(torch.tensor(2.0))
        - torch.lgamma(torch.tensor(2 * k + 1.0))
    )
    legendre_miss = float(measured.errors[k])
    says = f">= {2 * len(c)}" if power_degree is None else str(power_degree)
    flag = "" if says == str(measured.degree) else "  <- wrong"
    print(
        f"{name:14s} {measured.degree:6d} {says:>15s} {legendre_miss:13.1e} "
        f"{float(share) * legendre_miss:11.1e}{flag}"
    )

# %%
# Result
# ------

print()
if failures:
    print(f"[FAIL] {failures} method(s) do not have the degree they claim.")
    raise SystemExit(1)
print("All methods have the degree they claim, and every error estimate has the")
print("strength the solver assumes.")
