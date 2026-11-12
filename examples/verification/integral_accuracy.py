"""
Accuracy of complete integrations
=================================

This script checks the whole integrator, not just one method. For every
method, it integrates each test function in ``padaquad.integrand_dict`` over
[0, 1] at several tolerances, and compares the result with the exact answer.

What padaquad promises: it accepts a subinterval when its *estimated* error is
below the tolerance, ``max(atol, rtol * |integral|)``. The returned integral adds
up the accepted subintervals, so if every estimate were an upper bound on the
true error, the total error would be at most (number of subintervals) x
tolerance. That bound is what this script checks. It also prints the ratio of
the actual error to the tolerance, which is usually far below 1, because the
integral uses the higher-degree method while the estimate measures the
lower-degree one.

Run it with::

    python examples/verification/integral_accuracy.py

It exits with status 1 if any actual error exceeds the bound, except for the
known limitation of ``fehlberg2`` explained below, which is reported as
``[KNOWN]``.

Known limitation: ``fehlberg2``'s error estimate ignores curvature
------------------------------------------------------------------

``fehlberg2`` integrates with the weights [1/512, 255/256, 1/512] and estimates
its error as the difference from the lower-degree weights [1/256, 255/256, 0].
On a curved integrand both weight sets make almost the same error: on t^2 over
[0, 1] they give 0.25098 and 0.24902, where the exact value is 1/3. Their
difference, which is the estimate, cancels that shared error and keeps only
(f(end) - f(start)) / 512 times the width. Where the integrand bends more than
it rises, the estimate is far too small: on ``damped_sine`` at tolerance 1e-6,
subintervals with an estimated error near 1e-7 have a true error near 1e-4.
The estimate does become reliable on small enough subintervals (it shrinks like
h^2, the true error like h^3), so tight tolerances are less affected. Prefer
``bosh3``, ``dopri5`` or the Gauss-Kronrod and Clenshaw-Curtis methods when the
error estimate matters.
"""

# %%
# Integrate every test function with every method
# ------------------------------------------------
#
# Each run uses float64, the CPU and a fixed batch size, and the random starting
# mesh is seeded, so the output is the same on every machine.

from __future__ import annotations

import torch

from padaquad import UNIFORM_METHODS, VARIABLE_METHODS, integrand_dict, integrate

TOLERANCES = (1e-6, 1e-10)  # used for both atol and rtol
mesh_init = torch.tensor([0.0], dtype=torch.float64)
mesh_final = torch.tensor([1.0], dtype=torch.float64)

runs = [("uniform", name) for name in UNIFORM_METHODS]
runs += [("variable", name) for name in VARIABLE_METHODS]

# Methods whose error estimate is known not to bound the error (see above).
KNOWN_LIMITATIONS = {"fehlberg2"}

failures = 0
known = 0
worst_ratio = 0.0
print(
    f"{'method':24s} {'sampling':8s} {'tolerance':>9s} {'worst error/tolerance':>21s} ",
    end="",
)
print(f"{'subintervals':>12s}")
for sampling, name in runs:
    for tol in TOLERANCES:
        worst = 0.0
        n_sub_at_worst = 0
        for f, solution, _ in integrand_dict.values():
            torch.manual_seed(0)
            result = integrate(
                f,
                method=name,
                sampling=sampling,
                atol=tol,
                rtol=tol,
                mesh_init=mesh_init,
                mesh_final=mesh_final,
                device="cpu",
                max_batch=100_000,
                take_gradient=False,
            )
            exact = float(solution(mesh_init, mesh_final))
            got = float(result.integral)
            n_sub = len(result.h)
            tolerance = max(tol, tol * abs(got))
            error = abs(got - exact)
            bound = n_sub * tolerance
            if error > bound:
                status = "KNOWN" if name in KNOWN_LIMITATIONS else "FAIL"
                known += status == "KNOWN"
                failures += status == "FAIL"
                print(
                    f"  [{status}] {name} on {f.__name__}: error {error:.1e} > bound "
                    f"{bound:.1e} ({n_sub} subintervals x tolerance {tolerance:.0e})"
                )
            if error / tolerance >= worst:
                worst, n_sub_at_worst = error / tolerance, n_sub
        note = "  (known limitation)" if name in KNOWN_LIMITATIONS else ""
        if not note:
            worst_ratio = max(worst_ratio, worst)
        print(
            f"{name:24s} {sampling:8s} {tol:9.0e} {worst:21.2e} {n_sub_at_worst:12d}{note}"
        )

# %%
# Result
# ------

print()
print(
    f"Largest actual error / tolerance, known limitations excluded: {worst_ratio:.2e}"
)
if known:
    print(
        f"[KNOWN] {known} integration(s) of a method with a known limitation exceeded"
    )
    print("        the bound (see the explanation at the top of this script).")
if failures:
    print(f"[FAIL] {failures} integration(s) exceeded the error bound.")
    raise SystemExit(1)
print("Every other integration is within (number of subintervals) x tolerance.")
