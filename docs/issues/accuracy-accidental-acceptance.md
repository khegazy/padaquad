# Unresolved subintervals accepted because the error estimate is small by chance

Status: open. Observed by the planning of `split-waste` (2026-11-25, at
`0590ccf`), re-checked on branch `split_waste` (base `b43da3d`, 2026-11-26)
over ten random starting meshes. Nothing is fixed here.

## Summary

A call can return `converged=True` with an integral far outside the requested
tolerance. In each case below, a subinterval on which the integrand is far
from resolved is accepted because its error estimate (the difference between
the method and its embedded lower-degree method on the same nodes) happens to
be below the tolerance. Whether it happens depends on the random starting
mesh, so it shows up for some seeds and not others:

| Integrand on [0, 1] | Method | `atol` | Seeds (of 10) with error > 10 `atol` | Error then |
|---|---|---|---|---|
| sin(2e5 t) | gk21 | 1e-6 | 8 | 1.6e-6 to 2.8e-3 |
| sin(2e5 t) | gk21 | 1e-8 | 6 | 1.9e-7 to 2.2e-5 |
| sin(2e5 t) | gk21 | 1e-10 | 0 | |
| Gaussian peak of width 1e-4 | gk21 | 1e-6, 1e-8, 1e-10 | 10 (peak missed) | 2.5e-4 (the whole peak) |
| Gaussian peak of width 1e-4 | gk31 / cc33 | 1e-6, 1e-8, 1e-10 | 9 / 10 (peak missed) | the whole peak |
| Gaussian peak of width 1e-3 | gk7 / gk15 / gk21 / cc17 | 1e-8 | 8 / 5 / 2 / 4 (peak missed) | 2.5e-3 (the whole peak) |
| jump at 0.3183… | gk7 | 1e-6 / 1e-8 / 1e-10 | 1 / 2 / 2 | 1.9e-4 (seed 0), 1.4e-7 (seed 1) |

Every such call checked (those of the code below, `atol=1e-8`) reports
`converged=True`: the final check judges each accepted subinterval by the same
error estimate.

## Evidence

Common setup: `rtol=0`, float64, CPU, `take_gradient=False`,
`max_batch=1_000_000`, a fresh random starting mesh of 11 subintervals,
seeds 0 to 9, `split_rule="halve"` (the default). The Gaussian peaks are
`exp(-(t - A)^2 / (2 s^2))` with `A = 0.3183098861837907`, the jump is
`(t > A)`. Exact integrals: `(1 - cos(2e5)) / 2e5`, `s sqrt(2 pi)` (to 1e-30
on [0, 1]) and `1 - A`.

```python
import math
import torch
from padaquad import integrate

A = 0.3183098861837907
lo = torch.tensor([0.0], dtype=torch.float64)
hi = torch.tensor([1.0], dtype=torch.float64)
cases = {
    "sin(2e5 t), gk21": (
        lambda t: torch.sin(2e5 * t),
        "gk21",
        (1 - math.cos(2e5)) / 2e5,
    ),
    "peak 1e-4, gk21": (
        lambda t: torch.exp(-0.5 * ((t - A) / 1e-4) ** 2),
        "gk21",
        1e-4 * math.sqrt(2 * math.pi),
    ),
    "jump, gk7": (lambda t: (t > A).to(t.dtype), "gk7", 1 - A),
}
for name, (f, method, exact) in cases.items():
    for seed in range(10):
        torch.manual_seed(seed)
        r, _ = integrate(
            f,
            method,
            atol=1e-8,
            rtol=0.0,
            mesh_init=lo,
            mesh_final=hi,
            device="cpu",
            take_gradient=False,
            max_batch=1_000_000,
        )
        print(
            name,
            seed,
            f"{abs(float(r.integral) - exact):.1e}",
            r.converged,
            len(r.h),
            r.max_split_depth,
        )
```

What the runs show (the full sweep over tolerances and methods, and the
per-subinterval errors, came from two scripts in a scratch directory that was
deleted on 2026-11-26; the code above reproduces the `atol=1e-8` rows, and the
other rows change only `atol`, the method or the integrand):

1. **sin(2e5 t), gk21, `atol=1e-8`, seed 0**: error 2.2e-5 over 13,036
   subintervals. Against the exact integral of each subinterval (mpmath, 40
   digits), two subintervals carry all of it (1.5e-5 and 7.3e-6). Both are
   2.14e-3 wide: 68 periods of the integrand sampled by 21 nodes (0.31 nodes
   per period). Their error estimates are 9.9e-9 and 5.0e-9, below `atol`.
   Every other subinterval's error is below 1e-8.
2. **Peak of width 1e-4, gk21**: in 9 of 10 seeds no subinterval of the first
   pass fails (11 subintervals, 0.09 wide on average; gk21's nodes are 1e-3
   to 7e-3 apart there, 4e-4 across a mesh point, so the peak, 2.4e-4 wide at
   half height, falls between nodes), the call ends after one pass, and the integral is about 0. In the
   tenth (seed 8) one subinterval is split once and the peak is missed by its
   halves. The same holds at every tolerance from 1e-6 to 1e-10.
3. **Peak of width 1e-3**: missed (one pass, integral about 0) in 2 of 10 seeds
   with gk21 at `atol=1e-8` (seeds 3 and 9), more often with fewer nodes.
4. **Jump, gk7, seed 0**: 15 subintervals, depth 4; the subinterval holding the
   jump is accepted with an error of 1.9e-4. With the other seeds the jump is
   refined 14 to 22 levels deep.

### A related case: 100 points of infinite derivative (2026-11-26)

`f(t) = sum_j sqrt(|t - c_j|)` with 100 positions `c_j = frac(0.5 + j g)`,
`g = (sqrt(5) - 1)/2`, on [0, 1]; exact integral from the antiderivative
`(2/3) sign(t - c) |t - c|^1.5`. `gk21`, `atol=1e-8`, the common setup above
(default `error_norm`), seeds 0 to 9, PadaQuad 2.0.0 (`f01df5c`), on a
Perlmutter CPU node. Every call reports `converged=True`:

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Error | 1.3e-6 | 1.2e-6 | 1.7e-6 | 2.0e-6 | 1.5e-6 | 1.1e-6 | 1.3e-6 | 6.9e-6 | 7.0e-7 | 1.2e-6 |
| Subintervals | 635 | 630 | 643 | 630 | 635 | 625 | 628 | 635 | 653 | 623 |

The errors are 70 to 690 times `atol`. Unlike the cases above, most of this
is not shown to be an accidental acceptance: each subinterval is tested on its
own, so the total may reach the number of subintervals times `atol` (about
6.3e-6 here), and nine of the ten errors are below that. Seed 7 (6.9e-6) is
above it. Whether its excess comes from a subinterval accepted by chance was
not checked (it would need each subinterval's actual error against its
estimate). Before 2.0.0 (benchmark branch, 2026-11-24) the same integrand gave
3e-5 with one seed.

## Why it happens

The error estimate of every method is the difference between two rules of
different degree on the same nodes ({cite:p}`gonnet2012`, Sec. 2.5, the
category of estimates PadaQuad's methods belong to). The following is my
reasoning, not taken from a paper: both rules see the integrand only at the
nodes. When the integrand varies on a scale shorter than the node spacing,
the samples alias it: both rules integrate the same few samples of a function
neither resolves, and their difference has no reason to be as large as their
error; on some subintervals it is small by chance (case 1). When a feature
lies entirely between nodes, both rules see nothing and agree exactly (cases 2
and 3); no estimate built from the values at those nodes can see it. The
random starting mesh decides where the nodes fall, hence the dependence on
the seed. Case 4 is the first mechanism at a discontinuity, with only 7
nodes.

`split_rule="accelerate"` changes which subintervals are evaluated and so
avoids some of these cases by chance (sin(2e5 t) at `atol=1e-8`: 4 seeds of 10
instead of 6; the jump with gk7: 1 and 0 instead of 2 at `atol` 1e-8 and
1e-10); it misses the peaks exactly as halving does. It is not a remedy.

## Options

None is implemented; each costs evaluations of `f` on integrands that do not
need them, which is why they are options for the user to decide.

1. **Document it** (user guide, `ug-convergence`): `converged=True` means every
   subinterval passed its own error estimate, not that the integral is within
   the tolerance; integrands with features narrower than the node spacing of
   the starting mesh need a finer starting mesh (`N_init_steps`, or a `mesh`
   placed with knowledge of the feature), and oscillatory integrands a smaller
   tolerance or a higher-degree method. Costs nothing.
2. **A minimum depth**: split every starting subinterval at least k times
   before accepting it. Costs about 2^k times the points on easy integrands
   and does not help case 2 unless 2^k raises the node density enough.
3. **A second, independent estimate on accepted subintervals**, for example
   comparing a subinterval's result with the sum of its halves' (as adaptive
   Simpson does), only for subintervals accepted in the first passes. Costs
   the halves' evaluations on every such subinterval.
4. **A finer default starting mesh**: the `split-waste` plan measured 5 times
   the points of an easy integrand at `N_init_steps=50` and 90 times at 1000
   (its section 2.5); rejected there as a default.
5. **Detect unresolved subintervals from the values**, for example from how
   the method's interpolant's coefficients decay: an idea, not checked
   against any paper or measurement here.

## References

- `gonnet2012` in `docs/references.bib`, Sec. 2.5: error estimates from the
  difference of two rules of different degree.
