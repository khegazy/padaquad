# `merge_rule="predicted"` costs time per distinct width ratio

Status: open (a speed cost, not an accuracy problem). Found 2026-11-15 by the code
review of `merge_rule="predicted"` (PR #69). Written so that another agent can pick
it up without the original session.

## Summary

To predict the error of merging two neighbouring subintervals, `"predicted"`
builds weight vectors for each distinct ratio of their widths
(`_merged_pair_weights` in `padaquad/quadrature/base.py`). That costs O(C · U) per
ratio (C nodes, U distinct node fractions): about 2 µs for gk21 and 20 µs for cc65
on 8 threads of a Perlmutter GPU node. Meshes grown by bisection from equal
starting subintervals have few ratios (1, 2 and 1/2 plus rounding variants), and
the vectors are cached on the solver. A starting mesh of many random subintervals
has one ratio per pair, its warm-started meshes keep them, and the cache is
bypassed when a chunk of pairs has more than 256 ratios. The CHANGELOG entry of
1.4.1 ("Cost") states this.

## Evidence (measured)

f = e^{-t} sin 40t + a Lorentzian, a cold call from `N_init_steps` random
subintervals, then 5 warm calls with `same_f=True` (median shown).

| Method, `N_init_steps` | `power_law` cold / warm | `predicted` cold / warm |
|---|---|---|
| gk21, 13 (default) | 8.5 / 2.0 ms | 6.5 / 2.4 ms |
| gk21, 10^4 | 23.8 / 4.4 ms | 92.6 / 11.2 ms |
| cc65, 10^4 | 26.6 / 6.5 ms | 244.8 / 31.3 ms |

Perlmutter GPU node, 8 CPU threads, 2026-11-15. On a login node the same 10^4
cases gave 15.9 against 4.9 ms (gk21) and 95.9 against 6.2 ms (cc65) per warm
call. Building `mesh_optimal` alone on 10^5 random subintervals took 1.9 s for cc65
with `"predicted"` and 0.002 s with `"power_law"`.

For an integrand that is expensive to evaluate the difference is small in
relative terms. Merged subintervals are evaluated again on the next call, so the
cost does not affect accuracy.

## Options (not decided)

1. Keep it documented (current state).
2. Apply the weights with two batched matrix products instead of forming the
   `[R, 2, C, U]` products: a prototype took 2.5 instead of 5.2 ms (gk21) and 44.6
   instead of 86.0 ms (cc65) per 1,024 ratios. Its weights differ by at most
   5.6e-17, so rounding-level merge decisions could change. It halves the cost
   but does not change how it grows with the number of ratios.
3. Above some number of distinct ratios, fall back to `"power_law"` for that call
   with a logged message. This changes results for such meshes.

Coarser quantization of the ratio is not an option: it moves the prediction by
more than the rounding floor.
