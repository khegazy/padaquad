# `_rec_remove` took time quadratic in the number of subintervals

Status: fixed on branch `improve_pruning` (2026-11-14), in the commit that adds this
file. Found while planning the merge-rule change in
[`predicted-merge-rule.md`](predicted-merge-rule.md); the cause was identified by its
adversarial review ([`predicted-merge-rule-review.md`](predicted-merge-rule-review.md),
Finding 2 and Section 4). Written so that someone who was not in the session can
follow it.

## Summary

Building `mesh_optimal` after an integration could take almost the whole call: 9.1 s
of a 9.5 s call with 10^5 subintervals, for gk21, cc65 and dopri5 alike. The time was
in `AdaptiveQuadrature._rec_remove`, which decides which flagged pairs of neighbouring
subintervals are merged. Its loop un-flagged about two entries per pass over the whole
mask, so a mask that was mostly True took about N/2 passes of O(N) work each.

It was first noticed as "dopri5 is slow" because the speed test used a tolerance at
which dopri5's pairs all qualify for merging and gk21's do not. It is not specific to
dopri5: any call in which most pairs qualify is affected, under any merge rule.

## Terms

- **Pair i**: subintervals i and i + 1. `_prune_excess_mesh` flags a pair when the two
  can be merged into one.
- **Mask**: the boolean vector of flags, one entry per pair. Two adjacent flagged pairs
  share a subinterval, so only one of them can be merged.
- **Run**: a maximal block of consecutive True entries in the mask.

## Evidence (measured)

1. **Profile.** dopri5, 10^5 equal subintervals on [0, 1], f = e^{-t} sin 3t,
   atol = rtol = 1e-30, `max_adaptive_splits=0`, 8 CPU threads, `main` at `2b04ea1`
   (`/pscratch/sd/k/khegazy/u028/review/profile_dopri5_100000.txt`):

   | Function | Self time | Cumulative time | Calls |
   |---|---|---|---|
   | `_rec_remove` | 6.03 s | 9.17 s | 1 (48,405 loop passes) |
   | `torch.any` inside the loop | 2.16 s | 2.16 s | 48,405 |
   | `torch.concatenate` inside the loop | 0.80 s | 0.80 s | 48,404 |
   | `_adaptively_increase_mesh` | 0.01 s | 0.12 s | 2 |

   Total 9.55 s; every pair was flagged, and `mesh_optimal` had 50,076 subintervals.
2. **Growth.** On an all-True mask the loop took 0.35 s at N = 10^4 and 3.2 s at
   4·10^4 (`/pscratch/sd/k/khegazy/u028/pred/rec_remove_check.py`); about 15 minutes
   are expected at 10^6.
3. **Not specific to a method.** At atol = rtol = 1e-8, where most pairs merge, building
   `mesh_optimal` from 10^5 equal subintervals took 9.13 s (gk21), 8.95 s (cc65) and
   9.15 s (dopri5) (`pred/speed_merge.py`).

## The fix

The loop's rule is: within each run, keep the first flagged entry and every second one
after it (TTT → TFT, TTTT → TFTF). That is a parity within runs, computable in one pass:

1. mark the first entry of each run (`mask & ~previous`);
2. carry the index of the latest run start forward (`torch.cummax`);
3. keep a flagged entry when its distance from that start is even.

Results:

- **Same output.** Identical to the loop on 4,956 random masks (lengths 0–39, 100,
  1,000 and 10,000; flag densities 0 to 1) in `pred/rec_remove_check.py`, and on 2,580
  masks in the new test `TestRecRemove::test_matches_former_loop`, which keeps the old
  loop as its reference.
- **Time.** 2 ms at N = 10^5 and 20 ms at N = 10^6 on an all-True mask.
- **Whole call.** Building `mesh_optimal` from 10^5 equal subintervals at
  atol = rtol = 1e-8 now takes 0.068 s (gk21), 0.101 s (cc65) and 0.064 s (dopri5),
  against about 9 s before.
- **Input.** The new version does not modify the mask it is given; the loop wrote to it
  in place.

## Why it matters for the merge-rule change

The planned default merge rule (rule T in `predicted-merge-rule.md`) makes more of the
merges that are safe, so more pairs are flagged and the loop would have run more often.
The fix is therefore made first, in its own commit.
