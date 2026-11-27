# Restore the test calls of `integrate` (`restore-test-calls`)

**Status:** design note, 2026-11-26, branch `restore_test_calls`, base
`a2b805c` (tip of `speedups_integration`). No plan existed; this note records
what the user decided, what is restored and how. Scratch:
`/pscratch/sd/k/khegazy/padaquad_speedup_plans/restore-test-calls/impl/`.

## What the user decided

On `main` (`d11df47`), `integrate` without `max_batch` (call or constructor)
and without `state` called `f` on 10, 100 and 1,000 copies of one point before
the first pass, stopping at the order of the first pass, only to find a batch
size that runs (`_initialize_max_batch`, decision 11 of the memory-state plan
of 2026-11-18). The `startup-evaluations` plan removed them (commit
`2d969f1`). The user's words now: "Bring back the test calls when max_batch is
not provided", and "If an OOM occurs with N=10 then multiply by 0.5 (instead
of 0.1). Otherwise, for N>10 multiply by 0.75", and "use 0.5 below 10 points".

The rule: when a test call of N points runs out of memory, the next test call
has `_shrink_batch(N)` points with the factor 0.5 if N <= 10 and 0.75 if
N > 10 (rounded as `_shrink_batch` rounds: half to even, always below N, at
least 1), until one runs; the batch size has then converged and no larger
test call follows. Out-of-memory retries inside the integration loop keep 0.75.

## What is restored, and what is not

- **Restored:** the test calls, exactly when no `max_batch` (call or
  constructor) and no `state` is given; sizes 10, 100, 1,000 while not above
  `min(first pass, 1,000)`; the batch size starts at the largest that ran
  (1 when the first pass is below 10 points), converged if one ran out of
  memory. `_LARGEST_TEST_CALL = 1000` comes back and `_LARGEST_FIRST_CALL`
  goes: the first call of the first pass is again at most 10 times the
  largest test call (10,000 points), which is what `_LARGEST_FIRST_CALL`
  encoded.
- **Not restored:** the one-point shape-check call (commit `1183318`); the
  output is checked in `_call_f`, which the test calls now go through as
  well. No option is added. The test calls do not go through the memory
  benchmark's `_measure_call` (it needs readers and times the call); they go
  through `_call_f`, the one place every call of `f` in the integration
  already goes through (out-of-memory recognition, release of memory, shape
  check).
- **The test point.** The first node of the first subinterval of the starting
  mesh (`_compute_nodes` of `mesh[0]`, `mesh[1]`, after `_setup_initial_mesh`,
  so after any cut of a given `mesh=` to the bounds; detached, so the test
  calls carry no graph of bounds that require grad). The integration evaluates
  this point in its first pass. For the Gauss-Kronrod methods it is inside
  the interval, so `f` is never called at a bound (an integrable singularity
  at a bound still works); for methods with a node at the start of a
  subinterval it is `mesh_init`, as on `main`. It also covers a `mesh=` given
  without bounds. No separate "probe" from the bounds is reintroduced. The
  copies are made with `repeat` (a real [N, T] tensor, like every call's
  input), not `expand`.
- **Where:** inline in the `else` branch of the batch-size block of
  `integrate` (base.py), about 10 lines with a comment; `_shrink_batch` gets a
  `factor` argument (default `_OOM_BACKOFF`). No new method.

## Hazards

1. **`take_gradient=True` and a batch below one subinterval.** The smallest
   test call allowed is C points with `take_gradient=True` (1 otherwise): a
   retry is `max(smallest, _shrink_batch(N, factor))`, and the call at that
   size is made with `can_shrink=False`, so if it runs out of memory
   `_call_f` raises `torch.OutOfMemoryError`, as the loop does today when one
   subinterval does not fit. So the chain never settles below C when C points
   fit (changed on 2026-11-26 by the user's decision, option B of the review:
   a failed test call of C points or fewer is not retried, `can_shrink = n >
   smallest`, so a failed 10-point call with C > 10 raises at once; the text
   that follows describes the design before that change), and C points are
   tried before it is concluded that they do not (also
   after a 10-point test call fails for a method with C > 10). `_call_f`'s
   message gets one clause saying that a call holds at least one point, and
   with `take_gradient=True` one subinterval of C points. The existing
   `ValueError` check (a given or a state's converged batch below C) is then
   unreachable from the test calls, so its two wordings stay correct and
   need no third. Tests: `gk21` with a limit of 21 (chain 10, 100, 75, 56,
   42, 32, 24, 21) and `cc65` with 70 integrate; a limit of 20 with `gk21`
   raises `torch.OutOfMemoryError` naming one subinterval; a limit of 9
   with `take_gradient=True` tries 10 then 21 and raises.
2. **Evaluations of `f`.** The test calls add exactly 10, 110 or 1,110 points
   (1, 2 or 3 calls) for a first pass of 10 to 99, 100 to 999, or 1,000 or
   more points, none below 10 points, none with `max_batch` or a `state`;
   nothing else is added. `main` added one more point (the shape check) on
   every call. `tests/test_memory_state.py` pins this for every start case,
   both `take_gradient`, with and without simulated out-of-memory errors.
3. **Numbers.** Without an out-of-memory error, the first call of the first
   pass is the same as on the base (`10 * _batch` exceeds the first pass up
   to 10,000 points; the call then raises `_batch` to its size), so every
   returned number is the base's, bit for bit, for a deterministic `f`. With
   an out-of-memory error in a test call, the batch size of the first pass
   is the largest test call that ran, converged, and with
   `take_gradient=True` the batches of the first pass are formed from it, so
   the tolerance reference of each batch changes (within the tolerance); this
   reproduces `main` except where `main` converged below C (it crashed) and
   where the 0.5 rule differs (test calls of at most 10 points that fail).
   Shown by the grid comparison against the base (no out-of-memory errors:
   identical) and by the out-of-memory cases against `main`.
4. **Random generator.** On `main` the order was: one-point shape check
   (`f`), then the random starting mesh, then the test calls, then the first
   pass. (The task message says the test calls ran before the mesh was drawn;
   they did not: `_setup_initial_mesh` came first, `base.py:1155` against
   `1173` at `d11df47`.) The test point needs the starting mesh, so the order
   here is mesh, test calls, first pass: `main`'s order without the shape
   check. A deterministic `f` gets the same mesh as on `main` and the base.
   An `f` that draws from the global generator gets the base's mesh (no call
   of `f` before it), not `main`'s (which followed the shape check); its
   draws in the first pass follow the test calls' draws, as on `main`.
5. **Memory benchmark.** Unchanged.

## Tests

- `tests/test_memory_state.py`: the test calls stop at 1,000 or the order of
  the first pass (both `take_gradient`); a failed test call shrinks by 0.75
  above 10 points and 0.5 at or below 10, until one runs; with
  `take_gradient=True` not below C (the one-subinterval test extended, and
  the error when C points do not fit); the points of `f` are exactly the
  integration's plus the test calls (every start case).
- `tests/test_interior_nodes.py`: the test calls use the first node, never a
  mesh point; a singularity at a bound converges without `max_batch`.
- Others that pinned "no call before the first pass" are found by running
  the suite.

## Implemented (branch `restore_test_calls`, base `a2b805c`)

- `21e0291`: this note.
- `7c7dd86`: the test calls, inline in the `else` branch of the batch-size
  block of `integrate` (`base.py`); `_shrink_batch(n, factor=_OOM_BACKOFF)`;
  `_LARGEST_TEST_CALL = 1000` replaces `_LARGEST_FIRST_CALL`; `_call_f`'s
  "cannot be reduced" message names the one-point and one-subinterval floors;
  tests (`test_memory_state.py`, `test_interior_nodes.py`, and docstrings or
  counts in `test_memory_benchmark.py`, `test_bug_regressions.py`,
  `test_memory_state_adversarial.py`, `test_methods_registry.py`).
- `5c5b8bd`: user guide (`devices_memory.md`, `troubleshooting.md`), README,
  quick start, CLAUDE.md, CHANGELOG 2.0.0 entries, docstrings of `integrate`,
  the constructor and `IntegrationState`.
- `ab383c7`: rewrap of `_call_f`'s docstring.

Measured (scripts and logs in the scratch directory above):

- **Evaluations** (`count_startup.py`, CPU; elementwise and tanh-network `f`,
  both `take_gradient`): extra points against the base 0 / 10 / 110 / 1,110 /
  1,110 / 1,110 for first passes of 8, 63, 231, 1,050, 10,500 and 42,000
  points (0 / 1 / 2 / 3 / 3 / 3 extra calls); 0 with `max_batch` or a
  `state`. `main` had 1 / 11 / 111 / 1,111 and 1 with `max_batch` or a state.
  Integral, error, contributions, state mesh and state batch size equal the
  base's in every case.
- **Bit-for-bit grid** (`startgrid.py`, 248 cases, 3,294 tensors: first
  passes in every range, gk21, cc17, bosh3, `interpolatory3_variable`,
  elementwise and network `f`, both `take_gradient`, warm starts, a given
  `max_batch`, `trade_speed_for_memory`, `record_evaluations=False`,
  float32): 0 differences against the base on CPU and on CUDA (A100), the
  calls of `f` identical after the test calls (160 cases without, 26 with
  10 points, 26 with 110, 36 with 1,110). No snapshot changed.
- **Out-of-memory chains** (`count_oom.py`, simulated limit): the branch
  reproduces `main`'s test-call chain without its one-point call above 10
  points (limit 60, gk21: 10, 100, 75, 56; batch 56 converged, also with
  `take_gradient=True`, where the base gave 42). It differs from `main` where
  the user's 0.5 rule applies (limit 9: batch 5, `main` 8; limit 4: 2,
  `main` 4) and where `main` crashed below C (limit 21, gk21,
  `take_gradient=True`: 21, `main` `AssertionError` at 18; cc65 limit 70: 65).
  The returned integral and error equal the base's in all these cases
  (`sin(40 t) + t`; the batching of `take_gradient=True` differed only in the
  limit-60 case, where both form batches of 42 points).
- **Random generator** (`rng_check.py`): a deterministic `f` gets the same
  starting mesh as `main` and the base; an `f` that draws from the global
  generator gets the base's starting mesh, and its later draws follow the
  test calls' draws, so its integral differs from the base's in the last bits.
- **Time** (A100, GPU 0 of nid001821, alone; medians of 21 calls, two rounds,
  base / branch / `main`): a cold default gk21 call without `max_batch`,
  `exp(-t) sin(40 t)`: 4.38, 4.22 / 4.57, 4.36 / 5.53, 5.52 ms; a 3×512 tanh
  network: 4.36, 4.33 / 5.04, 4.81 / 6.08, 5.91 ms (with
  `take_gradient=True` 6.05, 6.07 / 6.53, 6.45 / 8.11, 8.22 ms); a 6×2048
  network: 5.09, 5.13 / 6.15, 6.08 / 7.52, 7.43 ms, and from a mesh of 500
  subintervals 37.4 (one slow run; min 30.6), 30.9 / 34.1, 35.1 / 39.1,
  38.8 ms. With `max_batch` or a state the branch equals the base within
  0.1 ms (6×2048: 5.11, 5.07 against 5.12, 5.12 ms). So the test calls cost
  0.15 to 1 ms per cold call on the default mesh and 3 to 4 ms from a large
  mesh, against the base; the branch is faster than `main` everywhere (other
  changes since `main`, and no one-point call).
- **Suite** (nid001821, at `5c5b8bd`): base 4,744 passed / 27 skipped on
  CUDA and 4,388 / 383 on CPU; branch 4,748 / 27 and 4,393 / 382; no
  failures on either. Docs build (`-W`) passes.

## Not done

- No option to turn the test calls off (none was asked for).
- The halving below 10 points can leave a batch size well below what fits
  (an `f` that holds 9 points gets 5); this is the user's rule, documented
  under the user guide's limitations.
- The starting mesh of an `f` that draws random numbers cannot match `main`'s
  without the one-point call before the mesh, which stays removed.
