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
   fit, and C points are tried before it is concluded that they do not (also
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
