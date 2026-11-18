# Critique of "Making the memory measurement fast: brainstorm"

Adversarial review of `docs/issues/2026-11-17-memory-benchmark-speed-brainstorm.md`
(the "brainstorm"), written 2026-11-17 on branch `fix_warm_start` (commit 5af0d76).
Every claim below was checked against the raw logs and JSON in
`/pscratch/sd/k/khegazy/padaquad_membench/`, the code, git history, or a page I
opened. My check scripts are in `/pscratch/sd/k/khegazy/padaquad_membench/critique/`
(`calls.py`, `stopsim.py`, `replay.py`, `fits.py`, `share.py`, `scipy_import.py`,
`reset_cost.py`, `partial_grad.py`, `nnls_subsets.py`). No GPU node was allocated.

Terms. A **test call** is one call of f inside the memory benchmark
(`_setup_memory_checks`). The **stop rules** are the five reasons the benchmark's
search ends (`memory.py:579-589`). **E** is the reader's fixed reading-error
allowance on Linux, `E = 2 · (CPUs in the affinity mask) · max(32, 2 · online CPUs)`
pages (`memory.py:1152`); on a 128-CPU node it is 256 MiB. The **oracle** run is
`time_hard.py`'s "integration alone", a call with `max_batch=result.max_batch`.

## 1. Summary

I report 18 findings: 2 wrong, 3 unsupported, 5 misleading, 6 incomplete and 2
minor (units, a citation), plus 18 claims I checked and found correct (section 4).

The three most serious:

1. **The mechanism is wrong on the CPU.** The test calls are not "sized to fill the
   machine's memory": a replay of the search on the logged readings shows that all
   14 CPU searches ended because the next call was predicted to take more than 1 s
   (12 runs) or hit the 1e9 cap (2 runs), using at most 17.9 GiB of about 428 GB
   usable. The convergence rule never fired, because the cost per point it tests
   includes E (256 MiB here), which a cheap integrand dilutes only at hundreds of
   millions of points.
2. **Option D is recommended on evidence that cannot tell it apart from the
   cheaper options**, and its design misses that in the `take_gradient=True` and
   `conserve_memory=True` paths a pass is one batch, so growing the batch from 128
   points changes which subintervals are judged against which running integral,
   and therefore changes the mesh and the result (the re-check that would correct
   this is off for `take_gradient=True`).
3. **The CPU "reader slows the next call by 18 to 23 %" result is unsupported.**
   The two loops in `checks.py` ran one after the other, 7 times each, on a node
   shared with a GPU job; an interleaved re-check found no slowdown; and the stated
   cause (pages returned by `malloc_trim` must be mapped again) cannot apply to
   80 MB and 400 MB tensors, which glibc allocates with `mmap` and returns on every
   free anyway.

**Does the recommendation survive?** Only in part. Removing the 0.05 s sleep at
the end of the benchmark (part of A.1) survives and is supported by git history.
Removing the garbage collection there too does not. Dropping scipy (A.2) is sound.
"Design D as the default" does not follow from the evidence: the same test set is
explained by a 1 s per-call time limit and an E-dominated convergence test, and a
change to those stop rules (a few lines, option B2, which the brainstorm dismissed
without measurement) should be measured first.

## 2. Findings, most severe first

### F1. Wrong: the cause of the long CPU search

- **Claim** (Summary): "It is spent at the end of the search, in test calls sized
  to fill the machine's memory (up to 4.8e8 points and 14.3 GB for an integral that
  needed 253 points)". Section 2.3, item 1: "The search needs three readings above
  the floor and then two calls with a stable cost per point, so a cheap integrand
  is pushed to hundreds of millions of points."
- **Verdict:** wrong (for the CPU); incomplete (for the GPU).
- **Evidence.**
  - `replay.py` re-runs the stop logic of `memory.py:784-928` on the logged
    readings of all 14 CPU runs in `cpu.json`, using padaquad's own
    `_fit_up_to_quadratic` and `_largest_n`. With E = 256 MiB or 128 MiB it
    reproduces every logged sequence of N exactly, and every search ends by stop
    rule 5 ("N_next < 2N"): 12 because `n_time` (calls predicted at most 1 s) fell
    below 2N, 2 (`sine_squared`) because the next doubling passed the 1e9 cap.
    Stop rule 2 (cost per point stable) fired in none of them. With E = 0 the
    replay contradicts 3 logs (`damped_sine` 1e-6 would have stopped at 240.7
    million points, not 481.5 million; both `sine_squared` runs at 400 million,
    not 800 million), so E is what kept rule 2 from firing.
  - The reason: rule 2 tests `c = s (M + E) / N` (`memory.py:810`, `851-856`).
    For `damped_sine`, M ≈ 30.5 bytes per point (14,694 MiB at 481,464,000
    points, `cpu.json`), so E = 256 MiB is 1 % of M only at about 880 million
    points. Until then c keeps falling by more than 1 % per doubling.
  - The largest CPU call used 17.9 GiB (`sine_squared`, 800 million points,
    18,313 MiB). The usable memory implied by the stored `max_batch` values of the
    `mlp_512x3` runs is 427.8 GB (`fits.py`), so the largest call used about 4 %
    of it. Memory was not the limit.
  - The structure is fixed by the code: after a call longer than 0.1 s the step is
    2 (`memory.py:901`), and calls stop at a predicted 1 s (`memory.py:884`), so a
    cheap CPU search ends with calls of roughly 0.1, 0.2, 0.4 and 0.8 s. That is
    why the CPU benchmark took 0.64 to 1.78 s of test calls "whatever the
    integral".
  - On the GPU (E = 0) rule 2 fired in all 14 runs (`replay.py cuda`). The two
    long GPU searches were stretched by one reading out of proportion: for
    `wolf_schlegel` the reading at 1e6 points was 47.26 MiB where the neighbours
    give 45.78 MiB (+3.2 %), and for `mlp_512x3` (tol 1e-6) the reading at 340
    points was 5.21 MiB where the neighbours give 3.99 MiB (+30 %). Rule 2 needs
    two consecutive changes below 1 %, so one off reading costs two more 10×
    calls; without it `wolf_schlegel` would have stopped at 1e7 points (0.003 s of
    calls) instead of 1e9 (0.26 to 0.89 s). This is my reading of the logged
    numbers.
- **Proposed statement:** "On the 128-CPU node every CPU search ended because the
  next call was predicted to take more than 1 s (or passed 1e9 points), using at
  most 4 % of the memory. The convergence test never fired, because the cost per
  point it compares includes the fixed reading error E (256 MiB here), which a
  cheap integrand dilutes only near 1e9 points. On the GPU, the long searches came
  from one reading out of proportion, which resets the two-call convergence test."
  Add that E grows with the square of the CPU count, so on a laptop (E of a few
  MiB) the CPU search would stop much earlier; the 0.7 to 1.8 s is specific to
  many-core nodes.

### F2. Unsupported: option D as the default, and its missing coupling to results

- **Claim** (Summary, section 4): "Recommendation: A now, then D." and "Design D
  as the default".
- **Verdict:** unsupported, with an important missing risk.
- **Evidence.**
  - The emulation shows only that skipping the benchmark saves the benchmark. On
    this test set no limit was ever reached (the brainstorm says so in 2.4 and
    question 6), so options B2 and C would show the same saving, and F1 shows the
    CPU cost is set by two stop rules that a few lines could change.
  - **Batch size changes the result outside the default path.** In the default
    path (`take_gradient=False`, `conserve_memory=False`) a pass evaluates every
    waiting subinterval, in calls of at most `max_batch` points
    (`_evaluate_f_on_split_nodes`, `base.py:2091-2170`), so the batch size only
    changes how the pass is chunked, and the emulation (which only covered this
    path) could not see any effect on the mesh. With `take_gradient=True`
    (`_evaluate_f_on_full_nodes`, `base.py:1918-1988`) and with
    `conserve_memory=True` (`base.py:2172-2330`) a pass is one batch. Starting D
    at 100 to 128 points means the first passes hold 4 to 5 of the 11 starting
    subintervals of gk21; in absolute mode their tolerance is
    `max(atol, rtol·|running integral|)` (`base.py:992-1007`), computed from an
    incomplete integral. The re-check of accepted subintervals that corrects this
    drift is disabled for `take_gradient=True` (`base.py:1032-1034`). The project
    memory note "take_gradient changes batching → changes the denominator" records
    the same effect. D would therefore change meshes, evaluation counts and,
    when the early running integral is larger than the final one, could accept
    subintervals too loosely; `converged` would flag it only afterwards.
  - **Convergence as designed would not happen on the CPU.** D's item 3 stops
    measuring when "the fit is ready by today's rule (... cost per point changed
    by less than 1 % twice)". On the measured node that rule fired in 0 of 14 CPU
    runs (F1); under D the readings come from calls of varied sizes, which makes
    a two-consecutive-call test less stable still. So on many-core CPUs D would
    either read memory around every call for the whole integration or stop only at
    the memory budget, which a cheap integral never reaches.
- **Proposed statement:** "D removes the benchmark but couples the batch size to
  the result in the `take_gradient=True` and `conserve_memory=True` paths, and its
  convergence rule must be redesigned (the present one does not fire on many-core
  CPUs). Before designing D, measure B2 (a smaller per-call time limit, and a
  convergence test without E) and C on the same scripts; they need far less code."

### F3. Unsupported: the CPU reader slows the next call by 18 to 23 %

- **Claim** (D, item 5): "the call after it was slower, because the returned memory
  must be mapped again: 2.74 to 3.38 ms for sin on 1e7 points (+23 %) and 271 to
  320 ms for mlp on 1e5 points (+18 %) ... This is why measuring must stop at
  convergence."
- **Verdict:** unsupported; the mechanism is unlikely for these sizes.
- **Evidence.**
  - Design of `checks.py:22-35`: 3 warm-up calls, then all 7 plain calls, then all
    7 calls after a reset. No interleaving, so any drift (the CPU and GPU scripts
    ran at the same time on the node, brainstorm 2.1) lands entirely on the second
    loop. Seven samples per arm, medians only, no spread reported. The plain loop
    calls `sync()` twice, which is a no-op on the CPU, so that difference does not
    matter.
  - Mechanism: glibc serves any request above its mmap threshold with `mmap` and
    returns it with `munmap` on `free`; the threshold can rise dynamically but not
    above `DEFAULT_MMAP_THRESHOLD_MAX` = 4·1024·1024·sizeof(long) = 32 MiB on 64-bit
    systems (`man mallopt`, section `M_MMAP_THRESHOLD`, read on this system). The
    tensors here are 80 MB (1e7 float64) and about 410 MB (1e5 × 512 float64), so
    they are mapped fresh and faulted in on every call with or without
    `malloc_trim`. Trimming can only affect allocations below 32 MiB.
  - Re-check (`reset_cost.py`, login node, 8 threads, noisy): calls interleaved in
    alternating order, with no reset, `malloc_trim` only, `clear_refs` only, and
    both. `sin` on 1e7 points, 40 calls per arm: medians 12.03, 12.28, 12.60 and
    12.75 ms, interquartile ranges overlapping (11.8 to 14.0 ms). MLP on 1e5
    points, 15 per arm: 1386, 1364, 1404 and 1380 ms. No slowdown beyond noise.
    Different machine and thread count from the original, so this does not
    disprove a slowdown on the GPU node; it shows the original result is not
    robust.
- **Proposed statement:** "A reset costs 0.16 to 0.34 ms (measured). Whether it
  slows the next call is not established: the one measurement ran the two cases
  in sequence, 7 times each, on a shared node, and an interleaved re-check on a
  login node found no difference. Large tensors are mapped fresh on every call
  regardless of the trim." Measuring should still stop at convergence, but for the
  reading's own cost and its side effects (see section 3), not for this reason.

### F4. Incomplete: "I did not investigate the cause" of the unstable limits

- **Claim** (2.4): "In the first two pairs both searches made the same test calls,
  so the readings or the free memory differed ... I did not investigate the cause."
- **Verdict:** incomplete; the cause is in the JSON.
- **Evidence** (`fits.py`, refitting the reliable readings with padaquad's
  `_fit_up_to_quadratic` and solving the stored `max_batch` for the usable memory):
  - Free memory did not differ: the implied usable memory is 427.8 GB in both CPU
    `mlp_512x3` runs and 70.8 GB in all four GPU `mlp_512x3` runs.
  - The readings differed slightly, and the fitted N² coefficient d amplified it.
    CPU: the reading at 200,000 points was 2,673 MiB in one run and 2,346 MiB in
    the other (14 %); d came out 1.7e-2 bytes per point² against 0, and
    `max_batch` 4,182,912 against 27,993,951. GPU with `take_gradient=True`: the
    reading at 1,000 points was 15.797 against 15.641 MiB (1 %); d came out 0.188
    against 0.0052 bytes per point², and `max_batch` 338,878 against 1,199,344.
    A fit over N ≤ 1,000 is extrapolated by a factor of about 1,000.
  - Third pair (GPU, 4,636,742 against 2,881,525): in the first run a slow call at
    N = 10 (0.083 s of compute) cut the next step to 34 points (the worst-case time
    rule, 10·sqrt(1 s / 0.083 s) ≈ 34.8), and the off reading at 340 points (F1)
    delayed convergence to 3.1 million points. In the second run the readings at
    10, 100 and 1,000 points were proportional and rule 2 fired at 1,000.
- **Proposed statement:** "Free memory was the same; the readings differed by 1 to
  14 %, and the N² term of a fit extrapolated far beyond its readings turned that
  into a factor 3.5 to 6.7 in `max_batch`." This matters for D, whose item 2 reuses
  this fit unchanged.

### F5. Incomplete: the one-time cost of the first benchmark is not only scipy

- **Claim** (2.2 footnote, 2.3 item 3, A.2): the first benchmark's extra time is
  "the one-time scipy import (0.58 s)".
- **Verdict:** incomplete. The scipy import itself is confirmed (section 4).
- **Evidence.** Time inside the first benchmark that is neither test calls nor
  `_release_memory`:
  - `hard_cpu.log`, sin(1e3 t): 1.311 − 0.676 − 0.049 = 0.586 s. This matches
    scipy.
  - `hard_cuda.log`, sin(1e3 t): 1.271 − 0.004 − 0.103 = 1.164 s. Scipy explains
    half; about 0.58 s is unexplained.
  - `cpu.json`, first row (`damped_sine`, 1e-6): cold call minus test calls minus
    warm integration = 3.044 − 1.596 − 0.0035 = 1.445 s, against 0.036 to 0.069 s
    in later rows; about 0.8 s is unexplained after scipy.
  - `cuda.json` first row: 0.971 s by the same subtraction, against 0.097 to
    0.400 s in later rows.
  - `cpu.json` also shows 0.816 s in the first `take_gradient=True` row
    (`mlp_64x2_grad`, 1e-6), against 0.039 s in its repeat; this is probably the
    first backward pass of the process and belongs to the integration (conjecture).
- **Proposed statement:** "The first benchmark of a process carried 0.59 to 1.45 s
  of one-time cost in the four logs, of which the scipy import is 0.58 s; the rest
  was not attributed." A.2 then removes only part of it on the GPU.

### F6. Misleading: the ranges in the Summary come from one script only

- **Claim** (Summary): "On a CPU node the benchmark took 0.7 to 1.3 s ... 0.9 to
  118 times the integration ... On a GPU it took 0.09 to 0.66 s".
- **Verdict:** misleading. All numbers match `hard_cpu.log` and `hard_cuda.log`
  exactly, but `time_benchmark.py` (`cpu.json`, `cuda.json`) is ignored.
- **Evidence** (`share.py`): in `cpu.json`, test calls alone (without the release,
  the fits or the one-time costs) took 0.637 to 1.783 s, and 109 to 525 times the
  integration (integrations of 253 points, 3.4 to 8.6 ms). In `cuda.json`, test
  calls alone took up to 0.926 s (`wolf_schlegel` 1e-10), which the brainstorm
  itself quotes in 2.3 ("0.3 to 0.9 s") but not in the Summary.
- On the suspicion that 118× is inflated by the scipy import: it is, but little.
  Without the 0.586 s one-time cost, sin(1e3 t) gives (1.311 − 0.586) / 0.0111 =
  65×, and the largest ratio among the other runs is 99× (mlp(300 t)). The other
  script gives larger ratios, so the headline understates rather than overstates.
- **Proposed statement:** "The CPU benchmark took 0.6 to 1.8 s in 20 runs of two
  scripts, 1 to 525 times the integration; on the GPU 0.09 s to about 1 s, of
  which about 0.09 s is the fixed end cost." State that the "12 runs" are
  `time_hard.py` and that 28 more runs (`time_benchmark.py`) agree with section
  2.4's first claim.

### F7. Misleading: the option D table

- **Claim** (D, emulation table and Summary): "ran the whole integration in 0.02 to
  1.6 s on the CPU where benchmark plus integration took 0.8 to 2.7 s"; "On
  sin(3e6 t) the emulation was slower than the integration alone".
- **Verdict:** numbers confirmed; framing misleading.
- **Evidence.**
  - Every number in the table matches `hard_cpu.log` / `hard_cuda.log`.
  - Against the oracle integration (a fairer baseline for the cost of D itself),
    the emulation was slower in 5 of 6 CPU runs, not only sin(3e6 t): 1.73×,
    0.81×, 1.32×, 1.52×, 1.58×, 1.57× (`hard_cpu.log`, ramp ÷ oracle). On the GPU
    0.86× to 1.20×, inside the stated noise.
  - The "today" column is one cold run; the D column is the median of 3 warm runs
    with different random starting meshes. In absolute terms the bias is small
    against a benchmark of about 1 s (cold minus benchmark equals the oracle
    within 3 ms on the CPU for sin(1e5 t)), so the conclusion "D saves the
    benchmark" stands; the per-row speed-ups are not like for like.
  - `ramp_batches` (`time_hard.py:26-46`) implements the stated rule (start 128,
    grow by min(10 B, B·sqrt(budget / (M + E))), freeze below 2 B), but it reads
    memory around every call even after freezing, skips `_split_f_output` and
    `_check_f_output_finite` (which the real loop runs on every call,
    `base.py:2073-2077`), would fail on an integrand that returns tracked
    variables (`len(out)`, `out.nbytes` on a tuple), has no out-of-memory retry,
    ignores `_oom_max_batch`, fixes `total_mem_usage` at 0.9, and covers only the
    default path. It also differs from the design in item 2 (it uses the N²
    worst case from the last reading, not the fit, and never tests convergence).
    The brainstorm states "Only the take_gradient=False path" but not the rest.
- **Proposed statement:** "On this test set the emulated D removed the benchmark;
  its integration was 1.3 to 1.7 times slower than the oracle on the CPU in 5 of 6
  integrals (partly the reading on every call), and it emulates only the default
  path without tracked variables, out-of-memory handling or the convergence rule."

### F8. Incomplete: option A.1 also drops the garbage collection

- **Claim** (A.1): "End the benchmark with `torch.cuda.empty_cache()` only; keep the
  garbage collection and the sleep for the out-of-memory path, which is what the
  sleep's comment is about. Saves about 0.08 s per benchmark on a GPU."
- **Verdict:** the sleep part is confirmed and better supported than the brainstorm
  says; the garbage-collection part is unsupported.
- **Evidence.**
  - `git log -S"time.sleep(0.05)"`: the sleep entered in 1d73ffe and 4dc0b1d,
    inside the out-of-memory handlers only, with the comment about a faulted
    stream. Before commit 3c5823a ("refactor: merge the split handlers' batch loop
    and the OOM cleanup", 2026-11-08, described as "No change in results"), the
    normal end of the benchmark ran `gc.collect()` and `torch.cuda.empty_cache()`
    with no sleep (`git show 3c5823a^:padaquad/quadrature/base.py`, lines
    2857-2859). The refactor routed that end through `_release_memory`, which
    added the sleep as a side effect.
  - The garbage collection at the end has been there in every version. It frees
    reference cycles that hold f's tensors, which `empty_cache` cannot release
    and which the following `_get_usable_memory` would count as used. Dropping it
    saves 0.025 to 0.04 s (the brainstorm's own numbers) for that risk; nothing in
    the evidence tests it.
- **Proposed statement:** "Restore the pre-3c5823a end of the benchmark: garbage
  collection and `empty_cache`, no sleep (the sleep was added there by a
  refactor). Saves 0.05 s per GPU benchmark." Keep the 0.08 s figure only if the
  collection is shown to be unnecessary.

### F9. Wrong: option C's description of existing behavior

- **Claim** (C): "readings this small are below the CPU noise floor, so no fit is
  possible and only the existing worst-case rule applies (memory grows at most like
  N² from the last reading). That rule gives a limit of millions of points for a
  cheap integrand, and a cautious one for an expensive integrand."
- **Verdict:** wrong about the code; the safety claim is unsupported.
- **Evidence.** When no reading reaches the floor, today's benchmark raises
  `RuntimeError` ("No test call of f used at least the noise floor", `memory.py:
  945-950`); the worst case from an unreliable reading is used only to size the
  next test call (`memory.py:780-782`). On the CPU every logged reading at N ≤ 253
  is below the 4 MiB floor (`cpu.json`), and such readings are often just the
  input-plus-output bytes (`_tensor_bytes`, limitation 4). For example,
  `mlp_512x3` at 100 points read 0.002 MiB, about 21 bytes per point, against
  about 12 KB per point at large N (`cpu.json`). Whether N² extrapolation from such
  a reading is "cautious" for an integrand whose true memory grows like N² was not
  examined. (On the GPU, readings at 100 points are above the 10 KB floor, so C
  could fit there.)
- **Proposed statement:** "C needs a new rule for the case with no reliable
  reading (today it raises), and the safety of extrapolating from readings below
  the floor must be shown."

### F10. Unsupported: option B2's cost in robustness

- **Claim** (B2): "Each rule exists for a failure listed in
  `docs/user_guide/devices_memory.md` (limitations 7 to 9), so speed is bought with
  robustness one for one."
- **Verdict:** unsupported.
- **Evidence.** Limitation 7 is "a single large reading", a property of the fit,
  not of a stop rule; limitations 8 and 9 concern N² growth and the 1 % rule
  (`devices_memory.md:138-156`). The 1 s per-call limit, the rule that actually
  ended every CPU search (F1), is not tied to any listed limitation; its comment
  calls it "the backstop" (`memory.py:866-868`). No run measured a looser rule. By
  F1, the 1 % rule never fired on the CPU, so on this node loosening it would cost
  nothing it currently provides.
- **Proposed statement:** "B2 is the cheapest change aimed at the actual cause on
  the CPU. Its cost is a longer extrapolation of the fit (limitation 9 grows by the
  factor the last test call shrinks). It should be measured, not estimated."

### F11. Incomplete: "a better starting N saves under 3 %"

- **Claim** (2.3 end, B1): "A better starting N would save at most the first line of
  the table in item 1 (0.035 s of 1.596 s)"; "Saves under 3 %".
- **Verdict:** confirmed on the CPU; not shown on the GPU.
- **Evidence** (`share.py`). CPU, 14 runs: calls below the floor took 0.2 to 2.2 %
  of test-call time, calls with N ≤ 1,000 took 0.1 to 1.4 %. GPU, 14 runs: calls
  with N ≤ 1,000 took 42 to 100 % of test-call time in 10 runs. In absolute terms
  these are milliseconds, except that a single slow small call (0.048 s at N = 10
  in `mlp_64x2`, 0.083 s in `mlp_512x3`, both 1e-6) also reshaped the search; in
  the `mlp_512x3` case it was followed by 0.46 s of further test calls (F4). My
  conjecture is that these slow calls are first-use costs of new CUDA kernels,
  which the integration would pay anyway; not measured.
- **Proposed statement:** "On the CPU the start of the search costs under 3 %. On
  the GPU the start can matter through one-time costs of the first calls; a better
  starting N would not remove them."

### F12. Misleading: "a fixed, far smaller batch cost little"

- **Claim** (2.4): "sin(3e6 t) with max_batch=10,000 took 1.81 s on the CPU (929
  calls) against 1.22 s, and 1.33 s against 1.18 s on the GPU."
- **Verdict:** numbers confirmed (`hard_cpu.log`: 1.8070 s, 929 calls, oracle
  1.2237 s; `hard_cuda.log`: 1.3293 s, 901 calls, oracle 1.1802 s), framing
  misleading.
- **Evidence.** That is 48 % slower on the CPU and 13 % on the GPU; for sin(1e5 t)
  the CPU figure is 0.1626 s against 0.1015 s, 60 % slower. One run each.
- **Proposed statement:** "A batch of 10,000 was 48 to 60 % slower than the oracle
  on the CPU and within noise on the GPU, but still faster than benchmark plus
  integration today."

### F13. Incomplete: what D fixes in limitation 6

- **Claim** (D): "The readings are of the real thing: real values of t (limitation
  6 today: one input value)".
- **Verdict:** overstated.
- **Evidence.** D stops measuring at convergence (item 3), so values of t reached
  after that are not measured, and the limit grows from the subintervals of the
  first passes, which cover the interval coarsely. An integrand whose memory peaks
  in a region refined late is not covered.
- **Proposed statement:** "D measures the values of t of the passes before
  convergence; memory that depends on t elsewhere is still not seen."

### F14. Misleading: the GPU cache leftover, without its consequence

- **Claim** (D, item 5): the 11 % leftover, the 1/(g − 1) rule, and the proposal to
  empty the cache at each growth.
- **Verdict:** confirmed, but the reason it matters is missing.
- **Evidence.** 1,323 / 11,719 MiB = 11.3 %; 1/(10 − 1) = 11.1 %. The derivation
  (previous calls' blocks sum to M/g + M/g² + ... = M/(g − 1)) is an upper bound,
  reached only when no older block can serve any allocation of the new call.
  The consequence the brainstorm omits: `_get_cuda_memory` counts reserved but
  unallocated cache as free (`memory.py:473-485`), so a fragmented cache inflates
  the usable memory used to size the next batch. With the doubling the brainstorm
  proposes for `take_gradient=True`, the bound is 100 % of the last call.
- **Proposed statement:** add "the leftover matters because padaquad counts the
  cache as free memory; emptying it at each growth is therefore required, not
  optional, when growth is by doubling."

### F15. Incomplete: Python references at the end of a pass

- **Claim** (D, item 5): `y_step_eval` is deleted but still held by `kept_y` and
  `intermediate_results`; `method_output`, `nodes`, `loss` stay bound. "Worth
  deleting these names."
- **Verdict:** correct as read (`base.py:1101`, `1130-1137`, `1153`, `1169`), but
  of little consequence.
- **Evidence.** After line 1101 these names hold the accepted part of the previous
  pass (filtered by `keep_mask`), not f's working memory: for T = D = 1 in float64,
  nodes and values are 16 bytes per point, against 30 bytes per point (sin) to
  about 12 KB per point (the MLP) of f's working memory (`cpu.json`). The batch
  size is computed from live free memory at every pass (`base.py:1860-1864`), so
  the leftover is already accounted for, today and under D.
- **Proposed statement:** "Small and already counted; not a priority."

### F16. Misleading: "the 95 %" and the table of section 2.3

- **Claim** (2.3, item 1): "The last test calls (CPU: about 95 % of the time)" with
  a table of `damped_sine` "1.596 s in total".
- **Verdict:** numbers confirmed, mixed sources.
- **Evidence.** The table is `cpu.json` (time_benchmark.py, first row), whose
  `bench` is the sum of test calls only. In that row the cold call took 3.044 s,
  so the test calls were 52 % of it. The 95 % holds for the non-first rows of
  `hard_cpu.log` (0.650 / 0.693 = 94 % to 1.277 / 1.312 = 97 %). Every row of the
  table matches the JSON (checked with `calls.py`).
- **Proposed statement:** name the script and say "96 % of the test-call time".

### F17. Minor: units in B4

- **Claim** (B4): "about 270 bytes · N² ... would need 2.7 TB at 100,000 points".
- **Verdict:** arithmetic correct in decimal units (270 × 1e10 = 2.7e12 bytes); in
  the 1024³ units the brainstorm says it uses, it is 2,515 GB. The source
  description is at `memory.py:1065-1069` as cited.

### F18. Minor: the Accelerate citation

- **Claim** (7): Accelerate "starts at 128 and retries with a smaller batch after each
  out-of-memory error (half according to the documentation; newer source multiplies
  by 0.9). Source: https://huggingface.co/docs/accelerate/main/usage_guides/memory".
- **Verdict:** content confirmed; the cited page does not support all of it.
- **Evidence.** The cited page says the batch is decreased "in half after each
  failed run" and does not give a default start. The source
  (`src/accelerate/utils/memory.py` on the `main` branch,
  https://raw.githubusercontent.com/huggingface/accelerate/main/src/accelerate/utils/memory.py,
  downloaded 2026-11-17) has `starting_batch_size: int = 128` (line 96) and
  `batch_size = int(batch_size * 0.9)` (line 141), with the docstring saying 0.9
  (lines 101, 112).
- **Proposed statement:** cite the source file for 128 and 0.9 and the guide for
  "in half".

## 3. Missing risks and options

1. **Results depend on the batch in two of the three paths** (F2). Any rule that
   starts small changes meshes and evaluation counts for `take_gradient=True` and
   `conserve_memory=True`.
2. **`conserve_memory=True` layout.** The carried partial subinterval is laid out
   with `% max_batch` arithmetic (`base.py:2260-2310`). The batch already varies
   between passes today, but by small amounts; tenfold changes between passes are
   untested there (my reasoning, not run).
3. **Side effects of reading on every pass.** On the GPU the reset clears the
   user's peak statistics (`max_memory_allocated`, `max_memory_reserved`) for the
   device; on Linux it resets `VmHWM` and what `getrusage` or `/usr/bin/time` report
   as peak (`memory.py:1028-1036`). Today this happens during the benchmark only;
   under D it happens throughout the integration, which interferes with users who
   profile their own code.
4. **Many-core CPUs never converge** under the present rule (F1, F2). D needs a
   convergence test that does not include E, or one based on the change between
   readings rather than the cost per point.
5. **Time per point that grows with N.** Today and under D the batch is sized by
   memory only. D observes call times for free and could stop growing when the
   time per point rises; the brainstorm does not consider it.
6. **Several solvers, threads or processes on one device.** Readings cover the
   whole process or device (limitation 5). Under D they are taken during the
   integration, when other work (data loading, another solver, NCCL in a
   distributed run) is more likely to be running than during a dedicated
   benchmark.
7. **Missing option: fix the stop rules.** Compute the convergence test without E
   (in 3 of 14 CPU runs this alone would have saved the last call, 0.8 to 0.9 s;
   `replay.py` with E = 0), make it robust to one off reading, and lower the
   per-call time limit. A few lines, aimed at the measured cause.
8. **The user's idea 1, second half.** "Require the user to give a guess" is read
   only as "the user must pass `max_batch`" (B3). A guess could instead be an
   optional hint of the batch the integral needs, used to stop the benchmark there
   (C with a user-supplied target). Not discussed.

## 4. Claims checked and found correct

- All 12 rows of both tables in 2.2: points, largest call, integration alone,
  benchmark, test calls, ratio and `max_batch` match `hard_cpu.log` and
  `hard_cuda.log` (ratios recomputed: 118, 6.8, 0.87, 20, 99, 55; 39, 0.69, 0.09,
  1.7, 24, 2.1).
- Every row of the `damped_sine` table in 2.3 matches `cpu.json`, in 1024³ units
  (1,837 MiB = 1.79 GiB; 14,694 MiB = 14.35 GiB); the last four calls sum to
  1.535 s, 96 % of 1.596 s.
- `wolf_schlegel` on the GPU: 1e9 points, 44.70 GiB, 0.301 and 0.926 s; `mlp` on
  the GPU: 3,133,216 points, 35.88 GiB, 0.468 s (`cuda.json`).
- In all 12 `time_hard.py` runs, and in all 28 `time_benchmark.py` runs, the largest
  call of the integration was far below the measured limit.
- The three pairs of unstable limits in 2.4 are quoted correctly, and the first two
  pairs made identical test calls.
- `_release_memory` times: 0.087 to 0.132 s on the GPU (`hard_cuda.log`), 0.033 to
  0.049 s on the CPU (`hard_cpu.log`); 0.087 s of the 0.092 s sin(1e5 t) GPU
  benchmark.
- The first `_fit_up_to_quadratic` cost is the scipy import: on the login node
  `from scipy.optimize import nnls` took 0.563 to 0.573 s in three fresh processes,
  the first `nnls` call 0.3 ms (`scipy_import.py`); padaquad imports scipy nowhere
  else (grep).
- A.2's method is equivalent to nonnegative least squares for three columns:
  over 20,000 random problems, the best feasible subset solution matched
  `scipy.optimize.nnls` to 2.6e-15 (`nnls_subsets.py`). The argument: the NNLS
  solution solves unconstrained least squares on its positive columns, so it is one
  of the subset solutions, and every feasible one has a residual at least as large.
  Caveat: include the empty subset (8 candidates, not 7) for completeness.
- `take_gradient=True` has no out-of-memory retry: `_evaluate_f_on_full_nodes` calls
  f once (`base.py:1971`; its docstring says so at 1935) and `loss.backward()` at
  `base.py:1157` is not inside any `try`.
- A backward pass that raises can leave some gradients written: re-run of part (c)
  on the CPU gave `a.grad = [2, 2, 2]`, `b.grad = None` with the failing branch built
  first, and both None with it built last (`partial_grad.py`).
- The emulation rule in `ramp_batches` matches its description in the brainstorm,
  and D's final limits (1,280 to 12.8 million) match `B_final` in the logs.
- The option D table's numbers and call counts match the logs.
- The reading costs: reset plus two reads 0.185 ms CPU, 0.144 ms GPU median
  (`cpu.json`, `cuda.json`), consistent with the 0.16 to 0.34 ms and 0.08 to 0.09 ms
  quoted from `checks.py`.
- GPU leftover 1,323 MiB of 11,719 MiB (11.3 %) and the 1/(g − 1) estimate of 11.1 %
  (as an upper bound, F14).
- File and line references: `memory.py:523`, `175`, `504-521`, `839-928`, `942`,
  `1065`; `base.py:885`, `1169`, `1860`, all correct in the current checkout.
- B4 arithmetic (F17) and the 255 MiB at N = 1,000 behind it (270 bytes · 1e6 =
  257 MiB).
- The Lightning description: the version 1.1.8 page says `power` "keep[s]
  multiplying the batch size by 2, until we get an OOM error" and `binsearch`
  doubles and then searches between the last success and the failure.
- Limitation numbers cited for D (1, 3, 6, 10, 11, 12) point to the right items of
  `devices_memory.md`.

## 5. Confidence

Sure:

- F1: the replay reproduces every logged CPU and GPU sequence of N with the code's
  own fit functions, and shows which stop rule ended each search. The exact E on
  the node is not logged, but 128 and 256 MiB both reproduce the logs and E = 0
  does not.
- F4: the usable memory implied by the stored `max_batch` is identical within each
  pair, so the difference is in the fitted coefficients.
- F8 history, F9 code path, F12 and F16 numbers, the NNLS equivalence, and the
  partial-gradient result.

Less sure:

- F2's claim that small first batches change results under `take_gradient=True` is
  read from the code and supported by the project memory note; I did not run it.
- F3: my re-check ran on a login node with 8 threads, not on the GPU node with 64.
  It shows the original result is fragile; it does not prove there is no slowdown
  there.
- F5: I could not attribute the unexplained 0.6 to 0.8 s of first-run cost. Doing
  so needs a GPU node, which I judged would not change any recommendation.
- F11's explanation of the slow small GPU calls (first use of new kernels) is a
  conjecture.
- Section 3, items 2, 5 and 6, are my reasoning, not tested.

Not checked: any behavior on macOS or Windows; the `checks.py` part (b) GPU numbers
(output exists only in the original conversation; I checked only their arithmetic);
`first_call.py` outputs beyond the scipy import (same reason).
