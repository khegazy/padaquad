# Making the memory measurement fast: brainstorm

Status: open, nothing decided, no code changed. Written 2026-11-17 on branch
`fix_warm_start` (commit 5af0d76, release 2.0.0), and revised the same day after
an adversarial review, `2026-11-17-memory-benchmark-speed-critique.md` (findings
F1 to F18; section 8 lists what each one changed here), and again after option E
was emulated and measured (section 2.6). Written so that a later session can pick
it up without the original conversation.

Terms used below. The **memory benchmark** is `_setup_memory_checks`
(`padaquad/quadrature/memory.py:523`): before an integration without
`max_batch`, it calls the integrand f on N copies of one point, for growing N,
and fits the memory of one call. A **test call** is one of those calls. A
**pass** is one turn of the main loop of `integrate`. The **batch limit** is the
largest number of points given to f in one call (`max_batch`). The **default
path** is `take_gradient=False` with `conserve_memory=False`. **E** is the
allowance for reading error that the Linux CPU reader adds to every reading
(`memory.py:1152`).

## Summary

- **The problem is confirmed.** On a 128-CPU node the benchmark's test calls took
  0.6 to 1.8 s whatever the integral, in 20 runs of two scripts, while the
  integration alone took 0.003 to 1.2 s. On a GPU the benchmark took 0.09 s to
  about 1 s.
- **The cause on the CPU is two stopping rules, not the machine's memory.** Every
  CPU search ended because the next test call was predicted to take more than 1 s
  (or passed 1e9 points), after using at most 4 % of the memory. The rule meant to
  stop it earlier, "the cost per point is stable", never fired: the cost it tests
  includes E, which is 256 MiB on this node and grows with the square of the CPU
  count (section 2.3).
- **On the GPU** the cost is a 0.05 s sleep at the end of every benchmark, which a
  refactor added by accident, plus occasional long searches caused by a single
  memory reading out of proportion.
- **The first benchmark of a process** costs a further 0.6 to 1.5 s, of which the
  scipy import is 0.58 s; the rest is not explained.
- **The number it measures was not needed in any of the 40 runs**, and it is not
  stable: the same integrand gave limits up to 6.7 times apart, because the fit is
  extrapolated up to thousands of times beyond its largest test call (section
  2.4).
- **Options** (section 3): A, remove the fixed costs; B, your idea 1, of which the
  useful part is fixing the stopping rules; C, stop the benchmark at the size the
  integral needs; D, your idea 2, measure during the integration; E, new, run the
  benchmark only as far as the integration asks, and continue it when a pass asks
  for more.
- **Option E was emulated and measured (section 2.6), and it works.** In 63
  comparisons (21 cases, 3 starting meshes each, CPU and GPU, with and without
  gradients) it gave the same integral, the same evaluation points and the same
  gradients as a run with a known batch limit, bit for bit, with no out-of-memory
  error. On the CPU node, where the benchmark takes 0.7 to 1.7 s today, the whole
  run took 1.03 to 1.46 times the integration alone in the better of two
  variants. Three things were worse than
  I had predicted: the test calls evaluate more points than my bound, a garbage
  collection or a GPU cache emptying after each group of test calls costs more
  than the test calls themselves, and where the memory limit is really reached E
  is no faster than today.
- **Recommendation.** Design E for `integrate`, in the variant that grows the test
  calls in steps of 10, with no garbage collection and no cache emptying after
  test calls. Remove the sleep and fix the stopping rules (A.1 and B2) for the
  explicit benchmark `get_max_batch(f=...)`, or as a stopgap; a simulation puts
  the CPU test calls at 0.10 to 0.24 s with them. Section 5 lists what you need
  to decide.

## 1. How it works today

1. `integrate` without `max_batch` calls `_setup_memory_checks` on every call
   (`padaquad/quadrature/base.py:885`). Nothing is kept between calls (a
   decision of 2.0.0; not reopened here).
2. The benchmark calls f on N = 1, 10, 100, ... copies of one point. A reading
   counts only above a noise floor (10 KB on a GPU, 4 MiB on a CPU). The next N is
   10 times the last, or 2 times after a call longer than 0.1 s, and never more
   than what is predicted to fit in memory and to take 1 s (`memory.py:873-902`).
3. It stops at the first of (`memory.py:839-928`):
   - the cost per point c = s (M + E) / N changed by less than 1 % over each of
     the last two calls, with three readings above the floor ("cost stable");
   - a call took more than 1 s;
   - N reached 1e9;
   - the next call would be less than twice the current one, because of memory or
     of the 1 s prediction;
   - 60 s in total.
4. It ends with `_release_memory` (`memory.py:942`): a garbage collection, and on
   a GPU `torch.cuda.empty_cache()` and `time.sleep(0.05)` (`memory.py:504-521`).
5. During the integration, every pass sizes its batch from the fit and the
   memory free at that moment (`_get_max_f_evals`, `base.py:1860`). In the default
   path a pass evaluates every waiting subinterval, in calls of at most the batch
   limit, so the limit changes only how the pass is cut into calls. With
   `take_gradient=True` or `conserve_memory=True` a pass is one batch, so the
   limit decides which subintervals a pass holds.

## 2. Evidence (measured 2026-11-17)

### 2.1 Setup

One Perlmutter GPU node: AMD EPYC 7763 (128 logical cores, 503 GB RAM), NVIDIA
A100 80 GB, torch 2.8.0, Linux with glibc. Method gk21, interval [0, 1],
float64. Everything is throwaway and lives in
`/pscratch/sd/k/khegazy/padaquad_membench/`; scratch is purged, so copy it if it
should be kept.

- `time_hard.py` (logs `hard_cpu`, `hard_cuda`): 6 harder integrals per device,
  12 runs. It times the benchmark, its test calls and `_release_memory`
  separately, and emulates option D.
- `time_benchmark.py` (`cpu.json`, `cuda.json`): 7 easy integrands, two
  tolerances, per device, 28 runs, with the reading of every test call.
- `first_call.py`, `checks.py`: one-time costs and three small checks.
- `critique/`: the reviewer's scripts. `critique/replay.py` reruns the search
  logic on the logged readings.
- `whatif/whatif.py`: my simulation of changed stopping rules (section 2.5).
- `optE/emulate_E.py` (logs `E_cpu`, `E_cuda`, and `E2_*.json` for the second
  run): the emulation of option E (section 2.6). It ran on a second node of the
  same CPU type with 251 GB RAM and an A100 40 GB.

"Integration alone" is a call with `max_batch=result.max_batch` (median of 3).
CPU and GPU ran at the same time on the node, and GPU timings scattered by tens
of percent between repeats, so differences below about 30 % mean nothing. Memory
is in units of 1024³ bytes, as in the code.

### 2.2 Benchmark against integration

From `time_hard.py`: `take_gradient=False`, atol = rtol = 1e-9. "Peaks" is a sum
of 40 Lorentzians of width 1e-5. "mlp" is a network with 3 hidden layers of 512
tanh units, its input multiplied by the number shown. The points differ between
CPU and GPU because the starting mesh is random.

**CPU**

| Integrand | Points | Largest call of f | Integration alone (s) | Benchmark (s) | Of which test calls (s) | Benchmark ÷ integration | Measured `max_batch` |
|---|---|---|---|---|---|---|---|
| sin(1e3 t) | 4,347 | 1,702 | 0.0111 | 1.311 † | 0.676 | 118 | 1e9 |
| sin(1e5 t) | 332,695 | 110,676 | 0.1015 | 0.693 | 0.650 | 6.8 | 1e9 |
| sin(3e6 t) | 7,679,079 | 2,575,678 | 1.2237 | 1.059 | 1.006 | 0.87 | 1e9 |
| 40 peaks | 25,737 | 2,668 | 0.0403 | 0.810 | 0.773 | 20 | 354,551,929 |
| mlp(300 t) | 437 | 253 | 0.0133 | 1.312 | 1.277 | 99 | 24,781,674 |
| mlp(1e4 t) | 575 | 253 | 0.0208 | 1.141 | 1.093 | 55 | 28,121,839 |

**GPU**

| Integrand | Points | Largest call of f | Integration alone (s) | Benchmark (s) | Of which test calls (s) | Benchmark ÷ integration | Measured `max_batch` |
|---|---|---|---|---|---|---|---|
| sin(1e3 t) | 4,945 | 1,472 | 0.0324 | 1.271 † | 0.004 | 39 | 1e9 |
| sin(1e5 t) | 359,421 | 70,656 | 0.1329 | 0.092 | 0.003 | 0.69 | 1e9 |
| sin(3e6 t) | 7,815,101 | 2,249,446 | 1.1802 | 0.104 | 0.003 | 0.09 | 1e9 |
| 40 peaks | 26,105 | 2,668 | 0.0948 | 0.164 | 0.028 | 1.7 | 58,919,365 |
| mlp(300 t) | 437 | 253 | 0.0281 | 0.664 | 0.537 | 24 | 4,637,513 |
| mlp(1e4 t) | 667 | 253 | 0.0450 | 0.094 | 0.005 | 2.1 | 2,881,527 |

† The first benchmark of the process; it includes one-time costs (section 2.3,
item 4). Without them the first CPU row is 65 times the integration.

The 28 runs of `time_benchmark.py` are easier integrals (253 points, 3.4 to 8.6
ms) and give larger ratios. On the CPU the test calls alone took 0.637 to 1.783 s,
109 to 525 times the integration. On the GPU they took 0.002 to 0.926 s.

### 2.3 Why the benchmark is slow

1. **CPU: the search runs until a call takes about 1 s.** The test calls of
   `damped_sine` (`cpu.json`, first row; 1.596 s of test calls in total):

   | N | Memory read | Time (s) |
   |---|---|---|
   | 1 to 100,000 (6 calls) | below the 4 MiB floor | 0.035 |
   | 601,830 | 21 MiB | 0.006 |
   | 6,018,300 | 199 MiB | 0.020 |
   | 60,183,000 | 1.8 GB | 0.110 |
   | 120,366,000 | 3.6 GB | 0.213 |
   | 240,732,000 | 7.2 GB | 0.405 |
   | 481,464,000 | 14.3 GB | 0.807 |

   The last four calls are 96 % of the test-call time. The reviewer's replay
   reproduces the sequence of N of all 14 CPU runs from the logged readings (I
   reran it). It shows:
   - 12 searches ended because the next call was predicted to take more than 1 s,
     and 2 because the next doubling passed 1e9 points. The code reports the
     second case as memory (limitation 14 in the user guide). The largest call
     used 17.9 GB of about 428 GB usable, 4 %.
   - The "cost stable" rule fired in none of them. It tests c = s (M + E) / N.
     On this node E = 2 · 128 · 256 pages = 256 MiB if the process may run on all
     128 CPUs (E was not logged; 128 and 256 MiB both reproduce the logs and 0
     does not). `damped_sine` uses about 30 bytes per point, so E is 1 % of M only
     near 880 million points; until then c keeps changing by more than 1 %.
   - E is 2 · (CPUs the process may use) · max(32, 2 · CPUs online) pages, so it
     grows with the square of the CPU count. On a laptop it is a few MiB and the
     search would stop much earlier. The 0.6 to 1.8 s is specific to many-core
     nodes; I have no measurement on a small machine.
   - After a call longer than 0.1 s the step is 2, and calls stop at a predicted
     1 s, so a cheap CPU search always ends with calls of about 0.1, 0.2, 0.4
     and 0.8 s.
2. **GPU: one reading out of proportion stretches the search.** On the GPU E is 0
   and "cost stable" ended all 14 searches. It needs two changes below 1 % in a
   row, so one off reading costs two more calls, each 10 times larger. For
   `wolf_schlegel` the reading at 1e6 points was 3.2 % high and the search went to
   1e9 points (44.7 GB, 0.3 to 0.9 s) instead of 1e7. For mlp a reading at 340
   points was 30 % high and the search went to 3.1 million points (35.9 GB,
   0.47 s).
3. **The end of every benchmark: `_release_memory`.** Measured 0.075 to 0.14 s on
   the GPU (0.05 s sleep plus a garbage collection of 0.025 to 0.04 s) and 0.03
   to 0.05 s on the CPU. On the GPU this is nearly all of a benchmark whose test
   calls are small (0.087 s of 0.092 s for sin(1e5 t)). Git history: the sleep was
   written for the out-of-memory handlers only. Until commit 3c5823a (a refactor,
   2026-11-08) the normal end of the benchmark ran the garbage collection and
   `empty_cache()` with no sleep; the refactor routed it through
   `_release_memory`.
4. **Once per process.** The first benchmark carried 0.59 to 1.45 s that is
   neither test calls nor `_release_memory`. The import
   `from scipy.optimize import nnls` (`memory.py:175`) is 0.58 s of it (measured;
   the comment there says 0.25 s). On the CPU in `time_hard.py` that is all of it.
   On the GPU about 0.58 s more, and in the first row of `cpu.json` about 0.8 s
   more, are not attributed.
5. **Every call repeats items 1 to 3** when no `max_batch` is passed.
6. **The start of the search.** On the CPU the calls below the noise floor took
   0.2 to 2.2 % of the test-call time. On the GPU a single small call was
   sometimes slow (0.048 s and 0.083 s at N = 10 for the two networks) and
   changed the rest of the search. The reviewer's conjecture is that these are
   first uses of GPU routines, which the integration would pay anyway.

### 2.4 The measured number: not needed, and not stable

- **Not needed.** In all 40 runs the largest call of f was far below the measured
  limit, so the limit never changed what the integration did.
- **Not stable.** Two consecutive benchmarks of the same integrand gave: mlp on
  the GPU with `take_gradient=True`, 338,878 and 1,199,344; mlp on the CPU,
  4,182,912 and 27,993,951; mlp on the GPU, 4,636,742 and 2,881,525.
  - Free memory was the same within each pair (427.8 GB on the CPU, 70.8 GB on
    the GPU, computed back from the stored limits).
  - The readings differed by 1 to 14 %, and the N² term of the fit turned that
    into factors of 3.5 to 6.7. Example, CPU: the reading at 200,000 points was
    2,673 against 2,346 MiB; the fitted N² coefficient was 1.7e-2 bytes against 0.
  - The fit is extrapolated far beyond its readings: the limit was 1.25 to 140
    times the largest test call on the CPU and up to 8,210 times on the GPU (all
    28 runs of `time_benchmark.py`).
- **A fixed smaller batch.** sin(3e6 t) with `max_batch=10,000` took 1.81 s on
  the CPU (929 calls) against 1.22 s, 48 % slower; sin(1e5 t) was 60 % slower. On
  the GPU the difference was 13 %, inside the noise. One run each. A crude limit
  costs something on the CPU, but less than the benchmark does today.

### 2.5 Simulated: changed stopping rules

`whatif/whatif.py` reruns the search with changed rules on memory and time curves
interpolated from the logged readings of `time_benchmark.py`. With today's rules
it reproduces the sequence of N of all 28 runs. These are simulations, not
measurements: they contain no timing noise and no new readings.

| Rules | CPU test calls (s), range and median | CPU limit ÷ largest test call | GPU test calls (s), range and median | GPU limit ÷ largest test call |
|---|---|---|---|---|
| Today | 0.64 to 1.78, 1.54 | 1.25 to 140 | 0.002 to 0.93, 0.009 | 1 to 8,210 |
| E out of the stability test | 0.64 to 1.72, 0.97 | 2.5 to 140 | unchanged | unchanged |
| and skip one off reading | 0.64 to 1.72, 0.97 | 2.5 to 140 | 0.002 to 0.47, 0.008 | 1.5 to 8,210 |
| and 0.1 s per call instead of 1 s | 0.10 to 0.24, 0.15 | 25 to 1,360 | 0.002 to 0.26, 0.008 | 36 to 9,230 |
| and 5 % instead of 1 %, 0.02 s per call | 0.02 to 0.09, 0.035 | 162 to 3,640 | 0.001 to 0.10, 0.004 | 360 to 290,000 |

- With the 0.1 s rule the simulated limit was 0.06 to 6.15 times today's on the
  CPU (the 6.15 is the run whose logged limit was the low outlier of section 2.4)
  and equal to today's on the GPU.
- With the last row it was 0.02 to 1.0 times today's on the CPU and 0.01 to 4.9
  times on the GPU.
- Shorter searches buy time with a longer extrapolation of the fit, which is the
  cause of the unstable limits in section 2.4.

### 2.6 Measured: option E, emulated

`optE/emulate_E.py` patches the solver while it runs; no library code was
changed. It replaces the up-front benchmark by the rule of option E (section 3):
before each pass, run test calls until 10 times the largest one covers the pass,
with the limit min(10 × the largest test call, what the memory model allows). It
reuses the solver's own reader, test call, fit and worst-case rule, and it covers
all the evaluation paths because it only sets the batch limit.

Two variants of how far a group of test calls goes:

- **Tenth:** the test call is no larger than a tenth of the pass, so the limit
  just covers the pass. Almost every growing pass then needs a new test call.
- **Decade:** the test calls keep their steps of 10, so the limit jumps by a
  factor of 10 and later passes need none until they outgrow it.

Method, chosen to meet the review's objection to my first emulation: every mode
ran with the same 3 random starting meshes (seeded), the modes were interleaved
in rotating order, all runs were warm, and the CPU and GPU runs were made one
after the other. "Reference" is the integration alone, with the batch limit a
benchmark measured before. Times are medians of 3 runs, in seconds. "grad" means
`take_gradient=True` with learnable weights. "LIMITED" marks cases in which the
memory limit is really reached: usable memory set to about 2 GB (CPU) or 0.6 GB
(GPU) through `total_mem_usage`, and an integrand whose memory grows like N²
(one gradient per point, GPU only).

**CPU** (128 CPUs, 251 GB)

| Integrand | Today | Reference | E, tenth | E, decade | Decade ÷ reference |
|---|---|---|---|---|---|
| sin(1e3 t) | 0.749 | 0.0113 | 0.0156 | 0.0142 | 1.26 |
| sin(1e5 t) | 0.869 | 0.0965 | 0.1152 | 0.0650 | 0.67 |
| sin(3e6 t) | 3.076 | 1.373 | 1.429 | 1.416 | 1.03 |
| 40 peaks | 0.918 | 0.0400 | 0.0561 | 0.0478 | 1.20 |
| mlp(300 t) | 1.033 | 0.0116 | 0.0168 | 0.0170 | 1.46 |
| mlp(4 t) sin(1e5 t) | 2.854 | 1.350 | 1.967 | 1.864 | 1.38 |
| mlp64, grad | 1.620 | 0.0053 | 0.0169 | 0.0069 | 1.32 |
| mlp64 sin(3e3 t), grad | 1.621 | 0.0273 | 0.0707 | 0.0326 | 1.19 |
| mlp512 sin(3e3 t), grad | 1.541 | 0.1402 | 0.2217 | 0.1793 | 1.28 |
| LIMITED mlp(4 t) sin(1e5 t) | 1.914 | 1.615 | 1.962 | 1.885 | 1.17 |

**GPU** (A100 40 GB)

| Integrand | Today | Reference | E, tenth | E, decade | Decade ÷ reference |
|---|---|---|---|---|---|
| sin(1e3 t) | 0.1099 | 0.0222 | 0.0278 | 0.0264 | 1.19 |
| sin(1e5 t) | 0.1798 | 0.0653 | 0.1187 | 0.0798 | 1.22 |
| sin(3e6 t) | 1.1334 | 0.9986 | 1.0914 | 1.0422 | 1.04 |
| 40 peaks | 0.1773 | 0.0720 | 0.0840 | 0.0794 | 1.10 |
| mlp(300 t) | 0.1213 | 0.0211 | 0.0291 | 0.0248 | 1.18 |
| mlp(4 t) sin(1e5 t) | 0.2374 | 0.1039 | 0.2326 | 0.1408 | 1.35 |
| mlp64, grad | 0.0954 | 0.0072 | 0.0091 | 0.0090 | 1.25 |
| mlp64 sin(3e3 t), grad | 0.1400 | 0.0357 | 0.0525 | 0.0409 | 1.15 |
| mlp512 sin(3e3 t), grad | 0.1489 | 0.0380 | 0.0572 | 0.0454 | 1.20 |
| LIMITED mlp(4 t) sin(1e5 t) | 0.2180 | 0.1015 | 0.2197 | 0.1303 | 1.28 |
| LIMITED N² memory | 8.558 | 7.243 | 12.203 | 8.965 | 1.24 |

The 0.67 in the CPU table is noise: the three reference runs of that case were
0.096, 0.065 and 0.111 s.

What held as predicted:

1. **Results are unchanged.** In all 63 comparisons against the reference (10
   CPU and 11 GPU cases, 3 starting meshes each) E evaluated the same number of
   points, ended with the same mesh size, and returned an integral and, with
   gradients, parameter gradients that differ by exactly 0. The number of calls
   of f also matched, except where the limit cut a pass into more calls.
2. **The cost follows the integral.** For the five smallest integrals on the CPU
   the decade variant added 2 to 8 ms to the reference, where today's benchmark
   adds 0.7 to 1.6 s; today's run takes 19 to 235 times as long as the decade
   variant there, and 1.0 to 235 times over all ten CPU cases. On this GPU
   today's benchmark is about 0.09 to 0.1 s, nearly all of it the fixed end cost,
   so the gain is smaller: today's run takes 0.95 to 10.6 times as long.
3. **No memory failure.** No out-of-memory error and no "max_batch is 0" warning
   in any run. Where the limit was reached on the GPU, E's limit agreed with
   today's: 39,235 and 35,882 (tenth, decade) against 38,915, and 8,368 and 8,393
   against 8,370 for the N² integrand.

What was worse than predicted:

1. **More test points.** I predicted at most 11 to 20 % of the largest pass.
   Measured: 11 to 61 % for the tenth variant (3 to 15 % of all points of the
   integration, in up to 16 groups of test calls) and 5 to 93 % for the decade
   variant (2 to 44 % of all points, in 1 to 5 groups). My bound assumed test
   calls only at each factor of 10; the tenth variant tests again at almost
   every pass, and the decade variant can test a size close to the pass itself.
2. **Test calls are slower per point than passes when f dominates.** For mlp sin
   on the CPU, the tenth variant's test calls were 10 % of the points but took
   0.31 s, 23 % of the reference time. In a second run of 5 repeats the decade
   variant made a test call of 100,000 points for a pass barely larger, and its
   test calls took 0.52 s in that run; the median was still 1.84 s against 1.70 s
   for the reference and 3.02 s today.
3. **Anything done after each group of test calls must be nearly free.**
   - A garbage collection after each group cost 25 to 40 ms per group and made
     small runs 4 to 13 times slower than the reference on the CPU (0.149 s
     against 0.0156 s for sin(1e3 t)).
   - Emptying the GPU cache after each group also cost time. Second run, 5
     repeats, N² integrand: 9.24 s with it and 7.37 s without, against 6.89 s for
     the reference and 8.45 s today; the limit was the same (8,393 and 8,390) and
     nothing ran out of memory.
4. **Where the limit is really reached, E is not faster than today.** Today's
   search ends early there, so its benchmark is short: LIMITED mlp on the CPU,
   1.91 s today against 1.89 to 1.96 s with E.
5. **On the CPU with the limit reached, the limits differed:** 57,873 today,
   41,559 and 31,088 for the two variants. The usable memory was set from the
   free memory at the start of each run, which changes between runs on a node,
   so I cannot say how much of this is the method.

One side result: skipping `malloc_trim` in the reader's reset changed nothing
(decade variant, 5 repeats: 1.84 s with the trim, 1.88 s without, for mlp sin;
0.035 s and 0.031 s for mlp64 sin with gradients). This agrees with the review's
finding F3 that a reading does not slow the calls after it.

## 3. Options

### Option A: remove the fixed costs (mine; independent of the others)

1. **Remove the sleep from the normal end of the benchmark.** Run the garbage
   collection and `empty_cache()` there, as before commit 3c5823a, and keep the
   sleep for the out-of-memory path. Saves 0.05 s per GPU benchmark. I first
   proposed dropping the garbage collection too (0.025 to 0.04 s more); the review
   is right that nothing supports that: it frees reference cycles that hold f's
   tensors, which the memory query that follows would count as used.
2. **Fit without scipy.** The fit has three unknowns that must be nonnegative. It
   equals the best of the least-squares solutions over the 8 subsets of the three
   terms (the empty one included) that have nonnegative coefficients; the reviewer
   checked this against `scipy.optimize.nnls` on 20,000 random problems (agreement
   to 2.6e-15). About 10 new lines against 0.58 s once per process. It removes
   only part of the first-benchmark cost on the GPU (section 2.3, item 4).

A alone leaves the CPU test calls at 0.6 to 1.8 s.

### Option B: a better first guess, or a guess from the user (your idea 1)

- **B1, a better starting N.** On the CPU the start of the search is under 3 % of
  the time. On the GPU the first calls can be slow, but for one-time reasons a
  different starting N would not remove. Not recommended.
- **B2, fix the stopping rules.** This is the cheapest change aimed at the
  measured cause. I had dismissed it on an estimate; the simulation in section 2.5
  replaces that.
  1. Take E out of the stability test. E stays in the stored fit, where it
     belongs: it is an error of a reading, not a cost per point.
  2. Let the stability test skip one reading out of proportion.
  3. Lower the 1 s limit per test call to 0.1 s.

  Simulated: CPU test calls 0.10 to 0.24 s (today 0.6 to 1.8 s), GPU at most
  0.26 s (today 0.93 s). Cost: the fit is extrapolated further (25 to 1,360 times
  beyond the largest test call on the CPU, against 1.25 to 140 today), so the
  limit is less certain. The 1 s rule is not tied to any listed limitation; its
  comment calls it the backstop (`memory.py:866-868`). What B2 cannot do: an
  up-front benchmark must reach three readings above the 4 MiB CPU floor, so it
  evaluates 1e5 to 1e7 points even for an integral of 253 points.
- **B3, the user must pass `max_batch`.** No measurement and no cost. It moves the
  work to every user, which the benchmark exists to avoid.
- **B4, a fixed default** (for example 100,000 points) without measurement. Unsafe
  for an integrand whose memory grows like N²: the popcornn-like integrand
  described at `memory.py:1065` (about 270 bytes · N²) would need about 2,500 GB
  at 100,000 points.
- **B5, an optional hint from the user** (raised by the review): the number of
  points per call the integral is expected to need, used to stop the benchmark
  there. Options C and E get that number from the integration itself, so I do not
  see what a hint adds.

### Option C: stop the benchmark at the size the first pass needs (mine)

Give the benchmark the number of points of the first pass (253 for the default
mesh with gk21) and stop the search there: 3 or 4 test calls of a few
milliseconds.

- Correction: I wrote that the existing worst-case rule would then set the limit.
  It does not. When no reading reaches the noise floor, today's code raises
  (`memory.py:945-950`), and on the CPU every reading at N ≤ 253 is below the
  floor. C needs a new rule for that case.
- A limit fixed after the first pass cannot follow an integral that grows. Option
  E is C with that fixed.

### Option D: measure during the integration (your idea 2)

No test calls. Calls of f in the integration are read with the existing memory
reader, and the batch limit grows only when a pass has more points than the
limit.

What speaks for it:

- No evaluation is wasted: every reading is a batch of the integral.
- The limit grows only as far as the integral needs, so the fit is never
  extrapolated far.
- With `take_gradient=True` the reading can span the backward pass, replacing the
  factor 2.1 guess (limitation 10).
- No time rules are needed. The 1 s and 60 s rules, `limit_benchmark_time`, and
  limitations 11 and 12 (slow integrands raise) go away.

**Emulation (throwaway, `time_hard.py`).** Rule: start at 128 points; read memory
around every call; after a call of full size, the next limit is
min(10 · B, B · sqrt(budget / M)); stop growing when that is below 2 · B.

| Integrand | CPU: benchmark + integration today (s) | CPU: emulated D (s) | CPU: emulated D ÷ integration alone | GPU: today (s) | GPU: emulated D (s) | GPU: emulated D ÷ integration alone |
|---|---|---|---|---|---|---|
| sin(1e3 t) | 1.323 † | 0.019 | 1.73 | 1.305 † | 0.039 | 1.20 |
| sin(1e5 t) | 0.792 | 0.082 | 0.81 | 0.233 | 0.157 | 1.18 |
| sin(3e6 t) | 2.724 | 1.611 | 1.32 | 1.590 | 1.391 | 1.18 |
| 40 peaks | 0.853 | 0.061 | 1.52 | 0.286 | 0.086 | 0.90 |
| mlp(300 t) | 1.334 | 0.021 | 1.58 | 0.719 | 0.025 | 0.90 |
| mlp(1e4 t) | 1.163 | 0.033 | 1.57 | 0.140 | 0.039 | 0.86 |

† Includes the one-time costs. What the emulation does and does not show:

- It shows that skipping the benchmark saves the benchmark. Options B2, C and E
  would show a similar saving on this test set, because no limit was ever
  reached. It is not evidence for D over them.
- The "today" column is one cold run and the D column a median of 3 warm runs
  with other random meshes, so the rows are not like for like.
- On the CPU the emulated integration was 1.3 to 1.7 times slower than the
  integration alone in 5 of 6 integrals (reading memory around every call is part
  of that); on the GPU the difference is inside the noise.
- It covers only the default path. It has no out-of-memory handling, no tracked
  variables, no convergence rule, and it sizes from the last reading, not from
  the fit.

**Design questions, with a proposal for each.** Items marked (yours) come from
your message; the rest are mine.

1. **The result can depend on the batch.** In the default path the limit only
   cuts a pass into calls, so D cannot change the result there. With
   `take_gradient=True` or `conserve_memory=True` a pass is one batch. A first
   pass of 100 points holds 4 of the 11 starting subintervals of gk21, and they
   are judged against a running integral that is far from complete. The re-check
   that corrects this is off with `take_gradient=True` (`base.py:1032-1034`). So
   D started small would change meshes and evaluation counts, and could accept
   subintervals too loosely. A way out, my reasoning and not tested: evaluate the
   first pass whole when it is small, and let the limit at least double after
   every pass. Each pass can at most double the number of waiting points (every
   evaluated subinterval splits into at most two), so such a limit never cuts a
   pass until memory really limits it, as today. A starting mesh too large to
   evaluate unseen (a warm start with many subintervals) keeps the problem.
2. **Starting limit.** 100 points (yours). Today's benchmark never makes a call
   more than 10 times larger than one that ran; a first batch of 253 points after
   the one-point check is a larger step than that. Two tiny test calls of 10 and
   100 points before the first pass would keep today's caution.
3. **Growth.** Reuse the benchmark's model (`memory_model`,
   `_fit_up_to_quadratic`, `_largest_n`), fed with readings of real calls. The
   limit is min(10 × the largest call so far, what the model allows), and only
   goes up.
4. **Convergence** (yours: it must not keep changing). My first proposal reused
   today's "cost stable" rule; that rule fired in 0 of 14 CPU runs, so it would
   never end the measuring on a many-core node. Replacement, mine: read memory
   only around a call at least twice as large as any call read before. Then no
   convergence test is needed. The limit changes only at such calls, at most about
   30 times between 1 and 1e9 points and usually fewer than 10, and the readings
   stop by themselves when the calls stop growing. An out-of-memory error lowers
   the limit as today (× 0.75) and ends the growth for the call.
5. **Out of memory while growing.**
   - Default path on a GPU: the existing retry in `_evaluate_f_in_batches`
     handles it.
   - CPU: a batch that does not fit can be killed by the operating system with no
     Python error (limitation 10), so growth must be predicted, never tried, as
     the test calls are today.
   - `take_gradient=True`: an error inside `loss.backward()` cannot be retried.
     Measured (`checks.py` part c, repeated by the reviewer): a backward pass that
     raises midway had already written the gradient of one of two parameters, so
     a retry would count it twice. Growth there must be cautious, and such an
     error stays an error, as today (`_evaluate_f_on_full_nodes` has no retry).
6. **Leftover memory when the limit changes** (yours).
   - GPU cache: PyTorch keeps freed blocks, and blocks sized for smaller batches
     may not serve a larger one. Measured: growing from 100 to 1e6 points in steps
     of 10 left 13,042 MiB reserved for a call whose peak was 11,719 MiB, so
     1,323 MiB (11 %) was leftover. For a growth factor g the leftover is at most
     1/(g − 1) of the last call (my derivation; up to 100 % for doubling, not
     measured). It matters because padaquad counts the cache as free memory
     (`memory.py:473-485`), so a fragmented cache inflates the memory used to
     size the next batch. `torch.cuda.empty_cache()` after each growth removed
     it, took 0.1 to 7 ms, and did not slow the next call. With growth by
     doubling, emptying the cache at each growth is required, not optional.
   - Python references: at the end of a pass `kept_y`, `intermediate_results`,
     `method_output`, `nodes` and `loss` stay bound while the next pass calls f
     (`base.py:1101-1169`). They hold the accepted results of the previous pass,
     16 bytes per point, against 30 bytes to 12 KB per point of f's working
     memory, and the batch is sized from live free memory. Small and already
     counted; not a priority.
   - Cost of a reading: 0.16 to 0.34 ms on the CPU and 0.08 to 0.09 ms on the GPU
     (measured). I also reported that the call after a CPU reading was 18 to 23 %
     slower. That is not established: the two loops ran one after the other, 7
     calls each, on a shared node; the reviewer's interleaved repeat on a login
     node found no difference; and my explanation cannot hold, because glibc gives
     blocks above 32 MiB their own pages and returns them at every free, with or
     without the trim.
7. **Other effects of reading during the integration** (from the review).
   - Each reading resets the peak-memory statistics of the device or process
     (`memory.py:1028-1036`), which a user profiling their own code would notice.
     Item 4 keeps this to a few resets per call; today the benchmark does 4 to 12.
   - Readings cover the whole process or device (limitation 5). During an
     integration other work (data loading, another solver) is more likely to be
     running than during a separate benchmark.
   - The `conserve_memory=True` layout uses arithmetic modulo the batch limit
     (`base.py:2260-2310`); tenfold changes between passes are untested there.
   - D measures the values of t of the passes it reads. Memory that depends on t
     elsewhere is still not seen (limitation 6 is reduced, not removed).
   - D sees call times for free and could also stop growing when the time per
     point rises; today nothing sizes the batch by time.
8. **macOS and Windows.** There a reading misses memory that f reuses, and during
   an integration every call after the first reuses memory. Readings will be too
   low (limitation 1), so the model cannot be trusted. I could not measure this.
9. **What changes for the user.** `result.max_batch` needs a new meaning when the
   limit was never reached. `get_max_batch(f=...)` can keep the present benchmark.
   The start warning, the user-guide section on limitations,
   `tests/test_memory_benchmark.py` and `tests/test_oom_recovery.py` must follow.

### Option E: run the benchmark only as far as the integration asks (new, mine)

Keep test calls, but never test a size the integral has not asked for. Emulated
and measured in section 2.6; the list below is the design as first written, and
"Measured" after it says what the measurement changed.

1. **Before the first pass:** run the search as today (1, 10, 100, ...) and stop
   as soon as the limit covers the first pass.
2. **The limit** is min(10 × the largest test call, what the model allows in the
   memory budget). The factor 10 is the step the search already takes between
   test calls, so nothing is evaluated further beyond a test call than today's
   next test call would be. Today the limit is up to 8,210 times the largest test
   call (section 2.4).
3. **When a pass has more points than the limit:** continue the same search, with
   its readings kept, up to one tenth of the pass, then evaluate the pass.
4. **No reading above the noise floor** (cheap integrand on a CPU): the model is
   the worst case the code already uses to size the next test call (memory grows
   at most like N² from the last reading, `memory.py:780-782`). Nothing raises.
5. **No time rules.** A test call is at most a tenth to a fifth of a pass the
   integration is about to run, so the 1 s and 60 s rules and
   `limit_benchmark_time` are not needed.

What it costs and what it keeps:

- **Cost follows the integral.** For the default first pass of 253 points: three
  test calls of 1, 10 and 100 points, which took 1 to 4 ms in most of the logged
  runs (one GPU run had a slow first call, section 2.3 item 6), plus the end of
  the benchmark. Later test calls add at most about 11 to
  20 % of the largest pass's evaluations (my derivation: steps of 10 up to a
  tenth of the pass, steps of 2 in the worst case). Not measured.
- **No effect on results.** The limit covers every pass before it runs unless
  memory really limits it, so the passes hold the same subintervals as today, in
  all three paths.
- **No new risk with gradients.** Test calls are forward only, as today; the
  factor 2.1 stays.
- **Less extrapolation than today**, so the unstable limits of section 2.4 should
  shrink (my expectation, not tested).
- **It keeps** limitation 6 (one value of t) and limitation 10 (the gradient
  factor is a guess), which D would reduce.
- **macOS and Windows:** the existing rule caps the limit at the largest test
  call, so test calls there must be as large as the pass. The cost still follows
  the integral.
- **Code:** the search loop of `_setup_memory_checks` must keep its readings and
  resume, and `_evaluate_f_on_mesh` must ask for it when a pass exceeds the
  limit. The reader, the fit, the worst-case rule and the out-of-memory handling
  are reused. The stability test and the time rules can be removed.

**Measured (section 2.6).**

- Confirmed: results unchanged in all 63 comparisons, gradients included; no
  out-of-memory error; the run costs 1.03 to 1.46 times the integration alone
  on the CPU and 1.04 to 1.35 times on the GPU (decade variant); where the limit
  is reached on the GPU it agrees with today's.
- Corrected: the test calls evaluate 5 to 93 % of the largest pass, not 11 to
  20 %. Step 3 above ("up to one tenth of the pass") is the tenth variant, which
  needs a group of test calls at almost every growing pass and was the slower
  one (0.0707 s against 0.0326 s for mlp64 sin with gradients on the CPU, 0.233 s
  against 0.141 s for mlp sin on the GPU). Use the decade variant: keep the steps
  of 10.
- New rules the measurement adds: no garbage collection after a group of test
  calls (25 to 40 ms each), and no emptying of the GPU cache there (9.24 s
  against 7.37 s for the N² integrand).
- Not improved: runs in which the memory limit is really reached. E costs about
  the same as today there.
- Not tested: an out-of-memory error inside a test call, `conserve_memory=True`,
  tracked variables, a warm start from a large mesh, float32, methods other than
  gk21, and macOS or Windows.

### Comparison

| | A: fixed costs | B2: fix stopping rules | C: stop at first pass | D: measure in the loop | E: benchmark on demand |
|---|---|---|---|---|---|
| Time before integrating (CPU node) | 0.6 to 1.8 s | 0.10 to 0.24 s (simulated) | milliseconds | 0 | 2 to 8 ms for small integrals (measured) |
| Follows the size of the integral | no | no | no, limit stays small | yes | yes |
| Wasted evaluations | up to 1e9 points | up to 4e7 points (simulated) | hundreds | none | 5 to 93 % of the largest pass (measured) |
| Can change results | no | no | no | yes, outside the default path | no (63 of 63 identical) |
| Extrapolation of the fit | up to 8,210 times | longer than today | needs a new rule | short | at most 10 times |
| Risk with gradients | as today | as today | as today | a failed backward pass cannot be retried | as today |
| New code | a few lines | a few lines | small | largest | moderate |
| Evidence | measured | simulated | none | emulated, default path only | emulated, 21 cases, with gradients and with the limit reached |

## 4. Recommendation (revised after measuring E)

1. **Design E for `integrate`,** in the decade variant, with no garbage
   collection and no cache emptying after test calls. It is the only option
   measured to make the cost follow the integral without changing results: 2 to
   8 ms on top of a small integral where today adds 0.7 to 1.6 s. Its test calls
   cost more than I predicted, up to about the time of one pass, so expect up to
   roughly 1.5 times the integration alone, not 1.1.
2. **A.1 and B2 for the explicit benchmark,** `get_max_batch(f=...)`, which keeps
   the full search: remove the sleep, take E out of the stability test, let the
   test skip one off reading, and lower the limit per test call. These are a few
   lines and can also go in first as a stopgap for `integrate`. The 0.10 to
   0.24 s for them is a simulation and must be measured.
3. **A.2 (no scipy) if 0.58 s once per process matters to you.** Under E the fit
   is still used, so the import still happens once the fit has three readings.
4. **D later, if at all.** Its remaining advantages over E are no wasted
   evaluations and a real measurement of the backward pass. Its costs are the
   dependence of results on the batch outside the default path, the backward pass
   that cannot be retried, and readings taken while other work runs. I recommended
   D as the default before the review; the evidence did not support that.

## 5. Decisions needed

1. **Go ahead with E?** Measured cost: a few milliseconds on a small integral,
   and up to about the time of one pass on an integral that grows (1.03 to 1.46
   times the integration alone on the CPU). If that is acceptable, the next step
   is a design document for E.
2. **`result.max_batch` under E** when the integral never reached the limit. The
   limit at the end is safe but can be small (1,000 for a small integral), which
   would slow a later, harder call that passes it on as a fixed `max_batch`. The
   model's larger estimate was never run. Or the value could be `None`.
3. **A `max_batch` passed by the user:** I assume it stays a fixed size with no
   test calls. Correct?
4. **The garbage collection.** Today the benchmark ends with one, and the review
   defends it. E cannot afford one after each group of test calls. Is one at the
   end of the first group acceptable (about 30 ms on a small integral), or none?
5. **B2 as a stopgap, and for `get_max_batch(f=...)`.** A 0.1 s limit per test
   call means a longer extrapolation of the fit. Acceptable?
6. **Your slow case.** Which integrand, device and `take_gradient` gave "twice
   the integration"? E should be run on it before the design is fixed; my cases
   in which the limit is reached are constructed, not real workloads.
7. **Only if D:** is cautious growth plus "out of memory in the backward pass
   raises, as today" acceptable, and what should happen on macOS and Windows?

## 6. Confidence

Sure (measured, or reproduced from the logs with padaquad's own functions):

- Which rule ended every search, and that E keeps the stability test from firing
  on this node.
- The history of the sleep in `_release_memory`.
- The scipy import time, and that subset least squares equals the nonnegative fit.
- A backward pass that raises can leave some gradients written.
- The GPU cache leftover for growth in steps of 10, and the cost of emptying it.
- The cause of the unstable limits (same free memory, different fitted N² term).
- Option E leaves results unchanged and removes the benchmark's cost on the
  cases of section 2.6 (measured on one node, like for like).

Less sure:

- Section 2.5 is a simulation on interpolated curves. It reproduces today's
  searches, but the changed rules were not run.
- Option E was emulated by patching the solver at run time, not implemented. A
  real implementation can differ in small costs. Timings are medians of 3 (5 in
  the second run) and scatter by 10 to 30 % between repeats, so differences
  between the two variants below that size are not established.
- Whether E makes the limit more stable was not tested. Where the limit was
  reached on the CPU, the three methods gave limits between 31,088 and 57,873,
  and I cannot separate the method from the changing free memory.
- E's list of untested situations (section 3, option E).
- The way out in D's item 1 (first pass whole, limit at least doubling) is my
  reasoning, not tested.
- That small first batches change results with `take_gradient=True` is read from
  the code, not run.
- All timings are from one many-core node; the CPU numbers depend on E and so on
  the CPU count.
- The integrands are one-dimensional and mostly cheap. The cases in which the
  memory limit is reached (section 2.6) are constructed: a lowered
  `total_mem_usage`, and one integrand with memory growing like N² on the GPU.
  None is a real workload.

Not known: the unexplained 0.6 to 0.8 s of the first benchmark of a process; any
behavior on macOS or Windows; behavior with a preloaded allocator such as
jemalloc (limitation 3).

## 7. Connections

- PyTorch Lightning's batch-size finder grows a training batch by doubling until
  an out-of-memory error (`power`), or doubles and then searches between the last
  success and the failure (`binsearch`). It finds the limit by failing, which
  padaquad cannot do on a CPU or in a backward pass. Source: the Lightning
  documentation of `batch_size_scaling`, version 1.1.8 (the interface has changed
  since),
  https://pytorch-lightning.readthedocs.io/en/1.1.8/api/pytorch_lightning.tuner.batch_size_scaling.html
- Hugging Face Accelerate's `find_executable_batch_size` starts at 128 and
  multiplies the batch by 0.9 after each out-of-memory error. Source:
  `src/accelerate/utils/memory.py` on the main branch (read by the reviewer on
  2026-11-17),
  https://raw.githubusercontent.com/huggingface/accelerate/main/src/accelerate/utils/memory.py
  The user guide still says "in half":
  https://huggingface.co/docs/accelerate/main/usage_guides/memory
- Option E is lazy evaluation applied to a measurement: compute it when it is
  first needed, and only as far as needed. The analogy is mine.

## 8. What the review changed

| Finding | Change in this file |
|---|---|
| F1, cause of the slow CPU search was wrong | Summary and section 2.3 rewritten: time rule and E, not memory |
| F2, D not supported as the default; results depend on the batch | Recommendation revised; D item 1 added; emulation caveats added |
| F3, slowdown after a CPU reading unsupported | Withdrawn in D item 6; listed as not known |
| F4, cause of the unstable limits | Section 2.4 gives the cause |
| F5, one-time cost is more than scipy | Section 2.3 item 4 |
| F6, ranges came from one script | Summary and section 2.2 use both scripts |
| F7, emulation table misleading | Ratio columns and caveats added |
| F8, A.1 also dropped the garbage collection | A.1 now removes only the sleep |
| F9, C misdescribed the code | Corrected in C; E states its own rule |
| F10, B2 dismissed without evidence | B2 simulated (section 2.5) and recommended |
| F11, starting N on the GPU | Section 2.3 item 6 and B1 |
| F12, fixed small batch | Section 2.4 states 48 to 60 % |
| F13, limitation 6 overstated | D item 7 |
| F14, why the cache leftover matters | D item 6 |
| F15, Python references are minor | D item 6 |
| F16, the "95 %" mixed sources | Section 2.3 item 1 names the script |
| F17, units in B4 | 2,500 GB |
| F18, Accelerate citation | Section 7 cites the source file |
| Missing options 7 and 8 | B2 and B5 |
| Missing risks 2, 3, 5, 6 | D item 7 |

After the review, option E was emulated and measured (section 2.6). That changed
the summary, option E's "Measured" list, the comparison table, the
recommendation and the decisions.
