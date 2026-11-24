# Performance

This page compares padaquad with four libraries that can compute the same
integrals: torchdiffeq {cite:p}`torchdiffeq` and diffrax {cite:p}`kidger2021`,
which solve differential equations, and scipy's `quad`
{cite:p}`quadpack_netlib` and quadax {cite:p}`quadax`, which are adaptive
quadrature routines. Every number below comes from the files
`docs/performance/results_<device>.json`, written by
`examples/speed_test/library_comparison.py`; the tables are rendered from
them when this page is built.

## What is compared

Each library integrates the same functions over $[0, 1]$, each with a known
integral:

- **torchdiffeq `dopri5`** (PyTorch) and **diffrax `Dopri5`** (JAX) solve the
  ODE $dy/dt = f(t)$ with $y(0) = 0$. An ODE solver takes its steps one after
  another, because in general each step needs the state the previous one
  produced.
- **scipy `quad`** (QUADPACK; `quad_vec` for the integrand with many outputs)
  and **quadax `quadgk`** (21 nodes per subinterval, JAX) split $[0, 1]$
  adaptively, one subinterval at a time.
- **padaquad `dopri5`** (the same Runge-Kutta weights used as a quadrature
  method) and **padaquad `gk21`** (its default method) evaluate the nodes of
  many subintervals in one call of the integrand.

scipy evaluates NumPy versions of the integrands and always runs on the CPU,
also in the GPU section, where its column compares padaquad on the GPU with
scipy on the CPU. diffrax and quadax evaluate JAX versions and run on the
section's device, as torchdiffeq and padaquad do with PyTorch versions. All
versions come from one definition per integrand (the `xp` argument of the
integrands in `padaquad.examples`).

The integrands are chosen for the case padaquad is built for: most of them
need hundreds to thousands of subintervals at this tolerance (the heading of
each table gives the number quadax needed), so padaquad can evaluate the nodes
of many subintervals at once, while an ODE solver takes thousands of steps in
sequence and scipy and quadax split one subinterval at a time.

- **Many subintervals:**
  - `chirp`: $2t\sin(10^4 t^2)$, about 1,600 periods whose frequency grows
    along $t$;
  - `spectrum`: 200 peaks of width $10^{-4}$ spread over $[0, 1]$;
  - `oscillatory`: 100 periods of $e^{-t}\sin(200\pi t)$;
  - `many_outputs`: $\sin(kt)$ for $k = 1, \dots, 1000$ at once;
  - `feature_network` and `feature_network_wide`: tanh networks with fixed
    random weights whose input is $\sin(2\pi k t)$ and $\cos(2\pi k t)$
    for $k = 1, 2, 4, \dots, 512$ (two hidden layers of 256 units; three of
    1,024). They are expensive to evaluate and oscillate up to 512 times.
- **Few subintervals, for contrast:** `damped_sine` (3.7 periods of a decaying
  sine), `sharp_peak` (one peak of width $10^{-3}$) and `neural_network_wide`
  (a tanh network of $4t$ with four hidden layers of 512 units: expensive but
  smooth).

The analytic integrands come from `hard_integrand_dict` and `integrand_dict`,
and the {doc}`examples <auto_examples/index>` show those of
`hard_integrand_dict`. The networks' reference integrals come from
`scipy.integrate.quad` at a tolerance of $10^{-14}$.

For each case the script records the median wall time of 7 runs after one
untimed warm-up run (which also compiles the JAX solvers), or of fewer runs
(at least one; the table says how many) when the warm-up run shows that 7 would
take more than a minute, with
`torch.cuda.synchronize` or `jax.block_until_ready` around each run on a GPU;
the smallest and largest time; the number of points at which the integrand was
evaluated (for diffrax, counted while it runs, in an untimed run); whether the
solver reported success (a † marks one that did not); and the error against
the exact integral, absolute and relative to the largest exact value.
padaquad runs through `padaquad.integrate`, which builds a new solver on every
call, with `take_gradient=False` and a fixed `max_batch` for each integrand
and device: the value padaquad's memory benchmark (`get_max_batch(f=...)`)
measured once on the hardware of the results (`MAX_BATCH` in the script), so
no run measures memory. No case needs more than about 110,000 evaluations in
all, so no call of the integrand is limited by it. The
benchmark runs on an otherwise idle compute node, never on a shared login
node.

(perf-tolerance)=
## One tolerance for all

Every library runs at `atol = 1e-8` and `rtol = 0`, and measures the error of
a vector-valued integrand by its largest component (the max norm). The
libraries decide that they are done in different ways; this is why that
choice is the closest to one common tolerance. In the notation below, $e$ is
an error estimate, $y$ the running integral (the ODE state) and $I$ the
integral.

- **torchdiffeq and diffrax** accept an ODE step from $y_n$ to $y_{n+1}$ when

  $$
  \left\| \frac{e_n}{\text{atol} + \text{rtol}\,\max(|y_n|, |y_{n+1}|)} \right\| \le 1
  $$ (perf-ode)

  (diffrax: $< 1$), where $e_n$ is the difference between the 5th- and
  4th-order solutions over the step; the norm is RMS by default.
- **scipy and quadax** stop when the sum of all subintervals' error estimates
  is small enough:

  $$
  \sum_k e_k \le \max(\text{atol},\ \text{rtol}\,\|I\|),
  $$ (perf-quadpack)

  with QUADPACK's error estimate, which enlarges the raw difference between
  the Kronrod and Gauss results by up to a factor of about 200 (`quad_vec`
  stops at a sum below an eighth of the tolerance {cite:p}`scipy_quadvec`).
- **padaquad** accepts each subinterval on its own when

  $$
  \frac{\|e_k\|}{\max(\text{atol},\ \text{rtol}\,\|I\|)} < 1,
  $$ (perf-padaquad)

  with $e_k$ the raw difference between the method and its lower-degree
  companion ({doc}`user_guide/error_control`).

With `rtol = 0` all of them compare an error with the same number, `atol`,
and the max norm makes every output meet it. Two differences remain, and no
choice of tolerances removes them:

- **Local versus global.** torchdiffeq, diffrax and padaquad test each step or
  subinterval on its own, so their total error can reach the number of steps
  or subintervals times `atol`; scipy and quadax bound the sum.
- **The size of the estimate.** QUADPACK's enlarged estimate is larger than
  the raw one, so scipy and quadax are the strictest. (An ODE solver
  continues with its 5th-order solution while it estimates the error of the
  4th-order one, so its actual error is usually well below its estimate.)

So equal tolerances do not give equal accuracy: **read the errors as well as
the times.** `atol = 1e-8` is close to scipy's and quadax's default
($1.49\cdot10^{-8}$), far above float64 rounding, and loose enough for every solver to finish
every integrand in under half a minute per run.

## Results

```{include} reference/generated/performance_table.md
```

## What the results show

The lines under each table are generated from its results file, like the
tables. The observations below are written by hand from the files measured
on 2026-11-24 (NVIDIA A100 80 GB and AMD EPYC 7763, padaquad commit
`12ebfef`); check them again whenever a results file changes. Speedups are
padaquad `gk21`'s.

- **Many subintervals on the GPU: padaquad is the fastest of the five on the
  four integrands that need about 1,000 subintervals.** It takes 42 to 61 ms
  on `chirp`, `spectrum`, `feature_network` and `feature_network_wide`, where
  the next fastest library takes 156 and 421 ms (scipy, on the CPU, on `chirp`
  and `spectrum`) or 326 and 358 ms (quadax, on the two networks). That is 3.7
  to 84 times as fast as scipy, 5.3 to 12 times as fast as quadax, 9.4 to 38
  times as fast as diffrax and 130 to 500 times as fast as torchdiffeq. Each
  of padaquad's calls of the integrand evaluates the nodes of every
  subinterval waiting to be checked; the other libraries evaluate one step or
  one subinterval at a time.
- **With fewer subintervals, the gap closes.** On `many_outputs` (126
  subintervals in quadax, 1,000 outputs) padaquad is 3 and 3.9 times as fast
  as scipy and diffrax and 0.87 times as fast as quadax; on `oscillatory` (64
  subintervals) it is 0.45 times as fast as scipy and 0.54 times as fast as
  quadax.
- **On the CPU, padaquad is the fastest on the two expensive networks**: 4.3
  and 1.7 times as fast as quadax, the next fastest, and 6.1 to 40 times as
  fast as scipy and diffrax. Each evaluation costs the most there, and
  padaquad evaluates many points per call of the network.
- **On cheap integrands on the CPU, diffrax and quadax are faster**: diffrax
  takes 2.2 ms on `chirp` (padaquad 41 ms) and quadax 0.32 ms on
  `oscillatory` (padaquad 12 ms). They compile the whole solve, integrand
  included, into one program (that this explains the times is our reading,
  not a measurement). padaquad is still 1.8 times as fast as quadax on
  `spectrum`, 1.4 and 1.1 times as fast as quadax and diffrax on
  `many_outputs`, and 3.9 and 7.2 times as fast as scipy on `chirp` and
  `spectrum`.
- **On the three integrands that need few subintervals**, padaquad takes 5.6
  to 29 ms, and scipy, quadax and (on the CPU) diffrax are faster, by up to
  240 times (diffrax on `sharp_peak`, CPU).
- **Why padaquad is slower there** (measured on the same node with
  `padaquad.integrate` alone; `many_outputs` on the GPU, 35 to 44 ms per
  call):
  - The integrand takes 0.6 ms of it. The GPU is busy for 7 ms: one call
    launches about 1,300 small GPU operations and waits for the GPU 227
    times (to read a result back, for example whether any subinterval is
    still waiting, or for a check), so the time is set by Python and these
    waits, not by the arithmetic.
  - That cost comes back on every turn of the loop (one call of the
    integrand, then splitting and recording): about 3 to 4 ms per turn, and 5 to
    10 turns per integral. The number of turns grows with how many times a
    subinterval is halved, not with the number of subintervals; quadax's time
    grows with the number of subintervals, at about 0.15 ms per iteration on
    this GPU. So padaquad wins once each of its turns handles many
    subintervals (our reading of these numbers: about 20 or more).
  - The record of accepted subintervals, including the integrand's values at
    every node (1,000 outputs here), is copied to the CPU after every turn
    (`result_device="cpu"`, the default) and rebuilt on every insertion: 13 ms
    of the 35 ms. With `result_device="cuda"` the call takes 22 ms, faster
    than quadax's 37 ms.
  - After the loop, padaquad builds the mesh it returns for a later call
    (`state.mesh`), merging subintervals with predicted errors: 2 to 6 ms on
    every call, although the benchmark does not reuse it.
  - On the CPU, torch's 128 threads slow these many small operations: with 8
    threads `many_outputs` takes 39 ms instead of 54 ms, `spectrum` 41 instead
    of 58, `chirp` 20 instead of 26. The benchmark uses all 128, as XLA does.
- **padaquad's times vary from run to run** more than the other libraries':
  its slowest of 7 runs is often 2 to 6 times its fastest, and a median can
  move by 30 to 80 % between two runs (`damped_sine` on the GPU: 6.8 ms on
  the 40 GB A100, 12 ms here; 7 ms when timed alone on this node). Its time
  is mostly small operations on the CPU, which react to anything else the
  machine does (a likely cause, not a measured one).
- **ODE solvers can step over narrow features.** On `spectrum`, torchdiffeq's
  and diffrax's errors are $7.6\cdot10^{-6}$ to $1.1\cdot10^{-5}$, about a
  thousand times `atol`, while scipy, quadax and padaquad `gk21` stay below
  $10^{-9}$: a step whose error estimate misses a peak skips part of it. On
  the networks, their errors are $3\cdot10^{-7}$ to $7\cdot10^{-6}$.
- **Accuracy of padaquad.** `gk21`'s error is at most $3\cdot10^{-8}$
  (`feature_network`, CPU): a test per subinterval allows a total error of
  many times `atol` ({ref}`perf-tolerance`). padaquad `dopri5`'s error is far
  larger on the integrands with many subintervals, up to $6\cdot10^{-4}$
  (`spectrum`, CPU): its error estimate is too small there. Use `gk21` for
  them.
- **Fewer timed runs.** torchdiffeq on `chirp` (GPU and CPU) and `spectrum`
  (GPU), and torchdiffeq and diffrax on `feature_network_wide` (CPU, 16 and
  27 s per run) were timed 2 to 6 times instead of 7; the tables say how
  many.

## Reproduce

On a compute node with a GPU and nothing else running (an interactive
allocation or a batch job, never a login node, whose other users change the
timings):

```bash
examples/speed_test/run_benchmark.sh setup   # once: builds the environment
examples/speed_test/run_benchmark.sh         # GPU run, then CPU run
```

`setup` uses [uv](https://docs.astral.sh/uv/) to build an environment with the
CUDA builds of PyTorch and JAX and `pip install -e ".[bench]"` (the compared
libraries, at the versions measured here); `PADAQUAD_BENCH_VENV` sets where
(default `$SCRATCH/padaquad_bench`). The run writes
`docs/performance/results_cuda.json` and `results_cpu.json`, which record the
commit they measured; commit them from a clean working tree, rebuild this
page, and revise "What the results show" for the new numbers.
