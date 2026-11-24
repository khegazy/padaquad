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
call, with `take_gradient=False` and a `max_batch` fixed before any timing:
100,000 for the analytic integrands, and for the networks the value of
`get_max_batch(f=...)`, so the search for a batch size is never timed. The
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
on 2026-11-24 (NVIDIA A100 and AMD EPYC 7763, padaquad commit `7ce788c`);
check them again whenever a results file changes. Speedups are padaquad
`gk21`'s.

- **Many subintervals on the GPU: padaquad is the fastest of the five on the
  four integrands that need about 1,000 subintervals.** It takes 38 to 65 ms
  on `chirp`, `spectrum`, `feature_network` and `feature_network_wide`, where
  the next fastest library takes 153 to 404 ms (scipy, on the CPU, on `chirp`
  and `spectrum`) or 316 and 358 ms (quadax, on the two networks). That is 3.4
  to 95 times as fast as scipy, 7.7 to 11 times as fast as quadax, 10 to 35
  times as fast as diffrax and 160 to 460 times as fast as torchdiffeq. Each
  of padaquad's calls of the integrand evaluates the nodes of every
  subinterval waiting to be checked; the other libraries evaluate one step or
  one subinterval at a time.
- **With fewer subintervals, the gap closes.** On `many_outputs` (126
  subintervals in quadax, 1,000 outputs) padaquad is 3 to 4 times as fast as
  scipy and diffrax and 0.88 times as fast as quadax; on `oscillatory` (64
  subintervals) it is 0.61 times as fast as scipy and 0.77 times as fast as
  quadax.
- **On the CPU, padaquad is the fastest on the two expensive networks**: 4.4
  and 2.1 times as fast as quadax, the next fastest, and 7 to 49 times as
  fast as scipy and diffrax. Each evaluation costs the most there, and
  padaquad evaluates many points per call of the network.
- **On cheap integrands on the CPU, diffrax and quadax are faster**: diffrax
  takes 2.4 ms on `chirp` (padaquad 26 ms) and quadax 0.48 ms on
  `oscillatory` (padaquad 12 ms). They compile the whole solve, integrand
  included, into one program (that this explains the times is our reading,
  not a measurement). padaquad is still 1.5 times as fast as quadax on
  `spectrum`, about as fast as diffrax on `many_outputs`, and 6.1 and 6.8
  times as fast as scipy on `chirp` and `spectrum`.
- **padaquad pays a fixed cost per call**: it takes 6 to 12 ms on the three
  integrands that need few subintervals (254 to 576 evaluations).
  `padaquad.integrate` builds a new solver and a starting mesh on every call;
  where the time goes has not been profiled. There scipy, quadax and (on the
  CPU) diffrax are faster, by up to 240 times (diffrax on `sharp_peak`, CPU).
- **ODE solvers can step over narrow features.** On `spectrum`, torchdiffeq's
  and diffrax's errors are $8\cdot10^{-6}$ to $1.1\cdot10^{-5}$, about a
  thousand times `atol`, while scipy, quadax and padaquad `gk21` stay below
  $10^{-9}$: a step whose error estimate misses a peak skips part of it. On
  the networks, their errors are $10^{-6}$ to $1.4\cdot10^{-5}$.
- **Accuracy of padaquad.** `gk21`'s error is at most $3\cdot10^{-8}$
  (`feature_network`, CPU): a test per subinterval allows a total error of
  many times `atol` ({ref}`perf-tolerance`). padaquad `dopri5`'s error is far
  larger on the integrands with many subintervals, up to $6\cdot10^{-4}$
  (`spectrum`, CPU): its error estimate is too small there. Use `gk21` for
  them.
- **Fewer timed runs.** torchdiffeq on `chirp` (GPU and CPU) and `spectrum`
  (GPU), and torchdiffeq and diffrax on `feature_network_wide` (CPU, 17 and
  26 s per run) were timed 2 to 6 times instead of 7; the tables say how
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
