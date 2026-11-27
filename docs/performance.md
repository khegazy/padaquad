# Performance

This page compares PadaQuad with four libraries that can compute the same
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
- **PadaQuad** evaluates the nodes of many subintervals in one call of the
  integrand, in three settings:
  - `dopri5`, the same Runge-Kutta weights used as a quadrature method;
  - `gk21`, its default method, with the default options;
  - `gk21 accelerate`, `gk21` as one would call it for a single integral
    that needs many subintervals: `split_rule="accelerate"`, which splits a
    subinterval in four where halving it would not be enough
    ({ref}`ug-four-pieces`), and `build_state_mesh=False`, which skips
    building the mesh a later call could start from (none of the other
    libraries builds one).

scipy evaluates NumPy versions of the integrands and always runs on the CPU,
also in the GPU section, where its column compares PadaQuad on the GPU with
scipy on the CPU. diffrax and quadax evaluate JAX versions and run on the
section's device, as torchdiffeq and PadaQuad do with PyTorch versions. All
versions come from one definition per integrand (the `xp` argument of the
integrands in `padaquad.examples`).

The integrands are chosen for the case PadaQuad is built for: most of them
need hundreds to thousands of subintervals at this tolerance (the heading of
each table gives the number quadax needed), so PadaQuad can evaluate the nodes
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
the exact integral, absolute and relative to the largest exact value. It
collects Python's garbage before each case, keeps every run's time in the
results file, and records the number of PyTorch threads and of CPUs the
process could use, which change the times ({ref}`ug-cpu-threads`). PadaQuad
and torchdiffeq run with 8 PyTorch threads (`--torch-threads`), a count close
to the fastest for PadaQuad's many small operations; JAX and scipy choose their
own.
PadaQuad runs through `padaquad.integrate`, which builds a new solver on every
call, with `take_gradient=False` and a fixed `max_batch` for each integrand
and device: the value PadaQuad's memory benchmark (`get_max_batch(f=...)`)
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
- **PadaQuad** accepts each subinterval on its own when

  $$
  \frac{\|e_k\|}{\max(\text{atol},\ \text{rtol}\,\|I\|)} < 1,
  $$ (perf-padaquad)

  with $e_k$ the raw difference between the method and its lower-degree
  companion ({doc}`user_guide/error_control`).

With `rtol = 0` all of them compare an error with the same number, `atol`,
and the max norm makes every output meet it. Two differences remain, and no
choice of tolerances removes them:

- **Local versus global.** torchdiffeq, diffrax and PadaQuad test each step or
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
on 2026-11-26 (NVIDIA A100 80 GB and AMD EPYC 7763, PadaQuad 2.0.0 at commit
`f8e78b2`, 8 PyTorch threads); check them again whenever a results file
changes. Speedups are PadaQuad `gk21`'s, with its default options, unless a
line names the `gk21 accelerate` row.

- **On the GPU, PadaQuad is the fastest of the five on all six integrands that
  need many subintervals.** `gk21` takes 14.5 to 28.8 ms on `chirp`,
  `spectrum`, `oscillatory`, `many_outputs`, `feature_network` and
  `feature_network_wide`. That is 1.1 to 140 times as fast as scipy, 1.3 to 25
  times as fast as quadax, 9.4 to 72 times as fast as diffrax and 130 to 950
  times as fast as torchdiffeq. The margin is smallest where the fewest
  subintervals are needed (`oscillatory`, 64 in quadax: 1.1 times scipy, 1.3
  times quadax) and largest on the expensive networks (63 and 140 times
  scipy). Each of PadaQuad's calls of the integrand evaluates the nodes of
  every subinterval waiting to be checked; the other libraries evaluate one
  step or one subinterval at a time.
- **On the CPU, PadaQuad is the fastest on the two networks and on
  `many_outputs`.** On the networks `gk21` is 5.1 and 2.4 times as fast as
  quadax, the next fastest library, and 8.3 to 57 times as fast as diffrax and
  scipy (PadaQuad's own `dopri5` row is faster still there, at a larger error).
  On `many_outputs` it is 2.8 times as fast as diffrax, 3.9 times as fast as
  quadax and 5.8 times as fast as scipy.
- **On cheap integrands on the CPU, diffrax and quadax are faster**: diffrax
  takes 2.2 ms on `chirp` (PadaQuad 13 ms) and quadax 0.33 ms on
  `oscillatory` (PadaQuad 8.1 ms). They compile the whole solve, integrand
  included, into one program (that this explains the times is our reading,
  not a measurement). PadaQuad is still 1.6 and 3.5 times as fast as quadax on
  `chirp` and `spectrum`, and 12 and 14 times as fast as scipy.
- **On the three integrands that need few subintervals**, PadaQuad takes 5.3
  to 18 ms on the GPU and 4.3 to 9.1 ms on the CPU, and scipy, quadax and
  diffrax are faster on most of them, by up to 190 times (diffrax on
  `sharp_peak`, CPU). On `neural_network_wide`, an expensive integrand, `gk21`
  is 2.9 times as fast as scipy on the GPU and 1.9 times on the CPU.
- **Why PadaQuad is slower there** (profiled on the same node with
  `padaquad.integrate` alone, `gk21`):
  - A call has a fixed cost: `damped_sine` finishes in one pass and takes
    4.5 ms on the GPU and 3.4 ms on the CPU, of which the integrand takes
    0.09 and 0.05 ms. On the GPU the call launches 157 small operations and
    waits for the GPU 25 times; the GPU is busy for 0.5 ms.
  - Each further pass (one call of the integrand on every waiting
    subinterval, then the error control and the mesh update) adds about 2.6 to
    3.0 ms on the GPU and 1 ms on the CPU, whatever the integrand costs:
    `sharp_peak` takes 7 passes and 20 ms on the GPU.
  - quadax spends about 0.15 ms per subinterval on this GPU (its loop runs
    `max_ninter + 1` iterations: 19.1 ms for 129 on `oscillatory`), and the
    compiled JAX solvers microseconds on the CPU. So on the GPU PadaQuad wins
    once a pass handles about 17 to 20 subintervals (our reading of these
    numbers), and on the CPU only when the integrand is expensive.
  - The `gk21 accelerate` row is 1.5 ms faster than `gk21` on the GPU and the
    CPU on `damped_sine` and `neural_network_wide`, where it evaluates the same
    points: that is the cost of building the mesh for a later call
    (`build_state_mesh=True`, the default), which a single integral does not
    need (our inference from equal point counts). Its four-way splits saved
    evaluations only on `many_outputs` (2,961 instead of 3,381 on the GPU) and,
    slightly, on `chirp`.
  - The causes that the earlier version of this page listed (the record copied
    to the CPU on every pass, kept in order through a dictionary, garbage
    collection) were fixed in PadaQuad 2.0.0
    (`docs/issues/per-turn-overhead.md` in the repository); what is left is
    the cost of many small operations launched from Python in every pass.
- **Run-to-run spread.** PadaQuad's slowest of 7 runs is within 6 % of its
  fastest for half of its rows. A few rows have one slow run among the seven
  (the first run of `chirp` with `dopri5` on the GPU, 96 ms instead of
  30 ms); the medians are not affected.
- **ODE solvers can step over narrow features.** On `spectrum`, torchdiffeq's
  and diffrax's errors are $7.6\cdot10^{-6}$ to $1.1\cdot10^{-5}$, about a
  thousand times `atol`, while scipy, quadax and PadaQuad `gk21` stay below
  $10^{-9}$: a step whose error estimate misses a peak skips part of it. On
  the networks, their errors are $2.8\cdot10^{-7}$ to $7\cdot10^{-6}$.
- **Accuracy of PadaQuad.** `gk21`'s error is at most $3.1\cdot10^{-8}$
  (`feature_network`, CPU): a test per subinterval allows a total error of
  many times `atol` ({ref}`perf-tolerance`). PadaQuad `dopri5`'s error is far
  larger on the integrands with many subintervals, up to $5.7\cdot10^{-4}$
  (`spectrum`, CPU): its error estimate is too small there. Use `gk21` for
  them.
- **Fewer timed runs.** torchdiffeq on `chirp` (GPU and CPU), `spectrum` (GPU)
  and `feature_network_wide` (CPU), and diffrax on `feature_network_wide`
  (CPU, 27 s per run) were timed 2 to 6 times instead of 7; the tables say
  how many.

## Reproduce

On a compute node with an 80 GB A100 and nothing else running (an interactive
allocation or a batch job, never a login node, whose other users change the
timings; on Perlmutter, `salloc -C "gpu&hbm80g"`). PadaQuad's `max_batch`
values were measured on that hardware; on other hardware the script stops and
asks for `--measure-max-batch`.

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
