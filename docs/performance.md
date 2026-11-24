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
($1.49\cdot10^{-8}$), far above float64 rounding, and loose enough for the ODE
solvers to finish the oscillatory integrand in seconds.

## Results

```{include} reference/generated/performance_table.md
```

## What the results show

The lines under each table are generated from its results file, like the
tables. The observations below are written by hand from the files measured
on 2026-11-23 (NVIDIA A100 and AMD EPYC 7763, padaquad commit `ea7247f`);
check them again whenever a results file changes.

- **padaquad pays a fixed cost per call**: `padaquad.integrate` takes at
  least about 5 to 7 ms even on `t`. Each call builds a new solver and a
  starting mesh; where the time goes has not been profiled. On a cheap
  integrand that needs few evaluations (`t`, `exp`, `sine_squared`), this cost
  is the whole time, and scipy (tens of microseconds) and the compiled JAX
  libraries (below a millisecond on the CPU) are faster, by up to about 580
  times (diffrax on `kink`, CPU).
- **Against torchdiffeq**, which takes its ODE steps one after another,
  padaquad `gk21` is faster on every integrand on the GPU (1.6 to 63 times;
  most on `oscillatory`, where torchdiffeq needs about 5,000 evaluations in
  sequence) and on every integrand but `t` and `t_squared` on the CPU.
- **With many outputs at once** (`many_outputs`, 1,000 integrals), padaquad
  `gk21` is the fastest of all five on the CPU, 1.05 times as fast as diffrax,
  1.3 times as fast as quadax and twice as fast as scipy's `quad_vec`, and on
  the GPU it is 2.9 times as fast
  as scipy and 3.8 times as fast as diffrax, and 0.82 times as fast as quadax.
- **JAX's compiled solvers are fast on the CPU**: diffrax and quadax compile
  the whole solve, integrand included, into one program, so on the analytic
  integrands with one output they take between 0.04 and 0.42 ms on the CPU.
  On the GPU they take longer on those integrands (0.8 to 140 ms), and
  padaquad `gk21` is faster than diffrax on `sine_squared`, `oscillatory` and
  `many_outputs`. (That each of their sequential steps waits for GPU work is
  our explanation, not a measurement.)
- **None of these integrands is both expensive and in need of many
  subintervals**, the case padaquad is built for: the two networks need only a
  few subintervals at this tolerance (padaquad never splits its 11 starting
  subintervals, 254 evaluations). Still, on the CPU, where the wide network's
  evaluations cost the most, padaquad `gk21` is 3.1 times as fast as diffrax
  and 1.3 times as fast as scipy on `neural_network_wide`.
- **Accuracy.** padaquad `gk21`'s error is below `atol` on every integrand
  (at most $7\cdot10^{-9}$). scipy and quadax, the strictest by construction
  ({ref}`perf-tolerance`), reach errors near float64 rounding. padaquad
  `dopri5`'s error reaches $8\cdot10^{-7}$ (`oscillatory`, CPU), and the ODE
  solvers' $10^{-6}$ to $3\cdot10^{-6}$ on `kink`: tests per step or per
  subinterval allow a total error of many times `atol`.

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
