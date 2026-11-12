# Performance

This page compares padaquad with torchdiffeq {cite:p}`torchdiffeq`, the
differentiable ODE solver of {cite:t}`chen2018`, on integrals with known
answers. Every number below comes from `docs/performance/results.json`, written
by `examples/speed_test/torchdiffeq_comparison.py`; the table is rendered from
that file when this page is built.

## What is compared

To compute $\int_0^1 f(t)\,dt$, torchdiffeq solves the ODE $dy/dt = f(t)$ with
$y(0) = 0$. An ODE solver must take its steps one after another, because in
general each step needs the state the previous one produced. padaquad uses the
fact that the integrand does not depend on $y$: it evaluates the nodes of many
subintervals in one call of $f$. The comparison measures what that buys on one
machine:

- **torchdiffeq `dopri5`**, solving $dy/dt = f(t)$;
- **padaquad `dopri5`**, the same Runge-Kutta weights used as a quadrature
  method, so the difference is mostly the batching;
- **padaquad `gk21`**, padaquad's default method.

The integrands are the five entries of `integrand_dict` (all on $[0, 1]$) and a
small neural network with fixed weights (two hidden layers of 64 tanh units;
reference value from `scipy.integrate.quad` at a tolerance of $10^{-14}$). Each
solver runs at `atol = rtol` of $10^{-6}$ and $10^{-9}$.

For each case the script records the median wall time of 7 runs after one
warm-up run (with `torch.cuda.synchronize` around each run on a GPU), the
smallest and largest time, the number of points at which $f$ was evaluated,
and the absolute error against the exact answer. padaquad runs with a fixed
`max_batch`, so its one-time memory benchmark is not timed, and with
`take_gradient=False`, because no gradient is computed (torchdiffeq's forward
solve is timed without one too).

**Read the errors as well as the times.** The two libraries apply the
tolerance differently: torchdiffeq tests each step's error against
$\text{atol} + \text{rtol}\,|y|$, padaquad each subinterval's against
$\max(\text{atol}, \text{rtol}\,|I|)$ ({doc}`user_guide/error_control`). Equal
tolerances therefore do not give equal accuracy.

## Results

```{include} reference/generated/performance_table.md
```

## What the results show

The two lines under the table are generated from the results file, like the
table itself.

- The gain is smallest on the simplest integrands (`t`, `t_squared`), where
  both solvers finish in about a millisecond and fixed per-call costs
  dominate, and largest where torchdiffeq needs many sequential steps
  (`sine_squared`, `damped_sine` at the tighter tolerance).
- On these smooth integrands `gk21` meets both tolerances on the starting mesh,
  without splitting a subinterval: its number of evaluations is the same at
  both tolerances.
- **Not measured here:** GPU runs. The batching is expected to matter more on a
  GPU, where one call on many points costs little more than a call on one
  point, but no GPU result is claimed until it is measured.

## Reproduce

```bash
pip install -e ".[dev]"   # includes torchdiffeq
python examples/speed_test/torchdiffeq_comparison.py --device cpu \
    --out docs/performance/results.json
```

Use `--device cuda` for a GPU. Commit the results file only from a clean
working tree: the file records the git commit, and marks uncommitted changes.
