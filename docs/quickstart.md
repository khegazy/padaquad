# Quickstart

## Installation

```bash
pip install padaquad
```

Or from source:

```bash
git clone https://github.com/khegazy/padaquad.git
cd padaquad
pip install -e .
```

Runtime dependencies: Python 3.10+, PyTorch, NumPy, SciPy, einops, psutil.

### For developers

```bash
pip install -e ".[dev]"
pre-commit install
```

The dev extras add pytest, ruff, mypy, pre-commit, typos, and torchdiffeq
(used by the speed-test benchmark only). To build this documentation site
locally, install the docs extras instead:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Your first integral

```python
import math
import torch
from padaquad import integrate

result = integrate(
    f=lambda t: torch.sin(t),
    method="gk21",  # default; Gauss-Kronrod 21-point pair (G10-K21)
    mesh_init=torch.tensor([0.0]),
    mesh_final=torch.tensor([math.pi]),
)
print(result.integral)  # tensor([2.0000])
print(result.integral_error)  # estimated absolute error
print(result.converged)  # True
```

The integrand `f` takes a tensor `t` of shape `[N, T]` (batched time
points) and returns a tensor of shape `[N, D]` (batched output values).
Because `f` depends only on `t`, the solver can evaluate many panels'
quadrature points simultaneously on GPU.

## One-shot vs. repeated calls

[`integrate()`](reference/padaquad/integrate.md) is the one-shot entry point:
it constructs a solver, runs the integration, and returns an
[`IntegrationResult`](reference/padaquad/results.md). For repeated calls — such
as a training loop where the integrand changes slightly each iteration —
instantiate the solver once via
[`adaptive_quadrature()`](reference/padaquad/runge_kutta.md) so that warm-start
cache state persists across iterations:

```python
from padaquad import adaptive_quadrature

solver = adaptive_quadrature(method="gk21", atol=1e-8, rtol=1e-8)
result = solver.integrate(f=..., mesh_init=..., mesh_final=...)
```

See the [User Guide](user_guide/concepts.md) for a deeper tour, and
[Differentiable integration](user_guide/autodiff.md) for taking gradients
through the integral.
