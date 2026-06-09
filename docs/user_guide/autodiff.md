# Differentiable integration

Every step of the integration is differentiable, so you can take gradients of
the result with respect to anything the integrand depends on. There are two
equivalent ways to compute $\nabla_\theta \int f_\theta(t)\,dt$:

```python
import torch
from padaquad import adaptive_quadrature, steps

theta = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
mesh_init = torch.tensor([0.0])
mesh_final = torch.tensor([torch.pi])


# (A) Backprop through the integral.
solver = adaptive_quadrature(
    sampling_type=steps.ADAPTIVE_UNIFORM,
    method="gk21",
    atol=1e-8,
    rtol=1e-8,
)
solver.integrate(
    f=lambda t: theta * torch.sin(t),
    mesh_init=mesh_init,
    mesh_final=mesh_final,
    take_gradient=True,  # per-batch backward; accumulates into theta.grad
)
print(theta.grad)  # 2.0


# (B) Integrate the gradient of the integrand directly.
df_dtheta = lambda t: torch.sin(t)  # closed-form derivative
out_b = solver.integrate(f=df_dtheta, mesh_init=mesh_init, mesh_final=mesh_final)
print(out_b.integral)  # also 2.0
```

The two paths agree to machine precision on smooth integrands; this
consistency is verified in `tests/test_autodiff_consistency.py`.

## Memory-bounded gradients

Setting `take_gradient=True` makes the solver call `loss.backward()` after each
accepted batch instead of holding the full autograd graph until the end. This
is essential when the number of panel evaluations is large enough that the
graph would otherwise exceed GPU memory: per-batch backward accumulates
gradients into `theta.grad` without holding the full graph.

Results from each batch after backward are `.detach()`-ed before being added
to the running record, so the graph for that batch can be released.

## The loss function

`loss_fxn` defaults to returning the integral itself. Because the default loss
is linear in the integral, the per-batch backward path is correctness-safe.
Supply your own `loss_fxn` to optimize a different objective of the integral.

## Training-loop pattern

```python
solver = adaptive_quadrature(method="gk21", atol=1e-8, rtol=1e-8)

for epoch in range(N_epochs):
    result = solver.integrate(
        f=f_theta,
        mesh_init=mesh_init,
        mesh_final=mesh_final,
        reuse_mesh=(epoch > 0),  # warm-start from the previous optimal mesh
        take_gradient=True,
    )
    optimizer.step()
    optimizer.zero_grad()
```

See [Advanced features](advanced.md) for warm-starting details, and the
[`IntegrationResult`](../reference/padaquad/results.md) reference for the
`loss`, `gradient_taken`, and `y0` training-loop diagnostics.
