(ug-autodiff)=
# Gradients

Every operation in the integration is a PyTorch operation, so the integral can
be differentiated with respect to anything the integrand depends on. There are
two ways to compute $\nabla_\theta \int f_\theta(t)\,dt$:

```{include} ../../README.md
:start-after: "<!-- include-start: autodiff -->"
:end-before: "<!-- include-end: autodiff -->"
```

Path B uses the fact that differentiation and integration can be exchanged
when the integration domain does not depend on $\theta$ and $f_\theta$ is
smooth in $\theta$. The script
{doc}`../auto_examples/gradient_of_integral` runs both paths.

## `take_gradient`

With `take_gradient=True` (the default), `loss.backward()` runs after each
batch, so gradients accumulate in the parameters' `.grad` without the autograd
graph of the whole integration being held in memory. The recorded results are
then detached: `result.integral` cannot be differentiated again, and the
gradient is already in `.grad`. Call the optimizer's `zero_grad` before each
integration, as for any loss.

With `take_gradient=False`, the graph is kept, so the returned
`result.integral` can be differentiated like any tensor. This needs memory for
the graph of every evaluation of the integrand.

## `loss_fxn`

`loss_fxn` maps one batch's `IntegrationResult` to a scalar tensor; with
`take_gradient=True` its result is back-propagated after each batch, and
`result.loss` is the sum over batches. If it is None, the loss is the batch's
integral, so the accumulated gradient is the gradient of the integral. Because
the loss is applied batch by batch, a loss that is not linear in the integral
(for example its square) gives the sum of the batches' losses, not the loss of
the total: compute such a loss from `result.integral` with
`take_gradient=False` instead.
