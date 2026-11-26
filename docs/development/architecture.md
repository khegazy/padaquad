# Architecture

## Package layout

| Module | Contents |
|---|---|
| `padaquad/_frontend.py` | `integrate()` and the `adaptive_quadrature()` factory |
| `padaquad/quadrature/base.py` | `AdaptiveQuadrature`: the integration loop, node placement, error ratios, the optimal mesh |
| `padaquad/quadrature/uniform.py`, `variable.py` | The two solvers; they differ only in the method registry they select |
| `padaquad/quadrature/memory.py` | The memory benchmark and the batch-size budget |
| `padaquad/methods/` | `MethodClass` and the method registries, one module per family |
| `padaquad/results.py` | `IntegrationResult` |
| `padaquad/base.py` | `SolverBase` (tolerances, dtype, warm-start state), `steps`, defaults |
| `padaquad/distributed.py` | Device selection and multi-process setup |
| `padaquad/examples.py` | Test integrands with known integrals |

## Methods

Every method is an instance of one of two classes built from data:
`UniformMethodClass` (fixed node fractions and weights) or
`VariableMethodClass` (node fractions and a function that computes the weights
from the node positions). Both implement the interface of the abstract
`MethodClass`: `node_fractions()`, `weights(nodes)`, `degree`, `order` and
`error_exponent`. The solver reaches a method only through this interface, so a
new method needs no solver change. A solver uses its method converted to its
device and dtype (`MethodClass.to`): a copy built once per process and shared
by every solver of that method, device and dtype, so the registries never
change. Never write into a method's tensors in place (for example
`method.tableau.b.mul_(2)`): the change would reach every solver that shares
the method.

## The integration loop

1. **Starting mesh.** Without a given `mesh`, the domain is cut into about
   $\sqrt{N_\text{init}}$ equal pieces, each split again at random points, for
   about `N_init_steps` subintervals ({ref}`ug-mesh`).
2. **Batches.** Subintervals that still need evaluating are taken in batches
   that fit in memory. All their nodes are stacked into one tensor and the
   integrand is called once per batch.
3. **Accept or split.** Each subinterval's error estimate is compared with its
   tolerance ({ref}`ug-error-control`). Accepted subintervals are recorded;
   the others are split in half and go back into the queue.
4. **Gradients.** With `take_gradient=True`, each batch's contribution is
   back-propagated as soon as it is accepted, so the autograd graph of the
   whole integral is never held at once ({ref}`ug-autodiff`).
5. **Optimal mesh.** After convergence, neighbouring subintervals whose
   combined error is small are merged and the result is returned as the
   state's mesh, the starting mesh of the next warm-started call
   ({ref}`ug-optimal-mesh`).
