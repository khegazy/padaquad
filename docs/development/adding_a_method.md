# Adding a method

A quadrature method in padaquad is an object, not a subclass: an instance of
`UniformMethodClass` (fixed node fractions and weights) or `VariableMethodClass`
(node fractions and a function that computes the weights from the actual node
positions). The solvers reach a method only through the interface of the
abstract `MethodClass` (`node_fractions()`, `weights(nodes)`, `degree`, `order`,
`error_exponent`), so a new method needs no change to the solvers.

## What a method needs

| Item | Meaning |
|---|---|
| node fractions `c` | Positions of the nodes in a subinterval scaled to $[0, 1]$: a 1-D float64 tensor starting at 0, ending at 1, never decreasing. Methods whose nodes do not include the ends add them with weight 0 (as the Gauss-Kronrod methods do). |
| weights `b` | The method's weights; the subinterval's integral is $h \sum_i b_i f(t_i)$. They sum to 1. |
| error weights `b_error` | `b` minus the weights of a lower-degree method on the same nodes; the error estimate is $h \sum_i b^\text{error}_i f(t_i)$. |
| `degree` | The highest polynomial degree `b` integrates exactly. `order` is computed as `degree + 1`. |
| `error_exponent` | The degree of the lower-degree method plus 2: the error estimate on a subinterval of width $h$ shrinks like $h^k$ with this $k$. The solver's merge rule uses it. |
| `references` | Keys of `docs/references.bib` for the sources of the nodes and weights. Add the source there first, after opening it and finding the numbers in it. |

## Example: Simpson's rule

Simpson's rule has the nodes $0, 1/2, 1$ and the weights $1/6, 2/3, 1/6$, and is
exact for polynomials up to degree 3. The trapezoid rule on the end nodes,
weights $1/2, 0, 1/2$ (degree 1), gives the error estimate.

```python
import math

import torch

from padaquad import UNIFORM_METHODS, integrate
from padaquad.methods import UniformMethodClass, _Tableau

f64 = torch.float64
simpson = UniformMethodClass(
    degree=3,  # Simpson's rule is exact for cubics
    error_exponent=3,  # the trapezoid rule (degree 1) is subtracted: 1 + 2
    tableau=_Tableau(
        c=torch.tensor([0.0, 0.5, 1.0], dtype=f64),
        b=torch.tensor([1 / 6, 2 / 3, 1 / 6], dtype=f64),
        # b minus the trapezoid weights [1/2, 0, 1/2]
        b_error=torch.tensor([1 / 6 - 1 / 2, 2 / 3, 1 / 6 - 1 / 2], dtype=f64),
    ),
)

# Check the claimed degrees against the weights before using the method.
measured = simpson.measure_degree()
print(measured.degree, measured.lower_degree)  # 3 1

# Make it available to the uniform solver by name (lower case).
UNIFORM_METHODS["simpson"] = simpson
result = integrate(
    torch.sin,
    "simpson",
    atol=1e-10,
    rtol=1e-10,
    mesh_init=torch.tensor([0.0], dtype=f64),
    mesh_final=torch.tensor([math.pi], dtype=f64),
    device="cpu",
    max_batch=10_000,
)
print(round(float(result.integral), 8))  # 2.0
```

Registering the method in `UNIFORM_METHODS` at run time makes it available to
every solver created afterwards; each solver works on its own copy.

## A method whose weights depend on the node positions

`VariableMethodClass(degree, error_exponent, c, weight_function, references)`
takes a `weight_function` instead of fixed weights. It receives the nodes of
every subinterval scaled to $[0, 1]$, a tensor of shape `[N, C, T]`, and returns
`(b, b_error)`, two tensors of shape `[N, C]` in the same dtype and on the same
device. It must be a function defined at module level, not a lambda or a nested
function, so that solvers can be copied and pickled.
`_interpolatory3_weights` in `padaquad/methods/interpolatory.py` is the
pattern to follow; register the method in `VARIABLE_METHODS`.

## Adding a method to padaquad itself

1. Define it in the module of its family (`padaquad/methods/runge_kutta.py`,
   `gauss_kronrod.py`, `clenshaw_curtis.py` or `interpolatory.py`) and add it to
   that module's registry dictionary (`RK_METHODS`, `GK_METHODS`,
   `CC_METHODS` or `INTERPOLATORY_METHODS`). `UNIFORM_METHODS` and
   `VARIABLE_METHODS` are built from these, and the methods table on the
   {doc}`methods page <../user_guide/methods>` is generated from them, so it
   gains a row automatically. A new family needs its own module, a registry,
   and an entry in `FAMILIES` in `docs/_ext/methods_table.py`.
2. Put a comment above each block of numbers saying where it comes from
   (author, year, venue, table or equation), and list the sources in
   `references`.
3. Run the checks that every method must pass:
   - `tests/test_exactness.py` measures the degree in both directions and checks
     `error_exponent` against the lower-degree method;
   - `tests/test_tableau.py` checks the weights against an independent
     calculation for each family; add one for a new family;
   - `examples/verification/polynomial_degree.py` and
     `examples/verification/integral_accuracy.py` include the method
     automatically;
   - `tests/test_docs_methods_table.py` checks that its row is complete and that
     its references are in the bibliography.
4. Regenerate the snapshot values for the new method
   (`TPD_REGENERATE_SNAPSHOTS=1 pytest tests/test_snapshots.py`) and say in the
   commit that only new entries were added.
