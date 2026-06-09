# The method portfolio

The library ships thirteen quadrature rules across three families. Select one
by passing its name as the `method` argument to
[`integrate()`](../reference/padaquad/integrate.md) or
[`adaptive_quadrature()`](../reference/padaquad/runge_kutta.md).

| Family | Methods | Polynomial exactness | Notes |
|---|---|---|---|
| Gauss-Kronrod | `gk7`, `gk15`, `gk21`, `gk31` | 10 / 22 / 31 / 46 | Embedded $G_n$ / $K_{2n+1}$ pair, the canonical adaptive-quadrature workhorse since 1965 (Laurie 1997). `gk21` is the default. `gk7` (G3-K7, 9 evaluation slots) fills the low-node end. |
| Clenshaw-Curtis | `cc5`, `cc9`, `cc17`, `cc33`, `cc65` | 4 / 8 / 16 / 32 / 64 | Chebyshev nodes, **nested** by doubling. Excellent on analytic integrands (Trefethen 2008). |
| Runge-Kutta | `adaptive_heun`, `fehlberg2`, `bosh3`, `dopri5` | 1 / 1 / 2 / 4 | Embedded RK pairs from the ODE-solver literature. |

Plus two variable-node methods (`adaptive_heun` and `interpolatory3_variable`)
that re-weight existing evaluations on mesh splits — useful when integrand
evaluations are expensive.

## Choosing a method

For smooth integrands at moderate-to-high accuracy, prefer `gk21` (the
default) or `cc33`. The low-node methods (`gk7`, `cc5`, `cc9`) fill the gap
between the RK rules and the headline `gk15`/`cc17`: useful when the adaptive
controller refines aggressively and a high-node rule would waste evaluations
on smooth regions. The RK methods are kept for backwards-compatibility and as
low-order baselines.

## Where the methods live

The method registries are organized by family in the `padaquad.methods`
package:

- [`padaquad.methods.runge_kutta`](../reference/padaquad/methods/runge_kutta.md)
  — `adaptive_heun`, `fehlberg2`, `bosh3`, `dopri5`
- [`padaquad.methods.gauss_kronrod`](../reference/padaquad/methods/gauss_kronrod.md)
  — `gk7`, `gk15`, `gk21`, `gk31` (with builder)
- [`padaquad.methods.clenshaw_curtis`](../reference/padaquad/methods/clenshaw_curtis.md)
  — `cc5`, `cc9`, `cc17`, `cc33`, `cc65` (with FFT-based weights)
- [`padaquad.methods.interpolatory`](../reference/padaquad/methods/interpolatory.md)
  — variable `adaptive_heun`, `interpolatory3_variable`

The available method names are exposed as the
[`UNIFORM_METHODS`](../reference/padaquad/methods/index.md) and
`VARIABLE_METHODS` registries in the public API.
