# Error control

Each method computes both a primary integral and an embedded lower-order
estimate; their difference is the per-step error estimate. The adaptive
controller uses this to decide whether to accept a panel or split it.

Two orthogonal axes control the accept/reject decision: the **tolerance
reference** (`use_absolute_error_ratio`) and the **vector-error reduction
scheme** (`error_norm`).

## Tolerances

`atol` and `rtol` set the absolute and relative error targets (both default
`1e-5`). For a scalar integrand the accept threshold for the norm family is

$$|\text{step\_error}| < \max(\text{atol},\; \text{rtol}\cdot|I|)$$

(aligned with `scipy.integrate`; ratios $< 1$ accept, ratios $\geq 1$ split
the step at its midpoint).

## Tolerance reference (`use_absolute_error_ratio`)

- **Absolute** (default): the denominator is `atol + rtol * |total_integral|`.
  Every panel uses the same denominator. Best for path integrals where the
  total is the meaningful quantity.
- **Cumulative**: the denominator is `atol + rtol * |cumsum_to_step|`, growing
  with the running integral (traditional ODE-style). Per-panel ratios
  *decrease* as integration progresses.

When the integrand's mass is concentrated at later times, the absolute mode
can over-refine early panels because the running integral is still small.
The `error_integral_reference` argument accepts a float or tensor close to the
true total and uses it as the denominator from the first batch onward (no
effect in cumulative mode).

## Vector-error reduction (`error_norm`)

For vector-valued integrands (`D > 1`) the per-step error must be reduced to a
single accept/reject decision. Modeled on `scipy.integrate.quad_vec`'s `norm`,
there are two families — both reducing to
`|error| / (atol + rtol*|I|) < 1` when `D == 1`:

- **Norm family** — *reduce-then-compare* (scipy style): reduce the error
  vector to a scalar with the norm, compare to `atol + rtol * norm(integral)`,
  accept when `< 1`. Options:
    - `"2"` (default, L2 = $\sqrt{\sum e^2}$)
    - `"max"` (L∞)
    - `"rms"` ($\sqrt{\text{mean}(e^2)}$, padaquad's historical reduction)
    - a callable reducing the last (`D`) axis.

- **`"failure_fraction"`** — *per-component*: each output element is compared
  against its own `atol + rtol*|I_d|`; a panel is accepted when the fraction
  of failing elements is `<= mesh_failure_tolerance` (default `0.0` ⇒ every
  element must pass). This bounds the error of *every* output dimension rather
  than an aggregate.

Both families accept a panel with any non-finite (NaN/Inf) element (splitting
cannot fix it) and apply a scipy-style machine-precision **rounding floor** so
refinement stops once a panel's error is at or below round-off. The scheme is
set at construction (`error_norm=`, `mesh_failure_tolerance=`) and overridable
per call in [`integrate()`](../reference/padaquad/integrate.md).

## Merging

`error_ratios_2steps` is the combined error of consecutive step pairs, used
for merging. When `error_ratio_2steps < remove_cut` (default `0.1`), the pair
is merged. The controller ensures no two adjacent pairs are both flagged.

## Non-finite handling

With `error_on_nonfinite=True` (default), the solver raises a `ValueError`
naming the offending `t` when `f` returns NaN/Inf, preventing silent wrong
results.
