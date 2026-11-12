# Error control

(ug-error-control)=
## Tolerances and error ratios

On each subinterval $k$, a method computes the subinterval's integral $s_k$
and an error estimate $e_k$: the difference between the method and its
lower-degree {term}`embedded method` ({doc}`methods`). The subinterval is
accepted when its {term}`error ratio`

$$
r_k = \frac{|e_k|}{\tau_k}, \qquad
\tau_k = \max\!\bigl(\text{atol},\ \text{rtol}\,|I|,\ 50\,\varepsilon\,|s_k|\bigr),
$$ (error-ratio)

is below 1, and split at its midpoint otherwise. Here `atol` and `rtol` are
the absolute and relative tolerances, $I$ is the integral (see
{ref}`ug-error-reference` for which value of it), and $\varepsilon$ is the
machine epsilon of the dtype ($2.2 \times 10^{-16}$ in float64,
$1.2 \times 10^{-7}$ in float32).

The last term is the **rounding floor**: rounding alone makes an error of
about this size on a subinterval, so the tolerance is never set below it, and
refinement stops once a subinterval's error is at rounding level. This matters
when the requested tolerance is smaller than the dtype can resolve. SciPy's
`scipy.integrate.quad_vec` has a similar floor {cite:p}`scipy_quadvec`, with
three differences: it uses $50\,\varepsilon$ times the integral of $|f|$ over
the subinterval rather than the absolute value of the subinterval's
contribution; it raises the error estimate rather than the tolerance; and it
applies the floor only when it is larger than the smallest positive double.
QUADPACK also rescales its error estimate {cite:p}`quadpack_netlib`, which
padaquad does not.

The error estimate measures the embedded method's error, so it is usually much
larger than the error of the integral padaquad returns, which uses the
higher-degree method {cite:p}`gonnet2012`, footnote 20. The script
{doc}`examples/verification/integral_accuracy.py <../auto_examples/verification/integral_accuracy>`
compares actual errors with the tolerances; for the Gauss-Kronrod and
Clenshaw-Curtis methods the actual errors are many orders of magnitude below
them. The exception is `fehlberg2`, whose error estimate ignores curvature
({doc}`methods`).

### Why the error estimate shrinks like $h^{d+2}$

The solver's merge rule ({ref}`ug-optimal-mesh`) assumes that the error
estimate of a subinterval of width $h$ shrinks like $h^k$ with
$k = d + 2$, where $d$ is the degree of the embedded method. This is a
derivation (shown for Simpson's rule by {cite:t}`gonnet2012`, Eqs. (4)–(5)):
write $f$ on the subinterval as its Taylor polynomial of degree $d$ plus a
remainder of size $C h^{d+1}$. The embedded method integrates the polynomial
exactly, so its error is the error on the remainder, which is integrated over
a width $h$:

$$
e = O\!\left(h \cdot h^{d+1}\right) = O\!\left(h^{d+2}\right).
$$ (error-scaling)

Each method stores $k$ as `error_exponent`, and the test suite checks
$k = d + 2$ against the measured degree of every embedded method.

### Precision

padaquad runs in float64 (the default) or float32, following the dtype of the
bounds or mesh given to `integrate`. float16 is refused with a `ValueError`:
its rounding error, about $10^{-3}$, is larger than typical tolerances, so the
solver could not tell whether a subinterval meets its tolerance.

### Non-finite values

If the integrand returns NaN or Inf, `integrate` raises a `ValueError` naming
the offending points (`error_on_nonfinite=True`, the default). With
`error_on_nonfinite=False`, subintervals with a non-finite error ratio are
accepted instead of split, the non-finite value reaches the returned integral,
and `converged` is `False`. Either way the run ends.

(ug-error-norm)=
## Integrands with several outputs

When the integrand has $D > 1$ outputs, each subinterval has an error vector
$e_k$, and `error_norm` turns it into one accept-or-split decision:

- `"2"`, `"max"`, `"rms"`, or a callable norm: the norm
  $R(e_k)$ of the error vector is compared with
  $\max(\text{atol}, \text{rtol}\, R(I))$. `"2"` is the Euclidean norm, `"max"`
  the largest absolute component, `"rms"` the root-mean-square; a callable must
  reduce the last axis of a `[..., D]` tensor.
- `"failure_fraction"`: each component $d$ is compared with its own tolerance
  $\text{atol} + \text{rtol}\,|I_d|$, and the subinterval is accepted when the
  fraction of failing components is at most `mesh_failure_tolerance` (0 means
  every component must pass). This bounds the error of every output, not
  an aggregate. This scheme is padaquad's own.

For $D = 1$ the norm options all reduce to Eq. {eq}`error-ratio`.

**Relation to SciPy.** The options `"2"` and `"max"` and the tolerance
$\max(\text{epsabs}, \text{epsrel}\cdot\text{norm}(I))$ are those of
`scipy.integrate.quad_vec` {cite:p}`scipy_quadvec`. The acceptance differs:
`quad_vec` stops when the *sum* of the errors of all subintervals is below
one eighth of that tolerance, while padaquad accepts or splits each subinterval
by its own error ratio. SciPy has no `"rms"` or `"failure_fraction"` option.

(ug-error-reference)=
## Which integral the relative tolerance uses

The relative tolerance multiplies a reference value of the integral:

- **Absolute mode** (`use_absolute_error_ratio=True`): the whole integral.
  During the integration only part of it is known, so the solver uses the
  running total of the subintervals evaluated so far, and early batches get a
  different tolerance than they would with the final integral. When the
  running total is smaller than the final integral, early subintervals are
  refined more than needed. When it is larger, for example when positive and
  negative parts of the integrand cancel, early subintervals are accepted
  against a tolerance that is too loose. With `take_gradient=False`, accepted
  subintervals are checked again as the total changes and split if they fail;
  with `take_gradient=True` they are not, and `converged` reports the failure
  ({ref}`ug-convergence`). Passing `error_integral_reference`, a value close to
  the true integral, fixes the reference from the first batch.
- **Cumulative mode** (`use_absolute_error_ratio=False`): the running integral
  from `mesh_init` through each subinterval, as in ODE solvers. Tolerances then
  change along the domain; where the running integral passes through zero, the
  relative tolerance vanishes and only `atol` remains.

Whatever the mode, `converged` judges every accepted subinterval against the
final integral ({ref}`ug-convergence`).
