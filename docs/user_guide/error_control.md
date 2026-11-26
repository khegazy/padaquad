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

is below 1, and split at its midpoint otherwise (or into four pieces,
{ref}`ug-four-pieces`). Here `atol` and `rtol` are
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
PadaQuad does not.

The error estimate measures the embedded method's error, so it is usually much
larger than the error of the integral PadaQuad returns, which uses the
higher-degree method {cite:p}`gonnet2012`, footnote 20. The script
{doc}`examples/verification/integral_accuracy.py <../auto_examples/verification/integral_accuracy>`
compares actual errors with the tolerances; for the Gauss-Kronrod and
Clenshaw-Curtis methods the actual errors are many orders of magnitude below
them. The exception is `fehlberg2`, whose error estimate ignores curvature
({doc}`methods`).

### Why the error estimate shrinks like $h^{d+2}$

The merge rule `merge_rule="power_law"` ({ref}`ug-optimal-mesh`) assumes that the error
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

PadaQuad runs in float64 (the default) or float32, following the dtype of the
bounds or mesh given to `integrate`. float16 is refused with a `ValueError`:
its rounding error, about $10^{-3}$, is larger than typical tolerances, so the
solver could not tell whether a subinterval meets its tolerance.

### Non-finite values

If the integrand returns NaN or Inf, `integrate` raises a `ValueError` naming
the offending points (`runtime_checks=True`, the default). With
`runtime_checks=False`, subintervals with a non-finite error ratio are
accepted instead of split, the non-finite value reaches the returned integral,
and `converged` is `False`. Either way the run ends. `runtime_checks=False`
also skips the check, made once at the end of the call, that the accepted
subintervals cover the final mesh once; the finite check costs time and, for
integrands with many outputs, memory as large as a call's output. The mesh
itself stays in order by construction, so it is not checked.

(ug-four-pieces)=
### Splitting into four pieces

A failing subinterval is split at its midpoint (`split_rule="halve"`, the
default). Where the integrand is far from resolved, both halves usually fail
again, and the values computed on the subinterval are discarded at every
level: with `gk21` at `atol=1e-8`, about half of the points at which
$\sin(2 \times 10^5 t)$ or $\sin(1/(t + 10^{-3}))$ on $[0, 1]$ is evaluated
belong to subintervals that are then split, and each level costs one more pass
of the loop.

With `split_rule="accelerate"` (a constructor or `integrate` option), a failing
subinterval is split into four equal pieces when

- it is itself a piece of a split subinterval (its split count is at least 1),
  and
- its error ratio $r$ is at least $2^k$, where $k$ is the method's
  `error_exponent` (21 for `gk21`, 15 for `gk15`, 7 for `gk7`).

The four pieces are those that two successive halvings make, and they count as
two halvings in `max_split_depth` and against `max_adaptive_splits` (one
halving below the cap, the subinterval is halved). Subintervals judged again by
the re-check of accepted subintervals are halved. With
`error_norm="failure_fraction"`, whose accept quantity is a fraction rather
than a ratio, `split_rule="accelerate"` raises `ValueError` (pass
`split_rule="halve"` or a norm-family `error_norm`).

**Why $2^k$.** The error estimate of a subinterval of width $h$ is dominated by
the error of the method's embedded lower-degree method, of degree $k - 2$,
which is
$C h^k$ with $C$ proportional to the $(k-1)$-th derivative of $f$ there. Where
that derivative changes little across the subinterval, each half has

$$
e_{1/2} \approx \frac{e}{2^k}, \qquad r_{1/2} \approx \frac{r}{2^k}
$$ (half-ratio)

when the tolerance does not depend on $h$ (`atol`, `rtol` times the integral),
and $r_{1/2} \approx r / 2^{k-1}$ when it is the rounding floor, which shrinks
with $h$. Both halves would pass, and a four-way split would waste the
evaluation of two pieces, only if $r_{1/2} < 1$ for both, which
{eq}`half-ratio` excludes for $r \geq 2^k$: such a subinterval's halves would
fail and be split anyway, and the four-way split saves their evaluation and a
pass. This derivation is PadaQuad's own reasoning, not taken from a paper;
measured, a subinterval's estimate is exactly $2^k$ times each half's for a
polynomial of degree $k - 1$, for every method.

Before the estimate follows $h^k$ (where the integrand is far from resolved),
it can fall faster than $2^k$ per halving, and a four-way split can then cost
points. Measured over 5,024 runs (the 9 integrands below, 15 others, and 3,944
runs on sines, Gaussians, Runge functions and exponentials; `gk7` to `gk31`,
`cc17`, `cc33`; `atol` $10^{-4}$ to $10^{-12}$; random starting meshes), 11
needed more points than halving: 6 because halving had accepted an error above
10 times the tolerance by accident (5 of them a jump with `gk7`), and 5 by 1.3
to 6.1 % (`gk7`, `gk15`, `cc17`). None was `gk21`, `gk31` or `cc33`, and no
error was worse than 10 times halving's or 10 times the tolerance. The saving
depends on the integrand and the method:

| Runs | `gk21` points / passes | `gk15` | `gk7` | `cc33` |
|---|---|---|---|---|
| 9 integrands of the plan (405) | 0.985 / 0.956 | 0.885 / 0.877 | 0.798 / 0.788 | 0.932 / 0.938 |
| 15 other integrands (675) | 0.982 / 0.962 | 0.963 / 0.882 | 0.828 / 0.733 | 0.984 / 0.916 |
| smooth families (3,944) | 0.877 / 0.861 | 0.825 / 0.838 | 0.781 / 0.724 | 0.882 / 0.870 |

(totals relative to halving). For `gk21` the saving is in oscillatory
integrands refined to tight tolerances ($\sin(2 \times 10^4 t)$ at
`atol=1e-11`: 26 % fewer points, 7 passes instead of 10); integrands with an
isolated feature (a kink, a peak, a jump), whose failing pieces leave one
failing piece each, are mostly unchanged.

It does not save time when `f` is cheap: on an A100, a mesh update takes 53
GPU launches when halving; with `"accelerate"`, every update after the call's
first split takes 31 more for its test, and 35 more when it splits in four.
Over 270 runs (the 9 integrands of the plan, `gk7`, `gk15`, `gk21`, `cc17`,
`cc33`, `atol` $10^{-6}$ to $10^{-10}$, two random starting meshes), the total
time was 1 % less than halving's: 9 % less where it split in four, 7 % more
where it did not; `gk21`, which seldom splits in four there, took 5 % more
(geometric mean) and `gk7` 10 % less. Use it when `f` is expensive and the
integrand needs many levels of refinement.

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
  an aggregate. This scheme is PadaQuad's own.

For $D = 1$ the norm options all reduce to Eq. {eq}`error-ratio`.

**Relation to SciPy.** The options `"2"` and `"max"` and the tolerance
$\max(\text{epsabs}, \text{epsrel}\cdot\text{norm}(I))$ are those of
`scipy.integrate.quad_vec` {cite:p}`scipy_quadvec`. The acceptance differs:
`quad_vec` stops when the *sum* of the errors of all subintervals is below
one eighth of that tolerance, while PadaQuad accepts or splits each subinterval
by its own error ratio. SciPy has no `"rms"` or `"failure_fraction"` option.

(ug-error-reference)=
## Which integral the relative tolerance uses

The relative tolerance multiplies a reference value of the integral:

- **Absolute mode** (`use_absolute_error_ratio=True`): the whole integral.
  During the integration only an estimate of it is known, so the solver uses
  the running total of the subintervals evaluated so far: with
  `take_gradient=False` the whole interval, evaluated on the current mesh,
  which is coarse in the first passes; with `take_gradient=True` also only part
  of the interval, since each batch is judged as soon as it is evaluated. Early
  subintervals therefore get a different tolerance than they would with the
  final integral. When the
  running total is smaller than the final integral, early subintervals are
  refined more than needed. When it is larger, for example when positive and
  negative parts of the integrand cancel, early subintervals are accepted
  against a tolerance that is too loose. With `enforce_convergence=True` (the
  default when `take_gradient=False`), accepted subintervals are checked again
  in every batch, against the same total as the subintervals just evaluated,
  and split if they fail; the check uses their stored error estimates, so it
  costs no evaluation of the integrand. With `take_gradient=True` they are not
  checked again (a batch's gradient is already backpropagated when it is
  accepted), and `converged` reports the failure ({ref}`ug-convergence`).
  Passing `error_integral_reference`, a value close to the true integral, fixes
  the reference from the first batch. With `enforce_convergence=True`, that
  reference is dropped once every subinterval is accepted: the subintervals
  that fail against the integral are split, and the integration continues
  against the running total until it converges, or until
  `max_adaptive_splits` or a non-finite value stops the refinement.
- **Cumulative mode** (`use_absolute_error_ratio=False`): the running integral
  from `mesh_init` through each subinterval, as in ODE solvers. Tolerances then
  change along the domain; where the running integral passes through zero, the
  relative tolerance vanishes and only `atol` remains.

Whatever the mode, `converged` judges every accepted subinterval against the
final integral ({ref}`ug-convergence`).
