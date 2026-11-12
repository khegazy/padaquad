# Concepts

## Quadrature, not ODE solving

padaquad computes

$$
I = \int_a^b f(t)\,dt
$$ (concepts-integral)

for an integrand $f$ that you can evaluate at any point of the integration
variable $t$. The integrand may depend on extra arguments (`f_args`) and on
learnable parameters, but never on the running value of the integral. That is
the difference from an ODE solver such as torchdiffeq, which solves
$\dot y = f(t, y)$: there each step needs the state $y$ that the previous step
produced, so the steps run one after another. Because padaquad's integrand
does not depend on $y$, the integrals over different subintervals are
independent, and padaquad evaluates the nodes of many subintervals in one call
of $f$.

(ug-shapes)=
## The integrand contract

The integrand receives a batch of points and returns the values there:

- **input** `t`: a tensor of shape `[N, T]`, one row per point;
- **output**: a tensor of shape `[N, D]`, one row per point and one column per
  output.

`N` changes from call to call, so `f` must work for any `N`. `D` is the number
of integrals computed at once: an integrand with `D = 3` outputs gives three
integrals, and `result.integral` has shape `[D]`. `T` is the number of
coordinates of a point. Only `T = 1` is supported: the integration domain is an
interval. The integrand may also return a 2-tuple `(values, tracked_variables)`
to record extra quantities at the nodes without integrating them
({ref}`ug-tracked-variables`).

The integrand is called with the points in the dtype and on the device of the
solver, and its output must match them. `f(t, *f_args)` is called when
`f_args` is given.

## Subintervals, nodes and methods

padaquad splits $[a, b]$ into {term}`subintervals <subinterval>`, given by the
{term}`mesh`. On each subinterval $[l, r]$ of width $h = r - l$ a
{term}`method` evaluates the integrand at $C$ {term}`nodes <node>`
$t_i = l + c_i h$ and forms

$$
I_{[l, r]} \approx h \sum_{i=1}^{C} b_i\, f(t_i),
$$ (concepts-method)

where the $c_i$ are the method's node fractions and the $b_i$ its weights. The
sum of these over all subintervals is the integral. How accurate a method is on
one subinterval is described by its {term}`degree`; see
{doc}`methods`. How padaquad decides where to split is described in
{doc}`error_control`.
