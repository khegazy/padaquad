# Glossary

These are the words the documentation and the docstrings use. Each concept has
one word, and the code's internal names (for example `mesh_trackers`) are not
used in prose.

(glossary-integration-variable)=
integration variable `t`
: The variable of integration. padaquad computes $\int_a^b f(t)\,dt$. The code
  calls it `t`; the documentation calls it "the integration variable `t`". It
  need not be a time.

integrand
: The function `f` being integrated. It takes points of shape `[N, T]` and
  returns values of shape `[N, D]`. It depends only on `t` (and on optional
  extra arguments `f_args`), never on the running value of the integral.

(glossary-subinterval)=
subinterval
: One piece of the integration domain $[a, b]$. padaquad splits the domain into
  subintervals, applies a quadrature method on each, and adds the results.

(glossary-mesh)=
mesh
: The ordered endpoints of the subintervals, from `mesh_init` to `mesh_final`.
  A mesh with M points has M - 1 subintervals.

(glossary-node)=
node
: A point where the integrand is evaluated. A method places C nodes in every
  subinterval.

node fractions
: The node positions of a method inside a subinterval of width 1, from 0 to 1.
  A subinterval $[l, r]$ gets its nodes at $l + c_i (r - l)$.

(glossary-method)=
method
: A quadrature formula: node fractions plus weights. padaquad's methods are
  listed in `UNIFORM_METHODS` and `VARIABLE_METHODS` (for example `gk21`,
  `cc33`, `dopri5`).

weights
: The numbers $b_i$ that multiply the integrand values at the nodes. On a
  subinterval of width $h$ the method's result is $h \sum_i b_i f(t_i)$.

(glossary-degree)=
degree
: The highest polynomial degree a method integrates exactly. `gk21` has degree
  31: it integrates every polynomial of degree at most 31 without error, but not
  every polynomial of degree 32. `MethodClass.measure_degree()` measures it.

order
: The degree plus one. On a smooth integrand, the total error of a method
  applied on subintervals of width $h$ falls like $h^{\text{order}}$.

(glossary-embedded)=
embedded method
: A second, lower-degree method that uses some or all of the same nodes as the
  main method, so it costs no extra integrand evaluations. The two results are
  compared to estimate the error.

(glossary-error-estimate)=
error estimate
: The difference between the main method's and the embedded method's results
  on a subinterval. It estimates the embedded method's error, so it is usually
  larger than the main method's true error.

error exponent
: The power $k$ such that the error estimate on a subinterval of width $h$
  shrinks like $h^k$. It is the embedded method's degree plus 2.

tolerance
: The accuracy asked for. A subinterval is accepted when its error estimate is
  below $\max(\text{atol}, \text{rtol} \cdot |I|)$, where `atol` is the absolute
  tolerance, `rtol` the relative tolerance and $I$ the integral.

(glossary-error-ratio)=
error ratio
: A subinterval's error estimate divided by its tolerance. Below 1, the
  subinterval is accepted; otherwise it is split in half and both halves are
  evaluated again.

rounding floor
: The size of the error that float64 or float32 rounding alone makes on a
  subinterval. padaquad never asks for a tolerance below it, so it stops
  splitting once the error is at rounding level.

(glossary-warm-start)=
warm start
: Starting an integration from the mesh the previous integration ended with,
  instead of a fresh one. Requested with `same_f=True`.

uniform sampling
: The nodes of every subinterval sit at the method's fixed node fractions, so
  the weights are fixed numbers. Splitting a subinterval evaluates the
  integrand at fresh nodes in both halves.

variable sampling
: The weights are computed from where the nodes actually are, so a split can
  reuse the integrand values already computed and evaluate only the new nodes.

shape letters
: `N` is a number of subintervals (or points), `C` the number of nodes per
  subinterval, `T` the number of coordinates of `t` (1 in practice), `D` the
  number of outputs of the integrand, and `M` the number of mesh points.
