# Methods

A {term}`method` is a set of node fractions $c_i$ and weights $b_i$ that turns
integrand values on one subinterval into an integral, Eq. {eq}`concepts-method`.
Each method also carries a second, lower-degree set of weights, the
{term}`embedded method`; the difference between the two results is the
subinterval's error estimate ({doc}`error_control`).

(methods-table)=
## All methods

The table is generated from `UNIFORM_METHODS` and `VARIABLE_METHODS` when this
page is built. "Degree" is the highest polynomial degree the method integrates
exactly, and "order" is the degree plus one: on a smooth integrand, the error of
the method applied on subintervals of width $h$ falls like $h^\text{order}$
{cite:p}`gonnet2012`. The last numeric column is the degree of the lower-degree
method in the error estimate; the solver assumes the error estimate shrinks like
$h$ to the power of that degree plus 2.

```{include} ../reference/generated/methods_table.md
```

The Gauss-Kronrod methods add two nodes with weight zero at the ends of each
subinterval, so `gk21` evaluates the integrand at 23 points per subinterval of
which 21 carry weight.

**Check these numbers yourself.** Every degree in the table is measured from
the stored nodes and weights by `MethodClass.measure_degree()`, both in the test
suite and in the script
{doc}`examples/verification/polynomial_degree.py <../auto_examples/verification/polynomial_degree>`:

```python
from padaquad import UNIFORM_METHODS

measured = UNIFORM_METHODS["gk21"].measure_degree()
print(measured.degree, measured.lower_degree)  # 31 19
```

The measurement integrates Legendre polynomials rather than powers. Write
$x = 2t - 1$ for the subinterval $[0, 1]$ mapped to $[-1, 1]$. A high power
$x^{d+1}$ is almost entirely made of lower-degree polynomials, which the method
integrates exactly: its error on $x^{d+1}$ is its error on the Legendre
polynomial $P_{d+1}(x)$ times $(k!)^2 2^k / (2k)!$ with $k = d + 1$ (the
inverse of the leading coefficient of $P_k$; derivation), which is about
$4 \times 10^{-14}$ for `gk31`. Powers of $t$ itself are worse, with the
factor $(k!)^2 / (2k)!$. A test with powers therefore cannot see
where the higher-degree methods stop being exact in float64 arithmetic; the
script shows this for every method.

## How to choose

- **Smooth integrands, any accuracy: `gk21`** (the default) or `cc33`. A
  high-degree method needs few subintervals; on the benchmark integrands `gk21`
  meets tolerances down to $10^{-9}$ without splitting any subinterval of the
  starting mesh ({doc}`../performance`).
- **Rough or localized features: a low-degree method** (`gk7`, `cc5`, `cc9`)
  wastes fewer evaluations where the integrand is smooth while the mesh is
  refined around the feature.
- **The Runge-Kutta methods** (`adaptive_heun`, `fehlberg2`, `bosh3`,
  `dopri5`) are low-degree methods taken from ODE solvers. They are kept as
  baselines and for comparison with ODE solvers. Avoid `fehlberg2` when the
  error estimate matters (see below).
- **Variable sampling** (`interpolatory3_variable`) computes its weights from
  the positions of the nodes. The integration loop currently places every
  subinterval's nodes at the method's node fractions, $0, 1/2, 1$, where this
  method is Simpson's rule (degree 3).

## Gauss-Kronrod methods

The method `gkN` combines the $n$-point Gauss-Legendre method $G_n$ with its
$(2n+1)$-point Kronrod extension $K_{2n+1}$, which reuses the $n$ Gauss nodes
and adds $n + 1$ more. The integral is the Kronrod result and the error
estimate is the difference

$$
e = K_{2n+1} - G_n ,
$$ (gk-error)

the difference QUADPACK starts from {cite:p}`quadpack_netlib` (QUADPACK then
rescales it; padaquad does not), described by {cite:t}`gonnet2012`, Eq. (24).
The Kronrod method has degree at least $3n + 1$ {cite:p}`gonnet2012`, Sec. 2.4.
For odd $n$ that number is even, and a method with symmetric nodes and weights
integrates every odd power of $x$ exactly (both sides are zero), so the degree
is at least $3n + 2$ (derivation). The measured degrees are 11, 23, 31 and 47
for `gk7` ($n = 3$), `gk15` ($n = 7$), `gk21` ($n = 10$) and `gk31` ($n = 15$),
equal to these bounds; for `gk7`, which is also the first of Patterson's
nested extensions, {cite:t}`gonnet2012`, Sec. 2.4, gives degree 11. The Gauss
method $G_n$ has degree $2n - 1$.

The nodes and weights of `gk15`, `gk21` and `gk31` are the 33-digit values of
the QUADPACK routines `dqk15.f`, `dqk21.f` and `dqk31.f` {cite:p}`quadpack_netlib`,
compared digit by digit with those files. QUADPACK has no 7-point routine; the
values of `gk7` agree with an 80-digit computation of the Kronrod extension of
the 3-point Gauss method.

## Clenshaw-Curtis methods

The method `ccN` uses the $N = n + 1$ Chebyshev points

$$
x_k = \cos\frac{k\pi}{n}, \qquad k = 0, \dots, n,
$$ (cc-nodes)

on $[-1, 1]$ {cite:p}`trefethen2008`, Eq. (2.3), mapped to each subinterval,
and the weights that integrate the polynomial interpolating the integrand at
those points exactly. Such a method is exact for degree $n$
{cite:p}`trefethen2008`, Sec. 2; because $n$ is even for every method here and
the nodes and weights are symmetric, it is also exact for degree $n + 1$
{cite:p}`trefethen2008`, Theorem 5.2, Eq. (5.4). So `ccN` has degree $N$. The
weights are computed in `_clenshaw_curtis_weights_unit` from a cosine sum that
is algebraically the same as the program `clenshaw_curtis.m` in
{cite:t}`trefethen2008`, Sec. 2, and are checked against a direct solve of the
moment equations in the test suite.

Every other node of `ccN` is a node of the Clenshaw-Curtis method with
$(N + 1)/2$ points {cite:p}`trefethen2008`, Sec. 2, so that method costs no
extra evaluations, and the error estimate is the difference between the two.
Choosing this nested pair is padaquad's choice; {cite:t}`gonnet2012`, Sec. 6,
discusses nested Clenshaw-Curtis pairs in general.

## Runge-Kutta methods

The four Runge-Kutta methods use the weights of embedded Runge-Kutta pairs as
quadrature formulas: with an integrand that does not depend on $y$, one step of
a Runge-Kutta method is the quadrature formula {eq}`concepts-method` with the
method's nodes $c_i$ and weights $b_i$. A Runge-Kutta method of order $p$ has
degree at least $p - 1$ as a quadrature formula; the table gives the measured
degrees.

- `fehlberg2` is Fehlberg's RK1(2) pair {cite:p}`fehlberg1969`, Table XIV:
  nodes $0, 1/2, 1$, weights $1/512, 255/256, 1/512$, and error weights that
  subtract the first-order weights $1/256, 255/256, 0$.
- `adaptive_heun`, `bosh3` and `dopri5` were transcribed from torchdiffeq
  {cite:p}`torchdiffeq`, and their nodes and weights agree with SciPy's RK23 and
  RK45 tables {cite:p}`scipy_quadvec`. The original papers (Heun 1900;
  Bogacki and Shampine 1989; Dormand and Prince 1980) could not be opened to
  check them. The embedded weights of `dopri5`'s error estimate,
  $1951/21600, 0, 22642/50085, 451/720, -12231/42400, 649/6300, 1/60$, come from
  torchdiffeq; SciPy uses a different embedded set. Both satisfy the
  fourth-order conditions, and both have degree 3 as quadrature formulas.
- `adaptive_heun`'s error weights subtract the rectangle rule that uses the
  right end of the subinterval; the estimate has the same size as with Euler's
  method, which uses the left end.

**`fehlberg2`'s error estimate ignores curvature.** Its two weight sets make
almost the same error on a curved integrand: on $t^2$ over $[0, 1]$ they give
0.25098 and 0.24902, where the exact value is $1/3$. Their difference, the
error estimate, keeps only $h\,(f(\text{end}) - f(\text{start}))/512$. Where the
integrand bends more than it rises, the estimate is far too small: on
`damped_sine` at tolerance $10^{-6}$, subintervals with an estimated error near
$10^{-7}$ have a true error near $10^{-4}$, and the total error is larger than
the tolerances of the subintervals add up to. The script
{doc}`examples/verification/integral_accuracy.py <../auto_examples/verification/integral_accuracy>`
reports this.

## The variable-node method

`interpolatory3_variable` has three nodes, $0$, $a$ and $1$ on a subinterval of
width 1, and the weights

$$
b_0 = \frac{1}{2} - \frac{1}{6a}, \qquad
b_a = \frac{1}{6a(1-a)}, \qquad
b_1 = \frac{2 - 3a}{6(1-a)},
$$ (interp3-weights)

which integrate $1$, $t$ and $t^2$ exactly for any $a$ in $(0, 1)$; they are
the weights of {cite:t}`sanderse2019`, Eq. (119), with $c_2 = a$. The method has
degree 2 in general and degree 3 at $a = 1/2$, where it is Simpson's rule. Its
error estimate subtracts the trapezoidal rule on the end nodes (padaquad's
choice).
