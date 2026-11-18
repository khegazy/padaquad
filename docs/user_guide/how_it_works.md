# How it works

```{include} ../../README.md
:start-after: "<!-- include-start: how-it-works -->"
:end-before: "<!-- include-end: how-it-works -->"
```

(ug-mesh)=
## The starting mesh

Without a `mesh` argument, the starting mesh is built from `N_init_steps` on
the bounds `mesh_init` and `mesh_final`, which are then both required. With
$s = \lfloor\sqrt{N_\text{init}}\rfloor$, the domain is cut into $s$ equal
segments, and each segment receives $s + 1$ mesh points: at random positions
(sorted) by default, or evenly spaced with `random_initial_mesh=False`. The
mesh has $s(s+1)$ points and $s(s+1) - 1$ subintervals (`result.n_start_steps`
reports the number).

The random positions are the default because evenly spaced subintervals can
line up with evenly spaced features of the integrand, such as the zeros of a
sinusoid. When every subinterval sees the same part of a period, every error
estimate can be small while the integral is wrong. Call `torch.manual_seed`
before integrating to get the same mesh, and so the same result to the last
digit, on every run.

A given `mesh` must be a tensor of shape `[M, T]` with at least two increasing
points. Its ends are the bounds, unless bounds inside it are given, which cut
it ({ref}`ug-reuse`).

(ug-optimal-mesh)=
## The optimal mesh and the merge rule

After the integration converges, padaquad builds the optimal mesh,
`state.mesh`, the mesh a later call starts from when the state is passed as
`state=` ({ref}`ug-reuse`). Pairs of neighbouring subintervals
whose merged subinterval is predicted to pass with a margin are merged, so that a
mesh finer than needed shrinks, and subintervals that failed are split. The
merged subinterval is not evaluated during this step; the next call evaluates it.

Let two neighbouring subintervals $[a, m]$ and $[m, b]$ have contributions $Q_1$
and $Q_2$, error estimates $e_1$ and $e_2$, and let $\tau$ be the tolerance the
merged subinterval $[a, b]$ would get (in cumulative mode,
{ref}`ug-error-reference`, that of the right subinterval, where the merged one
ends). The pair is merged when its predicted error $P$ satisfies

$$
\frac{P}{\tau} < \texttt{remove\_cut}' ,
$$ (merge-condition)

where `merge_rule` sets $P$ and the cutoff $\texttt{remove\_cut}'$.

**`merge_rule="predicted"` (the default).** Every method places its nodes at
fixed fractions of a subinterval, including both ends, and the integrand values
at those nodes are already stored. On each neighbour, padaquad replaces the
integrand by the polynomial through that neighbour's values (barycentric
interpolation, {cite:t}`berrut2004`, Eqs. (3.2) and (4.2)), evaluates it at the
nodes the merged subinterval would have, and applies the method's weights. This
gives the merged subinterval's integral $\hat Q$ and error estimate $\hat E$
without evaluating the integrand. Then

$$
P = \max\!\left(|\hat E|,\; \left|\hat Q - (Q_1 + Q_2)\right|\right) + |e_1| + |e_2|
$$ (merge-prediction)

and the cutoff is `remove_cut` itself.

- $|\hat E|$ predicts whether the merged subinterval will pass its own error
  estimate on the next call.
- $|\hat Q - (Q_1 + Q_2)|$ compares the merged subinterval with the sum of its
  two halves, the test that adaptive Simpson quadrature uses to decide
  refinement ({cite:t}`gander2000`, Sec. 3). When the stored values resolve the
  integrand, it approximates the merged subinterval's true error. It stops a
  merge whose own error estimate would be blind: for a kink $|t - 1/3|$ next to
  much wider neighbours, merging can leave every node of the merged subinterval
  on the straight part, so that its error estimate is exactly 0 while its true
  error is many times the tolerance.
- $|e_1| + |e_2|$ keeps a margin for the neighbours' own errors.

**`merge_rule="power_law"`** (padaquad 1.3.0 and 1.4.0). $P = |e_1| + |e_2|$ and the cutoff
is $\texttt{remove\_cut} / 2^{k-1}$, with $k$ the method's `error_exponent`. If
the error estimate of a subinterval of width $h$ scaled as $K h^k$ with the same
$K$ for both halves, the merged subinterval of width $2h$ would have the error
$K (2h)^k = 2^{k-1}(|e_1| + |e_2|)$ (padaquad's own derivation,
{ref}`ug-error-control`). That holds only when the subintervals are small enough
for the leading Taylor term to dominate. On wider subintervals the error grows by
far less than $2^{k-1}$ (measured: about $2^8$ to $2^9$ where $2^{20}$ is assumed
for `gk21`), so this merge rule refuses merges that would pass, and warm-started
calls keep more subintervals than they need. When the estimates are at the
level of rounding error (high-degree methods at tight tolerances, or float32),
they do not shrink with $h$ and few pairs are merged.

**`merge_rule="sum"`** (padaquad 1.2.2). $P = |e_1| + |e_2|$ and the cutoff is
`remove_cut`. It ignores how the error grows when two subintervals merge, so
merged subintervals often fail when the next call evaluates them and are split
again.

The two older merge rules are kept to reproduce meshes of earlier versions.

**What no merge rule can see.** Every merge rule decides from the values already
stored. A feature of the integrand that lies between the stored nodes, such as a
narrow bump, is invisible to all of them. The next call evaluates the merged
subinterval at new nodes: if they hit the feature, its error estimate usually
fails and the subinterval is split again, but a method whose error estimate is
weak at that node can accept it with a large error (measured for `dopri5`, under
every merge rule). A merged mesh is only as reliable as the stored values'
coverage of the integrand.

Each call merges a subinterval at most once, so a mesh much finer than needed
shrinks over several calls. A merged subinterval keeps the sum of the pair's
contributions and error estimates until it is evaluated again.

The verification script
{doc}`examples/verification/merge_prediction.py <../auto_examples/verification/merge_prediction>`
runs warm-start chains with `"predicted"` and `"power_law"` for every method and
checks that no warm call ends less accurate than the call it started from, at
ordinary and at rounding-level tolerances and in float32.
