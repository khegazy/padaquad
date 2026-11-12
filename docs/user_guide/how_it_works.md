# How it works

```{include} ../../README.md
:start-after: "<!-- include-start: how-it-works -->"
:end-before: "<!-- include-end: how-it-works -->"
```

(ug-mesh)=
## The starting mesh

Without a `mesh` argument, and without a stored mesh from `same_f=True`, the
starting mesh is built from `N_init_steps` (default 13). With
$s = \lfloor\sqrt{N_\text{init}}\rfloor$, the domain is cut into $s$ equal
segments, and each segment receives $s + 1$ mesh points: at random positions
(sorted) by default, or evenly spaced with `random_initial_mesh=False`. The
mesh has $s(s+1)$ points and $s(s+1) - 1$ subintervals: 11 for the default
(`result.n_start_steps` reports it).

The random positions are the default because evenly spaced subintervals can
line up with evenly spaced features of the integrand, such as the zeros of a
sinusoid. When every subinterval sees the same part of a period, every error
estimate can be small while the integral is wrong. Call `torch.manual_seed`
before integrating to get the same mesh, and so the same result to the last
digit, on every run.

A given `mesh` must be a tensor of shape `[M, T]` with increasing values from
`mesh_init` to `mesh_final`.

(ug-optimal-mesh)=
## The optimal mesh and the merge rule

After the integration converges, padaquad builds `mesh_optimal`, the mesh a
later call with `same_f=True` starts from. Pairs of neighbouring subintervals
whose combined error is small are merged, so that a mesh finer than needed
shrinks, and subintervals that failed are split.

The rule that decides whether a pair is merged is padaquad's own derivation.
Let two neighbouring subintervals of width $h$ have error estimates $e_1$ and
$e_2$ and tolerance $\tau$, and let

$$
S = \frac{|e_1| + |e_2|}{\tau}
$$ (merge-pair-ratio)

be the sum of their error ratios. If the error estimate of a subinterval of
width $h$ scales as $K h^k$, where $k$ is the method's `error_exponent` and $K$
is about the same for both halves, then $S \approx 2 K h^k / \tau$, and the
merged subinterval of width $2h$ has the error estimate

$$
e_\text{merged} \approx K (2h)^k = 2^{k-1}\, S\, \tau .
$$ (merge-prediction)

Its error ratio is therefore about $2^{k-1} S$, and the pair is merged only
when that prediction is below `remove_cut` (default 0.1):

$$
S < \frac{\texttt{remove\_cut}}{2^{k-1}} .
$$ (merge-rule)

The merged subinterval is not evaluated again; its contribution and error
estimate are the sums of the pair's. In cumulative mode
({ref}`ug-error-reference`) the tolerance $\tau$ is that of the right
subinterval, where the merged one ends. Each call merges a subinterval at most
once, so a mesh much finer than needed shrinks over several calls. The
prediction assumes the error estimate shrinks like $h^k$; when the estimates
are at the level of rounding error (high-degree methods at tight tolerances, or
float32), they do not shrink with $h$ and few pairs are merged.
