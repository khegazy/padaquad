# Quick start

```{include} ../../README.md
:start-after: "<!-- include-start: quickstart -->"
:end-before: "<!-- include-end: quickstart -->"
```

The integral above uses the default method and tolerances. Without
`max_batch`, the first call of `f` evaluates the whole first pass, and later
calls grow by factors of 10 until `f` runs out of memory ({ref}`ug-memory`).
See
{doc}`../user_guide/index` for what each option does, and
{func}`padaquad.integrate` for the full list.

Next: when you integrate a similar integrand many times, for example in a
training loop, reuse the mesh and batch size of the last call:
{ref}`ug-reuse`.
