# Quick start

```{include} ../../README.md
:start-after: "<!-- include-start: quickstart -->"
:end-before: "<!-- include-end: quickstart -->"
```

The integral above uses the default method and tolerances, and a batch size
chosen from the free memory: the first call measures how much memory one
evaluation of `f` uses, which takes a few seconds; pass `max_batch` to skip
it. See
{doc}`../user_guide/index` for what each option does, and
{func}`padaquad.integrate` for the full list.

Next: when you integrate a similar integrand many times, for example in a
training loop, reuse the mesh and batch size of the last call:
{ref}`ug-reuse`.
