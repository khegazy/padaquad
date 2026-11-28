# PadaQuad

PadaQuad computes definite integrals $\int_a^b f(t)\,dt$ of a function you can
evaluate, in PyTorch. It splits $[a, b]$ into subintervals, refines the ones
whose error is too large, and evaluates the integrand at many points of many
subintervals in one call, on a CPU or a GPU. Gradients flow through the result,
so the integral can be part of a training loss.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started/index
:link-type: doc

Install PadaQuad and compute a first integral.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

How the integration works, how to choose a method, how the error is
controlled, gradients, reusing the mesh and batch size across calls, and
memory.
:::

:::{grid-item-card} API reference
:link: reference/index
:link-type: doc

Every public function, class and result field, generated from the
docstrings.
:::

:::{grid-item-card} Examples
:link: auto_examples/index
:link-type: doc

Runnable scripts with their output, including the checks of each method's
accuracy that you can run yourself.
:::
::::

```{include} ../README.md
:start-after: "<!-- include-start: what-for -->"
:end-before: "<!-- include-end: what-for -->"
```

## Comparisons with other libraries

```{include} ../README.md
:start-after: "<!-- include-start: comparisons -->"
:end-before: "<!-- include-end: comparisons -->"
```

```{toctree}
:hidden:
:maxdepth: 2

getting_started/index
user_guide/index
reference/index
auto_examples/index
performance
development/index
release_notes
glossary
references
about
```
