# Solvers

`adaptive_quadrature` returns one of these two classes; both take the same
keyword-only options, listed under the uniform solver. Build a solver once and
call its `integrate` method repeatedly, for example in a training loop, to warm
start each call from the last one's mesh ({ref}`ug-same-f`).

```{eval-rst}
.. currentmodule:: padaquad

.. autoclass:: UniformAdaptiveQuadrature

   .. automethod:: integrate
   .. automethod:: get_max_batch
   .. automethod:: set_dtype_by_input

.. autoclass:: VariableAdaptiveQuadrature

   Takes the same options and has the same methods as
   :class:`UniformAdaptiveQuadrature`.
```
