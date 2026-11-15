# Methods and options

The quadrature methods are objects held in two dictionaries. Their degrees,
orders and sources are in the {ref}`table on the methods page <methods-table>`,
which is generated from these dictionaries.

```{eval-rst}
.. currentmodule:: padaquad

.. py:data:: UNIFORM_METHODS
   :type: dict[str, MethodClass]

   The methods of the uniform-sampling solver, by name: ``adaptive_heun``,
   ``fehlberg2``, ``bosh3``, ``dopri5`` (Runge-Kutta); ``gk7``, ``gk15``,
   ``gk21``, ``gk31`` (Gauss-Kronrod); ``cc5``, ``cc9``, ``cc17``, ``cc33``,
   ``cc65`` (Clenshaw-Curtis). Each solver uses its own copy, so changing a
   solver's precision never changes these objects.

.. py:data:: VARIABLE_METHODS
   :type: dict[str, MethodClass]

   The methods of the variable-sampling solver, by name: ``adaptive_heun``
   (the same object as in :data:`UNIFORM_METHODS`) and
   ``interpolatory3_variable``.

.. autoclass:: steps
   :members:
   :undoc-members:

.. currentmodule:: padaquad.methods

.. autoclass:: MethodClass
   :members: measure_degree, node_fractions, weights, n_nodes, order, clone

.. autoclass:: UniformMethodClass

.. autoclass:: VariableMethodClass

.. autoclass:: DegreeMeasurement
```
