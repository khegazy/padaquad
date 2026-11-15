# Test integrands

```{eval-rst}
.. currentmodule:: padaquad

.. py:data:: integrand_dict
   :type: dict[str, tuple]

   Test integrands with known integrals, by name: ``t``, ``t_squared``,
   ``sine_squared``, ``exp`` and ``damped_sine``. Each value is a tuple
   ``(f, solution, error_cutoff)``: the integrand, a function
   ``solution(mesh_init, mesh_final)`` returning its exact integral, and the
   relative error the tests accept.

.. autofunction:: wolf_schlegel
```
