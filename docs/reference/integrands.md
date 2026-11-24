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

.. py:data:: hard_integrand_dict
   :type: dict[str, tuple]

   Harder test integrands with known integrals, in the same form:
   ``oscillatory`` (``damped_sine`` with ``w=100``, ``a=1``: 100 periods of
   ``exp(-t) * sin(200*pi*t)`` on [0, 1]), ``sharp_peak`` (a peak of width
   1e-3), ``kink`` (``sqrt(|t - 1/3|)``, whose derivative is infinite at 1/3)
   and ``many_outputs`` (``sin(k*t)`` for k = 1 to 1,000 at once). Their
   integrands, like those of ``integrand_dict``, take the array module as the
   keyword argument ``xp`` (``torch``, ``numpy`` or ``jax.numpy``).

.. autofunction:: wolf_schlegel
```
