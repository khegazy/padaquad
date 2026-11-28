Harder integrals
----------------

Integrands that make an adaptive method work: many oscillations (of one
frequency, or of a growing one), a narrow peak, a spectrum of 200 narrow
peaks, an infinite derivative, and a thousand outputs at once. Each script
integrates one of them (from ``padaquad.hard_integrand_dict``, the integrands
the Performance page also measures), compares the result with the exact
integral, and plots where PadaQuad made its subintervals small.
