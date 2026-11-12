Verification
------------

These scripts let you check padaquad's accuracy on your own machine. The
first measures the polynomial degree of every quadrature method from its
stored nodes and weights. The second integrates functions with known
integrals using every method and compares each result with the exact answer.
Both exit with an error status if a check fails, and the test suite runs them.
