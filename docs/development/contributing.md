# Contributing

## Set up

```bash
git clone https://github.com/khegazy/padaquad.git
cd padaquad
python -m venv .venv
.venv/bin/pip install -e ".[dev,docs]"
.venv/bin/pre-commit install
```

## Checks every change must pass

```bash
.venv/bin/pytest                         # tests and docstring examples
.venv/bin/ruff check padaquad/ tests/    # lint
.venv/bin/ruff format --check padaquad/  # formatting
.venv/bin/mypy padaquad/                 # types (run where torch is installed)
numpydoc lint $(git ls-files 'padaquad/*.py')  # docstring format
```

`tests/test_snapshots.py` pins the integral, error estimate and optimal-mesh
size of every method on every test integrand. If a change is meant to alter
those numbers, regenerate them with
`TPD_REGENERATE_SNAPSHOTS=1 .venv/bin/pytest tests/test_snapshots.py` and
explain why in the pull request; never regenerate them to make a failing test
pass.

`tests/test_exactness.py`, `tests/test_tableau.py` and the scripts in
`examples/verification/` check every method's degree and weights against
independent calculations. A new method must pass them.

## Build the documentation

```bash
.venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
```

The build fails on any warning. The API pages come from the docstrings (see
{doc}`docstrings`), the methods table from the method registries, and the
Performance table from `docs/performance/results.json`.
