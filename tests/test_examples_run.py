"""Run example scripts end-to-end to catch documentation rot.

Phase 6 added two example scripts (``examples/quadrature_basics.py``
and ``examples/gradient_of_integral.py``) as user-facing tutorials.
They live outside ``tests/`` and are not exercised by the rest of
the test suite, so a future API rename or signature change could
silently break them. These tests subprocess each example and assert
non-zero output and a 0 exit code, plus a few markers to confirm the
specific scripted assertions inside each example were reached.

The examples are also useful as integration smoke tests that exercise:

  - ``integrate(f, method=..., mesh_init=..., mesh_final=...)`` (one-shot)
  - ``adaptive_quadrature(...).integrate(f, ...)`` (class API)
  - ``take_gradient=True`` autograd flow
  - ``same_integrand=True`` warm-start across iterations
  - the analytic-vs-autodiff consistency check (path A vs path B)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
PY = sys.executable  # the venv-aware interpreter pytest is running under


def _run(script: Path) -> subprocess.CompletedProcess[str]:
    """Run an example script with the test interpreter and capture output."""
    return subprocess.run(
        [PY, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_quadrature_basics_runs_and_prints_known_results():
    """``examples/quadrature_basics.py`` exits 0 and prints expected
    section headers and convergence results.
    """
    script = EXAMPLES_DIR / "quadrature_basics.py"
    assert script.exists(), f"missing example: {script}"
    out = _run(script)
    assert out.returncode == 0, (
        f"quadrature_basics.py exited {out.returncode}; stderr=\n{out.stderr}"
    )

    # Verify the script reached each example section. These are
    # stable headers in the script's print() output.
    for marker in (
        "sin over [0, pi]:",
        "t^4 over [0, 1]:",
        "exp(-t^2) over [-2, 2]:",
        "damped sinusoid over [0, 4]",
        "warm-start loop",
    ):
        assert marker in out.stdout, (
            f"quadrature_basics.py output missing marker {marker!r}; "
            f"stdout=\n{out.stdout[:500]}"
        )

    # Each _check call prints "[OK ]" or "[FAIL]" — ensure no FAIL
    # leaked to stdout.
    assert "[FAIL]" not in out.stdout, (
        f"quadrature_basics.py reported a [FAIL]; stdout=\n{out.stdout}"
    )

    # Method comparison loop prints all five methods.
    for method in ("adaptive_heun", "bosh3", "dopri5", "gk21", "cc33"):
        assert method in out.stdout


def test_gradient_of_integral_runs_and_assertions_pass():
    """``examples/gradient_of_integral.py`` runs path-A and path-B,
    asserts they agree to 1e-6, and exits 0. Failure of any assertion
    inside the script is a non-zero exit; we check both.
    """
    script = EXAMPLES_DIR / "gradient_of_integral.py"
    assert script.exists(), f"missing example: {script}"
    out = _run(script)
    assert out.returncode == 0, (
        f"gradient_of_integral.py exited {out.returncode}; stderr=\n{out.stderr}"
    )

    # Verify the script printed both gradient paths and the agreement check.
    for marker in (
        "path A (autograd through integration)",
        "path B (integrate the derivative)",
        "All three agree to within 1e-6.",
    ):
        assert marker in out.stdout, (
            f"gradient_of_integral.py output missing marker {marker!r}; "
            f"stdout=\n{out.stdout[:500]}"
        )
