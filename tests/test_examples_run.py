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
  - ``same_f=True`` warm-start across iterations
  - the analytic-vs-autodiff consistency check (path A vs path B)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

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


VERIFICATION_DIR = EXAMPLES_DIR / "verification"


def test_verification_scripts_pass():
    """Every script in ``examples/verification/`` exits 0 and prints no
    ``[FAIL]``. Each script exits 1 itself when one of its checks fails."""
    scripts = sorted(VERIFICATION_DIR.glob("*.py"))
    assert {s.name for s in scripts} >= {
        "polynomial_degree.py",
        "integral_accuracy.py",
    }, f"missing verification scripts in {VERIFICATION_DIR}"
    for script in scripts:
        out = _run(script)
        assert out.returncode == 0, (
            f"{script.name} exited {out.returncode}; stdout=\n{out.stdout}\n"
            f"stderr=\n{out.stderr}"
        )
        assert "[FAIL]" not in out.stdout, f"{script.name}: stdout=\n{out.stdout}"


# Documents whose Python blocks are complete programs that must run.
DOCUMENTS = ("README.md", "docs/development/adding_a_method.md")
NOT_TESTED = "<!-- not-tested -->"


def _python_blocks() -> list[tuple[str, str]]:
    """("file:line", code) of every ```python block in ``DOCUMENTS`` that is not
    preceded by the line ``<!-- not-tested -->`` (used for sketches that need
    the reader's own model or data)."""
    blocks = []
    for document in DOCUMENTS:
        lines = (REPO_ROOT / document).read_text().splitlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "```python":
                start = i + 1
                end = lines.index("```", start)
                previous = next((x for x in reversed(lines[:i]) if x.strip()), "")
                if previous.strip() != NOT_TESTED:
                    blocks.append(
                        (f"{document}:{start + 1}", "\n".join(lines[start:end]))
                    )
                i = end
            i += 1
    return blocks


@pytest.mark.parametrize("where_and_code", _python_blocks(), ids=lambda b: b[0])
def test_document_python_block_runs(where_and_code: tuple[str, str]) -> None:
    """Each tested Python block of ``DOCUMENTS`` runs without error on the CPU."""
    where, code = where_and_code
    out = subprocess.run(
        [PY, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert out.returncode == 0, f"{where} failed:\n{out.stderr[-2000:]}"
