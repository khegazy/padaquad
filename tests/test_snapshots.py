"""Snapshot/golden-value tests — the safety net for internal refactors.

For each ``(method, integrand, tolerance)`` triple, run the solver
with a fixed seed, no warm-start, and CPU + float64. Record the
returned ``integral``, ``integral_error``, optimal-mesh length, and
total evaluation count, then assert subsequent runs reproduce
those values to ``1e-12``.

Snapshots live in ``tests/test_snapshots_data.json`` (committed to
the repo). Regenerate with::

    TPD_REGENERATE_SNAPSHOTS=1 pytest tests/test_snapshots.py

Regenerate is required:
  * before Phase 1 lands (the bug fixes are in code paths that this
    test exercises only if reuse_mesh is on, which it isn't here);
  * at every Phase 5 file-split commit, to verify bit-equivalence;
  * any time a method's adaptive controller is intentionally changed.

Variable-sampling cases are excluded because the variable solver is
currently non-functional (see ``test_variable_integration.py``).

Phase 0 of the quadrature alignment plan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from _helpers import (
    INTEGRAND_NAMES,
    SEED,
    T_FINAL,
    T_INIT,
    TAKE_GRADIENT_IDS,
    TAKE_GRADIENT_VALUES,
    UNIFORM_METHOD_NAMES,
    make_uniform_solver,
)

from padaquad import integrand_dict

DATA_FILE = Path(__file__).parent / "test_snapshots_data.json"
SNAPSHOT_TOL = 1e-12
REGENERATE = os.environ.get("TPD_REGENERATE_SNAPSHOTS") == "1"

# Three tolerance settings — loose, medium, tight — that exercise the
# adaptive controller across regimes.
TOLERANCES = {
    "loose": (1e-5, 1e-5),
    "medium": (1e-8, 1e-8),
    "tight": (1e-12, 1e-10),
}


def _case_key(method: str, integrand: str, tol_label: str) -> str:
    """Stable key for a snapshot case in the JSON data file."""
    return f"{method}|{integrand}|{tol_label}"


def _run_case(
    method: str,
    integrand: str,
    atol: float,
    rtol: float,
    take_gradient: bool = True,
) -> dict:
    """Run one solver call and record the values we want to pin.

    Sets the manual seed before each call so the random initial mesh
    is reproducible. Also temporarily sets the global default dtype to
    float64 (snapshots run CPU + float64 for cross-machine determinism)
    and restores it on exit. The restore matters: the solver's
    random-mesh path goes through ``torch.rand`` which inherits the
    default dtype, so leaving the default at float64 contaminates any
    float32 solver constructed in a *later* test (their initial mesh
    silently becomes float64 and trips dtype-mismatch assertions deep
    in the adaptive loop).

    ``take_gradient`` controls the integration code path but must not
    perturb the numerical result — the snapshot file stores a single
    set of expected values, and both modes must hit it.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(SEED)
        f, _solution_fxn, _cutoff = integrand_dict[integrand]
        solver = make_uniform_solver(method, atol=atol, rtol=rtol)
        output = solver.integrate(
            f,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            take_gradient=take_gradient,
        )
        return {
            "integral": output.integral.tolist(),
            "integral_error": output.integral_error.tolist(),
            "n_optimal_mesh": int(output.mesh_optimal.shape[0]),
        }
    finally:
        torch.set_default_dtype(previous_default_dtype)


def _generate_all() -> dict[str, dict]:
    """Compute snapshots for every (method, integrand, tolerance) case."""
    snaps: dict[str, dict] = {}
    for method in UNIFORM_METHOD_NAMES:
        for integrand in INTEGRAND_NAMES:
            for tol_label, (atol, rtol) in TOLERANCES.items():
                key = _case_key(method, integrand, tol_label)
                snaps[key] = _run_case(method, integrand, atol, rtol)
    return snaps


def _maybe_regenerate_data_file() -> None:
    """If TPD_REGENERATE_SNAPSHOTS=1, regenerate the data file in place."""
    if not REGENERATE:
        return
    snaps = _generate_all()
    DATA_FILE.write_text(json.dumps(snaps, indent=2, sort_keys=True) + "\n")


_maybe_regenerate_data_file()


def _load_snapshots() -> dict[str, dict]:
    if not DATA_FILE.exists():
        pytest.fail(
            f"Snapshot data file {DATA_FILE} not found. "
            f"Generate with: TPD_REGENERATE_SNAPSHOTS=1 pytest "
            f"tests/test_snapshots.py"
        )
    return json.loads(DATA_FILE.read_text())


@pytest.fixture(scope="module")
def snapshots() -> dict[str, dict]:
    """Load the committed snapshot data once per pytest module."""
    return _load_snapshots()


def _ids() -> list[tuple[str, str, str]]:
    return [
        (m, i, t)
        for m in UNIFORM_METHOD_NAMES
        for i in INTEGRAND_NAMES
        for t in TOLERANCES
    ]


@pytest.mark.parametrize("take_gradient", TAKE_GRADIENT_VALUES, ids=TAKE_GRADIENT_IDS)
@pytest.mark.parametrize(
    ("method", "integrand", "tol_label"),
    _ids(),
    ids=str,
)
def test_snapshot(snapshots, method, integrand, tol_label, take_gradient):
    """Current solver output matches the committed snapshot to 1e-12.

    Both ``take_gradient=True`` and ``take_gradient=False`` must hit
    the same stored value — the snapshot key intentionally omits
    ``take_gradient`` because the two modes are expected to produce
    identical numerics. After the upcoming code-path split, any
    divergence between modes will fail this test against the single
    stored value, surfacing the bug.
    """
    key = _case_key(method, integrand, tol_label)
    expected = snapshots.get(key)
    if expected is None:
        pytest.fail(
            f"No snapshot for {key}; regenerate with TPD_REGENERATE_SNAPSHOTS=1."
        )
    atol, rtol = TOLERANCES[tol_label]
    got = _run_case(method, integrand, atol, rtol, take_gradient=take_gradient)

    # Compare element-wise, allowing 1e-12 absolute drift.
    for field, expected_val in expected.items():
        got_val = got[field]
        if isinstance(expected_val, list):
            for e, g in zip(expected_val, got_val, strict=True):
                assert abs(g - e) <= SNAPSHOT_TOL, (
                    f"{key} field={field}: got {got_val}, "
                    f"expected {expected_val}, drift {abs(g - e)} > {SNAPSHOT_TOL}"
                )
        else:
            assert got_val == expected_val, (
                f"{key} field={field}: got {got_val}, expected {expected_val}"
            )
