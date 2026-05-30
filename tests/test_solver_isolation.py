"""Cross-solver isolation tests.

Phase 4 of the quadrature alignment plan eliminated the
``UNIFORM_METHODS`` singleton mutation hazard by having ``_get_method``
``clone()`` the canonical singleton before applying any dtype/device
mutation. These tests directly verify the isolation guarantees that
clone provides — properties the conftest safety-belt fixture
previously had to enforce externally.

Each test exercises a scenario where, prior to the fix, two solvers
would have corrupted each other's tableau values via the shared
singleton. After the fix, each solver owns its own clone and these
scenarios are independent.

Coverage these tests add to the existing suite:

  - **Sequential dtype mixing**: a float64 solver followed by a
    float32 solver of the same method must not affect the float64
    solver's tableau precision.
  - **Concurrent dtype mixing**: two solvers built at different
    dtypes used in interleaved calls.
  - **Singleton untouched**: ``UNIFORM_METHODS["dopri5"].tableau.b``
    stays at its canonical float64 values regardless of how many
    solvers are constructed at any dtype.
"""

from __future__ import annotations

import torch
from tests._helpers import make_uniform_solver

from padaquad import UNIFORM_METHODS
from padaquad.base import steps
from padaquad.methods import _get_method


def _b_values(solver):
    return solver.method.tableau.b.detach().cpu().tolist()


def test_sequential_dtype_mixing_does_not_corrupt_float64_solver():
    """A float32 solver constructed AFTER a float64 solver must not
    degrade the float64 solver's tableau precision (singleton hazard)."""
    solver_a = make_uniform_solver("dopri5", atol=1e-8, rtol=1e-8)
    a_b_before = _b_values(solver_a)
    a_dtype_before = solver_a.method.tableau.b.dtype

    # This used to mutate the singleton via to_dtype(float32) and
    # leave its values at float32 precision even after restoration.
    solver_b = make_uniform_solver("dopri5", atol=1e-5, rtol=1e-5, dtype=torch.float32)
    assert solver_b.method.tableau.b.dtype == torch.float32

    # solver_a unchanged.
    assert solver_a.method.tableau.b.dtype == a_dtype_before == torch.float64
    assert _b_values(solver_a) == a_b_before


def test_concurrent_solvers_at_different_dtypes_remain_independent():
    """Two solvers constructed in any order can be used in any order
    without dtype drift between them."""
    solver_a = make_uniform_solver("bosh3", atol=1e-8, rtol=1e-8)
    solver_b = make_uniform_solver("bosh3", atol=1e-5, rtol=1e-5, dtype=torch.float32)

    # Run a round of integrations on each, alternating.
    a_b_snap = _b_values(solver_a)
    b_b_snap = _b_values(solver_b)

    for _ in range(3):
        solver_a.integrate(
            f=torch.sin,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([1.0], dtype=torch.float64),
        )
        solver_b.integrate(
            f=torch.sin,
            mesh_init=torch.tensor([0.0], dtype=torch.float32),
            mesh_final=torch.tensor([1.0], dtype=torch.float32),
        )

    assert solver_a.method.tableau.b.dtype == torch.float64
    assert solver_b.method.tableau.b.dtype == torch.float32
    assert _b_values(solver_a) == a_b_snap
    assert _b_values(solver_b) == b_b_snap


def test_global_singleton_never_mutated_by_solver_construction():
    """The canonical UNIFORM_METHODS["dopri5"] must remain bit-stable
    in both dtype AND values regardless of how many solvers we
    construct at any dtype.
    """
    canonical = UNIFORM_METHODS["dopri5"]
    snap_b = canonical.tableau.b.clone()
    snap_c = canonical.tableau.c.clone()
    snap_b_error = canonical.tableau.b_error.clone()
    snap_dtype = canonical.tableau.b.dtype

    # Construct several solvers across dtypes; values via _get_method
    # should clone the singleton, not mutate it.
    for dtype in (torch.float64, torch.float32, torch.float64, torch.float32):
        _get_method(steps.ADAPTIVE_UNIFORM, "dopri5", "cpu", dtype)

    assert canonical.tableau.b.dtype == snap_dtype == torch.float64
    assert torch.equal(canonical.tableau.b, snap_b)
    assert torch.equal(canonical.tableau.c, snap_c)
    assert torch.equal(canonical.tableau.b_error, snap_b_error)


def test_method_clone_returns_independent_object():
    """``MethodClass.clone()`` returns a deep copy: mutations on the
    clone don't reach the original, and clones are mutually
    independent.
    """
    canonical = UNIFORM_METHODS["dopri5"]
    clone1 = canonical.clone()
    clone2 = canonical.clone()

    # Identity: distinct objects.
    assert clone1 is not canonical
    assert clone2 is not canonical
    assert clone1 is not clone2
    assert clone1.tableau is not canonical.tableau
    assert clone1.tableau is not clone2.tableau

    # Tensor identity: tensors are distinct (not aliasing).
    assert clone1.tableau.b.data_ptr() != canonical.tableau.b.data_ptr()
    assert clone1.tableau.b.data_ptr() != clone2.tableau.b.data_ptr()

    # Mutation isolation: change clone1's dtype, others unaffected.
    clone1.to_dtype(torch.float32)
    assert clone1.tableau.b.dtype == torch.float32
    assert canonical.tableau.b.dtype == torch.float64
    assert clone2.tableau.b.dtype == torch.float64

    # Order is copied (not shared reference behind tableau).
    assert clone1.order == canonical.order


def test_clone_followed_by_dtype_round_trip_loses_clone_precision_only():
    """A float32 round-trip on a clone is lossy as expected, but the
    canonical singleton's precision is preserved.
    """
    canonical = UNIFORM_METHODS["dopri5"]
    canonical_b_before = canonical.tableau.b.clone()

    clone = canonical.clone()
    clone.to_dtype(torch.float32)
    clone.to_dtype(torch.float64)

    # Canonical untouched.
    assert torch.equal(canonical.tableau.b, canonical_b_before)
    # Clone has lost float32-level precision (irreversible round-trip).
    # We verify this is the EXPECTED outcome: clone differs from
    # canonical at the ~1e-8 level (float32 epsilon ~1.19e-7).
    diff = (clone.tableau.b - canonical.tableau.b).abs().max().item()
    assert diff > 0  # there IS a precision loss on the clone
    assert diff < 1e-6  # but it's at most float32 ULP scale
