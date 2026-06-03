"""Unit tests for the vector-error machinery, exercised on float32 and float64.

These target the pieces added with the ``error_norm`` schemes:

  - ``_reduce_norm``           (scipy-style norm reductions over the D axis)
  - ``_round_floor``           (machine-precision tolerance floor)
  - ``_compute_error_ratios``  (norm vs failure_fraction families, both modes)
  - ``_accept_reject_masks``   (keep/split decision)

The focus is subtle, dtype-sensitive behavior that a coarse end-to-end test
would miss:

  - each norm reduces correctly *and preserves the input dtype*;
  - the rounding floor scales with ``finfo(dtype).eps`` -- so the
    floor-as-tolerance crossover happens at a far larger tolerance in float32
    than float64, which can flip a panel's accept/reject decision;
  - the norm family's strict ``error_ratio < 1`` threshold (exactly 1 splits);
  - the failure-fraction boundary relies on a dtype-scaled epsilon, without
    which the float32 representation of e.g. ``1/3`` would spuriously fail an
    equal tolerance;
  - non-finite (NaN/Inf) panels are always accepted in both families.
"""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import make_solver_for_unit_test

DTYPES = [torch.float32, torch.float64]
DTYPE_IDS = ["f32", "f64"]


def _solver(
    dtype,
    error_norm="2",
    mesh_failure_tolerance=0.0,
    atol=1e-3,
    rtol=1e-3,
    use_absolute_error_ratio=True,
):
    return make_solver_for_unit_test(
        atol=atol,
        rtol=rtol,
        dtype=dtype,
        error_norm=error_norm,
        mesh_failure_tolerance=mesh_failure_tolerance,
        use_absolute_error_ratio=use_absolute_error_ratio,
    )


def _close(dtype):
    """allclose tolerance appropriate for the dtype."""
    return 1e-5 if dtype == torch.float32 else 1e-12


# ---------------------------------------------------------------------------
# _reduce_norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestReduceNormDtype:
    def test_each_norm_value(self, dtype):
        s = _solver(dtype)
        x = torch.tensor([[3.0, 4.0]], dtype=dtype)  # [1, 2]
        atol = _close(dtype)
        assert torch.allclose(
            s._reduce_norm(x, "2"), torch.tensor([5.0], dtype=dtype), atol=atol
        )
        assert torch.allclose(
            s._reduce_norm(x, "max"), torch.tensor([4.0], dtype=dtype), atol=atol
        )
        assert torch.allclose(
            s._reduce_norm(x, "rms"),
            torch.sqrt(torch.tensor([12.5], dtype=dtype)),
            atol=atol,
        )

    def test_dtype_preserved(self, dtype):
        s = _solver(dtype)
        x = torch.tensor([[3.0, 4.0], [1.0, 2.0]], dtype=dtype)
        for norm in ("2", "max", "rms"):
            assert s._reduce_norm(x, norm).dtype == dtype
        assert s._reduce_norm(x, lambda z: z.sum(dim=-1)).dtype == dtype

    def test_reduces_only_last_axis(self, dtype):
        s = _solver(dtype)
        x = torch.zeros(2, 4, 3, dtype=dtype)
        for norm in ("2", "max", "rms"):
            assert s._reduce_norm(x, norm).shape == (2, 4)

    def test_max_uses_abs(self, dtype):
        s = _solver(dtype)
        x = torch.tensor([[-7.0, 4.0]], dtype=dtype)
        assert torch.allclose(
            s._reduce_norm(x, "max"), torch.tensor([7.0], dtype=dtype)
        )

    def test_1d_equivalent_to_abs(self, dtype):
        s = _solver(dtype)
        x = torch.tensor([[3.0], [-4.0]], dtype=dtype)
        expected = torch.tensor([3.0, 4.0], dtype=dtype)
        for norm in ("2", "max", "rms"):
            assert torch.allclose(s._reduce_norm(x, norm), expected)

    def test_reads_self_error_norm_when_none(self, dtype):
        s = _solver(dtype, error_norm="max")
        x = torch.tensor([[-7.0, 4.0]], dtype=dtype)
        assert torch.allclose(s._reduce_norm(x), torch.tensor([7.0], dtype=dtype))

    def test_unknown_norm_raises(self, dtype):
        s = _solver(dtype)
        with pytest.raises(ValueError, match="error_norm"):
            s._reduce_norm(torch.zeros(1, 2, dtype=dtype), "bogus")


# ---------------------------------------------------------------------------
# _round_floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestRoundFloorDtype:
    def test_value_and_dtype(self, dtype):
        s = _solver(dtype)
        mq = torch.tensor([[2.0], [-4.0]], dtype=dtype)
        eps = torch.finfo(dtype).eps
        expected = 50.0 * eps * torch.tensor([[2.0], [4.0]], dtype=dtype)
        floor = s._round_floor(mq)
        assert floor.dtype == dtype
        assert torch.allclose(floor, expected)


def test_float32_floor_exceeds_float64_floor():
    """The rounding floor is dtype-sensitive: float32's eps is ~9 orders of
    magnitude larger, so for the same |s_k| the float32 floor dominates a much
    larger tolerance band."""
    mq32 = torch.tensor([[1.0]], dtype=torch.float32)
    mq64 = torch.tensor([[1.0]], dtype=torch.float64)
    floor32 = _solver(torch.float32)._round_floor(mq32).item()
    floor64 = _solver(torch.float64)._round_floor(mq64).item()
    assert floor32 > floor64
    assert floor32 > 1e-7  # ~5.96e-6
    assert floor64 < 1e-12  # ~1.11e-14


# ---------------------------------------------------------------------------
# _compute_error_ratios: norm family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestErrorRatiosNormDtype:
    def test_uses_same_norm_for_error_and_integral(self, dtype):
        """tol is built from the *same* norm as the error: e=[1,0], I=[3,4]
        gives 1/5 under '2' but 1/4 under 'max'."""
        e = torch.tensor([[1.0, 0.0]], dtype=dtype)
        integral = torch.tensor([3.0, 4.0], dtype=dtype)
        atol = _close(dtype)

        s2 = _solver(dtype, error_norm="2", atol=0.0, rtol=1.0)
        r2, _, per_dim2 = s2._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        assert per_dim2 is None
        assert math.isclose(r2[0].item(), 1.0 / 5.0, rel_tol=10 * atol)

        smax = _solver(dtype, error_norm="max", atol=0.0, rtol=1.0)
        rmax, _, _ = smax._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        assert math.isclose(rmax[0].item(), 1.0 / 4.0, rel_tol=10 * atol)

    def test_dtype_preserved(self, dtype):
        s = _solver(dtype)
        e = torch.tensor([[1e-3, 2e-3], [3e-3, 4e-3]], dtype=dtype)
        integral = torch.tensor([1.0, 1.0], dtype=dtype)
        r, r2, _ = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        assert r.dtype == dtype
        assert r2.dtype == dtype

    def test_floor_crossover_flips_decision_by_dtype(self, dtype):
        """With a tolerance below the float32 floor but above the float64 floor,
        the *same* inputs are accepted in float32 (floor caps the tolerance) but
        rejected in float64 (the real tolerance bites)."""
        s = _solver(dtype, error_norm="2", atol=1e-10, rtol=1e-10)
        e = torch.tensor([[1e-8]], dtype=dtype)
        mq = torch.tensor([[1.0]], dtype=dtype)
        integral = torch.tensor([1.0], dtype=dtype)
        r, _, _ = s._compute_error_ratios(
            mesh_quadrature_errors=e, mesh_quadratures=mq, integral=integral
        )
        eps = torch.finfo(dtype).eps
        floor = 50.0 * eps * 1.0
        tol = 1e-10 + 1e-10 * 1.0
        effective_tol = max(tol, floor)
        assert math.isclose(r[0].item(), 1e-8 / effective_tol, rel_tol=1e-4)
        if dtype == torch.float32:
            assert r[0].item() < 1.0  # floor caps tol -> accepted
        else:
            assert r[0].item() > 1.0  # real tol bites -> rejected

    def test_no_floor_without_mesh_quadratures(self, dtype):
        """Omitting mesh_quadratures disables the floor (ratio = err/tol)."""
        s = _solver(dtype, error_norm="2", atol=1e-10, rtol=1e-10)
        e = torch.tensor([[1e-8]], dtype=dtype)
        integral = torch.tensor([1.0], dtype=dtype)
        r, _, _ = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        assert math.isclose(r[0].item(), 1e-8 / (2e-10), rel_tol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestErrorRatiosCumulativeDtype:
    def test_cumulative_denominator_grows(self, dtype):
        """Cumulative mode: constant per-step error gives monotonically
        decreasing ratios as the running integral grows."""
        s = _solver(
            dtype, error_norm="2", atol=1e-8, rtol=1e-8, use_absolute_error_ratio=False
        )
        n = 6
        e = torch.full((n, 1), 1e-9, dtype=dtype)
        mq = torch.full((n, 1), 0.1, dtype=dtype)
        r, r2, _ = s._compute_error_ratios(
            mesh_quadrature_errors=e, mesh_quadratures=mq
        )
        assert r.dtype == dtype
        assert torch.all(r[1:] <= r[:-1] + _close(dtype))
        assert r2.shape == (n - 1,)

    def test_missing_quadratures_raises(self, dtype):
        s = _solver(dtype, use_absolute_error_ratio=False)
        with pytest.raises(ValueError, match="Must give"):
            s._compute_error_ratios(
                mesh_quadrature_errors=torch.tensor([[1e-3]], dtype=dtype)
            )


# ---------------------------------------------------------------------------
# _compute_error_ratios: failure_fraction family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestFailureFractionRatiosDtype:
    def test_fraction_and_per_dim(self, dtype):
        s = _solver(dtype, error_norm="failure_fraction", atol=0.0, rtol=1.0)
        # tol = |I_d| per element; ratios [2, 0.5, 0.5] -> one failure -> 1/3.
        e = torch.tensor([[2.0, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=dtype)
        integral = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
        frac, two_step, per_dim = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        assert frac.dtype == dtype
        assert per_dim.shape == (2, 3)
        assert torch.allclose(
            frac, torch.tensor([1.0 / 3.0, 0.0], dtype=dtype), atol=_close(dtype)
        )
        assert two_step.shape == (1,)

    def test_rounding_limited_element_not_a_failure(self, dtype):
        """An element whose error is at/below its rounding floor must not count
        as a failure even when the requested tolerance is far below it."""
        s = _solver(
            dtype, error_norm="failure_fraction", atol=1e-30, rtol=1e-30
        )
        # Element 0 error (1e-16) is below its floor 50*eps*|s_k| for both
        # dtypes (~1.1e-14 f64, ~6e-6 f32) -> rounding-limited -> not a failure.
        # Element 1 genuinely fails (large error vs tiny tol & tiny floor).
        e = torch.tensor([[1e-16, 10.0]], dtype=dtype)
        mq = torch.tensor([[1.0, 1e-12]], dtype=dtype)
        integral = torch.tensor([1.0, 1.0], dtype=dtype)
        frac, _, _ = s._compute_error_ratios(
            mesh_quadrature_errors=e, mesh_quadratures=mq, integral=integral
        )
        # Only element 1 fails -> fraction 1/2.
        assert math.isclose(frac[0].item(), 0.5, rel_tol=_close(dtype))


# ---------------------------------------------------------------------------
# _accept_reject_masks: norm family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestAcceptRejectMasksNorm:
    def test_strict_threshold(self, dtype):
        """error_ratio < 1 keeps; exactly 1.0 splits."""
        s = _solver(dtype, error_norm="2")
        ratios = torch.tensor([0.999, 1.0, 1.001], dtype=dtype)
        keep, remove = s._accept_reject_masks(ratios, None)
        assert keep.tolist() == [True, False, False]
        assert remove.tolist() == [False, True, True]

    def test_nonfinite_kept(self, dtype):
        s = _solver(dtype, error_norm="2")
        ratios = torch.tensor(
            [float("inf"), float("-inf"), float("nan"), 0.5], dtype=dtype
        )
        keep, _ = s._accept_reject_masks(ratios, None)
        assert keep.tolist() == [True, True, True, True]

    def test_masks_are_bool_and_complementary(self, dtype):
        s = _solver(dtype, error_norm="max")
        ratios = torch.tensor([0.5, 2.0, float("nan")], dtype=dtype)
        keep, remove = s._accept_reject_masks(ratios, None)
        assert keep.dtype == torch.bool
        assert remove.dtype == torch.bool
        assert torch.equal(remove, ~keep)
        assert not torch.any(keep & remove)
        assert torch.all(keep | remove)

    def test_callable_error_norm_uses_norm_branch(self, dtype):
        """A callable error_norm must route through the norm branch (ratio < 1),
        never the failure-fraction branch (which would dereference per_dim)."""
        s = _solver(dtype, error_norm=lambda x: torch.sqrt((x**2).sum(dim=-1)))
        ratios = torch.tensor([0.5, 1.5], dtype=dtype)
        keep, _ = s._accept_reject_masks(ratios, None)  # per_dim None must be ok
        assert keep.tolist() == [True, False]


# ---------------------------------------------------------------------------
# _accept_reject_masks: failure_fraction family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
class TestAcceptRejectMasksFailure:
    def _make(self, dtype, mft):
        return _solver(
            dtype,
            error_norm="failure_fraction",
            mesh_failure_tolerance=mft,
            atol=0.0,
            rtol=1.0,
        )

    def test_tol_zero_any_failure_splits(self, dtype):
        s = self._make(dtype, mft=0.0)
        # fractions 0, 1/3, 1 over D=3.
        e = torch.tensor(
            [[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [2.0, 2.0, 2.0]], dtype=dtype
        )
        integral = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
        frac, _, per_dim = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        keep, remove = s._accept_reject_masks(frac, per_dim)
        assert keep.tolist() == [True, False, False]
        assert remove.tolist() == [False, True, True]

    def test_third_boundary_relies_on_dtype_epsilon(self, dtype):
        """fraction == mesh_failure_tolerance must be ACCEPTED. With D=3 and one
        failure the fraction is 1/3, whose float32 value rounds slightly above
        1/3; only the dtype-scaled epsilon in the mask keeps it accepted."""
        s = self._make(dtype, mft=1.0 / 3.0)
        e = torch.tensor([[2.0, 0.5, 0.5]], dtype=dtype)
        integral = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
        frac, _, per_dim = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        keep, _ = s._accept_reject_masks(frac, per_dim)
        assert keep.tolist() == [True]

    def test_half_tolerance_D2(self, dtype):
        s = self._make(dtype, mft=0.5)
        # D=2: fractions 0, 0.5, 1.
        e = torch.tensor(
            [[0.5, 0.5], [2.0, 0.5], [2.0, 2.0]], dtype=dtype
        )
        integral = torch.tensor([1.0, 1.0], dtype=dtype)
        frac, _, per_dim = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        keep, _ = s._accept_reject_masks(frac, per_dim)
        assert keep.tolist() == [True, True, False]

    def test_nonfinite_panel_always_kept(self, dtype):
        """A panel with any NaN/Inf element is kept even if its finite elements
        would otherwise push the failure fraction past the tolerance."""
        s = self._make(dtype, mft=0.0)
        e = torch.tensor([[float("nan"), 2.0, 2.0]], dtype=dtype)
        integral = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
        frac, _, per_dim = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        # Two finite elements fail -> fraction 2/3 > mft, but the NaN forces keep.
        keep, _ = s._accept_reject_masks(frac, per_dim)
        assert keep.tolist() == [True]

    def test_masks_complementary(self, dtype):
        s = self._make(dtype, mft=0.5)
        e = torch.tensor([[0.5, 0.5], [2.0, 2.0]], dtype=dtype)
        integral = torch.tensor([1.0, 1.0], dtype=dtype)
        frac, _, per_dim = s._compute_error_ratios(
            mesh_quadrature_errors=e, integral=integral
        )
        keep, remove = s._accept_reject_masks(frac, per_dim)
        assert keep.dtype == torch.bool
        assert torch.equal(remove, ~keep)
