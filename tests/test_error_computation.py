"""Unit tests for norm reduction, _rec_remove, and error ratio computation."""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import make_solver_for_unit_test

# ---------------------------------------------------------------------------
# _reduce_norm
# ---------------------------------------------------------------------------


class TestReduceNorm:
    """Tests for _reduce_norm: scipy-style reductions over the last (D) axis."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_rms_1d_errors(self):
        """Single-dimension errors: any norm reduces to abs."""
        error = torch.tensor([[3.0], [-4.0]])  # [2, 1]
        result = self.solver._reduce_norm(error, error_norm="rms")
        assert torch.allclose(result, torch.tensor([3.0, 4.0]))

    def test_rms_multidim(self):
        """RMS across D=2: sqrt(mean([9, 16])) = sqrt(12.5)."""
        error = torch.tensor([[3.0, 4.0]])  # [1, 2]
        result = self.solver._reduce_norm(error, error_norm="rms")
        assert torch.allclose(result, torch.sqrt(torch.tensor([12.5])))

    def test_l2_multidim(self):
        """L2 (scipy '2') across D=2: sqrt(9 + 16) = 5."""
        error = torch.tensor([[3.0, 4.0]])
        result = self.solver._reduce_norm(error, error_norm="2")
        assert torch.allclose(result, torch.tensor([5.0]))

    def test_max_multidim(self):
        """L-infinity (scipy 'max') across D=2: max(|[-3, 4]|) = 4."""
        error = torch.tensor([[-3.0, 4.0]])
        result = self.solver._reduce_norm(error, error_norm="max")
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_callable(self):
        """A callable norm is honored (here: sum of abs along last axis)."""
        error = torch.tensor([[3.0, -4.0]])
        result = self.solver._reduce_norm(
            error, error_norm=lambda x: torch.abs(x).sum(dim=-1)
        )
        assert torch.allclose(result, torch.tensor([7.0]))

    def test_reads_self_error_norm(self):
        """With no override, _reduce_norm uses self.error_norm."""
        self.solver.error_norm = "max"
        error = torch.tensor([[-3.0, 4.0]])
        assert torch.allclose(self.solver._reduce_norm(error), torch.tensor([4.0]))

    def test_zero(self):
        """Zero errors give zero norms."""
        error = torch.zeros(5, 3)
        result = self.solver._reduce_norm(error, error_norm="2")
        assert torch.allclose(result, torch.zeros(5))


# ---------------------------------------------------------------------------
# _round_floor
# ---------------------------------------------------------------------------


class TestRoundFloor:
    """Tests for the machine-precision rounding floor."""

    def test_floor_value(self):
        solver = make_solver_for_unit_test()
        mesh_quadratures = torch.tensor([[2.0], [-4.0]], dtype=solver.dtype)
        floor = solver._round_floor(mesh_quadratures)
        eps = torch.finfo(solver.dtype).eps
        expected = 50.0 * eps * torch.tensor([[2.0], [4.0]], dtype=solver.dtype)
        assert torch.allclose(floor, expected)

    def test_floor_caps_tolerance_below_precision(self):
        """When the requested tolerance is below the rounding floor, the floor
        becomes the effective tolerance, so a tiny error is accepted (ratio < 1)
        instead of triggering endless refinement."""
        # atol/rtol far below what float64 can resolve relative to |s_k|.
        solver = make_solver_for_unit_test(atol=1e-20, rtol=1e-20)
        mesh_quadratures = torch.tensor([[1.0]], dtype=solver.dtype)
        integral = torch.tensor([1.0], dtype=solver.dtype)
        # Error below the rounding floor (50*eps*|s_k| ~ 1.1e-14).
        mesh_quadrature_errors = torch.tensor([[1e-16]], dtype=solver.dtype)

        with_floor, _, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors,
            mesh_quadratures=mesh_quadratures,
            integral=integral,
        )
        eps = torch.finfo(solver.dtype).eps
        # effective_tol = max(2e-20, 50*eps*1.0) = floor; ratio = err / floor.
        expected = 1e-16 / (50.0 * eps * 1.0)
        assert math.isclose(with_floor[0].item(), expected, rel_tol=1e-9)
        assert with_floor[0].item() < 1.0  # accepted, not split

        # Without mesh_quadratures (no floor) the same inputs blow up the ratio.
        no_floor, _, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )
        assert no_floor[0].item() > 1.0


# ---------------------------------------------------------------------------
# _rec_remove
# ---------------------------------------------------------------------------


class TestRecRemove:
    """Tests for _rec_remove: ensure no adjacent True values in a boolean mask."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def _check_no_adjacent(self, mask):
        """Helper: verify no two adjacent True values."""
        if len(mask) < 2:
            return
        adjacent = mask[:-1] & mask[1:]
        assert not torch.any(adjacent), f"Adjacent Trues found in {mask}"

    def test_no_adjacent_trues(self):
        """Already valid mask is unchanged."""
        mask = torch.tensor([True, False, True, False, True])
        result = self.solver._rec_remove(mask.clone())
        assert torch.equal(result, mask)

    def test_pair_at_start(self):
        """Adjacent pair at start: second is removed."""
        mask = torch.tensor([True, True, False, True])
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        assert result[0] == True  # noqa: E712  # First kept

    def test_all_true(self):
        """All True: result alternates True/False."""
        mask = torch.ones(5, dtype=torch.bool)
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        # At least ceil(N/2) Trues remain
        assert result.sum() >= 3

    def test_all_false(self):
        """All False: unchanged."""
        mask = torch.zeros(3, dtype=torch.bool)
        result = self.solver._rec_remove(mask.clone())
        assert torch.equal(result, mask)

    def test_single_element(self):
        """Single True element is unchanged."""
        mask = torch.tensor([True])
        result = self.solver._rec_remove(mask.clone())
        assert result[0] == True  # noqa: E712

    def test_two_both_true(self):
        """Two adjacent Trues: second is removed."""
        mask = torch.tensor([True, True])
        result = self.solver._rec_remove(mask.clone())
        expected = torch.tensor([True, False])
        assert torch.equal(result, expected)

    def test_three_adjacent(self):
        """Three adjacent Trues: result is [T, F, T]."""
        mask = torch.tensor([True, True, True])
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        assert result[0] == True  # noqa: E712
        assert result[2] == True  # noqa: E712

    def test_long_alternating(self):
        """Already alternating mask of length 20 is unchanged."""
        mask = torch.tensor([i % 2 == 0 for i in range(20)])
        result = self.solver._rec_remove(mask.clone())
        assert torch.equal(result, mask)

    def test_long_all_true(self):
        """All True of length 20: result has no adjacent Trues."""
        mask = torch.ones(20, dtype=torch.bool)
        result = self.solver._rec_remove(mask.clone())
        self._check_no_adjacent(result)
        assert result.sum() >= 10  # At least half remain


# ---------------------------------------------------------------------------
# _compute_error_ratios: norm family, absolute mode
# ---------------------------------------------------------------------------


class TestComputeErrorRatiosAbsolute:
    """Tests for absolute-mode error ratio computation (norm family)."""

    def _make_solver(self, atol=1e-3, rtol=1e-3):
        solver = make_solver_for_unit_test(atol=atol, rtol=rtol)
        solver.use_absolute_error_ratio = True
        return solver

    def test_basic(self):
        """Error ratios correctly identify passing and failing steps."""
        solver = self._make_solver(atol=1e-3, rtol=1e-3)
        # error_tol = atol + rtol * |integral| = 1e-3 + 1e-3 * 1.0 = 2e-3
        mesh_quadrature_errors = torch.tensor([[0.01], [0.001]])  # [2, 1]
        integral = torch.tensor([1.0])

        error_ratio, _error_ratio_2steps, _per_dim = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )

        # 0.01 / 2e-3 = 5.0 (failing), 0.001 / 2e-3 = 0.5 (passing)
        assert error_ratio[0] > 1.0
        assert error_ratio[1] < 1.0

    def test_zero_integral(self):
        """Zero integral: error_tol = atol only, no NaN."""
        solver = self._make_solver(atol=1e-3, rtol=1e-3)
        mesh_quadrature_errors = torch.tensor([[1e-4]])
        integral = torch.tensor([0.0])

        error_ratio, _, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )

        assert torch.isfinite(error_ratio).all()
        # 1e-4 / 1e-3 = 0.1
        assert torch.allclose(error_ratio, torch.tensor([0.1]))

    def test_2steps_shape(self):
        """N=3 steps: error_ratio has len 3, error_ratio_2steps has len 2."""
        solver = self._make_solver()
        mesh_quadrature_errors = torch.tensor([[0.001], [0.002], [0.003]])
        integral = torch.tensor([1.0])

        error_ratio, error_ratio_2steps, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )

        assert len(error_ratio) == 3
        assert len(error_ratio_2steps) == 2

    def test_single_step(self):
        """N=1: error_ratio has len 1, error_ratio_2steps has len 0."""
        solver = self._make_solver()
        mesh_quadrature_errors = torch.tensor([[0.001]])
        integral = torch.tensor([1.0])

        error_ratio, error_ratio_2steps, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )

        assert len(error_ratio) == 1
        assert len(error_ratio_2steps) == 0

    def test_multidim_l2(self):
        """D=2 with default '2' norm: reduce-then-compare L2."""
        solver = self._make_solver()  # default error_norm == "2"
        mesh_quadrature_errors = torch.tensor([[0.003, 0.004]])  # [1, 2]
        integral = torch.tensor([1.0, 1.0])

        error_ratio, _, per_dim = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )

        # err = L2([0.003, 0.004]) = 0.005; tol = 1e-3 + 1e-3 * L2([1, 1])
        tol = 1e-3 + 1e-3 * math.sqrt(2.0)
        assert error_ratio.shape == (1,)
        assert math.isclose(error_ratio[0].item(), 0.005 / tol, rel_tol=1e-6)
        # Norm family returns no per-dim ratio.
        assert per_dim is None

    def test_multidim_max(self):
        """D=2 with 'max' norm: reduce-then-compare L-infinity."""
        solver = self._make_solver()
        solver.error_norm = "max"
        mesh_quadrature_errors = torch.tensor([[0.003, 0.004]])
        integral = torch.tensor([1.0, 1.0])

        error_ratio, _, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )
        tol = 1e-3 + 1e-3 * 1.0  # max(|[1, 1]|) = 1
        assert math.isclose(error_ratio[0].item(), 0.004 / tol, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# _compute_error_ratios: norm family, cumulative mode
# ---------------------------------------------------------------------------


class TestComputeErrorRatiosCumulative:
    """Tests for cumulative-mode error ratio computation (norm family)."""

    def _make_solver(self, atol=1e-3, rtol=1e-3):
        solver = make_solver_for_unit_test(atol=atol, rtol=rtol)
        solver.use_absolute_error_ratio = False
        return solver

    def test_basic_with_mesh_quadratures(self):
        """Cumulative error ratios computed from mesh_quadratures."""
        solver = self._make_solver()
        mesh_quadrature_errors = torch.tensor([[0.001], [0.002]])
        mesh_quadratures = torch.tensor([[1.0], [2.0]])

        error_ratio, error_ratio_2steps, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors,
            mesh_quadratures=mesh_quadratures,
        )

        assert len(error_ratio) == 2
        assert len(error_ratio_2steps) == 1
        assert torch.isfinite(error_ratio).all()

    def test_with_cum_mesh_quadratures(self):
        """Passing cum_mesh_quadratures directly gives the same result."""
        solver = self._make_solver()
        mesh_quadrature_errors = torch.tensor([[0.001], [0.002]])
        mesh_quadratures = torch.tensor([[1.0], [2.0]])
        cum_mesh_quadratures = torch.cumsum(mesh_quadratures, dim=0)

        r1, r2_1, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors,
            mesh_quadratures=mesh_quadratures,
        )
        r3, r2_2, _ = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors,
            cum_mesh_quadratures=cum_mesh_quadratures,
        )

        assert torch.allclose(r1, r3)
        assert torch.allclose(r2_1, r2_2)

    def test_missing_args_raises(self):
        """Cumulative mode with no quadratures and no cumsum raises ValueError."""
        solver = self._make_solver()
        mesh_quadrature_errors = torch.tensor([[0.001]])

        with pytest.raises(ValueError, match="Must give"):
            solver._compute_error_ratios(mesh_quadrature_errors=mesh_quadrature_errors)


# ---------------------------------------------------------------------------
# _compute_error_ratios: failure_fraction family + _accept_reject_masks
# ---------------------------------------------------------------------------


class TestFailureFraction:
    """Tests for the per-component failure-fraction scheme."""

    def _make_solver(self, mesh_failure_tolerance=0.0, atol=1e-3, rtol=1e-3):
        solver = make_solver_for_unit_test(atol=atol, rtol=rtol)
        solver.error_norm = "failure_fraction"
        solver.mesh_failure_tolerance = mesh_failure_tolerance
        solver.use_absolute_error_ratio = True
        return solver

    def test_returns_fraction_and_per_dim(self):
        """failure_fraction is per-step [N]; per-dim ratio is [N, D]."""
        solver = self._make_solver()
        # tol = 1e-3 + 1e-3 * 1 = 2e-3 per element.
        # Step 0: [0.01 (fail), 0.0001 (pass)] -> fraction 0.5
        # Step 1: [0.0001, 0.0001] -> fraction 0.0
        mesh_quadrature_errors = torch.tensor([[0.01, 0.0001], [0.0001, 0.0001]])
        integral = torch.tensor([1.0, 1.0])

        failure_fraction, two_step, per_dim = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )

        assert per_dim.shape == (2, 2)
        assert torch.allclose(
            failure_fraction, torch.tensor([0.5, 0.0], dtype=solver.dtype)
        )
        assert two_step.shape == (1,)

    def test_mask_tol_zero_rejects_any_failure(self):
        """With mesh_failure_tolerance=0, any failing element splits the panel."""
        solver = self._make_solver(mesh_failure_tolerance=0.0)
        mesh_quadrature_errors = torch.tensor([[0.01, 0.0001], [0.0001, 0.0001]])
        integral = torch.tensor([1.0, 1.0])
        failure_fraction, _, per_dim = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )
        keep, remove = solver._accept_reject_masks(failure_fraction, per_dim)
        assert keep.tolist() == [False, True]
        assert remove.tolist() == [True, False]

    def test_mask_tol_half_accepts(self):
        """With mesh_failure_tolerance=0.5, a 50%-failing panel is accepted."""
        solver = self._make_solver(mesh_failure_tolerance=0.5)
        mesh_quadrature_errors = torch.tensor([[0.01, 0.0001]])
        integral = torch.tensor([1.0, 1.0])
        failure_fraction, _, per_dim = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )
        keep, _ = solver._accept_reject_masks(failure_fraction, per_dim)
        assert keep.tolist() == [True]

    def test_mask_nonfinite_panel_accepted(self):
        """A panel with a non-finite element is accepted regardless of tol."""
        solver = self._make_solver(mesh_failure_tolerance=0.0)
        mesh_quadrature_errors = torch.tensor([[float("nan"), 0.01]])
        integral = torch.tensor([1.0, 1.0])
        failure_fraction, _, per_dim = solver._compute_error_ratios(
            mesh_quadrature_errors=mesh_quadrature_errors, integral=integral
        )
        keep, _ = solver._accept_reject_masks(failure_fraction, per_dim)
        assert keep.tolist() == [True]


class TestAcceptRejectMasksNorm:
    """Norm-family acceptance: ratio < 1 keeps, non-finite keeps."""

    def test_threshold(self):
        solver = make_solver_for_unit_test()  # default error_norm == "2"
        error_ratios = torch.tensor([0.5, 1.0, 2.0])
        keep, remove = solver._accept_reject_masks(error_ratios, None)
        assert keep.tolist() == [True, False, False]
        assert remove.tolist() == [False, True, True]

    def test_nonfinite_kept(self):
        solver = make_solver_for_unit_test()
        error_ratios = torch.tensor([float("inf"), float("nan"), 0.1])
        keep, _ = solver._accept_reject_masks(error_ratios, None)
        assert keep.tolist() == [True, True, True]
