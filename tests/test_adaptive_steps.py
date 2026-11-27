"""Unit tests for _adaptively_increase_mesh: core adaptive refinement logic."""

from __future__ import annotations

import pytest
import torch
from _helpers import make_solver_for_unit_test

from padaquad.results import MethodOutput


def _make_method_output(N, D=1):
    """Create a synthetic MethodOutput with N steps and D output dims."""
    return MethodOutput(
        integral=torch.ones(D, dtype=torch.float64),
        integral_error=torch.ones(D, dtype=torch.float64) * 0.01,
        mesh_quadratures=torch.ones(N, D, dtype=torch.float64),
        mesh_quadrature_errors=torch.ones(N, D, dtype=torch.float64) * 0.01,
        h=torch.ones(N, 1, dtype=torch.float64) * 0.5,
    )


def _make_y_and_nodes(N, C, D, T):
    """y_step_eval [N, C, D] and nodes [N, C, T] with row i filled with the
    value ``i``, so a filter can be checked by reading the surviving row labels.
    """
    rows = torch.arange(N, dtype=torch.float64)
    y = rows.view(N, 1, 1).expand(N, C, D).clone()
    nodes = rows.view(N, 1, 1).expand(N, C, T).clone()
    return y, nodes


def _assert_columns_increasing(mesh):
    """Every column of a [M, T] barrier array is strictly increasing.

    For a vector mesh (T > 1) this is the per-component ordering invariant: each
    time dimension must advance monotonically along the (monotonic-by-
    construction) test meshes used here.
    """
    diffs = mesh[1:] - mesh[:-1]
    assert torch.all(diffs > 0), f"Columns not strictly increasing:\n{mesh}"


def _keep_mask(error_ratios):
    """The accept mask the integration loop passes for these error ratios
    (``_accept_reject_masks`` of the default norm: below 1, or non-finite)."""
    return make_solver_for_unit_test()._accept_reject_masks(error_ratios, None)[0]


def _assert_transfer(idxs_transfer, barriers, barriers_new):
    """``idxs_transfer`` gives the position in the new mesh of every old mesh
    point (the map the record's mesh indices follow): increasing, and the new
    mesh holds each old point there, bit for bit."""
    assert torch.all(idxs_transfer[1:] > idxs_transfer[:-1])
    assert torch.equal(barriers_new[idxs_transfer], barriers)


class TestAdaptivelyAddSteps:
    """Tests for AdaptiveQuadrature._adaptively_increase_mesh.

    The method returns a 9-tuple:
    (method_output, y, tracked, nodes, mesh_new, trackers_new, idxs_transfer,
     keep_idx, split_counts_new); keep_idx holds the batch positions of the
    accepted subintervals.
    """

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_all_pass(self):
        """All error_ratios < 1: barriers unchanged, all trackers False."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([0.5, 0.3])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, trackers_new, mesh_idx, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == len(barriers)
        assert not torch.any(trackers_new[:2])
        assert len(er_kept) == 2
        assert mo_out.mesh_quadratures.shape[0] == 2
        assert mesh_idx.tolist() == [0, 1, 2]

    def test_all_fail(self):
        """All error_ratios >= 1: midpoints inserted, no subinterval kept."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 1.5])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _trackers_new, mesh_idx, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        # 2 midpoints added: len goes from 3 to 5
        assert len(barriers_new) == 5
        assert len(er_kept) == 0
        assert mo_out.mesh_quadratures.shape[0] == 0
        # The old points move past the inserted midpoints.
        assert mesh_idx.tolist() == [0, 2, 4]
        _assert_transfer(mesh_idx, barriers, barriers_new)

    def test_mixed_pass_fail(self):
        """First passes, second fails: 1 midpoint added."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([0.5, 2.0])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _trackers_new, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        # 1 midpoint added: len goes from 3 to 4
        assert len(barriers_new) == 4
        assert len(er_kept) == 1
        assert mo_out.mesh_quadratures.shape[0] == 1

    def test_none_method_output(self):
        """method_output=None (post-convergence): returns None, barriers updated."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 0.5])

        mo_out, y_out, _tracked, t_out, barriers_new, _, _, _, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        assert mo_out is None
        assert y_out is None
        assert t_out is None
        # 1 midpoint added for the failing step
        assert len(barriers_new) == 4

    def test_midpoint_placement(self):
        """Midpoint of failing step [0, 1] is exactly 0.5."""
        barriers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([3.0])

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        assert torch.allclose(barriers_new[1], torch.tensor([0.5], dtype=torch.float64))

    def test_barrier_ordering(self):
        """After multiple fails, barriers remain sorted ascending."""
        barriers = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([2.0, 2.0, 2.0])

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        diffs = barriers_new[1:, 0] - barriers_new[:-1, 0]
        assert torch.all(diffs > 0), f"Barriers not sorted: {barriers_new[:, 0]}"

    def test_tracker_new_steps_true(self):
        """New midpoint positions are marked True in trackers."""
        barriers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([2.0])

        _, _, _, _, _barriers_new, trackers_new, _, _, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        # After split: barriers = [0, 0.5, 1]. Both step 0 and step 1 need eval.
        assert trackers_new[0] == True  # noqa: E712
        assert trackers_new[1] == True  # noqa: E712

    def test_method_output_filtered(self):
        """3 steps, middle fails: method_output retains 2 accepted rows."""
        barriers = torch.tensor([[0.0], [0.33], [0.67], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([0.5, 2.0, 0.3])
        mo = _make_method_output(3)

        mo_out, _, _, _, _, _, _, er_kept, _ = self.solver._adaptively_increase_mesh(
            mo, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        assert mo_out.mesh_quadratures.shape[0] == 2
        assert mo_out.h.shape[0] == 2
        # The positions in the batch of the accepted rows, which place them
        # in the record.
        assert er_kept.tolist() == [0, 2]

    def test_single_step_passes(self):
        """Single step that passes: barriers unchanged."""
        barriers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([0.5])

        _, _, _, _, barriers_new, trackers_new, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == 2
        assert trackers_new[0] == False  # noqa: E712
        assert len(er_kept) == 1

    def test_keep_mask_drives_accept_split(self):
        """keep_mask decides alone: False splits, True accepts."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, torch.tensor([False, True]), None, None, barriers, idxs, trackers
            )
        )
        # One step split -> one midpoint added (3 -> 4); one step kept.
        assert len(barriers_new) == 4
        assert er_kept.tolist() == [1]
        assert mo_out.mesh_quadratures.shape[0] == 1


class TestVectorMeshOrdering:
    """Vector mesh (T > 1): midpoints, per-component ordering, idxs_transfer.

    The mesh barrier array has shape [M, T]; every transfer/insert operation
    must act per component and keep each column ordered. These meshes are
    monotonic in every column by construction so per-column ordering is
    assertable.
    """

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_midpoint_per_component_T2(self):
        """Single failing T=2 panel: midpoint is the per-component average."""
        barriers = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([3.0])

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        assert barriers_new.shape == (3, 2)
        assert torch.allclose(
            barriers_new[1], torch.tensor([0.5, 1.0], dtype=torch.float64)
        )
        _assert_columns_increasing(barriers_new)

    def test_midpoint_per_component_T3(self):
        """T=3 with non-uniform spans: each component averaged independently."""
        barriers = torch.tensor(
            [[0.0, 10.0, -4.0], [2.0, 14.0, 0.0]], dtype=torch.float64
        )
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([3.0])

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        assert barriers_new.shape == (3, 3)
        assert torch.allclose(
            barriers_new[1], torch.tensor([1.0, 12.0, -2.0], dtype=torch.float64)
        )

    def test_all_fail_T2_ordering(self):
        """Two T=2 panels both fail: full expanded mesh matches hand value."""
        barriers = torch.tensor(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]], dtype=torch.float64
        )
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 2.0])

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        expected = torch.tensor(
            [[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 1.5], [1.0, 2.0]],
            dtype=torch.float64,
        )
        assert torch.allclose(barriers_new, expected)
        _assert_columns_increasing(barriers_new)

    def test_idxs_transfer_returned_T2(self):
        """The returned idxs_transfer places every old mesh point of a vector
        mesh in the new mesh: integer positions, no coordinates compared."""
        barriers = torch.tensor(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]], dtype=torch.float64
        )
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 2.0])

        _, _, _, _, barriers_new, _, mesh_idx, _, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        _assert_transfer(mesh_idx, barriers, barriers_new)
        assert mesh_idx.tolist() == [0, 2, 4]

    def test_consecutive_fails_T2_ordering(self):
        """Three adjacent T=2 fails (stresses the cumsum offset): columns stay
        ordered and three midpoints are inserted."""
        barriers = torch.tensor(
            [[0.0, 0.0], [0.25, 1.0], [0.5, 2.0], [1.0, 3.0]], dtype=torch.float64
        )
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([2.0, 2.0, 2.0])

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        assert len(barriers_new) == 7  # 4 + 3 midpoints
        _assert_columns_increasing(barriers_new)


class TestNonFiniteErrorRatios:
    """Non-finite (NaN/Inf) error ratios are accepted, never split.

    Splitting cannot fix a non-finite panel (the boundary node regenerates the
    NaN/Inf), so the legacy mask rule accepts them so the loop terminates.
    """

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_nan_ratio_accepted(self):
        barriers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([float("nan")])

        _, _, _, _, barriers_new, trackers_new, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == 2  # no midpoint inserted
        assert trackers_new[0] == False  # noqa: E712  accepted -> done
        assert len(er_kept) == 1

    def test_inf_ratio_accepted(self):
        barriers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([float("inf")])

        _, _, _, _, barriers_new, trackers_new, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == 2
        assert trackers_new[0] == False  # noqa: E712
        assert len(er_kept) == 1

    def test_mixed_finite_and_nonfinite(self):
        """[0.5, inf, 2.0]: keep the small and the non-finite, split only 2.0."""
        barriers = torch.tensor([[0.0], [0.33], [0.67], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([0.5, float("inf"), 2.0])

        _, _, _, _, barriers_new, _, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        # Only the 2.0 panel splits -> exactly one midpoint added (4 -> 5).
        assert len(barriers_new) == 5
        assert len(er_kept) == 2  # 0.5 and inf accepted


class TestSplitCounts:
    """max_adaptive_splits cap and split_counts propagation."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_split_counts_none_returns_none(self):
        """split_counts=None (post-convergence caller) -> split_counts_new None."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 0.5])

        *_, split_counts_new = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        assert split_counts_new is None

    def test_children_get_parent_plus_one(self):
        """A split panel's two children are bumped to parent_count + 1."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 0.5])  # panel 0 splits
        split_counts = torch.tensor([0, 0, 0])

        *_, split_counts_new = self.solver._adaptively_increase_mesh(
            None,
            _keep_mask(error_ratios),
            None,
            None,
            barriers,
            idxs,
            trackers,
            split_counts=split_counts,
            max_adaptive_splits=5,
        )
        # mesh_new = [0, 0.25, 0.5, 1.0]; children of [0,0.5] are barriers 0
        # and 0.25 (count 1); kept panel start (0.5) and end (1.0) stay 0.
        assert split_counts_new.tolist() == [1, 1, 0, 0]

    def test_split_counts_transferred(self):
        """Counts on untouched panels carry to their shifted positions."""
        barriers = torch.tensor([[0.0], [0.25], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([2.0, 0.5, 0.5])  # only panel 0 splits
        split_counts = torch.tensor([0, 2, 5, 1])

        *_, split_counts_new = self.solver._adaptively_increase_mesh(
            None,
            _keep_mask(error_ratios),
            None,
            None,
            barriers,
            idxs,
            trackers,
            split_counts=split_counts,
            max_adaptive_splits=10,
        )
        # mesh_new = [0, 0.125, 0.25, 0.5, 1.0]; children of panel 0 -> count 1,
        # the 2/5/1 counts ride along to positions 2/3/4.
        assert split_counts_new.tolist() == [1, 1, 2, 5, 1]

    def test_at_max_splits_accepted_not_split(self):
        """A failing panel already at max_adaptive_splits is accepted; a
        below-cap sibling still splits."""
        barriers = torch.tensor([[0.0], [0.25], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        # Panels 0 and 1 both fail; panel 0 is already at the cap.
        error_ratios = torch.tensor([2.0, 2.0, 0.5])
        split_counts = torch.tensor([3, 0, 0, 0])

        _, _, _, _, barriers_new, _, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None,
                _keep_mask(error_ratios),
                None,
                None,
                barriers,
                idxs,
                trackers,
                split_counts=split_counts,
                max_adaptive_splits=3,
            )
        )
        # Only panel 1 splits -> one midpoint (4 -> 5). Panel 0 accepted despite
        # ratio 2.0, so it is in er_kept.
        assert len(barriers_new) == 5
        assert er_kept.tolist() == [0, 2]

    def test_below_max_still_splits(self):
        """Below the cap, a high-error panel splits and bumps the count."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 0.5])
        split_counts = torch.tensor([1, 0, 0])

        _, _, _, _, barriers_new, _, _, _, split_counts_new = (
            self.solver._adaptively_increase_mesh(
                None,
                _keep_mask(error_ratios),
                None,
                None,
                barriers,
                idxs,
                trackers,
                split_counts=split_counts,
                max_adaptive_splits=5,
            )
        )
        assert len(barriers_new) == 4  # split happened
        # Children of the depth-1 panel become depth 2.
        assert split_counts_new.tolist() == [2, 2, 0, 0]

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    @pytest.mark.parametrize("T", [1, 2])
    def test_too_narrow_to_halve_accepted_not_split(self, dtype, T):
        """A failing subinterval whose ends are neighboring floating-point
        numbers (its midpoint rounds onto an end, in every coordinate) is
        accepted, as at the cap, with no cap given; a failing wide sibling
        still splits, and the mesh stays strictly increasing."""
        one = torch.tensor(1.0, dtype=dtype)
        after_one = torch.nextafter(one, torch.tensor(2.0, dtype=dtype))
        points = torch.stack([torch.zeros((), dtype=dtype), one, after_one])
        # T = 2: the second coordinate is twice the first, also neighbors.
        barriers = points.unsqueeze(-1) * torch.arange(1, T + 1, dtype=dtype)
        idxs = torch.tensor([0, 1])
        assert self.solver._splittable(barriers, idxs, None, None).tolist() == [
            True,
            False,
        ]

        _, _, _, _, barriers_new, trackers_new, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None,
                torch.tensor([False, False]),
                None,
                None,
                barriers,
                idxs,
                torch.tensor([True, True, False]),
            )
        )
        assert er_kept.tolist() == [1]
        assert len(barriers_new) == 4
        _assert_columns_increasing(barriers_new)
        assert trackers_new.tolist() == [True, True, False, False]


class TestDataFiltering:
    """y_step_eval / nodes / tracked_step_eval / method_output filtering.

    These args are passed as real (non-None) tensors so the keep_mask filtering
    branches are exercised, including D > 1 and T > 1 trailing axes.
    """

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_y_step_eval_filtered_to_kept(self):
        """D=3 integrand: middle panel rejected -> kept rows survive in order."""
        barriers = torch.tensor([[0.0], [0.33], [0.67], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([0.5, 2.0, 0.3])  # keep rows 0 and 2
        y, _ = _make_y_and_nodes(N=3, C=2, D=3, T=1)

        _, y_out, _, _, _, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), y, None, barriers, idxs, trackers
        )
        assert y_out.shape == (2, 2, 3)
        assert y_out[:, 0, 0].tolist() == [0.0, 2.0]

    def test_nodes_filtered_to_kept(self):
        """nodes [N, C, T=2]: kept rows survive in order."""
        barriers = torch.tensor(
            [[0.0, 0.0], [0.33, 0.33], [0.67, 0.67], [1.0, 1.0]], dtype=torch.float64
        )
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([0.5, 2.0, 0.3])
        _, nodes = _make_y_and_nodes(N=3, C=2, D=1, T=2)

        _, _, _, nodes_out, _, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, nodes, barriers, idxs, trackers
        )
        assert nodes_out.shape == (2, 2, 2)
        assert nodes_out[:, 0, 0].tolist() == [0.0, 2.0]

    def test_tracked_step_eval_tuple_filtered(self):
        """Each tracked-variable tensor in the tuple is filtered to kept rows."""
        barriers = torch.tensor([[0.0], [0.33], [0.67], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([0.5, 2.0, 0.3])
        rows = torch.arange(3, dtype=torch.float64)
        track_a = rows.view(3, 1).expand(3, 5).clone()
        track_b = rows.view(3, 1, 1).expand(3, 2, 4).clone()

        _, _, tracked_out, _, _, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None,
            _keep_mask(error_ratios),
            None,
            None,
            barriers,
            idxs,
            trackers,
            tracked_step_eval=(track_a, track_b),
        )
        assert tracked_out[0].shape == (2, 5)
        assert tracked_out[1].shape == (2, 2, 4)
        assert tracked_out[0][:, 0].tolist() == [0.0, 2.0]
        assert tracked_out[1][:, 0, 0].tolist() == [0.0, 2.0]

    def test_method_output_resummed_D_gt_1(self):
        """D=3: integral is re-summed per component over the kept rows only."""
        barriers = torch.tensor([[0.0], [0.33], [0.67], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([0.5, 2.0, 0.3])  # drop the middle row
        mo = _make_method_output(3, D=3)

        mo_out, _, _, _, _, _, _, _, _ = self.solver._adaptively_increase_mesh(
            mo, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        assert mo_out.mesh_quadratures.shape == (2, 3)
        # Two kept rows of all-ones -> 2.0 per component.
        assert torch.allclose(
            mo_out.integral, torch.full((3,), 2.0, dtype=torch.float64)
        )
        assert torch.allclose(
            mo_out.integral_error, torch.full((3,), 0.02, dtype=torch.float64)
        )


class TestPartialAndBoundary:
    """Partial batches (skipped panels) and boundary-panel splits."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_noncontiguous_mesh_idxs(self):
        """Batch covers only panels 1 and 3 of a 4-panel mesh; skipped panels'
        barriers and trackers ride along to their shifted positions."""
        barriers = torch.tensor(
            [[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64
        )
        trackers = torch.tensor([True, True, True, True, False])
        idxs = torch.tensor([1, 3])
        error_ratios = torch.tensor([0.5, 2.0])  # panel 1 passes, panel 3 splits

        _, _, _, _, barriers_new, trackers_new, mesh_idx, _, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        expected = torch.tensor(
            [[0.0], [0.25], [0.5], [0.75], [0.875], [1.0]], dtype=torch.float64
        )
        assert torch.allclose(barriers_new, expected)
        # Panel 1 accepted -> barrier 1 cleared; skipped panels 0 and 2 keep
        # their True tracker at shifted positions; new midpoint (idx 4) is True.
        assert trackers_new.tolist() == [True, False, True, True, True, False]
        _assert_transfer(mesh_idx, barriers, barriers_new)

    def test_extra_split_positions_in_any_order(self):
        """The recorded subintervals to split again come in the record's row
        order, not ascending; the update sorts them with the batch's, so the
        mesh, trackers and idxs_transfer are those of ascending positions."""
        barriers = torch.linspace(0, 1, 6, dtype=torch.float64)[:, None]
        trackers = torch.tensor([False, True, False, False, False, False])
        outputs = [
            self.solver._adaptively_increase_mesh(
                None,
                torch.tensor([False]),
                None,
                None,
                barriers,
                torch.tensor([1]),
                trackers.clone(),
                extra_split_idxs=torch.tensor(extras),
            )
            for extras in ([4, 0, 2], [0, 2, 4])
        ]
        for slot in (4, 5, 6):
            assert torch.equal(outputs[0][slot], outputs[1][slot])
        assert len(outputs[0][4]) == 10
        _assert_transfer(outputs[0][6], barriers, outputs[0][4])

    def test_split_first_panel(self):
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 0.5])  # only the first panel splits

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        expected = torch.tensor([[0.0], [0.25], [0.5], [1.0]], dtype=torch.float64)
        assert torch.allclose(barriers_new, expected)

    def test_split_last_panel(self):
        """Splitting the final panel exercises the in-bounds idxs_new+1 read."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([0.5, 2.0])  # only the last panel splits

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
        )
        expected = torch.tensor([[0.0], [0.5], [0.75], [1.0]], dtype=torch.float64)
        assert torch.allclose(barriers_new, expected)

    def test_error_ratio_exactly_one_splits(self):
        """error_ratio == 1.0 is on the reject side of the < 1.0 accept rule."""
        barriers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([1.0])

        _, _, _, _, barriers_new, _, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None, _keep_mask(error_ratios), None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == 3  # split
        assert len(er_kept) == 0
        assert torch.allclose(barriers_new[1], torch.tensor([0.5], dtype=torch.float64))

    def test_all_pass_with_split_counts(self):
        """N_t_add == 0 path with split_counts: mesh and counts unchanged."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([0.5, 0.3])
        split_counts = torch.tensor([0, 1, 2])

        _, _, _, _, barriers_new, _, _, _, split_counts_new = (
            self.solver._adaptively_increase_mesh(
                None,
                _keep_mask(error_ratios),
                None,
                None,
                barriers,
                idxs,
                trackers,
                split_counts=split_counts,
                max_adaptive_splits=5,
            )
        )
        assert len(barriers_new) == 3
        assert split_counts_new.tolist() == [0, 1, 2]


class TestFourPieces:
    """With ``error_ratios`` (``split_rule="accelerate"``), a failing piece of a
    split subinterval (split count >= 1) whose error ratio is at least
    2**error_exponent (8 for the unit-test solver's bosh3) is split in four: two halvings in one update, with one read of the device."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def _update(self, ratios, split_counts, mesh=None, cap=None, **kwargs):
        """The update of a batch holding every subinterval of ``mesh``."""
        n = len(ratios)
        if mesh is None:
            mesh = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:, None]
        return self.solver._adaptively_increase_mesh(
            None,
            _keep_mask(ratios),
            None,
            None,
            mesh,
            torch.arange(n),
            torch.arange(len(mesh)) < n,
            split_counts=split_counts,
            max_adaptive_splits=cap,
            **kwargs,
        )

    def test_equals_two_successive_halvings(self):
        """Subintervals 1 and 4 (split count >= 1, ratio >= 2**k) are split in
        four, 2 (ratio below 2**k) and 3 (split count 0) are halved, 0 is
        accepted: the mesh, trackers and split counts are bit for bit those of
        halving every failing subinterval and then both halves of 1 and 4, and
        idxs_transfer is the composition of the two transfers."""
        four = 2.0**self.solver.method.error_exponent
        ratios = torch.tensor(
            [0.5, 10 * four, 0.9 * four, 10 * four, four], dtype=torch.float64
        )
        counts = torch.tensor([0, 1, 1, 0, 2, 0])
        _, _, _, _, mesh, trackers, transfer, keep_idx, new_counts = self._update(
            ratios, counts, error_ratios=ratios
        )

        _, _, _, _, mesh1, trackers1, transfer1, _, counts1 = self._update(
            ratios, counts
        )
        halves = torch.tensor(
            [transfer1[1], transfer1[1] + 1, transfer1[4], transfer1[4] + 1]
        )
        _, _, _, _, mesh2, trackers2, transfer2, _, counts2 = (
            self.solver._adaptively_increase_mesh(
                None,
                torch.zeros(4, dtype=torch.bool),
                None,
                None,
                mesh1,
                halves,
                trackers1,
                split_counts=counts1,
            )
        )
        assert keep_idx.tolist() == [0]
        assert len(mesh) == 6 + 3 + 1 + 1 + 3
        assert torch.equal(mesh, mesh2)
        assert torch.equal(trackers, trackers2)
        assert torch.equal(new_counts, counts2)
        assert torch.equal(transfer, transfer2[transfer1])
        _assert_columns_increasing(mesh)
        _assert_transfer(
            transfer, torch.linspace(0, 1, 6, dtype=torch.float64)[:, None], mesh
        )
        # The pieces of a four-way split get the parent's count + 2.
        assert new_counts.tolist() == [0, 3, 3, 3, 3, 2, 2, 1, 1, 4, 4, 4, 4, 0]
        assert trackers.tolist() == [False] + [True] * 12 + [False]

    def test_without_error_ratios_every_split_is_a_halving(self):
        """``error_ratios=None`` (the default rule, the re-check, Step 7, the
        state's mesh) halves whatever the ratio."""
        ratios = torch.tensor([1e8, 1e8], dtype=torch.float64)
        mesh = self._update(ratios, torch.tensor([3, 3, 0]))[4]
        assert len(mesh) == 5

    @pytest.mark.parametrize(("cap", "n_points"), [(2, 4), (3, 8), (None, 8)])
    def test_split_cap(self, cap, n_points):
        """A subinterval one halving below the cap is halved, not split in
        four, so no split count exceeds the cap."""
        ratios = torch.tensor([1e5, 1e5], dtype=torch.float64)
        mesh, *_, new_counts = self._update(
            ratios, torch.tensor([1, 1, 0]), cap=cap, error_ratios=ratios
        )[4:]
        assert len(mesh) - 1 == n_points
        assert cap is None or int(new_counts.max()) <= cap

    def test_extra_split_positions_are_halved(self):
        """Recorded subintervals that failed the re-check are halved in the
        same update as the batch's four-way split."""
        mesh = torch.linspace(0, 1, 5, dtype=torch.float64)[:, None]
        ratios = torch.tensor([1e5], dtype=torch.float64)
        _, _, _, _, mesh_new, trackers, transfer, _, counts = (
            self.solver._adaptively_increase_mesh(
                None,
                _keep_mask(ratios),
                None,
                None,
                mesh,
                torch.tensor([2]),
                torch.tensor([False, False, True, False, False]),
                split_counts=torch.tensor([1, 1, 1, 1, 0]),
                extra_split_idxs=torch.tensor([3, 0]),
                error_ratios=ratios,
            )
        )
        assert counts.tolist() == [2, 2, 1, 3, 3, 3, 3, 2, 2, 0]
        assert trackers.tolist() == [True] * 2 + [False] + [True] * 6 + [False]
        _assert_columns_increasing(mesh_new)
        _assert_transfer(transfer, mesh, mesh_new)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
    def test_too_narrow_to_quarter_is_halved(self, dtype):
        """A subinterval whose ends are two floating-point numbers apart has a
        midpoint between them, but its quarter points round onto it or an end:
        it can be halved but not split in four, so it is halved."""
        one = torch.tensor(1.0, dtype=dtype)
        up = torch.tensor(2.0, dtype=dtype)
        two_ulps = torch.nextafter(torch.nextafter(one, up), up)
        mesh = torch.stack([torch.zeros((), dtype=dtype), one, two_ulps])[:, None]
        idxs = torch.tensor([0, 1])
        assert self.solver._splittable(mesh, idxs, None, None).tolist() == [True, True]
        assert self.solver._splittable(mesh, idxs, None, None, halvings=2).tolist() == [
            True,
            False,
        ]
        ratios = torch.tensor([1e5, 1e5], dtype=dtype)
        mesh_new = self._update(
            ratios, torch.tensor([1, 1, 0]), mesh=mesh, error_ratios=ratios
        )[4]
        assert len(mesh_new) == 3 + 3 + 1
        _assert_columns_increasing(mesh_new)
