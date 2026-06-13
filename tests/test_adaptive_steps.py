"""Unit tests for _adaptively_increase_mesh: core adaptive refinement logic."""

from __future__ import annotations

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


def _rebuilt_mesh_indices(barriers):
    """The mesh_indices dict the method should return for ``barriers``: each
    full barrier coordinate mapped to its position in the ordered mesh."""
    return {tuple(b.tolist()): i for i, b in enumerate(barriers)}


class TestAdaptivelyAddSteps:
    """Tests for AdaptiveQuadrature._adaptively_increase_mesh.

    The method returns a 9-tuple:
    (method_output, y, tracked, nodes, mesh_new, trackers_new, mesh_indices,
     error_ratios_kept, split_counts_new).
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
                mo, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == len(barriers)
        assert not torch.any(trackers_new[:2])
        assert len(er_kept) == 2
        assert mo_out.mesh_quadratures.shape[0] == 2
        # mesh_indices maps every barrier coordinate to its mesh position.
        assert mesh_idx == {
            tuple(b.tolist()): i for i, b in enumerate(barriers_new)
        }

    def test_all_fail(self):
        """All error_ratios >= 1: midpoints inserted, error_ratios_kept empty."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 1.5])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _trackers_new, mesh_idx, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        # 2 midpoints added: len goes from 3 to 5
        assert len(barriers_new) == 5
        assert len(er_kept) == 0
        assert mo_out.mesh_quadratures.shape[0] == 0
        # mesh_indices reflects the inserted midpoints and the bumped positions.
        assert mesh_idx == {
            tuple(b.tolist()): i for i, b in enumerate(barriers_new)
        }

    def test_mixed_pass_fail(self):
        """First passes, second fails: 1 midpoint added."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([0.5, 2.0])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _trackers_new, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, error_ratios, None, None, barriers, idxs, trackers
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
                None, error_ratios, None, None, barriers, idxs, trackers
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
            None, error_ratios, None, None, barriers, idxs, trackers
        )
        assert torch.allclose(barriers_new[1], torch.tensor([0.5], dtype=torch.float64))

    def test_barrier_ordering(self):
        """After multiple fails, barriers remain sorted ascending."""
        barriers = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([2.0, 2.0, 2.0])

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, error_ratios, None, None, barriers, idxs, trackers
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
                None, error_ratios, None, None, barriers, idxs, trackers
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
            mo, error_ratios, None, None, barriers, idxs, trackers
        )
        assert mo_out.mesh_quadratures.shape[0] == 2
        assert mo_out.h.shape[0] == 2
        assert len(er_kept) == 2

    def test_single_step_passes(self):
        """Single step that passes: barriers unchanged."""
        barriers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, False])
        idxs = torch.tensor([0])
        error_ratios = torch.tensor([0.5])

        _, _, _, _, barriers_new, trackers_new, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == 2
        assert trackers_new[0] == False  # noqa: E712
        assert len(er_kept) == 1

    def test_explicit_masks_override_scalar_rule(self):
        """When keep_mask/remove_mask are given they drive accept/reject,
        independent of the scalar error_ratios values."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        # Scalar rule would keep both (< 1), but the explicit masks reject step 0.
        error_ratios = torch.tensor([0.5, 0.5])
        keep_mask = torch.tensor([False, True])
        remove_mask = torch.tensor([True, False])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _, _, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo,
                error_ratios,
                None,
                None,
                barriers,
                idxs,
                trackers,
                keep_mask=keep_mask,
                remove_mask=remove_mask,
            )
        )
        # One step split -> one midpoint added (3 -> 4); one step kept.
        assert len(barriers_new) == 4
        assert len(er_kept) == 1
        assert mo_out.mesh_quadratures.shape[0] == 1


class TestVectorMeshOrdering:
    """Vector mesh (T > 1): midpoints, per-component ordering, mesh_indices.

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
            None, error_ratios, None, None, barriers, idxs, trackers
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
            None, error_ratios, None, None, barriers, idxs, trackers
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
            None, error_ratios, None, None, barriers, idxs, trackers
        )
        expected = torch.tensor(
            [[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 1.5], [1.0, 2.0]],
            dtype=torch.float64,
        )
        assert torch.allclose(barriers_new, expected)
        _assert_columns_increasing(barriers_new)

    def test_mesh_indices_returned_T2(self):
        """The returned mesh_indices keys every multi-element barrier vector.

        Extends the T=1 mesh_indices check in TestAdaptivelyAddSteps to vector
        meshes; ``len == len(barriers_new)`` guards against coordinate
        collisions among the multi-element keys.
        """
        barriers = torch.tensor(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]], dtype=torch.float64
        )
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 2.0])

        _, _, _, _, barriers_new, _, mesh_idx, _, _ = (
            self.solver._adaptively_increase_mesh(
                None, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        assert mesh_idx == _rebuilt_mesh_indices(barriers_new)
        assert len(mesh_idx) == len(barriers_new)

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
            None, error_ratios, None, None, barriers, idxs, trackers
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
                None, error_ratios, None, None, barriers, idxs, trackers
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
                None, error_ratios, None, None, barriers, idxs, trackers
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
                None, error_ratios, None, None, barriers, idxs, trackers
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
            None, error_ratios, None, None, barriers, idxs, trackers
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
            error_ratios,
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
            error_ratios,
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
                error_ratios,
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
        # ratio 2.0, so its ratio is in er_kept.
        assert len(barriers_new) == 5
        assert len(er_kept) == 2
        assert 2.0 in er_kept.tolist()

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
                error_ratios,
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
            None, error_ratios, y, None, barriers, idxs, trackers
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
            None, error_ratios, None, nodes, barriers, idxs, trackers
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
            error_ratios,
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
            mo, error_ratios, None, None, barriers, idxs, trackers
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
                None, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        expected = torch.tensor(
            [[0.0], [0.25], [0.5], [0.75], [0.875], [1.0]], dtype=torch.float64
        )
        assert torch.allclose(barriers_new, expected)
        # Panel 1 accepted -> barrier 1 cleared; skipped panels 0 and 2 keep
        # their True tracker at shifted positions; new midpoint (idx 4) is True.
        assert trackers_new.tolist() == [True, False, True, True, True, False]
        assert mesh_idx == _rebuilt_mesh_indices(barriers_new)

    def test_split_first_panel(self):
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 0.5])  # only the first panel splits

        _, _, _, _, barriers_new, _, _, _, _ = self.solver._adaptively_increase_mesh(
            None, error_ratios, None, None, barriers, idxs, trackers
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
            None, error_ratios, None, None, barriers, idxs, trackers
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
                None, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == 3  # split
        assert len(er_kept) == 0
        assert torch.allclose(
            barriers_new[1], torch.tensor([0.5], dtype=torch.float64)
        )

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
                error_ratios,
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
