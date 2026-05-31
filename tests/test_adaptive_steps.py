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


class TestAdaptivelyAddSteps:
    """Tests for AdaptiveQuadrature._adaptively_increase_mesh."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_all_pass(self):
        """All error_ratios < 1: barriers unchanged, all trackers False."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([0.5, 0.3])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, trackers_new, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == len(barriers)
        assert not torch.any(trackers_new[:2])
        assert len(er_kept) == 2
        assert mo_out.mesh_quadratures.shape[0] == 2

    def test_all_fail(self):
        """All error_ratios >= 1: midpoints inserted, error_ratios_kept empty."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([2.0, 1.5])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _trackers_new, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                mo, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        # 2 midpoints added: len goes from 3 to 5
        assert len(barriers_new) == 5
        assert len(er_kept) == 0
        assert mo_out.mesh_quadratures.shape[0] == 0

    def test_mixed_pass_fail(self):
        """First passes, second fails: 1 midpoint added."""
        barriers = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, False])
        idxs = torch.tensor([0, 1])
        error_ratios = torch.tensor([0.5, 2.0])
        mo = _make_method_output(2)

        mo_out, _, _, _, barriers_new, _trackers_new, er_kept, _ = (
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

        mo_out, y_out, _tracked, t_out, barriers_new, _, _, _ = (
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

        _, _, _, _, barriers_new, _, _, _ = self.solver._adaptively_increase_mesh(
            None, error_ratios, None, None, barriers, idxs, trackers
        )
        assert torch.allclose(barriers_new[1], torch.tensor([0.5], dtype=torch.float64))

    def test_barrier_ordering(self):
        """After multiple fails, barriers remain sorted ascending."""
        barriers = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
        trackers = torch.tensor([True, True, True, False])
        idxs = torch.tensor([0, 1, 2])
        error_ratios = torch.tensor([2.0, 2.0, 2.0])

        _, _, _, _, barriers_new, _, _, _ = self.solver._adaptively_increase_mesh(
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

        _, _, _, _, _barriers_new, trackers_new, _, _ = (
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

        mo_out, _, _, _, _, _, er_kept, _ = self.solver._adaptively_increase_mesh(
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

        _, _, _, _, barriers_new, trackers_new, er_kept, _ = (
            self.solver._adaptively_increase_mesh(
                None, error_ratios, None, None, barriers, idxs, trackers
            )
        )
        assert len(barriers_new) == 2
        assert trackers_new[0] == False  # noqa: E712
        assert len(er_kept) == 1
