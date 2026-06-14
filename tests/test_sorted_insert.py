"""Unit tests for _mesh_order, _merge_positions and _insert_sorted_results."""

from __future__ import annotations

import torch
from _helpers import make_solver_for_unit_test


class TestMeshOrder:
    """Tests for _mesh_order: map a panel's left barrier to its mesh position."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_1d(self):
        """1-D barriers resolve to their mesh positions."""
        mesh = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
        mi = self.solver._get_mesh_indices(mesh)
        barriers = torch.tensor([[0.5], [0.0], [1.0]], dtype=torch.float64)
        order = self.solver._mesh_order(barriers, mi)
        assert order.tolist() == [1, 0, 2]

    def test_multi_dim_keys_on_full_vector(self):
        """Multi-D barriers are matched on the whole vector, not coordinate 0.

        Two barriers share coordinate 0 but differ in coordinate 1; the order
        must come from the full-vector mesh position.
        """
        mesh = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 2.0]], dtype=torch.float64
        )
        mi = self.solver._get_mesh_indices(mesh)
        barriers = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        order = self.solver._mesh_order(barriers, mi)
        assert order.tolist() == [1, 0]


class TestMergePositions:
    """Tests for _merge_positions: combined-array placement from mesh orders."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_interleave(self):
        """Interleaving record [0,2,4] and new [1,3] gives [0,1,2,3,4]."""
        idxs_keep, idxs_input = self.solver._merge_positions(
            torch.tensor([0, 2, 4]), torch.tensor([1, 3])
        )
        assert idxs_keep.tolist() == [0, 2, 4]
        assert idxs_input.tolist() == [1, 3]

    def test_insert_at_start(self):
        idxs_keep, idxs_input = self.solver._merge_positions(
            torch.tensor([1, 2]), torch.tensor([0])
        )
        assert idxs_input.tolist() == [0]
        assert idxs_keep.tolist() == [1, 2]

    def test_insert_at_end(self):
        idxs_keep, idxs_input = self.solver._merge_positions(
            torch.tensor([0, 1]), torch.tensor([2])
        )
        assert idxs_input.tolist() == [2]
        assert idxs_keep.tolist() == [0, 1]

    def test_many_into_one(self):
        idxs_keep, idxs_input = self.solver._merge_positions(
            torch.tensor([2]), torch.tensor([0, 1, 3])
        )
        assert idxs_keep.tolist() == [2]
        assert idxs_input.tolist() == [0, 1, 3]


class TestInsertSortedResults:
    """Tests for _insert_sorted_results: merge tensors at pre-computed positions."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_1d(self):
        """Merging 1D tensors produces correctly ordered output."""
        record = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
        result = torch.tensor([2.0, 4.0], dtype=torch.float64)
        idxs_keep, idxs_input = self.solver._merge_positions(record, result)

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        assert torch.allclose(merged, expected)

    def test_2d(self):
        """Merging 2D tensors [N, D] preserves correct ordering."""
        record = torch.tensor(
            [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=torch.float64
        )
        result = torch.tensor([[2.0, 20.0], [4.0, 40.0]], dtype=torch.float64)
        idxs_keep, idxs_input = self.solver._merge_positions(
            record[:, 0], result[:, 0]
        )

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        assert merged.shape == (5, 2)
        expected_col0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        assert torch.allclose(merged[:, 0], expected_col0)

    def test_3d(self):
        """Merging 3D tensors [N, C, T] preserves correct ordering."""
        record = torch.tensor(
            [[[1.0, 1.5]], [[3.0, 3.5]], [[5.0, 5.5]]], dtype=torch.float64
        )  # [3, 1, 2]
        result = torch.tensor(
            [[[2.0, 2.5]], [[4.0, 4.5]]], dtype=torch.float64
        )  # [2, 1, 2]
        idxs_keep, idxs_input = self.solver._merge_positions(
            record[:, 0, 0], result[:, 0, 0]
        )

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        assert merged.shape == (5, 1, 2)
        expected_first = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        assert torch.allclose(merged[:, 0, 0], expected_first)

    def test_values_preserved(self):
        """All original values from both record and result appear in merged output."""
        record = torch.tensor([10.0, 30.0, 50.0], dtype=torch.float64)
        result = torch.tensor([20.0, 40.0], dtype=torch.float64)
        idxs_keep, idxs_input = self.solver._merge_positions(record, result)

        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )

        for val in record:
            assert val in merged, f"Record value {val} missing from merged"
        for val in result:
            assert val in merged, f"Result value {val} missing from merged"
