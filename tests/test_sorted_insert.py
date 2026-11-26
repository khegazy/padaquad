"""Unit tests for _merge_positions and _insert_sorted_results."""

from __future__ import annotations

import torch
from _helpers import make_solver_for_unit_test


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

    def test_record_in_acceptance_order(self):
        """The record's rows are in the order they were accepted, not in mesh
        order, and the mesh indices have gaps (subintervals still waiting):
        each row gets its rank in the union, the position that puts the merged
        array in mesh order."""
        idxs_keep, idxs_input = self.solver._merge_positions(
            torch.tensor([9, 3, 6]), torch.tensor([4, 1])
        )
        assert idxs_keep.tolist() == [4, 1, 3]
        assert idxs_input.tolist() == [2, 0]
        merged = self.solver._insert_sorted_results(
            torch.tensor([9.0, 3.0, 6.0]),
            idxs_keep,
            torch.tensor([4.0, 1.0]),
            idxs_input,
        )
        assert merged.tolist() == [1.0, 3.0, 4.0, 6.0, 9.0]


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
        idxs_keep, idxs_input = self.solver._merge_positions(record[:, 0], result[:, 0])

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

    def test_record_is_read_not_changed(self):
        """The merged record is a new tensor and the old record is unchanged."""
        record = torch.tensor([[10.0], [30.0], [50.0]], dtype=torch.float64)
        before = record.clone()
        result = torch.tensor([[20.0], [40.0]], dtype=torch.float64)
        idxs_keep, idxs_input = self.solver._merge_positions(record[:, 0], result[:, 0])
        merged = self.solver._insert_sorted_results(
            record, idxs_keep, result, idxs_input
        )
        assert torch.equal(record, before)
        assert merged.data_ptr() != record.data_ptr()
        assert torch.equal(
            merged[:, 0],
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64),
        )
