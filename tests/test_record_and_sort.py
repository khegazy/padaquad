"""Unit tests for _record_results and _sort_record."""

from __future__ import annotations

import torch
from _helpers import make_solver_for_unit_test

from padaquad.results import IntegrationResult


def _make_integral_output(t_start, t_end, N=2, C=4, D=1):
    """Create an IntegrationResult with N steps spanning [t_start, t_end]."""
    boundaries = torch.linspace(t_start, t_end, N + 1, dtype=torch.float64)
    t_left = boundaries[:-1]
    t_right = boundaries[1:]
    # Build [N, C, 1] time tensor
    c = torch.linspace(0, 1, C, dtype=torch.float64)
    t = t_left[:, None, None] + c[None, :, None] * (t_right - t_left)[:, None, None]
    h = (t_right - t_left).unsqueeze(-1)

    return IntegrationResult(
        integral=torch.tensor([0.5], dtype=torch.float64),
        integral_error=torch.tensor([0.01], dtype=torch.float64),
        nodes=t,
        h=h,
        y=torch.ones(N, C, D, dtype=torch.float64),
        mesh_quadratures=torch.ones(N, D, dtype=torch.float64) * 0.1,
        mesh_quadrature_errors=torch.ones(N, D, dtype=torch.float64) * 0.001,
        error_ratios=torch.ones(N, dtype=torch.float64) * 0.5,
        loss=torch.tensor([1.0], dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# _record_results
# ---------------------------------------------------------------------------


class TestRecordResults:
    """Tests for _record_results: accumulate batches into a sorted record."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_first_batch_initializes(self):
        """Empty record is populated with all keys from first batch."""
        record = {}
        results = _make_integral_output(0.0, 0.5)
        record = self.solver._record_results(record, False, results)
        assert "integral" in record
        assert "nodes" in record
        assert "h" in record
        assert "y" in record
        assert "mesh_quadratures" in record
        assert torch.equal(record["integral"], results.integral)

    def test_first_batch_detaches(self):
        """With take_gradient=True, first batch values are detached."""
        record = {}
        results = _make_integral_output(0.0, 0.5)
        # Make integral require grad
        results.integral = results.integral.clone().requires_grad_(True)
        results.loss = results.loss.clone().requires_grad_(True)
        record = self.solver._record_results(record, True, results)
        assert not record["integral"].requires_grad
        assert not record["loss"].requires_grad

    def test_second_batch_merges(self):
        """Two non-overlapping batches merge correctly in time order."""
        record = {}
        batch1 = _make_integral_output(0.5, 1.0, N=2)
        batch2 = _make_integral_output(0.0, 0.5, N=2)

        record = self.solver._record_results(record, False, batch1)
        record = self.solver._record_results(record, False, batch2)

        assert record["nodes"].shape[0] == 4  # 2 + 2 steps
        # Times should be sorted ascending
        assert torch.all(record["nodes"][1:, 0, 0] - record["nodes"][:-1, 0, 0] > 0)

    def test_integral_accumulated(self):
        """Integral values are summed across batches."""
        record = {}
        batch1 = _make_integral_output(0.0, 0.5, N=1)
        batch1.integral = torch.tensor([0.3], dtype=torch.float64)
        batch2 = _make_integral_output(0.5, 1.0, N=1)
        batch2.integral = torch.tensor([0.7], dtype=torch.float64)

        record = self.solver._record_results(record, False, batch1)
        record = self.solver._record_results(record, False, batch2)

        assert torch.allclose(
            record["integral"], torch.tensor([1.0], dtype=torch.float64)
        )

    def test_loss_accumulated(self):
        """Loss values are summed across batches."""
        record = {}
        batch1 = _make_integral_output(0.0, 0.5, N=1)
        batch1.loss = torch.tensor([1.0], dtype=torch.float64)
        batch2 = _make_integral_output(0.5, 1.0, N=1)
        batch2.loss = torch.tensor([2.0], dtype=torch.float64)

        record = self.solver._record_results(record, False, batch1)
        record = self.solver._record_results(record, False, batch2)

        assert torch.allclose(record["loss"], torch.tensor([3.0], dtype=torch.float64))

    def test_time_ordering_after_merge(self):
        """Batch 2 has earlier times but record remains sorted."""
        record = {}
        batch1 = _make_integral_output(0.6, 1.0, N=1)
        batch2 = _make_integral_output(0.0, 0.3, N=1)

        record = self.solver._record_results(record, False, batch1)
        record = self.solver._record_results(record, False, batch2)

        assert torch.all(record["nodes"][1:, 0, 0] - record["nodes"][:-1, 0, 0] > 0)


# ---------------------------------------------------------------------------
# _sort_record
# ---------------------------------------------------------------------------


class TestSortRecord:
    """Tests for _sort_record: sort per-step tensors by time."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def _make_record(self, t_starts):
        """Build a record dict with N steps at given start times."""
        N = len(t_starts)
        C, D = 4, 1
        t = torch.zeros(N, C, 1, dtype=torch.float64)
        for i, ts in enumerate(t_starts):
            t[i, :, 0] = torch.linspace(ts, ts + 0.1, C)
        return {
            "integral": torch.tensor([1.0], dtype=torch.float64),
            "loss": torch.tensor([2.0], dtype=torch.float64),
            "nodes": t,
            "h": torch.ones(N, 1, dtype=torch.float64) * 0.1,
            "y": torch.arange(N, dtype=torch.float64)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(N, C, D),
            "mesh_quadratures": torch.arange(N, dtype=torch.float64).unsqueeze(-1),
            "mesh_quadrature_errors": torch.ones(N, D, dtype=torch.float64) * 0.01,
            "error_ratios": torch.ones(N, dtype=torch.float64) * 0.5,
            "integral_error": torch.tensor([0.01], dtype=torch.float64),
        }

    def test_already_sorted(self):
        """Ascending times: unchanged."""
        record = self._make_record([0.1, 0.3, 0.5])
        original_t = record["nodes"].clone()
        record = self.solver._sort_record(record)
        assert torch.equal(record["nodes"], original_t)

    def test_reverse_order(self):
        """Descending times: sorted to ascending."""
        record = self._make_record([0.5, 0.3, 0.1])
        record = self.solver._sort_record(record)
        assert torch.all(record["nodes"][1:, 0, 0] - record["nodes"][:-1, 0, 0] > 0)

    def test_single_step(self):
        """Single step: trivially sorted."""
        record = self._make_record([0.5])
        record = self.solver._sort_record(record)
        assert record["nodes"].shape[0] == 1

    def test_scalars_untouched(self):
        """integral and loss are not reordered."""
        record = self._make_record([0.5, 0.3, 0.1])
        record = self.solver._sort_record(record)
        assert torch.equal(record["integral"], torch.tensor([1.0], dtype=torch.float64))
        assert torch.equal(record["loss"], torch.tensor([2.0], dtype=torch.float64))

    def test_all_keys_sorted_consistently(self):
        """All per-step keys are reordered identically."""
        record = self._make_record([0.5, 0.1, 0.3])
        record = self.solver._sort_record(record)
        # After sorting by t, mesh_quadratures should follow same order
        # Original: [0.5→idx0, 0.1→idx1, 0.3→idx2]
        # Sorted: [0.1→idx1, 0.3→idx2, 0.5→idx0]
        assert torch.allclose(
            record["mesh_quadratures"][:, 0],
            torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64),
        )
