"""Unit tests for _record_results, _sort_record and _flatten_output."""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import make_solver_for_unit_test, needs_cuda

from padaquad import adaptive_quadrature
from padaquad.methods import UniformMethodClass, _Tableau
from padaquad.results import IntegrationResult


def _batch(mesh_index, C=4, D=1, T=1):
    """An IntegrationResult of one batch whose subinterval with left mesh index
    i holds the value i in every per-subinterval field (nodes, h, y,
    contributions, errors, ratios), so an order can be read off any field."""
    tag = torch.as_tensor(mesh_index, dtype=torch.float64)
    N = len(tag)
    return IntegrationResult(
        integral=torch.tensor([0.5], dtype=torch.float64),
        integral_error=torch.tensor([0.01], dtype=torch.float64),
        nodes=tag.view(N, 1, 1).expand(N, C, T).clone(),
        h=tag.view(N, 1).expand(N, T).clone(),
        y=tag.view(N, 1, 1).expand(N, C, D).clone(),
        mesh_quadratures=tag.view(N, 1).expand(N, D).clone(),
        mesh_quadrature_errors=tag.view(N, 1).expand(N, D).clone(),
        error_ratios=tag.clone(),
        loss=torch.tensor([1.0], dtype=torch.float64),
        mesh_init=torch.zeros(T, dtype=torch.float64),
        mesh_final=torch.ones(T, dtype=torch.float64),
        gradient_taken=False,
        y0=torch.zeros(D, dtype=torch.float64),
    )


def _record(solver, *mesh_indices, take_gradient=False, **kwargs):
    """Record one batch per list of left mesh indices, in the given order."""
    record, chunks = {}, {}
    for mesh_index in mesh_indices:
        record = solver._record_results(
            record,
            chunks,
            take_gradient,
            _batch(mesh_index, **kwargs),
            torch.tensor(mesh_index),
        )
    return record, chunks


# ---------------------------------------------------------------------------
# _record_results
# ---------------------------------------------------------------------------


class TestRecordResults:
    """Tests for _record_results: append batches in the order they are accepted."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    def test_first_batch_initializes(self):
        """The first batch creates the sums, the loop's fields with the mesh
        indices, and one chunk per other field."""
        record, chunks = _record(self.solver, [0, 1])
        assert set(record) == {
            "integral",
            "integral_error",
            "loss",
            "mesh_index",
            "mesh_quadratures",
            "mesh_quadrature_errors",
        }
        assert set(chunks) == {"nodes", "h", "y", "error_ratios"}
        assert all(len(parts) == 1 for parts in chunks.values())
        assert record["mesh_index"].tolist() == [0, 1]
        assert torch.equal(record["integral"], torch.tensor([0.5], dtype=torch.float64))

    def test_take_gradient_detaches(self):
        """With take_gradient=True every recorded tensor is detached."""
        results = _batch([0])
        results.integral = results.integral.clone().requires_grad_(True)
        results.loss = results.loss.clone().requires_grad_(True)
        results.y = results.y.clone().requires_grad_(True)
        record, chunks = {}, {}
        record = self.solver._record_results(
            record, chunks, True, results, torch.tensor([0])
        )
        assert not record["integral"].requires_grad
        assert not record["loss"].requires_grad
        assert not chunks["y"][0].requires_grad

    def test_later_batches_are_appended(self):
        """Rows stay in the order they were accepted: the loop's fields are
        one tensor, the other fields one chunk per batch, nothing merged."""
        record, chunks = _record(self.solver, [2, 0], [3, 1])
        assert record["mesh_index"].tolist() == [2, 0, 3, 1]
        assert record["mesh_quadratures"][:, 0].tolist() == [2.0, 0.0, 3.0, 1.0]
        assert [part[:, 0, 0].tolist() for part in chunks["y"]] == [[2, 0], [3, 1]]

    def test_sums_accumulated(self):
        """The integral, its error and the loss are summed across batches."""
        record, _ = _record(self.solver, [0], [1], [2])
        assert torch.equal(record["integral"], torch.tensor([1.5], dtype=torch.float64))
        assert torch.equal(record["loss"], torch.tensor([3.0], dtype=torch.float64))

    def test_tracked_variables_are_chunk_keys(self):
        """Each tracked variable is recorded under its own key, as a chunk."""
        record, chunks = {}, {}
        for mesh_index in ([1], [0]):
            batch = _batch(mesh_index)
            batch.tracked_variables = (batch.nodes.clone(), 2 * batch.nodes)
            record = self.solver._record_results(
                record, chunks, False, batch, torch.tensor(mesh_index)
            )
        assert torch.equal(
            torch.cat(chunks["tracked_variables_0"]), torch.cat(chunks["nodes"])
        )
        assert torch.equal(
            torch.cat(chunks["tracked_variables_1"]), 2 * torch.cat(chunks["nodes"])
        )

    def test_changed_tracked_variables_raise(self):
        """A batch whose tracked variables differ from the record's raises."""
        record, chunks = _record(self.solver, [0])
        batch = _batch([1])
        batch.tracked_variables = (batch.nodes.clone(),)
        with pytest.raises(ValueError, match="different set of tracked variables"):
            self.solver._record_results(record, chunks, False, batch, torch.tensor([1]))

    def test_a_view_of_a_larger_array_is_copied(self):
        """A chunk that is a view of a larger array (the accepted rows moved to
        the front of a pass's values) is copied, so the array can be freed; a
        tensor that owns exactly its memory is kept without a copy."""
        record, chunks = _record(self.solver, [0])
        batch = _batch([1, 2])
        pass_values = torch.zeros(5, 4, 1, dtype=torch.float64)
        batch.y = pass_values[:2]  # the kept rows of a pass of 5
        whole = batch.nodes
        self.solver._record_results(record, chunks, False, batch, torch.tensor([1, 2]))
        y_chunk = chunks["y"][-1]
        assert y_chunk.data_ptr() != pass_values.data_ptr()
        assert y_chunk.untyped_storage().nbytes() == y_chunk.nbytes
        assert torch.equal(y_chunk, pass_values[:2])
        assert chunks["nodes"][-1] is whole


# ---------------------------------------------------------------------------
# _sort_record
# ---------------------------------------------------------------------------


class TestSortRecord:
    """Tests for _sort_record: put the record of a finished integration in mesh order."""

    def setup_method(self):
        self.solver = make_solver_for_unit_test()

    @pytest.mark.parametrize("T", [1, 2])
    def test_scrambled_to_mesh_order(self, T):
        """Rows recorded out of mesh order (two batches) come out in mesh
        order in every field, by their integer mesh index alone (any T); the
        sums are untouched, the chunks emptied and mesh_index removed."""
        record, chunks = _record(self.solver, [2, 0], [3, 1], T=T)
        record = self.solver._sort_record(record, chunks, "cpu")
        expected = torch.arange(4, dtype=torch.float64)
        for key in ("nodes", "h", "y", "mesh_quadratures", "mesh_quadrature_errors"):
            assert torch.equal(record[key].reshape(4, -1)[:, 0], expected), key
        assert torch.equal(record["error_ratios"], expected)
        assert torch.equal(record["integral"], torch.tensor([1.0], dtype=torch.float64))
        assert torch.equal(record["loss"], torch.tensor([2.0], dtype=torch.float64))
        assert "mesh_index" not in record
        assert all(len(parts) == 0 for parts in chunks.values())

    def test_one_batch_in_order_is_not_copied(self):
        """A record of one batch in mesh order keeps its tensors: no field is
        copied, the values included (the common case of one pass)."""
        record, chunks = _record(self.solver, [0, 1, 2])
        before = {key: parts[0] for key, parts in chunks.items()}
        before["mesh_quadratures"] = record["mesh_quadratures"]
        record = self.solver._sort_record(record, chunks, "cpu")
        assert all(record[key] is value for key, value in before.items())

    def test_batches_in_order_are_joined(self):
        """Batches recorded from left to right are joined in their order."""
        record, chunks = _record(self.solver, [0, 1], [2], [3, 4])
        record = self.solver._sort_record(record, chunks, "cpu")
        assert record["y"][:, 0, 0].tolist() == [0, 1, 2, 3, 4]

    @needs_cuda
    def test_chunk_fields_end_on_the_given_device(self):
        """The fields of chunks are moved to the given device; the loop's
        fields stay on the solver's."""
        solver = adaptive_quadrature("uniform", "bosh3", device="cuda")
        record, chunks = {}, {}
        for mesh_index in ([1], [0]):
            batch = _batch(mesh_index)
            for key in (
                "nodes",
                "h",
                "y",
                "mesh_quadratures",
                "mesh_quadrature_errors",
            ):
                setattr(batch, key, getattr(batch, key).cuda())
            batch.error_ratios = batch.error_ratios.cuda()
            record = solver._record_results(
                record, chunks, False, batch, torch.tensor(mesh_index, device="cuda")
            )
        record = solver._sort_record(record, chunks, "cpu")
        assert record["y"].device.type == "cpu"
        assert record["nodes"].device.type == "cpu"
        assert record["mesh_quadratures"].device.type == "cuda"
        assert record["y"][:, 0, 0].tolist() == [0, 1]

    @pytest.mark.parametrize(
        "mesh_index", [[[0, 1], [1]], [[0, 3], [1]]], ids=["twice", "gap"]
    )
    def test_record_that_does_not_cover_the_mesh_once_raises(self, mesh_index):
        """With runtime_checks, a subinterval recorded twice or a missing one
        (the record must cover the final mesh once) raises."""
        record, chunks = _record(self.solver, *mesh_index)
        with pytest.raises(AssertionError, match="cover the final mesh once"):
            self.solver._sort_record(record, chunks, "cpu")

    def test_gradient_passes_through_the_ordering(self):
        """With take_gradient=False the record keeps the autograd graph: the
        gradient of the ordered values reaches each recorded value."""
        record, chunks = {}, {}
        leaves = []
        for mesh_index in ([2, 0], [1]):
            batch = _batch(mesh_index)
            batch.y = batch.y.clone().requires_grad_()
            leaves.append(batch.y)
            record = self.solver._record_results(
                record, chunks, False, batch, torch.tensor(mesh_index)
            )
        record = self.solver._sort_record(record, chunks, "cpu")
        weights = torch.arange(1.0, 4.0, dtype=torch.float64).view(3, 1, 1)
        (record["y"] * weights).sum().backward()
        assert leaves[0].grad[:, 0, 0].tolist() == [3.0, 1.0]
        assert leaves[1].grad[:, 0, 0].tolist() == [2.0]


def _cancelling_peak(t):
    """A smooth part cancelled by a narrow peak: the re-check and a dropped
    error_integral_reference remove accepted subintervals."""
    return (
        1.0
        + 0.5 * torch.sin(4.0 * math.pi * t)
        - 100.0 / math.sqrt(math.pi) * torch.exp(-1e4 * (t - 0.37) ** 2)
    )


def _tracked_peak(t):
    """The cancelling peak with two tracked variables."""
    return _cancelling_peak(t), (2 * t, torch.cos(t).repeat(1, 3))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"take_gradient": True, "max_batch": 40},
        {"error_integral_reference": 10.0},
        {"use_absolute_error_ratio": False},
        {"f": _tracked_peak},
        {"f": _tracked_peak, "trade_speed_for_memory": True},
        {"record_evaluations": False},
        {"max_adaptive_splits": 3},
        {"method": "gk21"},
    ],
    ids=[
        "default",
        "take_gradient",
        "step7",
        "cumulative",
        "tracked",
        "trade",
        "no_record",
        "split_cap",
        "gk21",
    ],
)
def test_every_record_path_returns_mesh_order(options):
    """Through every path of the record (removals by the re-check and by a
    dropped reference, cumulative mode, tracked variables, values moved early or
    not recorded, the split cap, batches of take_gradient=True, a method
    without end nodes), the returned subintervals are in mesh order and tile
    the interval, and runtime_checks=False returns the same numbers bit for
    bit."""
    options = dict(options)
    f = options.pop("f", _cancelling_peak)
    ctor = {
        key: options.pop(key)
        for key in ("use_absolute_error_ratio", "method")
        if key in options
    }
    out = {}
    for checks in (True, False):
        solver = adaptive_quadrature(
            "uniform",
            **({"method": "bosh3"} | ctor),
            atol=1e-9,
            rtol=1e-6,
            device="cpu",
            runtime_checks=checks,
        )
        torch.manual_seed(0)
        out[checks] = solver.integrate(
            f,
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
            mesh_final=torch.tensor([1.0], dtype=torch.float64),
            **({"max_batch": 64, "take_gradient": False} | options),
        )
    result, state = out[True]
    nodes = result.nodes[:, 0]
    assert torch.all(nodes[1:] > nodes[:-1])
    assert len(state.mesh) > 2
    assert abs(float(result.h.sum()) - 1.0) < 1e-12
    other, other_state = out[False]
    for key in ("integral", "integral_error", "nodes", "h", "mesh_quadratures", "loss"):
        assert torch.equal(getattr(result, key), getattr(other, key)), key
    if result.y is not None:
        assert torch.equal(result.y, other.y)
    assert torch.equal(state.mesh, other_state.mesh)


@needs_cuda
@pytest.mark.parametrize("stage", ["removal", "final_ordering"])
def test_fields_are_copied_one_at_a_time(stage):
    """A removal by the re-check (each field's chunks joined) and the final
    ordering (each field written in mesh order) free the copy of one field
    before the next field is copied: with two tracked variables of F bytes
    each, neighbors in the record, the GPU holds at most one field more than
    the record (a name left holding the previous field's joined rows, or its
    last chunk, made it two)."""
    solver = adaptive_quadrature("uniform", "bosh3", device="cuda")
    record, chunks = {}, {}
    for mesh_index in ([3], [2, 0, 1]):  # out of mesh order, the last chunk large
        batch = _batch(mesh_index)
        for key in ("nodes", "h", "y", "mesh_quadratures", "mesh_quadrature_errors"):
            setattr(batch, key, getattr(batch, key).cuda())
        batch.error_ratios = batch.error_ratios.cuda()
        tracked = torch.ones(len(mesh_index), 4, 2**17, dtype=torch.float64).cuda()
        batch.tracked_variables = (tracked, 2 * tracked)
        record = solver._record_results(
            record, chunks, False, batch, torch.tensor(mesh_index, device="cuda")
        )
    del batch, tracked  # the record's chunks are the only names of the values
    field = 4 * 4 * 2**17 * 8
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if stage == "removal":
        zero = torch.zeros(1, dtype=torch.float64, device="cuda")
        keep = torch.tensor([True, False, True, True], device="cuda")
        solver._remove_failed_record_panels(
            record, keep, chunks, lambda r: r.integral, zero, zero + 1, "cuda"
        )
    else:
        solver._sort_record(record, chunks, "cuda")
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before < 1.3 * field


def _weighted(f, d):
    """f's one output repeated d times, each with its own weight."""
    w = torch.linspace(1.0, 1.001, d, dtype=torch.float64)
    return lambda t: f(t) * w


@needs_cuda
@pytest.mark.parametrize(
    ("method", "f", "options", "bound"),
    [
        ("gk21", _weighted(torch.sin, 4096), {"N_init_steps": 400}, 1.3),
        ("gk21", _weighted(lambda t: torch.sin(2e3 * t), 256), {}, 2.05),
        ("bosh3", _weighted(_cancelling_peak, 256), {}, 2.05),
        ("bosh3", _weighted(torch.sin, 4096), {"N_init_steps": 400}, 2.1),
    ],
    ids=["gk21_one_pass", "gk21_passes", "bosh3_removals", "bosh3_one_pass"],
)
def test_record_device_peak_at_the_end(method, f, options, bound):
    """The record's device holds at most about twice the recorded values R over
    a whole call: the final ordering's chunks and array in mesh order (2 R),
    or for one pass of a method without end nodes the values once (the join is
    a view). A CPU solver keeps its record on the GPU (result_device="cuda"),
    the code path of the host with exact allocator numbers. The join copied
    the values (2.15 R for gk21) and the reuse mesh held up to six arrays of
    one number per subinterval and output beside the values (2.5 R for
    bosh3, whose four nodes make such an array a quarter of R)."""
    solver = adaptive_quadrature(
        "uniform", method, atol=1e-9, rtol=1e-6, device="cpu", dtype=torch.float64
    )
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result, _ = solver.integrate(
        f,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([1.0], dtype=torch.float64),
        take_gradient=False,
        random_initial_mesh=False,
        result_device="cuda",
        **options,
    )
    torch.cuda.synchronize()
    values = len(result.h) * solver.C * result.y.shape[-1] * 8
    assert torch.cuda.max_memory_allocated() - before < bound * values


class TestFlattenOutput:
    """_flatten_output writes the joined values into one array."""

    @staticmethod
    def _joined(panel_array):
        """The former two-copy formula, as the reference."""
        return torch.concatenate(
            [panel_array[0], torch.flatten(panel_array[1:, 1:], 0, 1)], dim=0
        )

    @pytest.mark.parametrize("n", [1, 2, 5])
    @pytest.mark.parametrize("c", [2, 4])
    @pytest.mark.parametrize("rest", [(), (3,), (2, 3)])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.int64])
    def test_equals_concatenation(self, n, c, rest, dtype):
        panel_array = torch.arange(n * c * math.prod(rest)).reshape(n, c, *rest)
        panel_array = panel_array.to(dtype)
        solver = make_solver_for_unit_test()
        out = solver._flatten_output(panel_array)
        assert out.dtype == dtype
        assert torch.equal(out, self._joined(panel_array))

    @pytest.mark.parametrize(
        "fractions",
        [[0.2, 0.8], [0.0, 2 / 3], [1 / 3, 1.0]],
        ids=["none", "start", "end"],
    )
    def test_no_shared_node_keeps_every_row(self, fractions):
        """Without a node at both ends neighbouring subintervals share no
        node (Gauss-Kronrod has none, a Radau-type rule one), so every row is
        an evaluated point of its own: N * C rows, in order, a view of the
        array (no copy: a copy held the values twice at the end of a call)."""
        solver = make_solver_for_unit_test()
        c = torch.tensor(fractions, dtype=torch.float64)
        solver.method = UniformMethodClass(
            degree=0,
            error_exponent=2,
            tableau=_Tableau(c=c, b=torch.full_like(c, 0.5), b_error=0 * c),
        )
        panel_array = torch.arange(3 * 2 * 4, dtype=torch.float64).reshape(3, 2, 4)
        out = solver._flatten_output(panel_array)
        assert torch.equal(out, panel_array.reshape(6, 4))
        assert out.data_ptr() == panel_array.data_ptr()

    def test_gradient_equals_concatenation(self):
        """The gradient reaches every value as through the concatenation."""
        g = torch.Generator().manual_seed(0)
        values = torch.randn(4, 3, 2, generator=g, dtype=torch.float64)
        weights = torch.randn(4 * 2 + 1, 2, generator=g, dtype=torch.float64)
        solver = make_solver_for_unit_test()
        grads = []
        for flatten in (solver._flatten_output, self._joined):
            panel_array = values.clone().requires_grad_()
            (flatten(panel_array) * weights).sum().backward()
            grads.append(panel_array.grad)
        assert torch.equal(grads[0], grads[1])
