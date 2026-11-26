"""Unit tests for SolverBase methods: _set_dtype, set_dtype_by_input, _check_variables (with the bound defaults of _setup_integral_bounds), _integral_loss, device selection."""

from __future__ import annotations

import inspect
import warnings

import pytest
import torch
from _helpers import (
    T_FINAL,
    T_INIT,
    constant_integrand,
    make_solver_for_unit_test,
    make_uniform_solver,
    needs_cuda,
)

from padaquad import adaptive_quadrature, integrand_dict
from padaquad.distributed import _resolve_device
from padaquad.quadrature.base import AdaptiveQuadrature
from padaquad.results import IntegrationResult

# ---------------------------------------------------------------------------
# _set_dtype
# ---------------------------------------------------------------------------


class TestSetDtype:
    """Tests for SolverBase._set_dtype: dtype conversion."""

    def test_float16_raises(self):
        """float16 is refused at _set_dtype time (Bug B4 fix). Its ~1e-3
        precision floor exceeds typical adaptive tolerances; the
        previous "loose-tolerance" path silently produced wrong results.
        """
        solver = make_solver_for_unit_test()
        with pytest.raises(ValueError, match=r"float16|coarse"):
            solver._set_dtype(torch.float16)

    def test_invalid_dtype_raises(self):
        """Non-float dtype raises ValueError."""
        solver = make_solver_for_unit_test()
        try:
            solver._set_dtype(torch.int32)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_noop_same_dtype(self):
        """Calling with the same dtype is a no-op."""
        solver = make_solver_for_unit_test()
        solver._set_dtype(torch.float64)
        solver._set_dtype(torch.float64)
        assert solver.dtype == torch.float64

    def test_converts_y0(self):
        """y0 dtype matches after conversion."""
        solver = make_solver_for_unit_test()
        solver._set_dtype(torch.float32)
        assert solver.init_y0.dtype == torch.float32


# ---------------------------------------------------------------------------
# set_dtype_by_input
# ---------------------------------------------------------------------------


class TestTolerances:
    """``atol`` and ``rtol`` are stored as Python floats, finite and >= 0."""

    def test_stored_as_floats(self):
        """Floats, also when given as a 0-d tensor; 0 is allowed."""
        solver = make_solver_for_unit_test(atol=torch.tensor(1e-7), rtol=0)
        assert (type(solver.atol), type(solver.rtol)) == (float, float)
        assert (solver.atol, solver.rtol) == (float(torch.tensor(1e-7)), 0.0)

    def test_dtype_round_trip_keeps_them(self):
        """A float64 call after a float32 call on the same solver has the error
        ratios of a float64 call before it (as tensors of the solver's dtype,
        the tolerances stayed rounded to float32 after the switch back)."""
        solver = make_solver_for_unit_test("gk21", atol=1e-9, rtol=1e-7)
        ratios = []
        for dtype in (torch.float64, torch.float32, torch.float64):
            torch.manual_seed(0)
            a, b = (torch.tensor([x], dtype=dtype) for x in (0.0, 30.0))
            result, _ = solver.integrate(
                torch.sin, mesh_init=a, mesh_final=b, take_gradient=False
            )
            ratios.append(result.error_ratios)
        assert torch.equal(ratios[0], ratios[2])

    @pytest.mark.parametrize(
        "bad",
        [-1e-6, float("nan"), float("inf"), [1e-5, 1e-9], torch.tensor([1e-5, 1e-9])],
        ids=["negative", "nan", "inf", "list", "tensor"],
    )
    @pytest.mark.parametrize("name", ["atol", "rtol"])
    def test_not_one_finite_number_raises(self, name, bad):
        """A negative or non-finite tolerance, or one that is not one number (a
        tolerance per output component), raises ValueError naming it."""
        with pytest.raises(ValueError, match=rf"{name} must be a finite number >= 0"):
            make_solver_for_unit_test(**{name: bad})


class TestSetDtypeByInput:
    """Tests for SolverBase.set_dtype_by_input: infer dtype from tensors."""

    def test_infers_from_t(self):
        """Dtype inferred from t when provided."""
        solver = make_solver_for_unit_test()
        solver.set_dtype_by_input(mesh=torch.tensor([0.0], dtype=torch.float32))
        assert solver.dtype == torch.float32

    def test_infers_from_t_init(self):
        """Dtype inferred from mesh_init when t is None."""
        solver = make_solver_for_unit_test()
        solver.set_dtype_by_input(
            mesh=None, mesh_init=torch.tensor([0.0], dtype=torch.float32)
        )
        assert solver.dtype == torch.float32

    def test_infers_from_t_final(self):
        """Dtype inferred from mesh_final when t and mesh_init are None."""
        solver = make_solver_for_unit_test()
        solver.set_dtype_by_input(
            mesh=None,
            mesh_init=None,
            mesh_final=torch.tensor([1.0], dtype=torch.float32),
        )
        assert solver.dtype == torch.float32

    def test_all_none_noop(self):
        """All None leaves dtype unchanged."""
        solver = make_solver_for_unit_test()
        solver.set_dtype_by_input(mesh=None, mesh_init=None, mesh_final=None)
        assert solver.dtype == torch.float64

    def test_priority_t_over_t_init(self):
        """t takes priority over mesh_init."""
        solver = make_solver_for_unit_test()
        solver.set_dtype_by_input(
            mesh=torch.tensor([0.0], dtype=torch.float32),
            mesh_init=torch.tensor([0.0], dtype=torch.float64),
        )
        assert solver.dtype == torch.float32

    def test_integrate_infers_dtype_from_bounds(self):
        """integrate() given float32 bounds on a float64 solver runs in float32.

        Goes through integrate(), not set_dtype_by_input directly: integrate()
        once passed mesh_final=mesh_init, so a float32 mesh_final was ignored.
        A call now needs both bounds (or a mesh), so both are float32 here.
        """
        solver = make_solver_for_unit_test()  # float64
        result, _ = solver.integrate(
            constant_integrand,
            mesh_init=torch.tensor([0.0], dtype=torch.float32),
            mesh_final=torch.tensor([2.0], dtype=torch.float32),
            max_batch=10000,
        )
        assert solver.dtype == torch.float32
        assert result.integral.dtype == torch.float32
        assert torch.allclose(result.integral, torch.tensor([2.0]))


# ---------------------------------------------------------------------------
# _check_variables
# ---------------------------------------------------------------------------


class TestCheckVariables:
    """Tests for the defaults of a call: SolverBase._check_variables (f, y0) and
    AdaptiveQuadrature._setup_integral_bounds (the bounds), with dtype
    conversion."""

    def test_fills_defaults(self):
        """None f and y0 are the constructor's; bounds have no default."""
        solver = make_solver_for_unit_test()
        _f, y0 = solver._check_variables(f=torch.sin)
        assert torch.equal(y0, solver.init_y0)
        with pytest.raises(ValueError, match="mesh_init and mesh_final"):
            solver._setup_integral_bounds(None, None, None)

    def test_overrides_with_args(self):
        """Explicit args override stored defaults."""
        solver = make_solver_for_unit_test()
        custom_t_init = torch.tensor([2.0], dtype=torch.float64)
        custom_t_final = torch.tensor([5.0], dtype=torch.float64)
        custom_y0 = torch.tensor([10.0], dtype=torch.float64)

        _, y0 = solver._check_variables(f=torch.sin, y0=custom_y0)
        _, mesh_init, mesh_final = solver._setup_integral_bounds(
            None, custom_t_init, custom_t_final
        )
        assert torch.equal(mesh_init, custom_t_init)
        assert torch.equal(mesh_final, custom_t_final)
        assert torch.equal(y0, custom_y0)

    def test_converts_dtype(self):
        """Tensors are converted to solver's dtype."""
        solver = make_solver_for_unit_test()
        t_init_f32 = torch.tensor([0.0], dtype=torch.float32)
        y0_f32 = torch.tensor([1.0], dtype=torch.float32)
        _, mesh_init, _ = solver._setup_integral_bounds(
            None, t_init_f32, torch.tensor([1.0], dtype=torch.float32)
        )
        _, y0 = solver._check_variables(f=torch.sin, y0=y0_f32)
        assert mesh_init.dtype == torch.float64
        assert y0.dtype == torch.float64

    def test_missing_f_raises(self):
        """No integrand here or at construction raises ValueError."""
        solver = make_solver_for_unit_test()
        assert solver.init_f is None
        with pytest.raises(ValueError, match="No integrand given"):
            solver._check_variables()

    def test_falls_back_to_constructor_f(self):
        """f=None uses the integrand given at construction."""
        solver = make_solver_for_unit_test()
        solver.init_f = torch.sin
        f, _ = solver._check_variables()
        assert f is torch.sin


# ---------------------------------------------------------------------------
# _integral_loss
# ---------------------------------------------------------------------------


def _result_with_integral(integral: torch.Tensor) -> IntegrationResult:
    """IntegrationResult whose only meaningful field is ``integral``."""
    empty = torch.zeros(0)
    return IntegrationResult(
        integral=integral,
        integral_error=torch.zeros_like(integral),
        mesh_init=torch.zeros(1),
        mesh_final=torch.ones(1),
        nodes=empty,
        h=empty,
        y=empty,
        mesh_quadratures=empty,
        mesh_quadrature_errors=empty,
        error_ratios=empty,
        gradient_taken=False,
        y0=torch.zeros_like(integral),
    )


class TestIntegralLoss:
    """Tests for SolverBase._integral_loss: default loss returns integral."""

    def test_returns_integral(self):
        """Default loss is the integral value itself."""
        solver = make_solver_for_unit_test()
        output = _result_with_integral(torch.tensor([5.0]))
        result = solver._integral_loss(output)
        assert torch.equal(result, torch.tensor([5.0]))

    def test_multidim(self):
        """Works with multi-dimensional integral."""
        solver = make_solver_for_unit_test()
        output = _result_with_integral(torch.tensor([1.0, 2.0]))
        result = solver._integral_loss(output)
        assert torch.equal(result, torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

CPU, CUDA0, CUDA1 = map(torch.device, ("cpu", "cuda:0", "cuda:1"))
needs_2_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")


class TestDeviceSelection:
    """Tests for _resolve_device and the solver's resolved ``device``."""

    @pytest.mark.parametrize(
        ("device", "index_default", "distributed", "available", "expected"),
        [
            (None, 0, False, True, CUDA0),
            (None, 0, False, False, CPU),
            ("cuda", 1, False, True, CUDA1),
            ("cuda:1", 0, False, True, CUDA1),
            (CUDA1, 0, False, True, CUDA1),
            ("cuda:1", 1, True, True, CUDA1),
            ("cpu", 0, False, True, CPU),
            ("cpu:0", 0, False, True, CPU),
            ("cuda:2", 0, False, True, ValueError),  # index not visible
            ("meta", 0, False, True, ValueError),  # unsupported type
            ("cuda:1", 0, True, True, ValueError),  # conflicts with rank's cuda:0
            ("cuda", 0, False, False, ValueError),  # CUDA unavailable
        ],
    )
    def test_resolve_device(
        self, monkeypatch, device, index_default, distributed, available, expected
    ):
        """Every spelling resolves to bare cpu or an explicitly indexed cuda, and
        invalid requests raise ValueError instead of failing later."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        if expected is ValueError:
            with pytest.raises(ValueError):
                _resolve_device(device, index_default, distributed)
        else:
            assert _resolve_device(device, index_default, distributed) == expected

    @pytest.mark.parametrize(
        ("device", "result_device", "expected", "expected_result"),
        [
            ("cpu", "cpu", CPU, CPU),
            (CPU, None, CPU, CPU),  # None keeps results on the solver's device
            (None, "cpu", CUDA0 if torch.cuda.is_available() else CPU, CPU),
            pytest.param("cpu", "cuda", CPU, CUDA0, marks=needs_cuda),
            # Index-less "cuda" resolves to the solver's GPU, not cuda:0.
            pytest.param("cuda:1", "cuda", CUDA1, CUDA1, marks=needs_2_gpus),
            pytest.param(CUDA1, "cuda:0", CUDA1, CUDA0, marks=needs_2_gpus),
        ],
    )
    def test_solver_device(self, device, result_device, expected, expected_result):
        """The solver and its results land on exactly the requested devices,
        nothing is allocated on an unrequested GPU 0, and the global default /
        current CUDA device are unchanged."""
        default = torch.get_default_device()
        # current_device() also initializes CUDA, which the stats reset requires.
        current = torch.cuda.current_device() if torch.cuda.is_available() else None
        check_gpu0 = current is not None and CUDA0 not in (expected, expected_result)
        if check_gpu0:
            torch.cuda.reset_accumulated_memory_stats(0)
        f, solution, _ = integrand_dict["t_squared"]

        def f_on_device(t):
            assert t.device == expected
            return f(t)

        solver = make_uniform_solver("gk21", device=device)
        result, _ = solver.integrate(
            f_on_device,
            mesh_init=T_INIT,
            mesh_final=T_FINAL,
            result_device=result_device,
        )

        assert solver.device == expected
        assert (solver.get_cuda_memory_usage() == -1) == (expected.type == "cpu")
        assert result.nodes.device == expected_result
        assert torch.allclose(result.integral.cpu(), solution(T_INIT, T_FINAL))
        assert torch.get_default_device() == default
        if current is not None:
            assert torch.cuda.current_device() == current
        if check_gpu0:
            assert torch.cuda.memory_stats(0)["allocated_bytes.all.allocated"] == 0


# ---------------------------------------------------------------------------
# Host waits (CUDA)
# ---------------------------------------------------------------------------


def _lines(function, text=""):
    """The (file, line) of every source line of ``function`` that contains
    ``text`` (every line by default)."""
    function = inspect.unwrap(function)
    source, start = inspect.getsourcelines(function)
    file = inspect.getsourcefile(function)
    return {(file, start + i) for i, line in enumerate(source) if text in line}


def _host_waits(fn, where=None):
    """Run ``fn`` and count the operations that made the host wait for the GPU,
    only those at the (file, line) locations ``where`` when given (a warning
    names the Python line that waited)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    # Not the notice "Synchronization debug mode is a prototype feature ..."
    return sum(
        "called a synchronizing" in str(w.message)
        and (where is None or (w.filename, w.lineno) in where)
        for w in caught
    )


@needs_cuda
class TestHostWaits:
    """The fixed cost of a call on a GPU: steps that need no device value read none."""

    def test_construction_waits_for_nothing(self):
        """Building a solver copies nothing from the host once its method has
        been converted (the first solver of a method, device and dtype copies
        the tables): no tolerance or default y0 tensor is copied to the GPU."""
        adaptive_quadrature("uniform", "gk21", device="cuda")
        waits = _host_waits(
            lambda: adaptive_quadrature("uniform", "gk21", device="cuda")
        )
        assert waits == 0

    @pytest.mark.parametrize("runtime_checks", [True, False])
    def test_starting_mesh_reads_once_per_check(self, runtime_checks):
        """A fresh starting mesh reads the device for its order check (with
        runtime_checks), nothing else."""
        solver = adaptive_quadrature(
            "uniform", "gk21", device="cuda", runtime_checks=runtime_checks
        )
        a, b = (torch.tensor([x], dtype=torch.float64, device="cuda") for x in (0, 1))
        waits = _host_waits(lambda: solver._setup_initial_mesh(None, a, b, True, 13))
        assert waits == runtime_checks

    @pytest.mark.parametrize("four_pieces", [False, True])
    @pytest.mark.parametrize("values_on", ["cuda", "cpu"])
    def test_mesh_update_reads_once(self, values_on, four_pieces):
        """The mesh update reads the device once, for the number of accepted
        subintervals and of those split in four, whatever it keeps: method
        output, values moved within their array, tracked variables, nodes,
        split counts, the split cap, the four-way splits of
        split_rule="accelerate" (error_ratios). Values on result_device
        ("cpu", trade_speed_for_memory) cost one more copy of the kept
        positions per array."""
        solver = adaptive_quadrature("uniform", "gk21", device="cuda")
        n = 6
        mesh = torch.linspace(0, 1, n + 1, dtype=torch.float64, device="cuda")[:, None]
        trackers = torch.arange(n + 1, device="cuda") < n
        idxs = torch.arange(n, device="cuda")
        nodes = solver._compute_nodes(mesh[:-1], mesh[1:])
        y = torch.sin(nodes).to(values_on)
        output = solver._calculate_integral(nodes, y.to("cuda"), mesh[:-1], mesh[1:])
        keep = torch.tensor([True, False, True, True, False, True], device="cuda")
        split_counts = torch.ones(n + 1, dtype=torch.long, device="cuda")
        ratios = torch.full((n,), 1e5, dtype=torch.float64, device="cuda")

        def update():
            return solver._adaptively_increase_mesh(
                output,
                keep,
                y,
                nodes,
                mesh,
                idxs,
                trackers,
                tracked_step_eval=(y.clone(),),
                split_counts=split_counts,
                max_adaptive_splits=10,
                in_place=True,
                error_ratios=ratios if four_pieces else None,
            )

        meshes = []
        waits = _host_waits(lambda: meshes.append(update()[4]))
        assert waits == 1 + 2 * (values_on == "cpu")
        assert len(meshes[0]) == n + 1 + 2 * (3 if four_pieces else 1)

    @pytest.mark.parametrize("runtime_checks", [True, False])
    def test_evaluation_reads_only_for_the_finite_check(self, runtime_checks):
        """Evaluating and integrating the waiting subintervals reads the
        device only for the finite check of each call of f (runtime_checks):
        the batch size comes from the number of waiting subintervals, known on
        the host."""
        solver = adaptive_quadrature(
            "uniform", "gk21", device="cuda", runtime_checks=runtime_checks
        )
        mesh = torch.linspace(0, 1, 9, dtype=torch.float64, device="cuda")[:, None]
        waiting = torch.arange(8, device="cuda")
        solver._batch, solver._batch_converged = 3 * solver.C, True
        cuda = torch.device("cuda", torch.cuda.current_device())
        waits = _host_waits(
            lambda: solver._evaluate_f_on_mesh(
                torch.sin, (), mesh, waiting, False, cuda, float("inf")
            )
        )
        assert waits == 3 * runtime_checks  # three calls of f

    @pytest.mark.parametrize("result_device", [None, "cpu"])
    def test_loop_reads_once_per_pass_and_record_only_values(self, result_device):
        """Over an integration of several passes, the loop reads the device once
        per pass at its condition (which subintervals wait) and once when none
        is left. Recording the accepted subintervals reads nothing with the
        record on the GPU, and with result_device="cpu" only for the copy of
        the values to the host (y and the tracked variable); putting the record
        in mesh order reads twice (whether it is in order, and the runtime
        check that it covers the mesh once)."""
        calls = {"passes": 0, "records": 0}

        def run():
            solver = adaptive_quadrature(
                "uniform", "gk21", device="cuda", atol=1e-10, rtol=0.0
            )
            for name, key in (
                ("_evaluate_f_on_mesh", "passes"),
                ("_record_results", "records"),
            ):
                method = getattr(solver, name)

                def counted(*args, _method=method, _key=key, **kwargs):
                    calls[_key] += 1
                    return _method(*args, **kwargs)

                setattr(solver, name, counted)
            calls.update(passes=0, records=0)
            torch.manual_seed(0)
            solver.integrate(
                lambda t: (torch.sin(300 * t), (t,)),
                mesh_init=torch.tensor([0.0], dtype=torch.float64),
                mesh_final=torch.tensor([1.0], dtype=torch.float64),
                take_gradient=False,
                max_batch=10**6,
                result_device=result_device,
            )

        condition = _lines(AdaptiveQuadrature.integrate, "while len(waiting")
        assert len(condition) == 1
        assert _host_waits(run, condition) == calls["passes"] + 1
        assert calls["passes"] > 2
        record = _host_waits(run, _lines(AdaptiveQuadrature._record_results))
        assert record == (0 if result_device is None else 2 * calls["records"])
        if result_device is None:
            assert _host_waits(run, _lines(AdaptiveQuadrature._sort_record)) == 2

    def test_loop_reads_only_where_pinned(self):
        """The reads of the device that grow with the number of passes are the
        ones the tests of this class pin, and no others: per pass, the loop's
        condition and the mesh update; per call of f, its finite check; per
        re-check, the failing subintervals (the record on the GPU). Two
        integrals that take different numbers of passes therefore differ in
        their reads by exactly these; what is read before and after the loop
        is the same for both."""

        def reads(frequency):
            calls = {"update": 0, "recheck": 0, "f": 0}
            solver = adaptive_quadrature(
                "uniform", "gk21", device="cuda", atol=1e-10, rtol=0.0
            )
            for name, key in (
                ("_adaptively_increase_mesh", "update"),
                ("_failed_record_positions", "recheck"),
            ):
                method = getattr(solver, name)

                def counted(*args, _method=method, _key=key, **kwargs):
                    calls[_key] += 1
                    return _method(*args, **kwargs)

                setattr(solver, name, counted)

            def f(t):
                calls["f"] += 1
                return torch.sin(frequency * t)

            torch.manual_seed(0)
            waits = _host_waits(
                lambda: solver.integrate(
                    f,
                    mesh_init=torch.tensor([0.0], dtype=torch.float64),
                    mesh_final=torch.tensor([1.0], dtype=torch.float64),
                    take_gradient=False,
                    max_batch=10**6,
                    result_device=None,
                    build_state_mesh=False,
                )
            )
            # Every pass updates the mesh once (no update after the loop
            # without the state's mesh).
            pinned = 2 * calls["update"] + calls["f"] + calls["recheck"]
            return waits - pinned, calls["update"]

        (rest_few, passes_few), (rest_many, passes_many) = reads(30.0), reads(300.0)
        assert passes_many > passes_few + 2
        assert rest_many == rest_few

    @pytest.mark.parametrize("error_norm", ["2", "failure_fraction"])
    @pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "cumulative"])
    def test_error_control_reads_once_per_recheck(self, error_norm, absolute):
        """Judging a pass reads the device only in the re-check of accepted
        subintervals, once per pass (which of them fail and can be split): the
        batch's error ratios and masks, the running sums of cumulative mode
        and the final check read nothing, and whether any recorded subinterval
        is split comes from the size of the re-check's result, on the host."""
        calls = {"rechecks": 0}

        def run():
            solver = adaptive_quadrature(
                "uniform",
                "gk21",
                device="cuda",
                atol=1e-10,
                rtol=1e-8,
                error_norm=error_norm,
                use_absolute_error_ratio=absolute,
            )
            method = solver._failed_record_positions

            def counted(*args, **kwargs):
                calls["rechecks"] += 1
                return method(*args, **kwargs)

            solver._failed_record_positions = counted
            calls["rechecks"] = 0
            torch.manual_seed(0)
            solver.integrate(
                lambda t: torch.sin(1.0 / (t + 1e-3)).repeat(1, 3),
                mesh_init=torch.tensor([0.0], dtype=torch.float64),
                mesh_final=torch.tensor([1.0], dtype=torch.float64),
                take_gradient=False,
                max_batch=10**6,
            )

        where = set().union(
            *(
                _lines(getattr(AdaptiveQuadrature, name))
                for name in (
                    "_compute_error_ratios",
                    "_error_ratios_norm",
                    "_error_ratios_failure_fraction",
                    "_accept_reject_masks",
                    "_record_error_check",
                    "_failed_record_positions",
                    "_merge_positions",
                    "_insert_sorted_results",
                )
            ),
            _lines(AdaptiveQuadrature.integrate, "cum"),
            _lines(AdaptiveQuadrature.integrate, "extra_split_idxs"),
            _lines(AdaptiveQuadrature.integrate, "record_keep_mask"),
        )
        assert _host_waits(run, where) == calls["rechecks"]
        assert calls["rechecks"] > 2
