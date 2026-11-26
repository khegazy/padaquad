"""Unit tests for the adaptive_quadrature factory, MethodClass.weights through a
solver, the per-step node count C, and the size of a fresh starting mesh."""

from __future__ import annotations

import gc
import math

import pytest
import torch
from _helpers import (
    REMOVE_CUT,
    make_solver_for_unit_test,
    make_variable_solver_for_unit_test,
    needs_cuda,
)

from padaquad import (
    UniformAdaptiveQuadrature,
    VariableAdaptiveQuadrature,
    adaptive_quadrature,
    integrate,
)
from padaquad.base import steps

# ---------------------------------------------------------------------------
# adaptive_quadrature factory
# ---------------------------------------------------------------------------


class TestGetParallelRKSolver:
    """Tests for the factory function that creates uniform or variable solvers."""

    def test_uniform_string(self):
        """String 'uniform' returns a uniform solver."""
        solver = adaptive_quadrature(
            sampling_type="uniform",
            method="bosh3",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, UniformAdaptiveQuadrature)

    def test_default_is_uniform(self):
        """No sampling_type gives a uniform solver (the README's warm-start example)."""
        solver = adaptive_quadrature(method="gk21", atol=1e-8, rtol=1e-8)
        assert isinstance(solver, UniformAdaptiveQuadrature)

    def test_variable_string(self):
        """String 'variable' returns a variable solver."""
        solver = adaptive_quadrature(
            sampling_type="variable",
            method="adaptive_heun",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, VariableAdaptiveQuadrature)

    def test_uniform_enum(self):
        """steps.ADAPTIVE_UNIFORM enum returns a uniform solver."""
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_UNIFORM,
            method="bosh3",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, UniformAdaptiveQuadrature)

    def test_variable_enum(self):
        """steps.ADAPTIVE_VARIABLE enum returns a variable solver."""
        solver = adaptive_quadrature(
            sampling_type=steps.ADAPTIVE_VARIABLE,
            method="adaptive_heun",
            atol=1e-6,
            rtol=1e-6,
            remove_cut=REMOVE_CUT,
        )
        assert isinstance(solver, VariableAdaptiveQuadrature)

    def test_fixed_raises(self):
        """steps.FIXED raises ValueError naming it and the supported types."""
        with pytest.raises(ValueError, match=r"ADAPTIVE_UNIFORM.*got steps\.FIXED"):
            adaptive_quadrature(
                sampling_type=steps.FIXED,
                method="bosh3",
                atol=1e-6,
                rtol=1e-6,
                remove_cut=REMOVE_CUT,
            )

    def test_unknown_string_raises(self):
        """An unknown string raises ValueError (not KeyError) naming it."""
        with pytest.raises(ValueError, match=r"'bogus'.*'uniform'"):
            adaptive_quadrature(sampling_type="bogus", method="bosh3")


# ---------------------------------------------------------------------------
# Uniform rule weights
# ---------------------------------------------------------------------------


class TestUniformWeights:
    """Tests for UniformMethodClass.weights through a uniform solver."""

    def test_shape(self):
        """b and b_error have shape [N, C, 1] for bosh3 (C=4)."""
        solver = make_solver_for_unit_test("bosh3")
        t = torch.rand(5, 4, 1, dtype=torch.float64)
        b, b_error = solver.method.weights(t)
        assert b.shape == (5, 4, 1)
        assert b_error.shape == (5, 4, 1)

    def test_t_independent(self):
        """Different t values give identical weights for every panel."""
        solver = make_solver_for_unit_test("bosh3")
        t1 = torch.rand(3, 4, 1, dtype=torch.float64)
        t2 = torch.rand(7, 4, 1, dtype=torch.float64)
        b1, _ = solver.method.weights(t1)
        b2, _ = solver.method.weights(t2)
        assert torch.equal(b1[0], b2[0])
        assert torch.equal(b1, b1[:1].expand_as(b1))
        assert torch.equal(b2, b2[:1].expand_as(b2))

    def test_values_match_tableau(self):
        """Returned b matches method.tableau.b unsqueezed."""
        solver = make_solver_for_unit_test("bosh3")
        t = torch.rand(1, 4, 1, dtype=torch.float64)
        b, _ = solver.method.weights(t)
        expected = solver.method.tableau.b.unsqueeze(-1)
        assert torch.equal(b[0], expected)


# ---------------------------------------------------------------------------
# Variable-solver rule weights
# ---------------------------------------------------------------------------


class TestVariableWeights:
    """Tests for MethodClass.weights through a variable solver."""

    def test_shape_adaptive_heun(self):
        """adaptive_heun variable: b shape [N, 2, 1] (constant over panels)."""
        solver = make_variable_solver_for_unit_test("adaptive_heun")
        t = torch.tensor(
            [
                [[0.0], [1.0]],
                [[1.0], [2.0]],
                [[2.0], [3.0]],
                [[3.0], [4.0]],
                [[4.0], [5.0]],
            ],
            dtype=torch.float64,
        )
        b, b_error = solver.method.weights(t)
        # 2nd-order: weights are constant, expanded to [N, C, 1]
        assert b.shape == (5, 2, 1)
        assert b_error.shape == (5, 2, 1)

    def test_normalizes_to_unit(self):
        """generic3: mesh=[2,3,4] normalizes to c=[0, 0.5, 1] → Simpson weights."""
        solver = make_variable_solver_for_unit_test("interpolatory3_variable")
        t = torch.tensor([[[2.0], [3.0], [4.0]]], dtype=torch.float64)
        b, _ = solver.method.weights(t)
        # At a=0.5: b0=1/6, ba=2/3, b1=1/6
        expected = torch.tensor(
            [[[1.0 / 6], [2.0 / 3], [1.0 / 6]]], dtype=torch.float64
        )
        assert torch.allclose(b, expected, atol=1e-12)

    def test_different_positions_different_weights(self):
        """generic3: two steps with different midpoints produce different weights."""
        solver = make_variable_solver_for_unit_test("interpolatory3_variable")
        t = torch.tensor(
            [
                [[0.0], [0.3], [1.0]],  # a=0.3
                [[0.0], [0.7], [1.0]],  # a=0.7
            ],
            dtype=torch.float64,
        )
        b, _ = solver.method.weights(t)
        assert not torch.allclose(b[0], b[1])


# ---------------------------------------------------------------------------
# Number of quadrature points per step (solver.C)
# ---------------------------------------------------------------------------


class TestNumQuadraturePoints:
    """Tests that solver.C matches the method's node count on both solver types."""

    def test_uniform_bosh3(self):
        assert make_solver_for_unit_test("bosh3").C == 4

    def test_uniform_dopri5(self):
        assert make_solver_for_unit_test("dopri5").C == 5

    def test_uniform_adaptive_heun(self):
        assert make_solver_for_unit_test("adaptive_heun").C == 2

    def test_variable_adaptive_heun(self):
        assert make_variable_solver_for_unit_test("adaptive_heun").C == 2

    def test_variable_generic3(self):
        assert make_variable_solver_for_unit_test("interpolatory3_variable").C == 3


# ---------------------------------------------------------------------------
# Keyword-only constructor, factory arguments and integrate() routing
# ---------------------------------------------------------------------------

_LO = torch.tensor([0.0], dtype=torch.float64)
_HI = torch.tensor([1.0], dtype=torch.float64)
_BASE = {"mesh_init": _LO, "mesh_final": _HI, "device": "cpu", "max_batch": 10_000}


class TestConstructorSignature:
    """One keyword-only constructor names every solver option."""

    def test_every_option_by_keyword(self):
        """Each documented option is accepted by keyword and stored."""
        solver = UniformAdaptiveQuadrature(
            f=torch.cos,
            method="DOPRI5",
            atol=1e-7,
            rtol=1e-6,
            y0=torch.tensor([2.0], dtype=torch.float64),
            remove_cut=0.05,
            max_batch=1000,
            total_mem_usage=0.5,
            max_adaptive_splits=4,
            use_absolute_error_ratio=False,
            error_norm="max",
            mesh_failure_tolerance=0.25,
            runtime_checks=False,
            dtype=torch.float32,
            device="cpu",
            output_speed_info=False,
        )
        assert solver.method_name == "dopri5"
        assert solver.dtype == torch.float32
        assert float(solver.atol) == pytest.approx(1e-7)
        assert solver.remove_cut == 0.05
        assert solver.init_max_batch == 1000
        assert solver.init_total_mem_usage == 0.5
        assert solver.init_max_adaptive_splits == 4
        assert solver.use_absolute_error_ratio is False
        assert solver.init_error_norm == "max"
        assert solver.init_mesh_failure_tolerance == 0.25
        assert solver.init_runtime_checks is False
        assert float(solver.init_y0) == 2.0

    def test_positional_arguments_raise(self):
        with pytest.raises(TypeError, match="positional argument"):
            VariableAdaptiveQuadrature("interpolatory3_variable")

    def test_unknown_keyword_names_the_solver(self):
        with pytest.raises(TypeError, match=r"UniformAdaptiveQuadrature.*bogus"):
            UniformAdaptiveQuadrature(bogus=1, device="cpu")

    def test_remove_cut_at_least_one_raises(self):
        with pytest.raises(ValueError, match="remove_cut"):
            UniformAdaptiveQuadrature(remove_cut=1.0, device="cpu")

    def test_factory_takes_method_by_position(self):
        """``adaptive_quadrature(sampling_type, method)``; the method used to
        land in the constructor's first positional slot (remove_cut)."""
        solver = adaptive_quadrature(
            "variable", "interpolatory3_variable", device="cpu"
        )
        assert isinstance(solver, VariableAdaptiveQuadrature)
        assert solver.method_name == "interpolatory3_variable"
        assert solver.remove_cut == 0.1

    def test_failed_construction_prints_no_cleanup_error(self, capfd):
        """A constructor error is not followed by an ignored AttributeError
        from ``__del__`` on the half-built object."""
        with pytest.raises(TypeError):
            UniformAdaptiveQuadrature(bogus=1, device="cpu")
        gc.collect()
        assert "is_distributed" not in capfd.readouterr().err


class TestIntegrateRouting:
    """``padaquad.integrate`` sends call options to the solver's integrate
    call and the others to the constructor."""

    @pytest.mark.parametrize(
        "option",
        [
            {"N_init_steps": 5},
            {"loss_fxn": lambda r: r.integral.sum()},
            {"random_initial_mesh": False},
            {"error_integral_reference": 1.0},
            {"trade_speed_for_memory": True, "take_gradient": False},
            {"enforce_convergence": False},
        ],
        ids=lambda o: next(iter(o)),
    )
    def test_call_options_are_accepted(self, option):
        result, _ = integrate(torch.cos, **_BASE, **option)
        assert float(result.integral) == pytest.approx(0.8414709848, abs=1e-8)

    def test_f_args_reach_the_integrand(self):
        result, _ = integrate(lambda t, k: k * torch.cos(t), **_BASE, f_args=(2.0,))
        assert float(result.integral) == pytest.approx(2 * 0.8414709848, abs=1e-8)

    def test_constructor_options_are_used(self):
        result, _ = integrate(torch.cos, "interpolatory3_variable", "variable", **_BASE)
        assert float(result.integral) == pytest.approx(0.8414709848, abs=1e-5)

    def test_unknown_option_raises(self):
        with pytest.raises(TypeError, match="bogus"):
            integrate(torch.cos, **_BASE, bogus=1)


class TestDefaultMethod:
    """``method=None`` means the sampling type's default method."""

    def test_uniform_default_is_gk21(self):
        assert UniformAdaptiveQuadrature(device="cpu").method_name == "gk21"
        assert adaptive_quadrature("uniform", device="cpu").method_name == "gk21"

    def test_variable_default_is_interpolatory3(self):
        """``gk21`` is not a variable method; the variable solver used to raise
        ``ValueError: Unknown method 'gk21'`` without an explicit method."""
        expected = "interpolatory3_variable"
        assert VariableAdaptiveQuadrature(device="cpu").method_name == expected
        assert adaptive_quadrature("variable", device="cpu").method_name == expected

    def test_integrate_variable_without_method(self):
        result, _ = integrate(torch.cos, sampling="variable", **_BASE)
        assert float(result.integral) == pytest.approx(0.8414709848, abs=1e-6)


# ---------------------------------------------------------------------------
# Size of a fresh starting mesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_init_steps", "root"),
    [(1, 1), (13, 3), (16, 4), (100, 10), (1e4, 100), (3073**2 - 1, 3072)],
    ids=["1", "default-13", "square-16", "100", "float-1e4", "3073sq-1"],
)
def test_starting_mesh_has_isqrt_segments(n_init_steps, root):
    """A fresh starting mesh has k = isqrt(N_init_steps) segments of k + 1
    points: k (k + 1) mesh points. At 3073**2 - 1 the float32 square root used
    before rounded up to 3073 (on an AVX2 CPU with torch 2.8); a float N_init_steps
    is accepted as before."""
    solver = make_solver_for_unit_test()
    a, b = (torch.tensor([x], dtype=torch.float64) for x in (0.0, 1.0))
    mesh, trackers = solver._setup_initial_mesh(None, a, b, False, n_init_steps)
    assert len(mesh) == root * (root + 1)
    assert trackers.sum() == len(mesh) - 1
    assert not trackers[-1]


# Seeds, sizes and intervals whose random starting mesh had a repeated point
# (float32: offsets closer than the float spacing at the segment start round
# to one point; float64 at 10^5 points: torch.rand's float32 draws repeat) or,
# at 10^6 points on [1, 3] with seed 10, a segment's last point one unit in the
# last place above the next segment's first point; each failed the order
# assertion, or with runtime_checks=False gave subintervals of width 0.
_COLLIDING_MESHES = [
    (torch.float32, 10**4, (0.0, 1.0), range(20)),
    (torch.float32, 10**5, (0.0, 1.0), range(20)),
    (torch.float64, 10**5, (0.0, 1.0), range(20)),
    (torch.float32, 10**6, (1.0, 3.0), [10]),
    (torch.float32, 13, (1e4, 1e4 + 1), range(100)),
    (torch.float64, 10**4, (-1e10 - 1, -1e10), range(3)),
]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
@pytest.mark.parametrize("runtime_checks", [True, False])
@pytest.mark.parametrize(
    ("dtype", "n_init_steps", "interval", "seeds"),
    _COLLIDING_MESHES,
    ids=[
        "f32-1e4",
        "f32-1e5",
        "f64-1e5",
        "f32-1e6-reversed",
        "f32-13-far",
        "f64-1e4-neg",
    ],
)
def test_starting_mesh_drops_colliding_points(
    dtype, n_init_steps, interval, seeds, runtime_checks, device
):
    """A fresh random mesh is strictly increasing whatever rounding does, with
    the bounds as its ends bit for bit and their gradients: the points that
    round onto or below an earlier point are dropped (in each case at least one
    seed drops one), with runtime_checks on and off."""
    solver = adaptive_quadrature(
        "uniform", "gk21", device=device, dtype=dtype, runtime_checks=runtime_checks
    )
    a, b = (torch.tensor([x], dtype=dtype, device=device) for x in interval)
    a.requires_grad_(), b.requires_grad_()
    root = math.isqrt(n_init_steps)
    dropped = False
    for seed in seeds:
        torch.manual_seed(seed)
        mesh, trackers = solver._setup_initial_mesh(None, a, b, True, n_init_steps)
        assert bool((mesh[1:] > mesh[:-1]).all())
        assert torch.equal(mesh[0], a.detach()) and torch.equal(mesh[-1], b.detach())
        assert int(trackers.sum()) == len(mesh) - 1 and not trackers[-1]
        dropped |= len(mesh) < root * (root + 1)
        (2 * mesh[0] + 3 * mesh[-1]).sum().backward()
        assert float(a.grad) == 2 and float(b.grad) == 3
        a.grad = b.grad = None
    assert dropped


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
@pytest.mark.parametrize("runtime_checks", [True, False])
@pytest.mark.parametrize(
    ("dtype", "n_init_steps", "tol"),
    [(torch.float32, 10**4, 1e-5), (torch.float64, 10**5, 1e-10)],
    ids=["f32-1e4", "f64-1e5"],
)
def test_colliding_starting_mesh_integrates(
    dtype, n_init_steps, tol, runtime_checks, device
):
    """A starting mesh that had repeated points (the first such seed)
    integrates e^{-t} sin(40 t) on [0, 1] to its exact value, with no
    subinterval of width 0 in the result or the state's mesh."""
    exact = (40 - math.exp(-1) * (math.sin(40) + 40 * math.cos(40))) / 1601
    solver = adaptive_quadrature(
        "uniform",
        "gk7",
        atol=tol,
        rtol=tol,
        device=device,
        dtype=dtype,
        runtime_checks=runtime_checks,
    )
    a, b = (torch.tensor([x], dtype=dtype, device=device) for x in (0.0, 1.0))
    root = math.isqrt(n_init_steps)
    seed = next(
        seed
        for seed in range(20)
        if torch.manual_seed(seed) is not None
        and len(solver._setup_initial_mesh(None, a, b, True, n_init_steps)[0])
        < root * (root + 1)
    )
    torch.manual_seed(seed)
    result, state = solver.integrate(
        lambda t: torch.exp(-t) * torch.sin(40 * t),
        mesh_init=a,
        mesh_final=b,
        N_init_steps=n_init_steps,
        take_gradient=False,
        max_batch=10**6,
    )
    assert result.n_start_steps < root * (root + 1) - 1  # points were dropped
    assert bool((result.h > 0).all())
    assert bool((state.mesh[1:] > state.mesh[:-1]).all())
    assert float(result.integral) == pytest.approx(exact, rel=10 * tol)
