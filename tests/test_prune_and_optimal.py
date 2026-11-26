"""Unit tests for _prune_excess_mesh, _get_optimal_mesh and the merge rules."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from _helpers import (
    ATOL_LOOSE,
    REMOVE_CUT,
    RTOL_LOOSE,
    T_FINAL,
    T_INIT,
    UNIFORM_METHOD_NAMES,
    make_solver_for_unit_test,
    make_uniform_solver,
    needs_cuda,
)

from padaquad import (
    UNIFORM_METHODS,
    VARIABLE_METHODS,
    adaptive_quadrature,
    integrand_dict,
    integrate,
)
from padaquad.methods import UniformMethodClass, _Tableau

#: The merge rules whose meshes are checked end to end: the default and the
#: 1.3.0 and 1.4.0 one ("sum" merges pairs whose merged step fails, by design).
MERGE_RULES = ["predicted", "power_law"]
#: max_batch for the end-to-end merge tests below: their meshes are small, and an
#: explicit value skips the memory benchmark of make_uniform_solver.
SMALL_BATCH = 100_000

# ---------------------------------------------------------------------------
# _prune_excess_mesh
# ---------------------------------------------------------------------------


class TestPruneExcessT:
    """Tests for AdaptiveQuadrature._prune_excess_mesh."""

    def _make_mesh(self, t_start, t_end, N):
        """Create an [N + 1, 1] mesh of N equal subintervals."""
        return torch.linspace(t_start, t_end, N + 1, dtype=torch.float64).unsqueeze(-1)

    def test_no_pairs_below_cut(self):
        """All error_ratios_2steps above remove_cut: no pruning."""
        solver = make_solver_for_unit_test()
        mesh = self._make_mesh(0.0, 1.0, 4)
        ss = torch.ones(4, 1, dtype=torch.float64)
        se = torch.ones(4, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.5, 0.8, 0.3])  # All > 0.1

        m_p, ss_p, se_p = solver._prune_excess_mesh(
            mesh, ss, se, er2, solver.remove_cut
        )

        assert torch.equal(m_p, mesh)
        assert torch.equal(ss_p, ss)
        assert torch.equal(se_p, se)

    def test_one_pair_pruned(self):
        """First pair below cut: its shared mesh point goes, N-1 subintervals."""
        solver = make_solver_for_unit_test()
        mesh = self._make_mesh(0.0, 1.0, 4)
        ss = torch.ones(4, 1, dtype=torch.float64)
        se = torch.ones(4, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.8, 0.9])  # Pair 0 below the cut 0.1

        m_p, ss_p, se_p = solver._prune_excess_mesh(
            mesh, ss, se, er2, solver.remove_cut
        )

        assert torch.equal(m_p, mesh[[0, 2, 3, 4]])
        assert ss_p.shape[0] == se_p.shape[0] == 3

    def test_empty_error_ratios(self):
        """Single step (empty 2steps array): returns unchanged."""
        solver = make_solver_for_unit_test()
        mesh = self._make_mesh(0.0, 1.0, 1)
        ss = torch.ones(1, 1, dtype=torch.float64)
        se = torch.ones(1, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([])

        m_p, _ss_p, _se_p = solver._prune_excess_mesh(
            mesh, ss, se, er2, solver.remove_cut
        )

        assert torch.equal(m_p, mesh)

    def test_nonfinite_pair_ratio_not_merged(self):
        """A NaN or Inf pair ratio (non-finite step error) never merges."""
        solver = make_solver_for_unit_test()
        mesh = self._make_mesh(0.0, 1.0, 5)
        ss = torch.ones(5, 1, dtype=torch.float64)
        se = torch.ones(5, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([float("nan"), 0.8, float("inf"), 0.8])

        m_p, _, _ = solver._prune_excess_mesh(mesh, ss, se, er2, solver.remove_cut)

        assert torch.equal(m_p, mesh)

    def test_no_adjacent_merges(self):
        """Adjacent pairs both below cut: _rec_remove prevents double merge."""
        solver = make_solver_for_unit_test()
        mesh = self._make_mesh(0.0, 1.0, 5)
        ss = torch.ones(5, 1, dtype=torch.float64)
        se = torch.ones(5, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.01, 0.8, 0.01])  # Pairs 0,1 adjacent

        m_p, _, _ = solver._prune_excess_mesh(mesh, ss, se, er2, solver.remove_cut)

        # Only non-adjacent pairs merged: pair 0 and pair 3 (indices 0 and 3)
        # Pair 1 blocked by adjacency. Result: 5 - 2 = 3 subintervals, whose
        # shared mesh points 1 and 4 are gone.
        assert torch.equal(m_p, mesh[[0, 2, 3, 5]])

    def test_mesh_quadratures_accumulated(self):
        """After merge, sum of pair equals sum of originals; inputs unchanged."""
        solver = make_solver_for_unit_test()
        mesh = self._make_mesh(0.0, 1.0, 3)
        ss = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        se = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64)
        er2 = torch.tensor([0.01, 0.8])  # Pair 0 below cut

        _, ss_p, se_p = solver._prune_excess_mesh(mesh, ss, se, er2, solver.remove_cut)

        assert ss_p.shape[0] == 2
        # Merged: 1+2=3, remaining: 3
        assert torch.equal(
            ss_p[:, 0], torch.tensor([1.0 + 2.0, 3.0], dtype=torch.float64)
        )
        assert torch.equal(
            se_p[:, 0], torch.tensor([0.1 + 0.2, 0.3], dtype=torch.float64)
        )
        assert torch.equal(ss[:, 0], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    @pytest.mark.parametrize("method", ["bosh3", "dopri5", "gk21", "cc17"])
    def test_merge_inputs_per_rule(self, method):
        """_merge_inputs returns the merge prediction and cutoff of each
        merge_rule: "sum" and "power_law" add no prediction (None) to the
        summed errors, with the cutoffs remove_cut and remove_cut / 2**(k - 1),
        k = error_exponent; "predicted" returns the prediction, with the
        cutoff remove_cut; the prediction is positive for sin(3t), so an
        implementation that dropped it would fail."""
        sizes = {}
        mesh = self._make_mesh(0.0, 1.0, 3)
        for rule in ("sum", "power_law", "predicted"):
            solver = make_solver_for_unit_test(method, merge_rule=rule)
            t = solver._compute_nodes(mesh[:-1], mesh[1:])
            y = torch.sin(3 * t)
            b, _ = solver.method.weights(t)
            q = (mesh[1:] - mesh[:-1]) * (b * y).sum(1)
            sizes[rule] = solver._merge_inputs(y, q, mesh)
        k = make_solver_for_unit_test(method).method.error_exponent
        assert sizes["sum"] == (None, REMOVE_CUT)
        assert sizes["power_law"] == (None, REMOVE_CUT / 2 ** (k - 1))
        merge_prediction, cut = sizes["predicted"]
        assert cut == REMOVE_CUT
        predicted = solver._predict_merged_errors(y, q, mesh)
        assert torch.all(predicted > 0)
        assert torch.equal(merge_prediction, predicted)


# ---------------------------------------------------------------------------
# _get_optimal_mesh
# ---------------------------------------------------------------------------


class TestGetOptimalTStepBarriers:
    """Tests for _get_optimal_mesh: post-convergence mesh optimization."""

    def _make_record(self, solver, N, integral_val=1.0, error_scale=0.001):
        """Build a record dict with N uniform steps in [0, 1]."""
        return {
            "mesh_quadratures": torch.ones(N, 1, dtype=torch.float64)
            * (integral_val / N),
            "mesh_quadrature_errors": torch.ones(N, 1, dtype=torch.float64)
            * error_scale,
            "integral": torch.tensor([integral_val], dtype=torch.float64),
            "y": torch.full((N, solver.C, 1), integral_val, dtype=torch.float64),
        }

    def test_over_resolved_pruned(self):
        """Record with tiny errors → fewer barriers."""
        solver = make_solver_for_unit_test(atol=1e-3, rtol=1e-3)
        N = 20
        record = self._make_record(solver, N, error_scale=1e-10)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        assert len(barriers_opt) < len(barriers)

    def test_under_resolved_refined(self):
        """Record with large errors → more barriers."""
        solver = make_solver_for_unit_test(atol=1e-12, rtol=1e-12)
        N = 3
        record = self._make_record(solver, N, error_scale=10.0)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        assert len(barriers_opt) > len(barriers)

    def test_output_sorted(self):
        """Output barriers are strictly increasing."""
        solver = make_solver_for_unit_test(atol=1e-6, rtol=1e-6)
        N = 10
        record = self._make_record(solver, N, error_scale=0.001)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        diffs = barriers_opt[1:, 0] - barriers_opt[:-1, 0]
        assert torch.all(diffs > 0), f"Barriers not sorted: {barriers_opt[:, 0]}"

    def test_endpoints_preserved(self):
        """First and last barriers match 0 and 1."""
        solver = make_solver_for_unit_test(atol=1e-6, rtol=1e-6)
        N = 10
        record = self._make_record(solver, N, error_scale=0.001)
        barriers = torch.linspace(0, 1, N + 1, dtype=torch.float64).unsqueeze(-1)

        barriers_opt = solver._get_optimal_mesh(**record, mesh=barriers)

        assert torch.allclose(barriers_opt[0], torch.tensor([0.0], dtype=torch.float64))
        assert torch.allclose(
            barriers_opt[-1], torch.tensor([1.0], dtype=torch.float64)
        )


@pytest.mark.parametrize(
    ("device", "result_device"),
    [
        ("cpu", "cpu"),
        pytest.param("cuda", "cpu", marks=needs_cuda),
        pytest.param("cuda", None, marks=needs_cuda),
    ],
    ids=["cpu", "cuda-result_cpu", "cuda-result_cuda"],
)
@pytest.mark.parametrize("case", ["merge", "capped"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize(
    ("sampling", "method"),
    [("uniform", name) for name in UNIFORM_METHOD_NAMES]
    + [("variable", name) for name in VARIABLE_METHODS],
)
def test_mesh_optimal_ends_are_the_bounds(
    sampling, method, dtype, case, device, result_device
):
    """mesh_optimal starts and ends at the call's bounds bit for bit, so a
    later call given mesh=mesh_optimal and no bounds integrates the previous
    call's interval. "merge" starts from an
    over-resolved mesh, so the optimal mesh merges subintervals (not at both
    ends in every cell); "capped" stops short of the tolerance, so the optimal
    mesh splits failing subintervals. On a CUDA solver mesh_optimal is built
    on result_device and moved back to the solver's device."""
    tol = 1e-6 if dtype == torch.float64 else 1e-5
    solver = adaptive_quadrature(
        sampling,
        method,
        atol=tol,
        rtol=tol,
        device=device,
        dtype=dtype,
        max_batch=SMALL_BATCH,
    )
    if case == "merge":
        mesh = torch.linspace(0.3, 1.7, 129, dtype=dtype).unsqueeze(-1)
        out, out_state = solver.integrate(
            torch.exp, mesh=mesh, take_gradient=False, result_device=result_device
        )
    else:
        mesh = torch.tensor([[0.3], [4.0]], dtype=dtype)  # the bounds only
        out, out_state = solver.integrate(
            lambda t: torch.sin(20 * t),
            mesh_init=mesh[0],
            mesh_final=mesh[-1],
            max_adaptive_splits=1,
            take_gradient=False,
            result_device=result_device,
        )
    assert torch.equal(out_state.mesh[0].cpu(), mesh[0])
    assert torch.equal(out_state.mesh[-1].cpu(), mesh[-1])
    assert torch.all(torch.diff(out_state.mesh[:, 0]) > 0)


# ---------------------------------------------------------------------------
# Merge cutoff end to end: a merged step must pass when it is re-evaluated.
# ---------------------------------------------------------------------------


def _warm_start_passes(solver, f, cold_state) -> bool:
    """Re-integrate f from the cold call's state; True when no starting
    subinterval fails, i.e. none of that mesh's subintervals had to be split."""
    out, out_state = solver.integrate(
        f=f, mesh=cold_state.mesh, max_batch=cold_state.max_batch, take_gradient=False
    )
    return out.n_start_steps_split == 0


@pytest.mark.parametrize("merge_rule", MERGE_RULES)
@pytest.mark.parametrize("method", UNIFORM_METHOD_NAMES)
@pytest.mark.parametrize("integrand", ["sine_squared", "exp", "damped_sine"])
@pytest.mark.parametrize(
    ("atol", "rtol"), [(ATOL_LOOSE, RTOL_LOOSE), (1e-9, 1e-9)], ids=["loose", "tight"]
)
def test_optimal_mesh_passes_when_reevaluated(
    method, integrand, atol, rtol, merge_rule
):
    """Re-integrating the same integrand from its own mesh_optimal fails no
    step: merged steps are predicted to stay below tolerance. With
    merge_rule="sum" 7-22 % of these steps failed."""
    if (atol, rtol) == (1e-9, 1e-9) and method in ("adaptive_heun", "fehlberg2"):
        pytest.skip("1e5+ panels at this tolerance; covered at the loose one")
    f = integrand_dict[integrand][0]
    solver = make_uniform_solver(
        method, atol=atol, rtol=rtol, device="cpu", merge_rule=merge_rule
    )
    torch.manual_seed(0)
    cold, cold_state = solver.integrate(
        f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
    )
    assert _warm_start_passes(solver, f, cold_state)


@pytest.mark.parametrize(
    ("method", "f", "atol", "rtol"),
    [
        ("dopri5", lambda t: torch.cos(2 * torch.pi * t), 1e-14, 1e-6),
        ("cc17", lambda t: torch.sin(2 * torch.pi * 3.7 * t), 1e-12, 1e-8),
    ],
    ids=["dopri5-cos", "cc17-sin"],
)
@pytest.mark.parametrize("merge_rule", MERGE_RULES)
def test_optimal_mesh_passes_when_reevaluated_cumulative(
    method, f, atol, rtol, merge_rule
):
    """Cumulative tolerance with a running integral that falls toward 0
    (cos 2 pi t at t -> 1, sin 2 pi 3.7 t near t = 0.5). A merged pair is
    judged on re-evaluation against |cum| at its right end; the pair ratio
    used the larger of the two steps' |cum|, so pairs merged whose merged step
    failed (merged-step ratio 10.9 and 5.0 in these two cases)."""
    solver = make_uniform_solver(
        method,
        atol=atol,
        rtol=rtol,
        device="cpu",
        use_absolute_error_ratio=False,
        merge_rule=merge_rule,
    )
    torch.manual_seed(0)
    cold, cold_state = solver.integrate(
        f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
    )
    assert _warm_start_passes(solver, f, cold_state)


@pytest.mark.parametrize("use_absolute_error_ratio", [True, False])
@pytest.mark.parametrize("error_norm", ["2", "max", "failure_fraction"])
def test_pair_ratio_is_merged_step_ratio(use_absolute_error_ratio, error_norm):
    """error_ratios_2steps of a pair equals the error ratio the merged step
    gets when its error is the pair's summed error: same tolerance, including
    the cumulative tolerance at the merged step's right end (|cum| drops from
    1 to 1e-3 across this pair). Tolerances sit far above the rounding floor.
    D = 2 checks the vector reductions."""
    solver = make_solver_for_unit_test(
        atol=1e-14,
        rtol=1e-6,
        use_absolute_error_ratio=use_absolute_error_ratio,
        error_norm=error_norm,
    )
    quad = torch.tensor([[1.0, 2.0], [-0.999, -1.998]], dtype=torch.float64)
    err = torch.tensor([[1e-11, 3e-11], [2e-11, 1e-11]], dtype=torch.float64)
    integral = quad.sum(0)
    _, pair, _ = solver._compute_error_ratios(
        mesh_quadrature_errors=err, mesh_quadratures=quad, integral=integral
    )
    merged, _, merged_per_dim = solver._compute_error_ratios(
        mesh_quadrature_errors=err.sum(0, keepdim=True),
        mesh_quadratures=quad.sum(0, keepdim=True),
        integral=integral,
    )
    if error_norm == "failure_fraction":
        merged = merged_per_dim.amax(dim=-1)  # the pair ratio's reduction
    assert torch.allclose(pair, merged, rtol=1e-12, atol=0)


@pytest.mark.parametrize("merge_rule", MERGE_RULES)
@pytest.mark.parametrize("method", ["dopri5", "gk7", "gk21", "cc17"])
def test_repeated_warm_starts_prune_to_a_fixed_point(method, merge_rule):
    """Starting from an over-resolved 128-step mesh, repeated calls from the
    previous mesh_optimal halve the over-resolved steps (one merge level per call) until the
    mesh stops changing; no call fails a starting step and every integral
    stays within tolerance of the exact value."""
    f, solution, _ = integrand_dict["damped_sine"]
    exact = solution(T_INIT, T_FINAL)
    solver = make_uniform_solver(
        method, atol=1e-9, rtol=1e-7, device="cpu", merge_rule=merge_rule
    )
    mesh = torch.linspace(0.0, 1.0, 129, dtype=torch.float64).unsqueeze(-1)
    out, out_state = solver.integrate(f=f, mesh=mesh, take_gradient=False)
    sizes = [len(mesh) - 1, len(out_state.mesh) - 1]
    for _ in range(8):
        out, out_state = solver.integrate(
            f=f, mesh=out_state.mesh, max_batch=out_state.max_batch, take_gradient=False
        )
        assert out.n_start_steps_split == 0
        assert out.converged is True
        assert torch.allclose(out.integral.cpu(), exact, atol=1e-9, rtol=1e-6)
        sizes.append(len(out_state.mesh) - 1)
        if sizes[-1] == sizes[-2]:
            break
    assert sizes[-1] == sizes[-2], f"no fixed point: {sizes}"
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[-1] <= 0.6 * sizes[0], sizes


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
@pytest.mark.parametrize("take_gradient", [False, True])
@pytest.mark.parametrize(
    ("sampling", "method"),
    [("uniform", "gk21"), ("variable", "interpolatory3_variable")],
)
def test_build_state_mesh_false_skips_only_the_mesh(
    sampling, method, take_gradient, device, monkeypatch
):
    """build_state_mesh=False does not build the optimal mesh and returns
    state.mesh = None; the result, the batch size and the gradient equal the
    default call's bit for bit. A later call with that state and no bounds
    raises a ValueError that names the switch; with bounds it starts from a
    fresh mesh with the state's batch size, as a call without a state does."""
    w = torch.tensor([1.3], dtype=torch.float64, device=device, requires_grad=True)
    a, b = (torch.tensor([x], dtype=torch.float64) for x in (0.0, 1.0))

    def f(t):
        return w * torch.sin(9 * t) + t

    def call(build_state_mesh, **kwargs):
        solver = adaptive_quadrature(
            sampling, method, atol=1e-9, rtol=1e-9, device=device, dtype=torch.float64
        )
        built = []
        get_optimal_mesh = solver._get_optimal_mesh
        monkeypatch.setattr(
            solver,
            "_get_optimal_mesh",
            lambda **k: built.append(1) or get_optimal_mesh(**k),
        )
        torch.manual_seed(0)
        w.grad = None
        result, state = solver.integrate(
            f,
            mesh_init=a,
            mesh_final=b,
            take_gradient=take_gradient,
            build_state_mesh=build_state_mesh,
            **kwargs,
        )
        if not take_gradient:
            result.integral.sum().backward()
        return solver, result, state, w.grad.clone(), len(built)

    _, default, default_state, default_grad, n_built = call(True, max_batch=SMALL_BATCH)
    solver, skipped, state, grad, n_skipped = call(False, max_batch=SMALL_BATCH)
    assert (n_built, n_skipped) == (1, 0)
    assert state.mesh is None and default_state.mesh is not None
    assert (state.max_batch, state.max_batch_converged) == (
        default_state.max_batch,
        default_state.max_batch_converged,
    )
    assert torch.equal(grad, default_grad)
    for field in dataclasses.fields(default):
        x, y = getattr(default, field.name), getattr(skipped, field.name)
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y), field.name
        elif isinstance(x, tuple):
            assert all(map(torch.equal, x, y)), field.name
        else:
            assert x == y, field.name

    with pytest.raises(ValueError, match="build_state_mesh=False"):
        solver.integrate(f, state=state, take_gradient=take_gradient)
    # With bounds: a fresh mesh and the state's (given, so fixed) batch size.
    _, from_state, _, _, _ = call(True, state=state)
    assert torch.equal(from_state.integral, default.integral)
    assert from_state.n_start_steps == default.n_start_steps

    _, one_shot_state = integrate(
        f,
        mesh_init=a,
        mesh_final=b,
        device=device,
        max_batch=SMALL_BATCH,
        take_gradient=False,
        build_state_mesh=False,
    )
    assert one_shot_state.mesh is None


@pytest.mark.parametrize("n_steps", [1, 2])
def test_optimal_mesh_of_one_or_two_steps(n_steps):
    """One step has no pair to merge; an over-resolved pair merges into one
    step spanning both."""
    solver = make_solver_for_unit_test(atol=1e-3, rtol=1e-3)
    mesh = torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float64).unsqueeze(-1)
    mesh_opt = solver._get_optimal_mesh(
        mesh_quadratures=torch.full((n_steps, 1), 1.0 / n_steps, dtype=torch.float64),
        mesh_quadrature_errors=torch.full((n_steps, 1), 1e-12, dtype=torch.float64),
        integral=torch.tensor([1.0], dtype=torch.float64),
        mesh=mesh,
        y=torch.ones(n_steps, solver.C, 1, dtype=torch.float64),
    )
    assert torch.equal(mesh_opt, mesh[[0, -1]])


# ---------------------------------------------------------------------------
# merge_rule="predicted": the merged error predicted from the stored values
# ---------------------------------------------------------------------------


def _record(solver, edges, f):
    """Values and contributions of the subintervals between ``edges``."""
    nodes = solver._compute_nodes(edges[:-1], edges[1:])
    y = f(nodes)
    b, _ = solver.method.weights(nodes)
    return y, (edges[1:] - edges[:-1]) * (b * y).sum(1)


def _all_solvers():
    """One CPU solver per method of both registries (adaptive_heun once)."""
    return [make_solver_for_unit_test(name) for name in UNIFORM_METHOD_NAMES] + [
        adaptive_quadrature("variable", name, device="cpu", dtype=torch.float64)
        for name in VARIABLE_METHODS
        if name not in UNIFORM_METHOD_NAMES
    ]


@pytest.mark.parametrize("rho", [1.0, 2.0, 0.5, 0.37, 1e3, 1e-3])
def test_merged_pair_weights_exact_for_polynomials(rho):
    """Interpolating each neighbor through its U distinct nodes is exact for a
    polynomial of degree U - 1, so the predicted merged integral and error
    estimate, from the values at the neighbors' C nodes, equal the merged
    subinterval's own, to rounding, for every method and width ratio."""
    generator = torch.Generator().manual_seed(1)
    a, b = 0.2, 1.3
    m = a + (b - a) / (1.0 + rho)
    for solver in _all_solvers():
        # Degree U - 1, U counted independently of the library's own count
        n_distinct = len(torch.unique(solver.method.node_fractions()))
        coefficients = torch.randn(n_distinct, generator=generator, dtype=torch.float64)

        def poly(t, coefficients=coefficients):
            x = (t - 0.75) / 1.1
            return sum(c * x**k for k, c in enumerate(coefficients))

        edges = torch.tensor([[a], [m], [b]], dtype=torch.float64)
        halves = solver._compute_nodes(edges[:-1], edges[1:])  # [2, C, 1]
        weights = solver._merged_pair_weights(torch.tensor([rho], dtype=torch.float64))
        y = poly(halves[..., 0])  # [2, C]
        predicted = (b - a) * (weights[0, 0] @ y[0] + weights[0, 1] @ y[1])  # [2]

        merged = solver._compute_nodes(edges[:1], edges[2:])
        w, w_err = solver.method.weights(merged)
        values = poly(merged[0, :, 0])
        exact = (b - a) * torch.stack(
            [(w[0, :, 0] * values).sum(), (w_err[0, :, 0] * values).sum()]
        )
        scale = (b - a) * float((w[0, :, 0].abs() * values.abs()).sum())
        assert torch.allclose(predicted, exact, rtol=0, atol=1e-12 * scale), (
            solver.method_name,
            rho,
        )


def test_predicted_with_a_repeated_node_fraction(monkeypatch):
    """A method may repeat a node fraction (its values there are equal): the
    vectors give the repeat weight 0 and apply to the values as stored.
    Simpson's rule with its middle node written twice (weights 1/3 + 1/3)
    predicts the same merged errors as Simpson's rule, to rounding."""
    third, sixth = 1.0 / 3.0, 1.0 / 6.0
    rules = {
        "simpson": ([0.0, 0.5, 1.0], [sixth, 4 * sixth, sixth], [2 * third]),
        "simpson_twice": (
            [0.0, 0.5, 0.5, 1.0],
            [sixth, third, third, sixth],
            [third] * 2,
        ),
    }
    edges = torch.tensor([[0.0], [0.2], [0.45], [0.5], [1.0]], dtype=torch.float64)
    predicted = {}
    for name, (c, b, middle) in rules.items():
        c, b = (
            torch.tensor(c, dtype=torch.float64),
            torch.tensor(b, dtype=torch.float64),
        )
        # Simpson minus the trapezoidal rule
        b_error = torch.tensor([-third, *middle, -third], dtype=torch.float64)
        method = UniformMethodClass(
            degree=3, error_exponent=3, tableau=_Tableau(c=c, b=b, b_error=b_error)
        )
        monkeypatch.setitem(UNIFORM_METHODS, name, method)
        solver = make_solver_for_unit_test(name)
        y, q = _record(solver, edges, lambda t: torch.exp(3 * t) * torch.sin(7 * t))
        predicted[name] = solver._predict_merged_errors(y, q, edges)
    weights = solver._merged_pair_weights(torch.tensor([1.0, 2.5], dtype=torch.float64))
    assert torch.all(weights[..., 2] == 0)  # the repeat
    assert torch.allclose(
        predicted["simpson_twice"], predicted["simpson"], rtol=1e-13, atol=0
    )


def _lorentz_5(t):
    return 1.0 / (1.0 + 25.0 * (t - 0.5) ** 2)


@pytest.mark.parametrize(
    ("merge_rule", "merged"),
    [("predicted", True), ("sum", True), ("power_law", False)],
)
def test_preasymptotic_pair_merges(merge_rule, merged):
    """gk21 on 1/(1 + 25 (t - 1/2)^2) from [0, 1/2, 1] at atol 1e-2 (2.1 % of
    the integral 0.4761): both halves' estimates are 3.2e-9; the merged
    subinterval's estimate is 3.5e-4 and its true error 9.9e-8, so it passes
    easily. "power_law" predicts 2**20 times the halves' errors, a ratio of
    0.66, and refuses the merge; "predicted" merges it."""
    solver = make_uniform_solver(
        "gk21",
        atol=1e-2,
        rtol=1e-12,
        device="cpu",
        merge_rule=merge_rule,
        max_batch=SMALL_BATCH,
    )
    mesh = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    result, result_state = solver.integrate(
        f=_lorentz_5, mesh=mesh, take_gradient=False
    )
    expected = mesh[[0, -1]] if merged else mesh
    assert torch.equal(result_state.mesh.cpu(), expected)


def _warm_errors(solver, f, exact, mesh, n_warm):
    """Errors of a cold call from ``mesh`` and of ``n_warm`` warm calls, each
    from the previous call's mesh_optimal and max_batch, with the right-hand
    side of the warm-start check for each warm call: max(tol, 1.05 * cold
    error, summed rounding floor 50 eps |Q_i|), eps of the mesh's dtype."""
    eps = torch.finfo(mesh.dtype).eps
    tol = max(float(solver.atol), float(solver.rtol) * abs(exact))
    cold, cold_state = solver.integrate(f=f, mesh=mesh, take_gradient=False)
    cold_error = abs(float(cold.integral) - exact)
    rows = []
    warm_state = cold_state
    for _ in range(n_warm):
        warm, warm_state = solver.integrate(
            f=f,
            mesh=warm_state.mesh,
            max_batch=warm_state.max_batch,
            take_gradient=False,
        )
        floor = float((50 * eps * warm.mesh_quadratures.abs()).sum())
        bound = max(tol, 1.05 * cold_error, floor)
        rows.append((abs(float(warm.integral) - exact), bound, warm))
    return cold_error, rows


# 21 mesh points (float hex) captured from a seeded warm-start chain of gk31 on
# |t - 1/3|: a 4e-4-wide subinterval holds the kink, next to wide neighbors.
_KINK_MESH = [
    "0x0.0p+0", "0x1.dbb72aaaaaaacp-7", "0x1.166092aaaaaabp-3",
    "0x1.d009effffffffp-3", "0x1.5324a8aaaaaaap-2", "0x1.54d1276d55555p-2",
    "0x1.553c471e00000p-2", "0x1.55a766ceaaaaap-2", "0x1.567da63000000p-2",
    "0x1.59d6a3b555555p-2", "0x1.60889ec000000p-2", "0x1.6dec94d555555p-2",
    "0x1.88b4810000000p-2", "0x1.be44595555555p-2", "0x1.fc9e595555555p-2",
    "0x1.16e1fcaaaaaaap-1", "0x1.43a8ef5555555p-1", "0x1.cfa1815555555p-1",
    "0x1.d8a56a0000000p-1", "0x1.e1d8ceaaaaaabp-1", "0x1.0000000000000p+0",
]  # fmt: skip


@pytest.mark.parametrize("merge_rule", MERGE_RULES)
def test_kink_warm_starts_stay_accurate(merge_rule):
    """gk31 on |t - 1/3| (atol 1e-10, rtol 1e-9) from _KINK_MESH, six warm
    calls. Predicting only the merged error estimate (without the
    halves-versus-whole term) merges the kink's subinterval with ever wider
    neighbors until every merged node misses the kink: the estimate is then
    exactly 0 and the error 32.9 times the tolerance. With the term the
    merges are refused and every warm call stays at 0.09 of the tolerance."""
    third = 1.0 / 3.0
    exact = (third**2 + (1.0 - third) ** 2) / 2.0
    mesh = torch.tensor([[float.fromhex(x)] for x in _KINK_MESH], dtype=torch.float64)
    solver = make_uniform_solver(
        "gk31",
        atol=1e-10,
        rtol=1e-9,
        device="cpu",
        merge_rule=merge_rule,
        max_batch=SMALL_BATCH,
    )
    _, rows = _warm_errors(solver, lambda t: torch.abs(t - third), exact, mesh, 6)
    for error, bound, warm in rows:
        assert error <= bound, (error, bound)
        assert warm.converged is True


def test_rounding_level_warm_starts():
    """At rtol 1e-14 the tolerance of sin(200 t) (integral 130-205 times smaller
    than the sum of |contributions|) is below what float64 can certify; warm
    calls must stay within max(tol, 1.05 * cold error, summed rounding
    floor). The cancelling sin(2 pi t) (integral 0, so only the floors set
    the tolerance) is merged from 64 subintervals to at most 8 in four calls."""
    mesh = torch.linspace(0.0, 1.0, 65, dtype=torch.float64).unsqueeze(-1)
    solver = make_uniform_solver(
        "gk21", atol=0.0, rtol=1e-14, device="cpu", max_batch=SMALL_BATCH
    )
    exact = (1.0 - math.cos(200.0)) / 200.0
    _, rows = _warm_errors(solver, lambda t: torch.sin(200.0 * t), exact, mesh, 4)
    for error, bound, warm in rows:
        assert error <= bound, (error, bound)
        assert warm.converged is True

    solver = make_uniform_solver(
        "gk21", atol=0.0, rtol=1e-14, device="cpu", max_batch=SMALL_BATCH
    )
    _, rows = _warm_errors(solver, lambda t: torch.sin(2.0 * math.pi * t), 0.0, mesh, 4)
    for error, bound, _ in rows:
        assert error <= bound, (error, bound)
    assert len(rows[-1][2].h) <= 8


def test_merge_rule_option():
    """merge_rule defaults to "predicted", is checked at construction, and
    reaches the solver through adaptive_quadrature() and integrate()."""
    assert make_solver_for_unit_test().merge_rule == "predicted"
    with pytest.raises(ValueError, match="merge_rule must be one of"):
        make_solver_for_unit_test(merge_rule="bogus")
    solver = adaptive_quadrature("uniform", "gk21", device="cpu", merge_rule="sum")
    assert solver.merge_rule == "sum"
    result, result_state = integrate(
        torch.exp,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([1.0], dtype=torch.float64),
        device="cpu",
        max_batch=10_000,
        merge_rule="power_law",
    )
    assert abs(float(result.integral) - (math.e - 1.0)) < 1e-6


def test_predicted_falls_back_to_power_law_for_vector_t(caplog):
    """With T > 1 no interpolation along a subinterval is defined: the
    power-law cutoff is used and a warning says so."""
    solver = make_solver_for_unit_test("gk21")
    edges = torch.stack([torch.linspace(0, 1, 4), torch.linspace(0, 2, 4)], dim=-1)
    y = torch.ones(3, solver.C, 1, dtype=torch.float64)
    zeros = torch.zeros(3, 1, dtype=torch.float64)
    with caplog.at_level("WARNING", logger="padaquad"):
        merge_prediction, cut = solver._merge_inputs(y, zeros, edges.double())
    assert merge_prediction is None
    assert cut == REMOVE_CUT / 2 ** (solver.method.error_exponent - 1)
    assert "merge_rule='power_law'" in caplog.text


def test_predicted_detaches_and_caches():
    """The prediction carries no autograd graph (the record keeps one when
    take_gradient=False); its weight vectors are cached per width ratio, and
    the cache is emptied when the method is rebuilt (its bound:
    test_cache_eviction_gives_the_uncached_prediction)."""
    solver = make_solver_for_unit_test("gk21")
    edges = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64).unsqueeze(-1)
    theta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    y, q = _record(solver, edges, lambda t: torch.sin(theta * t))
    first = solver._predict_merged_errors(y, q, edges)
    assert not first.requires_grad
    assert len(solver._merge_weight_cache) == 2  # ratios 1 and 2
    assert torch.equal(solver._predict_merged_errors(y, q, edges), first)

    solver._set_dtype(torch.float32)
    assert solver._merge_weight_cache == {}


def test_predicted_chunking_matches_one_pass():
    """The prediction works in chunks of pairs and builds its weight vectors in
    blocks of width ratios, so memory stays bounded for any mesh. Shrinking both
    to a few entries gives the same result to rounding: the batched products
    may round differently for different batch sizes, so the check is against
    1e-13 of each pair's scale sum_i |b_i||y_i| h (the values compared are as
    small as 1e-20 where the contributions are 1e-3)."""
    edges = torch.cat(
        [
            torch.zeros(1),
            torch.rand(299, generator=torch.Generator().manual_seed(2)).sort().values,
            torch.ones(1),
        ]
    )
    edges = edges.double().unsqueeze(-1)
    for name in ("gk21", "cc65", "dopri5"):
        whole = make_solver_for_unit_test(name)
        nodes = whole._compute_nodes(edges[:-1], edges[1:])
        y = torch.stack([torch.sin(5 * nodes[..., 0]), torch.exp(nodes[..., 0])], -1)
        b, _ = whole.method.weights(nodes)
        h = edges[1:] - edges[:-1]
        q = h * (b * y).sum(1)
        scale = h * (b.abs() * y.abs()).sum(1)
        pieces = make_solver_for_unit_test(name)
        pieces._MERGE_CHUNK, pieces._RATIO_BLOCK, pieces._MERGE_CACHE_SIZE = 7, 5, 3
        difference = (
            pieces._predict_merged_errors(y, q, edges)
            - whole._predict_merged_errors(y, q, edges)
        ).abs()
        assert torch.all(difference <= 1e-13 * (scale[:-1] + scale[1:])), name


@pytest.mark.parametrize("n_outputs", [1, 4])
def test_predicted_works_in_bounded_chunks(monkeypatch, n_outputs):
    """The prediction runs on _MERGE_CHUNK // D pairs at a time (spied through
    _cached_merged_pair_weights, called once per chunk), so its float64
    temporaries have the same size for any record and any D, and the result
    equals that of one pass to rounding. Before 6f1a0dc a chunk held
    _MERGE_CHUNK pairs for any D (peak 5.0 times y at D = 256)."""
    solver = make_solver_for_unit_test("gk21")
    solver._MERGE_CHUNK = 8
    edges = torch.linspace(0.0, 1.0, 31, dtype=torch.float64).unsqueeze(-1)
    frequencies = torch.arange(1.0, n_outputs + 1.0, dtype=torch.float64)
    y, q = _record(solver, edges, lambda t: torch.sin(3 * t * frequencies))
    whole = make_solver_for_unit_test("gk21")._predict_merged_errors(y, q, edges)
    calls = []
    cached_weights = solver._cached_merged_pair_weights

    def spy(keys):
        calls.append(len(keys))
        return cached_weights(keys)

    monkeypatch.setattr(solver, "_cached_merged_pair_weights", spy)
    pieces = solver._predict_merged_errors(y, q, edges)
    assert len(calls) == math.ceil(29 / (8 // n_outputs))  # 29 pairs
    assert torch.allclose(pieces, whole, rtol=0, atol=1e-14 * float(q.abs().max()))


def test_predicted_refuses_pairs_without_a_finite_width_ratio():
    """A pair whose width ratio overflows (mesh [0, 1e-310, 1, 2]: ratio 1e310)
    or is NaN has no int64 key: converting a non-finite float to int64 is
    undefined and differs between platforms (on x86 CPUs it gives the most
    negative int64, read back as a ratio of 0). Such a pair gets an infinite
    prediction, so it is never merged; the other pairs are unaffected."""
    solver = make_solver_for_unit_test("gk21")
    edges = torch.tensor([[0.0], [1e-310], [1.0], [2.0]], dtype=torch.float64)
    y, q = _record(solver, edges, torch.exp)
    predicted = solver._predict_merged_errors(y, q, edges)
    assert torch.isposinf(predicted[0]).all()
    assert torch.isfinite(predicted[1]).all()


def test_cache_eviction_gives_the_uncached_prediction():
    """Chunks of 3 pairs bring new width ratios into a cache of 4 entries, so
    _cached_merged_pair_weights empties the full cache and rebuilds the
    chunk's ratios several times (no other test reaches that branch). The
    prediction must equal that of a solver whose cache never fills, also on a
    second pass that starts from the evicted cache, and the cache must stay
    within its bound."""
    # Widths 1, 1, 2, 2, 1, 4, 4, 0.5, 8, ...: ratios 1, 2, 1/2, 4, 1/8, ...
    widths = torch.tensor(
        [1, 1, 2, 2, 1, 4, 4, 0.5, 8, 8, 3, 3, 1, 0.25, 0.25, 5], dtype=torch.float64
    )
    edges = torch.cat([torch.zeros(1, dtype=torch.float64), widths.cumsum(0)])
    edges = (edges / edges[-1]).unsqueeze(-1)
    small = make_solver_for_unit_test("gk21")
    small._MERGE_CHUNK, small._MERGE_CACHE_SIZE = 3, 4
    y, q = _record(small, edges, lambda t: torch.sin(9 * t) + t**2)
    expected = make_solver_for_unit_test("gk21")._predict_merged_errors(y, q, edges)
    scale = float((q.abs()[:-1] + q.abs()[1:]).max())
    for _ in range(2):
        got = small._predict_merged_errors(y, q, edges)
        assert torch.allclose(got, expected, rtol=0, atol=1e-14 * scale)
        assert len(small._merge_weight_cache) <= small._MERGE_CACHE_SIZE


@pytest.mark.parametrize("error_norm", ["2", "failure_fraction"])
@pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "cumulative"])
def test_predicted_warm_starts_with_vector_output(error_norm, absolute):
    """D = 2 end to end under both error-norm families and both tolerance
    references: after a cold call from the seeded random start, warm calls
    re-split no starting subinterval and stay within tolerance in every
    component. Sensitive: with merge_rule="sum", or a "failure_fraction" that
    ignored the merge prediction, gk15 re-splits 2 starting subintervals here."""
    names = ("damped_sine", "sine_squared")

    def f(t):
        return torch.cat([integrand_dict[name][0](t) for name in names], dim=-1)

    exact = torch.cat([integrand_dict[name][1](T_INIT, T_FINAL) for name in names])
    solver = make_uniform_solver(
        "gk15",
        atol=1e-9,
        rtol=1e-9,
        device="cpu",
        error_norm=error_norm,
        use_absolute_error_ratio=absolute,
        max_batch=SMALL_BATCH,
    )
    torch.manual_seed(0)
    warm, warm_state = solver.integrate(
        f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False
    )
    for _ in range(3):
        warm, warm_state = solver.integrate(
            f=f,
            mesh=warm_state.mesh,
            max_batch=warm_state.max_batch,
            take_gradient=False,
        )
        assert warm.n_start_steps_split == 0
        assert warm.converged is True
        error = (warm.integral.cpu() - exact).abs()
        assert torch.all(error <= 1e-9 + 1e-9 * exact.abs()), error


def test_variable_solver_predicted_warm_chain():
    """The variable solver (interpolatory3_variable, whose weights
    VariableMethodClass.weights recomputes from the merged nodes) merges
    under "predicted": from 512 equal subintervals, three warm calls shrink
    the mesh to at most 128 without re-splitting a starting subinterval, and
    pass the warm-start check."""
    solver = adaptive_quadrature(
        "variable",
        "interpolatory3_variable",
        atol=1e-6,
        rtol=1e-6,
        device="cpu",
        dtype=torch.float64,
        max_batch=SMALL_BATCH,
    )
    mesh = torch.linspace(0.0, 1.0, 513, dtype=torch.float64).unsqueeze(-1)
    _, rows = _warm_errors(solver, torch.exp, math.e - 1, mesh, 3)
    for error, bound, warm in rows:
        assert error <= bound, (error, bound)
        assert warm.n_start_steps_split == 0
    assert len(rows[-1][2].h) <= 128


def test_float32_predicted_merges_where_power_law_does_not():
    """In float32 the error estimates of gk21 sit at the rounding level and do
    not shrink with the width, so "power_law" (cutoff remove_cut / 2**20)
    keeps at least 64 of 128 subintervals over six warm calls, while
    "predicted" merges down to at most 4; both pass the warm-start check."""
    mesh = torch.linspace(0.0, 1.0, 129, dtype=torch.float32).unsqueeze(-1)
    sizes = {}
    for merge_rule in ("predicted", "power_law"):
        solver = make_uniform_solver(
            "gk21",
            atol=1e-6,
            rtol=1e-6,
            dtype=torch.float32,
            device="cpu",
            merge_rule=merge_rule,
            max_batch=SMALL_BATCH,
        )
        _, rows = _warm_errors(solver, torch.exp, math.e - 1, mesh, 6)
        for error, bound, _ in rows:
            assert error <= bound, (merge_rule, error, bound)
        sizes[merge_rule] = len(rows[-1][2].h)
    assert sizes["predicted"] <= 4
    assert sizes["power_law"] >= 64


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("offset", [-1e3, 1e3])
def test_predicted_far_from_the_origin(dtype, offset):
    """On [offset, offset + 1/2], widths far below |t| on either side of 0:
    four warm calls merge 64 subintervals to fewer than 16 and pass the
    warm-start check. The warm calls pass no
    bounds and stay on [offset, offset + 1/2]."""
    a, b = offset, offset + 0.5
    mesh = torch.linspace(a, b, 65, dtype=dtype).unsqueeze(-1)
    solver = make_uniform_solver(
        "gk21", atol=1e-7, rtol=1e-6, dtype=dtype, device="cpu", max_batch=SMALL_BATCH
    )
    _, rows = _warm_errors(
        solver, lambda t: torch.cos(3 * (t - offset)), math.sin(1.5) / 3, mesh, 4
    )
    for error, bound, warm in rows:
        assert error <= bound, (error, bound)
        assert torch.equal(warm.mesh_init.cpu(), mesh[0])
        assert torch.equal(warm.mesh_final.cpu(), mesh[-1])
    assert len(rows[-1][2].h) < 16


def test_nan_integrand_does_not_break_the_prediction():
    """With runtime_checks=False a NaN integrand value makes the prediction
    NaN only in its own output component, for the pairs that touch it; the
    call returns a valid increasing mesh (a NaN pair ratio is never below the
    cutoff, so such a pair is not merged)."""
    solver = make_uniform_solver(
        "gk21",
        atol=1e-9,
        rtol=1e-9,
        device="cpu",
        max_batch=SMALL_BATCH,
        runtime_checks=False,
    )

    def f(t):
        inside = (t > 0.5) & (t < 0.51)
        return torch.cat(
            [torch.sin(t), torch.where(inside, float("nan"), torch.cos(t))], dim=-1
        )

    mesh = torch.linspace(0.0, 1.0, 33, dtype=torch.float64).unsqueeze(-1)
    optimal = solver.integrate(f=f, mesh=mesh, take_gradient=False)[1].mesh
    optimal = optimal.cpu()[:, 0]
    assert torch.isfinite(optimal).all()
    assert torch.all(optimal[1:] > optimal[:-1])
    predicted = solver._predict_merged_errors(*_record(solver, mesh, f), mesh)
    assert torch.isnan(predicted[:, 1]).any()
    assert torch.isfinite(predicted[:, 0]).all()


@needs_cuda
def test_predicted_on_gpu_with_either_result_device():
    """A CUDA solver builds mesh_optimal on result_device. One solver alternates
    between "cuda" and "cpu" with warm starts: the weight cache must
    follow the device it is used on, and the meshes must have the sizes of a
    CPU solver's, with no re-split and the integral within tolerance."""

    def f(t):
        return torch.exp(-t) * torch.sin(5 * t)

    exact = (5 - math.exp(-1) * (math.sin(5) + 5 * math.cos(5))) / 26
    mesh = torch.linspace(0.0, 1.0, 129, dtype=torch.float64).unsqueeze(-1)
    gpu, cpu = (
        make_uniform_solver(
            "gk21", atol=1e-9, rtol=1e-9, device=device, max_batch=SMALL_BATCH
        )
        for device in ("cuda", "cpu")
    )
    out_gpu, out_gpu_state = gpu.integrate(f=f, mesh=mesh, take_gradient=False)
    out_cpu, out_cpu_state = cpu.integrate(f=f, mesh=mesh, take_gradient=False)
    for result_device in ("cuda", "cpu", "cuda"):
        out_gpu, out_gpu_state = gpu.integrate(
            f=f,
            mesh=out_gpu_state.mesh,
            max_batch=out_gpu_state.max_batch,
            take_gradient=False,
            result_device=result_device,
        )
        out_cpu, out_cpu_state = cpu.integrate(
            f=f,
            mesh=out_cpu_state.mesh,
            max_batch=out_cpu_state.max_batch,
            take_gradient=False,
        )
        cached = list(gpu._merge_weight_cache.values())
        assert cached
        assert all(v.device.type == result_device for v in cached)
        assert out_gpu_state.mesh.device.type == "cuda"
        assert len(out_gpu_state.mesh) == len(out_cpu_state.mesh)
        assert out_gpu.n_start_steps_split == 0
        assert abs(float(out_gpu.integral) - exact) <= 1e-8
