"""Unit tests for _prune_excess_mesh, _get_optimal_mesh and the merge rules."""

from __future__ import annotations

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
)

from padaquad import VARIABLE_METHODS, adaptive_quadrature, integrand_dict, integrate

#: The merge rules whose meshes are checked end to end: the default and the
#: 1.3.0 one ("sum" merges pairs whose merged step fails, by design).
MERGE_RULES = ["predicted", "power_law"]
#: max_batch for the end-to-end merge tests below: their meshes are small, and an
#: explicit value skips the memory benchmark of make_uniform_solver.
SMALL_BATCH = 100_000

# ---------------------------------------------------------------------------
# _prune_excess_mesh
# ---------------------------------------------------------------------------


class TestPruneExcessT:
    """Tests for AdaptiveQuadrature._prune_excess_mesh."""

    def _make_t(self, solver, t_start, t_end, N):
        """Create [N, C, 1] time tensor with uniform steps."""
        boundaries = torch.linspace(t_start, t_end, N + 1, dtype=torch.float64)
        return solver._compute_nodes(
            boundaries[:-1].unsqueeze(-1), boundaries[1:].unsqueeze(-1)
        )

    def test_no_pairs_below_cut(self):
        """All error_ratios_2steps above remove_cut: no pruning."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 4)
        ss = torch.ones(4, 1, dtype=torch.float64)
        se = torch.ones(4, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.5, 0.8, 0.3])  # All > 0.1

        t_p, _ss_p, _se_p = solver._prune_excess_mesh(t, ss, se, er2, solver.remove_cut)

        assert t_p.shape[0] == 4
        assert torch.equal(t_p, t)

    def test_one_pair_pruned(self):
        """First pair below cut: output has N-1 steps."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 4)
        ss = torch.ones(4, 1, dtype=torch.float64)
        se = torch.ones(4, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.8, 0.9])  # Pair 0 below the cut 0.1

        t_p, _, _ = solver._prune_excess_mesh(t, ss, se, er2, solver.remove_cut)

        assert t_p.shape[0] == 3

    def test_empty_error_ratios(self):
        """Single step (empty 2steps array): returns unchanged."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 1)
        ss = torch.ones(1, 1, dtype=torch.float64)
        se = torch.ones(1, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([])

        t_p, _ss_p, _se_p = solver._prune_excess_mesh(t, ss, se, er2, solver.remove_cut)

        assert t_p.shape[0] == 1
        assert torch.equal(t_p, t)

    def test_nonfinite_pair_ratio_not_merged(self):
        """A NaN or Inf pair ratio (non-finite step error) never merges."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 5)
        ss = torch.ones(5, 1, dtype=torch.float64)
        se = torch.ones(5, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([float("nan"), 0.8, float("inf"), 0.8])

        t_p, _, _ = solver._prune_excess_mesh(t, ss, se, er2, solver.remove_cut)

        assert torch.equal(t_p, t)

    def test_no_adjacent_merges(self):
        """Adjacent pairs both below cut: _rec_remove prevents double merge."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 5)
        ss = torch.ones(5, 1, dtype=torch.float64)
        se = torch.ones(5, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.01, 0.8, 0.01])  # Pairs 0,1 adjacent

        t_p, _, _ = solver._prune_excess_mesh(t, ss, se, er2, solver.remove_cut)

        # Only non-adjacent pairs merged: pair 0 and pair 3 (indices 0 and 3)
        # Pair 1 blocked by adjacency. Result: 5 - 2 = 3 steps
        assert t_p.shape[0] == 3

    def test_mesh_quadratures_accumulated(self):
        """After merge, sum of pair equals sum of originals."""
        solver = make_solver_for_unit_test()
        t = self._make_t(solver, 0.0, 1.0, 3)
        ss = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        se = torch.ones(3, 1, dtype=torch.float64) * 0.01
        er2 = torch.tensor([0.01, 0.8])  # Pair 0 below cut

        _, ss_p, _ = solver._prune_excess_mesh(t, ss, se, er2, solver.remove_cut)

        assert ss_p.shape[0] == 2
        # Merged: 1+2=3, remaining: 3
        assert torch.allclose(ss_p[0], torch.tensor([3.0], dtype=torch.float64))

    @pytest.mark.parametrize("method", ["bosh3", "dopri5", "gk21", "cc17"])
    def test_merge_inputs_per_rule(self, method):
        """_merge_inputs returns the cutoff and pair errors of each merge_rule:
        "sum" and "power_law" keep the summed errors (None) with the cutoffs
        remove_cut and remove_cut / 2**(k - 1), k = error_exponent;
        "predicted" returns pair errors of at least the summed errors with the
        cutoff remove_cut."""
        sizes = {}
        for rule in ("sum", "power_law", "predicted"):
            solver = make_solver_for_unit_test(method, merge_rule=rule)
            t = self._make_t(solver, 0.0, 1.0, 3)
            y = torch.sin(3 * t)
            b, b_err = solver.method.weights(t)
            h = t[:, -1] - t[:, 0]  # [N, 1]
            q = h * (b * y).sum(1)
            e = (h * (b_err * y).sum(1)).abs()
            sizes[rule] = solver._merge_inputs(t, y, q, e)
        k = make_solver_for_unit_test(method).method.error_exponent
        assert sizes["sum"] == (None, REMOVE_CUT)
        assert sizes["power_law"] == (None, REMOVE_CUT / 2 ** (k - 1))
        pair_errors, cut = sizes["predicted"]
        assert cut == REMOVE_CUT
        assert pair_errors.shape == (2, 1)
        assert torch.all(pair_errors >= e[:-1] + e[1:])


# ---------------------------------------------------------------------------
# _get_optimal_mesh
# ---------------------------------------------------------------------------


class TestGetOptimalTStepBarriers:
    """Tests for _get_optimal_mesh: post-convergence mesh optimization."""

    def _make_record(self, solver, N, integral_val=1.0, error_scale=0.001):
        """Build a record dict with N uniform steps in [0, 1]."""
        boundaries = torch.linspace(0, 1, N + 1, dtype=torch.float64)
        t = solver._compute_nodes(
            boundaries[:-1].unsqueeze(-1), boundaries[1:].unsqueeze(-1)
        )
        return {
            "nodes": t,
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


# ---------------------------------------------------------------------------
# Merge cutoff end to end: a merged step must pass when it is re-evaluated.
# ---------------------------------------------------------------------------


def _warm_start_passes(solver, f) -> bool:
    """Re-integrate f from the solver's stored mesh; True when no starting step
    fails, i.e. none of the stored mesh's steps had to be split."""
    out = solver.integrate(f=f, same_f=True, take_gradient=False)
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
    solver.integrate(f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False)
    assert _warm_start_passes(solver, f)


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
    solver.integrate(f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False)
    assert _warm_start_passes(solver, f)


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
    """Starting from an over-resolved 128-step mesh, repeated same_f
    calls halve the over-resolved steps (one merge level per call) until the
    mesh stops changing; no call fails a starting step and every integral
    stays within tolerance of the exact value."""
    f, solution, _ = integrand_dict["damped_sine"]
    exact = solution(T_INIT, T_FINAL)
    solver = make_uniform_solver(
        method, atol=1e-9, rtol=1e-7, device="cpu", merge_rule=merge_rule
    )
    mesh = torch.linspace(0.0, 1.0, 129, dtype=torch.float64).unsqueeze(-1)
    out = solver.integrate(f=f, mesh=mesh, take_gradient=False)
    sizes = [len(mesh) - 1, len(out.mesh_optimal) - 1]
    for _ in range(8):
        out = solver.integrate(f=f, same_f=True, take_gradient=False)
        assert out.n_start_steps_split == 0
        assert out.converged is True
        assert torch.allclose(out.integral.cpu(), exact, atol=1e-9, rtol=1e-6)
        sizes.append(len(out.mesh_optimal) - 1)
        if sizes[-1] == sizes[-2]:
            break
    assert sizes[-1] == sizes[-2], f"no fixed point: {sizes}"
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[-1] <= 0.6 * sizes[0], sizes


@pytest.mark.parametrize("n_steps", [1, 2])
def test_optimal_mesh_of_one_or_two_steps(n_steps):
    """One step has no pair to merge; an over-resolved pair merges into one
    step spanning both."""
    solver = make_solver_for_unit_test(atol=1e-3, rtol=1e-3)
    mesh = torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float64).unsqueeze(-1)
    mesh_opt = solver._get_optimal_mesh(
        nodes=solver._compute_nodes(mesh[:-1], mesh[1:]),
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


def _distinct_fractions(solver):
    """Distinct node fractions of the solver's method and their first indices."""
    c = solver.method.node_fractions().to(torch.float64)
    new = torch.ones_like(c, dtype=torch.bool)
    new[1:] = c[1:] != c[:-1]
    return c[new], torch.nonzero(new)[:, 0]


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
    estimate equal the merged subinterval's own, to rounding, for every method
    and width ratio."""
    generator = torch.Generator().manual_seed(1)
    a, b = 0.2, 1.3
    m = a + (b - a) / (1.0 + rho)
    for solver in _all_solvers():
        u, _ = _distinct_fractions(solver)
        coefficients = torch.randn(len(u), generator=generator, dtype=torch.float64)

        def poly(t, coefficients=coefficients):
            x = (t - 0.75) / 1.1
            return sum(c * x**k for k, c in enumerate(coefficients))

        edges = torch.tensor([[a], [m], [b]], dtype=torch.float64)
        halves = solver._compute_nodes(edges[:-1], edges[1:])  # [2, C, 1]
        weights, first = solver._merged_pair_weights(
            torch.tensor([rho], dtype=torch.float64)
        )
        y = poly(halves[..., 0])[:, first]  # [2, U]
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
    result = solver.integrate(f=_lorentz_5, mesh=mesh, take_gradient=False)
    expected = mesh[[0, -1]] if merged else mesh
    assert torch.equal(result.mesh_optimal.cpu(), expected)


def _warm_errors(solver, f, exact, mesh, n_warm):
    """Errors of a cold call from ``mesh`` and of ``n_warm`` same_f calls, with
    the right-hand side of the warm-start check for each warm call:
    max(tol, 1.05 * cold error, summed rounding floor 50 eps |Q_i|)."""
    eps = torch.finfo(torch.float64).eps
    tol = max(float(solver.atol), float(solver.rtol) * abs(exact))
    cold = solver.integrate(f=f, mesh=mesh, take_gradient=False)
    cold_error = abs(float(cold.integral) - exact)
    rows = []
    for _ in range(n_warm):
        warm = solver.integrate(f=f, same_f=True, take_gradient=False)
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
    result = integrate(
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
    nodes = solver._compute_nodes(edges[:-1].double(), edges[1:].double())
    y = torch.ones(3, solver.C, 1, dtype=torch.float64)
    zeros = torch.zeros(3, 1, dtype=torch.float64)
    with caplog.at_level("WARNING", logger="padaquad"):
        pair_errors, cut = solver._merge_inputs(nodes, y, zeros, zeros)
    assert pair_errors is None
    assert cut == REMOVE_CUT / 2 ** (solver.method.error_exponent - 1)
    assert "merge_rule='power_law'" in caplog.text


def test_predicted_needs_nodes_at_the_fractions():
    """The prediction assumes every node sits at the method's node fractions;
    a record that breaks this is refused rather than mispredicted."""
    solver = make_solver_for_unit_test("gk21")
    edges = torch.linspace(0, 1, 4, dtype=torch.float64).unsqueeze(-1)
    nodes = solver._compute_nodes(edges[:-1], edges[1:]).clone()
    nodes[1, 5, 0] += 1e-3
    y = torch.ones(3, solver.C, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="node fractions"):
        solver._predict_merged_errors(nodes, y, torch.ones(3, 1, dtype=torch.float64))


def test_predicted_detaches_and_caches():
    """The prediction carries no autograd graph (the record keeps one when
    take_gradient=False); its weight vectors are cached per width ratio, the
    cache is bounded, and it is emptied when the method is rebuilt."""
    solver = make_solver_for_unit_test("gk21")
    edges = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64).unsqueeze(-1)
    nodes = solver._compute_nodes(edges[:-1], edges[1:])
    theta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    y = torch.sin(theta * nodes)
    b, _ = solver.method.weights(nodes)
    q = (nodes[:, -1] - nodes[:, 0]) * (b * y).sum(1)
    first = solver._predict_merged_errors(nodes, y, q)
    assert not first.requires_grad
    assert len(solver._merge_weight_cache) == 3  # ratios 1 and 2, and the indices
    assert torch.equal(solver._predict_merged_errors(nodes, y, q), first)

    many = torch.cat(
        [torch.zeros(1), torch.rand(400, generator=torch.Generator().manual_seed(0))]
    )
    edges = torch.cat([many.sort().values, torch.ones(1)]).double().unsqueeze(-1)
    nodes = solver._compute_nodes(edges[:-1], edges[1:])
    y = torch.sin(3 * nodes)
    q = (nodes[:, -1] - nodes[:, 0]) * (solver.method.weights(nodes)[0] * y).sum(1)
    solver._predict_merged_errors(nodes, y, q)
    assert len(solver._merge_weight_cache) <= solver._MERGE_CACHE_SIZE + 1

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
        h = nodes[:, -1] - nodes[:, 0]
        q = h * (b * y).sum(1)
        scale = h * (b.abs() * y.abs()).sum(1)
        pieces = make_solver_for_unit_test(name)
        pieces._MERGE_CHUNK, pieces._RATIO_BLOCK, pieces._MERGE_CACHE_SIZE = 7, 5, 3
        difference = (
            pieces._predict_merged_errors(nodes, y, q)
            - whole._predict_merged_errors(nodes, y, q)
        ).abs()
        assert torch.all(difference <= 1e-13 * (scale[:-1] + scale[1:])), name
