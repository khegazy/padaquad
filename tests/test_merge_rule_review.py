"""Tests proposed by the code review of merge_rule="predicted".

Each test says what it checks and why the author's tests do not already cover
it. See docs/issues/merge-rule-code-review.md for the review; the author keeps
or drops each test.
"""

from __future__ import annotations

import math

import pytest
import torch
from _helpers import T_FINAL, T_INIT, make_solver_for_unit_test, make_uniform_solver

from padaquad import adaptive_quadrature, integrand_dict

#: max_batch for the end-to-end tests: their meshes are small, and an explicit
#: value skips the memory benchmark of make_uniform_solver.
SMALL_BATCH = 100_000
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _record(solver, edges, f):
    """Nodes, values and contributions of the subintervals between ``edges``."""
    nodes = solver._compute_nodes(edges[:-1], edges[1:])
    y = f(nodes)
    b, _ = solver.method.weights(nodes)
    q = (nodes[:, -1] - nodes[:, 0]) * (b * y).sum(1)
    return nodes, y, q


def _warm_chain(solver, f, mesh, n_warm, **bounds):
    """A cold call from ``mesh`` and ``n_warm`` same_f calls; returns all results."""
    cold = solver.integrate(f=f, mesh=mesh, take_gradient=False)
    return [cold] + [
        solver.integrate(f=f, same_f=True, take_gradient=False, **bounds)
        for _ in range(n_warm)
    ]


def test_cache_eviction_gives_the_uncached_prediction():
    """The branch of _cached_merged_pair_weights that empties a full cache and
    rebuilds the chunk's ratios is never reached by the author's tests (their
    small-cache test has more ratios per chunk than the cache holds, so it
    always takes the uncached branch). Here chunks of 3 pairs bring new ratios
    into a cache of 4 entries, so it is evicted several times; the prediction
    must equal that of a solver whose cache never fills, and the cache must
    stay within its bound."""
    # Widths 1, 1, 2, 2, 1, 4, 4, 0.5, 8, ...: ratios 1, 2, 1/2, 4, 1/8, ...
    widths = torch.tensor(
        [1, 1, 2, 2, 1, 4, 4, 0.5, 8, 8, 3, 3, 1, 0.25, 0.25, 5], dtype=torch.float64
    )
    edges = torch.cat([torch.zeros(1, dtype=torch.float64), widths.cumsum(0)])
    edges = (edges / edges[-1]).unsqueeze(-1)
    small = make_solver_for_unit_test("gk21")
    small._MERGE_CHUNK, small._MERGE_CACHE_SIZE = 3, 4
    large = make_solver_for_unit_test("gk21")
    nodes, y, q = _record(small, edges, lambda t: torch.sin(9 * t) + t**2)
    expected = large._predict_merged_errors(nodes, y, q)
    scale = (q.abs()[:-1] + q.abs()[1:]).max()
    for _ in range(2):  # a second pass starts from the evicted cache
        got = small._predict_merged_errors(nodes, y, q)
        assert torch.allclose(got, expected, rtol=0, atol=1e-14 * float(scale))
        assert len(small._merge_weight_cache) <= small._MERGE_CACHE_SIZE + 1


@pytest.mark.parametrize("error_norm", ["2", "failure_fraction"])
@pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "cumulative"])
def test_predicted_warm_starts_with_vector_output(error_norm, absolute):
    """End to end with D = 2 under both error-norm families and both tolerance
    references: the author's end-to-end merge tests are D = 1 with the "2"
    norm, and the per-component pair errors of "failure_fraction" are only
    unit-tested. After a cold call from the seeded random start, the warm
    calls must re-split no starting subinterval and stay within tolerance in
    every component. Sensitive: with merge_rule="sum" (also what
    "failure_fraction" did if it ignored pair_errors) gk15 re-splits 2
    starting subintervals here in every mode."""
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
    solver.integrate(f=f, mesh_init=T_INIT, mesh_final=T_FINAL, take_gradient=False)
    for _ in range(3):
        warm = solver.integrate(f=f, same_f=True, take_gradient=False)
        assert warm.n_start_steps_split == 0
        assert warm.converged is True
        error = (warm.integral.cpu() - exact).abs()
        assert torch.all(error <= 1e-9 + 1e-9 * exact.abs()), error


def test_variable_solver_predicted_warm_chain():
    """The variable solver under "predicted" (interpolatory3_variable, whose
    weights are recomputed from the merged nodes by VariableMethodClass.weights)
    is exercised by the author's tests only through the exactness test and the
    verification script. From 512 equal subintervals the warm calls must shrink
    the mesh without re-splitting and stay within tolerance."""
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
    results = _warm_chain(solver, torch.exp, mesh, 3)
    assert len(results[-1].h) <= 128
    for warm in results[1:]:
        assert warm.n_start_steps_split == 0
        assert abs(float(warm.integral) - (math.e - 1)) <= 1e-6 * math.e


def test_float32_predicted_merges_where_power_law_does_not():
    """In float32 the high-degree error estimates sit at the rounding level, so
    "power_law" (cutoff remove_cut / 2**20 for gk21) merges almost nothing,
    while "predicted" merges down to a few subintervals; both stay within
    tolerance. Covers the float32 node check and float32 weights end to end
    in the suite's own tests (the verification script prints but does not
    compare mesh sizes)."""
    mesh = torch.linspace(0.0, 1.0, 129, dtype=torch.float32).unsqueeze(-1)
    sizes = {}
    for rule in ("predicted", "power_law"):
        solver = make_uniform_solver(
            "gk21",
            atol=1e-6,
            rtol=1e-6,
            dtype=torch.float32,
            device="cpu",
            merge_rule=rule,
            max_batch=SMALL_BATCH,
        )
        results = _warm_chain(solver, torch.exp, mesh, 6)
        for warm in results:
            assert abs(float(warm.integral) - (math.e - 1)) <= 2e-6 * math.e
        sizes[rule] = len(results[-1].h)
    assert sizes["predicted"] <= 4
    assert sizes["power_law"] >= 64


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("offset", [-1e3, 1e3])
def test_predicted_far_from_the_origin(dtype, offset):
    """Meshes far from 0, on either side, with widths far below |t|: the node
    check must accept the solver's own records (its scale is each
    subinterval's largest |node|), and warm calls must merge and stay within
    tolerance. Warm calls pass mesh_init and mesh_final: a same_f call after
    a mesh-only call otherwise integrates the solver's default interval."""
    a, b = offset, offset + 0.5
    mesh = torch.linspace(a, b, 65, dtype=dtype).unsqueeze(-1)
    bounds = {
        "mesh_init": torch.tensor([a], dtype=dtype),
        "mesh_final": torch.tensor([b], dtype=dtype),
    }
    solver = make_uniform_solver(
        "gk21",
        atol=1e-7,
        rtol=1e-6,
        dtype=dtype,
        device="cpu",
        max_batch=SMALL_BATCH,
    )

    def f(t):
        return torch.cos(3 * (t - offset))

    results = _warm_chain(solver, f, mesh, 4, **bounds)
    exact = math.sin(1.5) / 3
    assert len(results[-1].h) < 16
    for warm in results:
        assert abs(float(warm.integral) - exact) <= 1e-6 * exact + 1e-7


def test_take_gradient_does_not_change_the_merges():
    """With one batch, take_gradient=True (record values detached per batch)
    and take_gradient=False (record values keep the autograd graph, which the
    predictor must detach) see the same values, so mesh_optimal must be the
    same. No author test runs "predicted" with take_gradient=True or with a
    graph in the record end to end."""
    theta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)

    def f(t):
        return torch.sin(theta * t) + t**2

    mesh = torch.linspace(0.0, 1.0, 129, dtype=torch.float64).unsqueeze(-1)
    meshes = []
    for take_gradient in (True, False):
        solver = make_uniform_solver(
            "gk21", atol=1e-9, rtol=1e-9, device="cpu", max_batch=SMALL_BATCH
        )
        out = solver.integrate(f=f, mesh=mesh, take_gradient=take_gradient)
        meshes.append(out.mesh_optimal)
    assert len(meshes[0]) < len(mesh)
    assert torch.equal(meshes[0], meshes[1])


def test_nan_integrand_does_not_break_the_prediction():
    """With error_on_nonfinite=False a NaN integrand value makes the prediction
    NaN for the pairs that touch it. The call must return, and the mesh must
    still be a valid increasing mesh; a NaN pair ratio is never below the
    cutoff, so it is not merged."""
    solver = make_uniform_solver(
        "gk21",
        atol=1e-9,
        rtol=1e-9,
        device="cpu",
        max_batch=SMALL_BATCH,
        error_on_nonfinite=False,
    )

    def f(t):
        inside = (t > 0.5) & (t < 0.51)
        return torch.cat(
            [torch.sin(t), torch.where(inside, float("nan"), torch.cos(t))], dim=-1
        )

    mesh = torch.linspace(0.0, 1.0, 33, dtype=torch.float64).unsqueeze(-1)
    out = solver.integrate(f=f, mesh=mesh, take_gradient=False)
    optimal = out.mesh_optimal.cpu()[:, 0]
    assert torch.isfinite(optimal).all()
    assert torch.all(optimal[1:] > optimal[:-1])
    nodes, y, q = _record(solver, mesh.cpu(), f)
    predicted = solver._predict_merged_errors(nodes, y, q)
    assert torch.isnan(predicted[:, 1]).any()
    assert torch.isfinite(predicted[:, 0]).all()


@requires_cuda
def test_predicted_on_gpu_with_either_result_device():
    """A CUDA solver builds mesh_optimal on result_device: CPU by default, the
    GPU when asked. One solver alternates between the two with same_f warm
    starts; the weight cache must follow the device (it holds float64 vectors
    of the device it was last used on), and the meshes must equal those of a
    CPU solver up to the integrand's last-bit differences between devices
    (compared through the warm calls' integrals and mesh sizes)."""

    def f(t):
        return torch.exp(-t) * torch.sin(5 * t)

    exact = (5 - math.exp(-1) * (math.sin(5) + 5 * math.cos(5))) / 26
    mesh = torch.linspace(0.0, 1.0, 129, dtype=torch.float64).unsqueeze(-1)
    gpu = make_uniform_solver(
        "gk21", atol=1e-9, rtol=1e-9, device="cuda", max_batch=SMALL_BATCH
    )
    cpu = make_uniform_solver(
        "gk21", atol=1e-9, rtol=1e-9, device="cpu", max_batch=SMALL_BATCH
    )
    out_gpu = gpu.integrate(f=f, mesh=mesh, take_gradient=False)
    out_cpu = cpu.integrate(f=f, mesh=mesh, take_gradient=False)
    for result_device in ("cuda", "cpu", "cuda"):
        out_gpu = gpu.integrate(
            f=f, same_f=True, take_gradient=False, result_device=result_device
        )
        out_cpu = cpu.integrate(f=f, same_f=True, take_gradient=False)
        cached = [v for k, v in gpu._merge_weight_cache.items() if k is not None]
        assert cached
        assert all(v.device.type == torch.device(result_device).type for v in cached)
        assert out_gpu.mesh_optimal.device.type == "cuda"
        assert len(out_gpu.mesh_optimal) == len(out_cpu.mesh_optimal)
        assert out_gpu.n_start_steps_split == 0
        assert abs(float(out_gpu.integral) - exact) <= 1e-9 * 10


@requires_cuda
def test_rec_remove_on_gpu_matches_cpu():
    """The one-pass _rec_remove uses cummax, whose CUDA kernel differs from the
    CPU one; the author's comparison with the former loop runs on CPU only."""
    solver = make_solver_for_unit_test()
    generator = torch.Generator().manual_seed(0)
    for length in [0, 1, 2, 3, 17, 1000, 100_000]:
        for density in (0.2, 0.8, 1.0):
            mask = torch.rand(length, generator=generator) < density
            on_gpu = solver._rec_remove(mask.cuda())
            assert on_gpu.device.type == "cuda"
            assert torch.equal(on_gpu.cpu(), solver._rec_remove(mask))
