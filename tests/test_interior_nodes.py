"""Methods whose nodes are all inside the subinterval (the Gauss-Kronrod methods).

The integrand is evaluated only at the points that carry a weight: never at a
mesh point, also not in the start-up calls. So an integrable singularity at a
bound needs no special treatment, the result lists exactly the evaluated
points, and the width and the gradient with respect to a bound come from the
mesh points.
"""

from __future__ import annotations

import math

import pytest
import torch

from padaquad import UNIFORM_METHODS, adaptive_quadrature

GAUSS_KRONROD = ["gk7", "gk15", "gk21", "gk31"]
#: Points per subinterval, 2n + 1 for the n-point Gauss method.
POINTS = {"gk7": 7, "gk15": 15, "gk21": 21, "gk31": 31}
F64 = torch.float64
A, B = torch.tensor([0.0], dtype=F64), torch.tensor([1.0], dtype=F64)


def _solver(method, **options):
    return adaptive_quadrature("uniform", method, device="cpu", **options)


@pytest.mark.parametrize("method", GAUSS_KRONROD)
def test_nodes_are_interior_and_counted(method):
    """C = 2n + 1 nodes per subinterval, all strictly inside (0, 1)."""
    c = UNIFORM_METHODS[method].node_fractions()
    assert UNIFORM_METHODS[method].n_nodes == POINTS[method]
    assert float(c[0]) > 0.0
    assert float(c[-1]) < 1.0


@pytest.mark.parametrize("method", GAUSS_KRONROD)
@pytest.mark.parametrize("max_batch", [None, 50])
def test_f_gets_c_points_per_subinterval_and_no_mesh_point(method, max_batch):
    """Every pass calls f at C points per waiting subinterval (in one or more
    calls), and none of them is a point of the pass's mesh; nor is any point of
    the start-up calls (made without max_batch) a point of the starting mesh.
    (The centre node of a subinterval that is then split becomes a mesh point
    of the halves, which do not evaluate it.)"""
    received = []

    def f(t):
        received.append(t.detach().clone())
        return 1 / ((t - 0.37) ** 2 + 1e-4)

    solver = _solver(method, atol=1e-10, rtol=0)
    evaluate = solver._evaluate_f_on_mesh
    passes = []

    def spy(f_, f_args, mesh, mesh_trackers, *args):
        start = len(received)
        out = evaluate(f_, f_args, mesh, mesh_trackers, *args)
        points = torch.cat(received[start:])
        assert len(points) == solver.C * int(mesh_trackers.sum())
        assert not torch.isin(points, mesh.detach()).any()
        passes.append((start, mesh.detach().clone()))
        return out

    solver._evaluate_f_on_mesh = spy
    torch.manual_seed(0)
    result, _ = solver.integrate(
        f, mesh_init=A, mesh_final=B, take_gradient=False, max_batch=max_batch
    )
    assert result.converged
    assert len(passes) > 1  # several passes
    first_pass, starting_mesh = passes[0]
    # The test calls of 10, 100, ... points
    assert (first_pass > 0) == (max_batch is None)
    for t in received[:first_pass]:
        assert not torch.isin(t, starting_mesh).any()
    exact = 100 * (math.atan(63) + math.atan(37))
    assert abs(float(result.integral) - exact) < 1e-8


@pytest.mark.parametrize("method", GAUSS_KRONROD)
@pytest.mark.parametrize("max_batch", [None, 10_000])
@pytest.mark.parametrize(
    ("f", "exact"),
    [(lambda t: 1 / torch.sqrt(t), 2.0), (torch.log, -1.0)],
    ids=["inverse_sqrt", "log"],
)
def test_integrable_singularity_at_a_bound(method, max_batch, f, exact):
    """1/sqrt(t) and log(t) are infinite at t = 0, a bound. With every node
    inside its subinterval f is never called there, so runtime_checks finds
    no NaN or Inf and the integral converges to the exact value."""
    torch.manual_seed(0)
    result, _ = _solver(method, atol=1e-6, rtol=0).integrate(
        f,
        mesh_init=A,
        mesh_final=B,
        take_gradient=False,
        max_batch=max_batch,
        runtime_checks=True,
    )
    assert result.converged
    assert abs(float(result.integral) - exact) < 1e-6


@pytest.mark.parametrize("method", ["cc17", "dopri5", "interpolatory3_variable"])
def test_methods_with_end_nodes_still_raise_at_a_singular_bound(method):
    """Clenshaw-Curtis, the Runge-Kutta and the variable methods evaluate the
    ends of every subinterval, so f(0) = inf raises (runtime_checks)."""
    sampling = "variable" if method == "interpolatory3_variable" else "uniform"
    solver = adaptive_quadrature(sampling, method, device="cpu", atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="non-finite"):
        solver.integrate(
            lambda t: 1 / torch.sqrt(t),
            mesh_init=A,
            mesh_final=B,
            take_gradient=False,
            max_batch=10_000,
        )


@pytest.mark.parametrize("method", ["gk21", "gk7", "cc17", "bosh3"])
def test_result_rows_are_the_evaluated_points(method):
    """result.y and the tracked variables are f's values at result.nodes, row
    for row: N * C rows for a method without end nodes, N * (C - 1) + 1 for one
    whose neighbouring subintervals share their mesh point."""

    def f(t):
        return torch.cos(7 * t), (t.clone(),)

    torch.manual_seed(0)
    solver = _solver(method, atol=1e-9, rtol=0)
    result, _ = solver.integrate(
        f, mesh_init=A, mesh_final=B, take_gradient=False, max_batch=10_000
    )
    n, c = len(result.h), solver.C
    shared = method not in GAUSS_KRONROD
    assert len(result.nodes) == (n * (c - 1) + 1 if shared else n * c)
    assert torch.equal(result.y, torch.cos(7 * result.nodes))
    assert torch.equal(result.tracked_variables[0], result.nodes)
    assert torch.all(result.nodes[1:] >= result.nodes[:-1])


@pytest.mark.parametrize("method", GAUSS_KRONROD)
@pytest.mark.parametrize("take_gradient", [False, True])
def test_bound_gradients_match_the_closed_form(method, take_gradient):
    """The width comes from the mesh points the nodes are placed from, so a
    bound's gradient reaches the integral through both: for the integral of
    w e^t over [a, b], d/da = -w e^a, d/db = w e^b and d/dw = e^b - e^a, with
    several batches when take_gradient is True."""
    lo, hi = 0.25, 1.5
    w = torch.tensor(2.0, dtype=F64, requires_grad=True)
    a = torch.tensor([lo], dtype=F64, requires_grad=True)
    b = torch.tensor([hi], dtype=F64, requires_grad=True)
    torch.manual_seed(0)
    result, _ = _solver(method, atol=1e-11, rtol=0).integrate(
        lambda t: w * torch.exp(t),
        mesh_init=a,
        mesh_final=b,
        take_gradient=take_gradient,
        max_batch=2 * POINTS[method],
    )
    if not take_gradient:
        result.integral.sum().backward()
    assert math.isclose(float(a.grad), -2 * math.exp(lo), rel_tol=1e-10)
    assert math.isclose(float(b.grad), 2 * math.exp(hi), rel_tol=1e-10)
    assert math.isclose(float(w.grad), math.exp(hi) - math.exp(lo), rel_tol=1e-10)
