"""
Benchmark padaquad against torchdiffeq on integrals with known answers.

torchdiffeq solves an ODE: to compute an integral it solves dy/dt = f(t) with
y(a) = 0, one step after another, because in general each step needs the state
the previous one produced. padaquad uses the fact that an integrand does not
depend on y: it evaluates many nodes of many subintervals in one call of f.
The comparison shows what that buys, and what it costs, on this machine.

For every integrand and every tolerance (atol = rtol), each solver integrates
over [0, 1] several times. The script records the median wall time (one
warm-up call excluded), the spread (smallest and largest time), the number of
integrand evaluations (points at which f was evaluated) and the absolute error
against the exact answer. The two libraries apply the tolerance differently
(torchdiffeq per step against ``atol + rtol |y|``; padaquad per subinterval
against ``max(atol, rtol |I|)``), so equal tolerances do not mean equal
accuracy: compare the errors as well as the times.

Run (writes JSON; the documentation renders the table from it)::

    python examples/speed_test/torchdiffeq_comparison.py --device cpu \
        --out docs/performance/results.json

``render_markdown(results)`` turns the JSON into the table on the Performance
page.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import scipy
import scipy.integrate
import torch

import padaquad
from padaquad import integrand_dict

if TYPE_CHECKING:
    from collections.abc import Callable

TOLERANCES = (1e-6, 1e-9)
SOLVERS = ("torchdiffeq dopri5", "padaquad dopri5", "padaquad gk21")
# A fixed batch size skips padaquad's memory benchmark, which is a one-time
# cost per call without max_batch and is not what this benchmark measures.
MAX_BATCH = 100_000


class CountingIntegrand:
    """Wrap an integrand f(t) -> [N, D] and count the points it is evaluated at."""

    def __init__(self, f: Callable[..., torch.Tensor]) -> None:
        self.f = f
        self.points = 0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        self.points += t.shape[0]
        return self.f(t)


def neural_network_integrand(device: str) -> Callable[..., torch.Tensor]:
    """A small fixed-weight network t -> R: the expensive-integrand case.

    Two hidden layers of 64 tanh units, weights drawn with seed 0, float64.
    """
    generator = torch.Generator().manual_seed(0)
    sizes = [1, 64, 64, 1]
    layers = [
        (
            torch.randn(n_out, n_in, generator=generator, dtype=torch.float64).to(
                device
            )
            / n_in**0.5,
            torch.randn(n_out, generator=generator, dtype=torch.float64).to(device),
        )
        for n_in, n_out in itertools.pairwise(sizes)
    ]

    def f(t: torch.Tensor) -> torch.Tensor:
        x = 4 * t  # stretch [0, 1] so the network varies over the interval
        for i, (w, b) in enumerate(layers):
            x = x @ w.T + b
            if i < len(layers) - 1:
                x = torch.tanh(x)
        return x

    return f


def cases(
    device: str,
) -> dict[str, tuple[Callable[..., torch.Tensor], float]]:
    """Integrand name -> (integrand on [N, 1] points, exact integral over [0, 1])."""
    lo = torch.tensor([0.0], dtype=torch.float64)
    hi = torch.tensor([1.0], dtype=torch.float64)
    out: dict[str, tuple[Callable[..., torch.Tensor], float]] = {}
    for name, (f, solution, _) in integrand_dict.items():
        # integrand_dict is not typed per entry; every integrand maps [N, 1] -> [N, 1].
        out[name] = (cast("Callable[..., torch.Tensor]", f), float(solution(lo, hi)))
    net = neural_network_integrand(device)
    reference, ref_error = scipy.integrate.quad(
        lambda s: float(net(torch.tensor([[s]], dtype=torch.float64, device=device))),
        0.0,
        1.0,
        epsabs=1e-14,
        epsrel=1e-14,
        limit=500,
    )
    if ref_error > 1e-12:
        raise RuntimeError(f"reference integral not accurate enough: {ref_error:.1e}")
    out["neural_network"] = (net, reference)
    return out


def run_once(solver: str, f: CountingIntegrand, tol: float, device: str) -> float:
    """Integrate f over [0, 1] with one solver and return the integral."""
    lo = torch.tensor([0.0], dtype=torch.float64, device=device)
    hi = torch.tensor([1.0], dtype=torch.float64, device=device)
    if solver == "torchdiffeq dopri5":
        import torchdiffeq  # noqa: PLC0415 (dev dependency; the docs import this file)

        y = torchdiffeq.odeint(
            lambda t, _y: f(t.reshape(1, 1)).reshape(1),
            torch.zeros(1, dtype=torch.float64, device=device),
            torch.cat([lo, hi]),
            method="dopri5",
            atol=tol,
            rtol=tol,
        )
        return float(y[-1, 0])
    method = solver.split()[1]
    result = padaquad.integrate(
        f,
        method,
        atol=tol,
        rtol=tol,
        mesh_init=lo,
        mesh_final=hi,
        device=device,
        max_batch=MAX_BATCH,
        take_gradient=False,
    )
    return float(result.integral[0])


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def measure(
    solver: str,
    f: Callable[..., torch.Tensor],
    exact: float,
    tol: float,
    device: str,
    repeats: int,
) -> dict[str, Any]:
    """Time one (solver, integrand, tolerance) case: warm-up excluded."""
    counted = CountingIntegrand(f)
    run_once(solver, counted, tol, device)  # warm-up, not timed
    times, evaluations, value = [], 0, 0.0
    for _ in range(repeats):
        counted.points = 0
        torch.manual_seed(0)  # padaquad's random starting mesh
        synchronize(device)
        start = time.perf_counter()
        value = run_once(solver, counted, tol, device)
        synchronize(device)
        times.append(time.perf_counter() - start)
        evaluations = counted.points
    return {
        "time_median_s": statistics.median(times),
        "time_min_s": min(times),
        "time_max_s": max(times),
        "evaluations": evaluations,
        "abs_error": abs(value - exact),
    }


def cpu_model() -> str:
    """The CPU's model name (macOS sysctl or Linux /proc/cpuinfo), else the
    architecture."""
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if out:
            return out
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.machine()


def metadata(device: str) -> dict[str, Any]:
    """Hardware, library versions, git commit and date of the run."""
    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    import torchdiffeq  # noqa: PLC0415 (dev dependency; the docs import this file)

    return {
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.startswith("cuda")
            else cpu_model()
        ),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchdiffeq": torchdiffeq.__version__,
        "padaquad": padaquad.__version__
        if hasattr(padaquad, "__version__")
        else "source",
        "scipy": scipy.__version__,
        "max_batch": MAX_BATCH,
    }


def render_markdown(results: dict[str, Any]) -> str:
    """Render the results JSON as the Markdown table of the Performance page."""
    meta = results["metadata"]
    caption = (
        f"Measured on {meta['device_name']} ({meta['device']}, {meta['torch_threads']} torch "
        f"threads), {meta['date_utc']}, padaquad commit `{meta['git_commit'][:7]}`"
        f"{' (uncommitted changes)' if meta['git_dirty'] else ''}; torch {meta['torch']}, "
        f"torchdiffeq {meta['torchdiffeq']}, Python {meta['python']}. Times are medians of "
        f"{results['repeats']} runs after one warm-up run."
    )
    lines = [
        caption,
        "",
        "| Integrand | Tolerance | Solver | Median time (ms) | Range (ms) | Evaluations | Abs. error | Speedup vs torchdiffeq |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results["rows"]:
        base = next(
            r
            for r in results["rows"]
            if r["integrand"] == row["integrand"]
            and r["tolerance"] == row["tolerance"]
            and r["solver"] == "torchdiffeq dopri5"
        )
        speedup = base["time_median_s"] / row["time_median_s"]
        lines.append(
            f"| {row['integrand']} | {row['tolerance']:.0e} | {row['solver']} | "
            f"{1e3 * row['time_median_s']:.2f} | {1e3 * row['time_min_s']:.2f}-"
            f"{1e3 * row['time_max_s']:.2f} | {row['evaluations']} | {row['abs_error']:.1e} | "
            f"{speedup:.1f}x |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda[:N]'")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True, help="output JSON file")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    rows = []
    for name, (f, exact) in cases(args.device).items():
        for tol in TOLERANCES:
            for solver in SOLVERS:
                row: dict[str, Any] = {
                    "integrand": name,
                    "tolerance": tol,
                    "solver": solver,
                }
                row.update(measure(solver, f, exact, tol, args.device, args.repeats))
                rows.append(row)
                print(
                    f"{name:15s} {tol:.0e} {solver:20s} {1e3 * row['time_median_s']:9.2f} ms "
                    f"{row['evaluations']:8d} evals  error {row['abs_error']:.1e}",
                    flush=True,
                )
    results = {"metadata": metadata(args.device), "repeats": args.repeats, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(render_markdown(results))


if __name__ == "__main__":
    main()
