"""
Benchmark padaquad against torchdiffeq, diffrax, scipy and quadax.

Every library integrates the same integrands over [0, 1], each with a known
integral, at one tolerance: ``atol = 1e-8`` and ``rtol = 0`` (the
Performance page explains the choice), with the error of a vector-valued
integrand measured by the largest component (the max norm) everywhere.

- torchdiffeq ``dopri5`` and diffrax ``Dopri5`` compute the integral by
  solving the ODE dy/dt = f(t), y(0) = 0, one step after another.
- scipy ``quad`` (QUADPACK; ``quad_vec`` for a vector-valued integrand) and
  quadax ``quadgk`` (21 nodes) split [0, 1] adaptively, one subinterval at a
  time.
- padaquad (``dopri5`` and ``gk21``) evaluates the nodes of many subintervals
  in one call of the integrand.

scipy evaluates NumPy versions of the integrands, always on the CPU; diffrax
and quadax evaluate JAX versions; torchdiffeq and padaquad evaluate torch
versions. All come from one definition per integrand (``padaquad.examples``,
whose integrands take the array module ``xp``).

For every (integrand, solver) the script records the median wall time of
``--repeats`` runs after one untimed warm-up run (which also compiles the JAX
solvers), the smallest and largest time, the number of points at which the
integrand was evaluated, whether the solver reported success, and the error
against the exact answer. padaquad runs with a fixed ``max_batch``, chosen
before any timing, so no batch-size search is timed.

Run on an otherwise idle compute node through ``run_benchmark.sh`` (it sets
the JAX environment and writes ``docs/performance/results_<device>.json``)::

    examples/speed_test/run_benchmark.sh

``render_markdown(results)`` turns a results file into the tables of the
Performance page; ``docs/_ext/performance_table.py`` calls it at every
documentation build, so this module imports JAX, diffrax, quadax and
torchdiffeq only inside the functions that run them.
"""

# The optional libraries (the bench extra) are imported where they are used,
# because the documentation build imports this file without them.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import datetime as dt
import functools
import itertools
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import scipy
import scipy.integrate
import torch

import padaquad
from padaquad import hard_integrand_dict, integrand_dict

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

ATOL = 1e-8
RTOL = 0.0
# The benchmark table's rows, in order: library -> the solver it runs.
LIBRARIES = {
    "torchdiffeq": "torchdiffeq dopri5",
    "diffrax": "diffrax Dopri5",
    "scipy": "scipy quad",
    "quadax": "quadax quadgk",
}
PADAQUAD_METHODS = ("dopri5", "gk21")
# padaquad's batch size for the cheap analytic integrands; the networks' is
# measured before timing (solver.get_max_batch(f=...)).
MAX_BATCH_ANALYTIC = 100_000
# Budgets large enough that no case stops on them: scipy's subintervals
# (QUADPACK allocates them but does no work for those it does not use) and
# diffrax's steps (its default, 4096, is below what dopri5 needs for the
# oscillatory integrand at atol = 1e-8).
SCIPY_LIMIT = 10_000
DIFFRAX_MAX_STEPS = 100_000
# quadax's subinterval budget per integrand. Its loop has a fixed length of
# max_ninter + 1 iterations (no-ops after convergence), so a large budget
# costs time on every call: each value is the smallest power of 2 at least
# twice the number of subintervals quadax needed here (the "ninter" of a run).
QUADAX_MAX_NINTER = {
    "t": 2,
    "t_squared": 2,
    "sine_squared": 8,
    "exp": 2,
    "damped_sine": 4,
    "oscillatory": 128,
    "sharp_peak": 32,
    "kink": 16,
    "many_outputs": 256,
    "neural_network": 8,
    "neural_network_wide": 8,
}
QUADAX_DEFAULT_MAX_NINTER = 1024  # for an integrand not listed above


class Case(NamedTuple):
    """One integrand: how to build it for an array module, and its integral."""

    build: Callable[[ModuleType, str], Callable[[Any], Any]]
    exact: np.ndarray
    analytic: bool


def network_weights(
    widths: list[int], seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Draw the weights of a tanh network t -> R with the given layer widths.

    Parameters
    ----------
    widths : list of int
        Layer widths, input (1) to output (1).
    seed : int
        Seed of the NumPy generator.

    Returns
    -------
    list of tuple
        ``(W, b)`` per layer, float64, ``W`` scaled by 1/sqrt(fan-in).
    """
    rng = np.random.default_rng(seed)
    return [
        (rng.standard_normal((n_out, n_in)) / n_in**0.5, rng.standard_normal(n_out))
        for n_in, n_out in itertools.pairwise(widths)
    ]


def build_network(
    weights: list[tuple[np.ndarray, np.ndarray]], xp: ModuleType, device: str
) -> Callable[[Any], Any]:
    """
    Return the network as an integrand of the array module xp.

    Parameters
    ----------
    weights : list of tuple
        From ``network_weights``.
    xp : module
        ``torch``, ``numpy`` or ``jax.numpy``.
    device : str
        The torch device of the weights (ignored for NumPy and JAX).

    Returns
    -------
    Callable
        ``f(t)``: points [N, 1] to values [N, 1].
    """
    if xp is torch:
        layers = [
            (torch.as_tensor(w, device=device), torch.as_tensor(b, device=device))
            for w, b in weights
        ]
    else:
        layers = [(xp.asarray(w), xp.asarray(b)) for w, b in weights]

    def f(t: Any) -> Any:
        """
        Evaluate the network at the points t.

        Parameters
        ----------
        t : array
            Points [N, 1].

        Returns
        -------
        array
            Values [N, 1].
        """
        x = 4 * t  # stretch [0, 1] so that the network varies over it
        for i, (w, b) in enumerate(layers):
            x = x @ w.T + b
            if i < len(layers) - 1:
                x = xp.tanh(x)
        return x

    return f


def cases() -> dict[str, Case]:
    """
    Return every integrand of the benchmark, by name.

    Returns
    -------
    dict
        Name -> ``Case``: the ``integrand_dict`` and ``hard_integrand_dict``
        integrands (exact integrals from their solution functions) and two
        tanh networks (two hidden layers of 64 units, four of 512; reference
        integrals from ``scipy.integrate.quad`` at a tolerance of 1e-14).

    Raises
    ------
    RuntimeError
        If a network's reference integral is not accurate to 1e-12.
    """
    lo = torch.tensor([0.0], dtype=torch.float64)
    hi = torch.tensor([1.0], dtype=torch.float64)
    out = {}
    for name, (f, solution, _) in {**integrand_dict, **hard_integrand_dict}.items():
        exact = solution(lo, hi).reshape(-1).numpy()
        out[name] = Case(
            lambda xp, _device, f=f: functools.partial(f, xp=xp), exact, analytic=True
        )
    for name, widths in (
        ("neural_network", [1, 64, 64, 1]),
        ("neural_network_wide", [1, 512, 512, 512, 512, 1]),
    ):
        weights = network_weights(widths, seed=0)
        f_np = build_network(weights, np, "cpu")
        reference, ref_error = scipy.integrate.quad(
            lambda s, f_np=f_np: float(f_np(np.array([[s]]))[0, 0]),
            0.0,
            1.0,
            epsabs=1e-14,
            epsrel=1e-14,
            limit=1000,
        )
        if ref_error > 1e-12:
            raise RuntimeError(f"{name}: reference integral error {ref_error:.1e}")
        out[name] = Case(
            functools.partial(build_network, weights),
            np.array([reference]),
            analytic=False,
        )
    return out


class Outcome(NamedTuple):
    """What one solve returns: the integral, evaluations, success, extras."""

    value: np.ndarray
    evaluations: int
    converged: bool
    extra: dict[str, Any]


def max_norm(error: torch.Tensor) -> torch.Tensor:
    """
    Return the largest absolute component (torchdiffeq's error norm here).

    Parameters
    ----------
    error : torch.Tensor
        The scaled error of one step.

    Returns
    -------
    torch.Tensor
        Its max norm.
    """
    return error.abs().max()


def make_solver(
    library: str, case: Case, device: str, max_batch: int, max_ninter: int
) -> Callable[[], Outcome]:
    """
    Build one solve of the integral over [0, 1], ready to be called and timed.

    Everything that needs to happen once (building the integrand, compiling a
    JAX solver, counting diffrax's evaluations) happens here or in the
    untimed warm-up call.

    Parameters
    ----------
    library : str
        A key of ``LIBRARIES``, or ``"padaquad <method>"``.
    case : Case
        The integrand.
    device : str
        ``"cpu"`` or ``"cuda"`` (scipy always runs on the CPU).
    max_batch : int
        padaquad's fixed batch size.
    max_ninter : int
        quadax's subinterval budget.

    Returns
    -------
    Callable
        A function of no arguments that solves once and returns an ``Outcome``.
    """
    outputs = len(case.exact)
    if library == "scipy":
        f_np = case.build(np, "cpu")
        if outputs == 1:

            def solve() -> Outcome:
                """Integrate with scipy.integrate.quad."""
                result = scipy.integrate.quad(
                    lambda s: float(f_np(np.array([[s]]))[0, 0]),
                    0.0,
                    1.0,
                    epsabs=ATOL,
                    epsrel=RTOL,
                    limit=SCIPY_LIMIT,
                    full_output=1,
                )
                # A failure appends a message to the 3-tuple of a success.
                return Outcome(
                    np.array([result[0]]), result[2]["neval"], len(result) == 3, {}
                )

        else:

            def solve() -> Outcome:
                """Integrate with scipy.integrate.quad_vec."""
                value, _, info = scipy.integrate.quad_vec(
                    lambda s: f_np(np.array([[s]]))[0],
                    0.0,
                    1.0,
                    epsabs=ATOL,
                    epsrel=RTOL,
                    norm="max",
                    limit=SCIPY_LIMIT,
                    full_output=True,
                )
                return Outcome(np.asarray(value), info.neval, info.status == 0, {})

        return solve

    if library == "torchdiffeq":
        import torchdiffeq

        counted = CountingIntegrand(case.build(torch, device))
        y0 = torch.zeros(outputs, dtype=torch.float64, device=device)
        span = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)

        def solve() -> Outcome:
            """Integrate by solving dy/dt = f(t) with torchdiffeq."""
            counted.points = 0
            y = torchdiffeq.odeint(
                lambda t, _y: counted(t.reshape(1, 1)).reshape(outputs),
                y0,
                span,
                method="dopri5",
                atol=ATOL,
                rtol=RTOL,
                options={"norm": max_norm},
            )
            return Outcome(y[-1].cpu().numpy(), counted.points, True, {})

        return solve

    if library.startswith("padaquad"):
        method = library.split()[1]
        counted = CountingIntegrand(case.build(torch, device))
        lo = torch.tensor([0.0], dtype=torch.float64, device=device)
        hi = torch.tensor([1.0], dtype=torch.float64, device=device)

        def solve() -> Outcome:
            """Integrate with padaquad at a fixed batch size."""
            counted.points = 0
            torch.manual_seed(0)  # padaquad's random starting mesh
            result, _ = padaquad.integrate(
                counted,
                method,
                atol=ATOL,
                rtol=RTOL,
                error_norm="max",
                mesh_init=lo,
                mesh_final=hi,
                device=device,
                max_batch=max_batch,
                take_gradient=False,
            )
            value = result.integral.reshape(-1).cpu().numpy()
            return Outcome(value, counted.points, bool(result.converged), {})

        return solve

    import jax
    import jax.numpy as jnp

    f_jax = case.build(jnp, device)
    if library == "diffrax":
        import diffrax
        import optimistix as optx

        def run(count: bool) -> tuple[Any, Any, Any]:
            """Solve dy/dt = f(t); with count, count f's calls at run time."""
            calls = []

            def vector_field(t: Any, _y: Any, _args: Any) -> Any:
                """Return f(t) for the ODE solver."""
                if count:
                    jax.debug.callback(lambda: calls.append(1))
                return f_jax(jnp.reshape(t, (1, 1))).reshape(outputs)

            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(vector_field),
                diffrax.Dopri5(),
                t0=0.0,
                t1=1.0,
                dt0=None,
                y0=jnp.zeros(outputs),
                stepsize_controller=diffrax.PIDController(
                    rtol=RTOL, atol=ATOL, norm=optx.max_norm
                ),
                max_steps=DIFFRAX_MAX_STEPS,
                throw=False,
            )
            okay = diffrax.is_successful(solution.result)
            if count:
                jax.block_until_ready(solution.ys)
                return solution.ys[-1], len(calls), okay
            return solution.ys[-1], solution.stats["num_steps"], okay

        # The evaluations, counted once at run time (untimed): a Python
        # counter inside f would count JAX traces, not calls.
        _, evaluations, _ = run(count=True)
        jitted = jax.jit(functools.partial(run, False))

        def solve() -> Outcome:
            """Integrate by solving dy/dt = f(t) with diffrax."""
            value, steps, okay = jax.block_until_ready(jitted())
            return Outcome(
                np.asarray(value), evaluations, bool(okay), {"steps": int(steps)}
            )

        return solve

    if library == "quadax":
        import quadax

        @jax.jit
        def run() -> tuple[Any, Any, Any, Any]:
            """Integrate with quadax.quadgk."""
            value, info = quadax.quadgk(
                lambda x: f_jax(jnp.reshape(x, (1, 1))).reshape(outputs),
                jnp.array([0.0, 1.0]),
                epsabs=ATOL,
                epsrel=RTOL,
                max_ninter=max_ninter,
                order=21,
                norm=jnp.inf,
                full_output=True,
            )
            return value, info.neval, info.status, info.info["ninter"]

        def solve() -> Outcome:
            """Integrate with quadax at its per-integrand budget."""
            value, neval, status, ninter = jax.block_until_ready(run())
            return Outcome(
                np.asarray(value).reshape(-1),
                int(neval),
                int(status) == 0,
                {"max_ninter": max_ninter, "ninter": int(ninter)},
            )

        return solve

    raise ValueError(f"unknown library {library!r}")


class CountingIntegrand:
    """Wrap an integrand f(t) -> [N, D] and count the points it is evaluated at."""

    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.f = f
        self.points = 0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate f at t and add the number of points to the count."""
        self.points += t.shape[0]
        return self.f(t)


def synchronize(device: str) -> None:
    """Wait for queued GPU work on ``device``, so a timer measures it."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def measure(
    solve: Callable[[], Outcome], exact: np.ndarray, device: str, repeats: int
) -> dict[str, Any]:
    """
    Time one solver on one integrand: one untimed warm-up, then ``repeats`` runs.

    Parameters
    ----------
    solve : Callable
        From ``make_solver``.
    exact : numpy.ndarray
        The exact integral, shape [D].
    device : str
        The device to synchronize around each run.
    repeats : int
        Number of timed runs.

    Returns
    -------
    dict
        Median, smallest and largest time (s), evaluations, absolute and
        relative error (max norm over the outputs), convergence, extras.
    """
    solve()  # warm-up, not timed
    times, outcome = [], None
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        outcome = solve()
        synchronize(device)
        times.append(time.perf_counter() - start)
    assert outcome is not None
    abs_error = float(np.max(np.abs(outcome.value - exact)))
    return {
        "time_median_s": statistics.median(times),
        "time_min_s": min(times),
        "time_max_s": max(times),
        "evaluations": int(outcome.evaluations),
        "abs_error": abs_error,
        "rel_error": abs_error / float(np.max(np.abs(exact))),
        "converged": bool(outcome.converged),
        "error": None,
        **outcome.extra,
    }


def padaquad_max_batch(case: Case, device: str) -> int:
    """
    Return padaquad's fixed batch size for an integrand, measured before timing.

    Parameters
    ----------
    case : Case
        The integrand.
    device : str
        The torch device.

    Returns
    -------
    int
        ``MAX_BATCH_ANALYTIC`` for an analytic integrand; for a network, the
        batch size padaquad's memory benchmark gives (``get_max_batch(f=...)``).
    """
    if case.analytic:
        return MAX_BATCH_ANALYTIC
    solver = padaquad.adaptive_quadrature(
        method="gk21", atol=ATOL, rtol=RTOL, device=device
    )
    measured = solver.get_max_batch(
        f=case.build(torch, device),
        t=torch.tensor([0.0], dtype=torch.float64),
        take_gradient=False,
    )
    assert measured is not None
    return measured


def cpu_model() -> str:
    """Return the CPU model (macOS sysctl or Linux /proc/cpuinfo), else the architecture."""
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


def metadata(device: str, repeats: int) -> dict[str, Any]:
    """
    Hardware, library versions, settings, git commit and date of the run.

    Parameters
    ----------
    device : str
        ``"cpu"`` or ``"cuda"``.
    repeats : int
        Timed runs per case.

    Returns
    -------
    dict
        The metadata recorded with the results.
    """
    import diffrax
    import jax
    import jaxlib
    import optimistix
    import quadax
    import torchdiffeq

    root = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        """Run git in the repository and return its output."""
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        # Uncommitted changes to tracked files, except the results files this
        # script writes (one device's run must not mark the next as dirty).
        "git_dirty": bool(
            git(
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                ":!docs/performance",
            )
        ),
        "device": device,
        "device_name": torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else cpu_model(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "jax_devices": [str(d) for d in jax.devices()],
        "env": {
            k: os.environ.get(k)
            for k in ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS")
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "padaquad": padaquad.__version__
            if hasattr(padaquad, "__version__")
            else "source",
            "torch": torch.__version__,
            "torchdiffeq": torchdiffeq.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "diffrax": diffrax.__version__,
            "optimistix": optimistix.__version__,
            "quadax": quadax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "atol": ATOL,
        "rtol": RTOL,
        "repeats": repeats,
        "max_batch_analytic": MAX_BATCH_ANALYTIC,
        "scipy_limit": SCIPY_LIMIT,
        "diffrax_max_steps": DIFFRAX_MAX_STEPS,
        "quadax_max_ninter": QUADAX_MAX_NINTER,
    }


def _ms(seconds: float) -> str:
    """Format seconds as milliseconds with three significant digits."""
    return f"{1e3 * seconds:.3g}"


def render_markdown(results: dict[str, Any]) -> str:
    """
    Render a results file as the Performance page's tables for one device.

    For every integrand: a table of the four libraries, then a table of
    padaquad's methods with one speedup column per library (the library's
    median time over padaquad's). A solver that did not report success is
    marked "†" (and so is every speedup against it); one that raised has no
    time, and every speedup against it is "—".

    Parameters
    ----------
    results : dict
        A results file written by ``main``.

    Returns
    -------
    str
        Markdown.
    """
    meta = results["metadata"]
    versions = ", ".join(f"{k} {v}" for k, v in meta["versions"].items())
    caption = (
        f"Measured on {meta['device_name']} ({meta['device']}; CPU {meta['cpu_model']}, "
        f"{meta['torch_threads']} torch threads), {meta['date_utc']}, padaquad commit "
        f"`{meta['git_commit'][:7]}`{' (uncommitted changes)' if meta['git_dirty'] else ''}. "
        f"atol = {meta['atol']:g}, rtol = {meta['rtol']:g}. Times are medians of "
        f"{meta['repeats']} runs after one untimed warm-up run. Versions: {versions}."
    )
    lines = [caption, ""]
    rows = results["rows"]
    for name, info in results["integrands"].items():
        mine = {r["solver"]: r for r in rows if r["integrand"] == name}
        if info["outputs"] == 1:
            what = f"integral {info['largest_abs']:.6g}"
        else:
            what = (
                f"{info['outputs']} outputs, largest integral {info['largest_abs']:.3g}"
            )
        lines += [f"#### `{name}` ({what})", ""]
        lines += [
            "| Library | Device | Median time (ms) | Range (ms) | Evaluations | Abs. error | Rel. error |",
            "|---|---|---|---|---|---|---|",
        ]
        for solver in LIBRARIES.values():
            r = mine.get(solver)
            if r is None:
                continue
            if r["error"] is not None:
                lines.append(
                    f"| {solver} | {r['device']} | failed: {r['error']} | | | | |"
                )
                continue
            mark = "" if r["converged"] else " †"
            lines.append(
                f"| {solver}{mark} | {r['device']} | {_ms(r['time_median_s'])} | "
                f"{_ms(r['time_min_s'])}-{_ms(r['time_max_s'])} | {r['evaluations']} | "
                f"{r['abs_error']:.1e} | {r['rel_error']:.1e} |"
            )
        header = "| padaquad | max_batch | Median time (ms) | Evaluations | Abs. error | Rel. error | "
        header += " | ".join(f"vs {library}" for library in LIBRARIES) + " |"
        lines += ["", header, "|" + "---|" * (6 + len(LIBRARIES))]
        for method in PADAQUAD_METHODS:
            r = mine.get(f"padaquad {method}")
            if r is None:
                continue
            mark = "" if r["converged"] else " †"
            cells = [
                f"{method}{mark}",
                str(r["max_batch"]),
                _ms(r["time_median_s"]),
                str(r["evaluations"]),
                f"{r['abs_error']:.1e}",
                f"{r['rel_error']:.1e}",
            ]
            for solver in LIBRARIES.values():
                base = mine.get(solver)
                if base is None or base["error"] is not None:
                    cells.append("—")
                else:
                    ratio = base["time_median_s"] / r["time_median_s"]
                    cells.append(f"{ratio:.1f}x" + ("" if base["converged"] else " †"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n" + _summary(results)


def _summary(results: dict[str, Any]) -> str:
    """
    Summarize the tables in sentences, from the same numbers.

    Parameters
    ----------
    results : dict
        A results file written by ``main``.

    Returns
    -------
    str
        One bullet per padaquad method and library.
    """
    rows = results["rows"]
    by_key = {(r["integrand"], r["solver"]): r for r in rows}
    lines = []
    for method in PADAQUAD_METHODS:
        for library, solver in LIBRARIES.items():
            speedups, worse = [], []
            for name in results["integrands"]:
                mine, base = (
                    by_key.get((name, f"padaquad {method}")),
                    by_key.get((name, solver)),
                )
                if mine is None or base is None or base["error"] is not None:
                    continue
                speedups.append(base["time_median_s"] / mine["time_median_s"])
                if mine["abs_error"] > base["abs_error"]:
                    worse.append(name)
            if not speedups:
                continue
            accuracy = (
                "its error is at or below that library's on every integrand"
                if not worse
                else "its error is larger on " + ", ".join(f"`{n}`" for n in worse)
            )
            lines.append(
                f"- **padaquad {method} vs {library}:** speedup from "
                f"{min(speedups):.2g}x to {max(speedups):.2g}x (below 1x, padaquad "
                f"is slower); {accuracy}."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run every case, write the JSON results and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True, help="output JSON file")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    # Every core of the allocation for torch, as XLA uses them.
    torch.set_num_threads(len(os.sched_getaffinity(0)))

    import jax

    jax.config.update("jax_enable_x64", True)
    platform_name = jax.devices()[0].platform
    if (platform_name == "gpu") != (args.device == "cuda"):
        raise RuntimeError(
            f"JAX runs on {platform_name} but --device is {args.device}: set "
            "JAX_PLATFORMS (run_benchmark.sh does)."
        )

    rows, integrands = [], {}
    for name, case in cases().items():
        integrands[name] = {
            "outputs": len(case.exact),
            "largest_abs": float(np.max(np.abs(case.exact))),
        }
        max_batch = padaquad_max_batch(case, args.device)
        solvers = [*LIBRARIES, *(f"padaquad {m}" for m in PADAQUAD_METHODS)]
        for library in solvers:
            solver_name = LIBRARIES.get(library, library)
            row: dict[str, Any] = {
                "integrand": name,
                "solver": solver_name,
                "device": "cpu" if library == "scipy" else args.device,
            }
            budget = QUADAX_MAX_NINTER.get(name, QUADAX_DEFAULT_MAX_NINTER)
            try:
                solve = make_solver(library, case, args.device, max_batch, budget)
                row.update(measure(solve, case.exact, args.device, args.repeats))
            except Exception as e:
                row.update(error=f"{type(e).__name__}: {e}"[:200], converged=False)
            if library.startswith("padaquad"):
                row["max_batch"] = max_batch
            rows.append(row)
            if row["error"] is None:
                print(
                    f"{name:20s} {solver_name:20s} {1e3 * row['time_median_s']:10.2f} ms "
                    f"{row['evaluations']:9d} evals  error {row['abs_error']:.1e}"
                    f"{'' if row['converged'] else '  (not converged)'}"
                    + (f"  ninter {row['ninter']}" if "ninter" in row else ""),
                    flush=True,
                )
            else:
                print(f"{name:20s} {solver_name:20s} FAILED {row['error']}", flush=True)
    results = {
        "metadata": metadata(args.device, args.repeats),
        "integrands": integrands,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(render_markdown(results))


if __name__ == "__main__":
    main()
