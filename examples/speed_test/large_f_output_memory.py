"""Measure what ``large_f_output`` saves: device memory, host memory and time.

Run on an otherwise idle GPU node (never a login node)::

    python examples/speed_test/large_f_output_memory.py            # the table
    python examples/speed_test/large_f_output_memory.py --overhead # per-call cost

The table integrates three integrands with D outputs, with ``large_f_output``
off and on (``take_gradient=False``, ``result_device="cpu"``), and reports the
GPU's peak memory above what was allocated before the call, the process's peak
resident memory on the host during the call (sampled every 5 ms), the wall time
and the number of integrand points:

- ``fixed``: ``sin(t f_d)`` on a fixed mesh of 1,000 subintervals
  (``max_adaptive_splits=0``), gk21: one pass, the values of every subinterval
  at once.
- ``adaptive``: a Lorentzian of width 1e-3 times D weights, from the default
  starting mesh, gk21, rtol 1e-6: many passes.
- ``network``: ``tanh(t W1) W2`` with 4,096 hidden units, on the fixed mesh:
  each call costs a matrix product of size [points, 4096] x [4096, D], so the
  integrand's compute and its own memory dominate.

``--overhead`` times the default path (``large_f_output=False``) on a cheap
integrand with many small calls, so the cost of integrating each call as it
returns shows; run it with the padaquad to compare first on ``PYTHONPATH``.

The integrands' imports stay inside functions so that importing this module is
cheap (the documentation build imports nothing from here, but keep it so).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Self

import psutil
import torch

import padaquad
from padaquad import adaptive_quadrature

GB = 2**30
N_FIXED = 1000


def _fixed_mesh(device: str) -> torch.Tensor:
    """Return the fixed mesh of N_FIXED equal subintervals of [0, 1], shape [M, 1]."""
    return torch.linspace(0.0, 1.0, N_FIXED + 1, dtype=torch.float64, device=device)[
        :, None
    ]


def _integrand(case: str, n_out: int, device: str):
    """Return the integrand of a case and its integrate options."""
    weights = torch.linspace(1.0, 2.0, n_out, dtype=torch.float64, device=device)
    if case == "fixed":
        return (lambda t: torch.sin(t * weights)), {
            "mesh": _fixed_mesh(device),
            "max_adaptive_splits": 0,
        }
    if case == "adaptive":
        bounds = {
            "mesh_init": torch.tensor([0.0], dtype=torch.float64, device=device),
            "mesh_final": torch.tensor([1.0], dtype=torch.float64, device=device),
        }
        return (lambda t: weights / (1.0 + ((t - 0.37) / 1e-3) ** 2)), bounds
    gen = torch.Generator(device=device).manual_seed(0)
    w1 = torch.randn(1, 4096, generator=gen, dtype=torch.float64, device=device)
    w2 = torch.randn(4096, n_out, generator=gen, dtype=torch.float64, device=device)
    return (lambda t: torch.tanh(t @ w1) @ w2 / 64.0), {
        "mesh": _fixed_mesh(device),
        "max_adaptive_splits": 0,
    }


class HostPeak:
    """Sample this process's resident memory every 5 ms; keep the largest."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.005)

    def __enter__(self) -> Self:
        """Start sampling."""
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop sampling and take a last sample."""
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def measure(case: str, n_out: int, large: bool, max_batch: int) -> dict[str, object]:
    """One integration; its device peak, host peak, time and points."""
    device = "cuda"
    f, options = _integrand(case, n_out, device)
    points = [0]

    def counted(t: torch.Tensor) -> torch.Tensor:
        points[0] += t.shape[0]
        return f(t)

    solver = adaptive_quadrature(
        "uniform", method="gk21", device=device, atol=1e-10, rtol=1e-6
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    row: dict[str, object] = {"case": case, "D": n_out, "large": large}
    host_before = psutil.Process().memory_info().rss
    try:
        with HostPeak() as host:
            start = time.perf_counter()
            torch.manual_seed(0)
            result, _ = solver.integrate(
                counted,
                take_gradient=False,
                max_batch=max_batch,
                large_f_output=large,
                result_device="cpu",
                **options,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        row |= {
            "device GB": (torch.cuda.max_memory_allocated() - base) / GB,
            "host GB": (host.peak - host_before) / GB,
            "seconds": seconds,
            "points": points[0],
            "subintervals": len(result.h),
        }
        del result
    except torch.OutOfMemoryError:
        row |= {"device GB": "out of memory"}
    torch.cuda.empty_cache()
    return row


def copy_bandwidth() -> float:
    """Measured device-to-host bandwidth for a 1 GB copy to pageable memory, GB/s."""
    x = torch.ones(GB // 8, dtype=torch.float64, device="cuda")
    x.cpu()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        x.cpu()
    torch.cuda.synchronize()
    return 3.0 / (time.perf_counter() - start)


def table(max_batch: int) -> None:
    """Print the Markdown table of every case."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()
    print(
        f"GPU: {torch.cuda.get_device_name()}, "
        f"{torch.cuda.get_device_properties(0).total_memory / GB:.0f} GB; "
        f"torch {torch.__version__}; commit {commit}; max_batch {max_batch}; "
        f"device-to-host copy {copy_bandwidth():.1f} GB/s"
    )
    print(
        "| integrand | D | large_f_output | device peak (GB) | host peak (GB) "
        "| time (s) | points | subintervals |"
    )
    print("|---|---|---|---|---|---|---|---|")
    # D = 400,000 only where few subintervals are recorded: the record keeps
    # every value on the host (1,000 gk21 subintervals would be 74 GB), and the
    # network's W2 alone would be 13 GB.
    sizes = {
        "fixed": (1_000, 10_000, 100_000),
        "adaptive": (1_000, 10_000, 100_000, 400_000),
        "network": (1_000, 10_000, 100_000),
    }
    for case, n_outs in sizes.items():
        for n_out in n_outs:
            for large in (False, True):
                row = measure(case, n_out, large, max_batch)
                cells = [
                    row.get(k, "")
                    for k in (
                        "device GB",
                        "host GB",
                        "seconds",
                        "points",
                        "subintervals",
                    )
                ]
                text = [f"{c:.2f}" if isinstance(c, float) else str(c) for c in cells]
                print(f"| {case} | {n_out:,} | {large} | " + " | ".join(text) + " |")


def overhead(repeats: int) -> None:
    """Time the default path on a cheap integrand with many small calls."""
    for device in ("cpu", "cuda"):
        times = []
        for _ in range(repeats):
            solver = adaptive_quadrature(
                "uniform", method="gk21", device=device, atol=1e-12, rtol=1e-10
            )
            bounds = {
                "mesh_init": torch.tensor([0.0], dtype=torch.float64, device=device),
                "mesh_final": torch.tensor([1.0], dtype=torch.float64, device=device),
            }
            torch.manual_seed(0)
            start = time.perf_counter()
            result, _ = solver.integrate(
                lambda t: torch.sin(200 * math.pi * t) * torch.exp(-t),
                take_gradient=False,
                max_batch=46,
                **bounds,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        times.sort()
        print(
            f"{padaquad.__file__}: {device}: {len(result.h)} subintervals, "
            f"median {times[len(times) // 2]:.3f} s, best {times[0]:.3f} s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--overhead", action="store_true")
    parser.add_argument("--max-batch", type=int, default=2300)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.overhead:
        overhead(args.repeats)
    else:
        table(args.max_batch)
