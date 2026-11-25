"""Measure the memory options of ``integrate``, from speed to memory.

Run on an otherwise idle GPU node (never a login node)::

    python examples/speed_test/trade_speed_for_memory.py --out results.json

Three integrands are integrated with ``take_gradient=False`` and every setting
of ``SETTINGS``, which go from speed to memory:

1. ``trade_speed_for_memory=False``: a pass's values stay on the GPU until all
   its subintervals are evaluated, and are integrated at once.
2. ``trade_speed_for_memory=True`` with ``evaluation_buffer``: values are
   integrated once that many bytes wait.
3. ``trade_speed_for_memory=True`` without a buffer: after every call of f.

each with ``result_device=None`` (everything stays on the GPU) and ``"cpu"``,
and each with ``record_evaluations`` True and False (the values are deleted as
soon as they are integrated). The integrands:

- ``light``: a chirp and 50 narrow peaks, 2 outputs in float64 at rtol 1e-10:
  little memory, but many subintervals and calls of f.
- ``medium network``: a multilayer perceptron with Fourier features of t,
  1,024 hidden units and 10,000 outputs, float32.
- ``large network``: the same with 4,096 hidden units and 100,000 outputs.

For each setting the script records the GPU's peak memory above what was
allocated before the call, during the integration loop (read when each batch is
recorded, so the work after the loop does not hide it) and over the whole call;
the process's peak resident memory on the host (sampled every 5 ms); the best
wall time of ``--repeats`` runs; and the points, calls of f and subintervals.
It checks that every setting gives the same integral and subintervals, bit for
bit, prints a Markdown table and writes everything to ``--out`` as JSON.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Self

import psutil
import torch
from torch import nn

import padaquad
from padaquad import adaptive_quadrature

GB = 2**30
#: The buffer of the ``evaluation_buffer`` settings, in bytes.
BUFFER = 256 * 2**20


def _settings() -> list[dict[str, Any]]:
    """Return every setting, from speed to memory, as integrate keywords."""
    settings = []
    for record, (trade, buffer), result_device in itertools.product(
        (True, False), ((False, None), (True, BUFFER), (True, None)), (None, "cpu")
    ):
        setting: dict[str, Any] = {
            "trade_speed_for_memory": trade,
            "result_device": result_device,
            "record_evaluations": record,
        }
        if buffer is not None:
            setting["evaluation_buffer"] = buffer
        settings.append(setting)
    return settings


SETTINGS = _settings()


class FourierMLP(nn.Module):
    """A multilayer perceptron of the features sin(w t) and cos(w t).

    The 64 frequencies w run from 1 to ``max_frequency``, so the outputs
    oscillate on the scale 1 / max_frequency and need many subintervals.
    """

    def __init__(self, hidden: int, outputs: int, max_frequency: float) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.linspace(1.0, max_frequency, 64))
        self.layers = nn.Sequential(
            nn.Linear(128, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the network at points t of shape [N, 1]; shape [N, outputs]."""
        phase = t * self.frequencies
        return self.layers(torch.cat([torch.sin(phase), torch.cos(phase)], -1))


def _light(device: str) -> tuple[Any, dict[str, Any]]:
    """Return a chirp sin(10^4 pi t^2) and 50 Gaussians of width 1e-3, float64."""
    gen = torch.Generator().manual_seed(0)
    centers = torch.rand(50, generator=gen, dtype=torch.float64).to(device)

    def f(t: torch.Tensor) -> torch.Tensor:
        peaks = torch.exp(-(((t - centers) / 1e-3) ** 2)).sum(-1, keepdim=True)
        return torch.cat([torch.sin(1e4 * math.pi * t**2), peaks], -1)

    return f, {"dtype": torch.float64, "atol": 1e-12, "rtol": 1e-10, "max_batch": 2300}


def _network(hidden: int, outputs: int, device: str) -> tuple[Any, dict[str, Any]]:
    """Return a FourierMLP in float32 with fixed random weights and no gradient."""
    torch.manual_seed(0)
    net = FourierMLP(hidden, outputs, max_frequency=2e4).to(device)
    net.requires_grad_(False)  # the integral of a trained network, no autograd
    return net, {"dtype": torch.float32, "atol": 1e-6, "rtol": 1e-5, "max_batch": 4600}


EXAMPLES = {
    "light": (
        (
            "A chirp and 50 narrow peaks: 2 outputs, float64, rtol 1e-10, gk21, "
            "max_batch 2,300"
        ),
        _light,
    ),
    "medium network": (
        (
            "Fourier-feature network, 1,024 hidden units, 10,000 outputs, float32, "
            "rtol 1e-5, gk21, max_batch 4,600"
        ),
        lambda device: _network(1024, 10_000, device),
    ),
    "large network": (
        (
            "Fourier-feature network, 4,096 hidden units, 100,000 outputs, float32, "
            "rtol 1e-5, gk21, max_batch 4,600"
        ),
        lambda device: _network(4096, 100_000, device),
    ),
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


def measure(f: Any, options: dict[str, Any], setting: dict[str, Any]) -> dict[str, Any]:
    """One integration on the GPU: its peaks, time, counts and result."""
    device = "cuda"
    dtype, max_batch = options["dtype"], options["max_batch"]
    counts = {"points": 0, "calls": 0}

    def counted(t: torch.Tensor) -> torch.Tensor:
        counts["points"] += t.shape[0]
        counts["calls"] += 1
        return f(t)

    loop_peak = [0]

    def loss_fxn(r: padaquad.IntegrationResult) -> torch.Tensor:
        # Called as each batch is recorded: the peak so far covers the pass.
        loop_peak[0] = max(loop_peak[0], torch.cuda.max_memory_allocated())
        return r.integral

    solver = adaptive_quadrature(
        "uniform",
        method="gk21",
        device=device,
        atol=options["atol"],
        rtol=options["rtol"],
    )
    bounds = {
        "mesh_init": torch.tensor([0.0], dtype=dtype, device=device),
        "mesh_final": torch.tensor([1.0], dtype=dtype, device=device),
    }
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    host_before = psutil.Process().memory_info().rss
    with HostPeak() as host:
        start = time.perf_counter()
        torch.manual_seed(0)  # the starting mesh is random
        result, _ = solver.integrate(
            counted,
            take_gradient=False,
            max_batch=max_batch,
            loss_fxn=loss_fxn,
            **bounds,
            **setting,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    row = {
        "loop GB": (loop_peak[0] - base) / GB,
        "call GB": (torch.cuda.max_memory_allocated() - base) / GB,
        "host GB": (host.peak - host_before) / GB,
        "seconds": seconds,
        "points": counts["points"],
        "calls": counts["calls"],
        "subintervals": len(result.h),
        "converged": bool(result.converged),
    }
    integral, widths = result.integral.cpu(), result.h.cpu()
    del result
    torch.cuda.empty_cache()
    return row | {"integral": integral, "h": widths}


def run_example(name: str, repeats: int) -> dict[str, Any]:
    """Every setting on one integrand; the same result bit for bit is checked."""
    description, build = EXAMPLES[name]
    f, options = build("cuda")
    measure(f, options, SETTINGS[1])  # warm-up: CUDA and cuBLAS start-up
    rows = []
    reference = None
    for setting in SETTINGS:
        runs = [measure(f, options, setting) for _ in range(repeats)]
        row = min(runs, key=lambda r: r["seconds"])
        integral, widths = row.pop("integral"), row.pop("h")
        if reference is None:
            reference = (integral, widths)
        assert torch.equal(integral, reference[0]), (name, setting)
        assert torch.equal(widths, reference[1]), (name, setting)
        rows.append({"setting": {k: str(v) for k, v in setting.items()}} | row)
        print(
            name,
            setting,
            {k: row[k] for k in ("loop GB", "call GB", "host GB", "seconds")},
            flush=True,
        )
    del f
    torch.cuda.empty_cache()
    return {"description": description, "rows": rows}


def _label(setting: dict[str, str]) -> str:
    """Return a short label of a setting for the Markdown table."""
    parts = ["trade" if setting["trade_speed_for_memory"] == "True" else "no trade"]
    if "evaluation_buffer" in setting:
        parts.append(f"buffer {int(setting['evaluation_buffer']) // 2**20} MB")
    parts.append(f"result_device {setting['result_device']}")
    if setting["record_evaluations"] == "False":
        parts.append("no record")
    return ", ".join(parts)


def main() -> None:
    """Run every example; print the Markdown tables and write the JSON file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("trade_speed_for_memory.json"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--examples", nargs="*", default=list(EXAMPLES))
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()
    meta = {
        "gpu": torch.cuda.get_device_name(),
        "gpu GB": round(torch.cuda.get_device_properties(0).total_memory / GB),
        "torch": torch.__version__,
        "commit": commit,
        "date": dt.datetime.now(tz=dt.UTC).date().isoformat(),
        "buffer bytes": BUFFER,
        "repeats": args.repeats,
    }
    results = {"meta": meta, "examples": {}}
    for name in args.examples:
        results["examples"][name] = run_example(name, args.repeats)
        args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(meta))
    for name, example in results["examples"].items():
        print(f"\n### {name}: {example['description']}\n")
        print(
            "| setting | GPU peak, loop (GB) | GPU peak, call (GB) | host peak (GB) | time (s) | calls | subintervals |"
        )
        print("|---|---|---|---|---|---|---|")
        for row in example["rows"]:
            print(
                f"| {_label(row['setting'])} | {row['loop GB']:.2f} | {row['call GB']:.2f} "
                f"| {row['host GB']:.2f} | {row['seconds']:.2f} | {row['calls']} "
                f"| {row['subintervals']} |"
            )


if __name__ == "__main__":
    main()
