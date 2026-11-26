"""Measure the memory options of ``integrate``, from speed to memory.

Run on an otherwise idle GPU node (never a login node)::

    python examples/speed_test/trade_speed_for_memory.py

Four integrands are integrated with ``take_gradient=False`` and every setting
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
  two hidden layers of 37 units and 100 outputs (9,979 parameters), float32.
- ``large network``: the same with 583 units and 1,000 outputs (999,679
  parameters).
- ``expensive field``: ``SteppedField``, 512 x 512 outputs (1 MB per point in
  float32) from 2,000 sequential steps, so a pass's values take tens of
  GB.

No ``max_batch`` is given, except for the expensive field: PadaQuad grows each
call until f runs out of memory. The expensive field is measured instead with
``FIELD_SETTINGS``: ``max_batch`` 3,000, 4,000 and 5,550, each with
``trade_speed_for_memory`` False and True, and ``record_evaluations=False``,
to show the batch sizes at which the default runs out of memory.

For each setting the script records the GPU's peak memory above what was
allocated before the call, during the integration loop (read when each batch is
recorded, so the work after the loop does not hide it) and over the whole call;
the process's peak resident memory on the host (sampled every 5 ms); the best
wall time of ``--repeats`` runs; the final batch size (Actual Batch), the points, calls of f
and subintervals; or that the setting ran out of memory outside a call of f.
It checks that every setting gives the same integral and subintervals, bit for
bit, writes everything to ``--out`` as JSON (by default
``docs/performance/memory_options.json``, which the user guide's section on
these options renders at every documentation build with ``render_markdown``)
and prints the same Markdown.
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
#: The JSON file the user guide renders (``docs/_ext/performance_table.py``).
DOCS_RESULTS = (
    Path(__file__).resolve().parents[2] / "docs" / "performance" / "memory_options.json"
)
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
#: The expensive field's settings (see the module docstring).
FIELD_SETTINGS = [
    {
        "max_batch": max_batch,
        "trade_speed_for_memory": trade,
        "record_evaluations": False,
    }
    for max_batch in (3000, 4000, 5550)
    for trade in (False, True)
]
#: The settings of each example that does not use ``SETTINGS``.
EXAMPLE_SETTINGS = {"expensive field": FIELD_SETTINGS}


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


class SteppedField(nn.Module):
    """A small state advanced by many sequential steps, then expanded into a field.

    The state z, of 64 numbers per point, starts at sin(w t) for 64
    frequencies w up to ``max_frequency`` and takes ``steps`` explicit Euler
    steps of dz/ds = sin(z + c) over s in [0, 1]. Each step is a few small GPU
    operations, one after another. The state is then expanded into a ``side`` x ``side`` field,
    sin(p) cos(p) with p = u v^T and u, v linear in z, computed at
    ``supersample`` times the resolution and averaged down, so each point has
    side^2 outputs and the temporary arrays inside a call are about
    4 supersample^2 times its output.
    """

    def __init__(
        self, steps: int, side: int, max_frequency: float, supersample: int = 1
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.steps = steps
        self.register_buffer("frequencies", torch.linspace(1.0, max_frequency, 64))
        self.register_buffer("phases", 2 * math.pi * torch.rand(64, generator=gen))
        self.supersample = supersample
        fine = side * supersample
        self.register_buffer("u", torch.randn(64, fine, generator=gen) / 8)
        self.register_buffer("v", torch.randn(64, fine, generator=gen) / 8)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the field at points t of shape [N, 1]; shape [N, side^2]."""
        z = torch.sin(t * self.frequencies)
        for _ in range(self.steps):
            z = z + torch.sin(z + self.phases) / self.steps
        p = (z @ self.u)[:, :, None] * (z @ self.v)[:, None, :]
        field = torch.sin(p) * torch.cos(p)
        if self.supersample > 1:
            field = nn.functional.avg_pool2d(field[:, None], self.supersample)[:, 0]
        return field.reshape(len(t), -1)


def _light(device: str) -> tuple[Any, dict[str, Any]]:
    """Return a chirp sin(10^4 pi t^2) and 50 Gaussians of width 1e-3, float64."""
    gen = torch.Generator().manual_seed(0)
    centers = torch.rand(50, generator=gen, dtype=torch.float64).to(device)

    def f(t: torch.Tensor) -> torch.Tensor:
        peaks = torch.exp(-(((t - centers) / 1e-3) ** 2)).sum(-1, keepdim=True)
        return torch.cat([torch.sin(1e4 * math.pi * t**2), peaks], -1)

    return f, {"dtype": torch.float64, "atol": 1e-12, "rtol": 1e-10}


def _network(hidden: int, outputs: int, device: str) -> tuple[Any, dict[str, Any]]:
    """Return a FourierMLP in float32 with fixed random weights and no gradient."""
    torch.manual_seed(0)
    net = FourierMLP(hidden, outputs, max_frequency=2e4).to(device)
    net.requires_grad_(False)  # the integral of a trained network, no autograd
    return net, {"dtype": torch.float32, "atol": 1e-6, "rtol": 1e-5}


def _field(device: str) -> tuple[Any, dict[str, Any]]:
    """Return a SteppedField of 512 x 512 outputs in float32."""
    field = SteppedField(steps=2000, side=512, max_frequency=3000.0).to(device)
    return field, {"dtype": torch.float32, "atol": 1e-6, "rtol": 1e-5}


EXAMPLES = {
    "light": (
        ("A chirp and 50 narrow peaks: 2 outputs, float64, rtol 1e-10, gk21"),
        _light,
    ),
    "medium network": (
        (
            "Fourier-feature network, two hidden layers of 37 units, 100 outputs "
            "(~10,000 parameters), float32, rtol 1e-5, gk21"
        ),
        lambda device: _network(37, 100, device),
    ),
    "large network": (
        (
            "Fourier-feature network, two hidden layers of 583 units, 1,000 outputs "
            "(~1,000,000 parameters), float32, rtol 1e-5, gk21"
        ),
        lambda device: _network(583, 1000, device),
    ),
    "expensive field": (
        (
            "A field of 512 x 512 outputs (1 MB per point) computed from a state of "
            "64 numbers after 2,000 sequential steps (`SteppedField`), float32, "
            "rtol 1e-5, gk21"
        ),
        _field,
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
    dtype = options["dtype"]
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
        try:
            # No max_batch: PadaQuad grows each call until f runs out of memory.
            result, state = solver.integrate(
                counted, take_gradient=False, loss_fxn=loss_fxn, **bounds, **setting
            )
        except torch.OutOfMemoryError:
            # Out of memory outside a call of f (which PadaQuad retries smaller).
            result = None
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    if result is None:
        torch.cuda.empty_cache()
        return {"out of memory": True, "seconds": seconds, "calls": counts["calls"]} | {
            "integral": None,
            "h": None,
        }
    row = {
        "loop GB": (loop_peak[0] - base) / GB,
        "call GB": (torch.cuda.max_memory_allocated() - base) / GB,
        "host GB": (host.peak - host_before) / GB,
        "seconds": seconds,
        "points": counts["points"],
        "calls": counts["calls"],
        "batch": state.max_batch,
        "subintervals": len(result.h),
        "converged": bool(result.converged),
    }
    integral, widths = result.integral.cpu(), result.h.cpu()
    del result
    torch.cuda.empty_cache()
    return row | {"integral": integral, "h": widths}


def run_example(name: str, repeats: int) -> dict[str, Any]:
    """Every setting on one integrand; the integrals are checked against each other.

    Each setting finds its own batch size, and a network's outputs can change in
    the last bits with the batch size (the GPU chooses its matrix-multiply
    method by size), so the integrals must agree to 10 times the tolerance;
    ``same bits`` records whether the integral and subintervals are also equal
    bit for bit to the first setting's.
    """
    description, build = EXAMPLES[name]
    f, options = build("cuda")
    settings = EXAMPLE_SETTINGS.get(name, SETTINGS)
    measure(f, options, settings[1])  # warm-up: CUDA and cuBLAS start-up
    rows = []
    reference = None
    for setting in settings:
        runs = [measure(f, options, setting) for _ in range(repeats)]
        finished = [r for r in runs if r["integral"] is not None]
        row = min(finished or runs, key=lambda r: r["seconds"])
        integral, widths = row.pop("integral"), row.pop("h")
        if integral is not None:
            if reference is None:
                reference = (integral, widths)
            assert torch.allclose(
                integral,
                reference[0],
                rtol=10 * options["rtol"],
                atol=10 * options["atol"],
            ), (name, setting)
            row["same bits"] = torch.equal(integral, reference[0]) and torch.equal(
                widths, reference[1]
            )
        rows.append({"setting": {k: str(v) for k, v in setting.items()}} | row)
        print(name, setting, row, flush=True)
    del f
    torch.cuda.empty_cache()
    return {"description": description, "rows": rows}


#: Each option's default, as written in the JSON file.
DEFAULTS = {
    "trade_speed_for_memory": "False",
    "evaluation_buffer": "None",
    "result_device": "cpu",
    "record_evaluations": "True",
}


def _memory(gb: float) -> str:
    """Format a peak given in GB, in MB below 1 GB."""
    return f"{gb * 1024:.1f} MB" if gb < 1 else f"{gb:.2f} GB"


def render_markdown(results: dict[str, Any]) -> str:
    """Render the results as Markdown: one table per example.

    Rows are ordered by ``trade_speed_for_memory``, ``evaluation_buffer``,
    ``result_device`` and ``record_evaluations``, the default value first;
    values other than the default are in bold. A setting that ran out of
    memory outside a call of f shows the calls made and the time until then.
    The examples in ``EXAMPLE_SETTINGS`` show ``max_batch`` and
    ``trade_speed_for_memory``, ordered by ``max_batch``, the points
    evaluated and one GPU peak, the whole call's; a final batch below
    ``max_batch`` (f ran out of memory and the call was retried smaller) is
    shown as "ran out of memory", in bold.
    """
    meta = results["meta"]
    lines = [
        (
            f"Measured on an {meta['gpu']} ({meta['gpu GB']} GB) with torch "
            f"{meta['torch']}, PadaQuad at commit {meta['commit']}, on {meta['date']}. "
            f"Time is the best of {meta['repeats']} runs after a warm-up."
        )
    ]
    for name, example in results["examples"].items():
        short = name in EXAMPLE_SETTINGS
        keys = ["max_batch", "trade_speed_for_memory"] if short else list(DEFAULTS)
        peaks = (
            [("GPU peak", "call GB")]
            if short
            else [
                ("GPU peak, loop", "loop GB"),
                ("GPU peak, call", "call GB"),
                ("host peak", "host GB"),
            ]
        )
        counts = ["Actual Batch", "calls"] + (["points"] if short else [])
        rows = sorted(
            example["rows"],
            key=lambda r: (
                [int(r["setting"].get("max_batch", 0))]
                + [r["setting"].get(k, "None") != v for k, v in DEFAULTS.items()]
            ),
        )
        finished = [r for r in rows if "subintervals" in r]
        lines += [
            "",
            f"#### {name.capitalize()}",
            "",
            f"{example['description']}."
            + (f" {finished[0]['subintervals']:,} subintervals." if finished else ""),
            "",
            "| "
            + " | ".join([f"`{k}`" for k in keys] + counts)
            + " | "
            + " | ".join([label for label, _ in peaks] + ["time"])
            + " |",
            "|" + "---|" * (len(keys) + len(counts) + len(peaks) + 1),
        ]
        for row in rows:
            cells = []
            for key in keys:
                value = row["setting"].get(key, "None")
                default = DEFAULTS.get(key, value)  # max_batch has no default
                if key == "evaluation_buffer":
                    if row["setting"]["trade_speed_for_memory"] == "False":
                        cells.append("not used")
                        continue
                    shown = "None" if value == "None" else f"{int(value) // 2**20} MB"
                elif key == "max_batch":
                    shown = f"{int(value):,}"
                else:
                    shown = f'"{value}"' if value == "cpu" else value
                cells.append(f"`{shown}`" if value == default else f"**`{shown}`**")
            if row.get("out of memory"):
                cells += ["", str(row["calls"])] + [""] * (len(counts) - 2)
                cells += ["out of memory"] + [""] * (len(peaks) - 1)
            else:
                batch = f"{row['batch']:,}"
                if row["batch"] < int(row["setting"].get("max_batch", row["batch"])):
                    # f ran out of memory and the call was retried smaller.
                    batch = "**ran out of memory**"
                cells += [batch, str(row["calls"])]
                cells += [f"{row['points']:,}"] if short else []
                cells += [_memory(row[key]) for _, key in peaks]
            cells.append(f"{row['seconds']:.2f} s")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run every example; print the Markdown tables and write the JSON file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DOCS_RESULTS)
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
    print(render_markdown(results))


if __name__ == "__main__":
    main()
