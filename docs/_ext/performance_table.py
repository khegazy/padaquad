"""
Sphinx extension: render the Performance page's table from the benchmark results.

The numbers on the Performance page come only from the files
``docs/performance/results_<device>.json`` (for example ``results_cpu.json``,
``results_cuda.json``), which ``examples/speed_test/torchdiffeq_comparison.py``
writes. At every build this extension renders each file, in name order, with
the benchmark's own ``render_markdown``, so the page shows measured numbers and
the table has one definition.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sphinx.application import Sphinx

ROOT = Path(__file__).resolve().parents[2]
RESULTS = sorted((ROOT / "docs" / "performance").glob("results_*.json"))
BENCHMARK = ROOT / "examples" / "speed_test" / "torchdiffeq_comparison.py"
OUTPUT = ROOT / "docs" / "reference" / "generated" / "performance_table.md"


def render() -> str:
    """Render one results section per committed results file, as Markdown."""
    spec = importlib.util.spec_from_file_location("torchdiffeq_comparison", BENCHMARK)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {BENCHMARK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not RESULTS:
        raise FileNotFoundError("no docs/performance/results_*.json file")
    sections = []
    for path in RESULTS:
        results = json.loads(path.read_text())
        device = results["metadata"]["device"]
        sections.append(f"### {device.upper()}\n\n" + module.render_markdown(results))
    return "\n".join(sections)


def _write_table(app: Sphinx) -> None:  # noqa: ARG001 (Sphinx event signature)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    if not OUTPUT.exists() or OUTPUT.read_text() != text:
        OUTPUT.write_text(text)


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the table generation at the start of every build."""
    app.connect("builder-inited", _write_table)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
