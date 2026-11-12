"""
Sphinx extension: render the Performance page's table from the benchmark results.

The numbers on the Performance page come only from
``docs/performance/results.json``, which
``examples/speed_test/torchdiffeq_comparison.py`` writes. At every build this
extension renders that file with the benchmark's own ``render_markdown``, so the
page shows measured numbers and the table has one definition.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sphinx.application import Sphinx

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "performance" / "results.json"
BENCHMARK = ROOT / "examples" / "speed_test" / "torchdiffeq_comparison.py"
OUTPUT = ROOT / "docs" / "reference" / "generated" / "performance_table.md"


def render() -> str:
    """The results table as Markdown, from the committed results file."""
    spec = importlib.util.spec_from_file_location("torchdiffeq_comparison", BENCHMARK)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {BENCHMARK}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.render_markdown(json.loads(RESULTS.read_text())))


def _write_table(app: Sphinx) -> None:  # noqa: ARG001 (Sphinx event signature)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    if not OUTPUT.exists() or OUTPUT.read_text() != text:
        OUTPUT.write_text(text)


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the table generation at the start of every build."""
    app.connect("builder-inited", _write_table)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
