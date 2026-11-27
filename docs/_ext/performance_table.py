"""
Sphinx extension: render the measured tables of the documentation from results files.

The numbers on the Performance page come only from the files
``docs/performance/results_<device>.json`` (for example ``results_cpu.json``,
``results_cuda.json``), which ``examples/speed_test/library_comparison.py``
writes; those of the user guide's memory options from
``docs/performance/memory_options.json``, which
``examples/speed_test/trade_speed_for_memory.py`` writes. At every build this
extension renders each file with its script's own ``render_markdown``, so the
pages show measured numbers and each table has one definition.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sphinx.application import Sphinx

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "docs" / "performance"
BENCHMARK = ROOT / "examples" / "speed_test" / "library_comparison.py"
OUTPUT = ROOT / "docs" / "reference" / "generated" / "performance_table.md"
MEMORY_RESULTS = RESULTS_DIR / "memory_options.json"
MEMORY_SCRIPT = ROOT / "examples" / "speed_test" / "trade_speed_for_memory.py"
MEMORY_OUTPUT = ROOT / "docs" / "reference" / "generated" / "memory_options_table.md"


def _load(script: Path) -> Any:
    """Import a measurement script as a module, for its ``render_markdown``."""
    spec = importlib.util.spec_from_file_location(script.stem, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(results_dir: Path = RESULTS_DIR) -> str:
    """Render each ``results_*.json`` file in ``results_dir`` as Markdown.

    One section per file, in name order, headed by the measured hardware.

    Raises ``FileNotFoundError`` when there is no results file, and
    ``ValueError`` naming the file when one cannot be read or rendered.
    """
    module = _load(BENCHMARK)
    paths = sorted(results_dir.glob("results_*.json"))
    if not paths:
        raise FileNotFoundError(f"no results_*.json file in {results_dir}")
    sections = []
    for path in paths:
        try:
            results = json.loads(path.read_text())
            meta = results["metadata"]
            heading = f"### {meta['device_name']} ({meta['device']})"
            sections.append(heading + "\n\n" + module.render_markdown(results))
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            raise ValueError(f"cannot render {path}: {error!r}") from error
    return "\n".join(sections)


def render_memory_options(path: Path = MEMORY_RESULTS) -> str:
    """Render the memory options' results file as Markdown, one table per integrand.

    Raises ``FileNotFoundError`` when the file is missing, and ``ValueError``
    naming the file when it cannot be read or rendered.
    """
    module = _load(MEMORY_SCRIPT)
    if not path.exists():
        raise FileNotFoundError(f"no results file {path}")
    try:
        return module.render_markdown(json.loads(path.read_text()))
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"cannot render {path}: {error!r}") from error


def _write_table(app: Sphinx) -> None:  # noqa: ARG001 (Sphinx event signature)
    for output, text in ((OUTPUT, render()), (MEMORY_OUTPUT, render_memory_options())):
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists() or output.read_text() != text:
            output.write_text(text)


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the table generation at the start of every build."""
    app.connect("builder-inited", _write_table)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
