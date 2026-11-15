"""The Performance page's tables are rendered from the committed results files.

``docs/_ext/performance_table.py`` renders one section per
``docs/performance/results_*.json`` file. These tests run it without Sphinx on
copies of the committed file: none, one, two, and malformed files.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
COMMITTED = DOCS / "performance" / "results_cpu.json"


def _render(results_dir: Path) -> str:
    spec = importlib.util.spec_from_file_location(
        "performance_table", DOCS / "_ext" / "performance_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render(results_dir)


def test_no_results_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="results_"):
        _render(tmp_path)


def test_one_file_is_headed_by_its_hardware(tmp_path: Path) -> None:
    shutil.copy(COMMITTED, tmp_path / "results_cpu.json")
    meta = json.loads(COMMITTED.read_text())["metadata"]
    text = _render(tmp_path)
    assert text.startswith(f"### {meta['device_name']} ({meta['device']})\n")
    assert text.count("### ") == 1


def test_two_files_render_in_name_order(tmp_path: Path) -> None:
    shutil.copy(COMMITTED, tmp_path / "results_cpu.json")
    results = json.loads(COMMITTED.read_text())
    results["metadata"].update(device="cuda", device_name="Test GPU")
    (tmp_path / "results_cuda.json").write_text(json.dumps(results))
    text = _render(tmp_path)
    assert text.count("### ") == 2
    assert text.index("(cpu)\n") < text.index("### Test GPU (cuda)\n")


@pytest.mark.parametrize("content", ["{not json", json.dumps({"rows": []})])
def test_unreadable_file_is_named(tmp_path: Path, content: str) -> None:
    (tmp_path / "results_bad.json").write_text(content)
    with pytest.raises(ValueError, match=r"results_bad\.json"):
        _render(tmp_path)
