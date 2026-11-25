"""The measured tables of the documentation are rendered from the committed results files.

``docs/_ext/performance_table.py`` renders one Performance section per
``docs/performance/results_*.json`` file, and the user guide's memory options
tables from ``docs/performance/memory_options.json``. These tests run it
without Sphinx on copies of the committed files: none, one, two, and malformed
files.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
COMMITTED = DOCS / "performance" / "results_cpu.json"


MEMORY_COMMITTED = DOCS / "performance" / "memory_options.json"


def _extension():
    spec = importlib.util.spec_from_file_location(
        "performance_table", DOCS / "_ext" / "performance_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(results_dir: Path) -> str:
    return _extension().render(results_dir)


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


def test_memory_options_render_every_setting_of_every_integrand() -> None:
    results = json.loads(MEMORY_COMMITTED.read_text())
    text = _extension().render_memory_options(MEMORY_COMMITTED)
    sections = text.split("\n#### ")[1:]
    assert len(sections) == len(results["examples"])
    for section, example in zip(sections, results["examples"].values(), strict=True):
        lines = section.splitlines()
        rows = [line for line in lines[lines.index("|" + "---|" * 8) + 1 :] if line]
        assert len(rows) == len(example["rows"]) == 12
        # The defaults come first; every later row differs in a bold value.
        assert "**" not in rows[0]
        assert all("**" in row for row in rows[1:])


def test_memory_options_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="memory_options"):
        _extension().render_memory_options(tmp_path / "memory_options.json")


@pytest.mark.parametrize("content", ["{not json", json.dumps({"meta": {}})])
def test_memory_options_unreadable_file_is_named(tmp_path: Path, content: str) -> None:
    path = tmp_path / "memory_options.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=r"memory_options\.json"):
        _extension().render_memory_options(path)
