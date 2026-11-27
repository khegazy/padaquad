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


def _device_headings(text: str) -> int:
    """Count the sections, one per results file (``### device``, not ``####``)."""
    return sum(line.startswith("### ") for line in text.splitlines())


def test_no_results_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="results_"):
        _render(tmp_path)


def test_one_file_is_headed_by_its_hardware(tmp_path: Path) -> None:
    shutil.copy(COMMITTED, tmp_path / "results_cpu.json")
    meta = json.loads(COMMITTED.read_text())["metadata"]
    text = _render(tmp_path)
    assert text.startswith(f"### {meta['device_name']} ({meta['device']})\n")
    assert _device_headings(text) == 1


def test_two_files_render_in_name_order(tmp_path: Path) -> None:
    shutil.copy(COMMITTED, tmp_path / "results_cpu.json")
    results = json.loads(COMMITTED.read_text())
    results["metadata"].update(device="cuda", device_name="Test GPU")
    (tmp_path / "results_cuda.json").write_text(json.dumps(results))
    text = _render(tmp_path)
    assert _device_headings(text) == 2
    assert text.index("(cpu)\n") < text.index("### Test GPU (cuda)\n")


@pytest.mark.parametrize("content", ["{not json", json.dumps({"rows": []})])
def test_unreadable_file_is_named(tmp_path: Path, content: str) -> None:
    (tmp_path / "results_bad.json").write_text(content)
    with pytest.raises(ValueError, match=r"results_bad\.json"):
        _render(tmp_path)


def _synthetic_results() -> dict:
    """A results file with known times: padaquad gk21 takes 1 ms, dopri5 2 ms;
    torchdiffeq 4 ms (timed once), diffrax 8 ms (not converged), scipy 16 ms,
    quadax raised."""
    meta = json.loads(COMMITTED.read_text())["metadata"]
    row = {
        "device": "cpu",
        "time_min_s": 0.0,
        "time_max_s": 1.0,
        "repeats": meta["repeats"],
        "evaluations": 10,
        "abs_error": 1e-9,
        "rel_error": 1e-9,
        "converged": True,
        "error": None,
    }
    rows = [
        {
            **row,
            "integrand": "f",
            "solver": "torchdiffeq dopri5",
            "time_median_s": 4e-3,
            "repeats": 1,
        },
        {
            **row,
            "integrand": "f",
            "solver": "diffrax Dopri5",
            "time_median_s": 8e-3,
            "converged": False,
        },
        {**row, "integrand": "f", "solver": "scipy quad", "time_median_s": 16e-3},
        {
            **row,
            "integrand": "f",
            "solver": "quadax quadgk",
            "converged": False,
            "error": "ValueError: boom",
        },
        {
            **row,
            "integrand": "f",
            "solver": "padaquad dopri5",
            "time_median_s": 2e-3,
            "max_batch": 7,
        },
        {
            **row,
            "integrand": "f",
            "solver": "padaquad gk21",
            "time_median_s": 1e-3,
            "max_batch": 7,
        },
    ]
    return {
        "metadata": meta,
        "integrands": {"f": {"outputs": 1, "largest_abs": 1.0}},
        "rows": rows,
    }


def test_one_benchmark_table_and_one_padaquad_table_per_integrand(
    tmp_path: Path,
) -> None:
    (tmp_path / "results_cpu.json").write_text(json.dumps(_synthetic_results()))
    text = _render(tmp_path)
    benchmark, padaquad = text.split("#### `f`")[1].split("\n\n")[1:3]
    # The benchmark table lists each library once and no padaquad method.
    for solver in (
        "torchdiffeq dopri5",
        "diffrax Dopri5 †",
        "scipy quad",
        "quadax quadgk",
    ):
        assert benchmark.count(solver) == 1
    assert "padaquad" not in benchmark.split("\n", 1)[1]
    assert "failed: ValueError: boom" in benchmark
    # A range from fewer timed runs than the others says how many.
    assert "| 0-1e+03 (1 run) |" in benchmark
    assert benchmark.count(" run") == 1
    # The padaquad table: one speedup column per library, never vs itself.
    assert padaquad.splitlines()[0].endswith(
        "| vs torchdiffeq | vs diffrax | vs scipy | vs quadax |"
    )
    gk21 = next(line for line in padaquad.splitlines() if line.startswith("| gk21"))
    dopri5 = next(line for line in padaquad.splitlines() if line.startswith("| dopri5"))
    # Speedup = the library's median time / padaquad's; † after a library that
    # did not converge; — for one that raised.
    assert gk21.endswith("| 4x | 8x † | 16x | — |")
    assert dopri5.endswith("| 2x | 4x † | 8x | — |")
    assert "| 7 |" in gk21  # the max_batch it ran with
    # PadaQuad's table has the range beside the median, like the libraries'.
    assert "| Median time (ms) | Range (ms) |" in padaquad.splitlines()[0]
    assert "| 1 | 0-1e+03 |" in gk21


def test_memory_options_render_every_setting_of_every_integrand() -> None:
    """Every integrand's table has a row per setting: the 12 of ``SETTINGS``,
    defaults first and every later row with a bold value; or, for the
    examples of ``EXAMPLE_SETTINGS``, those settings in order of max_batch."""
    extension = _extension()
    script = extension._load(extension.MEMORY_SCRIPT)
    results = json.loads(MEMORY_COMMITTED.read_text())
    text = extension.render_memory_options(MEMORY_COMMITTED)
    sections = text.split("\n#### ")[1:]
    assert len(sections) == len(results["examples"])
    assert set(script.EXAMPLE_SETTINGS) <= set(results["examples"])
    for section, (name, example) in zip(
        sections, results["examples"].items(), strict=True
    ):
        settings = script.EXAMPLE_SETTINGS.get(name, script.SETTINGS)
        short = name in script.EXAMPLE_SETTINGS
        lines = section.splitlines()
        columns = 7 if short else 10
        rows = [
            line for line in lines[lines.index("|" + "---|" * columns) + 1 :] if line
        ]
        assert len(example["rows"]) == len(rows) == len(settings)
        assert "**" not in rows[0]  # the default setting, or the smallest batch
        if short:
            batches = [int(r.split("|")[1].strip(" `").replace(",", "")) for r in rows]
            assert batches == sorted(batches)
        else:
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
