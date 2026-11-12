"""
Sphinx extension: build the methods table from padaquad's method registries.

The table on the methods page is generated from ``UNIFORM_METHODS`` and
``VARIABLE_METHODS`` when the documentation is built, so it cannot disagree with
the code: every number in it is an attribute of a method object, and its
citations are each method's ``references``. The rows can also be built without
Sphinx (``methods_table_rows``), which ``tests/test_docs_methods_table.py`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sphinx.application import Sphinx

# Registry of each family in padaquad.methods, in the order the table lists them.
FAMILIES = (
    ("Runge-Kutta", "RK_METHODS"),
    ("Gauss-Kronrod", "GK_METHODS"),
    ("Clenshaw-Curtis", "CC_METHODS"),
    ("Variable nodes", "INTERPOLATORY_METHODS"),
)
COLUMNS = (
    ("name", "Method"),
    ("family", "Family"),
    ("sampling", "Sampling"),
    ("nodes", "Nodes per subinterval"),
    ("degree", "Degree"),
    ("order", "Order"),
    ("estimate_degree", "Degree of the method subtracted in the error estimate"),
    ("references", "Sources"),
)
# Where the generated table is written (ignored by git; see .gitignore).
OUTPUT = (
    Path(__file__).resolve().parents[1] / "reference" / "generated" / "methods_table.md"
)


def methods_table_rows() -> list[dict[str, Any]]:
    """One row per registered method object (a method in both registries once)."""
    import padaquad.methods as registry  # noqa: PLC0415 (import at build time)

    rows, seen = [], set()
    for family, attribute in FAMILIES:
        for name, method in getattr(registry, attribute).items():
            if id(method) in seen:
                continue
            seen.add(id(method))
            sampling = [
                kind
                for kind, methods in (
                    ("uniform", registry.UNIFORM_METHODS),
                    ("variable", registry.VARIABLE_METHODS),
                )
                if methods.get(name) is method
            ]
            rows.append(
                {
                    "name": name,
                    "family": family,
                    "sampling": ", ".join(sampling),
                    "nodes": method.n_nodes,
                    "degree": method.degree,
                    "order": method.order,
                    "estimate_degree": method.error_exponent - 2,
                    "references": tuple(method.references),
                }
            )
    return rows


def render_methods_table() -> str:
    """The table as MyST Markdown, citing each method's sources."""
    lines = [
        "| " + " | ".join(title for _, title in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in methods_table_rows():
        cells = []
        for key, _ in COLUMNS:
            value = row[key]
            if key == "name":
                value = f"`{value}`"
            elif key == "references":
                value = "; ".join(f"{{cite:p}}`{ref}`" for ref in value)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_table(app: Sphinx) -> None:  # noqa: ARG001 (Sphinx event signature)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = render_methods_table()
    if not OUTPUT.exists() or OUTPUT.read_text() != text:
        OUTPUT.write_text(text)


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the table generation at the start of every build."""
    app.connect("builder-inited", _write_table)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
