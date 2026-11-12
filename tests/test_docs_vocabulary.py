"""The public docstrings and user-facing pages follow the writing rules.

docs/development/docstrings.md lists the rules. This test checks the ones a
search can check, in prose only (code spans and code blocks are removed first):

- the glossary's words: no "panel", "barrier", "rule" for a method (proper
  names such as "Simpson's rule" are allowed), or "time" for ``t``;
- in public docstrings (everything on the API reference pages): no bug IDs,
  "Phase N", "historical", or names of private functions and attributes.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import padaquad
from padaquad.methods import (
    DegreeMeasurement,
    MethodClass,
    UniformMethodClass,
    VariableMethodClass,
)

ROOT = Path(__file__).resolve().parents[1]
# Pages written for users; the development pages and the issue notes may name
# internals.
PAGES = [
    *sorted(
        p
        for p in (ROOT / "docs").rglob("*.md")
        if p.relative_to(ROOT / "docs").parts[0]
        not in {"development", "issues", "_build"}
        and "generated" not in p.parts
    ),
    ROOT / "README.md",
]

BANNED_WORDS = {
    "panel": r"\bpanels?\b",
    "barrier": r"\bbarriers?\b",
    "time for t": r"\btime (points?|mesh|steps?|dimensions?|variable|coordinates?)\b",
    "rule": r"\brules?\b",
}
# "rule" is allowed in these proper names and phrases.
ALLOWED_RULE = r"(trapezoidal|trapezoid|Simpson's|rectangle|merge|acceptance|accept-or-split) rules?"
PUBLIC_ONLY = {
    "bug ID": r"\bB\d+\b",
    "Phase N": r"\bPhase \d",
    "historical": r"\bhistorical\b",
    "private name": r"(?<![\w.`])_[a-z]\w*",
}


def _prose(text: str) -> str:
    """Text without code blocks, inline code, math and URLs."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"(?m)^(\s*)>>>.*$|^\s*\.\.\.\s.*$", " ", text)
    text = re.sub(r"(?ms)::\n\n(\s+).*?(?=\n\S|\Z)", " ", text)  # rst literal blocks
    text = re.sub(r"``[^`]*``|`[^`]*`", " ", text)
    text = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(ALLOWED_RULE, " ", text, flags=re.I)


def _public_docstrings() -> dict[str, str]:
    """Docstrings of every object on the API reference pages."""
    objects = {name: getattr(padaquad, name) for name in padaquad.__all__}
    objects.update(
        MethodClass=MethodClass,
        UniformMethodClass=UniformMethodClass,
        VariableMethodClass=VariableMethodClass,
        DegreeMeasurement=DegreeMeasurement,
    )
    docs = {}
    for name, obj in objects.items():
        if isinstance(obj, dict):  # documented by hand on the reference pages
            continue
        docs[name] = inspect.getdoc(obj) or ""
        if inspect.isclass(obj):
            for member, value in vars(obj).items():
                if member == "__init__" or not member.startswith("_"):
                    doc = inspect.getdoc(value)
                    if (doc and callable(value)) or isinstance(value, property):
                        docs[f"{name}.{member}"] = doc or ""
    solver = padaquad.UniformAdaptiveQuadrature
    for member in ("__init__", "integrate", "get_max_batch", "set_dtype_by_input"):
        docs[f"UniformAdaptiveQuadrature.{member}"] = inspect.getdoc(
            getattr(solver, member)
        )
    return docs


@pytest.mark.parametrize("name", sorted(_public_docstrings()))
def test_public_docstring_vocabulary(name: str) -> None:
    text = _prose(_public_docstrings()[name])
    # An object's own parameters may start with "_" (e.g. ``_y``); they are not
    # private code.
    obj = getattr(padaquad, name.split(".", maxsplit=1)[0], None)
    if callable(obj) and not inspect.isclass(obj):
        for parameter in inspect.signature(obj).parameters:
            text = re.sub(rf"\b{re.escape(parameter)}\b", " ", text)
    found = {
        label: re.findall(pattern, text, flags=re.I if label != "private name" else 0)
        for label, pattern in {**BANNED_WORDS, **PUBLIC_ONLY}.items()
    }
    found = {label: hits for label, hits in found.items() if hits}
    assert not found, f"{name}: {found}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_page_vocabulary(page: Path) -> None:
    text = _prose(page.read_text())
    found = {
        label: re.findall(pattern, text, flags=re.I)
        for label, pattern in BANNED_WORDS.items()
    }
    found = {label: hits for label, hits in found.items() if hits}
    assert not found, f"{page.relative_to(ROOT)}: {found}"


def _all_docstrings() -> dict[str, str]:
    """Every module, class and function docstring in padaquad/."""
    import ast  # noqa: PLC0415

    docs = {}
    for path in sorted((ROOT / "padaquad").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docs[f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 1)}"] = doc
    return docs


def test_docstring_math_uses_single_backslashes() -> None:
    """A doubled backslash in a docstring's value renders as a line break in
    MathJax; LaTeX commands need one backslash (use a raw docstring)."""
    doubled = [where for where, doc in _all_docstrings().items() if "\\\\" in doc]
    assert not doubled, f"docstrings with a doubled backslash: {doubled}"
