"""
Generate the README's reference list from ``docs/references.bib``.

The bibliography has one source, ``docs/references.bib``. The README's
"References" section is written from it between the markers
``<!-- references-start -->`` and ``<!-- references-end -->``;
``tests/test_docs_methods_table.py`` fails if the committed section differs
from what this script writes. Run it after editing the bibliography::

    python docs/_ext/readme_references.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "docs" / "references.bib"
README = ROOT / "README.md"
START, END = "<!-- references-start -->", "<!-- references-end -->"


def _entries(text: str) -> list[dict[str, str]]:
    """Parse the BibTeX entries of ``text`` into dictionaries of their fields."""
    entries = []
    for match in re.finditer(r"^@(\w+)\{([^,]+),(.*?)^\}", text, re.M | re.S):
        fields = {"entry_type": match.group(1).lower(), "key": match.group(2).strip()}
        for name, raw in re.findall(
            r"^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", match.group(3), re.M | re.S
        ):
            value = re.sub(r"[{}]", "", " ".join(raw.split()))
            fields[name.lower()] = value.replace("\\_", "_").replace("--", "-")
        entries.append(fields)
    return entries


def render() -> str:
    """Render the reference list as Markdown bullets, sorted by key."""
    lines = []
    for e in sorted(_entries(BIB.read_text()), key=lambda e: e["key"]):
        author = e.get("author", "").replace(" and ", ", ")
        title = e["title"] if e["title"][-1] in ".?!" else e["title"] + "."
        parts = [f"- {author} ({e.get('year', 'n.d.')}). **{title}**"]
        venue = e.get("journal") or e.get("booktitle") or e.get("howpublished")
        if e["entry_type"] == "techreport":
            venue = f"{e.get('institution', '')}, {e.get('number', '')}".strip(", ")
        if venue:
            issue = f"({e['number']})" if "number" in e and "journal" in e else ""
            where = f"{e['volume']}{issue}" if "volume" in e else ""
            pages = e.get("pages", "")
            parts.append(
                f"*{venue}*"
                + (f" {where}" if where else "")
                + (f", {pages}" if pages else "")
                + "."
            )
        if "eprint" in e:
            parts.append(f"arXiv:{e['eprint']}.")
        if "doi" in e:
            parts.append(f"[doi:{e['doi']}](https://doi.org/{e['doi']})")
        elif "url" in e:
            parts.append(f"<{e['url']}>")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def readme_with_references(readme: str) -> str:
    """Return ``readme`` with its reference section replaced by ``render()``."""
    start, end = readme.index(START) + len(START), readme.index(END)
    return readme[:start] + "\n" + render() + readme[end:]


if __name__ == "__main__":
    README.write_text(readme_with_references(README.read_text()))
