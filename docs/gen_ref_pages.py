"""Generate the API reference pages and navigation.

This script is run by the ``mkdocs-gen-files`` plugin at build time. It walks
the ``padaquad`` source tree and emits one Markdown page per public module,
each containing a single ``mkdocstrings`` ``:::`` directive. ``mkdocstrings``
then introspects the module and renders every public class, function, and
method together with its numpy-style docstring.

Modules (and packages) whose name begins with an underscore are private and
are skipped, so the reference reflects the public API surface only.
"""

from pathlib import Path

import mkdocs_gen_files

PACKAGE = "padaquad"

# Modules that are internal but not underscore-prefixed. ``distributed`` is the
# multi-GPU / SLURM support layer, documented as internal in CLAUDE.md.
EXCLUDE = {("padaquad", "distributed")}

nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / PACKAGE

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        # Private module (e.g. methods/_base.py): skip.
        continue

    # Skip any package whose path contains a private component.
    if any(part.startswith("_") for part in parts):
        continue

    if parts in EXCLUDE:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
