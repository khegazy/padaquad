"""Sphinx configuration for the padaquad documentation.

Build locally (from the repository root)::

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -b html docs docs/_build/html
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path

# The examples and docstring examples must not depend on the machine: no GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS / "_ext"))

project = "padaquad"
author = "Kareem Hegazy"
copyright = "2024-2026, Kareem Hegazy"  # noqa: A001 (Sphinx setting name)
release = importlib.metadata.version("padaquad")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "methods_table",
    "performance_table",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build",
    "issues",
    "**/.ipynb_checkpoints",
    "auto_examples/*.ipynb",
    # Generated tables, included by the methods and performance pages.
    "reference/generated/*_table.md",
]

# Types come from the docstrings (numpydoc), not from the annotations.
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autosummary_generate = True

numpydoc_show_class_members = False
numpydoc_xref_param_type = False

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "plain"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": str(ROOT / "examples"),
    "gallery_dirs": "auto_examples",
    # Only the tutorial scripts and the verification scripts; examples/pode and
    # examples/speed_test are not part of the gallery.
    "filename_pattern": r"/(quadrature_basics|gradient_of_integral|polynomial_degree|integral_accuracy)\.py$",
    "ignore_pattern": r"(pode|speed_test)",
    "download_all_examples": False,
    "remove_config_comments": True,
    # The examples print text and draw no figures, so no image scraper (and no
    # matplotlib) is needed.
    "image_scrapers": (),
    "reset_modules": (),
}

html_theme = "pydata_sphinx_theme"
html_title = f"padaquad {release}"
html_theme_options = {
    "github_url": "https://github.com/khegazy/padaquad",
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_context = {
    "github_user": "khegazy",
    "github_repo": "padaquad",
    "github_version": "main",
    "doc_path": "docs",
}
html_static_path: list[str] = []
