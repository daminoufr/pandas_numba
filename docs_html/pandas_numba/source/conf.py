# Configuration file for the Sphinx documentation builder.
#
# API documentation for the pandas_numba module (Pandas_nb jitclass + pandas
# bridge + objmode glue). Mirrors the CudaCode C++ docs setup, but for Python:
# autodoc/napoleon introspect pandas_numba.py directly -- no extraction step.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# source/ is docs_html/pandas_numba/source -> repo root is three parents up.
# Put the repo root on sys.path so autodoc can import pandas_numba.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _read_version() -> str:
    """Read the package version from pyproject.toml (single source of truth)."""
    try:
        import tomllib  # Python 3.11+ (Anaconda is 3.12)
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        return data["project"]["version"]
    except Exception:
        return "0.0.0"


# -- Project information -----------------------------------------------------

project = "pandas_numba"
copyright = "2026, Damien Loison"
author = "Damien Loison"
release = _read_version()
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------
# Pandas_nb is a numba jitclass: autodoc cannot auto-enumerate its methods
# (they are dispatcher objects), so methods are pulled in explicitly with
# .. automethod:: in api.rst. napoleon renders the Google-style docstrings.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "pandas_numba API"

html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "secondary_sidebar_items": ["page-toc"],
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/daminoufr/pandas_numba",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pandas_numba/",
            "icon": "fa-brands fa-python",
        },
    ],
    "logo": {
        "text": "pandas_numba",
    },
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "footer_start": ["copyright"],
    "footer_end": [],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for syntax highlighting -----------------------------------------
pygments_style = "sphinx"
highlight_language = "python"
add_module_names = False

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for copybutton --------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |PS> "
copybutton_prompt_is_regexp = True
