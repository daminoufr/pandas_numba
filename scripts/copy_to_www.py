#!/usr/bin/env python
r"""copy_to_www.py - publish the pandas_numba docs out of docs_html/.

Copies a curated subset of the built Sphinx site to the local www publish
folder, preserving each file's relative path. Default destination is
``..\www\daminoufr\applications\pandas_numba\docs_html`` (next to the
hand-written ``applications\pandas_numba\index.html`` landing page, which is one
level up and never touched). Override with ``--dest``. Mirrors CudaCode's
code/scripts/docs_html_copy.py, adapted to this site's single-page ``api.html``
layout.

What is copied is web content only: the pages, the search/index support, and
the ``_static`` / ``_sources`` / ``_sphinx_design_static`` / ``_modules``
(viewcode source) directories. Left behind as build-only: ``.doctrees/`` (Sphinx
pickles, never served), ``.buildinfo``, and the Sphinx inputs under
``docs_html/pandas_numba/source``.

``docs_html/`` is generated and gitignored; regenerate it with
``scripts/generate_html_docs.py`` (the orchestrator) before publishing.

Usage (PowerShell):
    D:\Anaconda\python.exe scripts\copy_to_www.py
    D:\Anaconda\python.exe scripts\copy_to_www.py --dry-run
    D:\Anaconda\python.exe scripts\copy_to_www.py --clean
    D:\Anaconda\python.exe scripts\copy_to_www.py --dest "D:\other\pandas_numba\docs_html"
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
SRC = PROJECT_ROOT / "docs_html"
DEFAULT_DEST = (PROJECT_ROOT.parent / "www" / "daminoufr" / "applications"
                / "pandas_numba" / "docs_html")

# The Sphinx build lives one level down; there is only one package
# (pandas_numba) and no root docs_html/index.html -- the landing page is the
# build's index.html.
_B = "pandas_numba/build"

# Files copied (relative to docs_html/); relative paths are preserved at the
# destination. The pages plus search/index support (searchindex.js, objects.inv,
# search.html, genindex.html) so the search box and index work.
FILES = [
    "doc_build_manifest.json",
    f"{_B}/index.html",
    f"{_B}/api.html",
    f"{_B}/genindex.html",
    f"{_B}/search.html",
    f"{_B}/searchindex.js",
    f"{_B}/objects.inv",
]

# Directories copied wholesale (relative to docs_html/). _static/ is what makes
# the pages render styled; _sources/ backs the "view page source" links;
# _modules/ holds the viewcode pages the autodoc "[source]" links point to.
DIRS = [
    f"{_B}/_static",
    f"{_B}/_sources",
    f"{_B}/_sphinx_design_static",
    f"{_B}/_modules",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the pandas_numba docs from docs_html/.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST),
                        help=f"Destination directory (default: {DEFAULT_DEST}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be copied, do nothing.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the destination directory first, so pages "
                             "renamed or removed by a later build do not linger.")
    args = parser.parse_args()

    dest_root = Path(args.dest)
    print(f"Source: {SRC}")
    print(f"Dest:   {dest_root}")
    print()

    if not (SRC / _B / "index.html").exists():
        print("The built site is missing. Regenerate it first:", file=sys.stderr)
        print(r"    D:\Anaconda\python.exe scripts\generate_html_docs.py --force",
              file=sys.stderr)
        return 1

    if args.clean and dest_root.exists():
        if args.dry_run:
            print(f"DRY  clean {dest_root}")
        else:
            shutil.rmtree(dest_root)
            print(f"OK   clean {dest_root}")
        print()

    copied = missing = 0

    for rel in FILES:
        src = SRC / rel
        dst = dest_root / rel
        if not src.exists():
            print(f"MISS file {rel}  (source not found: {src})")
            missing += 1
            continue
        if args.dry_run:
            print(f"DRY  file {rel} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"OK   file {rel}")
        copied += 1

    for rel in DIRS:
        src = SRC / rel
        dst = dest_root / rel
        if not src.is_dir():
            print(f"MISS dir  {rel}  (source not found: {src})")
            missing += 1
            continue
        n = sum(1 for _ in src.rglob("*") if _.is_file())
        if args.dry_run:
            print(f"DRY  dir  {rel}/ ({n} files) -> {dst}")
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"OK   dir  {rel}/ ({n} files)")
        copied += 1

    print()
    print(f"{'Would copy' if args.dry_run else 'Copied'}: {copied} item(s)   "
          f"Missing: {missing}")
    if not args.dry_run and missing == 0:
        print(f"\nPublished site entry point:\n    "
              f"{dest_root / _B / 'index.html'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
