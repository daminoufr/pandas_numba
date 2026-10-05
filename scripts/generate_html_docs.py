#!/usr/bin/env python
r"""HTML documentation orchestrator for pandas_numba.

Single entry point that regenerates the Sphinx HTML API documentation, tracking
a per-target timestamp so it only regenerates what is out of date. Modelled on
CudaCode's code/scripts/generate_html_docs.py, but for a pure-Python target:
Sphinx autodoc/napoleon introspect pandas_numba.py directly, so there is no
extraction/RST-generation pre-step -- the target is just ``sphinx -b html``.

Targets
-------
| --create name | kind   | output                                              |
|---------------|--------|-----------------------------------------------------|
| pandas_numba  | Python | docs_html/pandas_numba/build/index.html             |

Everything runs under Anaconda Python (CLAUDE.md), by absolute path.

Usage (PowerShell)
------------------
    D:\Anaconda\python.exe scripts/generate_html_docs.py
    D:\Anaconda\python.exe scripts/generate_html_docs.py --check
    D:\Anaconda\python.exe scripts/generate_html_docs.py --force
    D:\Anaconda\python.exe scripts/generate_html_docs.py --force --open
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
PYTHON = Path(sys.executable)
MANIFEST = REPO / "docs_html" / "doc_build_manifest.json"

TARGETS: dict[str, dict] = {
    "pandas_numba": {
        "kind": "Python",
        # The module is imported by autodoc, so it (and pyproject's version)
        # are the inputs; editing a docstring reruns the build.
        "inputs": ["code/pandas_numba.py", "pyproject.toml"],
        "source": "docs_html/pandas_numba/source",
        "build": "docs_html/pandas_numba/build",
    },
}


def newest_input_mtime(target: dict) -> float:
    newest = 0.0
    for pattern in target["inputs"]:
        for path in REPO.glob(pattern):
            if path.is_file():
                newest = max(newest, path.stat().st_mtime)
    # Hand-written Sphinx sources count as inputs too.
    source = REPO / target["source"]
    for extra in source.rglob("*"):
        if extra.is_file():
            newest = max(newest, extra.stat().st_mtime)
    # This orchestrator itself counts: editing it invalidates the docs.
    newest = max(newest, Path(__file__).stat().st_mtime)
    return newest


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def staleness(name: str, manifest: dict) -> tuple[bool, str]:
    target = TARGETS[name]
    index = REPO / target["build"] / "index.html"
    entry = manifest.get(name)
    if entry is None:
        return True, "never generated (no manifest entry)"
    if not index.exists():
        return True, "build/index.html is missing"
    last = entry.get("generated_at", 0.0)
    newest = newest_input_mtime(target)
    if newest > last:
        return True, f"inputs changed ({time.ctime(newest)} > {time.ctime(last)})"
    return False, f"fresh (generated {time.ctime(last)})"


def run(cmd: list[str], verbose: bool, dry_run: bool) -> bool:
    printable = " ".join(cmd)
    if dry_run:
        print(f"    [dry-run] {printable}")
        return True
    if verbose:
        print(f"    $ {printable}")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=not verbose, text=True)
    if proc.returncode != 0:
        print(f"    FAILED: {printable}", file=sys.stderr)
        if not verbose and proc.stdout:
            print(proc.stdout[-4000:], file=sys.stderr)
        if not verbose and proc.stderr:
            print(proc.stderr[-4000:], file=sys.stderr)
        return False
    return True


def generate(name: str, verbose: bool, dry_run: bool) -> bool:
    target = TARGETS[name]
    print(f"\n--- {name} ({target['kind']}) ---")
    build = REPO / target["build"]
    print(f"  step: sphinx -b html -> {target['build']}")
    if not dry_run and build.exists():
        shutil.rmtree(build)
    cmd = [str(PYTHON), "-m", "sphinx", "-b", "html",
           target["source"], target["build"]]
    if not verbose:
        cmd.insert(3, "-q")
    return run(cmd, verbose, dry_run)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--create", action="append", choices=[*TARGETS, "all"],
                    help="target(s) to (re)generate; repeatable (default: all)")
    ap.add_argument("--force", action="store_true", help="regenerate even if fresh")
    ap.add_argument("--check", action="store_true",
                    help="print stale/fresh per target and exit")
    ap.add_argument("--open", action="store_true",
                    help="open the built index.html in a browser afterwards")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    requested = args.create or ["all"]
    names = list(TARGETS) if "all" in requested else list(dict.fromkeys(requested))

    manifest = load_manifest()

    print("=" * 70)
    print("pandas_numba HTML documentation orchestrator")
    print("=" * 70)
    print(f"Python: {PYTHON}")

    verdicts = {n: staleness(n, manifest) for n in names}
    for name, (stale, why) in verdicts.items():
        print(f"  {name:<14} {'STALE' if stale else 'fresh'}  - {why}")

    if args.check:
        return 0

    failures = 0
    for name in names:
        stale, _ = verdicts[name]
        if not stale and not args.force:
            print(f"\n--- {name}: up to date, skipping (use --force to rebuild) ---")
            continue
        if generate(name, args.verbose, args.dry_run):
            if not args.dry_run:
                manifest[name] = {
                    "generated_at": time.time(),
                    "generated_at_human": time.ctime(),
                    "output": TARGETS[name]["build"] + "/index.html",
                }
                save_manifest(manifest)
            index = REPO / TARGETS[name]["build"] / "index.html"
            print(f"  OK -> {index}")
            if args.open and index.exists():
                webbrowser.open(index.as_uri())
        else:
            failures += 1

    print("\nDone!" if not failures else f"\n{failures} target(s) FAILED.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
