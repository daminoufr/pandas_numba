# CLAUDE.md — pandas_numba

Project-local instructions.

> ⚠️ **This file is tracked and PUBLIC.** It is committed to the repo and
> pushed to the public GitHub (`daminoufr/pandas_numba`), so everything in it is
> world-readable. Information here must be **strictly controlled**: never write
> secrets or anything whose disclosure would matter — no tokens/PATs, passwords,
> keys, or their values; no private file contents, absolute local paths that
> leak a home/username, personal data, other account names, or internal URLs.
> Describe auth and workflows by *mechanism and location* (e.g. "a PAT in
> Windows Credential Manager"), never by value. When in doubt, leave it out —
> or keep it in an untracked local note instead. A change here ships publicly on
> the next commit; review every edit with that in mind.

## What this is

A numba `jitclass` (`Pandas_nb`) holding pandas-style typed columns usable inside
`@njit` nopython code, plus a pandas bridge (`f_df_to_nb` / `f_nb_to_df` and
friends) and objmode glue
(`f_eval_expr`) to call back into pandas from jitted code. Full design notes live
in `docs/pandas_numba.md`.

## Layout

The shippable module is `code/pandas_numba.py` (the source root, so `code/` holds
only that). Tests and examples are in `tests/`, the design doc in `docs/`, the
solution-management scripts in `scripts/`, and packaging / CI / the Sphinx docs
tree at the repo root.

- `code/pandas_numba.py`         — core: `Pandas_nb`, the bridge functions, `f_eval_expr` (the only packaged file).
- `tests/pandas_numba_tests.py`  — pytest suite, driven by one `ALL_TYPES` table.
- `tests/simple_example.py`      — minimal end-to-end example (shared-view update + new column).
- `tests/simple_case_study.py`   — case-study data generator + njit-vs-python benchmark.
- `docs/pandas_numba.md`              — full data model, API, round-trip rules, constraints.
- `scripts/generate_html_docs.py`     — Sphinx HTML-docs orchestrator (`--check`/`--force`/`--open`).
- `scripts/copy_to_www.py`            — publish the built site to the local www folder.
- `pyproject.toml`         — packaging (single-module dist `pandas_numba`; `package-dir` maps to `code/`).
- `.github/workflows/release.yml` — PyPI publish via Trusted Publishing (OIDC).
- `docs_html/pandas_numba/source/` — hand-written Sphinx sources (conf.py, index.rst, api.rst, _static/).

Each file in `tests/` prepends `code/` to `sys.path`, so `import pandas_numba`
works both under pytest and when a file is run directly.

## Python

Interpreter is Anaconda (3.12.7), not on PATH — call it by full path:

```
D:\Anaconda\python.exe
```

A bare `python` is a different interpreter (3.13) and must not be used here.

## Tests

```
D:\Anaconda\python.exe -m pytest tests\pandas_numba_tests.py -v
```

A bare `D:\Anaconda\python.exe -m pytest` (run at the repo root) also works —
`pytest.ini` points `testpaths` at `tests`.

Add a column type by adding one row to `ALL_TYPES`; every per-type test then
covers it.

## Conventions

- Keep docs (`docs/pandas_numba.md`, `README.md`) in sync with code changes.
- Markdown tables are column-aligned in the source.

## GitHub

- Repo: **https://github.com/daminoufr/pandas_numba** (public, owner `daminoufr`).
  Remote `origin` over HTTPS.
- Auth: a fine-grained PAT, stored in **Windows Credential Manager** (via GCM),
  scoped to this repo (`credential.useHttpPath=true` in the local git config)
  with the permissions needed for push / workflow edits / PR create+merge — so
  those work from the CLI without re-auth. The token value lives only in GCM,
  never in `.git/config` or this file.
  - If a push 403s, the token has expired → the user reissues one and re-stores
    it into GCM with `git credential approve`.
- `main` is **protected**: no direct pushes. **Only when the user explicitly
  asks to land work on `main`** (or to release): commit on a branch, push it,
  then open + merge a PR (the token can do both via the API; no approval
  required — GitHub forbids self-approval anyway). Then
  `git checkout main && git pull --ff-only`.
  - This PR→merge flow is **not** what `commitall` means. `commitall` (see
    global CLAUDE.md) is add + commit + push on the **current** branch — if that
    is `dev`, it commits and pushes `dev` and stops there. Never create a
    branch, PR, or merge to `main` just because `commitall` was said or because
    `main` is protected. Merging to `main` happens only on an explicit request.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `claude_cmd_run.bat` is gitignored — never pushed. `CLAUDE.md` is tracked.

## Releasing to PyPI

Package name **`pandas_numba`** (installs as `pandas_numba` / `pandas-numba`).
Publishing is automated via **Trusted Publishing** — no token, no local `twine`:

1. Bump `version` in `pyproject.toml` (versions are immutable; never reuse one).
2. Land it on `main` via a PR (branch → push → PR → merge → `git pull --ff-only`).
3. GitHub → **Releases → Draft a new release** → tag `vX.Y.Z` → **Publish release**.
4. `.github/workflows/release.yml` builds (sdist+wheel) and publishes via OIDC
   Trusted Publishing — the PyPI project's trusted-publisher config pins this
   repo's `release.yml` and its `pypi` environment.

Verify after: `curl -s https://pypi.org/pypi/pandas_numba/json` → `info.version`.

5. **Publish the HTML docs** to the local www folder (they track the release):

   ```powershell
   D:\Anaconda\python.exe scripts\generate_html_docs.py --force
   D:\Anaconda\python.exe scripts\copy_to_www.py
   ```

   `generate_html_docs.py` rebuilds the Sphinx site (the version on the pages is
   read from `pyproject.toml`, so it follows the bump); `copy_to_www.py` copies
   the built site to
   `..\www\daminoufr\applications\pandas_numba\docs_html\` (served entry point
   `…\pandas_numba\build\index.html`). The hand-written
   `applications\pandas_numba\index.html` landing page, one level up, is never
   touched. Use `--dry-run` to preview, `--clean` to drop pages removed by a
   later build. `docs_html/` itself is generated + gitignored, so this publish
   is the only thing that ships the docs anywhere.

Manual fallback (only if Actions is unavailable) — build in an **isolated venv**
so `twine`'s newer deps don't disturb the base Anaconda environment:

```
D:\Anaconda\python.exe -m venv <scratch>\pubvenv
<scratch>\pubvenv\Scripts\python.exe -m pip install build twine
rm -rf dist build *.egg-info && <scratch>\pubvenv\Scripts\python.exe -m build
<scratch>\pubvenv\Scripts\python.exe -m twine check dist/*
# upload needs a PyPI API token (user supplies via file):
#   TWINE_USERNAME=__token__ TWINE_PASSWORD=<pypi-...> ... -m twine upload dist/*
```
