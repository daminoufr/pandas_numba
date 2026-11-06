# pandas_numba

A tiny numba `jitclass` holding pandas-style typed columns for use inside `@njit`
nopython code — plus a pandas DataFrame bridge and objmode glue to call back into
pandas from jitted code.

- Homepage: https://pypi.org/project/pandas-numba/
- Repository: https://github.com/daminoufr/pandas_numba
- PyPI: https://pypi.org/project/pandas-numba/

- **`Pandas_nb`** — a `jitclass` DataFrame-like container: one numpy array per
  column, keyed by title, one typed dict per element type.
- **`f_df_to_nb` / `f_nb_to_df`** (and friends) — the pandas bridge, module-level
  functions: `DataFrame → Pandas_nb` and back, with columns shared as views where
  possible.
- **`f_eval_expr` + objmode** — carry a DataFrame by handle and run pandas on the
  live frame from inside `@njit`.

## Requirements

- Python 3.12
- `numpy`, `pandas`, `numba`

## Installation

From PyPI:

```
pip install pandas_numba
```

Or the latest from GitHub:

```
pip install git+https://github.com/daminoufr/pandas_numba.git
```

## Quick start

```python
import pandas as pd
from numba import njit
from pandas_numba import f_df_to_nb, f_nb_to_np

df = pd.DataFrame({"a": [1., 2., 3.], "b": [1, 2, 3]}) # small dataframe
pandas_nb = f_df_to_nb(df)                          # columns shared as views

@njit
def f_update(pandas_nb):
    pandas_nb.m_floats["a"][:] = pandas_nb.m_floats["a"] * 2   # df will be changed (shared view)
    pandas_nb.f_add_int("c", pandas_nb.m_ints["b"] + 10)      # new int column (built in nopython)

f_update(pandas_nb)                     # run in nopython mode
df["c"] = f_nb_to_np(pandas_nb, "c")    # copy the new column back into df
print(df)                               # df is changed
#      a  b   c
# 0  2.0  1  11
# 1  4.0  2  12
# 2  6.0  3  13
```

(The same script lives in [`tests/simple_example.py`](tests/simple_example.py).)

## Calling Python (pandas) from a jitted function

A `DataFrame` has no numba type, so it cannot cross into nopython code. To run
real pandas from inside an `@njit` function, carry the frame **by handle**:
`f_register_df(nb, df)` stores `df` in a Python-side registry and writes its
`int64` id into `nb.m_df_id`. Inside the jitted code, an `objmode` block hops
back into Python and `f_eval_expr` evaluates an expression string with `np`,
`pd`, the registry `DF`, and `params` (your `Pandas_nb`) in scope:

```python
import pandas as pd
from numba import njit, objmode, float64
from pandas_numba import f_df_to_nb, f_register_df, f_eval_expr

df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
nb = f_df_to_nb(df)
f_register_df(nb, df)              # attach df by handle (sets nb.m_df_id)

@njit
def f_sum_via_pandas(nb):
    with objmode(tot=float64):     # hop from nopython back into Python
        tot = f_eval_expr("float(DF[params.m_df_id]['x'].sum())", nb)
    return tot

print(f_sum_via_pandas(nb))        # 10.0
```

Any pandas call is reachable this way, not just `.sum()`. Note `objmode` needs a
**static output type** (here `float64`), the registry holds `df` **weakly** (keep
your own reference alive while the handle is used), and each `f_eval_expr` round
trip costs on the order of ~15 µs, so it is for occasional calls, not hot loops.

## Example data

[`tests/simple_case_study.py`](tests/simple_case_study.py) generates a small top-of-book
order-book DataFrame (bid/ask prices and volumes over time) as a worked example
to feed through the bridge functions. `f_make_book(n)` scales it to `n` rows.

It also computes an indicator with a **per-row-variable window** (the window
length depends on recent activity — something pandas `rolling` cannot express
but an `@njit` loop does naturally), and times the jitted kernel against an
identical pure-Python twin on the same arrays. There is **one source of truth**:
the Python function is written once, and `numba.njit` compiles that same source —
it returns a new dispatcher without mutating the original, so the Python twin
stays directly callable for a fair, bit-identical comparison:

```python
# f_indicator_weighter_python(...) is the plain-Python kernel (the loop).
# The jitted twin is the SAME function, compiled at import time:
f_indicator_weighter_jit = njit(f_indicator_weighter_python)
```

Both return bit-identical results; the njit version is **~230× faster**:

```
mean_w_jit over 15483 rows (best of 5 runs):
  python :    194.899 ms
  njit   :      0.843 ms
  gain   :      231.2x faster
```

(Exact numbers vary by machine; the first njit call compiles, so the benchmark
warms it up before timing.)

## Documentation

See [`docs/pandas_numba.md`](docs/pandas_numba.md) for the full data model, API, the
pandas round-trip rules, the df-by-handle / string-eval objmode mechanism, and
the numba constraints.

A browsable **HTML API site** (Sphinx + autodoc, pydata theme) is generated from
the module's docstrings:

```
python scripts/generate_html_docs.py          # build if stale
python scripts/generate_html_docs.py --force   # always rebuild
python scripts/generate_html_docs.py --open    # rebuild and open in a browser
```

The hand-written Sphinx sources live in `docs_html/pandas_numba/source/`; the
built site is written to `docs_html/pandas_numba/build/index.html` (git-ignored,
regenerate on demand).

## Tests

```
python -m pytest tests/pandas_numba_tests.py -v
```

The suite is driven by one `ALL_TYPES` table — add a column type by adding one
row and every per-type test covers it.

## License

[MIT](LICENSE) © 2026 Damien Loison
