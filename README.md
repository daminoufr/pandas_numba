# pandas_numba

A tiny numba `jitclass` holding pandas-style typed columns for use inside `@njit`
nopython code — plus a pandas DataFrame bridge and objmode glue to call back into
pandas from jitted code.

Repository: https://github.com/daminoufr/pandas_numba

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

(The same script lives in [`simple_example.py`](simple_example.py).)

## Example data

[`simple_case_study.py`](simple_case_study.py) generates a small top-of-book
order-book DataFrame (bid/ask prices and volumes over time) as a worked example
to feed through the bridge functions. `f_make_book(n)` scales it to `n` rows.

## Documentation

See [`pandas_numba.md`](pandas_numba.md) for the full data model, API, the
pandas round-trip rules, the df-by-handle / string-eval objmode mechanism, and
the numba constraints.

## Tests

```
python -m pytest pandas_numba_tests.py -v
```

The suite is driven by one `ALL_TYPES` table — add a column type by adding one
row and every per-type test covers it.

## License

[MIT](LICENSE) © 2026 Damien Loison
