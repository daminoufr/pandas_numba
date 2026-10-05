# pandas_numba

A numba `jitclass` (`Pandas_nb`) that holds DataFrame-like columns usable inside
`@njit` nopython code, plus a pandas bridge (`f_df_to_nb` / `f_nb_to_df` and
friends) and a way to call
back into Python/pandas from jitted code (`f_eval_expr` + objmode). Core lives in
`code/pandas_numba.py`; tests and their `@njit` helpers in
`tests/pandas_numba_tests.py`.

## Data model

`Pandas_nb` keeps one numpy array per column, keyed by title. numba typed dicts
are homogeneous in value type, so there is **one dict per element type**
(`m_floats`, `m_ints`, `m_char3`, …); `m_titles[title]` stores an `int64` type
code selecting which dict holds the column. An `int64` `m_df_id` optionally
carries a source DataFrame by handle (see below); −1 means none.

Supported column types (each has a code in `m_titles` and a backing dict):

| Family      | dtypes                                     | Backing dict                        |
| ----------- | ------------------------------------------ | ----------------------------------- |
| float       | float64, float32                           | `m_floats`, `m_float32`             |
| int         | int64/32/16/8, uint64/32/16/8              | `m_ints`, `m_int32`, … `m_uint8`    |
| bool        | bool_                                      | `m_bool`                            |
| complex     | complex128, complex64                      | `m_complex128`, `m_complex64`       |
| datetime    | datetime64[ns], timedelta64[ns]  (ns only) | `m_datetime64ns`, `m_timedelta64ns` |
| char (S)    | S1, S3, S10, S100  (bytes)                 | `m_char1`, `m_char3`, …             |
| unicode (U) | U1, U3, U10, U100  (code points)           | `m_unicode1`, `m_unicode3`, …       |

Fixed-width text comes in exactly four widths: **1, 3, 10, 100** (bytes for `|S`,
code points for `|U`). Wider values raise `ValueError`. `CODE_TO_ATTR` maps each
code to its dict; `CHAR_WIDTH` maps the char codes to their width.

## API

### Building columns directly

`Pandas_nb()` then `nb.f_add_<type>(title, values=None, error_if_present=True)`,
e.g. `f_add_float`, `f_add_int`, `f_add_char3`, `f_add_unicode10`. `values=None`
makes an empty column — but only for numeric types; a fixed-width char/unicode
column **cannot be allocated inside numba**, so `values` is required there.
`error_if_present=True` rejects a duplicate title (pass `False` to overwrite;
jitclass methods take positional args only).

### pandas bridge — module-level functions

| Function                                    | Does                                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------------------- |
| `f_df_to_nb(df, str_types={})`              | DataFrame → `Pandas_nb` (one column per df column).                                |
| `f_nb_to_df(nb, df=None)`                   | `Pandas_nb` → fresh DataFrame, or add its missing columns to an existing `df`.     |
| `f_add_to_nb(nb, title, arr, str_types={})` | Add one numpy array as the column type matching its dtype. Returns `True`/`False`. |
| `f_nb_to_np(nb, title)`                     | The numpy array backing one column.                                                |
| `f_np_to_np_str(arr, kind="U")`             | Any array → smallest fitting fixed-width U (`kind="U"`) or S (`kind="S"`) array.   |

`str_types` is `{title: "U"|"S"}`: a listed column is run through
`f_np_to_np_str` to a fixed-width array and kept, instead of being dropped as
object. `f_add_to_nb` returns `False` (column dropped) for an object array with
no matching `str_types` entry, or an unsupported dtype raises `ValueError`.

### Sharing and round-trip

| Column                             | `f_df_to_nb`            | survives `df → nb → df`? |
| ---------------------------------- | ----------------------- | ------------------------ |
| numeric / `datetime64[ns]`         | **shared view** of df   | yes                      |
| native char (S)                    | **shared view** of df   | yes                      |
| object text, listed in `str_types` | fresh copy (not shared) | yes (as S/U)             |
| object text / unicode, not listed  | dropped                 | no                       |

A shared-view column edited in nopython writes straight back into the DataFrame;
a `str_types`-converted column is a fresh array, so changes must be copied back
by hand. On the way out, char columns are forced back to native `|S`; unicode is
emitted as object (which pandas cannot hold as a native column), so a second
`f_df_to_nb` drops it — numeric and char survive the round-trip, unicode does
not. `datetime64` is **ns only** (us → `ValueError`); tz-aware datetimes become
object and are dropped.

### Carrying a DataFrame by handle

A DataFrame has no numba type and cannot cross into nopython. `f_register_df(nb,
df)` stores `df` in a weak module registry (`_DF_REGISTRY`) under a fresh `int64`
id written to `nb.m_df_id`. Inside `@njit`, an objmode block runs pandas on the
live frame via `f_eval_expr`:

```python
with objmode(tot=float64):
    tot = f_eval_expr("float(DF[params.m_df_id]['x'].sum())", nb)
```

`f_eval_expr(expr, params)` evaluates `expr` with `np`, `pd`, `DF` (the registry)
and `params` in scope. The registry holds `df` **weakly** — it auto-evicts when
the caller drops `df` (no manual unregister), so the caller must keep `df` alive
while the handle is used; ids are never reused.

## numba constraints

- Numeric arrays **can** be built in nopython (`np.empty`, `np.arange`, …) and
  passed to `f_add_*`. Fixed-width char/unicode arrays **cannot** — build them in
  Python (`np.empty(n, dtype="U10")`) and pass them in.
- `np.timedelta64(...)` **cannot** be constructed in nopython; build the delta in
  Python and pass it in (adding it to a `datetime64[ns]` works).
- objmode needs **static output types**; a DataFrame/Series cannot be boxed —
  pass `.to_numpy()` or carry the frame by handle.
- Windows: a bare `np.array([1, 2, 3])` is **int32** on win64, so it routes to
  `INT32`/`m_int32`. pandas columns are int64, so `f_df_to_nb` is unaffected.

## Testing

The official suite is the pytest file `pandas_numba_tests.py`:

```
D:\Anaconda\python.exe -m pytest tests\pandas_numba_tests.py -v
```

It is driven by one `ALL_TYPES` table — **add a column type by adding one row**,
and every per-type test covers it. Calling `f_eval_expr` from nopython costs
≈ 15 µs/call (the bare objmode round-trip is ≈ 0.8 µs; the rest is Python
`eval`).

## Prior art (searched 2026-10-03)

"pandas columns inside numba" exists in **Bodo** (active) and the archived
**Intel SDC**, but both register a real DataFrame type via
`numba.experimental.structref` and compile pandas ops — whole frameworks.
`Pandas_nb` is the opposite: a tiny shared-buffer container, one dict per dtype.
That lightweight design, and the string-eval / df-by-handle objmode glue, are not
provided by any library; `structref` is the one official primitive worth knowing
(used by those libraries, not needed here).
