# pandas speed notes

Research notes on *where* pandas is slow and *why*, and on whether handing
compiled functions (numba `@njit`, Cython, C++) to `groupby(...).apply()` helps.
Background for why `pandas_numba` exists.

## Why many pandas operations are slow

The common thread: whenever pandas has to invoke the **Python interpreter once
per row / element / group** instead of running one vectorized pass in C, you pay
interpreter overhead (object boxing, the GIL, a function call) on *every*
iteration. At millions of rows that turns microseconds into minutes.

### The usual culprits

**Explicit row iteration**
- `df.iterrows()` — worst of the lot; builds a new `Series` object per row.
- `df.itertuples()` — faster than `iterrows` but still a Python loop.
- Any hand-written `for i in range(len(df))` indexing with `.iloc` / `.at`.

**`.apply()` / `.map()` with a Python callable**
- `df.apply(func, axis=1)` — calls `func` once per row.
- `series.apply(lambda x: ...)`, `df.map(...)` (formerly `applymap`) — once per element.
- The lambda runs in pure Python, so vectorization is lost even though it *looks* like one call.

**GroupBy / rolling with custom Python functions**
- `groupby(...).apply(python_func)` and `.agg(lambda ...)` / `.transform(lambda ...)`.
  Built-in reducers like `.sum()`, `.mean()`, `.count()` are *fast* — they are
  cythonized. It is the custom lambda that drops to Python.
- `rolling(...).apply(func, raw=False)` — the function runs per window in Python.
  (`raw=True` passes a NumPy array and helps, but it is still per-window.)

**`object`-dtype columns and `.str`**
- Anything stored as `object` (notably Python strings, mixed types) is processed
  element-by-element in Python.
- Most `.str` methods (`.str.contains`, `.str.extract`, `.str.replace`, …) loop
  internally over Python string objects.

**Per-element parsing / construction**
- `pd.to_datetime(col)` on irregular strings **without** `format=` — infers and
  parses each value in Python (passing `format=...` makes it far faster).
- Growing a frame in a loop (`pd.concat` / old `.append` inside a loop) — not
  per-row Python per se, but quadratic copying; same "repeated work" smell.

### The subset that genuinely cannot be vectorized

Some work is slow not by choice but by nature: **path-dependent / sequential
logic** where row *i* depends on a result computed at row *i−1*, or where the
window length itself varies per row. There is no single array expression for it,
so pandas has no vectorized path to offer — you are forced into a Python loop.
That is exactly where compiling the loop with `numba.njit` wins hundreds of×:
the loop is *necessary*, you just want it running as machine code instead of
interpreted Python.

### Rule of thumb

> If you passed pandas a **Python function** (a `lambda`, `def`, or `.apply`), or
> you are **looping rows yourself**, or you are touching **`object`/string** data,
> you are probably running Python per element — the slow path.
>
> If instead you expressed the whole operation as **column arithmetic, boolean
> masks, or built-in reductions**, it runs in C over the entire array — the fast path.

## Does `groupby(...).apply()` accept an njit / Cython / C++ function?

Mechanically, yes — `groupby(...).apply(f)` accepts **any callable**. But
"accepted" and "fast" are two different things.

### What `apply` passes, and why it bites compiled functions

`groupby(...).apply(f)` calls `f` **once per group**, passing a **pandas object**
(a sub-`DataFrame`, or a sub-`Series` for a `SeriesGroupBy`). That matters:

- **`@njit` (nopython)** has *no numba type for a DataFrame/Series*. Passing a raw
  `@njit` function makes numba try to compile for argument type "DataFrame" and
  raise a `TypingError`. It does **not** work directly.
- **Cython / C++** functions do not understand a pandas object either — they want
  C types, NumPy buffers, or typed memoryviews.

So in all three cases you must feed them **NumPy arrays**, not the pandas group,
via a thin Python wrapper:

```python
g.apply(lambda sub: njit_kernel(sub["x"].to_numpy(), sub["w"].to_numpy()))
```

The `lambda` is Python but runs only once per group; the heavy numeric work
inside `njit_kernel` is compiled. Same pattern for Cython (pass `.to_numpy()`, or
a typed memoryview) and for a C++ function (pass array pointers/buffers).

### The catch: the per-group machinery is the real cost

`groupby().apply()` has two cost layers:

| Layer        | What it is                                                                         | Does compiling `f` help? |
|--------------|------------------------------------------------------------------------------------|--------------------------|
| The body     | `f` itself — the numeric work.                                                      | Yes.                     |
| The dispatch | Splitting the frame, materializing a pandas sub-object per group, the Python call, concatenating results. | No — pure pandas/Python overhead. |

For **many small groups**, the dispatch layer dominates, so a compiled `f` barely
moves the needle. Compiling helps most when you have **few groups with a lot of
data each**, so the body's work outweighs the dispatch.

### Better options, roughly in order of payoff

1. **Built-in reducers** (`.sum()`, `.mean()`, `.min()`, `.cumsum()`, …) — already
   cythonized C. If your logic is expressible this way, nothing custom will beat it.

2. **`engine="numba"`** where pandas supports it — it compiles *your* function and
   applies it per group with less Python overhead (and can release the GIL).
   Available on `Rolling.apply` / `Expanding.apply`, and on groupby **`.agg` /
   `.aggregate`** and **`.transform`** (and experimentally `DataFrame.apply`). The
   function must take the numba-friendly signature `(values, index)` and be
   nopython-compatible. Note: **`GroupBy.apply` itself has no `engine` argument** —
   only agg/transform do.

   ```python
   def kernel(values, index):            # values, index are NumPy arrays
       ...
   g["x"].transform(kernel, engine="numba")
   ```

3. **One compiled pass over all groups** — the fastest, and the `pandas_numba`
   approach. Instead of `apply` splitting into N sub-frames, hand your `@njit`
   function the **whole columns as arrays plus the group labels/codes**, and loop
   over everything in a single nopython call:

   ```python
   codes = g.ngroup().to_numpy()               # integer group id per row
   out = njit_grouped(values, codes, n_groups) # one compiled loop, no per-group pandas objects
   ```

   This eliminates the dispatch layer entirely — no sub-object construction, no
   Python call per group — which is usually where the real win is.

### Bottom line

`apply` will take a compiled callable, but you must pass it arrays (a raw `@njit`
on a DataFrame errors), and you only speed up the function body, not pandas'
per-group dispatch. For the big win, either use `engine="numba"` on
`agg` / `transform`, or skip `apply` and do a single compiled loop over the
columns keyed by group codes.
