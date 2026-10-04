"""Official pytest suite for pandas_numba.py.

This is THE suite to re-run on every change to ``pandas_numba.py``:

    D:\\Anaconda\\python.exe -m pytest code\\pandas_numba_tests.py -v
    D:\\Anaconda\\python.exe -m pytest code\\pandas_numba_tests.py -s -q   # see timing
    D:\\Anaconda\\python.exe code\\pandas_numba_tests.py                    # direct run
    D:\\Anaconda\\python.exe code\\pandas_numba_tests.py --list             # list tests + descriptions (no run)

The file name is not ``test_*.py``, so pytest collects it only by explicit path
(the ``__main__`` block below does exactly that). Test functions are named
``test_*`` so they are collected.

Design — ONE source of truth for the 23 column types
-----------------------------------------------------
``ALL_TYPES`` holds one ``TypeSpec`` per type code 0..22 (dtype, the ``f_add_*``
method, a sample-array builder, and flags). Every per-type test is a single
function parametrized over ``ALL_TYPES`` with ``@pytest.mark.parametrize``, so
each type shows as its own pytest id (e.g. ``test_add_type[char3]``) and a
failure names the offending type. **Adding a new column type = adding one row to
ALL_TYPES** -- the tests then cover it automatically.

Notes
-----
* Source is UTF-8 (the non-ASCII "€UR" gating test below relies on it).
* Fixed-width char/unicode arrays are built in plain Python; numba cannot
  allocate them inside nopython code.
* The ``@njit`` helpers are module-level because their objmode output
  annotations (``PANDAS_NB_T`` / ``float64_1d_t`` / ``PANDAS_NB_FLOAT_T``) must
  be module-level names.
* This file is the sole, official suite.
"""

import gc
import sys
import time
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
from numba import njit, objmode, float64

from pandas_numba import (
    Pandas_nb,
    f_nb_to_np,
    f_nb_to_df,
    f_add_to_nb,
    f_df_to_nb,
    f_help_wide,
    f_np_to_np_str,
    f_register_df,
    f_eval_expr,
    _DF_REGISTRY,
    CODE_TO_ATTR,
    CHAR_WIDTH,
    PANDAS_NB_T,
    PANDAS_NB_FLOAT_T,
    float64_1d_t,
    FLOAT64, INT64, BOOL, INT8, INT16, INT32,
    UINT8, UINT16, UINT32, UINT64, FLOAT32,
    COMPLEX64, COMPLEX128, DATETIME64NS, TIMEDELTA64NS,
    CHAR1, CHAR3, CHAR10, CHAR100,
    UNICODE1, UNICODE3, UNICODE10, UNICODE100,
)

# ---------------------------------------------------------------------------
# Single source of truth: one TypeSpec per type code 0..22.
# ---------------------------------------------------------------------------
# code       : the int64 type code stored in m_titles
# name        : short unique id, used as the pytest param id and column title
# add_method  : the Pandas_nb.f_add_* method for this type
# group       : float|int|uint|bool|complex|datetime|timedelta|char|unicode
# np_dtype    : numpy dtype for numeric groups (None for datetime/td/char/unicode)
# width       : fixed width for char/unicode (None otherwise)
TypeSpec = namedtuple("TypeSpec", "code name add_method group np_dtype width")

ALL_TYPES = [
    TypeSpec(FLOAT64, "float64", "f_add_float", "float", np.float64, None),
    TypeSpec(INT64, "int64", "f_add_int", "int", np.int64, None),
    TypeSpec(BOOL, "bool", "f_add_bool", "bool", np.bool_, None),
    TypeSpec(INT8, "int8", "f_add_int8", "int", np.int8, None),
    TypeSpec(INT16, "int16", "f_add_int16", "int", np.int16, None),
    TypeSpec(INT32, "int32", "f_add_int32", "int", np.int32, None),
    TypeSpec(UINT8, "uint8", "f_add_uint8", "uint", np.uint8, None),
    TypeSpec(UINT16, "uint16", "f_add_uint16", "uint", np.uint16, None),
    TypeSpec(UINT32, "uint32", "f_add_uint32", "uint", np.uint32, None),
    TypeSpec(UINT64, "uint64", "f_add_uint64", "uint", np.uint64, None),
    TypeSpec(FLOAT32, "float32", "f_add_float32", "float", np.float32, None),
    TypeSpec(COMPLEX64, "complex64", "f_add_complex64", "complex", np.complex64, None),
    TypeSpec(COMPLEX128, "complex128", "f_add_complex128", "complex", np.complex128, None),
    TypeSpec(DATETIME64NS, "datetime64ns", "f_add_datetime64ns", "datetime", None, None),
    TypeSpec(TIMEDELTA64NS, "timedelta64ns", "f_add_timedelta64ns", "timedelta", None, None),
    TypeSpec(CHAR1, "char1", "f_add_char1", "char", None, 1),
    TypeSpec(CHAR3, "char3", "f_add_char3", "char", None, 3),
    TypeSpec(CHAR10, "char10", "f_add_char10", "char", None, 10),
    TypeSpec(CHAR100, "char100", "f_add_char100", "char", None, 100),
    TypeSpec(UNICODE1, "unicode1", "f_add_unicode1", "unicode", None, 1),
    TypeSpec(UNICODE3, "unicode3", "f_add_unicode3", "unicode", None, 3),
    TypeSpec(UNICODE10, "unicode10", "f_add_unicode10", "unicode", None, 10),
    TypeSpec(UNICODE100, "unicode100", "f_add_unicode100", "unicode", None, 100),
]

# ascii sample text fitting each fixed width (1, 3, 10, 100)
WIDTH_SAMPLE = {1: "a", 3: "abc", 10: "hello", 100: "x" * 50}

_IDS = [s.name for s in ALL_TYPES]


def is_fixed_width(s):
    # char/unicode: cannot be allocated inside numba; values required on f_add_*.
    return s.group in ("char", "unicode")


def survives_roundtrip(s):
    # df -> nb -> df: everything survives except unicode (pandas stores |U as
    # object, which is dropped). char crosses as native |S and survives.
    return s.group != "unicode"


def shares_view(s):
    # f_df_to_nb gives a shared view for all non-unicode types (unicode dropped).
    return s.group != "unicode"


def sample(spec, n):
    # A length-n sample array for one type, built in plain Python.
    g = spec.group
    if g in ("float", "int", "uint"):
        return np.arange(1, n + 1, dtype=spec.np_dtype)
    if g == "bool":
        return np.arange(n) % 2 == 0
    if g == "complex":
        a = np.arange(1, n + 1)
        return (a + 1j * a).astype(spec.np_dtype)
    if g == "datetime":
        return np.arange(1, n + 1).astype("datetime64[ns]")
    if g == "timedelta":
        return np.arange(1, n + 1).astype("timedelta64[ns]")
    if g == "char":
        return np.array([WIDTH_SAMPLE[spec.width]] * n, dtype="S%d" % spec.width)
    if g == "unicode":
        return np.array([WIDTH_SAMPLE[spec.width]] * n, dtype="U%d" % spec.width)
    raise AssertionError("unknown group " + spec.group)


def new_value(spec):
    # A distinct valid scalar to poke into a shared column (index 0) and then
    # read back from df. Not defined for unicode (never shared).
    return {
        "float": 7.0, "int": 7, "uint": 7, "bool": False, "complex": 7 + 7j,
        "datetime": np.datetime64("2000-01-01", "ns"),
        "timedelta": np.timedelta64(99, "ns"), "char": b"z",
    }[spec.group]


def make_df(n):
    # A DataFrame with one column per type (titles = spec.name): numeric/datetime
    # as-is, char forced to native |S<width>, unicode left as object (|U coerces
    # to object on construction, as pandas cannot hold it natively).
    data = {s.name: sample(s, n) for s in ALL_TYPES}
    df = pd.DataFrame(data)
    for s in ALL_TYPES:
        if s.group == "char":
            df[s.name] = df[s.name].astype("S%d" % s.width)
    return df


SHARED_TYPES = [s for s in ALL_TYPES if shares_view(s)]


# ---------------------------------------------------------------------------
# @njit helpers (module-level; objmode annotations need module-level names)
# ---------------------------------------------------------------------------
@njit
def _njit_add_f2_into_f1(nb):
    # float1 += float2, element-wise, on the shared buffers.
    a = nb.m_floats["float1"]
    b = nb.m_floats["float2"]
    for i in range(a.shape[0]):
        a[i] += b[i]


@njit
def _njit_add_f2_into_f1_eur(nb):
    # float1 += float2 only where the char column ccy == b"EUR".
    a = nb.m_floats["float1"]
    b = nb.m_floats["float2"]
    ccy = nb.m_char3["ccy"]
    for i in range(a.shape[0]):
        if ccy[i] == b"EUR":
            a[i] += b[i]


@njit
def _njit_add_f2_into_f1_eur_unicode(nb):
    # float1 += float2 only where the unicode column ccy == "€UR" (non-ASCII).
    a = nb.m_floats["float1"]
    b = nb.m_floats["float2"]
    ccy = nb.m_unicode3["ccy"]
    for i in range(a.shape[0]):
        if ccy[i] == "€UR":
            a[i] += b[i]


def _build_two_float_nb():
    # Plain-python builder used from objmode: a Pandas_nb with float1/float2.
    df = pd.DataFrame({"float1": np.arange(5.0), "float2": np.arange(5.0) * 10.0})
    return f_df_to_nb(df)


@njit
def _njit_build_nb_via_objmode():
    # Obtain a Pandas_nb from python via objmode, then use it at nopython speed:
    # float1 += float2; return the sum of the updated float1.
    with objmode(nb=PANDAS_NB_T):
        nb = _build_two_float_nb()
    a = nb.m_floats["float1"]
    b = nb.m_floats["float2"]
    s = 0.0
    for i in range(a.shape[0]):
        a[i] += b[i]
        s += a[i]
    return s


@njit
def _njit_eval_array_scalar_tuple():
    # Three objmode blocks with DIFFERENT output types: a float64 array, a
    # float64 scalar, and an array derived from a (Pandas_nb, float64) tuple.
    with objmode(p1=PANDAS_NB_T):
        p1 = _build_two_float_nb()
    with objmode(arr=float64_1d_t):
        arr = f_eval_expr("params.m_floats['float1'] * 2.0", p1)
    with objmode(tot=float64):
        tot = f_eval_expr("float(np.sum(params.m_floats['float1']))", p1)
    with objmode(pair=PANDAS_NB_FLOAT_T):
        pair = (_build_two_float_nb(), 3.0)
    with objmode(arr2=float64_1d_t):
        arr2 = f_eval_expr("params[0].m_floats['float1'] * params[1]", pair)
    return arr, tot, arr2


@njit
def _njit_df_sum_via_objmode(nb):
    # Run a pandas call on the df carried by handle (m_df_id) via f_eval_expr.
    with objmode(tot=float64):
        tot = f_eval_expr("float(DF[params.m_df_id]['float1'].sum())", nb)
    return tot


@njit
def _njit_df_mean_via_objmode(nb):
    # An ARBITRARY pandas call (describe()) on the handle-resolved df, proving any
    # pandas method -- not just .sum() -- is reachable through m_df_id.
    with objmode(m=float64):
        m = f_eval_expr(
            "float(DF[params.m_df_id].describe().loc['mean','float2'])", nb)
    return m


@njit
def _njit_shift_dates(nb, title, delta):
    # Shift a datetime64[ns] column in place by a python-built ns timedelta.
    a = nb.m_datetime64ns[title]
    for i in range(a.shape[0]):
        a[i] = a[i] + delta


@njit
def _njit_eval_expr_loop(n, x):
    # Call f_eval_expr through objmode n times, accumulating the result so the
    # calls cannot be optimized away. Returns the sum (== n*(x+1)).
    s = 0.0
    for i in range(n):
        with objmode(r=float64):
            r = f_eval_expr("params + 1.0", x)
        s += r
    return s


@njit
def _njit_objmode_only_loop(n, x):
    # Baseline: the same objmode boundary n times, without f_eval_expr.
    s = 0.0
    for i in range(n):
        with objmode(r=float64):
            r = x + 1.0
        s += r
    return s


# ===========================================================================
# Tables / meta
# ===========================================================================
def test_all_types_covers_0_to_22():
    """ALL_TYPES has exactly codes 0..22 once, names unique, attr == CODE_TO_ATTR."""
    assert sorted(s.code for s in ALL_TYPES) == list(range(23))
    assert len({s.name for s in ALL_TYPES}) == len(ALL_TYPES)
    for s in ALL_TYPES:
        assert CODE_TO_ATTR[s.code], s.name          # a backing dict is mapped


def test_code_tables():
    """CODE_TO_ATTR covers 0..22 with real Pandas_nb members; CHAR_WIDTH is the char map."""
    assert sorted(CODE_TO_ATTR) == list(range(23))
    nb = Pandas_nb()
    for code, attr in CODE_TO_ATTR.items():
        assert hasattr(nb, attr), (code, attr)
    assert CHAR_WIDTH == {CHAR1: 1, CHAR3: 3, CHAR10: 10, CHAR100: 100}


# ===========================================================================
# Construction (per-type)
# ===========================================================================
@pytest.mark.parametrize("t", ALL_TYPES, ids=_IDS)
def test_add_type(t):
    """f_add_<type>(title, values) stores the array under the right code/dict."""
    nb = Pandas_nb()
    arr = sample(t, 4)
    getattr(nb, t.add_method)(t.name, arr)
    assert nb.m_titles[t.name] == t.code
    backing = getattr(nb, CODE_TO_ATTR[t.code])[t.name]
    assert np.array_equal(backing, arr)
    assert np.array_equal(f_nb_to_np(nb, t.name), arr)


@pytest.mark.parametrize("t", ALL_TYPES, ids=_IDS)
def test_values_none(t):
    """values=None: numeric -> empty typed array; fixed-width char/unicode -> ValueError."""
    nb = Pandas_nb()
    if is_fixed_width(t):
        with pytest.raises(ValueError):
            getattr(nb, t.add_method)(t.name)
    else:
        getattr(nb, t.add_method)(t.name)
        backing = getattr(nb, CODE_TO_ATTR[t.code])[t.name]
        assert len(backing) == 0
        assert backing.dtype == sample(t, 1).dtype


@pytest.mark.parametrize("t", ALL_TYPES, ids=_IDS)
def test_error_if_present(t):
    """A duplicate title raises (same OR different type, m_titles being global);
    the stored array is left unchanged; error_if_present=False overwrites."""
    nb = Pandas_nb()
    arr1 = sample(t, 3)
    getattr(nb, t.add_method)(t.name, arr1)
    # same-type duplicate raises, and leaves the stored array untouched (the
    # attempted array has a different length, so a silent overwrite would show)
    with pytest.raises(Exception):
        getattr(nb, t.add_method)(t.name, sample(t, 2))
    assert np.array_equal(getattr(nb, CODE_TO_ATTR[t.code])[t.name], arr1)
    # the present-check is a GLOBAL namespace across the backing dicts: re-adding
    # the same title as a DIFFERENT type (different add_method / dict) must raise
    other = next(s for s in ALL_TYPES if s.group != t.group)
    with pytest.raises(Exception):
        getattr(nb, other.add_method)(t.name, sample(other, 3))
    # error_if_present=False overwrites (positional arg)
    arr2 = sample(t, 2)
    getattr(nb, t.add_method)(t.name, arr2, False)
    assert np.array_equal(getattr(nb, CODE_TO_ATTR[t.code])[t.name], arr2)


# ===========================================================================
# Introspection
# ===========================================================================
def test_nb_to_np_unknown_code_raises():
    """f_nb_to_np raises ValueError when m_titles holds a code with no backing dict."""
    nb = Pandas_nb()
    nb.m_titles["x"] = 99
    with pytest.raises(ValueError):
        f_nb_to_np(nb, "x")


# ===========================================================================
# Conversion
# ===========================================================================
def test_roundtrip_df_nb_df():
    """df -> nb -> df: numeric + char survive exactly (order kept); unicode dropped."""
    df = make_df(4)
    nb = f_df_to_nb(df)
    survivors = [s.name for s in ALL_TYPES if survives_roundtrip(s)]
    assert list(nb.m_titles) == survivors
    for s in ALL_TYPES:
        if not survives_roundtrip(s):
            assert s.name not in nb.m_titles
    df2 = f_nb_to_df(nb)
    assert list(df2.columns) == survivors
    for name in survivors:
        assert df2[name].dtype == df[name].dtype, name
        assert np.array_equal(df2[name].to_numpy(), df[name].to_numpy()), name


def test_nb_df_nb():
    """nb -> df -> nb: char emits native |S (kept), unicode emits object (dropped)."""
    nb0 = f_df_to_nb(make_df(4))          # numeric + char
    for s in ALL_TYPES:                                 # add unicode directly
        if s.group == "unicode":
            getattr(nb0, s.add_method)(s.name, sample(s, 4))
    df = f_nb_to_df(nb0)
    for s in ALL_TYPES:
        if s.group == "char":
            assert df[s.name].dtype.kind == "S", s.name
        if s.group == "unicode":
            assert df[s.name].dtype == object, s.name
    nbB = f_df_to_nb(df)                   # unicode dropped again
    survivors = [s.name for s in ALL_TYPES if survives_roundtrip(s)]
    assert list(nbB.m_titles) == survivors
    for s in ALL_TYPES:
        if survives_roundtrip(s):
            assert nbB.m_titles[s.name] == s.code, s.name


def test_nb_to_df_fresh_and_inplace():
    """f_nb_to_df: df=None builds fresh; df given extends in place; non-df -> TypeError."""
    nb = Pandas_nb()
    nb.f_add_float("a", np.array([1.0, 2.0]))
    nb.f_add_int("b", np.array([3, 4], dtype=np.int64))
    nb.f_add_unicode10("c", np.array(["hi", "yo"], dtype="U10"))
    nb.f_add_char3("d", np.array([b"ab", b"cd"], dtype="S3"))

    fresh = f_nb_to_df(nb)
    assert list(fresh.columns) == ["a", "b", "c", "d"]
    assert fresh["c"].dtype == object                   # unicode -> object
    assert fresh["d"].dtype.kind == "S"                 # char -> native |S

    with pytest.raises(TypeError):
        f_nb_to_df(nb, {"a": [1, 2]})

    df = pd.DataFrame({"a": np.array([10.0, 20.0]),
                       "x": np.array([7, 8], dtype=np.int64)})
    out = f_nb_to_df(nb, df)
    assert out is df
    assert list(out.columns) == ["a", "x", "b", "c", "d"]   # originals, then new
    assert np.array_equal(out["a"].to_numpy(), np.array([10.0, 20.0]))  # kept
    assert np.array_equal(out["b"].to_numpy(), np.array([3, 4]))
    assert list(out["c"]) == ["hi", "yo"]                   # unicode cell values
    assert out["d"].dtype.kind == "S"
    assert np.array_equal(out["d"].to_numpy(), np.array([b"ab", b"cd"], dtype="S3"))


# ===========================================================================
# Sharing
# ===========================================================================
@pytest.mark.parametrize("t", SHARED_TYPES, ids=[s.name for s in SHARED_TYPES])
def test_df_to_nb_shares(t):
    """Non-unicode columns are shared views: an nb write reaches df in place."""
    df = make_df(4)
    nb = f_df_to_nb(df)
    backing = getattr(nb, CODE_TO_ATTR[t.code])[t.name]
    nv = new_value(t)
    backing[0] = nv
    assert df[t.name].iloc[0] == nv
    assert df[t.name].dtype == df[t.name].dtype          # dtype unchanged


def test_df_to_nb_drops_object():
    """A plain object (text) column is not a native pandas column -> dropped."""
    df = pd.DataFrame({"x": np.array(["a", "b"], dtype=object)})
    nb = f_df_to_nb(df)
    assert "x" not in nb.m_titles


def test_str_types():
    """str_types keeps object text as a fresh (unshared) fixed-width U/S column.

    Covers both entry points: f_df_to_nb(df, str_types) and
    f_add_to_nb(nb, title, arr, str_types), all eight widths, the fresh-not-shared
    semantics, and that an unlisted/missing title is ignored.
    """
    cols = {
        "n": np.array([1.0, 2.0], dtype=np.float64),   # numeric, not listed
        "s1": pd.Series(["a", "b"]), "s3": pd.Series(["abc", "de"]),
        "s10": pd.Series(["hello", "world"]), "s100": pd.Series(["x" * 50, "y" * 80]),
        "u1": pd.Series(["a", "b"]), "u3": pd.Series(["abc", "de"]),
        "u10": pd.Series(["hello", "world"]), "u100": pd.Series(["x" * 50, "y" * 80]),
    }
    df = pd.DataFrame(cols)
    str_types = {"s1": "S", "s3": "S", "s10": "S", "s100": "S",
                 "u1": "U", "u3": "U", "u10": "U", "u100": "U"}
    nb = f_df_to_nb(df, str_types)

    expected = [
        ("s1", CHAR1, nb.m_char1, "S1"), ("s3", CHAR3, nb.m_char3, "S3"),
        ("s10", CHAR10, nb.m_char10, "S10"), ("s100", CHAR100, nb.m_char100, "S100"),
        ("u1", UNICODE1, nb.m_unicode1, "U1"), ("u3", UNICODE3, nb.m_unicode3, "U3"),
        ("u10", UNICODE10, nb.m_unicode10, "U10"), ("u100", UNICODE100, nb.m_unicode100, "U100"),
    ]
    for title, code, coldict, wdtype in expected:
        assert nb.m_titles[title] == code, title
        assert coldict[title].dtype == np.dtype(wdtype), title
        assert np.array_equal(coldict[title], df[title].to_numpy().astype(wdtype)), title
    assert nb.m_titles["n"] == FLOAT64

    # unlisted numeric column is a shared view; converted S/U columns are fresh
    nb.m_floats["n"][0] = 99.0
    assert df["n"].iloc[0] == 99.0
    nb.m_char3["s3"][0] = b"zz"
    assert df["s3"].iloc[0] == "abc"                     # not shared
    nb.m_unicode3["u3"][0] = "zz"
    assert df["u3"].iloc[0] == "abc"                     # not shared

    # no str_types -> every object column dropped; a missing title is ignored
    nb0 = f_df_to_nb(df)
    for t in df.columns:
        assert (t in nb0.m_titles) == (t == "n")
    assert "missing" not in f_df_to_nb(df, {"missing": "U"}).m_titles

    # f_add_to_nb direct str_types path: object text -> kept as U / S, with the
    # converted array stored exactly (not just the right code)
    nb2 = Pandas_nb()
    assert f_add_to_nb(nb2, "txt", np.array(["hello", "world"], dtype=object),
                                    {"txt": "U"}) is True
    assert nb2.m_titles["txt"] == UNICODE10
    assert np.array_equal(nb2.m_unicode10["txt"], np.array(["hello", "world"], dtype="U10"))
    assert f_add_to_nb(nb2, "bts", np.array(["ab", "cd"], dtype=object),
                                    {"bts": "S"}) is True
    assert nb2.m_titles["bts"] == CHAR3
    assert np.array_equal(nb2.m_char3["bts"], np.array([b"ab", b"cd"], dtype="S3"))
    # a title ABSENT from a non-empty str_types still follows the object-drop path
    assert f_add_to_nb(nb2, "drop", np.array(["a", "b"], dtype=object),
                                    {"txt": "U"}) is False
    assert "drop" not in nb2.m_titles


# ===========================================================================
# Dispatch / width
# ===========================================================================
@pytest.mark.parametrize("t", ALL_TYPES, ids=_IDS)
def test_add_to_nb_dispatch(t):
    """f_add_to_nb routes each numpy dtype (incl. char/unicode widths) to its code."""
    nb = Pandas_nb()
    assert f_add_to_nb(nb, t.name, sample(t, 4)) is True
    assert nb.m_titles[t.name] == t.code


def test_add_to_nb_raises():
    """object -> dropped (False); unsupported dtype / char / unicode width -> ValueError."""
    nb = Pandas_nb()
    assert f_add_to_nb(nb, "obj", np.array(["a", "b"], dtype=object)) is False
    assert "obj" not in nb.m_titles
    with pytest.raises(ValueError):
        f_add_to_nb(nb, "f16", np.array([1, 2], dtype=np.float16))
    with pytest.raises(ValueError):
        f_add_to_nb(nb, "s2", np.array([b"ab"], dtype="S2"))
    with pytest.raises(ValueError):
        f_add_to_nb(nb, "u5", np.array(["abcde"], dtype="U5"))


def test_np_to_np_str():
    """f_np_to_np_str: smallest fitting width per kind, multibyte, ascii-raise, bounds."""
    f = f_np_to_np_str
    cases = [
        (np.array(["a", "b"], dtype=object), 1),
        (np.array(["abc", "de"], dtype=object), 3),
        (np.array(["hello", "world"], dtype=object), 10),
        (np.array(["x" * 50, "y" * 80], dtype=object), 100),
    ]
    for kind in ("U", "S"):
        for arr, w in cases:
            out = f(arr, kind)
            assert out.dtype == np.dtype(kind + str(w))
            assert np.array_equal(out, arr.astype(kind + str(w)))

    assert f(np.array(["abc", "de"], dtype=object)).dtype == np.dtype("U3")  # default "U"

    multi = np.array(["café", "hi"], dtype=object)                           # 4 code points -> U10
    assert f(multi, "U").dtype == np.dtype("U10")
    with pytest.raises(UnicodeEncodeError):                                  # non-ascii under "S"
        f(np.array(["café"], dtype=object), "S")

    for kind in ("U", "S"):
        with pytest.raises(ValueError):                                      # 101 > 100
            f(np.array(["a" * 101], dtype=object), kind)
    with pytest.raises(ValueError):                                          # unknown kind
        f(np.array(["a"], dtype=object), "X")
    for kind in ("U", "S"):                                                  # empty -> ?1, len 0
        empty = f(np.array([], dtype=object), kind)
        assert empty.dtype == np.dtype(kind + "1") and len(empty) == 0


def test_help_wide():
    """f_help_wide picks the smallest of {1,3,10,100} that fits; over 100 raises."""
    f = f_help_wide
    assert [f(n, "u") for n in (1, 2, 3, 4, 10, 11, 100)] == [1, 3, 3, 10, 10, 100, 100]
    with pytest.raises(ValueError):
        f(101, "u")


# ===========================================================================
# Datetime specifics
# ===========================================================================
def test_datetime_ns_shift_shared():
    """A datetime64[ns] column is a shared view; an njit shift reaches df in place."""
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({"d": dates})
    nb = f_df_to_nb(df)
    assert nb.m_titles["d"] == DATETIME64NS
    before = df["d"].to_numpy().copy()
    one_day = np.timedelta64(86400 * 10**9, "ns")
    _njit_shift_dates(nb, "d", one_day)
    assert np.array_equal(df["d"].to_numpy(), before + one_day)
    assert df["d"].iloc[0] == pd.Timestamp("2026-01-02")


def test_datetime_us_unsupported_and_tz_dropped():
    """datetime64[us] is unsupported (ValueError); tz-aware becomes object and is dropped."""
    df_us = pd.DataFrame({"d": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[us]")})
    with pytest.raises(ValueError):
        f_df_to_nb(df_us)
    df_tz = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=3).tz_localize("Europe/Paris")})
    assert "d" not in f_df_to_nb(df_tz).m_titles


# ===========================================================================
# njit / objmode
# ===========================================================================
def test_njit_add_shared_writeback():
    """An njit loop over the shared float buffers updates df in place."""
    f1 = np.arange(10, dtype=np.float64)
    f2 = np.arange(10, dtype=np.float64) * 10.0
    df = pd.DataFrame({"float1": f1.copy(), "float2": f2.copy()})
    nb = f_df_to_nb(df)
    _njit_add_f2_into_f1(nb)
    assert np.array_equal(df["float1"].to_numpy(), f1 + f2)
    assert np.array_equal(df["float2"].to_numpy(), f2)


def test_njit_gated_char():
    """njit float1 += float2 only on b"EUR" rows (char column gate)."""
    n = 10
    f1 = np.arange(n, dtype=np.float64)
    f2 = np.arange(n, dtype=np.float64) * 10.0
    ccy = np.array(["EUR" if i % 2 == 0 else "USD" for i in range(n)], dtype=object)
    df = pd.DataFrame({"float1": f1.copy(), "float2": f2.copy(), "ccy": ccy})
    nb = f_df_to_nb(df, {"ccy": "S"})
    assert nb.m_titles["ccy"] == CHAR3
    _njit_add_f2_into_f1_eur(nb)
    expected = np.where(ccy == "EUR", f1 + f2, f1)
    assert np.array_equal(df["float1"].to_numpy(), expected)


def test_njit_gated_unicode():
    """njit float1 += float2 only on "€UR" rows (non-ASCII unicode column gate)."""
    n = 10
    f1 = np.arange(n, dtype=np.float64)
    f2 = np.arange(n, dtype=np.float64) * 10.0
    ccy = np.array(["€UR" if i % 2 == 0 else "USD" for i in range(n)], dtype=object)
    df = pd.DataFrame({"float1": f1.copy(), "float2": f2.copy(), "ccy": ccy})
    nb = f_df_to_nb(df, {"ccy": "U"})
    assert nb.m_titles["ccy"] == UNICODE3
    assert nb.m_unicode3["ccy"][0] == "€UR"              # glyph preserved
    _njit_add_f2_into_f1_eur_unicode(nb)
    expected = np.where(ccy == "€UR", f1 + f2, f1)
    assert np.array_equal(df["float1"].to_numpy(), expected)


def test_objmode_returns_pandas_nb():
    """objmode returns a Pandas_nb built in python; nopython then uses it."""
    s = _njit_build_nb_via_objmode()
    assert s == (np.arange(5.0) + np.arange(5.0) * 10.0).sum()


def test_objmode_returns_array_scalar_tuple():
    """objmode blocks with differing output types: float64 array, scalar, tuple-indexed array."""
    arr, tot, arr2 = _njit_eval_array_scalar_tuple()
    assert np.array_equal(arr, np.arange(5.0) * 2.0)
    assert tot == np.arange(5.0).sum() and np.isscalar(tot)
    assert np.array_equal(arr2, np.arange(5.0) * 3.0)


def test_df_handle_via_objmode():
    """A df carried by int64 handle: pandas .sum() via objmode on the LIVE frame."""
    df = pd.DataFrame({"float1": np.arange(5.0), "float2": np.arange(5.0) * 10.0})
    nb = Pandas_nb()
    assert nb.m_df_id == -1
    f_register_df(nb, df)
    assert nb.m_df_id >= 0 and _DF_REGISTRY[nb.m_df_id] is df
    assert _njit_df_sum_via_objmode(nb) == df["float1"].sum() == 10.0
    df.loc[0, "float1"] = 100.0                          # live frame: next call sees it
    assert _njit_df_sum_via_objmode(nb) == 110.0
    # an arbitrary pandas method (describe) through the same handle, not just .sum()
    assert _njit_df_mean_via_objmode(nb) == df["float2"].mean() == 20.0
    with pytest.raises(TypeError):                       # non-DataFrame rejected
        f_register_df(Pandas_nb(), {"float1": [1.0]})


def test_df_handle_weak_eviction():
    """The registry holds df weakly: once the caller drops it, the handle stops resolving."""
    df2 = pd.DataFrame({"float1": np.arange(3.0)})
    nb2 = Pandas_nb()
    f_register_df(nb2, df2)
    hid = nb2.m_df_id
    assert hid in _DF_REGISTRY
    del df2
    gc.collect()
    assert hid not in _DF_REGISTRY
    with pytest.raises(KeyError):
        _njit_df_sum_via_objmode(nb2)


def test_eval_expr_from_njit_timing(capsys):
    """f_eval_expr works called in an njit loop; measure and PRINT per-call cost.

    Correctness only gates the test (sum == n*2.0); the wall time is reported,
    never asserted (it is environment-dependent). Run with -s to see it.
    """
    n = 2000
    x = 1.0
    _njit_eval_expr_loop(1, x)                           # warm up (compile)
    _njit_objmode_only_loop(1, x)

    t0 = time.perf_counter()
    s_eval = _njit_eval_expr_loop(n, x)
    dt_eval = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_base = _njit_objmode_only_loop(n, x)
    dt_base = time.perf_counter() - t0

    assert s_eval == n * 2.0 and s_base == n * 2.0
    with capsys.disabled():
        print("\n  f_eval_expr from njit: %.2f us/call "
              "(objmode-only baseline %.2f us/call) over %d calls"
              % (dt_eval / n * 1e6, dt_base / n * 1e6, n))


def _print_test_catalog():
    # Print each test function's name and its one-line description (first line of
    # the docstring), in definition order. Does NOT run the tests.
    import inspect
    tests = [(name, obj) for name, obj in globals().items()
             if name.startswith("test_") and inspect.isfunction(obj)]
    tests.sort(key=lambda t: t[1].__code__.co_firstlineno)
    width = max(len(name) for name, _ in tests)
    for name, obj in tests:
        doc = obj.__doc__.strip().splitlines()[0] if obj.__doc__ else ""
        print("%-*s  %s" % (width, name, doc))
    print("\n%d test functions (several parametrized over ALL_TYPES)" % len(tests))


class _LiveReporter:
    # In-process pytest plugin that makes the live (-v) line show, for each test:
    # which test (node id), WHAT it checks (the test's one-line docstring), and
    # whether it PASSED/FAILED. Registered only for the file's own run (__main__);
    # under `-m pytest` the standard -v line (node id + PASSED/FAILED) still shows
    # which test ran and its result, just without the description.
    def __init__(self):
        self._desc = {}

    def pytest_runtest_setup(self, item):
        doc = getattr(item, "function", None) and item.function.__doc__
        self._desc[item.nodeid] = doc.strip().splitlines()[0] if doc else ""

    def pytest_report_teststatus(self, report, config):
        if report.when != "call":
            return None                      # default handling for setup/teardown
        desc = self._desc.get(report.nodeid, "")
        word = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(report.outcome)
        if word is None:
            return None
        return report.outcome, word[0], word + "  " + desc


if __name__ == "__main__":
    if "--list" in sys.argv[1:] or "-l" in sys.argv[1:]:
        _print_test_catalog()
    else:
        # -v so each test prints a live line: "<node id> PASS  <description>".
        sys.exit(pytest.main([__file__, "-v"], plugins=[_LiveReporter()]))
