# pandas_numba.py
#
# A numba jitclass "Pandas_nb" mimicking part of python pandas.
# Columns are numpy arrays keyed by title. numba typed dicts are homogeneous
# in their value type, so each numpy element type gets its own dict; m_titles
# records which dict each title belongs to, via an int64 type code.
#
# Array creation inside @njit: a numeric column (float/int/bool/complex/
# datetime) can be built from scratch inside a jitted routine (np.zeros,
# np.empty, np.arange, np.array([...]), ...) and handed to add_*. A unicode/
# char column cannot -- numba can read/store U/S arrays but not allocate one
# in nopython code -- so it must be allocated in plain python (np.empty(n,
# dtype="U10") or np.array([...], dtype="U10")) and then passed in.
#
# Carrying a pandas.DataFrame: a DataFrame has no numba type, so it cannot be a
# jitclass member, an njit argument, or a tuple element -- it only exists inside
# an objmode (python) block. To let a Pandas_nb "carry" a source df through
# nopython code, the df stays in a python-side registry (_DF_REGISTRY) and the
# jitclass stores only an int64 handle (m_df_id). f_register_df(nb, df) attaches
# one; an objmode callback resolves it (DF[params.m_df_id]) and runs any pandas
# call on it. The registry holds the df WEAKLY (a WeakValueDictionary), so the
# entry is evicted automatically once the caller drops the df -- no manual
# unregister, but the caller must keep the df alive while the handle is used.
# See _njit_df_sum_via_objmode / test_df_handle_via_objmode (both in
# pandas_numba_tests.py).
#
# Design note: we researched prior art (Intel SDC -- archived; Bodo -- active;
# and numba.experimental.structref, which SDC/Bodo build on). structref would add
# caching/AOT and weak-referenceable instances, but at the cost of heavy
# boilerplate and methods no longer callable directly from python. We keep the
# light jitclass design on purpose. See pandas_numba.md for the full writeup.
#
# Converters (module-level functions) Pandas_nb <-> pandas.DataFrame. Char columns
# cross as native fixed-width |S<n>; unicode cannot be a native pandas column
# (text is object dtype), so unicode (U*) columns are emitted as object and dropped
# by f_df_to_nb -- only numeric and char survive the round-trip.
#
# Convention: all changes must be tested. The official suite is the pytest file
# pandas_numba_tests.py (python -m pytest code/pandas_numba_tests.py) --
# add a column type by adding one row to its ALL_TYPES table.

"""Pandas-style typed columns usable inside numba ``@njit`` nopython code.

This module provides :class:`Pandas_nb`, a numba ``jitclass`` that holds one
numpy array per column (keyed by title), a small pandas bridge
(:func:`f_df_to_nb` / :func:`f_nb_to_df` and friends), and objmode glue
(:func:`f_register_df` + :func:`f_eval_expr`) for calling back into
Python/pandas from jitted code.

See ``pandas_numba.md`` for the full data model, round-trip rules, and the
numba constraints that shape the design. The pytest suite in
``pandas_numba_tests.py`` is driven by one ``ALL_TYPES`` table; add a column
type by adding one row.
"""

import weakref

import numpy as np
import pandas as pd
from numba import (
    boolean,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
    types,
)
from numba.typed import Dict
from numba.experimental import jitclass

# type codes stored in m_titles
FLOAT64 = 0
INT64 = 1
BOOL = 2
INT8 = 3
INT16 = 4
INT32 = 5
UINT8 = 6
UINT16 = 7
UINT32 = 8
UINT64 = 9
FLOAT32 = 10
COMPLEX64 = 11
COMPLEX128 = 12
DATETIME64NS = 13
TIMEDELTA64NS = 14
CHAR1 = 15
CHAR3 = 16
CHAR10 = 17
CHAR100 = 18
UNICODE1 = 19
UNICODE3 = 20
UNICODE10 = 21
UNICODE100 = 22

# type code -> the Pandas_nb member dict that stores columns of that code.
# Used by f_nb_to_np (and f_nb_to_df) to pick the backing dict by
# code instead of a long if-elif ladder. Pure-python lookup -- not used in any
# jitted code. CHAR_WIDTH maps the char codes to their native |S width.
CODE_TO_ATTR = {
    FLOAT64: "m_floats", INT64: "m_ints", BOOL: "m_bool",
    INT8: "m_int8", INT16: "m_int16", INT32: "m_int32",
    UINT8: "m_uint8", UINT16: "m_uint16", UINT32: "m_uint32", UINT64: "m_uint64",
    FLOAT32: "m_float32", COMPLEX64: "m_complex64", COMPLEX128: "m_complex128",
    DATETIME64NS: "m_datetime64ns", TIMEDELTA64NS: "m_timedelta64ns",
    CHAR1: "m_char1", CHAR3: "m_char3", CHAR10: "m_char10", CHAR100: "m_char100",
    UNICODE1: "m_unicode1", UNICODE3: "m_unicode3",
    UNICODE10: "m_unicode10", UNICODE100: "m_unicode100",
}
CHAR_WIDTH = {CHAR1: 1, CHAR3: 3, CHAR10: 10, CHAR100: 100}

# array types that must be built by a call (can't be constructed inside njit):
# fixed-width unicode / char (width is part of the type) and datetime/timedelta.
char1_t = types.Array(types.CharSeq(1), 1, "A")
char3_t = types.Array(types.CharSeq(3), 1, "A")
char10_t = types.Array(types.CharSeq(10), 1, "A")
char100_t = types.Array(types.CharSeq(100), 1, "A")
unicode1_t = types.Array(types.UnicodeCharSeq(1), 1, "A")
unicode3_t = types.Array(types.UnicodeCharSeq(3), 1, "A")
unicode10_t = types.Array(types.UnicodeCharSeq(10), 1, "A")
unicode100_t = types.Array(types.UnicodeCharSeq(100), 1, "A")
dt64ns_t = types.Array(types.NPDatetime("ns"), 1, "A")
td64ns_t = types.Array(types.NPTimedelta("ns"), 1, "A")
# a plain float64 1-D array type, named so it can annotate an objmode return
# (objmode needs a string or a module-level name, not an inline float64[:]).
float64_1d_t = float64[:]  # == types.Array(float64, 1, "A")

spec = [
    ("m_floats", types.DictType(types.unicode_type, float64[:])),
    ("m_ints", types.DictType(types.unicode_type, int64[:])),
    ("m_bool", types.DictType(types.unicode_type, boolean[:])),
    ("m_int8", types.DictType(types.unicode_type, int8[:])),
    ("m_int16", types.DictType(types.unicode_type, int16[:])),
    ("m_int32", types.DictType(types.unicode_type, int32[:])),
    ("m_uint8", types.DictType(types.unicode_type, uint8[:])),
    ("m_uint16", types.DictType(types.unicode_type, uint16[:])),
    ("m_uint32", types.DictType(types.unicode_type, uint32[:])),
    ("m_uint64", types.DictType(types.unicode_type, uint64[:])),
    ("m_float32", types.DictType(types.unicode_type, float32[:])),
    ("m_complex64", types.DictType(types.unicode_type, complex64[:])),
    ("m_complex128", types.DictType(types.unicode_type, complex128[:])),
    ("m_datetime64ns", types.DictType(types.unicode_type, dt64ns_t)),
    ("m_timedelta64ns", types.DictType(types.unicode_type, td64ns_t)),
    ("m_char1", types.DictType(types.unicode_type, char1_t)),
    ("m_char3", types.DictType(types.unicode_type, char3_t)),
    ("m_char10", types.DictType(types.unicode_type, char10_t)),
    ("m_char100", types.DictType(types.unicode_type, char100_t)),
    ("m_unicode1", types.DictType(types.unicode_type, unicode1_t)),
    ("m_unicode3", types.DictType(types.unicode_type, unicode3_t)),
    ("m_unicode10", types.DictType(types.unicode_type, unicode10_t)),
    ("m_unicode100", types.DictType(types.unicode_type, unicode100_t)),
    ("m_titles", types.DictType(types.unicode_type, int64)),
    # int64 handle into the python-side _DF_REGISTRY: a source pandas.DataFrame
    # cannot be a jitclass member (no numba type -- "non-precise type pyobject"),
    # so the df stays in python and only this handle crosses nopython. -1 means
    # "no df attached"; f_register_df sets it. See test_df_handle_via_objmode
    # (in pandas_numba_tests.py).
    ("m_df_id", int64),
]


@jitclass(spec)
class Pandas_nb:
    """A numba ``jitclass`` holding pandas-style typed columns.

    Each column is a 1-D numpy array keyed by title. numba typed dicts are
    homogeneous in their value type, so there is **one backing dict per element
    type** (``m_floats``, ``m_ints``, ``m_char3``, ...); ``m_titles[title]``
    stores an ``int64`` type code selecting which dict holds the column.
    ``CODE_TO_ATTR`` maps each code to its dict name.

    Columns are added with one method per type::

        f_add_<type>(title, values=None, error_if_present=True)

    for example :meth:`f_add_float`, :meth:`f_add_int`, :meth:`f_add_char3`,
    :meth:`f_add_unicode10`. The contract is uniform across types:

    * ``values=None`` creates an empty column -- but only for numeric families
      (float/int/bool/complex/datetime); a fixed-width char/unicode column
      **cannot be allocated inside numba**, so ``values`` is required there and
      ``None`` raises ``ValueError``.
    * ``error_if_present=True`` rejects a duplicate title; pass ``False`` to
      overwrite. jitclass methods take **positional arguments only**.

    Supported column families and their backing dicts:

    ===========  ==========================================  ==================================
    Family       dtypes                                      backing dict(s)
    ===========  ==========================================  ==================================
    float        float64, float32                            ``m_floats``, ``m_float32``
    int          int64/32/16/8, uint64/32/16/8               ``m_ints``, ``m_int32``, ... ``m_uint8``
    bool         bool\\_                                       ``m_bool``
    complex      complex128, complex64                       ``m_complex128``, ``m_complex64``
    datetime     datetime64[ns], timedelta64[ns] (ns only)   ``m_datetime64ns``, ``m_timedelta64ns``
    char (S)     S1, S3, S10, S100 (bytes)                   ``m_char1``, ``m_char3``, ...
    unicode (U)  U1, U3, U10, U100 (code points)             ``m_unicode1``, ``m_unicode3``, ...
    ===========  ==========================================  ==================================

    Fixed-width text comes in exactly four widths: **1, 3, 10, 100**. The
    ``int64`` member ``m_df_id`` optionally carries a source DataFrame by handle
    (see :func:`f_register_df`); ``-1`` means none.

    Instances are created with no arguments (``nb = Pandas_nb()``) and all
    dicts start empty. Prefer the module-level :func:`f_df_to_nb` to build one
    from a DataFrame.
    """

    def __init__(self):
        self.m_floats = Dict.empty(types.unicode_type, float64[:])
        self.m_ints = Dict.empty(types.unicode_type, int64[:])
        self.m_bool = Dict.empty(types.unicode_type, boolean[:])
        self.m_int8 = Dict.empty(types.unicode_type, int8[:])
        self.m_int16 = Dict.empty(types.unicode_type, int16[:])
        self.m_int32 = Dict.empty(types.unicode_type, int32[:])
        self.m_uint8 = Dict.empty(types.unicode_type, uint8[:])
        self.m_uint16 = Dict.empty(types.unicode_type, uint16[:])
        self.m_uint32 = Dict.empty(types.unicode_type, uint32[:])
        self.m_uint64 = Dict.empty(types.unicode_type, uint64[:])
        self.m_float32 = Dict.empty(types.unicode_type, float32[:])
        self.m_complex64 = Dict.empty(types.unicode_type, complex64[:])
        self.m_complex128 = Dict.empty(types.unicode_type, complex128[:])
        self.m_datetime64ns = Dict.empty(types.unicode_type, dt64ns_t)
        self.m_timedelta64ns = Dict.empty(types.unicode_type, td64ns_t)
        self.m_char1 = Dict.empty(types.unicode_type, char1_t)
        self.m_char3 = Dict.empty(types.unicode_type, char3_t)
        self.m_char10 = Dict.empty(types.unicode_type, char10_t)
        self.m_char100 = Dict.empty(types.unicode_type, char100_t)
        self.m_unicode1 = Dict.empty(types.unicode_type, unicode1_t)
        self.m_unicode3 = Dict.empty(types.unicode_type, unicode3_t)
        self.m_unicode10 = Dict.empty(types.unicode_type, unicode10_t)
        self.m_unicode100 = Dict.empty(types.unicode_type, unicode100_t)
        self.m_titles = Dict.empty(types.unicode_type, int64)
        self.m_df_id = -1  # no python df attached yet (see f_register_df)

    def f_add_float(self, title, values=None, error_if_present=True):
        """Add a ``float64`` column (representative of the numeric families).

        Every numeric ``f_add_<type>`` method (float/int/bool/complex/datetime,
        in all widths) follows this exact shape; only the dtype and backing
        dict differ.

        Args:
            title: Column name (used as the dict key and the type-code key).
            values: A 1-D numpy array of the matching dtype, or ``None`` for an
                empty column. Numeric arrays may be built inside ``@njit``.
            error_if_present: If ``True`` (default), raise ``ValueError`` when
                ``title`` already exists; pass ``False`` to overwrite.

        Raises:
            ValueError: If ``error_if_present`` and ``title`` is already present.
        """
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.float64)
        else:
            vals = values
        self.m_floats[title] = vals
        self.m_titles[title] = FLOAT64

    def f_add_int(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.int64)
        else:
            vals = values
        self.m_ints[title] = vals
        self.m_titles[title] = INT64

    def f_add_bool(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.bool_)
        else:
            vals = values
        self.m_bool[title] = vals
        self.m_titles[title] = BOOL

    def f_add_int8(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.int8)
        else:
            vals = values
        self.m_int8[title] = vals
        self.m_titles[title] = INT8

    def f_add_int16(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.int16)
        else:
            vals = values
        self.m_int16[title] = vals
        self.m_titles[title] = INT16

    def f_add_int32(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.int32)
        else:
            vals = values
        self.m_int32[title] = vals
        self.m_titles[title] = INT32

    def f_add_uint8(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.uint8)
        else:
            vals = values
        self.m_uint8[title] = vals
        self.m_titles[title] = UINT8

    def f_add_uint16(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.uint16)
        else:
            vals = values
        self.m_uint16[title] = vals
        self.m_titles[title] = UINT16

    def f_add_uint32(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.uint32)
        else:
            vals = values
        self.m_uint32[title] = vals
        self.m_titles[title] = UINT32

    def f_add_uint64(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.uint64)
        else:
            vals = values
        self.m_uint64[title] = vals
        self.m_titles[title] = UINT64

    def f_add_float32(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.float32)
        else:
            vals = values
        self.m_float32[title] = vals
        self.m_titles[title] = FLOAT32

    def f_add_complex64(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.complex64)
        else:
            vals = values
        self.m_complex64[title] = vals
        self.m_titles[title] = COMPLEX64

    def f_add_complex128(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.complex128)
        else:
            vals = values
        self.m_complex128[title] = vals
        self.m_titles[title] = COMPLEX128

    def f_add_datetime64ns(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.dtype('datetime64[ns]'))
        else:
            vals = values
        self.m_datetime64ns[title] = vals
        self.m_titles[title] = DATETIME64NS

    def f_add_timedelta64ns(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            vals = np.empty(0, dtype=np.dtype('timedelta64[ns]'))
        else:
            vals = values
        self.m_timedelta64ns[title] = vals
        self.m_titles[title] = TIMEDELTA64NS

    def f_add_char1(self, title, values=None, error_if_present=True):
        """Add a width-1 char (``|S1``) column (representative of text families).

        Every fixed-width text ``f_add_<type>`` method (char ``|S`` and unicode
        ``|U``, widths 1/3/10/100) follows this exact shape. Unlike the numeric
        methods, ``values`` is **required**: numba cannot allocate a fixed-width
        char/unicode array in nopython code, so it must be built in Python
        (e.g. ``np.empty(n, dtype='S1')``) and passed in.

        Args:
            title: Column name.
            values: A 1-D numpy array of the matching fixed-width dtype. Passing
                ``None`` raises ``ValueError``.
            error_if_present: If ``True`` (default), raise ``ValueError`` when
                ``title`` already exists; pass ``False`` to overwrite.

        Raises:
            ValueError: If ``error_if_present`` and ``title`` is already present,
                or if ``values`` is ``None``.
        """
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='S1')) and pass it in")
        self.m_char1[title] = values
        self.m_titles[title] = CHAR1

    def f_add_char3(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='S3')) and pass it in")
        self.m_char3[title] = values
        self.m_titles[title] = CHAR3

    def f_add_char10(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='S10')) and pass it in")
        self.m_char10[title] = values
        self.m_titles[title] = CHAR10

    def f_add_char100(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='S100')) and pass it in")
        self.m_char100[title] = values
        self.m_titles[title] = CHAR100

    def f_add_unicode1(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='U1')) and pass it in")
        self.m_unicode1[title] = values
        self.m_titles[title] = UNICODE1

    def f_add_unicode3(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='U3')) and pass it in")
        self.m_unicode3[title] = values
        self.m_titles[title] = UNICODE3

    def f_add_unicode10(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='U10')) and pass it in")
        self.m_unicode10[title] = values
        self.m_titles[title] = UNICODE10

    def f_add_unicode100(self, title, values=None, error_if_present=True):
        if error_if_present and title in self.m_titles:
            raise ValueError("title already present")
        if values is None:
            raise ValueError("cannot allocate a fixed-width unicode/char column inside numba; allocate it in python (e.g. np.empty(n, dtype='U100')) and pass it in")
        self.m_unicode100[title] = values
        self.m_titles[title] = UNICODE100


# Module-level converters between Pandas_nb and a real pandas.DataFrame.
# Char columns cross as native fixed-width |S<n> (n in 1/3/10/100), read back
# by itemsize. Unicode cannot be a native pandas column (text is object), so
# unicode (U*) columns are emitted as object and dropped by f_df_to_nb -- the
# round-trip preserves numeric and char only.


def f_nb_to_np(nb, title):
    """Return the numpy array backing one column of a :class:`Pandas_nb`.

    The column is selected by its stored ``int64`` type code via
    ``CODE_TO_ATTR``, which names the backing dict.

    Args:
        nb: The :class:`Pandas_nb` to read from.
        title: The column title.

    Returns:
        The 1-D numpy array backing the column.

    Raises:
        ValueError: If the column's stored type code is unknown.
    """
    code = nb.m_titles[title]
    attr = CODE_TO_ATTR.get(code)
    if attr is None:
        raise ValueError("unknown type code " + str(code) + " for column " + str(title))
    return getattr(nb, attr)[title]

def f_nb_to_df(nb, df=None):
    """Convert a :class:`Pandas_nb` to a pandas DataFrame.

    Char columns are forced back to their native fixed-width ``|S<width>`` so a
    later :func:`f_df_to_nb` reads the width directly. Unicode columns are
    emitted as object dtype (pandas has no native fixed-width unicode column),
    so they do not survive a second round-trip.

    Args:
        nb: The :class:`Pandas_nb` to convert.
        df: If ``None`` (default), build and return a fresh DataFrame. If a
            DataFrame is given, add to it only the ``nb`` columns whose titles
            are not already columns of ``df`` and return it; existing columns
            are left untouched.

    Returns:
        The fresh or augmented :class:`pandas.DataFrame`.

    Raises:
        TypeError: If ``df`` is neither ``None`` nor a :class:`pandas.DataFrame`.
    """
    if df is not None and not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame or None")
    data = {}
    char_widths = {}  # title -> width, for forcing native |S columns
    for title in nb.m_titles:
        data[title] = f_nb_to_np(nb, title)
        code = nb.m_titles[title]
        if code in CHAR_WIDTH:
            char_widths[title] = CHAR_WIDTH[code]
    # pandas coerces unicode/char arrays to object on construction; unicode
    # (|U) stays object (pandas cannot hold it), but char arrays are forced back
    # to their native |S<width> so f_df_to_nb reads the width directly.
    if df is None:
        out = pd.DataFrame(data)
        for title, w in char_widths.items():
            out[title] = out[title].astype("S" + str(w))
        return out
    for title, arr in data.items():
        if title not in df.columns:
            df[title] = arr
            if title in char_widths:
                df[title] = df[title].astype("S" + str(char_widths[title]))
    return df

def f_add_to_nb(nb, title, np_arr, str_types={}):
    """Add one numpy array to ``nb`` as the column type matching its dtype.

    The dtype is taken as-is and routed to the matching ``f_add_<type>``
    method.

    Args:
        nb: The :class:`Pandas_nb` to add to.
        title: The column title.
        np_arr: The 1-D numpy array to add.
        str_types: Optional ``{title: kind}`` with ``kind`` ``"U"`` or ``"S"``
            (same meaning as in :func:`f_df_to_nb`). If ``title`` is listed,
            ``np_arr`` is first run through :func:`f_np_to_np_str` to a
            fixed-width unicode (U) or char (S) array, so an object text array
            is **kept** as a U/S column instead of being dropped. The default
            ``{}`` is only ever read, so sharing one instance is safe.

    Returns:
        ``True`` if a column was added, ``False`` if the array was dropped
        (object dtype cannot be a native pandas column; see :func:`f_df_to_nb`).

    Raises:
        ValueError: On an unsupported dtype or char/unicode width.

    Note:
        On Windows a bare ``np.array([1, 2, 3])`` is ``int32`` on win64 (not
        ``int64``), so it routes to ``INT32`` / ``m_int32``. pandas columns are
        ``int64``, so :func:`f_df_to_nb` is unaffected; this only bites arrays
        built by hand without an explicit dtype.
    """
    if title in str_types:
        np_arr = f_np_to_np_str(np_arr, str_types[title])
    dt = np_arr.dtype
    if dt == np.float64:
        nb.f_add_float(title, np_arr)
    elif dt == np.int64:
        nb.f_add_int(title, np_arr)
    elif dt == np.bool_:
        nb.f_add_bool(title, np_arr)
    elif dt == np.int8:
        nb.f_add_int8(title, np_arr)
    elif dt == np.int16:
        nb.f_add_int16(title, np_arr)
    elif dt == np.int32:
        nb.f_add_int32(title, np_arr)
    elif dt == np.uint8:
        nb.f_add_uint8(title, np_arr)
    elif dt == np.uint16:
        nb.f_add_uint16(title, np_arr)
    elif dt == np.uint32:
        nb.f_add_uint32(title, np_arr)
    elif dt == np.uint64:
        nb.f_add_uint64(title, np_arr)
    elif dt == np.float32:
        nb.f_add_float32(title, np_arr)
    elif dt == np.complex64:
        nb.f_add_complex64(title, np_arr)
    elif dt == np.complex128:
        nb.f_add_complex128(title, np_arr)
    elif dt == np.dtype("datetime64[ns]"):
        nb.f_add_datetime64ns(title, np_arr)
    elif dt == np.dtype("timedelta64[ns]"):
        nb.f_add_timedelta64ns(title, np_arr)
    elif dt.kind == "S":
        # native fixed-width char column: width is the itemsize.
        n = np_arr.dtype.itemsize
        if n == 1:
            nb.f_add_char1(title, np_arr)
        elif n == 3:
            nb.f_add_char3(title, np_arr)
        elif n == 10:
            nb.f_add_char10(title, np_arr)
        elif n == 100:
            nb.f_add_char100(title, np_arr)
        else:
            raise ValueError("unsupported char width S" + str(n) + " for column " + str(title))
    elif dt.kind == "U":
        # native fixed-width unicode column: a |U element is 4 bytes per
        # code point, so the width in code points is itemsize // 4.
        n = np_arr.dtype.itemsize // 4
        if n == 1:
            nb.f_add_unicode1(title, np_arr)
        elif n == 3:
            nb.f_add_unicode3(title, np_arr)
        elif n == 10:
            nb.f_add_unicode10(title, np_arr)
        elif n == 100:
            nb.f_add_unicode100(title, np_arr)
        else:
            raise ValueError("unsupported unicode width U" + str(n) + " for column " + str(title))
    elif dt == object:
        # object columns are not a native pandas text column and map to no
        # nb type; skip them silently (this is how unicode text from a
        # DataFrame, stored as object, is dropped in f_df_to_nb).
        return False
    else:
        raise ValueError("unsupported dtype " + str(dt) + " for column " + str(title))
    return True

def f_df_to_nb(df, str_types={}):
    """Convert a pandas DataFrame to a :class:`Pandas_nb`, one column per column.

    Each column is added by dtype via :func:`f_add_to_nb`. Numeric and native
    char (``|S``) columns become **shared views** of the DataFrame's memory;
    object text columns are dropped unless listed in ``str_types``.

    Args:
        df: The source :class:`pandas.DataFrame`.
        str_types: Optional ``{title: kind}`` with ``kind`` ``"U"`` or ``"S"``.
            For a listed title, the column's array is first run through
            :func:`f_np_to_np_str` to a fixed-width unicode (U) or char (S)
            array, so it is kept as a U/S column instead of being dropped. The
            default ``{}`` is only ever read, so sharing one instance is safe.

    Returns:
        A new :class:`Pandas_nb` holding the convertible columns.

    Note:
        A ``str_types``-listed column is **not** shared with ``df`` --
        :func:`f_np_to_np_str` builds a fresh array (``astype``), so the two no
        longer back the same memory and edits must be copied by hand. Unlisted
        numeric/char columns stay shared views.

    Note:
        The ``str_types`` conversion is applied here and the result is passed to
        :func:`f_add_to_nb` **without** ``str_types``, so a listed column is
        converted exactly once (never double-converted).
    """
    nb = Pandas_nb()
    for title in df.columns:
        arr = df[title].to_numpy()
        if title in str_types:
            arr = f_np_to_np_str(arr, str_types[title])
        f_add_to_nb(nb, title, arr)
    return nb

def f_help_wide(need, unit):
    """Pick the smallest supported fixed width (1, 3, 10, 100) that fits ``need``.

    Args:
        need: The required width in the relevant units.
        unit: A label for the error message only (e.g. ``"bytes"``,
            ``"code points"``).

    Returns:
        The smallest of ``1, 3, 10, 100`` that is ``>= need``.

    Raises:
        ValueError: If ``need`` exceeds the largest supported width (100).
    """
    for w in (1, 3, 10, 100):
        if need <= w:
            return w
    raise ValueError(str(need) + " " + unit
                     + " exceeds largest supported width 100")

def f_np_to_np_str(np_arr, kind="U"):
    """Convert an array to a fixed-width numpy string array.

    Uses the smallest supported width (1, 3, 10, 100) that fits the longest
    value.

    Args:
        np_arr: The source numpy array.
        kind: ``"U"`` for unicode (width counted in code points) or ``"S"`` for
            bytes (width counted in bytes). ``"S"`` encodes with the ascii
            codec, so a non-ascii value raises ``UnicodeEncodeError``.

    Returns:
        A new numpy array of dtype ``<kind><width>`` (e.g. ``U10`` / ``S3``).

    Raises:
        ValueError: On an unknown ``kind``, or if the longest value exceeds the
            largest supported width (100).
    """
    if kind == "U":
        unit, label = 4, "code points"
    elif kind == "S":
        unit, label = 1, "bytes"
    else:
        raise ValueError("kind must be 'U' or 'S', got " + str(kind))
    # astype(kind) sizes the result to the longest element (width 1 for
    # empty input); need is that width in the kind's own units.
    s = np_arr.astype(kind)
    need = s.dtype.itemsize // unit
    width = f_help_wide(need, label)
    return s.astype(kind + str(width))


# ---------------------------------------------------------------------------
# Infrastructure shared by the njit routines and objmode callbacks below:
# the numba type of a Pandas_nb (needed to carry one across objmode), the
# python-side df registry + f_register_df, and the generic f_eval_expr callback.
# ---------------------------------------------------------------------------

# The numba type of a Pandas_nb instance -- needed to carry one across the
# objmode boundary (below). All instances share this single compile-time type.
PANDAS_NB_T = Pandas_nb.class_type.instance_type
# a heterogeneous 2-tuple (Pandas_nb, float64) -- shows objmode can carry a
# mixed bundle in one value (see f_test_help_nb_array_via_objmode).
PANDAS_NB_FLOAT_T = types.Tuple((PANDAS_NB_T, float64))


# python-side registry mapping an int64 handle -> the source pandas.DataFrame.
# The df cannot be a jitclass member or cross nopython (no numba type), so a
# Pandas_nb only stores the int64 handle (m_df_id) and the df stays here, in
# python. An objmode callback (f_eval_expr) resolves the handle via DF[...].
#
# It is a WeakValueDictionary, so the entry is automatically evicted when the df
# is garbage-collected -- this bounds the registry without any manual unregister.
# Automatic cleanup cannot be driven by the Pandas_nb itself: a jitclass supports
# no __del__ ("Method '__del__' is not supported.") and its boxed instance is not
# weak-referenceable ("cannot create weak reference"), so the only lifetime we can
# hang cleanup on is the df's. Consequence of holding the df WEAKLY: the registry
# does NOT keep it alive -- the caller must hold a df reference for as long as the
# handle is used, and a lookup after the df is gone raises KeyError (a handle is
# valid only while its df lives). Integer ids are never reused (monotonic counter),
# so a stale handle can never silently resolve to a different df.
#
# It must also be PICKLABLE: numba's objmode path serializes the callback and the
# globals it reaches (this registry, through f_eval_expr) to check serializability
# -- by value when the module is __main__. A plain WeakValueDictionary holds an
# internal weakref that cannot be pickled and breaks objmode compilation ("cannot
# pickle 'weakref.ReferenceType'"). The contents are never needed across that
# boundary, so __reduce__ serializes it as a fresh empty registry; at runtime
# objmode uses the real module-global instance.
class _WeakDfRegistry(weakref.WeakValueDictionary):
    def __reduce__(self):
        return (_WeakDfRegistry, ())


_DF_REGISTRY = _WeakDfRegistry()
_DF_NEXT_ID = [0]  # 1-element list so the counter can be bumped from a function


def f_register_df(nb, df):
    """Attach a pandas DataFrame to a :class:`Pandas_nb` by handle.

    Stores ``df`` in the module registry under a fresh ``int64`` id and writes
    that id into ``nb.m_df_id``, so the frame can be reached from nopython code
    via an objmode call to :func:`f_eval_expr`. Must be called from plain Python
    (it sets a jitclass member and touches the registry).

    Args:
        nb: The :class:`Pandas_nb` to tag with the handle.
        df: The source :class:`pandas.DataFrame`. It is **not** copied and is
            held **weakly**: :func:`f_eval_expr` sees later edits to it, but the
            caller must keep the frame alive for as long as the handle is used
            (ids are never reused).

    Returns:
        ``nb`` (for chaining).

    Raises:
        TypeError: If ``df`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    hid = _DF_NEXT_ID[0]
    _DF_NEXT_ID[0] += 1
    _DF_REGISTRY[hid] = df
    nb.m_df_id = hid
    return nb


def f_eval_expr(expr, params):
    """Evaluate a Python expression string from inside an objmode block.

    The generic callback that lets jitted code call back into Python/pandas.
    ``expr`` is evaluated with ``np``, ``pd``, the DataFrame registry ``DF``,
    and ``params`` in scope.

    Args:
        expr: A Python expression string, e.g.
            ``"params.m_floats['float1'] * 2.0"`` or, for a frame carried by
            handle, ``"DF[params.m_df_id]['x'].sum()"``.
        params: Any objmode-boxable value referenced as ``params`` in ``expr``
            (a :class:`Pandas_nb`, a numpy array, a float, a tuple of these,
            ...).

    Returns:
        Whatever ``expr`` evaluates to. The enclosing ``objmode`` block must
        declare a matching static output type.
    """
    return eval(expr, {"np": np, "pd": pd, "DF": _DF_REGISTRY, "params": params})
