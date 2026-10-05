pandas_numba
============

A numba ``jitclass`` (:class:`~pandas_numba.Pandas_nb`) that holds
DataFrame-like **typed columns usable inside** ``@njit`` **nopython code**, plus
a pandas bridge (:func:`~pandas_numba.f_df_to_nb` /
:func:`~pandas_numba.f_nb_to_df` and friends) and a way to call back into
Python/pandas from jitted code (:func:`~pandas_numba.f_eval_expr` + objmode).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Install
-------

.. code-block:: bat

   pip install pandas_numba

Quick start
-----------

.. code-block:: python

   import numpy as np, pandas as pd
   from numba import njit
   from pandas_numba import Pandas_nb, f_df_to_nb, f_nb_to_df

   df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
   nb = f_df_to_nb(df)          # numeric columns become SHARED VIEWS of df

   @njit
   def double_x(nb):
       col = nb.m_floats["x"]   # the numpy array backing column "x"
       for i in range(col.shape[0]):
           col[i] *= 2.0        # writes straight back into df (shared view)

   double_x(nb)
   print(df["x"].tolist())      # [2.0, 4.0, 6.0]

   out = f_nb_to_df(nb)         # Pandas_nb -> a fresh DataFrame

Data model
----------

:class:`~pandas_numba.Pandas_nb` keeps one numpy array per column, keyed by
title. numba typed dicts are homogeneous in value type, so there is **one dict
per element type** (``m_floats``, ``m_ints``, ``m_char3``, ...);
``m_titles[title]`` stores an ``int64`` type code selecting which dict holds the
column. An ``int64`` ``m_df_id`` optionally carries a source DataFrame by handle
(see below); ``-1`` means none.

.. list-table:: Supported column types
   :widths: 18 46 36
   :header-rows: 1

   * - Family
     - dtypes
     - Backing dict
   * - float
     - float64, float32
     - ``m_floats``, ``m_float32``
   * - int
     - int64/32/16/8, uint64/32/16/8
     - ``m_ints``, ``m_int32``, ... ``m_uint8``
   * - bool
     - bool\_
     - ``m_bool``
   * - complex
     - complex128, complex64
     - ``m_complex128``, ``m_complex64``
   * - datetime
     - datetime64[ns], timedelta64[ns] (ns only)
     - ``m_datetime64ns``, ``m_timedelta64ns``
   * - char (S)
     - S1, S3, S10, S100 (bytes)
     - ``m_char1``, ``m_char3``, ...
   * - unicode (U)
     - U1, U3, U10, U100 (code points)
     - ``m_unicode1``, ``m_unicode3``, ...

Fixed-width text comes in exactly four widths: **1, 3, 10, 100** (bytes for
``|S``, code points for ``|U``). Wider values raise ``ValueError``.
``CODE_TO_ATTR`` maps each code to its dict; ``CHAR_WIDTH`` maps the char codes
to their width.

Sharing and round-trip
----------------------

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Column
     - ``f_df_to_nb``
     - survives ``df → nb → df``?
   * - numeric / ``datetime64[ns]``
     - **shared view** of df
     - yes
   * - native char (S)
     - **shared view** of df
     - yes
   * - object text, listed in ``str_types``
     - fresh copy (not shared)
     - yes (as S/U)
   * - object text / unicode, not listed
     - dropped
     - no

A shared-view column edited in nopython writes straight back into the
DataFrame; a ``str_types``-converted column is a fresh array, so changes must be
copied back by hand. On the way out, char columns are forced back to native
``|S``; unicode is emitted as object (which pandas cannot hold as a native
column), so a second :func:`~pandas_numba.f_df_to_nb` drops it -- numeric and
char survive the round-trip, unicode does not. ``datetime64`` is **ns only**
(us → ``ValueError``); tz-aware datetimes become object and are dropped.

Carrying a DataFrame by handle
------------------------------

A DataFrame has no numba type and cannot cross into nopython.
:func:`~pandas_numba.f_register_df` stores ``df`` in a weak module registry
under a fresh ``int64`` id written to ``nb.m_df_id``. Inside ``@njit``, an
objmode block runs pandas on the live frame via
:func:`~pandas_numba.f_eval_expr`:

.. code-block:: python

   from numba import njit, objmode, float64
   from pandas_numba import f_register_df, f_eval_expr

   @njit
   def df_sum(nb):
       with objmode(tot=float64):
           tot = f_eval_expr("float(DF[params.m_df_id]['x'].sum())", nb)
       return tot

The registry holds ``df`` **weakly** -- it auto-evicts when the caller drops
``df`` (no manual unregister), so the caller must keep ``df`` alive while the
handle is used; ids are never reused.

numba constraints
-----------------

- Numeric arrays **can** be built in nopython (``np.empty``, ``np.arange``, ...)
  and passed to ``f_add_*``. Fixed-width char/unicode arrays **cannot** -- build
  them in Python (``np.empty(n, dtype="U10")``) and pass them in.
- ``np.timedelta64(...)`` **cannot** be constructed in nopython; build the delta
  in Python and pass it in (adding it to a ``datetime64[ns]`` works).
- objmode needs **static output types**; a DataFrame/Series cannot be boxed --
  pass ``.to_numpy()`` or carry the frame by handle.
- Windows: a bare ``np.array([1, 2, 3])`` is **int32** on win64, so it routes to
  ``INT32`` / ``m_int32``. pandas columns are int64, so ``f_df_to_nb`` is
  unaffected.

API reference
-------------

See :doc:`api` for the complete symbol reference.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
