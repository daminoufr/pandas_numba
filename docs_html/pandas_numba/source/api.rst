API reference
=============

.. currentmodule:: pandas_numba

Each entry below links to its ``[source]`` and carries a collapsible **Example
from tests** — the exact function from the test suite
(`pandas_numba_tests.py`), pulled in live so it never drifts from the code. The
objmode entries show both the ``@njit`` helper and the test that drives it.

The ``Pandas_nb`` jitclass
--------------------------

.. autoclass:: Pandas_nb

.. dropdown:: Example from tests — construct, add a column, read it back
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_add_type
      :language: python

Adding columns
~~~~~~~~~~~~~~~

Columns are added with one method per type, all sharing the signature
``f_add_<type>(title, values=None, error_if_present=True)``. The two methods
below are representative: every **numeric** family follows ``f_add_float`` and
every **fixed-width text** family follows ``f_add_char1``.

.. automethod:: Pandas_nb.f_add_float

.. automethod:: Pandas_nb.f_add_char1

.. dropdown:: Example from tests — f_add_float / f_add_char3 (fresh + in-place DataFrame)
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_nb_to_df_fresh_and_inplace
      :language: python

The full set of ``f_add_*`` methods, by family:

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Family (follows)
     - Methods
   * - numeric (``f_add_float``)
     - ``f_add_float``, ``f_add_float32``, ``f_add_int``, ``f_add_int8``,
       ``f_add_int16``, ``f_add_int32``, ``f_add_uint8``, ``f_add_uint16``,
       ``f_add_uint32``, ``f_add_uint64``, ``f_add_bool``, ``f_add_complex64``,
       ``f_add_complex128``, ``f_add_datetime64ns``, ``f_add_timedelta64ns``
   * - text (``f_add_char1``)
     - ``f_add_char1``, ``f_add_char3``, ``f_add_char10``, ``f_add_char100``,
       ``f_add_unicode1``, ``f_add_unicode3``, ``f_add_unicode10``,
       ``f_add_unicode100``

The pandas bridge
-----------------

.. autofunction:: f_df_to_nb

.. dropdown:: Example from tests — numeric/char columns are shared views of df
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_df_to_nb_shares
      :language: python

.. autofunction:: f_nb_to_df

.. dropdown:: Example from tests — df=None builds fresh; df given extends in place
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_nb_to_df_fresh_and_inplace
      :language: python

.. autofunction:: f_add_to_nb

.. dropdown:: Example from tests — dtype dispatch, and the dropped/raising cases
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_add_to_nb_dispatch
      :language: python

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_add_to_nb_raises
      :language: python

.. autofunction:: f_nb_to_np

.. dropdown:: Example from tests — unknown type code raises ValueError
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_nb_to_np_unknown_code_raises
      :language: python

.. autofunction:: f_np_to_np_str

.. dropdown:: Example from tests — smallest fitting width, multibyte, ascii-raise, bounds
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_np_to_np_str
      :language: python

.. autofunction:: f_help_wide

.. dropdown:: Example from tests — smallest of {1, 3, 10, 100} that fits
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_help_wide
      :language: python

Carrying a DataFrame by handle (objmode)
----------------------------------------

.. autofunction:: f_register_df

.. autofunction:: f_eval_expr

.. dropdown:: Example from tests — pandas on the live frame via an int64 handle
   :class-title: example-dropdown
   :color: muted

   The ``@njit`` side (``objmode`` hops back into Python and calls
   ``f_eval_expr``):

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: _njit_df_sum_via_objmode
      :language: python

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: _njit_df_mean_via_objmode
      :language: python

   The test that registers the frame and drives both calls:

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: test_df_handle_via_objmode
      :language: python

.. dropdown:: Example from tests — f_eval_expr with three different objmode output types
   :class-title: example-dropdown
   :color: muted

   .. literalinclude:: ../../../tests/pandas_numba_tests.py
      :pyobject: _njit_eval_array_scalar_tuple
      :language: python
