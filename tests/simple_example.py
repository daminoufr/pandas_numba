# Homepage:   https://pypi.org/project/pandas-numba/
# Repository: https://github.com/daminoufr/pandas_numba
# PyPI:       https://pypi.org/project/pandas-numba/
# [MIT](LICENSE) © 2026 Damien Loison

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))  # code/ on path (module is there)

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
