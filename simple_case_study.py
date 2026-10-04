# simple_case_study.py
#
# Case study data: top-of-book (first level) snapshots for a single asset, as a
# pandas DataFrame. One row = one snapshot of the best bid/ask and their sizes.
#
# Columns:
#   timestamp  quote time (datetime64[ns]); starts 2026-09-02 08:00:00, each
#              next row arrives a random 0.1-3 s later (cumulative)
#   bid        best buy price; a gaussian random walk: bid[t] = bid[t-1] +
#              N(0, 0.02), first bid uniform in [100, 105], kept to 2 decimals
#   bid_vol    best buy volume,  integer in [1, 500]
#   ask        best sell price, = bid + spread (2 decimals)
#   ask_vol    best sell volume, integer in [1, 500]
#   flag       random "H" or "V" (string / object column)
#
# Spread (ask - bid) is a tick multiple with a fixed mix: 92% 0.01, 7% 0.02,
# 1% 0.03 (exact counts for n=100, proportional + shuffled for other n).
#
# Starts at 100 rows; f_make_book(n) scales it up. Plain pandas/numpy (data
# generation, not a jitted hot path).

import numpy as np
import pandas as pd
from numba import njit, objmode

# The mean_w_jit indicator (f_indicators_by_nb / f_indicators_jit) is computed in
# an @njit hot loop through the pandas_numba bridge. f_eval_expr is imported
# bare so the objmode block can name it, as f_test_help_nb_df_sum_via_objmode does.
import pandas_numba as pn
from pandas_numba import f_eval_expr

SPREADS = np.array([0.01, 0.02, 0.03])   # the three possible spreads
SPREAD_P = np.array([0.92, 0.07, 0.01])  # their target proportions
PRICE_LO, PRICE_HI = 100.0, 105.0        # bid/ask bounds
VOL_LO, VOL_HI = 1, 500                  # volume bounds (inclusive)
START = np.datetime64("2026-09-02T08:00:00", "ns")  # first quote time
GAP_LO, GAP_HI = 0.1, 3.0                # inter-arrival gap bounds, seconds
PRICE_SIGMA = 0.02                       # std of the per-step gaussian price move


def f_spread_counts(n):
    # Exact per-spread row counts for n rows: round each proportion, then put
    # the rounding remainder on the most common spread (0.01) so they sum to n.
    # For n=100 this is exactly [92, 7, 1].
    counts = np.round(SPREAD_P * n).astype(np.int64)
    counts[0] += n - counts.sum()
    return counts


def f_make_book(n=100, seed=0):
    # Build an n-row top-of-book DataFrame (see module header). Reproducible for
    # a given seed. Spreads follow the 92/7/1 mix exactly; bid is a gaussian
    # random walk (step N(0, PRICE_SIGMA), first bid uniform in [100, 105])
    # kept to 2 decimals, ask = bid + spread; volumes are integers in [1, 500].
    # NB the walk is unbounded, so over many rows bid may drift outside [100,105].
    rng = np.random.default_rng(seed)

    # timestamps: row 0 at START, each later row a random GAP_LO..GAP_HI s after
    # the previous one. gaps are rounded to ns and accumulated.
    gaps_s = np.concatenate(([0.0], rng.uniform(GAP_LO, GAP_HI, n - 1)))
    offset_ns = np.round(np.cumsum(gaps_s) * 1e9).astype(np.int64).astype("timedelta64[ns]")
    timestamp = START + offset_ns

    counts = f_spread_counts(n)
    spread = np.repeat(SPREADS, counts)          # 92x0.01, 7x0.02, 1x0.03, ...
    rng.shuffle(spread)                          # mix them across the rows

    # bid is a gaussian random walk: first bid uniform in [100, 105], then each
    # step adds N(0, PRICE_SIGMA). round to 2 decimals; ask = bid + spread.
    steps = rng.normal(0.0, PRICE_SIGMA, n)
    steps[0] = rng.uniform(PRICE_LO, PRICE_HI)   # row 0 is the starting price
    bid = np.round(np.cumsum(steps), 2)
    ask = np.round(bid + spread, 2)

    bid_vol = rng.integers(VOL_LO, VOL_HI + 1, n)
    ask_vol = rng.integers(VOL_LO, VOL_HI + 1, n)

    flag = rng.choice(np.array(["H", "V"]), n)   # object (text) column

    return pd.DataFrame(
        {"timestamp": timestamp, "bid": bid, "bid_vol": bid_vol,
         "ask": ask, "ask_vol": ask_vol, "flag": flag}
    )


def f_markout_grid(df):
    # Step 1: sample the prevailing book onto a regular 1-second grid. For each
    # whole second, merge_asof (direction="backward") takes the last quote at or
    # before that instant -- the book "as of" the second mark, carrying the last
    # quote forward across seconds with no new quote.
    grid = pd.date_range(df["timestamp"].iloc[0].ceil("s"),
                         df["timestamp"].iloc[-1], freq="1s")
    book = pd.merge_asof(pd.DataFrame({"timestamp": grid}), df,
                         on="timestamp", direction="backward")
    return book


def f_indicators(df):
    # Add price indicators to a (1-second grid) book, in place, and return it:
    #   mid     plain mid, (bid + ask) / 2
    #   wmid    volume-weighted mid, weighted by the OPPOSITE side's size
    #           (the micro-price): (bid*ask_vol + ask*bid_vol)/(bid_vol+ask_vol)
    #           -- more size on the bid pushes the fair price up toward the ask.
    #           (For same-side weighting use bid*bid_vol + ask*ask_vol instead.)
    #   mean_10 trailing mean of mid over the last up-to-10 rows (10 s on a 1s grid)
    #   mean_30 trailing mean of mid over the last up-to-30 rows (30 s)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["wmid"] = ((df["bid"] * df["ask_vol"] + df["ask"] * df["bid_vol"])
                  / (df["bid_vol"] + df["ask_vol"]))
    df["mean_10"] = df["mid"].rolling(10, min_periods=1).mean()
    df["mean_30"] = df["mid"].rolling(30, min_periods=1).mean()
    return df


def f_indicators_by_nb(df):
    # Compute the mean_w_jit indicator through the pandas_numba bridge instead
    # of in pandas. Build a Pandas_nb over the (1s-grid) book -- numeric columns
    # are shared views, flag is forced to a char column so the njit loop can
    # count "H" -- carry df by int64 handle, then fill + attach mean_w_jit in an
    # @njit routine. Returns (df, nb); df gains the new float32 mean_w_jit column.
    nb = pn.f_df_to_nb(df, {"flag": "S"})   # mid/vols shared; flag -> CHAR1
    pn.f_register_df(nb, df)                             # attach df by handle
    f_indicators_jit(nb)                                   # njit: fill + push mean_w_jit
    return df, nb


@njit
def f_indicator_weighter_jit(mid, bv, av, flag):
    # The mean_w_jit kernel, kept separate so f_indicators_jit stays pure
    # orchestration. Pure numeric: takes the mid (float64), bid_vol / ask_vol
    # (int64) and flag (char) arrays, returns a fresh float32 array.
    #
    # out[i] is a volume-weighted mean of mid over a window whose length is
    # driven by recent activity: with h = number of "H" flags in the last 20
    # rows (i included), the window is the n = 2*h rows ending at i, clamped to
    # available history. Each row is weighted by its total size (bid_vol+ask_vol):
    #   out[i] = Σ mid[k]*(bid_vol[k]+ask_vol[k]) / Σ (bid_vol[k]+ask_vol[k])
    # If n == 0 (no "H" in the last 20 rows) it falls back to mid[i]. The
    # per-row-variable window is exactly what pandas rolling cannot express but
    # an njit loop does naturally.
    n_rows = mid.shape[0]
    out = np.empty(n_rows, dtype=np.float32)
    for i in range(n_rows):
        # h = count of "H" in the last 20 rows (i included)
        lo20 = i - 19
        if lo20 < 0:
            lo20 = 0
        h = 0
        for j in range(lo20, i + 1):
            if flag[j] == b"H":
                h += 1
        n = 2 * h
        if n == 0:
            out[i] = np.float32(mid[i])            # no H -> current mid
            continue
        lo = i - n + 1                             # window of n rows ending at i
        if lo < 0:
            lo = 0
        num = 0.0
        den = 0.0
        for j in range(lo, i + 1):
            w = np.float64(bv[j] + av[j])          # total size weight
            num += mid[j] * w
            den += w
        out[i] = np.float32(num / den)
    return out


@njit
def f_indicators_jit(nb):
    # Orchestration only: compute the mean_w_jit kernel, attach it to nb as a
    # float32 column, and push that column onto the carried df. No return: nb is
    # mutated and df is written in place.
    out = f_indicator_weighter_jit(nb.m_floats["mid"], nb.m_ints["bid_vol"],
                                   nb.m_ints["ask_vol"], nb.m_char1["flag"])
    nb.f_add_float32("mean_w_jit", out)
    # push the fresh (non-shared) float32 column onto the carried df via a pandas
    # call in objmode -- eval is expression-only, so set it with __setitem__.
    with objmode():
        f_eval_expr(
            "DF[params.m_df_id].__setitem__('mean_w_jit', params.m_float32['mean_w_jit'])",
            nb)


def f_set_shared_and_un_shared(df, panda_nb):
    # Add three columns to df, then bridge them into panda_nb with f_add_to_nb:
    # int8 and native |S3 cross as SHARED views (df[col].to_numpy() is a view,
    # stored as-is), while the object text column is kept only via str_types="U"
    # -- f_np_to_np_str builds a fresh |U array, so that column is UNSHARED.
    n = len(df)
    df["shared_int8"] = np.array([1] * n, dtype=np.int8)
    df["shared_char3"] = np.array(["AAA"] * n, dtype="S3")
    df["unshared_unicode"] = np.array(["AAA"] * n, dtype=object)
    pn.f_add_to_nb(panda_nb, "shared_int8", df["shared_int8"].to_numpy())
    pn.f_add_to_nb(panda_nb, "shared_char3", df["shared_char3"].to_numpy())
    pn.f_add_to_nb(panda_nb, "unshared_unicode",
                                df["unshared_unicode"].to_numpy(),
                                {"unshared_unicode": "U"})


def f_change_nb_values(nb):
    # Change the three columns' values in the Pandas_nb (plain python). int8 and
    # char3 are shared views of df, so df sees these edits; unshared_unicode is a
    # fresh column, so df does NOT -- the next print differs in two columns only.
    nb.m_int8["shared_int8"][:] = 2
    nb.m_char3["shared_char3"][:] = b"BBB"
    nb.m_unicode3["unshared_unicode"][:] = "BBB"


def f_test_book():
    df = f_make_book(10000, seed=0)

    book = f_indicators(f_markout_grid(df))
    book, nb = f_indicators_by_nb(book)
    f_set_shared_and_un_shared(book, nb)        # add 3 cols to book + nb
    print('initial book')
    print(book.head(10).to_string())            # 10 first lines (initial)
    f_change_nb_values(nb)                      # mutate the 3 cols IN nb (python)
    print('mutate the 3 cols IN pandas_nb (not df): df differs in the 2 shared cols only')
    print(book.head(10).to_string())            # differs in the 2 shared cols only


def f_main():
    f_test_book()


if __name__ == "__main__":
    f_main()
