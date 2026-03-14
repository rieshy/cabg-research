#!/usr/bin/env python3
"""
pullback_candles.py

Simulates the Cabg Trade Signals order flow logic on M5 GBPUSD data and
extracts the pullback candle preceding each structural break:
  - MRH break → candle with the lowest low in the pullback window
  - MRL break → candle with the highest high in the pullback window

Pullback window:
  MRH break: [prevMrhBar+1 .. breakBar]   if green (down-first candle)
             [prevMrhBar+1 .. breakBar-1]  if red   (up-first candle)
  MRL break: [prevMrlBar+1 .. breakBar]   if red   (up-first candle)
             [prevMrlBar+1 .. breakBar-1]  if green (down-first candle)

Input:  data/GBPUSD_5_*.csv  (Unix timestamp + OHLC, downloaded from TradingView)
Output: output/pullback_candles.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
OF_DOWN      = -2
OF_TEST_UP   = -1
OF_TEST_DOWN =  1
OF_UP        =  2

ATR_PERIOD      = 14
RESET_STALE_M   = 20
RESET_STALE_N   = 3.0

OF_NAMES = {
    OF_UP: "UP", OF_DOWN: "DOWN",
    OF_TEST_UP: "TEST_UP", OF_TEST_DOWN: "TEST_DOWN",
}

DATA_DIR    = Path(__file__).parent / "data"
OUTPUT_FILE = Path(__file__).parent / "output" / "pullback_candles.csv"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("GBPUSD_5_*.csv"))
    if not files:
        sys.exit(f"No GBPUSD_5_*.csv files found in {data_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = (df.drop_duplicates(subset="time")
            .sort_values("time")
            .reset_index(drop=True))
    return df


# ── ATR (Wilder smoothing — matches Pine Script ta.atr) ──────────────────────
def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().to_numpy()


# ── Pivot-finding helpers (port of Pine Script findNewMrl/MrhOffset) ──────────
# In Pine Script, offset=N means "N bars ago from current bar".
# Translated: bar at offset N from bar_idx  =>  array index  bar_idx - N
# "older" direction (larger offset) => lower array index

def find_new_mrl_offset(hi: np.ndarray, lo: np.ndarray,
                         start_offset: int, bar_idx: int) -> int:
    """
    Scan backward from start_offset to find where the new MRL should sit.
    Stops at the most recent pivot low or inside-bar edge; falls back to
    start_offset if nothing is found.
    """
    result = start_offset
    for offset in range(start_offset, bar_idx):
        i     = bar_idx - offset      # current candidate index
        i_old = i - 1                 # one bar older (offset + 1)
        if i_old < 0:
            break

        inside = hi[i] < hi[i_old] and lo[i] > lo[i_old]
        if inside:
            i_old2 = i - 2
            prev_pivot = (i_old2 >= 0
                          and lo[i_old] <= lo[i]
                          and lo[i_old] <= lo[i_old2])
            result = offset + 1 if prev_pivot else offset
            break

        if offset == 0:
            is_pivot = i > 0 and lo[i] <= lo[i - 1]
        else:
            i_new = i + 1            # one bar newer (offset - 1)
            is_pivot = (i_new <= bar_idx
                        and i_old >= 0
                        and lo[i] <= lo[i_new]
                        and lo[i] <= lo[i_old])
        if is_pivot:
            result = offset
            break
    return result


def find_new_mrh_offset(hi: np.ndarray, lo: np.ndarray,
                         start_offset: int, bar_idx: int) -> int:
    """
    Scan backward from start_offset to find where the new MRH should sit.
    """
    result = start_offset
    for offset in range(start_offset, bar_idx):
        i     = bar_idx - offset
        i_old = i - 1
        if i_old < 0:
            break

        inside = hi[i] < hi[i_old] and lo[i] > lo[i_old]
        if inside:
            i_old2 = i - 2
            prev_pivot = (i_old2 >= 0
                          and hi[i_old] >= hi[i]
                          and hi[i_old] >= hi[i_old2])
            result = offset + 1 if prev_pivot else offset
            break

        if offset == 0:
            is_pivot = i > 0 and hi[i] >= hi[i - 1]
        else:
            i_new = i + 1
            is_pivot = (i_new <= bar_idx
                        and i_old >= 0
                        and hi[i] >= hi[i_new]
                        and hi[i] >= hi[i_old])
        if is_pivot:
            result = offset
            break
    return result


# ── Candle morphology ─────────────────────────────────────────────────────────
def candle_features(o: float, h: float, l: float, c: float,
                    atr: float) -> dict:
    body       = abs(c - o)
    crange     = h - l
    upper_shad = h - max(o, c)
    lower_shad = min(o, c) - l
    mid_body   = (max(o, c) + min(o, c)) / 2.0
    return {
        "candle_direction":  "green" if c >= o else "red",
        "candle_range":      round(crange,                              5),
        "candle_range_atr":  round(crange     / atr,                   4) if atr else None,
        "body":              round(body,                                5),
        "body_pct":          round(body       / crange,                4) if crange else None,
        "body_position_pct": round((mid_body  - l) / crange * 100,     1) if crange else None,
        "upper_shadow":      round(upper_shad,                         5),
        "upper_shadow_pct":  round(upper_shad / crange,                4) if crange else None,
        "lower_shadow":      round(lower_shad,                         5),
        "lower_shadow_pct":  round(lower_shad / crange,                4) if crange else None,
        "atr":               round(atr,                                5),
    }


# ── Main simulation ───────────────────────────────────────────────────────────
def simulate(df: pd.DataFrame) -> pd.DataFrame:
    hi  = df["high"].to_numpy()
    lo  = df["low"].to_numpy()
    op  = df["open"].to_numpy()
    cl  = df["close"].to_numpy()
    ts  = df["time"].to_numpy()
    atr = compute_atr(df)

    n = len(df)

    # Initial state (mirrors Pine Script var declarations)
    order_flow    = OF_UP if cl[0] >= op[0] else OF_DOWN
    mrh_bar       = 0
    mrh_price     = hi[0]
    mrl_bar       = 0
    mrl_price     = lo[0]
    pullback_seen = False

    records = []

    def record_break(break_dir: str, break_bar: int,
                     prev_level_bar: int, of_state: int,
                     broken_level: float) -> None:
        """Find the pullback candle in the window and append a record."""
        is_green = cl[break_bar] >= op[break_bar]

        if break_dir == "up":
            # MRH break: lowest low in [prevMrhBar+1 .. breakBar or breakBar-1]
            win_end   = break_bar if is_green else break_bar - 1
            win_start = prev_level_bar + 1
            if win_start > win_end:
                return
            pb_idx = win_start + int(np.argmin(lo[win_start:win_end + 1]))
        else:
            # MRL break: highest high in [prevMrlBar+1 .. breakBar or breakBar-1]
            win_end   = break_bar if not is_green else break_bar - 1
            win_start = prev_level_bar + 1
            if win_start > win_end:
                return
            pb_idx = win_start + int(np.argmax(hi[win_start:win_end + 1]))

        pb_atr = atr[pb_idx]
        if np.isnan(pb_atr):
            return  # ATR not yet warm; skip

        records.append({
            "break_time":      ts[break_bar],
            "break_direction": break_dir,
            "of_state":        OF_NAMES[of_state],
            "broken_level":    round(broken_level, 5),
            "prev_level_time": ts[prev_level_bar],
            "pullback_time":   ts[pb_idx],
            "pullback_open":   op[pb_idx],
            "pullback_high":   hi[pb_idx],
            "pullback_low":    lo[pb_idx],
            "pullback_close":  cl[pb_idx],
            "window_bars":     win_end - win_start + 1,
            **candle_features(op[pb_idx], hi[pb_idx],
                              lo[pb_idx], cl[pb_idx], float(pb_atr)),
        })

    # ── Per-bar loop ──────────────────────────────────────────────────────────
    for idx in range(1, n):
        is_green = cl[idx] >= op[idx]

        # ── Stale level reset ─────────────────────────────────────────────────
        stale_reset = (
            idx - mrh_bar > RESET_STALE_M and
            idx - mrl_bar > RESET_STALE_M and
            not np.isnan(atr[idx]) and
            mrh_price - mrl_price > RESET_STALE_N * atr[idx]
        )
        if stale_reset:
            if is_green:
                mrl_off   = find_new_mrl_offset(hi, lo, 0, idx)
                mrh_bar   = idx;             mrh_price = hi[idx]
                mrl_bar   = idx - mrl_off;   mrl_price = lo[idx - mrl_off]
            else:
                mrh_off   = find_new_mrh_offset(hi, lo, 0, idx)
                mrl_bar   = idx;             mrl_price = lo[idx]
                mrh_bar   = idx - mrh_off;   mrh_price = hi[idx - mrh_off]
            order_flow    = OF_UP if is_green else OF_DOWN
            pullback_seen = False
            continue

        # Pullback detection — must run before break detection
        if order_flow == OF_TEST_UP   and hi[idx] < mrh_price:
            pullback_seen = True
        if order_flow == OF_TEST_DOWN and lo[idx] > mrl_price:
            pullback_seen = True

        # ── MRH processing ────────────────────────────────────────────────────
        def process_mrh():
            nonlocal order_flow, mrh_bar, mrh_price, mrl_bar, mrl_price, pullback_seen

            if hi[idx] < mrh_price:
                return

            # start_offset for findNewMrlOffset:
            # green candle (low first) → current bar's low is valid → 0
            # red candle   (high first) → skip current bar for MRL  → 1
            mrl_start = 0 if is_green else 1

            if order_flow == OF_UP:
                if mrh_bar == idx - 1:
                    # Consecutive expansion — just extend, no pullback
                    mrh_bar   = idx
                    mrh_price = hi[idx]
                elif hi[idx] > mrh_price:
                    prev = mrh_bar
                    record_break("up", idx, prev, order_flow, mrh_price)
                    mrl_off   = find_new_mrl_offset(hi, lo, mrl_start, idx)
                    mrh_bar   = idx;        mrh_price = hi[idx]
                    mrl_bar   = idx - mrl_off; mrl_price = lo[idx - mrl_off]

            elif order_flow == OF_TEST_DOWN and hi[idx] > mrh_price:
                prev = mrh_bar
                record_break("up", idx, prev, order_flow, mrh_price)
                mrl_off   = find_new_mrl_offset(hi, lo, mrl_start, idx)
                order_flow = OF_UP
                mrh_bar   = idx;        mrh_price = hi[idx]
                mrl_bar   = idx - mrl_off; mrl_price = lo[idx - mrl_off]

            elif order_flow == OF_DOWN and hi[idx] > mrh_price:
                prev = mrh_bar
                record_break("up", idx, prev, order_flow, mrh_price)
                mrl_off   = find_new_mrl_offset(hi, lo, mrl_start, idx)
                order_flow    = OF_TEST_UP
                mrh_bar   = idx;        mrh_price = hi[idx]
                mrl_bar   = idx - mrl_off; mrl_price = lo[idx - mrl_off]
                pullback_seen = False

            elif order_flow == OF_TEST_UP and hi[idx] > mrh_price:
                if pullback_seen:
                    prev = mrh_bar
                    record_break("up", idx, prev, order_flow, mrh_price)
                    mrl_off   = find_new_mrl_offset(hi, lo, mrl_start, idx)
                    order_flow = OF_UP
                    mrh_bar   = idx;        mrh_price = hi[idx]
                    mrl_bar   = idx - mrl_off; mrl_price = lo[idx - mrl_off]
                else:
                    mrh_bar   = idx
                    mrh_price = hi[idx]

        # ── MRL processing ────────────────────────────────────────────────────
        def process_mrl():
            nonlocal order_flow, mrh_bar, mrh_price, mrl_bar, mrl_price, pullback_seen

            if lo[idx] > mrl_price:
                return

            # start_offset for findNewMrhOffset:
            # red candle   (high first) → current bar's high is valid → 0
            # green candle (low first)  → skip current bar for MRH   → 1
            mrh_start = 1 if is_green else 0

            if order_flow == OF_DOWN:
                if mrl_bar == idx - 1:
                    mrl_bar   = idx
                    mrl_price = lo[idx]
                elif lo[idx] < mrl_price:
                    prev = mrl_bar
                    record_break("down", idx, prev, order_flow, mrl_price)
                    mrh_off   = find_new_mrh_offset(hi, lo, mrh_start, idx)
                    mrl_bar   = idx;        mrl_price = lo[idx]
                    mrh_bar   = idx - mrh_off; mrh_price = hi[idx - mrh_off]

            elif order_flow == OF_TEST_UP and lo[idx] < mrl_price:
                prev = mrl_bar
                record_break("down", idx, prev, order_flow, mrl_price)
                mrh_off   = find_new_mrh_offset(hi, lo, mrh_start, idx)
                order_flow = OF_DOWN
                mrl_bar   = idx;        mrl_price = lo[idx]
                mrh_bar   = idx - mrh_off; mrh_price = hi[idx - mrh_off]

            elif order_flow == OF_UP and lo[idx] < mrl_price:
                prev = mrl_bar
                record_break("down", idx, prev, order_flow, mrl_price)
                mrh_off   = find_new_mrh_offset(hi, lo, mrh_start, idx)
                order_flow    = OF_TEST_DOWN
                mrl_bar   = idx;        mrl_price = lo[idx]
                mrh_bar   = idx - mrh_off; mrh_price = hi[idx - mrh_off]
                pullback_seen = False

            elif order_flow == OF_TEST_DOWN and lo[idx] < mrl_price:
                if pullback_seen:
                    prev = mrl_bar
                    record_break("down", idx, prev, order_flow, mrl_price)
                    mrh_off   = find_new_mrh_offset(hi, lo, mrh_start, idx)
                    order_flow = OF_DOWN
                    mrl_bar   = idx;        mrl_price = lo[idx]
                    mrh_bar   = idx - mrh_off; mrh_price = hi[idx - mrh_off]
                else:
                    mrl_bar   = idx
                    mrl_price = lo[idx]

        # Green: low first (MRL), then high (MRH)
        # Red:   high first (MRH), then low (MRL)
        if is_green:
            process_mrl()
            process_mrh()
        else:
            process_mrh()
            process_mrl()

    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_DIR)
    t0 = pd.to_datetime(df["time"].iloc[0],  unit="s")
    t1 = pd.to_datetime(df["time"].iloc[-1], unit="s")
    print(f"  {len(df):,} bars  |  {t0.date()} → {t1.date()}")

    print("Running simulation...")
    results = simulate(df)
    print(f"  {len(results):,} break events recorded")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved → {OUTPUT_FILE}")

    print("\nBreaks by direction:")
    print(results["break_direction"].value_counts().to_string())
    print("\nBreaks by OF state:")
    print(results["of_state"].value_counts().to_string())
    print("\nCandle range (× ATR):")
    print(results["candle_range_atr"].describe().round(3).to_string())
    print("\nLower shadow (% of range):")
    print(results["lower_shadow_pct"].describe().round(3).to_string())
    print("\nUpper shadow (% of range):")
    print(results["upper_shadow_pct"].describe().round(3).to_string())
