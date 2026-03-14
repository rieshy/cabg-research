#!/usr/bin/env python3
"""
compute_outcomes.py

For each pullback candle in pullback_candles.csv, computes trade outcomes:

  Entry:  open of the bar after the pullback candle
  SL:     pullback_low  - SL_OFFSET  (buy)
          pullback_high + SL_OFFSET  (sell)
  R:      entry - sl  (buy)  /  sl - entry  (sell)

  Targets:
    t_struct  =  broken_level (MRH for buy, MRL for sell)
    t2..t5    =  entry ± N×R

  For each target: did price reach the target BEFORE hitting SL,
  within MAX_BARS bars of entry? Inconclusive → False.

  Bar color determines intrabar price sequence:
    green (close >= open):  low before high
    red   (close <  open):  high before low
  The open of each bar is checked first to catch gaps.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent / "data"
SIGNALS_FILE = Path(__file__).parent / "output" / "pullback_candles.csv"
OUTPUT_FILE  = Path(__file__).parent / "output" / "outcomes.csv"

SL_OFFSET = 0.0020   # 20 base points
MAX_BARS  = 200


# ── Data loading ──────────────────────────────────────────────────────────────
def load_price_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("GBPUSD_5_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return (df.drop_duplicates(subset="time")
              .sort_values("time")
              .reset_index(drop=True))


# ── Intrabar hit detection ────────────────────────────────────────────────────
def check_bar(is_buy: bool, o: float, h: float, l: float, c: float,
              sl: float, target: float) -> str:
    """
    Returns 'target', 'sl', or 'none' for a single bar.

    Open is checked first to handle gaps.
    Then bar color determines the intrabar order:
      green → low before high
      red   → high before low
    """
    if is_buy:
        if o <= sl:     return 'sl'
        if o >= target: return 'target'
        if c >= o:                        # green: low first
            if l <= sl:     return 'sl'
            if h >= target: return 'target'
        else:                             # red: high first
            if h >= target: return 'target'
            if l <= sl:     return 'sl'
    else:
        if o >= sl:     return 'sl'
        if o <= target: return 'target'
        if c >= o:                        # green: low first
            if l <= target: return 'target'
            if h >= sl:     return 'sl'
        else:                             # red: high first
            if h >= sl:     return 'sl'
            if l <= target: return 'target'
    return 'none'


def scan_target(is_buy: bool,
                hi: np.ndarray, lo: np.ndarray,
                op: np.ndarray, cl: np.ndarray,
                start: int, sl: float, target: float) -> bool:
    """
    Scan up to MAX_BARS bars from start.
    Returns True if target hit before SL, False otherwise (including inconclusive).
    """
    end = min(start + MAX_BARS, len(hi))
    for i in range(start, end):
        result = check_bar(is_buy, op[i], hi[i], lo[i], cl[i], sl, target)
        if result == 'target': return True
        if result == 'sl':     return False
    return False  # inconclusive


# ── Main computation ──────────────────────────────────────────────────────────
def compute_outcomes(signals: pd.DataFrame,
                     prices: pd.DataFrame) -> pd.DataFrame:
    ts_to_idx = {int(ts): idx for idx, ts in enumerate(prices["time"])}

    hi = prices["high"].to_numpy()
    lo = prices["low"].to_numpy()
    op = prices["open"].to_numpy()
    cl = prices["close"].to_numpy()

    records = []
    skipped = 0

    for _, row in signals.iterrows():
        pb_idx = ts_to_idx.get(int(row["pullback_time"]))
        if pb_idx is None or pb_idx + 1 >= len(prices):
            skipped += 1
            continue

        entry_idx = pb_idx + 1
        entry     = op[entry_idx]
        is_buy    = row["break_direction"] == "up"

        if is_buy:
            sl = row["pullback_low"]  - SL_OFFSET
            r  = entry - sl
        else:
            sl = row["pullback_high"] + SL_OFFSET
            r  = sl - entry

        if r <= 0:
            skipped += 1
            continue

        t_struct = float(row["broken_level"])

        # Structural target must be on the correct side of entry
        if is_buy  and t_struct <= entry: skipped += 1; continue
        if not is_buy and t_struct >= entry: skipped += 1; continue

        t2 = entry + 2 * r if is_buy else entry - 2 * r
        t3 = entry + 3 * r if is_buy else entry - 3 * r
        t4 = entry + 4 * r if is_buy else entry - 4 * r
        t5 = entry + 5 * r if is_buy else entry - 5 * r

        # t_struct expressed in R multiples
        t_struct_r = (t_struct - entry) / r if is_buy else (entry - t_struct) / r

        rec = dict(row)
        rec.update({
            "entry":       round(entry,      5),
            "sl":          round(sl,         5),
            "r":           round(r,          5),
            "r_atr":       round(r / row["atr"], 4),
            "t_struct":    round(t_struct,   5),
            "t_struct_r":  round(t_struct_r, 2),
            "t2":          round(t2,         5),
            "t3":          round(t3,         5),
            "t4":          round(t4,         5),
            "t5":          round(t5,         5),
        })

        for name, target in [("t_struct", t_struct),
                              ("t2", t2), ("t3", t3),
                              ("t4", t4), ("t5", t5)]:
            rec[f"hit_{name}"] = scan_target(
                is_buy, hi, lo, op, cl, entry_idx, sl, target)

        records.append(rec)

    print(f"  Skipped: {skipped:,}")
    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    prices  = load_price_data(DATA_DIR)
    signals = pd.read_csv(SIGNALS_FILE)
    print(f"  {len(signals):,} signals")

    print("Computing outcomes...")
    df = compute_outcomes(signals, prices)
    print(f"  {len(df):,} outcomes computed")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved → {OUTPUT_FILE}")

    print("\nHit rates (target reached before SL within 200 bars):")
    total = len(df)
    for col in ["hit_t_struct", "hit_t2", "hit_t3", "hit_t4", "hit_t5"]:
        n   = df[col].sum()
        pct = n / total * 100
        print(f"  {col:15s}  {pct:5.1f}%  ({int(n):,} / {total:,})")

    print(f"\nt_struct_r distribution (structural target in R multiples):")
    print(df["t_struct_r"].describe().round(2).to_string())
