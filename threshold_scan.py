#!/usr/bin/env python3
"""
threshold_scan.py

Shows how many pullback candles survive various combinations of
defining-feature thresholds for a Pin Bar.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(__file__).parent / "output" / "pullback_candles.csv"

LONG_SHADOW_THRESHOLDS  = [0.40, 0.45, 0.50, 0.55, 0.60]
SHORT_SHADOW_THRESHOLDS = [0.20, 0.25, 0.30, 0.35]
ATR_MIN, ATR_MAX        = 0.5, 3.0


def run(df: pd.DataFrame) -> None:
    df = df.copy()
    df["long_shadow_pct"] = np.where(
        df["break_direction"] == "up",
        df["lower_shadow_pct"],
        df["upper_shadow_pct"],
    )
    df["short_shadow_pct"] = np.where(
        df["break_direction"] == "up",
        df["upper_shadow_pct"],
        df["lower_shadow_pct"],
    )

    total = len(df)
    print(f"Total candles: {total:,}\n")

    # ── Step 1: ATR size filter alone ────────────────────────────────────────
    atr_mask = df["candle_range_atr"].between(ATR_MIN, ATR_MAX)
    after_atr = atr_mask.sum()
    print(f"After ATR filter ({ATR_MIN}–{ATR_MAX}×): "
          f"{after_atr:,}  ({after_atr/total*100:.1f}%)\n")

    # ── Step 2: shadow combinations on ATR-filtered set ──────────────────────
    base = df[atr_mask].copy()
    base_n = len(base)

    header = f"{'long≥':>6}  {'short≤':>6}  {'n':>6}  {'%base':>6}  {'%total':>7}"
    print(header)
    print("─" * len(header))

    for long_min in LONG_SHADOW_THRESHOLDS:
        for short_max in SHORT_SHADOW_THRESHOLDS:
            mask = (
                (base["long_shadow_pct"]  >= long_min) &
                (base["short_shadow_pct"] <= short_max)
            )
            n = mask.sum()
            print(f"{long_min:>6.0%}  {short_max:>6.0%}  "
                  f"{n:>6,}  {n/base_n*100:>5.1f}%  {n/total*100:>6.1f}%")
        print()

    # ── Step 3: best candidate — show up/down split ───────────────────────────
    print("\n── Direction split for each threshold combo (after ATR filter) ──\n")
    print(f"{'long≥':>6}  {'short≤':>6}  {'up':>6}  {'down':>6}  {'ratio up':>9}")
    print("─" * 46)
    for long_min in LONG_SHADOW_THRESHOLDS:
        for short_max in SHORT_SHADOW_THRESHOLDS:
            mask = (
                (base["long_shadow_pct"]  >= long_min) &
                (base["short_shadow_pct"] <= short_max)
            )
            sub  = base[mask]
            n_up   = (sub["break_direction"] == "up").sum()
            n_down = (sub["break_direction"] == "down").sum()
            ratio  = n_up / (n_up + n_down) * 100 if (n_up + n_down) else 0
            print(f"{long_min:>6.0%}  {short_max:>6.0%}  "
                  f"{n_up:>6,}  {n_down:>6,}  {ratio:>8.1f}%")
        print()


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    run(df)
