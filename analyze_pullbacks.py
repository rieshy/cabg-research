#!/usr/bin/env python3
"""
analyze_pullbacks.py

Analyzes the morphology of pullback candles extracted by pullback_candles.py.

For an UP break  (market broke above MRH), the pullback candle's:
  - long_shadow_pct  = lower_shadow_pct  (rejection from below)
  - short_shadow_pct = upper_shadow_pct

For a DOWN break (market broke below MRL), the pullback candle's:
  - long_shadow_pct  = upper_shadow_pct  (rejection from above)
  - short_shadow_pct = lower_shadow_pct

The goal is to understand what shadow/body/size thresholds characterise
a "Pin Bar" — i.e. a candle where the market rejected one extreme and
committed to the opposite direction.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE  = Path(__file__).parent / "output" / "pullback_candles.csv"
OUTPUT_FILE = Path(__file__).parent / "output" / "analysis.txt"

PERCENTILES = [10, 25, 50, 75, 90, 95]


# ── Helpers ───────────────────────────────────────────────────────────────────
def pct_stats(series: pd.Series, label: str) -> str:
    s = series.dropna()
    pct = np.percentile(s, PERCENTILES)
    lines = [
        f"  {label}",
        f"    n={len(s):,}   mean={s.mean():.3f}   std={s.std():.3f}   "
        f"min={s.min():.3f}   max={s.max():.3f}",
        "    percentiles: " + "  ".join(
            f"p{p}={v:.3f}" for p, v in zip(PERCENTILES, pct)
        ),
    ]
    return "\n".join(lines)


def section(title: str) -> str:
    bar = "─" * 70
    return f"\n{bar}\n{title}\n{bar}"


# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(df: pd.DataFrame) -> str:
    out = []
    out.append("PULLBACK CANDLE MORPHOLOGY ANALYSIS")
    out.append(f"Total break events: {len(df):,}")

    # ── Normalise shadow labels per break direction ───────────────────────────
    # long_shadow  = the shadow in the direction of rejection (the "pin" shadow)
    # short_shadow = the opposite shadow
    df = df.copy()
    df["long_shadow_pct"]  = np.where(
        df["break_direction"] == "up",
        df["lower_shadow_pct"],   # up break: rejection from below
        df["upper_shadow_pct"],   # down break: rejection from above
    )
    df["short_shadow_pct"] = np.where(
        df["break_direction"] == "up",
        df["upper_shadow_pct"],
        df["lower_shadow_pct"],
    )
    # body_position_pct: for up breaks higher = better (body near top)
    # flip down breaks so that "high" = body near the rejection end in both cases
    df["body_position_signal"] = np.where(
        df["break_direction"] == "up",
        df["body_position_pct"],
        100 - df["body_position_pct"],
    )

    # ── Overall ───────────────────────────────────────────────────────────────
    out.append(section("OVERALL — ALL BREAKS"))
    out.append(pct_stats(df["candle_range_atr"],    "candle range (× ATR)"))
    out.append(pct_stats(df["long_shadow_pct"],     "long shadow (% of range)  ← the pin shadow"))
    out.append(pct_stats(df["short_shadow_pct"],    "short shadow (% of range) ← opposite shadow"))
    out.append(pct_stats(df["body_pct"],            "body (% of range)"))
    out.append(pct_stats(df["body_position_signal"],"body position (0=at SL end, 100=at signal end)"))

    # ── Split by break direction ───────────────────────────────────────────────
    for direction, label in [("up", "UP BREAKS  (bull pin candidate)"),
                              ("down", "DOWN BREAKS  (bear pin candidate)")]:
        sub = df[df["break_direction"] == direction]
        out.append(section(f"{label}   n={len(sub):,}"))
        out.append(pct_stats(sub["candle_range_atr"],     "candle range (× ATR)"))
        out.append(pct_stats(sub["long_shadow_pct"],      "long shadow (% of range)"))
        out.append(pct_stats(sub["short_shadow_pct"],     "short shadow (% of range)"))
        out.append(pct_stats(sub["body_pct"],             "body (% of range)"))
        out.append(pct_stats(sub["body_position_signal"], "body position (0=SL end, 100=signal end)"))

        out.append(f"\n  Candle direction split:")
        out.append(sub["candle_direction"].value_counts().to_string())

    # ── Split by OF state ─────────────────────────────────────────────────────
    out.append(section("LONG SHADOW by OF STATE"))
    for state in ["UP", "DOWN", "TEST_UP", "TEST_DOWN"]:
        sub = df[df["of_state"] == state]
        out.append(pct_stats(sub["long_shadow_pct"], f"{state}  (n={len(sub):,})"))

    # ── Window size distribution ───────────────────────────────────────────────
    out.append(section("PULLBACK WINDOW SIZE (bars between prev level and break)"))
    out.append(pct_stats(df["window_bars"], "window bars"))
    vc = df["window_bars"].clip(upper=20).value_counts().sort_index()
    out.append("  Distribution (capped at 20):")
    for bars, count in vc.items():
        bar_str = "█" * int(count / len(df) * 400)
        label_str = f"20+" if bars == 20 else str(int(bars))
        out.append(f"    {label_str:>3}  {bar_str}  {count:,}")

    return "\n".join(out)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    report = analyze(df)
    print(report)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_FILE.write_text(report)
    print(f"\nSaved → {OUTPUT_FILE}")
