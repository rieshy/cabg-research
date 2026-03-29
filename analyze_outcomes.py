#!/usr/bin/env python3
"""
analyze_outcomes.py

Reads pin_outcomes.csv and produces a full analysis report.
Run this instead of rerunning scan_pins.py when you just want to see results.

Sections:
  1. Baseline (no filtering)
  2. OF state alignment (with-flow vs counter-flow)
  3. OF state detail (all 4 states)
  4. Session
  5. Wick length — long_shadow_pct buckets
  6. Wick length — long_shadow_atr buckets
  7. Combinations: OF alignment × Session
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE  = Path(__file__).parent / "output" / "pin_outcomes.csv"
OUTPUT_FILE = Path(__file__).parent / "output" / "outcomes_report.txt"

MULTIPLES  = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
HIT_COLS   = [f"hit_t{str(m).replace('.','_')}" for m in MULTIPLES]
BREAKEVEN  = {c: 1/(m+1)*100 for c, m in zip(HIT_COLS, MULTIPLES)}
SESSIONS   = ["Asian", "London", "Overlap", "NY", "Off"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def col_header() -> str:
    return "  ".join(f"{str(m)+'R':>7}" for m in MULTIPLES)

def be_row() -> str:
    return "  ".join(f"{BREAKEVEN[c]:>6.1f}%" for c in HIT_COLS)

def rates_row(sub: pd.DataFrame) -> str:
    return "  ".join(f"{sub[c].mean()*100:>6.1f}%" for c in HIT_COLS)

def table(df: pd.DataFrame, group_col: str, groups: list,
          label_width: int = 12) -> str:
    lines = []
    pad = " " * (label_width + 10)
    lines.append(f"  {'':>{label_width}}  {'n':>6}  {col_header()}")
    lines.append("  " + "─" * (label_width + 8 + len(col_header()) + 18))
    lines.append(f"  {'breakeven':>{label_width}}  {'':>6}  {be_row()}")
    lines.append("")
    for g in groups:
        sub = df[df[group_col] == g]
        flag_row = rates_row(sub)
        lines.append(f"  {str(g):>{label_width}}  {len(sub):>6}  {flag_row}")
    lines.append(f"  {'TOTAL':>{label_width}}  {len(df):>6}  {rates_row(df)}")
    return "\n".join(lines)

def section(title: str) -> str:
    return f"\n{'═' * 72}\n{title}\n{'═' * 72}"


# ── Feature engineering ───────────────────────────────────────────────────────
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # OF alignment
    df["with_flow"] = (
        ((df["direction"] == "buy")  & df["of_state"].isin(["UP",   "TEST_UP"])) |
        ((df["direction"] == "sell") & df["of_state"].isin(["DOWN", "TEST_DOWN"]))
    )
    df["flow_label"] = df["with_flow"].map({True: "with flow", False: "counter flow"})

    # Session (UTC hours)
    df["hour"] = pd.to_datetime(df["time"], unit="s").dt.hour
    def session(h):
        if  0 <= h <  8: return "Asian"
        if  8 <= h < 13: return "London"
        if 13 <= h < 16: return "Overlap"
        if 16 <= h < 21: return "NY"
        return "Off"
    df["session"] = df["hour"].map(session)

    # Wick — pct buckets
    bins   = [0.50, 0.60, 0.70, 0.80, 1.01]
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-1.00"]
    df["wick_pct_bucket"] = pd.cut(df["long_shadow_pct"],
                                   bins=bins, labels=labels, right=False)

    # Wick — ATR buckets (quartiles)
    df["long_shadow_atr"] = df["candle_range_atr"] * df["long_shadow_pct"]
    q25, q50, q75 = df["long_shadow_atr"].quantile([.25, .50, .75])
    atr_bins   = [0, q25, q50, q75, 999]
    atr_labels = [f"<{q25:.2f}", f"{q25:.2f}-{q50:.2f}",
                  f"{q50:.2f}-{q75:.2f}", f">{q75:.2f}"]
    df["wick_atr_bucket"] = pd.cut(df["long_shadow_atr"],
                                   bins=atr_bins, labels=atr_labels)
    df["_atr_labels"] = atr_labels[0]   # store for reference
    df["_q25"] = q25; df["_q50"] = q50; df["_q75"] = q75

    return df, atr_labels


# ── Report ────────────────────────────────────────────────────────────────────
def build_report(df: pd.DataFrame, atr_labels: list) -> str:
    out = []
    out.append("PIN BAR OUTCOMES ANALYSIS")
    out.append(f"Total signals: {len(df):,}  "
               f"(buy: {(df['direction']=='buy').sum():,}  "
               f"sell: {(df['direction']=='sell').sum():,})")

    # 1. Baseline
    out.append(section("1. BASELINE — no filtering"))
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}  (n={len(sub):,})")
        out.append(f"  {'':>12}  {'':>6}  {col_header()}")
        out.append(f"  {'breakeven':>12}  {'':>6}  {be_row()}")
        out.append(f"  {'hit rate':>12}  {len(sub):>6}  {rates_row(sub)}")

    # 2. OF alignment
    out.append(section("2. OF STATE ALIGNMENT"))
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "flow_label",
                         ["with flow", "counter flow"], label_width=14))

    # 3. OF state detail
    out.append(section("3. OF STATE DETAIL"))
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "of_state",
                         ["UP", "DOWN", "TEST_UP", "TEST_DOWN"], label_width=12))

    # 4. Session
    out.append(section("4. SESSION"))
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "session", SESSIONS, label_width=10))

    # 5. Wick pct
    out.append(section("5. WICK LENGTH — long_shadow_pct buckets"))
    wick_pct_labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-1.00"]
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "wick_pct_bucket", wick_pct_labels, label_width=12))

    # 6. Wick ATR
    out.append(section("6. WICK LENGTH — long_shadow_atr quartile buckets"))
    q25 = df["_q25"].iloc[0]; q50 = df["_q50"].iloc[0]; q75 = df["_q75"].iloc[0]
    out.append(f"  ATR quartiles: p25={q25:.3f}  p50={q50:.3f}  p75={q75:.3f}")
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "wick_atr_bucket", atr_labels, label_width=14))

    # 7. Short shadow — pct buckets
    out.append(section("7. SHORT SHADOW — short_shadow_pct buckets"))
    short_pct_labels = ["0.00-0.10", "0.10-0.20", "0.20-0.30"]
    df["short_pct_bucket"] = pd.cut(df["short_shadow_pct"],
                                    bins=[0.0, 0.10, 0.20, 0.31],
                                    labels=short_pct_labels, right=False)
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "short_pct_bucket", short_pct_labels, label_width=12))

    # Short shadow — ATR buckets
    out.append(section("7b. SHORT SHADOW — short_shadow_atr quartile buckets"))
    df["short_shadow_atr"] = df["candle_range_atr"] * df["short_shadow_pct"]
    sq25, sq50, sq75 = df["short_shadow_atr"].quantile([.25, .50, .75])
    satr_labels = [f"<{sq25:.3f}", f"{sq25:.3f}-{sq50:.3f}",
                   f"{sq50:.3f}-{sq75:.3f}", f">{sq75:.3f}"]
    df["short_atr_bucket"] = pd.cut(df["short_shadow_atr"],
                                    bins=[0, sq25, sq50, sq75, 999],
                                    labels=satr_labels)
    out.append(f"  ATR quartiles: p25={sq25:.3f}  p50={sq50:.3f}  p75={sq75:.3f}")
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "short_atr_bucket", satr_labels, label_width=14))

    # 8. Candle body color
    out.append(section("7. PIN BODY COLOR"))
    out.append("  buy pin:  green = closed up (body toward signal end)")
    out.append("            red   = closed down (body toward SL end)\n")
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "candle_direction",
                         ["green", "red"], label_width=8))

    # 9. Prev bar pivot confirmation
    out.append(section("9. PREV BAR PIVOT CONFIRMATION"))
    out.append("  buy pin:  prev_low  > pin_low   (prev bar has higher low)")
    out.append("  sell pin: prev_high < pin_high  (prev bar has lower high)\n")
    df["pivot_label"] = df["prev_confirms"].map({True: "confirmed", False: "not confirmed"})
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "pivot_label",
                         ["confirmed", "not confirmed"], label_width=15))

    # 10. Candle range — relative (ATR) buckets
    out.append(section("10. CANDLE RANGE — candle_range_atr buckets"))
    range_atr_labels = ["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0"]
    df["range_atr_bucket"] = pd.cut(df["candle_range_atr"],
                                    bins=[0.5, 1.0, 1.5, 2.0, 3.01],
                                    labels=range_atr_labels, right=False)
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "range_atr_bucket", range_atr_labels, label_width=10))

    # 10b. Candle range — absolute (pips) quartile buckets
    out.append(section("10b. CANDLE RANGE — absolute (ATR units × ATR) quartile buckets"))
    df["atr_val"]       = df["r"] / df["r_atr"]                      # ATR in price units
    df["candle_range"]  = df["candle_range_atr"] * df["atr_val"]     # absolute range
    df["range_pips"]    = (df["candle_range"] * 10000).round(1)      # pips (4-decimal pair)
    rq25, rq50, rq75 = df["range_pips"].quantile([.25, .50, .75])
    range_abs_labels = [f"<{rq25:.1f}", f"{rq25:.1f}-{rq50:.1f}",
                        f"{rq50:.1f}-{rq75:.1f}", f">{rq75:.1f}"]
    df["range_abs_bucket"] = pd.cut(df["range_pips"],
                                    bins=[0, rq25, rq50, rq75, 9999],
                                    labels=range_abs_labels)
    out.append(f"  Pip quartiles: p25={rq25:.1f}  p50={rq50:.1f}  p75={rq75:.1f}")
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "range_abs_bucket", range_abs_labels, label_width=12))

    # 11. Distance to structural level (MRH/MRL) in R multiples
    out.append(section("11. DISTANCE TO STRUCTURAL LEVEL — R multiples"))
    out.append("  buy:  (mrh - entry) / r    sell: (entry - mrl) / r\n")
    df["dist_level_r"] = np.where(
        df["direction"] == "buy",
        (df["mrh"] - df["entry"]) / df["r"],
        (df["entry"] - df["mrl"]) / df["r"],
    )
    dq25, dq50, dq75 = df["dist_level_r"].quantile([.25, .50, .75])
    dist_labels = [f"<{dq25:.1f}R", f"{dq25:.1f}-{dq50:.1f}R",
                   f"{dq50:.1f}-{dq75:.1f}R", f">{dq75:.1f}R"]
    df["dist_level_bucket"] = pd.cut(df["dist_level_r"],
                                     bins=[0, dq25, dq50, dq75, 9999],
                                     labels=dist_labels)
    out.append(f"  R quartiles: p25={dq25:.2f}  p50={dq50:.2f}  p75={dq75:.2f}")
    out.append(f"  (distribution stats:  mean={df['dist_level_r'].mean():.2f}"
               f"  max={df['dist_level_r'].max():.2f})")
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "dist_level_bucket", dist_labels, label_width=12))

    # 11b. Distance to structural level — ATR multiples
    out.append(section("11b. DISTANCE TO STRUCTURAL LEVEL — ATR multiples"))
    df["atr_val2"]     = df["r"] / df["r_atr"]
    df["dist_level_atr"] = np.where(
        df["direction"] == "buy",
        (df["mrh"] - df["entry"]) / df["atr_val2"],
        (df["entry"] - df["mrl"]) / df["atr_val2"],
    )
    daq25, daq50, daq75 = df["dist_level_atr"].quantile([.25, .50, .75])
    dist_atr_labels = [f"<{daq25:.2f}", f"{daq25:.2f}-{daq50:.2f}",
                       f"{daq50:.2f}-{daq75:.2f}", f">{daq75:.2f}"]
    df["dist_atr_bucket"] = pd.cut(df["dist_level_atr"],
                                   bins=[0, daq25, daq50, daq75, 9999],
                                   labels=dist_atr_labels)
    out.append(f"  ATR quartiles: p25={daq25:.3f}  p50={daq50:.3f}  p75={daq75:.3f}")
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "dist_atr_bucket", dist_atr_labels, label_width=12))

    # 12. OF alignment × Session
    out.append(section("12. COMBINATION — OF alignment × Session"))
    df["combo"] = df["flow_label"].str[:4] + " / " + df["session"]
    combos = [f"{f[:4]} / {s}"
              for f in ["with", "counter"]
              for s in SESSIONS]
    for label, mask in [("ALL",  slice(None)),
                         ("BUY",  df["direction"] == "buy"),
                         ("SELL", df["direction"] == "sell")]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        out.append(f"\n  {label}")
        out.append(table(sub, "combo", combos, label_width=18))

    return "\n".join(out)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = pd.read_csv(INPUT_FILE)
    df, atr_labels = enrich(df_raw)
    report = build_report(df, atr_labels)
    print(report)
    OUTPUT_FILE.write_text(report)
    print(f"\nSaved → {OUTPUT_FILE}")
