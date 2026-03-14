#!/usr/bin/env python3
"""
scan_pins.py

Scans every bar for Pin Bar patterns and computes trade outcomes.
No causal filtering — every qualifying pin is traded (unrealistic baseline).

Pin defining criteria:
  buy  pin: lower_shadow >= LONG_SHADOW_MIN  AND  upper_shadow <= SHORT_SHADOW_MAX
  sell pin: upper_shadow >= LONG_SHADOW_MIN  AND  lower_shadow <= SHORT_SHADOW_MAX
  both:     candle range within [ATR_MIN, ATR_MAX] × ATR

Trade mechanics:
  Entry:  open of the next bar
  SL:     pin_low  - SL_OFFSET  (buy)
          pin_high + SL_OFFSET  (sell)
  R:      entry - sl  (buy)  /  sl - entry  (sell)
  T_struct: MRH (buy) / MRL (sell) — OF state captured at pin close
  T2–T5:  entry ± N×R

Outcome: did price reach target before SL within MAX_BARS bars?
Bar color determines intrabar sequence (green = low first, red = high first).
Open checked first on every bar to handle gaps.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Defining feature thresholds ───────────────────────────────────────────────
LONG_SHADOW_MIN  = 0.50
SHORT_SHADOW_MAX = 0.30
ATR_MIN          = 0.5
ATR_MAX          = 3.0

# ── Trade mechanics ───────────────────────────────────────────────────────────
SL_OFFSET = 0.0020
MAX_BARS  = 200

# ── OF simulation parameters ──────────────────────────────────────────────────
ATR_PERIOD    = 14
RESET_STALE_M = 20
RESET_STALE_N = 10.0

OF_DOWN      = -2
OF_TEST_UP   = -1
OF_TEST_DOWN =  1
OF_UP        =  2
OF_NAMES     = {OF_UP:"UP", OF_DOWN:"DOWN",
                OF_TEST_UP:"TEST_UP", OF_TEST_DOWN:"TEST_DOWN"}

DATA_DIR    = Path(__file__).parent / "data"
OUTPUT_FILE = Path(__file__).parent / "output" / "pin_outcomes.csv"


# ── Helpers (copied from pullback_candles.py) ─────────────────────────────────
def load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("GBPUSD_5_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return (df.drop_duplicates(subset="time")
              .sort_values("time")
              .reset_index(drop=True))


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> np.ndarray:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().to_numpy()


def find_new_mrl_offset(hi, lo, start_offset, bar_idx):
    result = start_offset
    for offset in range(start_offset, bar_idx):
        i, i_old = bar_idx - offset, bar_idx - offset - 1
        if i_old < 0: break
        inside = hi[i] < hi[i_old] and lo[i] > lo[i_old]
        if inside:
            i_old2 = i_old - 1
            pivot = i_old2 >= 0 and lo[i_old] <= lo[i] and lo[i_old] <= lo[i_old2]
            result = offset + 1 if pivot else offset
            break
        if offset == 0:
            is_pivot = i > 0 and lo[i] <= lo[i - 1]
        else:
            i_new = i + 1
            is_pivot = i_new <= bar_idx and i_old >= 0 and lo[i] <= lo[i_new] and lo[i] <= lo[i_old]
        if is_pivot: result = offset; break
    return result


def find_new_mrh_offset(hi, lo, start_offset, bar_idx):
    result = start_offset
    for offset in range(start_offset, bar_idx):
        i, i_old = bar_idx - offset, bar_idx - offset - 1
        if i_old < 0: break
        inside = hi[i] < hi[i_old] and lo[i] > lo[i_old]
        if inside:
            i_old2 = i_old - 1
            pivot = i_old2 >= 0 and hi[i_old] >= hi[i] and hi[i_old] >= hi[i_old2]
            result = offset + 1 if pivot else offset
            break
        if offset == 0:
            is_pivot = i > 0 and hi[i] >= hi[i - 1]
        else:
            i_new = i + 1
            is_pivot = i_new <= bar_idx and i_old >= 0 and hi[i] >= hi[i_new] and hi[i] >= hi[i_old]
        if is_pivot: result = offset; break
    return result


def check_bar(is_buy, o, h, l, c, sl, target):
    """Returns 'target', 'sl', or 'none'. Open checked first for gaps."""
    if is_buy:
        if o <= sl:     return 'sl'
        if o >= target: return 'target'
        if c >= o:
            if l <= sl:     return 'sl'
            if h >= target: return 'target'
        else:
            if h >= target: return 'target'
            if l <= sl:     return 'sl'
    else:
        if o >= sl:     return 'sl'
        if o <= target: return 'target'
        if c >= o:
            if l <= target: return 'target'
            if h >= sl:     return 'sl'
        else:
            if h >= sl:     return 'sl'
            if l <= target: return 'target'
    return 'none'


def scan_target(is_buy, hi, lo, op, cl, start, sl, target):
    end = min(start + MAX_BARS, len(hi))
    for i in range(start, end):
        r = check_bar(is_buy, op[i], hi[i], lo[i], cl[i], sl, target)
        if r == 'target': return True
        if r == 'sl':     return False
    return False  # inconclusive


# ── Main ──────────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame) -> pd.DataFrame:
    hi  = df["high"].to_numpy()
    lo  = df["low"].to_numpy()
    op  = df["open"].to_numpy()
    cl  = df["close"].to_numpy()
    ts  = df["time"].to_numpy()
    atr = compute_atr(df)
    n   = len(df)

    # OF simulation state
    order_flow    = OF_UP if cl[0] >= op[0] else OF_DOWN
    mrh_bar       = 0;  mrh_price = hi[0]
    mrl_bar       = 0;  mrl_price = lo[0]
    pullback_seen = False

    records = []

    for idx in range(1, n - 1):   # -1: need at least one bar after for entry
        is_green = cl[idx] >= op[idx]
        a = atr[idx]
        if np.isnan(a): continue

        # ── Pullback detection ────────────────────────────────────────────────
        if order_flow == OF_TEST_UP   and hi[idx] < mrh_price: pullback_seen = True
        if order_flow == OF_TEST_DOWN and lo[idx] > mrl_price: pullback_seen = True

        # ── Stale level reset ─────────────────────────────────────────────────
        stale = (idx - mrh_bar > RESET_STALE_M and
                 idx - mrl_bar > RESET_STALE_M and
                 mrh_price - mrl_price > RESET_STALE_N * a)
        if stale:
            if is_green:
                off = find_new_mrl_offset(hi, lo, 0, idx)
                mrh_bar = idx;       mrh_price = hi[idx]
                mrl_bar = idx - off; mrl_price = lo[idx - off]
            else:
                off = find_new_mrh_offset(hi, lo, 0, idx)
                mrl_bar = idx;       mrl_price = lo[idx]
                mrh_bar = idx - off; mrh_price = hi[idx - off]
            order_flow = OF_UP if is_green else OF_DOWN
            pullback_seen = False
            continue

        # ── Pin bar detection — runs before OF update, same as Pine Script ────
        c_range = hi[idx] - lo[idx]
        if c_range > 0 and ATR_MIN <= c_range / a <= ATR_MAX:
            upper_shad = hi[idx] - max(op[idx], cl[idx])
            lower_shad = min(op[idx], cl[idx]) - lo[idx]
            u_pct = upper_shad / c_range
            l_pct = lower_shad / c_range

            for is_buy, long_pct, short_pct in [
                (True,  l_pct, u_pct),   # buy pin:  long lower shadow
                (False, u_pct, l_pct),   # sell pin: long upper shadow
            ]:
                if long_pct < LONG_SHADOW_MIN or short_pct > SHORT_SHADOW_MAX:
                    continue

                entry_idx = idx + 1
                entry     = op[entry_idx]

                if is_buy:
                    sl = lo[idx] - SL_OFFSET
                    r  = entry - sl
                    t_struct = mrh_price
                else:
                    sl = hi[idx] + SL_OFFSET
                    r  = sl - entry
                    t_struct = mrl_price

                if r <= 0: continue
                if is_buy  and t_struct <= entry: continue
                if not is_buy and t_struct >= entry: continue

                multiples = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
                targets   = [(f"t{str(m).replace('.','_')}",
                               entry + m*r if is_buy else entry - m*r)
                              for m in multiples]

                body     = abs(cl[idx] - op[idx])
                mid_body = (max(op[idx], cl[idx]) + min(op[idx], cl[idx])) / 2

                rec = {
                    "time":             ts[idx],
                    "direction":        "buy" if is_buy else "sell",
                    "of_state":         OF_NAMES[order_flow],
                    "candle_direction": "green" if is_green else "red",
                    # Candle morphology
                    "candle_range_atr": round(c_range / a,                       4),
                    "long_shadow_pct":  round(long_pct,                          4),
                    "short_shadow_pct": round(short_pct,                         4),
                    "body_pct":         round(body / c_range,                    4),
                    "body_position_pct":round((mid_body - lo[idx]) / c_range * 100, 1),
                    # Trade levels
                    "entry":            round(entry,    5),
                    "sl":               round(sl,       5),
                    "r":                round(r,        5),
                    "r_atr":            round(r / a,    4),
                    "mrh":              round(mrh_price,5),
                    "mrl":              round(mrl_price,5),
                }

                for name, tgt in targets:
                    rec[f"hit_{name}"] = scan_target(
                        is_buy, hi, lo, op, cl, entry_idx, sl, tgt)

                records.append(rec)

        # ── OF market structure update ────────────────────────────────────────
        def process_mrh():
            nonlocal order_flow, mrh_bar, mrh_price, mrl_bar, mrl_price, pullback_seen
            if hi[idx] < mrh_price: return
            s = 0 if is_green else 1
            if order_flow == OF_UP:
                if mrh_bar == idx - 1:
                    mrh_bar = idx; mrh_price = hi[idx]
                elif hi[idx] > mrh_price:
                    off = find_new_mrl_offset(hi, lo, s, idx)
                    mrh_bar = idx; mrh_price = hi[idx]
                    mrl_bar = idx - off; mrl_price = lo[idx - off]
            elif order_flow == OF_TEST_DOWN and hi[idx] > mrh_price:
                off = find_new_mrl_offset(hi, lo, s, idx)
                order_flow = OF_UP
                mrh_bar = idx; mrh_price = hi[idx]
                mrl_bar = idx - off; mrl_price = lo[idx - off]
            elif order_flow == OF_DOWN and hi[idx] > mrh_price:
                off = find_new_mrl_offset(hi, lo, s, idx)
                order_flow = OF_TEST_UP
                mrh_bar = idx; mrh_price = hi[idx]
                mrl_bar = idx - off; mrl_price = lo[idx - off]
                pullback_seen = False
            elif order_flow == OF_TEST_UP and hi[idx] > mrh_price:
                if pullback_seen:
                    off = find_new_mrl_offset(hi, lo, s, idx)
                    order_flow = OF_UP
                    mrh_bar = idx; mrh_price = hi[idx]
                    mrl_bar = idx - off; mrl_price = lo[idx - off]
                else:
                    mrh_bar = idx; mrh_price = hi[idx]

        def process_mrl():
            nonlocal order_flow, mrh_bar, mrh_price, mrl_bar, mrl_price, pullback_seen
            if lo[idx] > mrl_price: return
            s = 1 if is_green else 0
            if order_flow == OF_DOWN:
                if mrl_bar == idx - 1:
                    mrl_bar = idx; mrl_price = lo[idx]
                elif lo[idx] < mrl_price:
                    off = find_new_mrh_offset(hi, lo, s, idx)
                    mrl_bar = idx; mrl_price = lo[idx]
                    mrh_bar = idx - off; mrh_price = hi[idx - off]
            elif order_flow == OF_TEST_UP and lo[idx] < mrl_price:
                off = find_new_mrh_offset(hi, lo, s, idx)
                order_flow = OF_DOWN
                mrl_bar = idx; mrl_price = lo[idx]
                mrh_bar = idx - off; mrh_price = hi[idx - off]
            elif order_flow == OF_UP and lo[idx] < mrl_price:
                off = find_new_mrh_offset(hi, lo, s, idx)
                order_flow = OF_TEST_DOWN
                mrl_bar = idx; mrl_price = lo[idx]
                mrh_bar = idx - off; mrh_price = hi[idx - off]
                pullback_seen = False
            elif order_flow == OF_TEST_DOWN and lo[idx] < mrl_price:
                if pullback_seen:
                    off = find_new_mrh_offset(hi, lo, s, idx)
                    order_flow = OF_DOWN
                    mrl_bar = idx; mrl_price = lo[idx]
                    mrh_bar = idx - off; mrh_price = hi[idx - off]
                else:
                    mrl_bar = idx; mrl_price = lo[idx]

        if is_green: process_mrl(); process_mrh()
        else:        process_mrh(); process_mrl()

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_DIR)
    t0 = pd.to_datetime(df["time"].iloc[0],  unit="s")
    t1 = pd.to_datetime(df["time"].iloc[-1], unit="s")
    print(f"  {len(df):,} bars  |  {t0.date()} → {t1.date()}")

    print("Scanning pins and computing outcomes...")
    results = run(df)
    print(f"  {len(results):,} pin signals found")

    buys  = results[results["direction"] == "buy"]
    sells = results[results["direction"] == "sell"]
    print(f"  buy: {len(buys):,}   sell: {len(sells):,}")

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved → {OUTPUT_FILE}")

    hit_cols = [c for c in results.columns if c.startswith("hit_")]
    breakeven = {f"hit_t{str(m).replace('.','_')}": 1/(m+1)*100
                 for m in [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]}

    print(f"\n{'Target':>10}  {'Breakeven':>10}  {'All':>8}  {'Buy':>8}  {'Sell':>8}")
    print("─" * 52)
    for col in hit_cols:
        be  = breakeven.get(col, 0)
        all_pct  = results[col].mean() * 100
        buy_pct  = buys[col].mean()   * 100
        sell_pct = sells[col].mean()  * 100
        flag = " ✓" if all_pct >= be else ""
        print(f"  {col.replace('hit_t',''):>8}R  {be:>9.1f}%  "
              f"{all_pct:>7.1f}%  {buy_pct:>7.1f}%  {sell_pct:>7.1f}%{flag}")
