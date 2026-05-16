"""
Black-Scholes Mispricing Scanner
=================================
Loads all options data → runs B-S pricing on GPU in one big batch →
flags rows where market price diverges from theoretical price.

Two mismatch checks:
  1. hist_vol_mismatch : |BS(hist_vol) - market| / market  ← main signal
  2. iv_bs_check       : |BS(IV_provided) - market| / market ← data-quality check

Usage
-----
    python scanner.py                    # all symbols
    python scanner.py NIFTY HDFCBANK    # specific symbols only
    python scanner.py --threshold 0.10  # override rel threshold (10%)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bs_engine import DEVICE, black_scholes_batch
from dashboard import build_dashboard
from config import (
    ABS_MISMATCH_THRESHOLD,
    GPU_BATCH_SIZE,
    MAX_MONEYNESS,
    MIN_MARKET_PRICE,
    MIN_OI,
    MIN_VOLUME,
    REL_MISMATCH_THRESHOLD,
)
from data_loader import build_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── GPU batch pricing ─────────────────────────────────────────────────────────

def _price_in_batches(df: pd.DataFrame, r: float) -> np.ndarray:
    """
    Run Black-Scholes on the full DataFrame in chunks that fit in GPU memory.
    Returns BS prices array aligned with df index.
    """
    N = len(df)
    bs_prices = np.full(N, np.nan, dtype=np.float64)

    for start in range(0, N, GPU_BATCH_SIZE):
        end   = min(start + GPU_BATCH_SIZE, N)
        chunk = df.iloc[start:end]

        bs_prices[start:end] = black_scholes_batch(
            S      = chunk["spot"].to_numpy(dtype=np.float64),
            K      = chunk["strike"].to_numpy(dtype=np.float64),
            T      = chunk["T"].to_numpy(dtype=np.float64),
            sigma  = chunk["hist_vol"].to_numpy(dtype=np.float64),
            is_put = chunk["is_put"].to_numpy(dtype=bool),
            r      = r,
        )

    return bs_prices


def _price_iv_batch(df: pd.DataFrame, r: float) -> np.ndarray:
    """Same but uses the IV that came with the data (where available)."""
    mask = df["iv_provided"].notna() & (df["iv_provided"] > 0)
    out  = np.full(len(df), np.nan, dtype=np.float64)

    sub = df[mask]
    if sub.empty:
        return out

    out[mask.to_numpy()] = black_scholes_batch(
        S      = sub["spot"].to_numpy(dtype=np.float64),
        K      = sub["strike"].to_numpy(dtype=np.float64),
        T      = sub["T"].to_numpy(dtype=np.float64),
        sigma  = sub["iv_provided"].to_numpy(dtype=np.float64),
        is_put = sub["is_put"].to_numpy(dtype=bool),
        r      = r,
    )
    return out


# ── analysis ──────────────────────────────────────────────────────────────────

def add_mismatch_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── historical-vol B-S price ──────────────────────────────────────────────
    log.info("Running GPU Black-Scholes (hist vol) on %d rows via %s …", len(df), DEVICE)
    t0 = time.perf_counter()
    df["bs_price_hist"] = _price_in_batches(df, df["r"].iloc[0])
    log.info("  Done in %.2fs", time.perf_counter() - t0)

    # ── IV verification B-S price ─────────────────────────────────────────────
    log.info("Running GPU Black-Scholes (provided IV) on %d rows …", df["iv_provided"].notna().sum())
    df["bs_price_iv"] = _price_iv_batch(df, df["r"].iloc[0])

    # ── mismatch metrics ──────────────────────────────────────────────────────
    mp = df["market_price"]

    df["abs_mismatch_hist"] = (df["bs_price_hist"] - mp).abs()
    df["rel_mismatch_hist"] = df["abs_mismatch_hist"] / mp

    df["abs_mismatch_iv"]   = (df["bs_price_iv"] - mp).abs()
    df["rel_mismatch_iv"]   = df["abs_mismatch_iv"] / mp

    # Direction: positive = option is CHEAP vs B-S (buy signal), negative = EXPENSIVE
    df["edge_hist"] = df["bs_price_hist"] - mp  # positive → market underprices

    return df


def flag_mispricings(
    df:            pd.DataFrame,
    rel_threshold: float = REL_MISMATCH_THRESHOLD,
    abs_threshold: float = ABS_MISMATCH_THRESHOLD,
) -> pd.DataFrame:
    """Return only rows that breach both the relative AND absolute threshold."""
    breach = (
        (df["rel_mismatch_hist"] >= rel_threshold) &
        (df["abs_mismatch_hist"] >= abs_threshold)
    )
    return df[breach].copy()


def print_summary(df: pd.DataFrame, mispricings: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  BLACK-SCHOLES MISPRICING SCANNER  —  SUMMARY")
    print("=" * 70)
    print(f"  Device            : {DEVICE}")
    print(f"  Total rows priced : {len(df):,}")
    print(f"  Mispricings found : {len(mispricings):,}")

    if not mispricings.empty:
        pct = len(mispricings) / len(df) * 100
        print(f"  Mispricing rate   : {pct:.2f}%")

        by_sym = mispricings.groupby("symbol").size().sort_values(ascending=False)
        print(f"\n  Top symbols by mismatch count:")
        for sym, cnt in by_sym.head(10).items():
            print(f"    {sym:<20} {cnt:>6,}")

        print(f"\n  Avg relative mismatch : {mispricings['rel_mismatch_hist'].mean():.1%}")
        print(f"  Avg absolute mismatch : ₹{mispricings['abs_mismatch_hist'].mean():.2f}")

        overpriced  = (mispricings["edge_hist"] < 0).sum()
        underpriced = (mispricings["edge_hist"] > 0).sum()
        print(f"\n  Overpriced  (sell signal) : {overpriced:,}")
        print(f"  Underpriced (buy signal)  : {underpriced:,}")

    if "abs_mismatch_iv" in df.columns:
        iv_rows = df["abs_mismatch_iv"].notna()
        bad_iv  = (df.loc[iv_rows, "rel_mismatch_iv"] > 0.01).sum()
        print(f"\n  IV data-quality check   : {bad_iv:,} rows where BS(IV) ≠ market (>1%)")

    print("=" * 70)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BS Mispricing Scanner")
    parser.add_argument(
        "symbols", nargs="*",
        help="Restrict to these symbols (e.g. NIFTY HDFCBANK). Default: all.",
    )
    parser.add_argument(
        "--threshold", type=float, default=REL_MISMATCH_THRESHOLD,
        help="Relative mismatch threshold (default %(default)s)",
    )
    parser.add_argument(
        "--abs-threshold", type=float, default=ABS_MISMATCH_THRESHOLD,
        help="Absolute mismatch threshold in ₹ (default %(default)s)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Show top-N mispricings in console output (default %(default)s)",
    )
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else None

    # ── 1. load data ──────────────────────────────────────────────────────────
    log.info("Loading option records …")
    t0 = time.perf_counter()
    df = build_records(symbols)
    if df.empty:
        log.error("No records loaded — check OPTIONS_DIR / symbol names.")
        sys.exit(1)

    log.info("Loaded %d records in %.2fs", len(df), time.perf_counter() - t0)

    # ── quality filters ───────────────────────────────────────────────────────
    before = len(df)
    df = df[df["market_price"] >= MIN_MARKET_PRICE]
    df = df[df["volume"] >= MIN_VOLUME]          # remove stale zero-volume rows
    df = df[df["oi"] >= MIN_OI]                  # remove low open-interest contracts

    # Moneyness filter: |S/K - 1| <= MAX_MONEYNESS
    df["moneyness"] = df["spot"] / df["strike"] - 1
    df = df[df["moneyness"].abs() <= MAX_MONEYNESS]

    df = df.reset_index(drop=True)
    log.info(
        "%d → %d records after quality filters (price≥%.2f, vol≥%d, oi≥%d, moneyness≤±%.0f%%)",
        before, len(df), MIN_MARKET_PRICE, MIN_VOLUME, MIN_OI, MAX_MONEYNESS * 100,
    )

    # ── 2. GPU pricing + mismatch ─────────────────────────────────────────────
    t_gpu = time.perf_counter()
    df = add_mismatch_columns(df)
    gpu_time = time.perf_counter() - t_gpu

    # ── 3. flag mispricings ───────────────────────────────────────────────────
    mispricings = flag_mispricings(df, args.threshold, args.abs_threshold)

    # ── 4. console summary ────────────────────────────────────────────────────
    print_summary(df, mispricings)

    # ── 5. HTML dashboard ─────────────────────────────────────────────────────
    out_dir  = Path(__file__).parent / "output"
    dash_path = build_dashboard(
        df=df,
        mispricings=mispricings,
        gpu_device=str(DEVICE),
        load_time=time.perf_counter() - t0,
        gpu_time=gpu_time,
        out_path=out_dir / "dashboard.html",
    )
    log.info("Dashboard saved → %s", dash_path)


if __name__ == "__main__":
    main()
