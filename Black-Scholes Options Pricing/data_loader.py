"""
Loads spot prices and options data; computes rolling historical volatility.

Filename conventions
--------------------
Weekly  (NIFTY / BANKNIFTY):   SYMBOL + YY + M + DD + STRIKE + CE/PE
  e.g.  NIFTY2651921100CE  → 2026-05-19, strike 21100
Monthly (stocks):              SYMBOL + YY + MMM + STRIKE + CE/PE
  e.g.  HDFCBANK26MAY1000CE → last Thursday of May 2026, strike 1000
"""
import os
import re
import logging
from calendar import monthrange
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import OPTIONS_DIR, SPOT_DIR, HIST_VOL_WINDOW, RISK_FREE_RATE

log = logging.getLogger(__name__)

# ── expiry date helpers ───────────────────────────────────────────────────────

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def _last_thursday(year: int, month: int) -> datetime:
    """Return the last Thursday of the given month (NSE monthly expiry)."""
    _, n_days = monthrange(year, month)
    for day in range(n_days, 0, -1):
        if datetime(year, month, day).weekday() == 3:  # Thursday = 3
            return datetime(year, month, day)
    raise ValueError(f"No Thursday in {year}-{month}?")


# ── filename parser ───────────────────────────────────────────────────────────

# Weekly:  NIFTY26 5 19 21100 CE
_WEEKLY_RE  = re.compile(r"^([A-Z&\-]+)(\d{2})(\d)(\d{2})(\d+)(CE|PE)$")
# Monthly: HDFCBANK 26 MAY 1000 CE
_MONTHLY_RE = re.compile(r"^([A-Z&\-]+)(\d{2})([A-Z]{3})(\d+)(CE|PE)$")


def parse_option_filename(filename: str):
    """
    Returns (symbol, expiry_date, strike, option_type) or None on failure.
    option_type is 'CE' or 'PE'.
    """
    name = filename.replace(".csv", "")

    m = _WEEKLY_RE.match(name)
    if m:
        sym, yy, mm, dd, strike, otype = m.groups()
        try:
            expiry = datetime(2000 + int(yy), int(mm), int(dd))
        except ValueError:
            return None
        return sym, expiry, int(strike), otype

    m = _MONTHLY_RE.match(name)
    if m:
        sym, yy, mon, strike, otype = m.groups()
        if mon not in MONTH_MAP:
            return None
        year  = 2000 + int(yy)
        month = MONTH_MAP[mon]
        expiry = _last_thursday(year, month)
        return sym, expiry, int(strike), otype

    return None


# ── spot data ─────────────────────────────────────────────────────────────────

def load_spot(symbol: str) -> pd.DataFrame | None:
    """
    Load daily OHLCV for *symbol* from dataset/.
    Returns DataFrame indexed by date (tz-naive), or None if file missing.
    """
    path = os.path.join(SPOT_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date").set_index("date")
    return df


def compute_hist_vol(spot_df: pd.DataFrame, window: int = HIST_VOL_WINDOW) -> pd.Series:
    """
    Annualised historical volatility (rolling log-return std × √252).
    Index aligns with spot_df.
    """
    log_ret = np.log(spot_df["close"] / spot_df["close"].shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ── options data ──────────────────────────────────────────────────────────────

def load_option(path: str) -> pd.DataFrame | None:
    """Load one option CSV; return None if empty after filtering."""
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.sort_values("date").set_index("date")
    return df


# ── master record builder ─────────────────────────────────────────────────────

def build_records(symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Walk the options directory and build a flat DataFrame with one row per
    (symbol, date, contract) containing all fields needed for B-S pricing.

    Columns
    -------
    symbol, date, strike, option_type, expiry,
    market_price, volume, oi, spot, hist_vol, T, r,
    iv_provided  (implied vol from data, or NaN)
    """
    rows = []
    all_symbols = sorted(os.listdir(OPTIONS_DIR))
    if symbols:
        all_symbols = [s for s in all_symbols if s in symbols]

    for sym in all_symbols:
        sym_dir = os.path.join(OPTIONS_DIR, sym)
        if not os.path.isdir(sym_dir):
            continue

        spot_df = load_spot(sym)
        if spot_df is None or len(spot_df) < HIST_VOL_WINDOW + 1:
            log.debug("No spot data for %s — skipping", sym)
            continue

        hist_vol_series = compute_hist_vol(spot_df)

        for fname in os.listdir(sym_dir):
            if not fname.endswith(".csv"):
                continue
            parsed = parse_option_filename(fname)
            if parsed is None:
                log.debug("Unparseable filename: %s/%s", sym, fname)
                continue
            _, expiry, strike, otype = parsed

            opt_df = load_option(os.path.join(sym_dir, fname))
            if opt_df is None:
                continue

            expiry_date = pd.Timestamp(expiry)

            for trade_date, row in opt_df.iterrows():
                market_price = row.get("close", np.nan)
                if pd.isna(market_price) or market_price <= 0:
                    continue

                # Spot on trade date
                if trade_date not in spot_df.index:
                    continue
                S = spot_df.loc[trade_date, "close"]
                if pd.isna(S) or S <= 0:
                    continue

                # Historical vol on trade date
                hv = hist_vol_series.get(trade_date, np.nan)
                if pd.isna(hv) or hv <= 0:
                    continue

                # Time to expiry in years (calendar days / 365)
                days_left = (expiry_date - trade_date).days
                T = max(days_left / 365.0, 0.0)

                rows.append({
                    "symbol":       sym,
                    "date":         trade_date,
                    "strike":       strike,
                    "option_type":  otype,
                    "expiry":       expiry_date,
                    "market_price": market_price,
                    "volume":       row.get("volume", 0),
                    "oi":           row.get("oi", 0),
                    "spot":         S,
                    "hist_vol":     hv,
                    "T":            T,
                    "r":            RISK_FREE_RATE,
                    "iv_provided":  row.get("iv", np.nan),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["is_put"] = df["option_type"] == "PE"
    return df
