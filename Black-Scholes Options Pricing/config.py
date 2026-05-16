import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
OPTIONS_DIR = os.path.join(DATASET_DIR, "options")
SPOT_DIR    = DATASET_DIR

# Black-Scholes parameters
RISK_FREE_RATE   = 0.07   # India 10-year G-Sec (~7%)
HIST_VOL_WINDOW  = 30     # trading days for rolling historical volatility
MIN_MARKET_PRICE = 0.05   # ignore options with price below this (noise)
MIN_VOLUME       = 1      # exclude stale zero-volume rows (stale last-traded prices)
MIN_OI           = 100    # exclude contracts with negligible open interest
MAX_MONEYNESS    = 0.30   # exclude options deeper than 30% ITM or OTM (illiquid extremes)

# Mismatch reporting thresholds
ABS_MISMATCH_THRESHOLD = 0.50   # flag if |BS - market| >= ₹0.50
REL_MISMATCH_THRESHOLD = 0.15   # flag if |BS - market| / market >= 15%

# Batch size for GPU tensor allocation (tune down if OOM)
GPU_BATCH_SIZE = 500_000
