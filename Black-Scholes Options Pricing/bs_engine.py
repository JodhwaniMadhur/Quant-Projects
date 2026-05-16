"""
Black-Scholes pricing engine backed by PyTorch.

Auto-selects device in order: CUDA → MPS (Apple Silicon) → CPU.
All public functions accept plain NumPy arrays and return NumPy arrays;
tensor lifetimes stay inside this module.
"""
import math
import numpy as np
import torch

# ── device selection ────────────────────────────────────────────────────────

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _best_device()
DTYPE  = torch.float32   # float64 on MPS has limited kernel support


def _to(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=DTYPE, device=DEVICE)


# ── core formula ─────────────────────────────────────────────────────────────

_SQRT2 = math.sqrt(2.0)

def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF via erf — works on every torch backend."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _bs_call_put(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (call_price, put_price) tensors.
    Rows where T <= 0 or sigma <= 0 get their intrinsic value.
    """
    valid = (T > 1e-6) & (sigma > 1e-6) & (S > 0) & (K > 0)

    # Safe compute: clamp to avoid log(0) / div-by-zero on invalid rows
    S_safe     = torch.where(valid, S,     torch.ones_like(S))
    K_safe     = torch.where(valid, K,     torch.ones_like(K))
    T_safe     = torch.where(valid, T,     torch.ones_like(T))
    sigma_safe = torch.where(valid, sigma, torch.ones_like(sigma))

    sqrt_T = torch.sqrt(T_safe)
    d1 = (torch.log(S_safe / K_safe) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    disc = torch.exp(torch.tensor(-r, dtype=DTYPE, device=DEVICE) * T_safe)
    call_bs = S_safe * _norm_cdf(d1)  - K_safe * disc * _norm_cdf(d2)
    put_bs  = K_safe * disc * _norm_cdf(-d2) - S_safe * _norm_cdf(-d1)

    # Intrinsic value for expired / zero-vol options
    call_intr = torch.clamp(S - K, min=0.0)
    put_intr  = torch.clamp(K - S, min=0.0)

    call = torch.where(valid, call_bs, call_intr)
    put  = torch.where(valid, put_bs,  put_intr)
    return call, put


# ── public batch API ─────────────────────────────────────────────────────────

def black_scholes_batch(
    S:      np.ndarray,
    K:      np.ndarray,
    T:      np.ndarray,
    sigma:  np.ndarray,
    is_put: np.ndarray,
    r:      float,
) -> np.ndarray:
    """
    Vectorised Black-Scholes on GPU.

    Parameters
    ----------
    S       : spot price         shape (N,)
    K       : strike             shape (N,)
    T       : years to expiry    shape (N,)
    sigma   : annualised vol     shape (N,)
    is_put  : bool array         shape (N,)   True → put (PE), False → call (CE)
    r       : scalar risk-free rate

    Returns
    -------
    bs_price : np.ndarray  shape (N,)   — NaN where inputs are degenerate
    """
    S_t      = _to(S)
    K_t      = _to(K)
    T_t      = _to(T)
    sigma_t  = _to(sigma)
    is_put_t = torch.as_tensor(is_put, dtype=torch.bool, device=DEVICE)

    call, put = _bs_call_put(S_t, K_t, T_t, r, sigma_t)
    price = torch.where(is_put_t, put, call)

    return price.cpu().numpy().astype(np.float64)


def compute_greeks_batch(
    S:      np.ndarray,
    K:      np.ndarray,
    T:      np.ndarray,
    sigma:  np.ndarray,
    is_put: np.ndarray,
    r:      float,
) -> dict[str, np.ndarray]:
    """
    Compute delta, gamma, vega, theta for all options in one GPU pass.
    Returns dict with keys: delta, gamma, vega, theta  (all shape (N,)).
    """
    S_t      = _to(S)
    K_t      = _to(K)
    T_t      = _to(T)
    sigma_t  = _to(sigma)
    is_put_t = torch.as_tensor(is_put, dtype=torch.bool, device=DEVICE)

    valid = (T_t > 1e-6) & (sigma_t > 1e-6) & (S_t > 0) & (K_t > 0)

    S_safe     = torch.where(valid, S_t,     torch.ones_like(S_t))
    K_safe     = torch.where(valid, K_t,     torch.ones_like(K_t))
    T_safe     = torch.where(valid, T_t,     torch.ones_like(T_t))
    sigma_safe = torch.where(valid, sigma_t, torch.ones_like(sigma_t))

    sqrt_T  = torch.sqrt(T_safe)
    d1 = (torch.log(S_safe / K_safe) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    # Standard normal PDF
    pdf_d1 = torch.exp(-0.5 * d1 ** 2) / _SQRT2 / math.sqrt(math.pi)

    disc = torch.exp(torch.tensor(-r, dtype=DTYPE, device=DEVICE) * T_safe)

    delta_call = _norm_cdf(d1)
    delta_put  = delta_call - 1.0
    delta = torch.where(is_put_t, delta_put, delta_call)

    gamma = pdf_d1 / (S_safe * sigma_safe * sqrt_T)

    vega  = S_safe * pdf_d1 * sqrt_T / 100.0  # per 1% vol move

    theta_call = (
        -(S_safe * pdf_d1 * sigma_safe) / (2.0 * sqrt_T)
        - r * K_safe * disc * _norm_cdf(d2)
    ) / 252.0
    theta_put = (
        -(S_safe * pdf_d1 * sigma_safe) / (2.0 * sqrt_T)
        + r * K_safe * disc * _norm_cdf(-d2)
    ) / 252.0
    theta = torch.where(is_put_t, theta_put, theta_call)

    nan_mask = ~valid

    def to_np(t):
        out = t.cpu().numpy().astype(np.float64)
        out[nan_mask.cpu().numpy()] = np.nan
        return out

    return {
        "delta": to_np(delta),
        "gamma": to_np(gamma),
        "vega":  to_np(vega),
        "theta": to_np(theta),
    }
