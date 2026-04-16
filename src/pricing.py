"""
pricing.py
==========
BRC payoff calculation, BGK continuity correction, and price aggregation.

TONIGHT'S TASK — MODULE 4  (~10 min)
--------------------------------------
This is the financial heart of the pricer.
Work through the functions in order:
  bgk_adjusted_barrier  →  compute_brc_payoff  →  price_brc  →  compare_flat_vs_skew

The payoff logic in compute_brc_payoff has 3 cases — think through each one.

Run sanity check:  python src/pricing.py
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BGK_BETA = 0.5826   # = -ζ(½) / √(2π)  (Broadie-Glasserman-Kou, 1997)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BRCParams:
    S0: float
    barrier_pct: float   # e.g. 0.60 for a 60% barrier
    coupon: float        # annualised coupon, e.g. 0.08 for 8%
    T: float             # maturity in years
    r: float             # risk-free rate (continuous)
    notional: float = 1000.0


@dataclass
class PriceResult:
    price: float
    std_err: float
    ci_low: float
    ci_high: float
    barrier_hit_pct: float   # fraction of paths [0, 1] that hit the barrier


# ---------------------------------------------------------------------------
# 1. BGK continuity correction
# ---------------------------------------------------------------------------

def bgk_adjusted_barrier(barrier: float, sigma: float, dt: float) -> float:
    """
    Adjust a discrete monitoring barrier to its continuous equivalent.

    For a DOWN-AND-IN barrier (BRC case):
        B_adj = B · exp( -β* · σ · √Δt )

    The adjusted barrier is *lower* than B, meaning the effective barrier
    is harder to breach → the BRC is slightly more expensive.

    Parameters
    ----------
    barrier : unadjusted barrier level  (e.g. 60.0 if S0=100, barrier_pct=0.60)
    sigma   : volatility used in simulation
    dt      : time step size (e.g. 1/252 for daily)

    Formula reminder
    ----------------
    return barrier * np.exp(-BGK_BETA * sigma * np.sqrt(dt))
    """
    return barrier * np.exp(-BGK_BETA * sigma * np.sqrt(dt))


# ---------------------------------------------------------------------------
# 2. BRC payoff per path
# ---------------------------------------------------------------------------

def compute_brc_payoff(
    paths: np.ndarray,
    params: BRCParams,
    sigma: float,
    use_bgk: bool = True,
) -> np.ndarray:
    """
    Compute the BRC payoff for every simulated path.

    Payoff formula (per path):
    ─────────────────────────
    Let:
      - barrier_hit  = (min of path) ≤ B_adj
      - S_T          = terminal spot price  (last column of path)
      - coupon_cash  = notional · coupon · T    (total coupon, not annualised!)

    Case 1 — barrier NEVER touched:
        payoff = notional + coupon_cash

    Case 2 — barrier touched AND S_T ≥ S0:
        payoff = notional + coupon_cash    (investor recovers — full capital)

    Case 3 — barrier touched AND S_T < S0:
        payoff = notional · (S_T / S0) + coupon_cash    (capital loss)

    Implementation steps
    --------------------
    n_steps     = paths.shape[1] - 1
    dt          = params.T / n_steps
    barrier     = params.S0 * params.barrier_pct
    if use_bgk:
        barrier = bgk_adjusted_barrier(barrier, sigma, dt)

    S_T          = paths[:, -1]                         # shape (n_sims,)
    barrier_hit  = paths.min(axis=1) <= barrier         # bool array

    coupon_cash  = params.notional * params.coupon * params.T

    payoffs = np.where(
        ~barrier_hit,
        params.notional + coupon_cash,                              # case 1 & 2 (no hit)
        np.where(
            S_T >= params.S0,
            params.notional + coupon_cash,                          # case 2 (hit, recovery)
            params.notional * (S_T / params.S0) + coupon_cash,     # case 3 (loss)
        ),
    )
    return payoffs
    """
    n_steps     = paths.shape[1] - 1
    dt          = params.T / n_steps
    barrier     = params.S0 * params.barrier_pct
    if use_bgk:
        barrier = bgk_adjusted_barrier(barrier, sigma, dt)
    S_T          = paths[:, -1]                         # shape (n_sims,)
    barrier_hit  = paths.min(axis=1) <= barrier         # bool array
    coupon_cash  = params.notional * params.coupon * params.T
    payoffs = np.where(
        ~barrier_hit,
        params.notional + coupon_cash,                              # case 1 & 2 (no hit)
        np.where(
            S_T >= params.S0,
            params.notional + coupon_cash,                          # case 2 (hit, recovery)
            params.notional * (S_T / params.S0) + coupon_cash,     # case 3 (loss)
        ),
    )
    return payoffs
    


# ---------------------------------------------------------------------------
# 3. Price BRC from paths
# ---------------------------------------------------------------------------

def price_brc(
    paths: np.ndarray,
    params: BRCParams,
    sigma: float,
    use_bgk: bool = True,
) -> PriceResult:
    """
    Aggregate payoffs into a discounted price with confidence interval.

    Steps
    -----
    payoffs        = compute_brc_payoff(paths, params, sigma, use_bgk)
    n_sims         = len(payoffs)
    discount       = np.exp(-params.r * params.T)

    price          = discount * np.mean(payoffs)
    std_err        = discount * np.std(payoffs, ddof=1) / np.sqrt(n_sims)
    ci_low         = price - 1.96 * std_err
    ci_high        = price + 1.96 * std_err
    barrier_hit_pct = (paths.min(axis=1) <= params.S0 * params.barrier_pct).mean()

    return PriceResult(price, std_err, ci_low, ci_high, barrier_hit_pct)
    """
    payoffs         = compute_brc_payoff(paths, params, sigma, use_bgk)
    n_sims          = len(payoffs)
    discount        = np.exp(-params.r * params.T)
    price           = discount * np.mean(payoffs)
    std_err         = discount * np.std(payoffs, ddof=1) / np.sqrt(n_sims)
    ci_low          = price - 1.96 * std_err
    ci_high         = price + 1.96 * std_err
    barrier_hit_pct = (paths.min(axis=1) <= params.S0 * params.barrier_pct).mean()
    return PriceResult(price, std_err, ci_low, ci_high, barrier_hit_pct)


# ---------------------------------------------------------------------------
# 4. Hero feature: flat vol vs skew-adjusted vol (CRN)
# ---------------------------------------------------------------------------

def compare_flat_vs_skew(
    paths: np.ndarray,
    params: BRCParams,
    sigma_flat: float,
    sigma_skew: float,
) -> dict:
    """
    Price the BRC twice — once with flat vol, once with skew vol —
    using the SAME set of paths (Common Random Numbers).

    CRN ensures the variance of the *difference* is small.

    Returns
    -------
    dict with keys:
      "flat"         : PriceResult under flat vol
      "skew"         : PriceResult under skew-adjusted vol
      "diff_bps"     : (skew_price - flat_price) / notional * 10_000  [basis points]

    Think about it:
      diff_bps > 0 means skew increases the price of the BRC
      (the DIPoP is more expensive → the product is worth more to the issuer,
       and a larger coupon needs to be offered to the investor)
    """
    result_flat = price_brc(paths, params, sigma_flat)
    result_skew = price_brc(paths, params, sigma_skew)
    diff_bps = (result_skew.price - result_flat.price) / params.notional * 10_000
    return {"flat": result_flat, "skew": result_skew, "diff_bps": diff_bps}


# ---------------------------------------------------------------------------
# Sanity check  (run: python src/pricing.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from monte_carlo import simulate_paths

    # Parameters
    params = BRCParams(
        S0=100.0,
        barrier_pct=0.65,
        coupon=0.08,
        T=1.0,
        r=0.03,
        notional=1000.0,
    )
    sigma_flat = 0.20
    sigma_skew = 0.28   # vol at barrier (deep OTM, left skew boosts it)

    # Simulate once — use same paths for both prices (CRN)
    paths = simulate_paths(
        params.S0, params.r, sigma_skew, params.T,
        n_steps=252, n_sims=50_000, antithetic=True, seed=42
    )

    # BGK check
    barrier = params.S0 * params.barrier_pct
    B_adj = bgk_adjusted_barrier(barrier, sigma_flat, 1/252)
    print(f"Barrier      : {barrier:.2f}")
    print(f"BGK adjusted : {B_adj:.4f}  (should be slightly below {barrier:.2f})\n")

    # Price comparison
    result = compare_flat_vs_skew(paths, params, sigma_flat, sigma_skew)
    flat = result["flat"]
    skew = result["skew"]
    print(f"Flat vol price   : {flat.price:.2f}  ± {flat.std_err:.3f}")
    print(f"  Barrier hit %  : {flat.barrier_hit_pct*100:.1f}%")
    print(f"Skew vol price   : {skew.price:.2f}  ± {skew.std_err:.3f}")
    print(f"  Barrier hit %  : {skew.barrier_hit_pct*100:.1f}%")
    print(f"\nSkew premium     : {result['diff_bps']:+.1f} bps")
    print("(positive = skew makes BRC more expensive)")
