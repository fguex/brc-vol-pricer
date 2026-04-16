"""
vol_surface.py
==============
Volatility surface utilities:
  1. Black-Scholes closed-form call pricer
  2. Implied vol inversion (Brent's method)
  3. Quadratic smile fit:  σ(m) = σ_atm + α·m + β·m²
  4. Vol look-up at arbitrary strike (e.g. barrier)

TONIGHT'S TASK — MODULE 2  (~15 min)
--------------------------------------
Work through the four functions in order — each builds on the previous one.
The most important one to understand conceptually is implied_vol():
you are "inverting" the BS formula numerically.

Run sanity check:  python src/vol_surface.py
"""

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SmileParams:
    sigma_atm: float   # ATM vol level  σ_atm
    alpha: float       # skew coefficient  α  (negative for equity left skew)
    beta: float        # curvature coefficient  β  (positive, smile shape)
    maturity: float    # T in years (informational, not used in formula)


# ---------------------------------------------------------------------------
# 1. Black-Scholes call price
# ---------------------------------------------------------------------------

def bs_call_price(S: float, K: float, T: float,
                  r: float, sigma: float, q: float = 0.0) -> float:
    """
    Return the Black-Scholes price of a European call option.

    Parameters
    ----------
    S     : spot price
    K     : strike
    T     : time to maturity (years)
    r     : risk-free rate (continuous, annualised)
    sigma : volatility (annualised)
    q     : continuous dividend yield (default 0)

    Formula reminder
    ----------------
    d1 = [ ln(S/K) + (r - q + 0.5·σ²)·T ] / (σ·√T)
    d2 = d1 - σ·√T
    C  = S·e^{-q·T}·N(d1) - K·e^{-r·T}·N(d2)

    Use: norm.cdf(x) from scipy.stats for N(·)
    Edge case: if T <= 0 return max(S - K, 0)
    """
    if T <= 0:
        return max(S - K, 0.0)
    else:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    raise NotImplementedError("Implement bs_call_price()")


# ---------------------------------------------------------------------------
# 2. Implied volatility (numerical inversion)
# ---------------------------------------------------------------------------

def implied_vol(market_price: float, S: float, K: float,
                T: float, r: float, q: float = 0.0) -> float:
    """
    Find the implied volatility σ such that bs_call_price(..., σ) = market_price.

    Uses Brent's root-finding method on the interval [1e-4, 5.0].

    Returns np.nan if:
      - T <= 0
      - market_price is below intrinsic value (no solution exists)
      - brentq raises ValueError (e.g. same sign at both ends)

    Implementation hint
    -------------------
    objective = lambda sigma: bs_call_price(S, K, T, r, sigma, q) - market_price
    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan
    """
    # -----------------------------------------------------------------------
    if (T <= 0 or market_price < max(S*np.exp(-q*T) - K*np.exp(-r*T), 0)):
        return np.nan
    # -----------------------------------------------------------------------
    objective = lambda sigma: bs_call_price(S, K, T, r, sigma, q) - market_price
    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan
    
# ---------------------------------------------------------------------------
# 3. Quadratic smile fit
# ---------------------------------------------------------------------------

def fit_smile(strikes: np.ndarray, impl_vols: np.ndarray,
              forward: float, maturity: float) -> SmileParams:
    """
    Fit the quadratic parametric smile:
        σ(m) = σ_atm + α·m + β·m²     where m = ln(K/F)

    Parameters
    ----------
    strikes   : array of strikes where implied vol is known
    impl_vols : corresponding implied volatilities (same length, no NaNs)
    forward   : F = S·e^{(r-q)·T}
    maturity  : T (stored in SmileParams, not used in fitting)

    Implementation hint
    -------------------
    1. log_m = np.log(strikes / forward)          # log-moneyness array
    2. Build the design matrix A:
         each row = [1, m_i, m_i²]
         A = np.column_stack([np.ones_like(log_m), log_m, log_m**2])
    3. Use np.linalg.lstsq(A, impl_vols, rcond=None) to solve for [σ_atm, α, β]
       (Ordinary Least Squares — closed form, no optimiser needed)
    4. Clip σ_atm > 0 and β > 0 for stability
    5. Return SmileParams(...)

    Bonus: filter out NaN implied vols before fitting.
    """
    log_m = np.log(strikes / forward)
    A = np.column_stack([np.ones_like(log_m), log_m, log_m**2])
    coeffs, _, _, _ = np.linalg.lstsq(A, impl_vols, rcond=None)
    sigma_atm, alpha, beta = coeffs
    sigma_atm = max(sigma_atm, 1e-4)
    beta = max(beta, 1e-4)
    return SmileParams(sigma_atm=sigma_atm, alpha=alpha, beta=beta, maturity=maturity)


# ---------------------------------------------------------------------------
# 4. Vol look-up at a given strike (e.g. barrier)
# ---------------------------------------------------------------------------

def get_vol_for_strike(K: float, S: float, T: float,
                       r: float, params: SmileParams,
                       q: float = 0.0) -> float:
    """
    Evaluate the fitted smile at strike K.

    Formula
    -------
    F = S · exp((r - q) · T)
    m = ln(K / F)
    σ = σ_atm + α·m + β·m²
    return np.clip(σ, 0.01, 2.0)   # safety clip

    This is the vol you will use for the barrier (deep OTM put).
    Notice: m < 0 for K < F (OTM put), and with α < 0
    the skew increases σ for OTM puts — that is the whole point.
    """

    F = S * np.exp((r - q) * T)
    m = np.log(K / F)
    sigma = params.sigma_atm + params.alpha * m + params.beta * m**2
    return np.clip(sigma, 0.01, 2.0)



# ---------------------------------------------------------------------------
# Sanity check  (run: python src/vol_surface.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Test bs_call_price --
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.03, 0.20
    price = bs_call_price(S, K, T, r, sigma)
    print(f"ATM call price: {price:.4f}  (expect ~10.45 for S=K=100, T=1, r=3%, σ=20%)")

    # -- Test implied_vol --
    iv = implied_vol(price, S, K, T, r)
    print(f"Recovered implied vol: {iv:.4f}  (expect 0.2000)")

    # -- Test fit_smile --
    # Synthetic data: true smile = 0.20 - 0.05*m + 0.10*m²
    fwd = S * np.exp(r * T)
    test_strikes = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)
    log_m = np.log(test_strikes / fwd)
    true_vols = 0.20 - 0.05 * log_m + 0.10 * log_m**2
    params = fit_smile(test_strikes, true_vols, fwd, T)
    print(f"\nFitted smile params:")
    print(f"  σ_atm = {params.sigma_atm:.4f}  (expect ~0.20)")
    print(f"  α     = {params.alpha:.4f}  (expect ~-0.05)")
    print(f"  β     = {params.beta:.4f}   (expect ~0.10)")

    # -- Test get_vol_for_strike at barrier (60% of spot) --
    barrier = 0.60 * S
    sigma_barrier = get_vol_for_strike(barrier, S, T, r, params)
    print(f"\nVol at 60% barrier: {sigma_barrier:.4f}")
    print(f"ATM vol            : {params.sigma_atm:.4f}")
    print(f"Skew premium       : {(sigma_barrier - params.sigma_atm)*100:.1f} vol points")
