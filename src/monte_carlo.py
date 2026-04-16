"""
monte_carlo.py
==============
GBM path simulation for the BRC pricer.

TONIGHT'S TASK — MODULE 3  (~10 min)
--------------------------------------
Implement simulate_paths().
This is the numerically heavy core of the pricer.
Think about shapes: paths is (n_sims, n_steps+1).
The antithetic trick doubles your effective sample size for free.

Run sanity check:  python src/monte_carlo.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Core GBM path simulator
# ---------------------------------------------------------------------------

def simulate_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_sims: int,
    antithetic: bool = True,
    seed: int = None,
    q: float = 0.0,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths under the risk-neutral measure.

    GBM exact solution for one step Δt:
        S_{t+Δt} = S_t · exp( (r - q - ½σ²)·Δt  +  σ·√Δt·Z )
        where Z ~ N(0,1)

    Parameters
    ----------
    S0         : initial spot price
    r          : risk-free rate (continuous, annualised)
    sigma      : constant volatility (annualised)
    T          : total time horizon (years)
    n_steps    : number of time steps  (use 252 for daily steps over 1 year)
    n_sims     : number of Monte Carlo paths
                 If antithetic=True, must be even; you draw n_sims//2 normals
                 and mirror them.
    antithetic : if True, use antithetic variates (variance reduction)
    seed       : optional random seed for reproducibility
    q          : continuous dividend yield (default 0.0)

    Returns
    -------
    paths : np.ndarray of shape (n_sims, n_steps + 1)
            paths[:, 0]  == S0  (all paths start at S0)
            paths[:, -1] == S_T (terminal prices)

    Implementation steps
    --------------------
    dt     = T / n_steps
    drift  = (r - q - 0.5 * sigma**2) * dt      # scalar  ← q subtracted here
    diff   = sigma * np.sqrt(dt)                 # scalar

    if antithetic:
        half = n_sims // 2
        Z_half = rng.standard_normal((half, n_steps))   # shape (half, n_steps)
        Z = np.vstack([Z_half, -Z_half])                # shape (n_sims, n_steps)
    else:
        Z = rng.standard_normal((n_sims, n_steps))

    log_returns = drift + diff * Z               # shape (n_sims, n_steps)
    # cumsum along axis=1 gives cumulative log-return at each step
    log_paths   = np.cumsum(log_returns, axis=1) # shape (n_sims, n_steps)
    prices      = S0 * np.exp(log_paths)         # shape (n_sims, n_steps)
    # prepend the S0 column
    paths = np.hstack([np.full((n_sims, 1), S0), prices])
    """
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    if antithetic:
        half = n_sims // 2
        Z_half = np.random.standard_normal((half, n_steps))
        Z = np.vstack([Z_half, -Z_half])
    else:
        Z = np.random.standard_normal((n_sims, n_steps))
    log_returns = drift + diff * Z
    prices = S0 * np.exp(np.cumsum(log_returns, axis=1))  # shape (n_sims, n_steps)
    paths = np.hstack([np.full((n_sims, 1), S0), prices])  # shape (n_sims, n_steps+1)

    return paths


# ---------------------------------------------------------------------------
# 2. Skew-adjusted simulator (v1 simplification)
# ---------------------------------------------------------------------------

def simulate_paths_skew(
    S0: float,
    r: float,
    T: float,
    n_steps: int,
    n_sims: int,
    smile_params,            # SmileParams from vol_surface.py
    barrier_pct: float,      # barrier as fraction of S0, e.g. 0.60
    antithetic: bool = True,
    seed: int = None,
) -> np.ndarray:
    """
    Simulate paths using a single vol calibrated at the barrier strike.

    v1 simplification: rather than a full local vol surface, we use
    σ_skew = σ(K=barrier) as a constant vol for all paths.
    This is an approximation — note it for interviews.

    Implementation hint
    -------------------
    from vol_surface import get_vol_for_strike
    barrier_strike = S0 * barrier_pct
    sigma_skew = get_vol_for_strike(barrier_strike, S0, T, r, smile_params)
    return simulate_paths(S0, r, sigma_skew, T, n_steps, n_sims, antithetic, seed)
    """
    from vol_surface import get_vol_for_strike
    barrier_strike = S0 * barrier_pct
    sigma_skew = get_vol_for_strike(barrier_strike, S0, T, r, smile_params)
    return simulate_paths(S0, r, sigma_skew, T, n_steps, n_sims, antithetic, seed)
    

# ---------------------------------------------------------------------------
# Sanity check  (run: python src/monte_carlo.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S0, r, sigma, T, q = 100.0, 0.03, 0.20, 1.0, 0.02   # q=2% dividend yield
    n_steps = 252   # daily steps
    n_sims  = 10_000

    paths = simulate_paths(S0, r, sigma, T, n_steps, n_sims, antithetic=True, seed=42, q=q)

    print(f"Path shape          : {paths.shape}   (expect ({n_sims}, {n_steps+1}))")
    print(f"All paths start at  : {np.unique(paths[:, 0])}  (expect [100.0])")
    print(f"Mean S_T            : {paths[:, -1].mean():.2f}  (expect ~{S0 * np.exp((r - q) * T):.2f}  = S0·e^(r-q)T)")
    print(f"Std  S_T            : {paths[:, -1].std():.2f}")

    # Check antithetic symmetry: first half and second half should sum to ~2·forward per path
    half = n_sims // 2
    mean_sum = (paths[:half, -1] + paths[half:, -1]).mean()
    fwd = S0 * np.exp((r - q) * T)
    print(f"Mean(S_T + S_T_anti): {mean_sum:.2f}  (should be ~2·{fwd:.2f} = {2*fwd:.2f})")
