# 🔵 BRC Pricer — Practice Session Guide

> **Tonight's session: ~45–60 min**
> You'll implement every function from scratch, validate with the built-in sanity checks, and end with a working pricer.

---

## Setup (2 min)

```bash
cd /Users/felixguex/brc-vol-pricer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Session Roadmap

| # | File | Functions to implement | Validation | Time |
|---|------|------------------------|------------|------|
| 1 | `src/vol_surface.py` | `bs_call_price` → `implied_vol` → `fit_smile` → `get_vol_for_strike` | `python src/vol_surface.py` | ~15 min |
| 2 | `src/monte_carlo.py` | `simulate_paths` | `python src/monte_carlo.py` | ~10 min |
| 3 | `src/pricing.py` | `bgk_adjusted_barrier` → `compute_brc_payoff` → `price_brc` → `compare_flat_vs_skew` | `python src/pricing.py` | ~10 min |
| 4 | `src/market_data.py` | `fetch_market_data` → `get_risk_free_rate` | `python src/market_data.py` | ~10 min |
| 5 | `src/app.py` | Wire it all together | `streamlit run src/app.py` | bonus |

> **Recommended order**: start with `vol_surface.py` (pure maths, no network),
> then `monte_carlo.py` (numpy arrays), then `pricing.py` (business logic),
> then `market_data.py` (network, can fail gracefully), then the app.

---

## Module 1 — `vol_surface.py`

### Concept recap

The vol surface answers: *what volatility should I plug into Black-Scholes for this specific strike and maturity?*

For a BRC the critical strike is the **barrier** — typically 60–70% of spot. Because of the **left skew**, implied vol at the barrier is significantly higher than ATM vol. Ignoring this underprices the embedded down-and-in put.

### `bs_call_price(S, K, T, r, sigma, q=0.0)`

$$d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S e^{-qT} \mathcal{N}(d_1) - K e^{-rT} \mathcal{N}(d_2)$$

```python
# Hint: 5 lines
d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
```

**Sanity check**: ATM call (S=K=100, T=1, r=3%, σ=20%) ≈ **10.45**

---

### `implied_vol(market_price, S, K, T, r, q=0.0)`

You are *inverting* BS numerically — this is what every trading system does millions of times per day.

```python
# Hint: 6 lines
if T <= 0:
    return np.nan
objective = lambda sigma: bs_call_price(S, K, T, r, sigma, q) - market_price
try:
    return brentq(objective, 1e-4, 5.0)
except ValueError:
    return np.nan
```

**Sanity check**: `implied_vol(bs_call_price(100, 100, 1, 0.03, 0.20), 100, 100, 1, 0.03)` → **0.2000**

---

### `fit_smile(strikes, impl_vols, forward, maturity)`

Quadratic smile: $\sigma(m) = \sigma_{ATM} + \alpha m + \beta m^2$ where $m = \ln(K/F)$.

This is a **linear regression problem** — no gradient descent needed.

```python
# Hint: 5 lines
log_m = np.log(strikes / forward)
A = np.column_stack([np.ones_like(log_m), log_m, log_m**2])
coeffs, _, _, _ = np.linalg.lstsq(A, impl_vols, rcond=None)
sigma_atm, alpha, beta = coeffs
return SmileParams(max(sigma_atm, 1e-4), alpha, max(beta, 0), maturity)
```

---

### `get_vol_for_strike(K, S, T, r, params, q=0.0)`

```python
# Hint: 3 lines
F = S * np.exp((r - q) * T)
m = np.log(K / F)
return np.clip(params.sigma_atm + params.alpha * m + params.beta * m**2, 0.01, 2.0)
```

---

## Module 2 — `monte_carlo.py`

### Concept recap

You simulate many possible futures for the stock price. The average payoff across all futures, discounted back to today, is the fair price.

### `simulate_paths(S0, r, sigma, T, n_steps, n_sims, antithetic, seed)`

The key insight: instead of simulating $S_{t+1}$ from $S_t$ step by step (slow, accumulates errors), use the **exact solution** — simulate the *log-return* at each step:

$$\ln\frac{S_{t+\Delta t}}{S_t} = \underbrace{\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t}_{\text{drift}} + \underbrace{\sigma\sqrt{\Delta t} \cdot Z}_{\text{diffusion}}, \quad Z \sim \mathcal{N}(0,1)$$

```python
# Hint: ~10 lines
if seed is not None:
    rng = np.random.default_rng(seed)
else:
    rng = np.random.default_rng()

dt = T / n_steps
drift = (r - 0.5 * sigma**2) * dt
diff  = sigma * np.sqrt(dt)

if antithetic:
    half = n_sims // 2
    Z = np.vstack([rng.standard_normal((half, n_steps)),
                   rng.standard_normal((half, n_steps)) * -1])  # mirror
else:
    Z = rng.standard_normal((n_sims, n_steps))

log_returns = drift + diff * Z
prices = S0 * np.exp(np.cumsum(log_returns, axis=1))
return np.hstack([np.full((n_sims, 1), S0), prices])
```

**Sanity checks after implementing:**
- Shape must be `(n_sims, n_steps+1)`
- `paths[:, 0]` all equal to `S0`
- `mean(S_T)` ≈ `S0 · e^{rT}` (risk-neutral drift)

---

## Module 3 — `pricing.py`

### Concept recap

The BRC = bond + coupon − down-and-in put. Your job here is to evaluate the payoff function on each simulated path and discount back.

### `bgk_adjusted_barrier(barrier, sigma, dt)`

One line — the correction is tiny but matters for P&L:

$$B_{adj} = B \cdot e^{-\beta^* \sigma \sqrt{\Delta t}}, \quad \beta^* = 0.5826$$

---

### `compute_brc_payoff(paths, params, sigma, use_bgk)`

The three-case payoff logic (think of it as a decision tree):

```
barrier_hit?
  └─ No  → notional + coupon_cash                   (investor is happy)
  └─ Yes
       └─ S_T ≥ S0  → notional + coupon_cash         (recovery, still happy)
       └─ S_T < S0  → notional·(S_T/S0) + coupon_cash (capital loss)
```

Note: coupon is **always** paid — it compensates the investor for the barrier risk.

---

### `price_brc(paths, params, sigma, use_bgk)`

$$\hat{V}_0 = e^{-rT} \cdot \overline{\text{payoff}}, \quad \text{SE} = e^{-rT} \cdot \frac{\hat{\sigma}_{payoff}}{\sqrt{N}}$$

---

### `compare_flat_vs_skew(paths, params, sigma_flat, sigma_skew)`

This is the **hero feature**. Using the same paths (CRN):
1. Price once with `sigma_flat` (ATM vol — what you'd naively use)
2. Price once with `sigma_skew` (vol at barrier — skew-corrected)
3. Difference in basis points = impact of ignoring the skew

Expected result for a 65%-barrier 1Y BRC: **+50 to +150 bps** (skew makes it pricier).

---

## Module 4 — `market_data.py`

This is the plumbing. Work through it last so network issues don't block you.

```python
# fetch_market_data — core logic:
tkr  = yf.Ticker(ticker)
info = tkr.info
spot = info.get("regularMarketPrice") or info.get("previousClose")
div_yield = info.get("dividendYield") or 0.0
option_chain = {}
for expiry in tkr.options[:6]:
    opt = tkr.option_chain(expiry)
    option_chain[expiry] = {"calls": opt.calls, "puts": opt.puts}
return MarketData(ticker=ticker, spot=spot, rate=rate,
                  div_yield=div_yield, option_chain=option_chain)
```

---

## Quick verification sequence

After implementing all four modules, run these in order:

```bash
cd /Users/felixguex/brc-vol-pricer
source .venv/bin/activate

python src/vol_surface.py    # expect: ATM call ~10.45, recovered IV 0.2000
python src/monte_carlo.py    # expect: shape (10000, 253), mean S_T ~103
python src/pricing.py        # expect: prices ~950-1000, diff_bps > 0
python src/market_data.py    # expect: spot, expiry list, calls table
```

---

## Key concepts to be able to explain (structuring interview questions)

| Question | Where in the code |
|----------|------------------|
| What is a BRC? | `pricing.py` docstring + payoff formula |
| Why does the vol skew matter for barrier products? | `vol_surface.py` `get_vol_for_strike` |
| What is the BGK correction and why do we need it? | `pricing.py` `bgk_adjusted_barrier` |
| How does antithetic sampling work? | `monte_carlo.py` `simulate_paths` |
| What are Common Random Numbers? | `pricing.py` `compare_flat_vs_skew` |
| What is the Black-Scholes assumption being violated? | Constant σ → skew exists in practice |

---

## Architecture

```
brc-vol-pricer/
├── src/
│   ├── market_data.py   ← Module 4: yfinance wrapper
│   ├── vol_surface.py   ← Module 1: BS pricer + smile fitting
│   ├── monte_carlo.py   ← Module 2: GBM path simulation
│   ├── pricing.py       ← Module 3: BRC payoff + BGK + comparison
│   └── app.py           ← Bonus: Streamlit UI
├── BRC_Pricer_MasterDoc.html   ← Full mathematical reference
├── requirements.txt
└── SESSION_GUIDE.md            ← You are here
```
