"""
market_data.py
==============
Wrapper around yfinance for fetching:
  - spot price
  - dividend yield
  - option chain (calls + puts per expiry)

TONIGHT'S TASK — MODULE 1  (~10 min)
--------------------------------------
Implement fetch_market_data() following the TODO comments.
The other functions are helpers that call yf.Ticker internally.

Run a quick sanity check at the bottom of this file
by executing:  python src/market_data.py
"""

import yfinance as yf
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    ticker: str
    spot: float                  # Last closing price  (S0)
    rate: float                  # Risk-free rate used for pricing
    div_yield: float             # Continuous dividend yield q
    option_chain: dict = field(default_factory=dict)
    # option_chain layout: { "YYYY-MM-DD": {"calls": DataFrame, "puts": DataFrame} }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_market_data(ticker: str, rate: float = 0.03) -> MarketData:
    """
    Download spot price and option chain for `ticker` from Yahoo Finance.

    Parameters
    ----------
    ticker : str   e.g. "NESN.SW", "AAPL", "^SPX"
    rate   : float risk-free rate (annualised, continuous) — default 3%

    Returns
    -------
    MarketData

    Implementation hints
    --------------------
    Step 1 — Create a yf.Ticker object.

    Step 2 — Extract spot price:
        info = tkr.info
        spot = info.get("regularMarketPrice") or info.get("previousClose")

    Step 3 — Extract dividend yield:
        div_yield = info.get("dividendYield") or 0.0

    Step 4 — Build option_chain dict:
        for expiry in tkr.options[:6]:          # limit to 6 nearest expiries
            opt = tkr.option_chain(expiry)
            option_chain[expiry] = {"calls": opt.calls, "puts": opt.puts}

    Step 5 — Return MarketData(...)
    """
    # -----------------------------------------------------------------------
    tkr = yf.Ticker(ticker)
    fi = tkr.fast_info
    spot = fi.get("lastPrice") or fi.get("previousClose")
    # fast_info doesn't have dividendYield — fall back to history-based estimate
    try:
        div_yield = tkr.info.get("dividendYield") or 0.0
    except Exception:
        div_yield = 0.0
    option_chain = {}
    for expiry in tkr.options[:6]:       # first 6 expiries only
        opt = tkr.option_chain(expiry)
        option_chain[expiry] = {"calls": opt.calls, "puts": opt.puts}
    return MarketData(ticker=ticker, spot=spot, rate=rate, div_yield=div_yield, option_chain=option_chain)


def get_risk_free_rate() -> float:
    """
    Return an estimate of the risk-free rate.

    Simple v1: fetch the 13-week T-bill yield from Yahoo Finance (ticker "^IRX").
    The rate is quoted as a percentage — divide by 100.

    If the fetch fails, fall back to 0.03.

    Implementation hint
    -------------------
        tkr = yf.Ticker("^IRX")
        rate_pct = tkr.fast_info["lastPrice"]   # e.g. 5.2 → 0.052
        return rate_pct / 100
    """
    tkr = yf.Ticker("^IRX")
    try:
        rate_pct = tkr.fast_info["lastPrice"]
        return rate_pct / 100
    except Exception:
        return 0.03



# ---------------------------------------------------------------------------
# Sanity check  (run: python src/market_data.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Fetching AAPL market data …")
    mkt = fetch_market_data("AAPL")
    print(f"  Spot       : {mkt.spot:.2f}")
    print(f"  Rate       : {mkt.rate:.3f}")
    print(f"  Div yield  : {mkt.div_yield:.4f}")
    print(f"  Expiries   : {list(mkt.option_chain.keys())}")

    first_expiry = list(mkt.option_chain.keys())[0]
    calls = mkt.option_chain[first_expiry]["calls"]
    print(f"\nCalls for {first_expiry}:")
    print(calls[["strike", "lastPrice", "impliedVolatility"]].head(5))
