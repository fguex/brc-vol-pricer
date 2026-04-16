"""
app.py  —  Streamlit front-end for the BRC pricer.
Run:  streamlit run src/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from market_data import fetch_market_data
from vol_surface import fit_smile, get_vol_for_strike, implied_vol
from monte_carlo import simulate_paths
from pricing import BRCParams, compare_flat_vs_skew, PriceResult


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    st.sidebar.header("🔵 BRC Parameters")
    ticker      = st.sidebar.text_input("Ticker", "AAPL")
    barrier_pct = st.sidebar.slider("Barrier %", 50, 85, 65) / 100.0
    coupon      = st.sidebar.slider("Annual Coupon %", 2, 20, 8) / 100.0
    T           = st.sidebar.selectbox("Maturity (years)", [0.5, 1.0, 1.5, 2.0], index=1)
    n_sims      = st.sidebar.select_slider("MC Paths", [10_000, 50_000, 100_000], value=50_000)
    use_bgk     = st.sidebar.checkbox("BGK correction", value=True)
    return dict(ticker=ticker, barrier_pct=barrier_pct, coupon=coupon,
                T=T, n_sims=n_sims, use_bgk=use_bgk)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_vol_smile(strikes, market_vols, smile_params, forward, barrier_strike) -> go.Figure:
    k_grid      = np.linspace(strikes.min() * 0.9, strikes.max() * 1.1, 200)
    fitted_vols = [get_vol_for_strike(k, forward, smile_params.maturity, 0.0, smile_params)
                   for k in k_grid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=market_vols * 100, mode="markers",
                             name="Market IV", marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=k_grid, y=np.array(fitted_vols) * 100, mode="lines",
                             name="Fitted smile"))
    fig.add_vline(x=barrier_strike, line_dash="dash", line_color="red",
                  annotation_text="Barrier", annotation_position="top right")
    fig.update_layout(xaxis_title="Strike", yaxis_title="Implied Vol (%)", template="plotly_dark")
    return fig


def plot_price_comparison(result_flat: PriceResult, result_skew: PriceResult, notional: float) -> go.Figure:
    labels = ["Flat vol", "Skew-adjusted"]
    prices = [result_flat.price, result_skew.price]
    errors = [1.96 * result_flat.std_err, 1.96 * result_skew.std_err]
    fig = go.Figure(go.Bar(
        x=labels, y=prices,
        error_y=dict(type="data", array=errors, visible=True),
        marker_color=["#58a6ff", "#f78166"],
    ))
    fig.add_hline(y=notional, line_dash="dot", line_color="#7ee787",
                  annotation_text="Par", annotation_position="bottom right")
    fig.update_layout(yaxis_title="Price", template="plotly_dark", showlegend=False)
    return fig


def plot_sample_paths(paths: np.ndarray, params: BRCParams, n_display: int = 50) -> go.Figure:
    idx = np.random.choice(len(paths), size=min(n_display, len(paths)), replace=False)
    t   = np.linspace(0, params.T, paths.shape[1])
    fig = go.Figure()
    for i in idx:
        fig.add_trace(go.Scatter(x=t, y=paths[i], mode="lines",
                                 line=dict(width=0.5, color="rgba(88,166,255,0.15)"),
                                 showlegend=False))
    fig.add_hline(y=params.S0 * params.barrier_pct, line_color="red", line_dash="dash",
                  annotation_text=f"Barrier {params.barrier_pct*100:.0f}%",
                  annotation_position="bottom right")
    fig.update_layout(xaxis_title="Time (years)", yaxis_title="Spot price", template="plotly_dark")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="BRC Pricer", layout="wide", page_icon="🔵")
    st.title("🔵 Barrier Reverse Convertible Pricer")
    st.caption("Monte Carlo · Volatility Skew · BGK Correction")

    params_dict = render_sidebar()

    if not st.button("▶ Run Pricer", type="primary"):
        return

    # --- Step 1: Market data ---
    with st.spinner(f"Fetching market data for {params_dict['ticker']} …"):
        try:
            mkt = fetch_market_data(params_dict["ticker"])
        except Exception as e:
            st.error(f"Market data fetch failed: {e}")
            return

    st.success(f"Spot: **{mkt.spot:.2f}** · Rate: **{mkt.rate:.2%}** · Div yield: **{mkt.div_yield:.2%}**")

    # --- Step 2: Vol surface ---
    with st.spinner("Fitting volatility smile …"):
        expiry  = list(mkt.option_chain.keys())[1]
        calls   = mkt.option_chain[expiry]["calls"]
        T_exp   = max((pd.Timestamp(expiry) - pd.Timestamp.now()).days / 365, 1/252)
        forward = mkt.spot * np.exp((mkt.rate - mkt.div_yield) * T_exp)

        calls   = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()
        mids    = ((calls["bid"] + calls["ask"]) / 2).values
        strikes = calls["strike"].values
        ivs     = np.array([implied_vol(p, mkt.spot, k, T_exp, mkt.rate, mkt.div_yield)
                            for p, k in zip(mids, strikes)])
        mask    = ~np.isnan(ivs) & (ivs > 0.01) & (ivs < 3.0)

        if mask.sum() < 3:
            st.error("Not enough valid implied vols to fit smile. Try a different ticker or expiry.")
            return

        smile          = fit_smile(strikes[mask], ivs[mask], forward, T_exp)
        sigma_flat     = smile.sigma_atm
        barrier_strike = mkt.spot * params_dict["barrier_pct"]
        sigma_skew     = get_vol_for_strike(barrier_strike, mkt.spot, T_exp,
                                            mkt.rate, smile, mkt.div_yield)
        sigma_skew     = min(sigma_skew, 2.0 * sigma_flat)

        st.info(f"σ_ATM = **{sigma_flat:.1%}** · σ_barrier = **{sigma_skew:.1%}** · "
                f"Skew premium = **{(sigma_skew - sigma_flat)*100:+.1f} vol pts**")

        with st.expander("🔍 Vol surface diagnostics"):
            st.write(f"**Expiry:** {expiry}  |  **T_exp:** {T_exp:.3f}y  |  **Forward:** {forward:.2f}")
            st.write(f"**Valid IV points:** {mask.sum()} / {len(strikes)}")
            st.write(f"**Smile params:** σ_atm={smile.sigma_atm:.3f}  α={smile.alpha:.3f}  β={smile.beta:.3f}")
            st.write(f"**Barrier strike:** {barrier_strike:.2f}  ({params_dict['barrier_pct']:.0%} of {mkt.spot:.2f})")
            if sigma_skew >= 2.0 * sigma_flat:
                st.warning("⚠️ σ_skew capped at 2×σ_ATM — smile extrapolation unreliable at barrier strike.")

    # --- Step 3: Monte Carlo ---
    with st.spinner(f"Running {params_dict['n_sims']:,} Monte Carlo paths …"):
        brc_params = BRCParams(
            S0=mkt.spot, barrier_pct=params_dict["barrier_pct"],
            coupon=params_dict["coupon"], T=params_dict["T"],
            r=mkt.rate, notional=1000.0,
        )
        paths  = simulate_paths(mkt.spot, mkt.rate, sigma_skew, params_dict["T"],
                                n_steps=252, n_sims=params_dict["n_sims"],
                                antithetic=True, q=mkt.div_yield)
        result = compare_flat_vs_skew(paths, brc_params, sigma_flat, sigma_skew)

    with st.expander("🔍 Monte Carlo diagnostics"):
        expected_fwd = mkt.spot * np.exp((mkt.rate - mkt.div_yield) * params_dict["T"])
        st.write(f"**Mean S_T:** {paths[:,-1].mean():.2f}  (expect ~{expected_fwd:.2f})")
        st.write(f"**Barrier hit %:** {result['flat'].barrier_hit_pct*100:.1f}%  (typical: 5–30%)")
        if result["flat"].price < 500:
            st.error("❌ Price too low — check vol surface diagnostics.")

    # --- Step 4: Results ---
    flat, skew = result["flat"], result["skew"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Flat vol price",  f"{flat.price:.2f}", delta=f"hit {flat.barrier_hit_pct*100:.1f}%")
    col2.metric("Skew vol price",  f"{skew.price:.2f}", delta=f"hit {skew.barrier_hit_pct*100:.1f}%")
    col3.metric("Skew premium",    f"{result['diff_bps']:+.1f} bps")

    st.subheader("Volatility smile")
    st.plotly_chart(plot_vol_smile(strikes[mask], ivs[mask], smile, forward, barrier_strike),
                    use_container_width=True)

    st.subheader("Price comparison — flat vs skew")
    st.plotly_chart(plot_price_comparison(flat, skew, brc_params.notional),
                    use_container_width=True)

    st.subheader("Sample GBM paths (50 shown)")
    st.plotly_chart(plot_sample_paths(paths, brc_params), use_container_width=True)


if __name__ == "__main__":
    main()
