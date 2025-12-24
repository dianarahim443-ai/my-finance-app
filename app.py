# ==========================================================
# DIANA FINANCE AI — SOVEREIGN INSTITUTIONAL TERMINAL
# Cloud-Safe + Local-Heavy Architecture
# ==========================================================

# ---------- ENV FIX (MUST BE FIRST) ----------
import os
os.environ["PROPHET_BACKEND"] = "CMDSTANPY"

# ---------- CORE IMPORTS ----------
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import norm
import io
import warnings

warnings.filterwarnings("ignore")

# ---------- SAFE PROPHET LOADER ----------
PROPHET_AVAILABLE = True
try:
    from prophet import Prophet
    from prophet.plot import plot_components_plotly
except Exception as e:
    PROPHET_AVAILABLE = False
    PROPHET_ERROR = str(e)

# ==========================================================
# 1. APP CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Diana Finance AI | Sovereign Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. HIGH-END INSTITUTIONAL THEME
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)),
                url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
    background-size: cover;
}
.main .block-container {
    background: rgba(10, 10, 10, 0.97);
    border-radius: 40px;
    padding: 60px;
    border: 1px solid #222;
    box-shadow: 0 40px 120px rgba(0,0,0,1);
}
h1 { color: #FFD700 !important; font-weight: 900; font-size: 4rem !important; }
h2, h3 { color: #E0E0E0 !important; border-left: 5px solid #FFD700; padding-left: 15px; }
.stMetric {
    background: rgba(255,255,255,0.02);
    padding: 30px;
    border-radius: 25px;
    border-top: 5px solid #FFD700;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. QUANTITATIVE CORE ENGINE
# ==========================================================

class SovereignAnalytics:

    @staticmethod
    def standardize_data(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def compute_risk_ratios(returns):
        if returns.empty:
            return None

        rf_daily = 0.04 / 252
        mu = returns.mean()
        sigma = returns.std()

        sharpe = (mu - rf_daily) / sigma * np.sqrt(252) if sigma > 0 else 0
        downside = returns[returns < 0].std()
        sortino = (mu - rf_daily) / downside * np.sqrt(252) if downside > 0 else 0

        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        mdd = drawdown.min() * 100

        var_95 = norm.ppf(0.05, mu, sigma) * 100

        return {
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MDD": mdd,
            "VaR": var_95
        }

# ==========================================================
# 4. UI MODULES
# ==========================================================

def render_global_pulse():
    st.title("🏛️ Diana Sovereign Terminal")
    st.markdown("### *Institutional-Grade Multi-Asset Intelligence*")

    assets = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Gold": "GC=F",
        "Bitcoin": "BTC-USD",
        "10Y Yield": "^TNX"
    }

    cols = st.columns(len(assets))
    for i, (name, sym) in enumerate(assets.items()):
        try:
            df = yf.download(sym, period="2d", progress=False)
            df = SovereignAnalytics.standardize_data(df)
            price = df["Close"].iloc[-1]
            change = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
            cols[i].metric(name, f"{price:,.2f}", f"{change:+.2f}%")
        except:
            cols[i].metric(name, "N/A", "—")

    st.divider()

# ----------------------------------------------------------

def render_equity_intel():
    st.header("📈 Equity Intelligence Engine")

    c1, c2 = st.columns([3,1])
    ticker = c1.text_input("Ticker", "NVDA").upper()
    period = c2.selectbox("Lookback", ["1y","2y","5y","max"])

    if st.button("Run Institutional Analysis"):
        with st.spinner("Analyzing Capital Dynamics..."):
            data = yf.download(ticker, period=period, progress=False)
            data = SovereignAnalytics.standardize_data(data)

            prices = data["Close"].dropna()
            returns = prices.pct_change().dropna()

            metrics = SovereignAnalytics.compute_risk_ratios(returns)

            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
            k2.metric("Sortino", f"{metrics['Sortino']:.2f}")
            k3.metric("Max DD", f"{metrics['MDD']:.2f}%")
            k4.metric("VaR 95%", f"{metrics['VaR']:.2f}%")

            fig = px.line(prices, title=f"{ticker} Price Dynamics", template="plotly_dark")
            fig.update_traces(line_color="#FFD700", line_width=3)
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------

def render_ai_forecasting():
    st.header("🔮 Neural Forecasting Engine")

    if not PROPHET_AVAILABLE:
        st.error("Prophet is unavailable in this environment.")
        st.code(PROPHET_ERROR)
        st.info("Run locally with Python 3.10 + pinned NumPy to enable forecasting.")
        return

    asset = st.text_input("Forecast Asset", "BTC-USD")

    if st.button("Generate 90-Day Forecast"):
        with st.spinner("Training probabilistic neural model..."):
            raw = yf.download(asset, period="3y", progress=False).reset_index()
            df = pd.DataFrame({
                "ds": pd.to_datetime(raw["Date"]),
                "y": raw["Close"]
            }).dropna()

            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="Historical", line=dict(color="#00F2FF")))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="#FFD700", dash="dash")))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_lower"],
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.2)",
                line=dict(width=0),
                showlegend=False
            ))

            fig.update_layout(template="plotly_dark", title="Neural Price Projection")
            st.plotly_chart(fig, use_container_width=True)

            st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_advisor():
    st.header("💳 Behavioral Wealth Advisor")

    df = pd.DataFrame([
        {"Category": "Income", "Amount": 10500},
        {"Category": "Fixed", "Amount": -3000},
        {"Category": "Wealth", "Amount": -2800},
        {"Category": "Wants", "Amount": -900},
        {"Category": "Fixed", "Amount": -500},
        {"Category": "Wealth", "Amount": -600},
    ])

    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

    outflows = df[df["Amount"] < 0].copy()
    outflows["Abs"] = outflows["Amount"].abs()

    fig = px.pie(
        outflows,
        values="Abs",
        names="Category",
        hole=0.6,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 5. MASTER ROUTER
# ==========================================================

def main():
    render_global_pulse()

    st.sidebar.title("💎 Sovereign Control")
    nav = st.sidebar.radio(
        "Navigation",
        ["Equity Intelligence", "Neural Forecasting", "Wealth Advisor"]
    )

    if nav == "Equity Intelligence":
        render_equity_intel()
    elif nav == "Neural Forecasting":
        render_ai_forecasting()
    elif nav == "Wealth Advisor":
        render_wealth_advisor()

    st.sidebar.divider()
    st.sidebar.caption(f"Runtime Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()

