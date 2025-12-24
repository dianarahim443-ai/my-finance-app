import os
os.environ["PROPHET_BACKEND"] = "CMDSTANPY"
from prophet import Prophet
from prophet.plot import plot_components_plotly

# =========================================================
# DIANA FINANCE AI — SOVEREIGN GRAND-PRO EDITION
# Build: v12.0.4-Magnum | Cloud-Safe + Local-Pro
# =========================================================

import os
import io
import math
import time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
from scipy.stats import norm, skew, kurtosis

# ---------------------------------------------------------
# ENVIRONMENT SAFETY
# ---------------------------------------------------------
os.environ["PROPHET_BACKEND"] = "CMDSTANPY"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Diana Finance AI | Sovereign Grand-Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# GLOBAL STATE INIT
# ---------------------------------------------------------
if "boot_time" not in st.session_state:
    st.session_state.boot_time = datetime.now()

if "nav" not in st.session_state:
    st.session_state.nav = "Global Pulse"

# ---------------------------------------------------------
# HIGH-END THEME ENGINE
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        linear-gradient(rgba(0,0,0,0.94), rgba(0,0,0,0.94)),
        url("https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070");
    background-size: cover;
}

.main .block-container {
    background: rgba(10,10,10,0.98);
    border-radius: 40px;
    padding: 60px;
    border: 1px solid #222;
    box-shadow: 0 40px 120px rgba(0,0,0,1);
}

h1 {
    color: #FFD700 !important;
    font-size: 4.5rem !important;
    font-weight: 900;
    letter-spacing: -3px;
}

h2, h3 {
    color: #E6E6E6 !important;
    border-left: 5px solid #FFD700;
    padding-left: 15px;
}

.stMetric {
    background: rgba(255,255,255,0.03);
    padding: 30px;
    border-radius: 25px;
    border-top: 5px solid #FFD700;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# ANALYTICS CORE — INSTITUTIONAL GRADE
# =========================================================
class SovereignAnalytics:

    @staticmethod
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()

    @staticmethod
    def returns(series: pd.Series) -> pd.Series:
        return series.pct_change().dropna()

    @staticmethod
    def drawdown(returns: pd.Series) -> pd.Series:
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak

    @staticmethod
    def risk_metrics(returns: pd.Series) -> dict:
        rf = 0.04 / 252
        mu = returns.mean()
        sigma = returns.std()

        sharpe = (mu - rf) / sigma * np.sqrt(252) if sigma != 0 else 0
        downside = returns[returns < 0].std()
        sortino = (mu - rf) / downside * np.sqrt(252) if downside else 0

        dd = SovereignAnalytics.drawdown(returns).min() * 100
        var95 = norm.ppf(0.05, mu, sigma) * 100
        cvar = returns[returns <= np.percentile(returns, 5)].mean() * 100

        return {
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown %": dd,
            "VaR 95% %": var95,
            "CVaR %": cvar,
            "Skew": skew(returns),
            "Kurtosis": kurtosis(returns)
        }

# =========================================================
# GLOBAL PULSE — REAL-TIME TERMINAL
# =========================================================
def render_global_pulse():
    st.title("🏛️ Diana Sovereign")
    st.markdown("### *Institutional Multi-Asset Intelligence Terminal*")

    assets = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^IXIC",
        "Gold": "GC=F",
        "Bitcoin": "BTC-USD",
        "10Y Treasury": "^TNX"
    }

    cols = st.columns(len(assets))
    for i, (name, ticker) in enumerate(assets.items()):
        try:
            df = yf.download(ticker, period="2d", progress=False)
            df = SovereignAnalytics.standardize(df)
            price = df["Close"].iloc[-1]
            chg = (price / df["Close"].iloc[-2] - 1) * 100
            cols[i].metric(name, f"{price:,.2f}", f"{chg:+.2f}%")
        except:
            cols[i].metric(name, "N/A", "—")

    st.divider()

# =========================================================
# METHODOLOGY MODULE
# =========================================================
def render_methodology():
    st.header("🔬 Institutional Research Methodology")

    t1, t2, t3 = st.tabs([
        "Stochastic Modeling",
        "Neural Decomposition",
        "Risk Architecture"
    ])

    with t1:
        st.subheader("Geometric Brownian Motion")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Used for Monte Carlo pricing, VaR and tail risk.")

    with t2:
        st.subheader("Prophet Additive Decomposition")
        st.latex(r"y(t)=g(t)+s(t)+h(t)+\epsilon_t")

    with t3:
        st.subheader("Risk Stack")
        st.write("""
        • Sharpe / Sortino  
        • Max Drawdown  
        • Parametric & Conditional VaR  
        • Distribution Shape (Skew / Kurtosis)
        """)

# =========================================================
# EQUITY INTELLIGENCE MODULE
# =========================================================
def render_equity_intel():
    st.header("📈 Equity Intelligence Terminal")

    c1, c2 = st.columns([3,1])
    with c1:
        ticker = st.text_input("Ticker:", "NVDA").upper()
    with c2:
        period = st.selectbox("Lookback:", ["1y","2y","5y","max"])

    if st.button("Run Institutional Analysis"):
        df = yf.download(ticker, period=period, progress=False)
        df = SovereignAnalytics.standardize(df)

        prices = df["Close"]
        rets = SovereignAnalytics.returns(prices)
        metrics = SovereignAnalytics.risk_metrics(rets)

        cols = st.columns(len(metrics))
        for i, (k,v) in enumerate(metrics.items()):
            cols[i].metric(k, f"{v:.2f}")

        fig = px.line(prices, template="plotly_dark")
        fig.update_traces(line_color="#FFD700", line_width=3)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# AI FORECASTING — PROPHET (LAZY SAFE)
# =========================================================
def render_ai_forecasting():
    st.header("🔮 Neural Predictive Engine")

    asset = st.text_input("Forecast Asset:", "BTC-USD").upper()

    if st.button("Generate Forecast"):
        try:
            from prophet import Prophet
            from prophet.plot import plot_components_plotly
        except Exception as e:
            st.error("Prophet not available in this environment.")
            st.code(str(e))
            return

        df = yf.download(asset, period="3y", progress=False).reset_index()
        df = df[["Date","Close"]].dropna()
        df.columns = ["ds","y"]

        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], name="History"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            fill=None,
            mode="lines",
            line_color="rgba(255,215,0,0.2)"
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            fill="tonexty",
            mode="lines",
            line_color="rgba(255,215,0,0.2)"
        ))

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)

# =========================================================
# WEALTH MANAGEMENT MODULE
# =========================================================
def render_wealth_advisor():
    st.header("💳 Behavioral Wealth Intelligence")

    file = st.file_uploader("Upload Transactions CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame([
            {"Description":"Salary","Amount":12000,"Category":"Income"},
            {"Description":"Rent","Amount":-3200,"Category":"Fixed"},
            {"Description":"Equities","Amount":-3500,"Category":"Wealth"},
            {"Description":"Crypto","Amount":-800,"Category":"Wealth"},
            {"Description":"Lifestyle","Amount":-1100,"Category":"Wants"},
        ])

    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

    out = df[df["Amount"] < 0].copy()
    out["Abs"] = out["Amount"].abs()

    if not out.empty:
        fig = px.pie(
            out,
            values="Abs",
            names="Category",
            hole=0.6,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MASTER CONTROLLER
# =========================================================
def main():
    render_global_pulse()

    st.sidebar.title("💎 Sovereign Terminal")
    nav = st.sidebar.radio(
        "Navigation",
        [
            "Global Pulse",
            "Theoretical Framework",
            "Equity Intelligence",
            "Neural Forecasting",
            "Wealth Management Advisor"
        ]
    )

    if nav == "Theoretical Framework":
        render_methodology()
    elif nav == "Equity Intelligence":
        render_equity_intel()
    elif nav == "Neural Forecasting":
        render_ai_forecasting()
    elif nav == "Wealth Management Advisor":
        render_wealth_advisor()

    st.sidebar.divider()
    st.sidebar.caption(
        f"Session Start: {st.session_state.boot_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.sidebar.markdown("**Engine Build:** `v12.0.4-Magnum`")

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
