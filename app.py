
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.stats import norm
import os

# ==========================================
# 1. CORE ARCHITECTURE & THEME ENGINE
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Sovereign Grand-Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# Prophet Safe Config (VERY IMPORTANT)
# ------------------------------------------
os.environ["PROPHET_BACKEND"] = "CMDSTANPY"

# ==========================================
# Custom High-End Professional CSS
# ==========================================
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
    background: rgba(10,10,10,0.98);
    border-radius: 40px;
    padding: 60px;
    border: 1px solid #222;
    box-shadow: 0 30px 100px rgba(0,0,0,1);
}
h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; }
h2, h3 { color: #E0E0E0 !important; font-weight: 700; border-left: 5px solid #FFD700; padding-left: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. QUANTITATIVE ANALYSIS CORE
# ==========================================
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
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu - rf_daily) / sigma * np.sqrt(252) if sigma != 0 else 0
        downside_std = returns[returns < 0].std()
        sortino = (mu - rf_daily) / downside_std * np.sqrt(252) if downside_std > 0 else 0

        cum = (1 + returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()

        var95 = norm.ppf(0.05, mu, sigma) * 100
        return {"Sharpe": sharpe, "Sortino": sortino, "MDD": dd.min() * 100, "VaR": var95}

# ==========================================
# 3. GUI MODULES
# ==========================================
def render_global_pulse():
    st.title("🏛️ Diana Sovereign")
    st.markdown("### *Professional Multi-Asset Research & Capital Management*")

    assets = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^IXIC",
        "Gold": "GC=F",
        "Bitcoin": "BTC-USD",
        "10Y Treasury": "^TNX"
    }

    cols = st.columns(len(assets))
    for i, (name, sym) in enumerate(assets.items()):
        try:
            d = SovereignAnalytics.standardize_data(yf.download(sym, period="2d", progress=False))
            price = d["Close"].iloc[-1]
            chg = (price / d["Close"].iloc[-2] - 1) * 100
            cols[i].metric(name, f"{price:,.2f}", f"{chg:+.2f}%")
        except:
            cols[i].metric(name, "N/A", "0.00%")
    st.divider()

def render_methodology():
    st.header("🔬 Strategic Research Methodology")
    tab1, tab2, _ = st.tabs(["Stochastic Modeling", "Neural Decomposition", "Risk Framework"])

    with tab1:
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")

    with tab2:
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")

def render_equity_intel():
    st.header("📈 Equity Intelligence Terminal")
    ticker = st.text_input("Institutional Ticker:", "NVDA").upper()
    period = st.selectbox("Lookback Horizon:", ["1Y", "2Y", "5Y", "Max"])

    if st.button("Initialize Deep Research Run"):
        raw = SovereignAnalytics.standardize_data(
            yf.download(ticker, period=period.lower(), progress=False)
        )
        prices = raw["Close"]
        returns = prices.pct_change().dropna()
        metrics = SovereignAnalytics.compute_risk_ratios(returns)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
        c2.metric("Sortino", f"{metrics['Sortino']:.2f}")
        c3.metric("MDD", f"{metrics['MDD']:.2f}%")
        c4.metric("VaR", f"{metrics['VaR']:.2f}%")

        fig = px.line(prices, template="plotly_dark")
        fig.update_traces(line_color="#FFD700")
        st.plotly_chart(fig, use_container_width=True)

def render_ai_forecasting():
    st.header("🔮 Neural Predictive Engine (V4)")
    asset = st.text_input("Forecast Asset Target:", "BTC-USD").upper()

    if st.button("Generate AI Forecast"):
        try:
            from prophet import Prophet
            from prophet.plot import plot_components_plotly
        except Exception as e:
            st.error("❌ Prophet is not available on this deployment.")
            st.code(str(e))
            return

        raw = yf.download(asset, period="3y", progress=False).reset_index()
        df_p = pd.DataFrame({
            "ds": pd.to_datetime(raw["Date"]),
            "y": raw["Close"]
        }).dropna()

        m = Prophet(daily_seasonality=True)
        m.fit(df_p)
        forecast = m.predict(m.make_future_dataframe(periods=90))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_p["ds"], y=df_p["y"], name="Historical"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Behavioral Wealth Audit")
    up_file = st.file_uploader("Upload CSV", type=["csv"])

    if up_file:
        df = pd.read_csv(up_file)
    else:
        df = pd.DataFrame([
            {"Description": "Salary", "Amount": 10500, "Category": "Income"},
            {"Description": "Rent", "Amount": -3000, "Category": "Fixed"},
            {"Description": "Equities", "Amount": -2800, "Category": "Wealth"},
        ])

    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# ==========================================
# 4. MASTER CONTROLLER
# ==========================================
def main():
    render_global_pulse()

    st.sidebar.title("💎 Sovereign Terminal")
    nav = st.sidebar.radio(
        "Navigation Perspectives:",
        ["Theoretical Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management Advisor"]
    )

    if nav == "Theoretical Framework":
        render_methodology()
    elif nav == "Equity Intelligence":
        render_equity_intel()
    elif nav == "Neural Forecasting":
        render_ai_forecasting()
    else:
        render_wealth_advisor()

    if "session_time" not in st.session_state:
        st.session_state.session_time = datetime.now()

    st.sidebar.caption(
        f"Last Terminal Sync: {st.session_state.session_time.strftime('%H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
      
