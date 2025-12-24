import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from scipy.stats import norm

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Sovereign Grand-Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PREMIUM THEME
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
h1 { color: #FFD700 !important; font-weight: 900; font-size: 4rem !important; }
h2, h3 { color: #E0E0E0 !important; border-left: 5px solid #FFD700; padding-left: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. ANALYTICS CORE
# ==========================================
class SovereignAnalytics:
    @staticmethod
    def clean(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def risk_metrics(returns):
        rf = 0.04 / 252
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu - rf) / sigma * np.sqrt(252) if sigma != 0 else 0
        cum = (1 + returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        var95 = norm.ppf(0.05, mu, sigma) * 100
        return sharpe, dd.min() * 100, var95

# ==========================================
# 4. GLOBAL PULSE
# ==========================================
def render_global_pulse():
    st.title("🏛️ Diana Sovereign")
    st.markdown("### *Institutional Market Intelligence Platform*")

    assets = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Gold": "GC=F",
        "Bitcoin": "BTC-USD"
    }

    cols = st.columns(len(assets))
    for i, (name, sym) in enumerate(assets.items()):
        try:
            d = SovereignAnalytics.clean(yf.download(sym, period="2d", progress=False))
            price = d["Close"].iloc[-1]
            chg = (d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100
            cols[i].metric(name, f"{price:,.2f}", f"{chg:+.2f}%")
        except:
            cols[i].metric(name, "N/A", "0.00%")
    st.divider()

# ==========================================
# 5. EQUITY INTEL
# ==========================================
def render_equity():
    st.header("📈 Equity Intelligence Terminal")
    ticker = st.text_input("Ticker:", "NVDA").upper()
    period = st.selectbox("Lookback:", ["1y", "2y", "5y", "max"])

    if st.button("Run Analysis"):
        data = SovereignAnalytics.clean(yf.download(ticker, period=period, progress=False))
        prices = data["Close"].dropna()
        returns = prices.pct_change().dropna()

        sharpe, mdd, var95 = SovereignAnalytics.risk_metrics(returns)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sharpe", f"{sharpe:.2f}")
        c2.metric("Max Drawdown", f"{mdd:.2f}%")
        c3.metric("VaR 95%", f"{var95:.2f}%")

        fig = px.line(prices, title=f"{ticker} Price Dynamics", template="plotly_dark")
        fig.update_traces(line_color="#FFD700")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. CLOUD-SAFE FORECAST
# ==========================================
def render_forecast():
    st.header("🔮 AI Forecast Engine (Cloud-Safe)")
    ticker = st.text_input("Forecast Asset:", "BTC-USD").upper()

    if st.button("Generate Projection"):
        data = yf.download(ticker, period="2y", progress=False)
        prices = data["Close"].dropna()
        returns = prices.pct_change().dropna()

        mu, sigma = returns.mean(), returns.std()
        last_price = prices.iloc[-1]

        paths, days = 100, 60
        fig = go.Figure()

        for _ in range(paths):
            path = [last_price]
            for _ in range(days):
                path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            fig.add_trace(go.Scatter(y=path, mode="lines", opacity=0.08))

        fig.update_layout(
            title="Monte Carlo Forward Projection (60D)",
            template="plotly_dark",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. MAIN
# ==========================================
def main():
    render_global_pulse()

    st.sidebar.title("💎 Sovereign Terminal")
    nav = st.sidebar.radio(
        "Navigation:",
        ["Equity Intelligence", "AI Forecast"]
    )

    if nav == "Equity Intelligence":
        render_equity()
    else:
        render_forecast()

    if "load_time" not in st.session_state:
        st.session_state.load_time = datetime.now()
    st.sidebar.caption(f"Session Start: {st.session_state.load_time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()

