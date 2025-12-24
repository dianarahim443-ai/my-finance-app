import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. CLEAN START CONFIGURATION ---
# Setting this first prevents many layout-related redirect issues
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# --- 2. CORE QUANTITATIVE FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_global_metrics():
    """Live global market data fetcher for the header bar."""
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            if len(df) >= 2:
                price = float(df['Close'].iloc[-1])
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                data[name] = (price, change)
            else: data[name] = (0, 0)
        except: data[name] = (0, 0)
    return data

def run_monte_carlo(data, prediction_days=30, simulations=50):
    """Stochastic price path modeling using Geometric Brownian Motion (GBM)."""
    returns = data.pct_change().dropna()
    last_price = float(data.iloc[-1])
    daily_vol = returns.std()
    avg_daily_ret = returns.mean()
    simulation_df = pd.DataFrame()
    for i in range(simulations):
        prices = [last_price]
        for d in range(prediction_days):
            next_price = prices[-1] * np.exp(avg_daily_ret + daily_vol * np.random.normal())
            prices.append(next_price)
        simulation_df[i] = prices
    return simulation_df

# --- 3. MAIN APPLICATION INTERFACE ---

def main():
    # Header Section
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Master's Thesis Project | Quantitative Finance & Behavioral Economics_")
    
    # Global Live Metrics
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.markdown("---")

    # --- SIDEBAR NAVIGATION (Anti-Redirect Loop Method) ---
    # We use a simple selectbox. This avoids changing the URL with '#' or '?' 
    st.sidebar.title("🔬 Research Methodology")
    
    page = st.sidebar.selectbox(
        "Module Selector:", 
        ["🏠 Home & Documentation", 
         "📈 Equity Intelligence", 
         "🔮 AI Prediction", 
         "💳 Personal Finance AI"],
        index=0,
        key="main_nav" # Unique key to prevent state loss
    )

    # --- MODULE 1: HOME & DOCUMENTATION ---
    if page == "🏠 Home & Documentation":
        st.header("📑 Quantitative Research Documentation")
        tab1, tab2, tab3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI vs Traditional"])
        
        with tab1:
            st.subheader("AI System Architecture")
            st.markdown("""
            **1. Prophet Engine:**
            Utilizes a **decomposable time-series model** to analyze: **Trend**, **Seasonality**, and **Holidays**.
            """)
            
            st.markdown("""
            **2. Stochastic Risk Modeling:**
            Implemented via **Monte Carlo methods** based on the **Geometric Brownian Motion (GBM)** framework:
            """)
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.info("Parameters: $S_t$ = Asset Price, $\mu$ = Drift, $\sigma$ = Volatility, $W_t$ = Wiener Process.")

        with tab2:
            st.subheader("Backtest Methodology")
            st.write("""
            - **Universe:** Global Equities (via Yahoo Finance).
            - **Initial Capital:** $10,000 USD.
            - **Execution:** Zero slippage simulation based on daily closing prices.
            """)
            
        with tab3:
            st.subheader("Innovation: AI vs. Traditional Analysis")
            compare_data = {
                "Feature": ["Data Processing", "Trend Detection", "Risk Model", "Integration"],
                "Traditional": ["Manual Spreadsheets", "Linear/Static", "Variance-only", "Siloed"],
                "Diana AI": ["Automated API", "Non-Linear ML", "Stochastic GBM", "Holistic View"]
            }
            st.table(compare_data)

    # --- MODULE 2: EQUITY INTELLIGENCE ---
    elif page == "📈 Equity Intelligence":
        st.header("🔍 Backtesting & Alpha Generation")
        ticker = st.text_input("Enter Ticker:", "NVDA").upper()
        if st.button("Run Quantitative Analysis"):
            with st.spinner("Processing Market Data..."):
                stock_raw = yf.download(ticker, period="1y")['Close']
                if not stock_raw.empty:
                    stock_data = stock_raw.squeeze()
                    # Momentum Strategy (SMA 20)
                    signal = np.where(stock_data > stock_data.rolling(20).mean(), 1, 0)
                    returns = stock_data.pct_change() * pd.Series(signal).shift(1).values
                    ai_equity = 10000 * (1 + returns.fillna(0)).cumprod()
                    
                    st.subheader("Strategy Performance")
                    fig_perf = px.line(ai_equity, title="Equity Growth ($10,000 Initial)", template="plotly_dark")
                    fig_perf.update_traces(line_color='#FFD700')
                    st.plotly_chart(fig_perf, use_container_width=True)

                    st.subheader("Risk Forecasting (Monte Carlo)")
                    sim_results = run_monte_carlo(stock_data)
                    fig_mc = px.line(sim_results, template="plotly_dark", title="Potential 30-Day Paths")
                    fig_mc.update_layout(showlegend=False)
                    
                    st.plotly_chart(fig_mc, use_container_width=True)

    # --- MODULE 3: AI PREDICTION ---
    elif page == "🔮 AI Prediction":
        st.header("🔮 Time-Series Forecasting")
        symbol = st.text_input("Enter Asset (e.g., BTC-USD):", "BTC-USD").upper()
        if st.button("Generate Forecast"):
            df_raw = yf.download(symbol, period="2y").reset_index()
            if not df_raw.empty:
                df_p = df_raw[['Date', 'Close']].copy()
                df_p.columns = ['ds', 'y']
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                m = Prophet(daily_seasonality=True); m.fit(df_p)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                fig_f = px.line(forecast, x='ds', y='yhat', title=f"30-Day Predictive Trend", template="plotly_dark")
                st.plotly_chart(fig_f, use_container_width=True)

    # --- MODULE 4: PERSONAL FINANCE AI ---
    elif page == "💳 Personal Finance AI":
        st.header("💳 Intelligent Wealth Management")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df_u = pd.read_csv(uploaded)
            if 'Description' in df_u.columns and 'Amount' in df_u.columns:
                df_u['Amount'] = pd.to_numeric(df_u['Amount']).abs()
                fig = px.pie(df_u, values='Amount', names='Description', hole=0.5, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
