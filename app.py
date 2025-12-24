import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. SETTINGS & ANTI-LOOP CONFIG ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# --- 2. CORE QUANTITATIVE FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_global_metrics():
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            if len(df) >= 2:
                price = float(df['Close'].iloc[-1])
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def calculate_metrics(equity_curve, strategy_returns):
    rf = 0.02 / 252 
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    return total_return, sharpe, max_dd

def run_monte_carlo(data, prediction_days=30, simulations=50):
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

# --- 3. APP INTERFACE ---

def main():
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Master's Thesis Project | Quantitative Finance & Behavioral Economics_")

    # Header Metrics
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    # SIDEBAR NAVIGATION (URL-Safe Method)
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.selectbox("Select Module:", 
                                ["🏠 Home & Documentation", 
                                 "📈 Equity Intelligence", 
                                 "🔮 AI Prediction", 
                                 "💳 Personal Finance AI"])

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
            - **Initial Capital:** $10,000 USD.
            - **Rebalancing:** Daily closing prices.
            - **Risk-Free Rate:** 2% (Sharpe proxy).
            - **Execution:** Zero slippage institutional simulation.
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
            with st.spinner("Processing..."):
                stock_raw = yf.download(ticker, period="1y")['Close']
                if not stock_raw.empty:
                    stock_data = stock_raw.squeeze()
                    # Strategy logic
                    signal = np.where(stock_data > stock_data.rolling(20).mean(), 1, 0)
                    returns = stock_data.pct_change() * pd.Series(signal).shift(1).values
                    ai_equity = 10000 * (1 + returns.fillna(0)).cumprod()
                    
                    ret, sharpe, dd = calculate_metrics(ai_equity, returns.fillna(0))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Return", f"{ret:.2f}%")
                    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    c3.metric("Max Drawdown", f"{dd:.2f}%")

                    st.plotly_chart(px.line(ai_equity, title="Equity Curve", template="plotly_dark"), use_container_width=True)
                    
                    st.subheader("Monte Carlo Risk Forecasting")
                    sims = run_monte_carlo(stock_data)
                    st.plotly_chart(px.line(sims, template="plotly_dark").update_layout(showlegend=False), use_container_width=True)

    # --- MODULE 3: AI PREDICTION ---
    elif page == "🔮 AI Prediction":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Asset:", "BTC-USD").upper()
        if st.button("Generate Forecast"):
            df = yf.download(symbol, period="2y").reset_index()
            if not df.empty:
                df_p = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                m = Prophet().fit(df_p)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                st.plotly_chart(px.line(forecast, x='ds', y='yhat', template="plotly_dark"), use_container_width=True)

    # --- MODULE 4: PERSONAL FINANCE AI ---
    elif page == "💳 Personal Finance AI":
        st.header("💳 Intelligent Wealth Management")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df_u = pd.read_csv(uploaded)
            if 'Description' in df_u.columns and 'Amount' in df_u.columns:
                def categorize(d):
                    d = str(d).lower()
                    if any(x in d for x in ['shop', 'amazon']): return 'Discretionary'
                    if any(x in d for x in ['rent', 'bill']): return 'Fixed Obligations'
                    return 'Lifestyle/Other'
                
                df_u['Category'] = df_u['Description'].apply(categorize)
                df_u['Amount'] = pd.to_numeric(df_u['Amount']).abs()
                total = df_u['Amount'].sum()
                dis_pct = (df_u[df_u['Category'] == 'Discretionary']['Amount'].sum() / total) * 100 if total > 0 else 0
                
                col1, col2 = st.columns(2)
                col1.plotly_chart(px.pie(df_u, values='Amount', names='Category', hole=0.5, template="plotly_dark"), use_container_width=True)
                col2.metric("Discretionary Spending", f"{dis_pct:.1f}%")
                
                if dis_pct > 25:
                    st.warning(f"Optimization Alert: High discretionary spending detected. Reallocate to Alpha-seeking assets.")
                else: st.success("Cash flow optimized.")

    st.sidebar.divider()
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
