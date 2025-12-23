import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. System Configuration ---
st.set_page_config(page_title="Diana Finance AI | Academic Research", layout="wide")

# --- 2. Financial Logic & Functions ---

def get_ai_reasoning(ticker, combined_df):
    try:
        latest_price = float(combined_df['Stock'].iloc[-1])
        sma_20 = float(combined_df['Stock'].rolling(20).mean().iloc[-1])
        volatility = combined_df['Stock'].pct_change().std() * np.sqrt(252)
        reasons = []
        if latest_price > sma_20:
            reasons.append(f"• Momentum: Bullish (Price > 20-day SMA).")
        else:
            reasons.append(f"• Momentum: Bearish (Price < 20-day SMA).")
        reasons.append(f"• Annualized Volatility: {volatility:.1%}.")
        return reasons
    except: return ["Processing..."]

def calculate_metrics(equity_curve, strategy_returns):
    rf = 0.02 / 252 
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    return total_return, sharpe, max_dd

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
            else: data[name] = (0, 0)
        except: data[name] = (0, 0)
    return data

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

# --- 3. Main Application ---
def main():
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Master's Thesis Quantitative Finance Module_")
    st.markdown("---")

    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    st.sidebar.title("🔬 Research Methodology")
    page = st.sidebar.radio("Module Selector:", ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI", "Technical Documentation 📄"])

    if page == "Global Stock 360°":
        st.header("🔍 Equity Intelligence & Backtesting")
        ticker = st.text_input("Enter Ticker:", "NVDA").upper()
        if st.button("Run Full Institutional Analysis"):
            with st.spinner("Processing..."):
                stock_raw = yf.download(ticker, period="1y")['Close']
                market_raw = yf.download("^GSPC", period="1y")['Close']
                if not stock_raw.empty:
                    stock_data = stock_raw.squeeze()
                    market_data = market_raw.squeeze()
                    combined = pd.concat([stock_data, market_data], axis=1).dropna()
                    combined.columns = ['Stock', 'Market']
                    combined['Signal'] = np.where(combined['Stock'] > combined['Stock'].rolling(20).mean(), 1, 0)
                    combined['Strategy_Returns'] = combined['Stock'].pct_change() * combined['Signal'].shift(1)
                    ai_equity = 10000 * (1 + combined['Strategy_Returns'].fillna(0)).cumprod()
                    bh_equity = 10000 * (1 + combined['Stock'].pct_change().fillna(0)).cumprod()
                    ai_ret, ai_sharpe, ai_dd = calculate_metrics(ai_equity, combined['Strategy_Returns'].fillna(0))
                    bh_ret, _, _ = calculate_metrics(bh_equity, combined['Stock'].pct_change().fillna(0))
                    
                    st.subheader("🤖 AI Decision Reasoning")
                    for line in get_ai_reasoning(ticker, combined): st.write(line)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("AI Return", f"{ai_ret:.2f}%", f"{(ai_ret-bh_ret):.2f}% Alpha")
                    c2.metric("Sharpe Ratio", f"{ai_sharpe:.2f}")
                    c3.metric("Max Drawdown", f"{ai_dd:.2f}%")

                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(x=ai_equity.index, y=ai_equity, name='AI Strategy', line=dict(color='#FFD700')))
                    fig_perf.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name='Market', line=dict(color='gray', dash='dash')))
                    fig_perf.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_perf, use_container_width=True)

                    st.subheader("🎲 Monte Carlo Risk Forecasting")
                    sim_results = run_monte_carlo(combined['Stock'])
                    fig_mc = go.Figure()
                    for i in range(sim_results.columns.size):
                        fig_mc.add_trace(go.Scatter(y=sim_results[i], mode='lines', opacity=0.1, showlegend=False))
                    st.plotly_chart(fig_mc, use_container_width=True)
                else: st.error("Ticker not found.")

    elif page == "AI Wealth Prediction":
        st.header("🔮 Time-Series Forecasting (Prophet)")
        symbol = st.text_input("Enter Asset:", "BTC-USD").upper()
        if st.button("Generate Forecast"):
            raw_f = yf.download(symbol, period="2y").reset_index()
            if not raw_f.empty:
                df_p = raw_f[['Date', 'Close']].copy()
                df_p.columns = ['ds', 'y']
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                df_p['y'] = pd.to_numeric(df_p['y'].squeeze(), errors='coerce')
                df_p = df_p.dropna()
                m = Prophet(daily_seasonality=True); m.fit(df_p)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Predicted Trend', line=dict(color='cyan')))
                fig_f.update_layout(template="plotly_dark")
                st.plotly_chart(fig_f, use_container_width=True)

    elif page == "Personal Finance AI":
        st.header("💳 Expense Intelligence")
        uploaded = st.file_uploader("Choose CSV", type="csv")
        if uploaded:
            df_u = pd.read_csv(uploaded)
            if 'Description' in df_u.columns and 'Amount' in df_u.columns:
                def categorize(d):
                    d = str(d).lower()
                    if any(x in d for x in ['shop', 'amazon']): return 'Shopping'
                    if any(x in d for x in ['uber', 'food']): return 'Lifestyle'
                    return 'Fixed/Other'
                df_u['Category'] = df_u['Description'].apply(categorize)
                df_u['Amount'] = pd.to_numeric(df_u['Amount']).abs()
                fig_p = px.pie(df_u, values='Amount', names='Category', hole=0.4, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_p, use_container_width=True)

    elif page == "Technical Documentation 📄":
        st.header("📑 Quantitative Research Documentation")
        
        tab1, tab2, tab3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI vs Traditional"])
        
        with tab1:
            st.subheader("1. AI Methodology")
            st.write("""
            The platform utilizes a **Multi-Layered AI Approach**:
            - **Trend Analysis:** Facebook Prophet (Additive Regressive Model) decomposes time-series into trend, seasonality, and holidays.
            - **Risk Modeling:** Geometric Brownian Motion (GBM) generates 50+ stochastic paths to calculate the potential range of price movement.
            - **Execution Logic:** Momentum SMA Crossover (20-day) acts as the primary signal generator for active alpha pursuit.
            """)
            
            
        with tab2:
            st.subheader("2. Backtesting Scenario & Assumptions")
            st.info("**Practical Case Study:** Testing NVDA (Nvidia) over 12 months.")
            st.write("""
            **Experimental Setup:**
            - **Initial Capital:** $10,000.
            - **Trading Frequency:** Daily (End-of-day).
            - **Cost Basis:** Zero commissions (Institutional assumption).
            - **Risk-Free Rate:** 2% (Standard T-Bill proxy).
            
            **Risk Disclosure:**
            - *Survival Bias:* The model assumes the asset exists throughout the period.
            - *Slippage:* Real-market execution prices may vary from the closing price.
            """)

        with tab3:
            st.subheader("3. Innovation: AI vs Traditional Analysis")
            comparison_data = {
                "Feature": ["Data Handling", "Pattern Recognition", "Risk Assessment", "Speed"],
                "Traditional (Fundamental)": ["Manual Spreadsheet", "Linear/Historical", "Static Ratios", "Low"],
                "Diana AI (Quantitative)": ["Automated API (yFinance)", "Non-Linear Regression", "Stochastic (Monte Carlo)", "High/Real-time"]
            }
            st.table(comparison_data)
            st.success("Innovative Edge: The integration of behavioral personal finance with macro-equity forecasting provides a holistic 360° wealth management perspective.")

    st.sidebar.divider()
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
