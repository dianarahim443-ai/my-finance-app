import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# --- 2. CORE QUANTITATIVE FUNCTIONS ---

def get_ai_reasoning(ticker, combined_df):
    """Generates professional quantitative insights for the defense committee."""
    try:
        latest_price = float(combined_df['Stock'].iloc[-1])
        sma_20 = float(combined_df['Stock'].rolling(20).mean().iloc[-1])
        volatility = combined_df['Stock'].pct_change().std() * np.sqrt(252)
        
        reasons = []
        if latest_price > sma_20:
            reasons.append(f"• **Trend Analysis:** Bullish momentum confirmed (Price > 20-day SMA).")
        else:
            reasons.append(f"• **Trend Analysis:** Bearish pressure detected (Price < 20-day SMA).")
        
        reasons.append(f"• **Risk Metric:** Annualized Volatility at {volatility:.1%}. Strategy: {'Risk-On' if volatility < 0.25 else 'Risk-Mitigation'}.")
        return reasons
    except: return ["Analyzing market dynamics..."]

def calculate_metrics(equity_curve, strategy_returns):
    """Calculates KPIs: Sharpe Ratio, Max Drawdown, and Alpha."""
    rf = 0.02 / 252 
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    return total_return, sharpe, max_dd

@st.cache_data(ttl=3600)
def get_global_metrics():
    """Live global market data fetcher."""
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
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Master's Thesis Project | Quantitative Finance & Behavioral Economics_")
    st.markdown("---")

    # Header Metrics Bar
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    # Sidebar Navigation & Methodology
    st.sidebar.title("🔬 Research Methodology")
    
    with st.sidebar.expander("Academic Framework", expanded=True):
        st.markdown("""
        **1. Quantitative Strategy:**
        - Momentum-based Crossover (Alpha Gen).
        
        **2. Risk Architecture:**
        - Monte Carlo GBM Simulations (VaR).
        
        **3. Forecasting Engine:**
        - Additive Regression Models (Prophet).
        
        **4. Behavioral Analytics:**
        - Heuristic Spending Categorization.
        """)
        
    page = st.sidebar.radio("Module Selector:", 
                            ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI", "Technical Documentation 📄"])

    # --- MODULE 1: STOCK ANALYSIS ---
    if page == "Global Stock 360°":
        st.header("🔍 Equity Intelligence & Backtesting")
        ticker = st.text_input("Enter Institutional Ticker (e.g., AAPL, TSLA, NVDA):", "NVDA").upper()
        if st.button("Run Quantitative Backtest"):
            with st.spinner("Executing Backtest..."):
                stock_raw = yf.download(ticker, period="1y")['Close']
                market_raw = yf.download("^GSPC", period="1y")['Close']
                
                if not stock_raw.empty:
                    stock_data = stock_raw.squeeze()
                    market_data = market_raw.squeeze()
                    combined = pd.concat([stock_data, market_data], axis=1).dropna()
                    combined.columns = ['Stock', 'Market']
                    
                    # Logic: SMA 20 Crossover
                    combined['Signal'] = np.where(combined['Stock'] > combined['Stock'].rolling(20).mean(), 1, 0)
                    combined['Strategy_Returns'] = combined['Stock'].pct_change() * combined['Signal'].shift(1)
                    ai_equity = 10000 * (1 + combined['Strategy_Returns'].fillna(0)).cumprod()
                    bh_equity = 10000 * (1 + combined['Stock'].pct_change().fillna(0)).cumprod()
                    
                    ai_ret, ai_sharpe, ai_dd = calculate_metrics(ai_equity, combined['Strategy_Returns'].fillna(0))
                    bh_ret, _, _ = calculate_metrics(bh_equity, combined['Stock'].pct_change().fillna(0))
                    
                    st.subheader("🤖 AI Decision Reasoning")
                    for line in get_ai_reasoning(ticker, combined): st.write(line)

                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AI Strategy Return", f"{ai_ret:.2f}%", f"{(ai_ret-bh_ret):.2f}% Alpha")
                    c2.metric("Sharpe Ratio", f"{ai_sharpe:.2f}")
                    c3.metric("Max Drawdown", f"{ai_dd:.2f}%")

                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(x=ai_equity.index, y=ai_equity, name='AI Strategy (Active)', line=dict(color='#FFD700')))
                    fig_perf.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name='Benchmark (B&H)', line=dict(color='gray', dash='dash')))
                    fig_perf.update_layout(template="plotly_dark", title="Equity Growth Comparison")
                    st.plotly_chart(fig_perf, use_container_width=True)

                    st.subheader("🎲 Monte Carlo Risk Forecasting")
                    sim_results = run_monte_carlo(combined['Stock'])
                    fig_mc = go.Figure()
                    for i in range(sim_results.columns.size):
                        fig_mc.add_trace(go.Scatter(y=sim_results[i], mode='lines', opacity=0.1, showlegend=False))
                    fig_mc.update_layout(template="plotly_dark", title="Stochastic Price Path Projection (30D)")
                    st.plotly_chart(fig_mc, use_container_width=True)
                else: st.error("Ticker not found.")

    # --- MODULE 2: PROPHET FORECAST ---
    elif page == "AI Wealth Prediction":
        st.header("🔮 Time-Series Forecasting (Prophet)")
        symbol = st.text_input("Enter Asset for AI Forecasting:", "BTC-USD").upper()
        if st.button("Generate Forecast"):
            with st.spinner("Training ML Model..."):
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
                    fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Trend', line=dict(color='cyan')))
                    fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual', mode='markers', marker=dict(size=2)))
                    fig_f.update_layout(template="plotly_dark", title=f"Predictive Analytics: {symbol}")
                    st.plotly_chart(fig_f, use_container_width=True)

    # --- MODULE 3: PERSONAL FINANCE ---
    elif page == "Personal Finance AI":
        st.header("💳 Intelligent Wealth Management")
        st.info("Upload CSV with 'Description' and 'Amount' columns.")
        uploaded = st.file_uploader("Upload Ledger", type="csv")
        if uploaded:
            try:
                df_u = pd.read_csv(uploaded)
                if 'Description' in df_u.columns and 'Amount' in df_u.columns:
                    def advanced_categorize(d):
                        d = str(d).lower()
                        if any(x in d for x in ['shop', 'amazon', 'zara']): return 'Discretionary'
                        if any(x in d for x in ['uber', 'gas', 'snapp']): return 'Operational'
                        if any(x in d for x in ['food', 'cafe', 'restaurant']): return 'Lifestyle'
                        if any(x in d for x in ['rent', 'bill', 'insurance']): return 'Fixed'
                        return 'Other'
                    
                    df_u['Category'] = df_u['Description'].apply(advanced_categorize)
                    df_u['Amount'] = pd.to_numeric(df_u['Amount']).abs()
                    
                    # Dashboard Metrics
                    total = df_u['Amount'].sum()
                    discretionary = df_u[df_u['Category'] == 'Discretionary']['Amount'].sum()
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_p = px.pie(df_u, values='Amount', names='Category', hole=0.5, 
                                      template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold)
                        st.plotly_chart(fig_p, use_container_width=True)
                    with c2:
                        st.subheader("Top Expenses")
                        st.table(df_u.sort_values(by='Amount', ascending=False).head(5)[['Description', 'Amount']])
                    
                    st.divider()
                    st.subheader("🤖 AI Financial Insight")
                    dis_pct = (discretionary / total) * 100 if total > 0 else 0
                    if dis_pct > 25:
                        st.warning(f"Optimization Opportunity: Discretionary spending at {dis_pct:.1f}%. Consider reallocating to High-Alpha assets.")
                    else: st.success("Financial Health: Spending is within institutional safety limits.")
            except Exception as e: st.error(f"Error: {e}")

    # --- MODULE 4: DOCUMENTATION ---
    elif page == "Technical Documentation 📄":
        st.header("📑 Quantitative & Academic Documentation")
        t1, t2, t3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI vs Traditional"])
        
        with t1:
            st.subheader("AI System Architecture")
            st.write("""
            - **Prophet Engine:** Uses a decomposable time-series model with three main components: trend, seasonality, and holidays.
            - **Stochastic Risk:** Monte Carlo simulations utilize Geometric Brownian Motion: $dS_t = \mu S_t dt + \sigma S_t dW_t$.
            """)
            

        with t2:
            st.subheader("Assumptions & Limitations")
            st.write("""
            - **Assumptions:** Zero slippage, 2% risk-free rate, daily rebalancing.
            - **Limitations:** Past performance is not indicative of future results; model does not account for flash crashes or extreme macro-geopolitical shifts.
            """)

        with t3:
            st.subheader("Comparative Advantage")
            compare = {
                "Metric": ["Data Processing", "Trend Detection", "Risk Model", "Integration"],
                "Traditional Methods": ["Manual Excel", "Linear/Static", "Variance-only", "Siloed"],
                "Diana Finance AI": ["Automated API", "Non-Linear ML", "Stochastic GBM", "Holistic Portfolio"]
            }
            st.table(compare)

    st.sidebar.divider()
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
