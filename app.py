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

# --- 2. CORE ENGINES (Calculations) ---
@st.cache_data(ttl=3600)
def get_global_metrics():
    """Fetches live market indices for the top header."""
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            if len(df) >= 2:
                price = float(df['Close'].iloc[-1])
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                data[name] = (price, change)
        except: data[name] = (0.0, 0.0)
    return data

def calculate_risk_metrics(equity_curve, strategy_returns):
    """Academic KPIs for portfolio management."""
    rf = 0.02 / 252 
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    return total_return, sharpe, max_dd

def run_monte_carlo(data, prediction_days=30, simulations=50):
    """Geometric Brownian Motion (GBM) simulation."""
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
    
    # 3.1. Live Header Metrics
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    # 3.2. Sidebar Navigation (Safe Method)
    st.sidebar.title("🔬 Navigation Menu")
    page = st.sidebar.selectbox("Select Research Module:", 
                                ["🏠 Home & Documentation", 
                                 "📈 Equity Intelligence", 
                                 "🔮 AI Prediction", 
                                 "💳 Personal Finance AI"])

    # --- PAGE 1: DOCUMENTATION ---
    if page == "🏠 Home & Documentation":
        st.header("📑 Quantitative Research Documentation")
        tab1, tab2, tab3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI vs Traditional"])
        
        with tab1:
            st.subheader("AI System Architecture")
            st.markdown("""
            **1. Prophet Engine:** Utilizes a decomposable time-series model (Trend, Seasonality, Holidays).
            """)
            
            st.markdown("""
            **2. Stochastic Risk (GBM):** Monte Carlo simulations utilize Geometric Brownian Motion:
            """)
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.info("S_t: Price, mu: Drift, sigma: Volatility, W_t: Wiener Process.")

        with tab2:
            st.subheader("Methodology & Parameters")
            st.markdown("""
            - **Portfolio Start:** $10,000 USD.
            - **Rebalancing:** Daily frequency based on SMA-20 Signals.
            - **Data Source:** Yahoo Finance Institutional API.
            """)

        with tab3:
            st.subheader("Innovation: AI vs Traditional")
            compare = {
                "Feature": ["Data Processing", "Trend Detection", "Risk Model", "Integration"],
                "Traditional": ["Manual Spreadsheets", "Linear/Static", "Variance-only", "Siloed"],
                "Diana AI": ["Automated API", "Non-Linear ML", "Stochastic GBM", "Holistic Portfolio"]
            }
            st.table(compare)

    # --- PAGE 2: EQUITY ANALYSIS ---
    elif page == "📈 Equity Intelligence":
        st.header("🔍 Backtesting & Alpha Generation")
        ticker = st.text_input("Enter Ticker (e.g. NVDA, AAPL):", "NVDA").upper()
        if st.button("Execute Quantitative Run"):
            with st.spinner("Analyzing Market Dynamics..."):
                stock_raw = yf.download(ticker, period="1y")['Close']
                if not stock_raw.empty:
                    data = stock_raw.squeeze()
                    # Momentum Strategy
                    signal = np.where(data > data.rolling(20).mean(), 1, 0)
                    returns = data.pct_change() * pd.Series(signal).shift(1).values
                    equity = 10000 * (1 + returns.fillna(0)).cumprod()
                    
                    ret, sharpe, dd = calculate_risk_metrics(equity, returns.fillna(0))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Strategy Return", f"{ret:.2f}%")
                    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    c3.metric("Max Drawdown", f"{dd:.2f}%")

                    fig_eq = px.line(equity, title=f"Equity Growth: {ticker}", template="plotly_dark")
                    fig_eq.update_traces(line_color='#FFD700')
                    st.plotly_chart(fig_eq, use_container_width=True)

                    st.subheader("Stochastic Risk Simulation (Monte Carlo)")
                    sims = run_monte_carlo(data)
                    fig_mc = px.line(sims, template="plotly_dark", title="Potential 30-Day Paths")
                    fig_mc.update_layout(showlegend=False)
                    
                    st.plotly_chart(fig_mc, use_container_width=True)
                else: st.error("Ticker not found.")

    # --- PAGE 3: AI FORECAST ---
    elif page == "🔮 AI Prediction":
        st.header("🔮 Time-Series Forecasting")
        symbol = st.text_input("Asset to Forecast (e.g. BTC-USD):", "BTC-USD").upper()
        if st.button("Train AI Model"):
            with st.spinner("Building Prophet Model..."):
                df = yf.download(symbol, period="2y").reset_index()
                if not df.empty:
                    df_p = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                    df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                    m = Prophet(daily_seasonality=True).fit(df_p)
                    forecast = m.predict(m.make_future_dataframe(periods=30))
                    
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical"))
                    fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Forecast", line=dict(dash='dash')))
                    fig_f.update_layout(template="plotly_dark", title=f"30-Day Outlook: {symbol}")
                    st.plotly_chart(fig_f, use_container_width=True)

    # --- PAGE 4: PERSONAL FINANCE ---
    elif page == "💳 Personal Finance AI":
        st.header("💳 Intelligent Wealth Management")
        uploaded = st.file_uploader("Upload CSV (Description, Amount)", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'Description' in df.columns and 'Amount' in df.columns:
                def categorize(d):
                    d = str(d).lower()
                    if any(x in d for x in ['shop', 'amazon', 'uber']): return 'Discretionary'
                    if any(x in d for x in ['rent', 'bill', 'electric']): return 'Obligations'
                    return 'Lifestyle'
                df['Category'] = df['Description'].apply(categorize)
                df['Amount'] = pd.to_numeric(df['Amount']).abs()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.pie(df, values='Amount', names='Category', hole=0.5, template="plotly_dark"))
                with c2:
                    st.subheader("AI Behavioral Insight")
                    total = df['Amount'].sum()
                    dis_pct = (df[df['Category'] == 'Discretionary']['Amount'].sum() / total) * 100 if total > 0 else 0
                    if dis_pct > 25:
                        st.warning(f"Optimization Needed: Discretionary spending at {dis_pct:.1f}%. Suggest reallocating to Assets.")
                    else: st.success("Cash flow is institutionally optimized.")

    st.sidebar.divider()
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
