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
def calculate_metrics(equity_curve, strategy_returns):
    # Risk-free rate (European Central Bank / 10Y Bunds approx)
    rf = 0.02 / 252 
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    
    # Sharpe Ratio Calculation
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    
    # Max Drawdown
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
                price = df['Close'].iloc[-1]
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                data[name] = (price, change)
            else:
                data[name] = (0, 0)
        except: data[name] = (0, 0)
    return data

def run_monte_carlo(data, prediction_days=30, simulations=50):
    returns = data.pct_change().dropna()
    last_price = data.iloc[-1]
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

    # Global Market Metrics
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    st.sidebar.title("🔬 Research Methodology")with st.sidebar.expander("Academic Methodology"):
    st.write("""
    **Models used:**
    - Time-Series: Facebook Prophet (Additive Regressive Model)
    - Risk: Geometric Brownian Motion (GBM)
    - Strategy: Momentum-based SMA Crossover
    - Metrics: Log-returns for stationarity.
    """)
    page = st.sidebar.radio("Module Selector:", ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI"])

    if page == "Global Stock 360°":
        st.header("🔍 Equity Intelligence & Backtesting")
        ticker = st.text_input("Enter Ticker (e.g. NVDA, AAPL, TSLA):", "NVDA").upper()
        
        if st.button("Run Full Institutional Analysis"):
            with st.spinner("Fetching data and running simulations..."):
                # Data Retrieval
                stock_data = yf.download(ticker, period="1y")
                market_data = yf.download("^GSPC", period="1y")
                
                if not stock_data.empty:
                    # Align Data
                    combined = pd.concat([stock_data['Close'], market_data['Close']], axis=1).dropna()
                    combined.columns = ['Stock', 'Market']
                    
                    # 1. Strategy Logic (Moving Average Crossover)
                    # Academic Note: Using 20-day SMA as a signal for AI Agent
                    combined['Signal'] = np.where(combined['Stock'] > combined['Stock'].rolling(20).mean(), 1, 0)
                    combined['Strategy_Returns'] = combined['Stock'].pct_change() * combined['Signal'].shift(1)
                    
                    # 2. Performance Calculation
                    initial_cap = 10000
                    ai_equity = initial_cap * (1 + combined['Strategy_Returns'].fillna(0)).cumprod()
                    bh_equity = initial_cap * (1 + combined['Stock'].pct_change().fillna(0)).cumprod()

st.subheader("🤖 AI Decision Reasoning")
with st.expander("See why Diana issued this signal", expanded=True):
    explanation = get_ai_reasoning(ticker, combined)
    for line in explanation:
        st.write(line)
    
    st.caption(f"Analysis based on technical indicators and volatility regime for {ticker}.")
                    # Metrics calculation
                    ai_ret, ai_sharpe, ai_dd = calculate_metrics(ai_equity, combined['Strategy_Returns'].fillna(0))
                    bh_ret, _, bh_dd = calculate_metrics(bh_equity, combined['Stock'].pct_change().fillna(0))
                    
                    # 3. Display Metrics
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("AI Strategy Return", f"{ai_ret:.2f}%", f"{(ai_ret-bh_ret):.2f}% vs Market")
                    col2.metric("Sharpe Ratio (Risk Adj.)", f"{ai_sharpe:.2f}")
                    col3.metric("Max Drawdown", f"{ai_dd:.2f}%")

                    # 4. Interactive Plotting
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(x=ai_equity.index, y=ai_equity, name='Diana AI Strategy', line=dict(color='#FFD700', width=3)))
                    fig_perf.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name='Benchmark (Buy & Hold)', line=dict(color='gray', dash='dash')))
                    fig_perf.update_layout(title="Equity Curve: Strategy vs Passive Market", template="plotly_dark", hovermode="x unified")
                    st.plotly_chart(fig_perf, use_container_width=True)

                    # 5. Monte Carlo Simulation
                    st.divider()
                    st.subheader("🎲 Monte Carlo Risk Forecasting")
                    sim_results = run_monte_carlo(combined['Stock'])
                    
                    fig_mc = go.Figure()
                    for i in range(sim_results.columns.size):
                        fig_mc.add_trace(go.Scatter(y=sim_results[i], mode='lines', opacity=0.1, showlegend=False))
                    
                    expected_p = sim_results.iloc[-1].mean()
                    var_5 = np.percentile(sim_results.iloc[-1], 5)
                    
                    st.write(f"**Statistical Forecast (30 Days):** Expected Price: ${expected_p:.2f} | Value at Risk (VaR 95%): ${var_5:.2f}")
                    fig_mc.update_layout(title="Geometric Brownian Motion: 50 Possible Price Paths", template="plotly_dark")
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                else:
                    st.error("Ticker not found. Please check the symbol.")

    elif page == "AI Wealth Prediction":
        st.header("🔮 Time-Series Forecasting (Prophet)")
        symbol = st.text_input("Enter Asset for Forecast:", "BTC-USD").upper()
        if st.button("Generate Forecast"):
            df_raw = yf.download(symbol, period="2y").reset_index()
            df_p = df_raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            
            m = Prophet(daily_seasonality=True)
            m.fit(df_p)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Trend', line=dict(color='cyan')))
            fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name='Actual Price', mode='markers', marker=dict(size=2)))
            fig_f.update_layout(title=f"30-Day Predictive Trend for {symbol}", template="plotly_dark")
            st.plotly_chart(fig_f, use_container_width=True)

    elif page == "Personal Finance AI":
        st.header("💳 Financial Behavior Analysis")
        uploaded = st.file_uploader("Upload your transaction history (CSV)", type="csv")
        if uploaded:
            st.info("Module under construction: Integration with LLM for spending classification.")

    st.sidebar.divider()
    st.sidebar.info("📌 **Defense Tip:** Focus on the 'Sharpe Ratio' and 'Max Drawdown' when explaining the Backtesting results to the committee.")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
def get_ai_reasoning(ticker, combined_df):
    latest_price = combined_df['Stock'].iloc[-1]
    sma_20 = combined_df['Stock'].rolling(20).mean().iloc[-1]
    volatility = combined_df['Stock'].pct_change().std() * np.sqrt(252)
    
    reasons = []
    if latest_price > sma_20:
        reasons.append(f"• Price is above 20-day SMA (${sma_20:.2f}), indicating a **Bullish Trend**.")
    else:
        reasons.append(f"• Price is below 20-day SMA (${sma_20:.2f}), suggesting **Bearish Momentum**.")
        
    if volatility > 0.30:
        reasons.append("• High Annualized Volatility detected. Model suggests **Caution** (High Risk).")
    else:
        reasons.append("• Volatility is within stable limits, supporting a **Steady Accumulation** strategy.")
        
    return reasons
if __name__ == "__main__":
    main()
