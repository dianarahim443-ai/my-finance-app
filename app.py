import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. تنظیمات سیستمی ---
st.set_page_config(page_title="AI Finance & Research Platform", layout="wide")

# --- 2. توابع محاسباتی و منطق مالی ---

def calculate_asset_pricing_models(stock_returns, market_returns):
    rf = 0.02 / 252 
    combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
    combined.columns = ['stock', 'market']
    excess_stock = combined['stock'] - rf
    excess_market = combined['market'] - rf
    beta = np.cov(excess_stock, excess_market)[0, 1] / np.var(excess_market)
    capm_expected = rf + beta * (excess_market.mean())
    return beta, capm_expected

@st.cache_data(ttl=3600)
def get_global_metrics():
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            price = df['Close'].iloc[-1]
            change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def run_backtest(data, signals, initial_capital=10000):
    positions = signals.shift(1).fillna(0)
    returns = data.pct_change()
    strategy_returns = returns * positions
    equity_curve = initial_capital * (1 + strategy_returns).cumprod().fillna(initial_capital)
    return equity_curve

def display_backtest_results(equity_curve, benchmark_curve):
    st.subheader("📈 Backtesting & Performance Analysis")
    col1, col2, col3 = st.columns(3)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    benchmark_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100
    alpha = total_return - benchmark_return
    col1.metric("AI Strategy Return", f"{total_return:.2f}%")
    col2.metric("Market Return", f"{benchmark_return:.2f}%")
    col3.metric("Alpha", f"{alpha:.2f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name='Diana AI Strategy', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name='Market', line=dict(color='gray', dash='dash')))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def run_monte_carlo(data, prediction_days=30, simulations=50):
    returns = data.pct_change()
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

# --- 3. بدنه اصلی برنامه ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Global Market AI")
    st.markdown("---")

    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    st.sidebar.title("🔬 Research Methodology")
    page = st.sidebar.radio("Go to Module:", ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI"])

    if page == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        ticker = st.text_input("Enter Ticker:", "NVDA").upper()
        
        if st.button("Run Full Analysis"):
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            
            if not df.empty:
                st.subheader(f"Results for {ticker}")
                st.line_chart(df['Close'])

                # --- بخش 1: بک‌تست پیشرفته و مقایسه با بازار ---
                st.divider()
                st.header("🔬 Institutional Performance Attribution")
                
                with st.spinner("Calculating Academic Benchmarks..."):
                    market_ticker = "^GSPC" 
                    mkt_data = yf.download(market_ticker, period="1y")['Close']
                    combined_df = pd.concat([df['Close'], mkt_data], axis=1).dropna()
                    combined_df.columns = ['Stock', 'Market']
                    stock_rets = combined
