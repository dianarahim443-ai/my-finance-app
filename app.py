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
    """محاسبه عملکرد استراتژی بر اساس سیگنال‌ها"""
    positions = signals.shift(1).fillna(0)
    returns = data.pct_change()
    strategy_returns = returns * positions
    equity_curve = initial_capital * (1 + strategy_returns).cumprod().fillna(initial_capital)
    return equity_curve

def display_backtest_results(equity_curve, benchmark_curve):
    """نمایش بصری نتایج بک‌تست"""
    st.subheader("📈 Backtesting & Performance Analysis")
    col1, col2, col3 = st.columns(3)
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    benchmark_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100
    alpha = total_return - benchmark_return
    
    col1.metric("AI Strategy Return", f"{total_return:.2f}%")
    col2.metric("Market Return", f"{benchmark_return:.2f}%")
    col3.metric("Alpha (Excess Return)", f"{alpha:.2f}%", delta_color="normal")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name='Diana AI Strategy', line=dict(color='gold', width=3)))
    fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name='Market (Buy & Hold)', line=dict(color='gray', dash='dash')))
    
    fig.update_layout(title="Equity Growth Over Time", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def categorize_expenses(description):
    desc = description.lower()
    if any(word in desc for word in ['amazon', 'shop', 'buy']): return 'Shopping'
    if any(word in desc for word in ['uber', 'gas', 'snapp', 'train']): return 'Transport'
    if any(word in desc for word in ['restaurant', 'food', 'cafe']): return 'Dining'
    return 'Others'

# --- 3. بدنه اصلی برنامه ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Global Market AI")
    st.markdown("---")

    # ردیف شاخص‌های زنده
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    # منوی ناوبری
    st.sidebar.title("🔬 Research Methodology")
    page = st.sidebar.radio("Go to Module:", ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI"])

    if page == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        ticker = st.text_input("Enter Ticker (e.g. NVDA, AAPL, RACE):", "NVDA").upper()
        
        if st.button("Run Full Analysis"):
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            
            if not df.empty:
                # تحلیل تکنیکال
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                c2.metric("Annual Volatility", f"{(df['Close'].pct_change().std() * np.sqrt(252)):.2%}")
                
                # نمایش نمودار قیمت
                st.line_chart(df['Close'])

                # --- بخش بک‌تست (اضافه شده) ---
                st.divider()
                # شبیه‌سازی سیگنال هوشمند (در پروژه واقعی از مدل ML شما می‌آید)
                df['Signal'] = np.where(df['Close'] > df['Close'].rolling(20).mean(), 1, 0)
                
                equity = run_backtest(df['Close'], df['Signal'])
                benchmark = 10000 * (1 + df['Close'].pct_change()).cumprod().fillna(10000)
                
                display_backtest_results(equity, benchmark)
            else:
                st.error("Data not found.")

    elif page == "AI Wealth Prediction":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Ticker to Forecast:", "BTC-USD").upper()
        if st.button("Generate AI Prediction"):
            raw = yf.download(symbol, period="2y").reset_index()
            df_p = raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None) # رفع خطای زمانی
            
            model = Prophet()
            model.fit(df_p)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction'))
            st.plotly_chart(fig_f
