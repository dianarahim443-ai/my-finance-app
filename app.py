import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np

# --- 1. تنظیمات اولیه (Academic Standard) ---
st.set_page_config(page_title="Strategic Finance AI", layout="wide")

# --- 2. توابع کمکی (برای جلوگیری از Rate Limit) ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1y")
        return data if not data.empty else None
    except:
        return None

def categorize_expenses(desc):
    desc = str(desc).lower()
    if any(w in desc for w in ['amazon', 'shop', 'mall']): return 'Shopping'
    if any(w in desc for w in ['uber', 'bolt', 'train', 'gas']): return 'Transport'
    if any(w in desc for w in ['food', 'cafe', 'restaurant']): return 'Dining'
    return 'Fixed Costs / Others'

# --- 3. بدنه اصلی اپلیکیشن ---
def main():
    st.title("🌐 Global Financial Intelligence Platform")
    st.markdown("_Advanced Quant Analysis for MSc Finance Research_")

    # سایدبار برای جابجایی بین بخش‌ها
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to:", ["Market Pulse", "Personal Finance AI", "Asset Intelligence", "Forecasting"])

    # بخش اول: نمای بازار جهانی
    if choice == "Market Pulse":
        st.header("🌍 Global Market Indicators")
        tickers = {"S&P 500": "^GSPC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
        cols = st.columns(4)
        for i, (name, t) in enumerate(tickers.items()):
            df = get_market_data(t)
            if df is not None:
                price = df['Close'].iloc[-1]
                delta = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                cols[i].metric(name, f"{price:,.2f}", f"{delta:.2f}%")

    # بخش دوم: هوش مصنوعی مدیریت مخارج (NLP ساده)
    elif choice == "Personal Finance AI":
        st.header("💳 Intelligent Expense Categorizer")
        uploaded_file = st.file_uploader("Upload Bank Statement (CSV)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'Description' in df.columns and 'Amount' in df.columns:
                df['Category'] = df['Description'].apply(categorize_expenses)
                st.plotly_chart(px.pie(df, values='Amount', names='Category', hole=0.5))
            else:
                st.error("CSV must have 'Description' and 'Amount' columns.")

    # بخش سوم: تحلیل کمی سهام (بسیار مهم برای رزومه)
    elif choice == "Asset Intelligence":
        st.header("🔍 Quantitative Security Analysis")
        ticker = st.text_input("Enter Ticker (e.g. NVDA, AAPL):", "NVDA").upper()
        if st.button("Analyze"):
            df = get_market_data(ticker)
            if df is not None:
                returns = df['Close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252) # ریسک سالانه
                var_95 = np.percentile(returns, 5) # ارزش در معرض ریسک
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                c2.metric("Annual Volatility", f"{vol:.2%}")
                c3.metric("Daily VaR (95%)", f"{var_95:.2%}")
                
                st.plotly_chart(px.line(df, y='Close', title=f"{ticker} 1Y Trend"))
                

    # بخش چهارم: پیش‌بینی سری زمانی
    elif choice == "Forecasting":
        st.header("🔮 AI Wealth Projection")
        # دمو برای نمایش قدرت مدل Prophet
        dates = pd.date_range(start='2024-01-01', periods=100)
        values = np.random.normal(100, 10, 100).cumsum()
        df_f = pd.DataFrame({'ds': dates, 'y': values})
        
        m = Prophet().fit(df_f)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        st.plotly_chart(px.line(forecast, x='ds', y='yhat', title="30-Day Predictive Wealth Trend"))
        

    st.sidebar.divider()
    st.sidebar.info("MSc Finance Project | Built with Streamlit & Meta Prophet")

if __name__ == "__main__":
    main()
