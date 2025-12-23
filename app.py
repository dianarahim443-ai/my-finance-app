import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta

# --- 1. تنظیمات اولیه سیستم ---
st.set_page_config(page_title="Global Finance AI | Advanced Analytics", layout="wide")

# --- 2. موتور بهینه‌شده دریافت داده (برای جلوگیری از Rate Limit) ---
@st.cache_data(ttl=3600)
def get_advanced_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # دریافت دیتای یک ساله برای تحلیل دقیق‌تر
        df = stock.history(period="1y")
        return df if not df.empty else None
    except:
        return None

# --- 3. تابع دسته‌بندی مخارج ---
def categorize_expenses(description):
    description = str(description).lower()
    if any(word in description for word in ['amazon', 'shop', 'mall', 'buy']): return 'Shopping'
    if any(word in description for word in ['uber', 'gas', 'snapp', 'train', 'flight']): return 'Transport'
    if any(word in description for word in ['restaurant', 'food', 'cafe', 'pizza']): return 'Dining'
    return 'Other Expenses'

# --- 4. رابط کاربری اصلی ---
def main():
    st.title("🏛️ Professional Financial Intelligence Platform")
    
    # منوی ناوبری در سایدبار
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.radio("Go to:", ["Market Pulse", "Asset Intelligence", "Personal Finance AI", "Wealth Projection"])

    # --- بخش ۱: نمای کلی بازار (Global Pulse) ---
    if page == "Market Pulse":
        st.header("🌍 Global Market Performance")
        tickers = {"S&P 500": "^GSPC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
        cols = st.columns(4)
        for i, (name, t) in enumerate(tickers.items()):
            df = get_advanced_stock_data(t)
            if df is not None:
                price = df['Close'].iloc[-1]
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                cols[i].metric(name, f"{price:,.2f}", f"{change:.2f}%")

    # --- بخش ۲: تحلیل عمیق سهام (Asset Intelligence) - بخش درخواستی شما ---
    elif page == "Asset Intelligence":
        st.header("📊 Comprehensive Asset Analysis")
        ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL, TSLA, BTC-USD):", "NVDA").upper()
        
        if st.button("Generate Full Audit"):
            with st.spinner("Analyzing Market Data..."):
                df = get_advanced_stock_data(ticker)
                if df is not None:
                    # الف) محاسبات بازده و ریسک
                    returns = df['Close'].pct_change().dropna()
                    last_price = df['Close'].iloc[-1]
                    ann_volatility = returns.std() * np.sqrt(252) # نوسان سالانه
                    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) # شاخص شارپ
                    
                    # ب) نمایش کارت‌های آماری
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Price", f"${last_price:,.2f}")
                    c2.metric("Annual Risk (Volatility)", f"{ann_volatility:.2%}")
                    c3.metric("Sharpe Ratio (Efficiency)", f"{sharpe_ratio:.2f}")

                    # ج) نمودار قیمتی پیشرفته (Candlestick)
                    st.subheader("Price Action Analysis")
                    fig_candle = go.Figure(data=[go.Candlestick(x=df.index,
                                    open=df['Open'], high=df['High'],
                                    low=df['Low'], close=df['Close'])])
                    fig_candle.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_candle, use_container_width=True)
                    

                    # د) تحلیل توزیع بازدهی و ریسک سقوط
                    st.subheader("Risk Distribution (Statistical Profile)")
                    fig_dist = px.histogram(returns, nbins=50, marginal="box", title="Daily Returns Distribution")
                    st.plotly_chart(fig_dist, use_container_width=True)
                    st.info("💡 Tip: A wider distribution indicates higher uncertainty and market risk.")

                else:
                    st.error("Invalid ticker or connection issue. Please try again.")

    # --- بخش ۳: تحلیل مخارج شخصی ---
    elif page == "Personal Finance AI":
        st.header("💳 AI Expense Categorization")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            if 'Description' in data.columns and 'Amount' in data.columns:
                data['Category'] = data['Description'].apply(categorize_expenses)
                st.plotly_chart(px.pie(data, values='Amount', names='Category', hole=0.4))

    # --- بخش ۴: پیش‌بینی با هوش مصنوعی (Prophet) ---
    elif page == "Wealth Projection":
        st.header("🔮 AI Time-Series Forecasting")
        # ایجاد داده فرضی برای نمایش قابلیت مدل
        dates = pd.date_range(start=datetime.now()-timedelta(days=100), periods=100)
        y = np.random.normal(100, 10, 100).cumsum()
        df_p = pd.DataFrame({'ds': dates, 'y': y})
        
        m = Prophet().fit(df_p)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        st.plotly_chart(px.line(forecast, x='ds', y='yhat', title="30-Day Predictive Projection"))
        

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("MSc Finance Research Framework | v3.0 Global")

if __name__ == "__main__":
    main()
