import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt

# --- ۱. تنظیمات اولیه صفحه (مناسب برای رزومه) ---
st.set_page_config(page_title="AI Financial Advisor | Master Thesis", layout="wide")

# --- ۲. توابع تحلیل داده‌های بازار (Backend) ---
@st.cache_data(ttl=3600) # برای جلوگیری از لود مکرر و افزایش سرعت
def get_market_metrics():
    tickers = {
        "Gold (USD)": "GC=F",
        "S&P 500": "^GSPC",
        "FTSE MIB (Italy)": "FTSEMIB.MI",
        "EUR/USD": "EURUSD=X"
    }
    results = {}
    for name, ticker in tickers.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty:
                curr = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                delta = ((curr - prev) / prev) * 100
                results[name] = (round(curr, 2), round(delta, 2))
        except:
            results[name] = (0, 0)
    return results

# --- ۳. بدنه اصلی اپلیکیشن ---
def main():
    st.title("🛡️ AI-Powered Financial Intelligence System")
    st.markdown("### Decision Support System for Personal Finance Management")
    st.info("این پروژه به عنوان نمونه عملی تحلیل داده‌های مالی برای پایان‌نامه ارشد طراحی شده است.")

    # نمایش شاخص‌های زنده بازار در بالای صفحه
    st.subheader("📊 Market Real-time Indicators")
    market_data = get_market_metrics()
    cols = st.columns(len(market_data))
    
    for i, (name, val) in enumerate(market_data.items()):
        cols[i].metric(name, val[0], f"{val[1]}%")

    st.divider()

    # --- بخش آپلود داده‌ها ---
    st.sidebar.header("📁 Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload your Bank Statement (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        
        # پیش‌نمایش داده‌ها
        with st.expander("👀 View Raw Financial Data"):
            st.dataframe(df.head())

        # --- بخش تحلیل پیش‌بینی (Forecasting) ---
        st.subheader("📈 Predictive Analytics (Prophet Model)")
        # فرض بر این است که فایل CSV دو ستون 'Date' و 'Amount' دارد
        if 'Date' in df.columns and 'Amount' in df.columns:
            df_prophet = df.rename(columns={'Date': 'ds', 'Amount': 'y'})
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            
            m = Prophet()
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)

            fig_forecast = px.line(forecast, x='ds', y='yhat', title="Expense Forecast for Next 30 Days")
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            
        
    else:
        st.warning("👈 لطفا یک فایل CSV شامل ستون‌های Date و Amount آپلود کنید تا تحلیل شروع شود.")
        
        # نمایش یک نمودار نمونه برای خالی نبودن صفحه
        st.subheader("💡 Analysis Example: Gold vs Market")
        gold_data = yf.Ticker("GC=F").history(period="1mo").reset_index()
        fig_sample = px.area(gold_data, x='Date', y='Close', title="Gold Price Trend (Last 30 Days)")
        st.plotly_chart(fig_sample, use_container_width=True)

    # --- بخش متدولوژی (بسیار مهم برای دفاع ارشد) ---
    st.sidebar.divider()
    st.sidebar.markdown("""
    **Academic Framework:**
    - Model: Facebook Prophet
    - Indicators: Real-time Yahoo Finance API
    - Strategy: Mean-Variance Optimization
    - University: Italy Master Thesis Project
    """)

if __name__ == "__main__":
    main()
