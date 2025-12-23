import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. تنظیمات اولیه سیستم ---
st.set_page_config(page_title="QuantFinance AI | Research Platform", layout="wide")

# --- 2. توابع کمکی (Helper Functions) ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    data = yf.download(tickers, period="1y")['Close']
    return data

# --- 3. بدنه اصلی برنامه ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Quantitative Analysis")
    st.markdown("---")

    # منوی کناری برای ناوبری (بسیار مهم برای ساختار پایان‌نامه)
    st.sidebar.title("🔬 Methodology")
    menu = st.sidebar.radio("Select Analysis Module:", 
                           ["Market Intelligence", "Predictive Modeling", "Global Stock 360°"])

    # --- بخش اول: هوش بازار جهانی ---
    if menu == "Market Intelligence":
        st.header("🌍 Global Asset Correlation")
        tickers = ["^GSPC", "GC=F", "BTC-USD", "EURUSD=X"]
        df_market = get_market_data(tickers)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Correlation Matrix")
            corr = df_market.pct_change().dropna().corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        
        with col2:
            st.subheader("Risk-Return Profile")
            returns = df_market.pct_change().dropna()
            st.dataframe(returns.describe().T[['mean', 'std', 'min', 'max']])

    # --- بخش دوم: پیش‌بینی با هوش مصنوعی ---
    elif menu == "Predictive Modeling":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Enter Asset Ticker (e.g. NVDA):", "NVDA").upper()
        
        if st.button("Run AI Forecast"):
            df_raw = yf.download(symbol, period="5y").reset_index()
            df_prop = df_raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            m = Prophet(daily_seasonality=True)
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.1)'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.1)'))
            st.plotly_chart(fig, use_container_width=True)

    # --- بخش سوم: تحلیل ۳۶۰ درجه (بخش جدید که خطا داشت) ---
    elif menu == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        ticker_input = st.text_input("Enter Ticker (e.g., TSLA, AAPL, RACE):", "TSLA").upper()
        
        if ticker_input:
            stock = yf.Ticker(ticker_input)
            
            # نمایش متغیرهای کلیدی
            info = stock.info
            cols = st.columns(4)
            cols[0].metric("Price", f"${info.get('currentPrice', 'N/A')}")
            cols[1].metric("P/E Ratio", info.get('trailingPE', 'N/A'))
            cols[2].metric("Market Cap", f"{info.get('marketCap', 0):,}")
            cols[3].metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

            # نمودار تکنیکال با میانگین متحرک
            df_tech = stock.history(period="1y")
            df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
            
            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['Close'], name='Price'))
            fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA50'], name='MA50 Trend'))
            st.plotly_chart(fig_tech, use_container_width=True)
            

            # تحلیل سودآوری سالانه
            st.subheader("Annual Net Income (Financial Health)")
            try:
                income = stock.financials.loc['Net Income']
                st.bar_chart(income)
            except:
                st.warning("Financial statements not available for this ticker.")

    # فوتر مخصوص پایان‌نامه
    st.sidebar.divider()
    st.sidebar.caption("Project: AI-Driven Financial Analysis\nAcademic Year: 2024-2025")

if __name__ == "__main__":
    main()
