import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from scipy.stats import norm

# --- 1. تنظیمات سیستمی (Academic Presentation Mode) ---
st.set_page_config(page_title="QuantFinance AI | Research Platform", layout="wide")

# --- 2. موتور محاسبات کوانت (Advanced Quant Engine) ---
@st.cache_data(ttl=3600)
def get_advanced_analytics(ticker):
    try:
        data = yf.download(ticker, period="2y")['Close']
        if data.empty: return None
        
        returns = data.pct_change().dropna()
        
        # محاسبات شاخص‌های ریسک (مورد نیاز برای دفاع ارشد)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        var_95 = np.percentile(returns, 5) # Value at Risk
        volatility = returns.std() * np.sqrt(252)
        
        return {
            "data": data,
            "returns": returns,
            "sharpe": sharpe,
            "var": var_95,
            "volatility": volatility
        }
    except: return None

# --- 3. بدنه اصلی اپلیکیشن ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Quantitative Analysis")
    st.markdown("---")

    # سایدبار برای ناوبری متدولوژی
    st.sidebar.title("🔬 Methodology")
    menu = st.sidebar.radio("Select Analysis Module:", 
                           ["Market Intelligence", "Predictive Modeling", "Risk Management"])

    # --- ماژول ۱: هوش بازار و همبستگی ---
    if menu == "Market Intelligence":
        st.header("🌍 Global Asset Correlation & Performance")
        
        tickers = ["^GSPC", "GC=F", "BTC-USD", "EURUSD=X"]
        df_market = yf.download(tickers, period="1y")['Close'].pct_change().dropna()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Asset Correlation")
            corr = df_market.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col2:
            st.subheader("Performance Metrics")
            st.dataframe(df_market.describe().T[['mean', 'std', 'min', 'max']])

    # --- ماژول ۲: پیش‌بینی سری زمانی با Prophet ---
    elif menu == "Predictive Modeling":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Enter Asset Ticker:", "NVDA").upper()
        
        if st.button("Train AI Model"):
            with st.spinner("Optimizing Hyperparameters..."):
                df_raw = yf.download(symbol, period="5y").reset_index()
                df_prophet = df_raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                
                m = Prophet(daily_seasonality=True, interval_width=0.95)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=90)
                forecast = m.predict(future)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.1)'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.1)'))
                st.plotly_chart(fig, use_container_width=True)

    # --- ماژول ۳: مدیریت ریسک و توزیع بازدهی ---
    elif menu == "Risk Management":
        st.header("📉 Financial Risk Profiling")
        symbol = st.text_input("Enter Asset:", "AAPL").upper()
        
        analysis = get_advanced_analytics(symbol)
        if analysis:
            c1, c2, c3 = st.columns(3)
            c1.metric("Annualized Volatility", f"{analysis['volatility']:.2%}")
            c2.metric("Sharpe Ratio", f"{analysis['sharpe']:.2f}")
            c3.metric("Daily VaR (95%)", f"{analysis['var']:.2%}")
            
            st.subheader("Returns Distribution Analysis")
            fig_dist = px.histogram(analysis['returns'], nbins=100, marginal="box", 
                                     title=f"Statistical Distribution of {symbol} Returns")
            st.plotly_chart(fig_dist, use_container_width=True)
            
    st.sidebar.divider()
    st.sidebar.caption("Thesis Candidate: [Your Name] | University: [Your University]")

if __name__ == "__main__":
    main()
