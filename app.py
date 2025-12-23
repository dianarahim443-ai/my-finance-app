import streamlit as st
import pandas as pd
import yfinance as yf # برای قیمت‌های زنده بازار
from prophet import Prophet # برای پیش‌بینی آینده (Forecasting)
import plotly.express as px # برای نمودارهای تعاملی ارشد
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# تنظیمات تم و صفحه
st.set_page_config(page_title="Diana Finance AI - MSc Thesis", layout="wide")

# --- منوی سایدبار ---
st.sidebar.title("💎 Academic Dashboard")
page = st.sidebar.radio("Modules:", ["Technical Analysis", "Portfolio Optimization & Correlation"])
st.sidebar.divider()
st.sidebar.write("📌 **Focus:** Risk Management & Asset Allocation")

if page == "Technical Analysis":
    st.title("📈 Technical Inference Engine")
    ticker = st.text_input("Enter Ticker:", "NVDA").upper()
    
    if st.button('Analyze'):
        data = yf.download(ticker, period="1y")
        if not data.empty:
            prices = data['Close']
            ma20 = prices.rolling(window=20).mean()
            
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"${float(prices.iloc[-1]):.2f}")
            c2.write("### RL-based Strategy Signal")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(prices, label="Price")
            ax.plot(ma20, label="20-Day MA", linestyle="--")
            ax.legend()
            st.pyplot(fig)

elif page == "Portfolio Optimization & Correlation":
    st.title("🎯 Strategic Asset Allocation")
    tickers = st.text_input("Enter Tickers (space separated):", "AAPL MSFT GOOGL TSLA NVDA BTC-USD")
    t_list = tickers.split()

    if st.button('Run Deep Analysis'):
        with st.spinner('Calculating Financial Metrics...'):
            data = yf.download(t_list, period="2y")['Adj Close']
            
            # --- بخش ۱: بهینه‌سازی ---
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Optimal Weights (Max Sharpe)")
                fig_pie = px.pie(values=list(weights.values()), names=list(weights.keys()), hole=0.4)
                st.plotly_chart(fig_pie)
            
            # --- بخش ۲: نقشه حرارتی همبستگی (جدید) ---
            with col2:
                st.subheader("Assets Correlation Matrix")
                corr = data.pct_change().corr()
                fig_corr, ax_corr = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="RdYlGn", ax=ax_corr)
                st.pyplot(fig_corr)
            
            # --- بخش ۳: شاخص‌های عملکرد ---
            st.divider()
            perf = ef.portfolio_performance()
            i1, i2, i3 = st.columns(3)
            i1.metric("Expected Annual Return", f"{perf[0]:.2%}")
            i2.metric("Annual Volatility", f"{perf[1]:.2%}")
            i3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
            
            st.success("Analysis based on Modern Portfolio Theory (MPT)")
pip install yfinance pandas plotly
import plotly.express as px

# دریافت تاریخچه یک ساله طلا
gold_history = yf.Ticker("GC=F").history(period="1y").reset_index()

# رسم نمودار با Plotly
fig = px.line(gold_history, x='Date', y='Close', title='Trend of Gold Price (1 Year)')
st.plotly_chart(fig, use_container_width=True)
# فراخوانی تابع
market_data = get_market_metrics()

# ایجاد ستون برای نمایش
col1, col2, col3 = st.columns(3)

with col1:
    price, delta = market_data["Gold"]
    st.metric("Gold (USD)", f"${price}", f"{delta}%")

with col2:
    price, delta = market_data["Brent Oil"]
    st.metric("Brent Oil", f"${price}", f"{delta}%")

with col3:
    price, delta = market_data["FTSE MIB (Italy)"]
    st.metric("Italy Stock Index", f"{price}", f"{delta}%")
    import yfinance as yf
import streamlit as st

def get_market_metrics():
    # دریافت داده‌های طلا، نفت برنت و شاخص بورس ایتالیا
    tickers = {
        "Gold": "GC=F",
        "Brent Oil": "BZ=F",
        "FTSE MIB (Italy)": "FTSEMIB.MI"
    }
    
    results = {}
    for name, ticker in tickers.items():
        data = yf.Ticker(ticker).history(period="2d")
        if len(data) >= 2:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2]
            delta = ((current_price - prev_price) / prev_price) * 100
            results[name] = (round(current_price, 2), round(delta, 2))
    return results
    import plotly.express as px

# دریافت تاریخچه یک ساله طلا
gold_history = yf.Ticker("GC=F").history(period="1y").reset_index()

# رسم نمودار با Plotly
fig = px.line(gold_history, x='Date', y='Close', title='Trend of Gold Price (1 Year)')
st.plotly_chart(fig, use_container_width=True)
