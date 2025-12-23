import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# تنظیمات اولیه صفحه
st.set_page_config(page_title="Diana Finance AI", layout="wide")

# --- گام اول: تعریف منو در سایدبار ---
st.sidebar.title("🎓 Thesis Navigation")
page = st.sidebar.radio("Select a Module:", ["Standard Technical Analysis", "MSc Portfolio Optimization"])
st.sidebar.divider()
st.sidebar.info("Developed for MSc Financial Management Thesis")

# --- گام دوم: بخش اول (تحلیل تکنیکال و RL) ---
if page == "Standard Technical Analysis":
    st.title("🚀 سامانه هوشمند مدیریت سبد سهام (مدل RL)")
    ticker = st.text_input("نام نماد (مثلاً NVDA):", "AAPL").upper()

    if st.button('تحلیل هوشمند'):
        with st.spinner('در حال پردازش...'):
            data = yf.download(ticker, period="1y", auto_adjust=True)
            
            if data.empty:
                st.error("نماد یافت نشد.")
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                prices = data['Close']
                ma20 = prices.rolling(window=20).mean()
                
                curr_p = float(prices.iloc[-1])
                last_ma = float(ma20.iloc[-1])
                diff = (curr_p - last_ma) / last_ma

                if diff < -0.03:
                    res, advice = "BUY (خرید)", "قیمت پایین‌تر از میانگین؛ افزایش وزن سهم."
                elif diff > 0.03:
                    res, advice = "SELL (فروش)", "قیمت در اشباع؛ کاهش وزن و شناسایی سود."
                else:
                    res, advice = "HOLD (نگهداری)", "قیمت در محدوده تعادل؛ حفظ استراتژی."

                c1, c2 = st.columns(2)
                c1.metric("قیمت فعلی", f"${curr_p:.2f}")
                c2.metric("وضعیت", res)
                st.info(f"**تحلیل مدل:** {advice}")

                st.subheader(f"نمودار تحلیل روند {ticker}")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(prices.index, prices.values, label='Price', color='#1a73e8')
                ax.plot(ma20.index, ma20.values, label='RL Baseline', linestyle='--', color='#f4b400')
                ax.legend()
                st.pyplot(fig)

# --- گام سوم: بخش دوم (بهینه‌سازی پورتفوی) ---
elif page == "MSc Portfolio Optimization":
    st.header("🎯 Portfolio Optimization (MSc Thesis Module)")
    tickers_input = st.text_input("Enter Tickers (separated by space):", "AAPL MSFT GOOGL AMZN NVDA")
    tickers_list = tickers_input.split()

    if st.button('Run Financial Optimization'):
        try:
            with st.spinner('Optimizing...'):
                data = yf.download(tickers_list, period="3y")['Adj Close']
                if data.empty:
                    st.error("No data found.")
                else:
                    mu = expected_returns.mean_historical_return(data)
                    S = risk_models.sample_cov(data)
                    ef = EfficientFrontier(mu, S)
                    weights = ef.max_sharpe()
                    cleaned_weights = ef.clean_weights()
                    
                    st.subheader("Optimal Asset Allocation")
                    fig_pie = px.pie(
                        values=list(cleaned_weights.values()), 
                        names=list(cleaned_weights.keys()),
                        hole=0.4
                    )
                    st.plotly_chart(fig_pie)
                    
                    perf = ef.portfolio_performance()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Return", f"{perf[0]:.2%}")
                    col2.metric("Volatility", f"{perf[1]:.2%}")
                    col3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
