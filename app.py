import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="RL Portfolio Manager", layout="wide")
st.title("🚀 سامانه هوشمند مدیریت سبد سهام (مدل RL)")

ticker = st.text_input("نام نماد بورس بین‌الملل (مثلاً NVDA, TSLA):", "AAPL").upper()

if st.button('تحلیل هوشمند'):
    with st.spinner('در حال پردازش...'):
        # دانلود داده با تنظیمات اصلاح شده
        data = yf.download(ticker, period="1y", auto_adjust=True)
        
        if data.empty:
            st.error("نماد یافت نشد.")
        else:
            # --- بخش اصلاح خطا ---
            # حذف لایه‌های اضافی از نام ستون‌ها
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            prices = data['Close']
            ma20 = prices.rolling(window=20).mean()
            # --------------------
            
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

            # رسم نمودار اصلاح شده
            st.subheader(f"نمودار تحلیل روند {ticker}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(prices.index, prices.values, label='Price', color='#1a73e8')
            ax.plot(ma20.index, ma20.values, label='RL Baseline', linestyle='--', color='#f4b400')
            ax.legend()
            st.pyplot(fig)

st.sidebar.info("پروژه رساله دکتری مدیریت مالی")
yfinance
PyPortfolioOpt
pLotly
import streamlit as st
import yfinance as download
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import plotly.express as px

st.title("بخش مدیریت پرتفوی (ویژه دفاع ارشد)")

# ۱. دریافت نمادها از کاربر
tickers = st.text_input("نمادهای مورد نظر را با فاصله وارد کنید (مثلا: AAPL TSLA MSFT)", "AAPL TSLA MSFT")
tickers_list = tickers.split()

if st.button('محاسبه بهینه‌ترین سبد'):
    # ۲. دانلود داده‌ها
    data = download.download(tickers_list, period="1y")['Adj Close']
    
    # ۳. محاسبات مالی (مرز کارا)
mu = expected_returns.mean_historical_return(data) # بازده انتظاری
    S = risk_models.sample_cov(data) # ریسک (کوواریانس)
    
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe() # محاسبه بهترین نسبت سود به ریسک
    cleaned_weights = ef.clean_weights()
    
    # ۴. نمایش نتایج به صورت نمودار
    st.subheader("وزن‌های پیشنهادی برای هر سهم:")
    fig = px.pie(values=list(cleaned_weights.values()), names=list(cleaned_weights.keys()), title="Optimal Portfolio Allocation")
    st.plotly_chart(fig)

    # ۵. نمایش شاخص‌های عملکرد (برای سوالات استاد)
    perf = ef.portfolio_performance(verbose=True)
    st.write(f"بازده سالانه انتظاری: {perf[0]:.2%}")
    st.write(f"نوسان‌پذیری (ریسک): {perf[1]:.2%}")
    st.write(f"شاخص شارپ: {perf[2]:.2f}")
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import plotly.express as px

# اضافه کردن یک تیتر برای بخش آکادمیک
st.divider() 
st.header("🎯 Portfolio Optimization (MSc Thesis Module)")

# ورودی برای نمادهای سهام
tickers = st.text_input("Enter Tickers (separated by space) for Portfolio Analysis:", "AAPL MSFT GOOGL AMZN")
tickers_list = tickers.split()

if st.button('Run Financial Optimization'):
    try:
# ۱. دریافت داده‌های ۳ سال اخیر (استاندارد آکادمیک)
        data = yf.download(tickers_list, period="3y")['Adj Close']
        
        # ۲. محاسبه بازده و ریسک
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # ۳. بهینه‌سازی سبد سهام بر اساس شاخص شارپ (Sharpe Ratio)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        # ۴. نمایش خروجی به صورت نمودار Plotly (بسیار شیک برای دفاع)
        st.subheader("Optimal Asset Allocation")
        fig = px.pie(
            values=list(cleaned_weights.values()), 
            names=list(cleaned_weights.keys()),
hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig)
        
        # ۵. نمایش آمارهای کلیدی (این چیزیه که اساتید می‌پرسن)
        perf = ef.portfolio_performance()
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{perf[0]:.2%}")
        col2.metric("Annual Volatility (Risk)", f"{perf[1]:.2%}")
        col3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
        
        st.success("✅ This model uses Mean-Variance Optimization (Markowitz Theory).")
        
    except Exception as e:
        st.error(f"Error: {e}. Please check the
import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import plotly.express as px
st.sidebar.title("Thesis Navigation")
page = st.sidebar.radio("Select a Module:", ["Standard Technical Analysis", "MSc Portfolio Optimization"])
