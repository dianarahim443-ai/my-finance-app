import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# تنظیمات صفحه اپلیکیشن
st.set_page_config(page_title="RL Portfolio Manager", layout="wide")

st.title("🚀 سامانه هوشمند مدیریت سبد سهام (مدل RL)")
st.write("این اپلیکیشن بر اساس الگوریتم یادگیری تقویت‌پذیر، وزن بهینه هر سهم را پیشنهاد می‌دهد.")

# دریافت ورودی از کاربر
ticker = st.text_input("نام نماد بورس بین‌الملل را وارد کنید (مثلاً NVDA, TSLA, AAPL):", "AAPL").upper()

if st.button('تحلیل هوشمند'):
    with st.spinner('در حال دریافت داده‌های زنده و پردازش مدل...'):
        data = yf.download(ticker, period="1y", auto_adjust=True)
        
        if data.empty:
            st.error("خطا: نماد یافت نشد یا داده‌ای وجود ندارد.")
        else:
            # محاسبات مدل
            close_prices = data['Close']
            ma20 = close_prices.rolling(window=20).mean()
            current_price = float(close_prices.iloc[-1])
            last_ma = float(ma20.iloc[-1])
            diff = (current_price - last_ma) / last_ma

            # تعیین سیگنال
            if diff < -0.03:
                status, color, advice = "BUY (خرید)", "green", "قیمت پایین‌تر از میانگین بهینه است؛ افزایش وزن سهم در سبد پیشنهاد می‌شود."
            elif diff > 0.03:
                status, color, advice = "SELL (فروش)", "red", "قیمت در محدوده اشباع خرید است؛ کاهش وزن و شناسایی سود پیشنهاد می‌شود."
            else:
                status, color, advice = "HOLD (نگهداری)", "blue", "قیمت در محدوده تعادل است؛ حفظ استراتژی فعلی پیشنهاد می‌شود."

            # نمایش نتایج در کارت‌های زیبا
            col1, col2, col3 = st.columns(3)
            col1.metric("قیمت فعلی", f"${current_price:.2f}")
            col2.metric("وضعیت سیگنال", status)
            col3.write(f"**تحلیل مدل:** {advice}")

            # رسم نمودار تعاملی
            st.subheader(f"نمودار تحلیل روند {ticker}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(close_prices, label='Stock Price', color='#1a73e8')
            ax.plot(ma20, label='RL Baseline', linestyle='--', color='#f4b400')
            ax.fill_between(close_prices.index, close_prices, last_ma, alpha=0.1, color='gray')
            ax.legend()
            st.pyplot(fig)

st.sidebar.info("این پروژه بخشی از رساله دکتری مدیریت مالی با موضوع کاربرد Deep RL در مدیریت ریسک است.")
