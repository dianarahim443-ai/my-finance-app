import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# تنظیمات صفحه
st.set_page_config(page_title="RL Portfolio Manager", layout="wide")

st.title("🚀 سامانه هوشمند مدیریت سبد سهام (مدل RL)")
st.write("این اپلیکیشن بر اساس الگوریتم یادگیری تقویت‌پذیر، وزن بهینه هر سهم را پیشنهاد می‌دهد.")

ticker = st.text_input("نام نماد بورس بین‌الملل (مثلاً NVDA, TSLA):", "AAPL").upper()

if st.button('تحلیل هوشمند'):
    with st.spinner('در حال پردازش...'):
        # دریافت داده
        df = yf.download(ticker, period="1y", auto_adjust=True)
        
        if df.empty:
            st.error("نماد یافت نشد.")
        else:
            # اصلاح ساختار داده برای جلوگیری از خطا
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            prices = df['Close']
            
            # محاسبات
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

            # نمایش کارت‌ها
            c1, c2 = st.columns(2)
            c1.metric("قیمت فعلی", f"${curr_p:.2f}")
            c2.metric("وضعیت", res)
            st.info(advice)

            # رسم نمودار بدون خطا
            st.subheader(f"نمودار تحلیل {ticker}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(prices.index, prices.values, label='Price')
            ax.plot(ma20.index, ma20.values, label='MA20', linestyle='--')
            ax.legend()
            st.pyplot(fig)

st.sidebar.info("پروژه رساله دکتری مدیریت مالی")
