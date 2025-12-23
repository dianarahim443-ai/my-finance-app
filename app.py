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
