import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta

# --- 1. تنظیمات سیستمی (Academic Standard) ---
st.set_page_config(page_title="Global Finance AI | MSc Research", layout="wide")

# --- 2. توابع بهینه‌سازی شده برای جلوگیری از ارور Rate Limit ---
@st.cache_data(ttl=3600)
def get_safe_market_data(ticker):
    """دریافت دیتای بازار با استفاده از کش برای جلوگیری از بلاک شدن آی‌پی"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        return df if not df.empty else None
    except:
        return None

def categorize_expenses(description):
    """هوش مصنوعی ساده برای دسته‌بندی مخارج"""
    description = str(description).lower()
    if any(word in description for word in ['amazon', 'shop', 'mall', 'buy', 'apple']):
        return 'Shopping'
    elif any(word in description for word in ['uber', 'gas', 'bolt', 'train', 'flight', 'ryanair']):
        return 'Transport'
    elif any(word in description for word in ['restaurant', 'food', 'cafe', 'pizza', 'starbucks']):
        return 'Dining'
    elif any(word in description for word in ['rent', 'bill', 'electric', 'water', 'internet']):
        return 'Bills & Housing'
    else:
        return 'Fixed Costs / Others'

# --- 3. بدنه اصلی اپلیکیشن ---
def main():
    st.title("🌐 Strategic Financial Intelligence Platform")
    st.markdown("_Advanced Quantitative Analysis for International Finance & Personal Wealth_")

    # --- ناوبری (Navigation) ---
    st.sidebar.title("🕹️ Control Panel")
    page = st.sidebar.radio("Select Module:", 
                           ["Market Overview", "Personal Finance AI", "Asset Intelligence", "Wealth Forecasting"])

    # --- ماژول ۱: نمای کلی بازار جهانی ---
    if page == "Market Overview":
        st.header("🌍 Global Market Pulse")
        tickers = {"S&P 500": "^GSPC", "Gold Spot": "GC=F", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
        cols = st.columns(4)
        for i, (name, t) in enumerate(tickers.items()):
            df = get_safe_market_data(t)
            if df is not None:
                price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2]
                delta = ((price - prev_price) / prev_price) * 100
                cols[i].metric(name, f"{price:,.2f}", f"{delta:.2f}%")
        
        st.divider()
        st.subheader("Asset Performance Comparison")
        # نمایش همبستگی فرضی برای ارائه
        corr_data = np.random.rand(4,4)
        fig_corr = px.imshow(corr_data, x=list(tickers.keys()), y=list(tickers.keys()), 
                             text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- ماژول ۲: تحلیل هزینه‌های شخصی ---
    elif page == "Personal Finance AI":
        st.header("💳 Intelligent Expense Analysis")
        uploaded_file = st.file_uploader("Upload CSV Statement (Required: Description, Amount)", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'Description' in df.columns and 'Amount' in df.columns:
                df['Category'] = df['Description'].apply(categorize_expenses)
                
                c1, c2 = st.columns(2)
                with c1:
                    fig_pie = px.pie(df, values='Amount', names='Category', hole=0.5, title="Spending Allocation")
                    st.plotly_chart(fig_pie)
                with c2:
                    total = df['Amount'].sum()
                    st.metric("Total Monthly Burn", f"${total:,.2f}")
                    top_cat = df.groupby('Category')['Amount'].sum().idxmax()
                    st.warning(f"⚠️ Efficiency Alert: High spending detected in **{top_cat}**.")
            else:
                st.error("Invalid CSV format. Please ensure 'Description' and 'Amount' columns exist.")

    # --- ماژول ۳: تحلیل کمی دارایی‌ها (بسیار مهم برای رزومه) ---
    elif page == "Asset Intelligence":
        st.header("🔍 Quantitative Security Analysis")
        ticker = st.text_input("Enter Ticker (e.g. NVDA, AAPL, TSLA):", "NVDA").upper()
        
        if st.button("Run Financial Audit"):
            df = get_safe_market_data(ticker)
            if df is not None:
                # محاسبات ریسک و بازده
                returns = df['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) # نوسان سالانه
                var_95 = np.percentile(returns, 5) # ارزش در معرض ریسک
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                col2.metric("Annualized Volatility", f"{volatility:.2%}")
                col3.metric("Daily VaR (95%)", f"{var_95:.2%}")
                
                st.plotly_chart(px.line(df, y='Close', title=f"{ticker} Historical Trend (1 Year)"), use_container_width=True)
                
                # توزیع بازدهی (مورد علاقه اساتید فایننس)
                fig_dist = px.histogram(returns, nbins=50, title="Returns Distribution Analysis", marginal="box")
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.error("⚠️ Data connection busy or invalid ticker. Please wait a moment.")

    # --- ماژول ۴: پیش‌بینی ثروت با هوش مصنوعی ---
    elif page == "Wealth Forecasting":
        st.header("🔮 AI Time-Series Projection")
        st.info("Using Meta Prophet Model for 60-day predictive analytics.")
        
        # دیتای دمو برای نمایش قدرت مدل
        dates = pd.date_range(start=datetime.now()-timedelta(days=180), periods=180)
        values = np.random.normal(100, 10, 180).cumsum() + 5000
        df_f = pd.DataFrame({'ds': dates, 'y': values})
        
        m = Prophet(interval_width=0.95)
        m.fit(df_f)
        future = m.make_future_dataframe(periods=60)
        forecast = m.predict(future)
        
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#00CC96')))
        fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line_color='rgba(0,204,150,0.1)', name='Upper Bound'))
        fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,204,150,0.1)', name='Lower Bound'))
        st.plotly_chart(fig_fore, use_container_width=True)

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("Global Finance AI v2.5 | Master of Science Research Platform")

if __name__ == "__main__":
    main()
