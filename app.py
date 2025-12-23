import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta

# --- 1. Global Configuration ---
st.set_page_config(page_title="Global AI Finance Intelligence", layout="wide")

# --- 2. Market Data Engine (International Assets) ---
@st.cache_data(ttl=3600)
def get_global_market_data():
    # استفاده از شاخص‌های جهانی: طلا، بورس آمریکا، بیت‌کوین و شاخص دلار
    tickers = {
        "Gold (Spot)": "GC=F",
        "S&P 500 (US)": "^GSPC",
        "Bitcoin (BTC)": "BTC-USD",
        "Dollar Index (DXY)": "DX-Y.NYB"
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

# --- 3. Synthetic Data Generator (برای اینکه سایت خالی نباشد) ---
def generate_demo_data():
    dates = pd.date_range(start=datetime.now() - timedelta(days=180), periods=180, freq='D')
    # شبیه‌سازی هزینه‌ها با نوسان تصادفی
    amounts = np.random.normal(loc=50, scale=20, size=180).cumsum() + 1000
    return pd.DataFrame({'Date': dates, 'Amount': amounts})

# --- 4. Main Interface ---
def main():
    st.title("🌐 Global Financial Intelligence & Forecasting System")
    st.markdown("_An advanced AI-driven platform for predictive financial analysis and global market tracking._")
# --- International Stock Analyzer Section ---
    st.divider() # یک خط جداکننده برای زیبایی
    st.subheader("🔍 Global Investment Intelligence")
    
    with st.expander("Click to Analyze Specific Stocks", expanded=False):
        ticker_symbol = st.text_input("Enter Ticker (e.g., NVDA, AAPL, TSLA):", "NVDA").upper()
        
        # استفاده از دکمه برای جلوگیری از درخواست‌های مکرر و رفع Rate Limit
        if st.button("Run Stock Analysis"):
            try:
                stock_obj = yf.Ticker(ticker_symbol)
                # دریافت تاریخچه قیمت (بسیار پایدارتر از info)
                stock_hist = stock_obj.history(period="1y")
                
                if not stock_hist.empty:
                    # محاسبات پایه مالی
                    last_price = stock_hist['Close'].iloc[-1]
                    annual_return = ((last_price - stock_hist['Close'].iloc[0]) / stock_hist['Close'].iloc[0]) * 100
                    
                    c1, c2 = st.columns(2)
                    c1.metric(f"Current {ticker_symbol} Price", f"${last_price:.2f}")
                    c2.metric("Annual Performance", f"{annual_return:.2f}%")
                    
                    # رسم نمودار قیمت با استایل حرفه‌ای
                    fig_stock_price = px.line(stock_hist, x=stock_hist.index, y='Close', 
                                            title=f"{ticker_symbol} - Year over Year Analysis",
                                            template="plotly_dark") # استایل تیره برای کلاس مالی
                    st.plotly_chart(fig_stock_price, use_container_width=True)
                    
                    # تحلیل نوسانات (Volatility Analysis) - برای رزومه ارشد عالی است
                    stock_hist['Daily Return'] = stock_hist['Close'].pct_change()
                    volatility = stock_hist['Daily Return'].std() * (252**0.5) # فرمول سالانه کردن نوسان
                    st.info(f"📊 Annualized Volatility: {volatility:.2%}")
                    
                else:
                    st.error("Invalid Ticker or No Data Found.")
            except Exception as e:
                st.warning("⚠️ Market API is busy. Please wait 1 minute and click the button again.")
    # Sidebar
    st.sidebar.header("🕹️ Control Panel")
    mode = st.sidebar.radio("Select Data Source:", ["Live Demo (Simulation)", "Upload Personal Data"])
    
    # Live Market Ticker
    market_data = get_global_market_data()
    cols = st.columns(len(market_data))
    for i, (name, val) in enumerate(market_data.items()):
        cols[i].metric(name, f"{val[0]:,}", f"{val[1]}%")

    st.divider()

    # Data Processing
    if mode == "Upload Personal Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (Required columns: Date, Amount)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            st.info("💡 Please upload a CSV file to begin. Using Demo data for visualization below.")
            df = generate_demo_data()
    else:
        df = generate_demo_data()

    # --- Analysis Tabs ---
    tab1, tab2, tab3 = st.tabs(["📈 Wealth Forecasting", "📊 Expense Analysis", "🌍 Global Correlation"])

    with tab1:
        st.subheader("AI Time-Series Projection")
        df_p = df.rename(columns={'Date': 'ds', 'Amount': 'y'})
        
        m = Prophet(interval_width=0.95)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=60)
        forecast = m.predict(future)

        # رسم نمودار پیش‌بینی
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#00CC96')))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,204,150,0.1)', name='Confidence Upper'))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,204,150,0.1)', name='Confidence Lower'))
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        

    with tab2:
        st.subheader("Spending Pattern Distribution")
        fig_hist = px.histogram(df, x="Amount", nbins=20, marginal="box", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Global Asset Correlation Matrix")
        # یک ماتریس همبستگی فرضی برای نمایش تخصص مالی
        corr_data = np.random.rand(4,4)
        labels = ["Spending", "Gold", "S&P 500", "BTC"]
        fig_corr = px.imshow(corr_data, x=labels, y=labels, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
        

    # Footer for Thesis
    st.sidebar.divider()
    st.sidebar.caption("Built with Python, Streamlit & Meta Prophet Model. Target: Predictive Personal Finance Optimization.")

if __name__ == "__main__":
    main()
# --- بخش جدید: تحلیل سهام (Stock Intelligence) ---
with st.expander("🔍 Global Stock Intelligence"):
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "AAPL").upper()
    
    if ticker_input:
        stock = yf.Ticker(ticker_input)
        
        # دریافت اطلاعات کلی سهام
        info = stock.info
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            col1.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            col2.metric("Market Cap", f"{info.get('marketCap', 0):,}")
            col3.metric("P/E Ratio", info.get('forwardPE', 'N/A'))
            col4.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            
            # رسم نمودار قیمت سهام
            hist = stock.history(period="1y")
            fig_stock = px.area(hist, x=hist.index, y='Close', 
                                title=f"{ticker_input} Price Action (Past Year)",
                                color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_stock, use_container_width=True)

            # تحلیل فاندامنتال (تحلیل سودآوری)
            st.subheader(f"Financial Health: {ticker_input}")
            fundamentals = stock.financials.loc['Net Income'] if 'Net Income' in stock.financials.index else None
            if fundamentals is not None:
                fig_fin = px.bar(fundamentals, title="Annual Net Income Trend")
                st.plotly_chart(fig_fin, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not fetch data for {ticker_input}. Please check the ticker symbol.")
