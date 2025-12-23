import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta

# --- 1. تنظیمات سیستمی ---
st.set_page_config(page_title="AI Finance & Research Platform", layout="wide")

# --- 2. موتور محاسباتی (Cache برای سرعت بالا) ---
@st.cache_data(ttl=3600)
def get_global_metrics():
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            price = df['Close'].iloc[-1]
            change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def categorize_expenses(description):
    desc = description.lower()
    if any(word in desc for word in ['amazon', 'shop', 'buy']): return 'Shopping'
    if any(word in desc for word in ['uber', 'gas', 'snapp', 'train']): return 'Transport'
    if any(word in desc for word in ['restaurant', 'food', 'cafe']): return 'Dining'
    if any(word in desc for word in ['rent', 'bill', 'water']): return 'Bills'
    return 'Others'

# --- 3. بدنه اصلی برنامه ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Global Market AI")
    st.markdown("---")

    # ردیف شاخص‌های زنده جهانی
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    # منوی ناوبری
    st.sidebar.title("🔬 Research Methodology")
    page = st.sidebar.radio("Go to Module:", 
                           ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI"])

    # --- ماژول ۱: تحلیل جامع سهام بین‌المللی (بخش مورد نظر شما) ---
    if page == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        ticker = st.text_input("Enter International Ticker (e.g. NVDA, AAPL, TSLA, RACE):", "NVDA").upper()
        
        if st.button("Run Full Analysis"):
            with st.spinner("Analyzing Market Data..."):
                stock = yf.Ticker(ticker)
                df = stock.history(period="1y")
                
                if not df.empty:
                    # نمایش شاخص‌ها
                    info = stock.info
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                    c2.metric("Market Cap", f"{info.get('marketCap', 0):,}")
                    c3.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                    c4.metric("Annual Volatility", f"{(df['Close'].pct_change().std() * np.sqrt(252)):.2%}")

                    # نمودار تکنیکال پیشرفته
                    df['MA50'] = df['Close'].rolling(window=50).mean()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Market Price'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-Day Trend', line=dict(dash='dash')))
                    st.plotly_chart(fig, use_container_width=True)
                    

                    # تحلیل سودآوری
                    st.subheader("Financial Health (Net Income)")
                    try:
                        st.bar_chart(stock.financials.loc['Net Income'])
                    except:
                        st.info("Detailed financials not available for this ticker.")
                else:
                    st.error("Ticker not found or API Limit reached.")

    # --- ماژول ۲: پیش‌بینی با هوش مصنوعی ---
    elif page == "AI Wealth Prediction":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Ticker to Forecast:", "BTC-USD").upper()
        
        if st.button("Generate AI Prediction"):
            raw = yf.download(symbol, period="2y").reset_index()
            df_p = raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            model = Prophet(daily_seasonality=True)
            model.fit(df_p)
            future = model.make_future_dataframe(periods=60)
            forecast = model.predict(future)
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Expected'))
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.2)'))
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.2)'))
            st.plotly_chart(fig_f, use_container_width=True)

    # --- ماژول ۳: تحلیل امور مالی شخصی ---
    elif page == "Personal Finance AI":
        st.header("💳 NLP Expense Categorization")
        uploaded = st.file_uploader("Upload CSV Statement", type="csv")
        if uploaded:
            df_user = pd.read_csv(uploaded)
            if 'Description' in df_user.columns:
                df_user['Category'] = df_user['Description'].apply(categorize_expenses)
                st.write(df_user)
                fig_pie = px.pie(df_user, values='Amount', names='Category', hole=0.5)
                st.plotly_chart(fig_pie)

    # فوتر آکادمیک
    st.sidebar.divider()
    st.sidebar.caption("Thesis Candidate: Master's in Finance/AI\nAcademic Year: 2024-2025")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def run_backtest(data, signals, initial_capital=10000):
    """
    data: دیتای قیمت‌ها (Close price)
    signals: سیگنال‌های تولید شده توسط مدل شما (1 برای خرید، -1 برای فروش)
    """
    positions = signals.shift(1) # معامله در قیمت روز بعد از سیگنال
    returns = data.pct_change()
    strategy_returns = returns * positions
    
    # محاسبه ثروت انباشته (Equity Curve)
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()
    
    return equity_curve, strategy_returns
# --- مرحله ۱: تعریف تابع در بالای فایل ---
def display_backtest_results(equity_curve, benchmark_curve):
    st.subheader("📈 Performance Analysis")
    col1, col2, col3 = st.columns(3)
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    benchmark_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100
    
    col1.metric("AI Total Return", f"{total_return:.2f}%")
    col2.metric("Market Return", f"{benchmark_return:.2f}%")
    col3.metric("Alpha", f"{(total_return - benchmark_return):.2f}%")

    fig = go.Figure()
    # اضافه کردن محور X (تاریخ) برای علمی‌تر شدن نمودار
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name='Diana AI Strategy', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name='Market (Buy & Hold)', line=dict(color='gray', dash='dash')))
    
    fig.update_layout(title="Equity Curve Comparison", template="plotly_dark")
    st.plotly_chart(fig)

# --- مرحله ۲: استفاده از آن در بخش اصلی (Main) ---
# فرض کنید در اینجا مدل شما قیمت‌ها و سیگنال‌ها را محاسبه کرده است
if st.button("Analyze Performance"):
    # شما باید قبل از این خط، equity_curve را با تابعی که قبلا دادم محاسبه کرده باشید
    # مثلا:
    # equity_curve, _ = run_backtest(data['Close'], signals)
    # benchmark_curve = initial_capital * (1 + data['Close'].pct_change()).cumprod()
    
    display_backtest_results(equity_curve, benchmark_curve)
