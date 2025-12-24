import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. تنظیمات سیستمی (Configuration) ---
st.set_page_config(
    page_title="Diana Finance AI | Institutional Research",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. توابع محاسباتی پیشرفته (Quantitative Engines) ---

@st.cache_data(ttl=3600)
def get_global_metrics():
    """دریافت داده‌های زنده بازارهای جهانی برای هدر سایت"""
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            if len(df) >= 2:
                price = float(df['Close'].iloc[-1])
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                data[name] = (price, change)
        except: data[name] = (0.0, 0.0)
    return data

def calculate_advanced_metrics(equity_curve, strategy_returns):
    """محاسبه شاخص‌های تخصصی مدیریت سبد سهام"""
    rf = 0.02 / 252  # Risk-free rate (daily proxy)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    
    # Sharpe Ratio
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    
    # Maximum Drawdown
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    
    # Alpha (Simple proxy vs buy & hold)
    return total_return, sharpe, max_dd

def run_monte_carlo(data, prediction_days=30, simulations=50):
    """مدل‌سازی تصادفی مسیر قیمت با استفاده از حرکت براونی هندسی (GBM)"""
    returns = data.pct_change().dropna()
    last_price = float(data.iloc[-1])
    daily_vol = returns.std()
    avg_daily_ret = returns.mean()
    simulation_df = pd.DataFrame()
    for i in range(simulations):
        prices = [last_price]
        for d in range(prediction_days):
            next_price = prices[-1] * np.exp(avg_daily_ret + daily_vol * np.random.normal())
            prices.append(next_price)
        simulation_df[i] = prices
    return simulation_df

# --- 3. ساختار اصلی برنامه (Main Interface) ---

def main():
    # هدر و وضعیت بازارهای جهانی
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Master's Thesis Project | Quantitative Finance & Behavioral Economics_")
    
    metrics = get_global_metrics()
    if metrics:
        m_cols = st.columns(len(metrics))
        for i, (name, val) in enumerate(metrics.items()):
            m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    # --- ناوبری در سایدبار (Navigation بدون تغییر URL برای رفع لوپ) ---
    st.sidebar.title("🔬 Research Modules")
    page = st.sidebar.selectbox(
        "Choose a Section:",
        ["🏠 Home & Documentation", 
         "📈 Equity Intelligence", 
         "🔮 AI Prediction", 
         "💳 Personal Finance AI"]
    )

    # --- صفحه 1: مستندات آکادمیک ---
    if page == "🏠 Home & Documentation":
        st.header("📑 Quantitative Research Documentation")
        tab1, tab2, tab3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI vs Traditional"])
        
        with tab1:
            st.subheader("AI System Architecture")
            st.markdown("""
            **1. Prophet Engine:**
            Utilizes a **decomposable time-series model** (Harvey & Peters 1990) to analyze: **Trend**, **Seasonality**, and **Holidays**.
            """)
            st.markdown("""
            **2. Stochastic Risk Modeling:**
            Implemented via **Monte Carlo methods** based on the **Geometric Brownian Motion (GBM)** framework:
            """)
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.info("Where $S_t$ is asset price, $\mu$ is drift, $\sigma$ is volatility, and $W_t$ is the Wiener process.")

        with tab2:
            st.subheader("Backtest Methodology & Constraints")
            st.write("""
            - **Universe:** Global Equities & Commodities (via Yahoo Finance API).
            - **Initial Capital:** $10,000 USD (Virtual Basis).
            - **Lookback Period:** 252 Trading Days (1 Year).
            - **Execution:** Zero-slippage assumption for institutional simulation.
            """)
            
        with tab3:
            st.subheader("Innovation: AI vs. Traditional Analysis")
            compare_data = {
                "Feature": ["Data Processing", "Trend Detection", "Risk Model", "Integration"],
                "Traditional (Fundamental)": ["Manual Spreadsheets", "Linear/Static", "Variance-only", "Siloed"],
                "Diana AI (Quantitative)": ["Automated API", "Non-Linear ML", "Stochastic GBM", "Holistic View"]
            }
            st.table(compare_data)

    # --- صفحه 2: تحلیل هوشمند سهام ---
    elif page == "📈 Equity Intelligence":
        st.header("🔍 Backtesting & Alpha Generation")
        ticker = st.text_input("Enter Ticker (e.g., NVDA, AAPL, TSLA):", "NVDA").upper()
        
        if st.button("Execute Quantitative Run"):
            with st.spinner("Analyzing Market Dynamics..."):
                stock_raw = yf.download(ticker, period="1y")['Close']
                if not stock_raw.empty:
                    stock_data = stock_raw.squeeze()
                    
                    # استراتژی مومنتوم (SMA 20)
                    sma = stock_data.rolling(window=20).mean()
                    signal = np.where(stock_data > sma, 1, 0)
                    returns = stock_data.pct_change() * pd.Series(signal).shift(1).values
                    ai_equity = 10000 * (1 + returns.fillna(0)).cumprod()
                    
                    # محاسبه متریک‌های پیشرفته
                    ret, sharpe, dd = calculate_advanced_metrics(ai_equity, returns.fillna(0))
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Strategy Return", f"{ret:.2f}%")
                    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    col3.metric("Max Drawdown", f"{dd:.2f}%")

                    # رسم نمودار Equity Curve
                    fig_equity = px.line(ai_equity, title=f"AI Strategy Performance: {ticker}", template="plotly_dark")
                    fig_equity.update_traces(line_color='#FFD700', line_width=3)
                    st.plotly_chart(fig_equity, use_container_width=True)

                    # شبیه‌سازی مونت‌کارلو برای پیش‌بینی ریسک
                    st.subheader("Risk Modeling: Monte Carlo Path Simulations")
                    sim_results = run_monte_carlo(stock_data)
                    fig_mc = px.line(sim_results, template="plotly_dark", title="Potential 30-Day Stochastic Paths")
                    fig_mc.update_layout(showlegend=False, xaxis_title="Days Ahead", yaxis_title="Price ($)")
                    st.plotly_chart(fig_mc, use_container_width=True)
                else:
                    st.error("Ticker not found. Please check the symbol.")

    # --- صفحه 3: پیش‌بینی با هوش مصنوعی ---
    elif page == "🔮 AI Prediction":
        st.header("🔮 Time-Series Forecasting Engine")
        symbol = st.text_input("Asset for Forecasting (e.g., BTC-USD, MSFT):", "BTC-USD").upper()
        
        if st.button("Generate Predictive Model"):
            with st.spinner("Training Prophet Model..."):
                df_raw = yf.download(symbol, period="2y").reset_index()
                if not df_raw.empty:
                    df_p = df_raw[['Date', 'Close']].copy()
                    df_p.columns = ['ds', 'y']
                    df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                    
                    model = Prophet(daily_seasonality=True)
                    model.fit(df_p)
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    
                    # رسم نمودار پیش‌بینی
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical", line=dict(color="#636EFA")))
                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(color="#00CC96", dash='dash')))
                    fig_forecast.update_layout(title=f"30-Day Forecast for {symbol}", template="plotly_dark")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    st.success("Model trained successfully. Note: Predictions are probabilistic.")

    # --- صفحه 4: مدیریت مالی شخصی (رویکرد رفتاری) ---
    elif page == "💳 Personal Finance AI":
        st.header("💳 Behavioral Wealth Management")
        uploaded = st.file_uploader("Upload Transaction Data (CSV format with 'Description' and 'Amount')", type="csv")
        
        if uploaded:
            df_u = pd.read_csv(uploaded)
            if 'Description' in df_u.columns and 'Amount' in df_u.columns:
                def categorize_behavioral(desc):
                    desc = str(desc).lower()
                    if any(x in desc for x in ['amazon', 'uber', 'starbucks', 'shop', 'netflix']): return 'Discretionary (Non-Essential)'
                    if any(x in desc for x in ['rent', 'bill', 'electric', 'insurance']): return 'Fixed Obligations'
                    if any(x in desc for x in ['stock', 'crypto', 'invest', 'saving']): return 'Wealth Building'
                    return 'Lifestyle/Daily'

                df_u['Category'] = df_u['Description'].apply(categorize_behavioral)
                df_u['Amount'] = pd.to_numeric(df_u['Amount']).abs()
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_pie = px.pie(df_u, values='Amount', names='Category', hole=0.5, template="plotly_dark", title="Capital Allocation Mix")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c2:
                    st.subheader("Strategic Insights")
                    total = df_u['Amount'].sum()
                    dis_val = df_u[df_u['Category'] == 'Discretionary (Non-Essential)']['Amount'].sum()
                    dis_pct = (dis_val / total) * 100 if total > 0 else 0
                    
                    st.metric("Total Monthly Outflow", f"${total:,.2f}")
                    if dis_pct > 25:
                        st.warning(f"High Discretionary Spending: {dis_pct:.1f}%. Suggest reallocating 10% to Wealth Building.")
                    else:
                        st.success(f"Optimized Cash Flow: Discretionary at {dis_pct:.1f}%.")
            else:
                st.error("CSV must contain 'Description' and 'Amount' columns.")

    # فوتر سایدبار
    st.sidebar.divider()
    st.sidebar.caption(f"System Status: Operational")
    st.sidebar.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
