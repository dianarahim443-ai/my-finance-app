import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. تنظیمات سیستمی (System Configuration) ---
st.set_page_config(page_title="Diana Finance AI | Research Platform", layout="wide", initial_sidebar_state="expanded")

# --- 2. توابع محاسباتی هوشمند (Core Logic) ---

@st.cache_data(ttl=3600)
def get_global_metrics():
    """دریافت زنده شاخص‌های جهانی"""
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            if len(df) >= 2:
                price = float(df['Close'].iloc[-1])
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def calculate_metrics(equity_curve, strategy_returns):
    """محاسبه شاخص‌های عملکرد آکادمیک"""
    rf = 0.02 / 252 
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    return total_return, sharpe, max_dd

def run_monte_carlo(data, prediction_days=30, simulations=50):
    """شبیه‌سازی آماری مسیر قیمت (GBM)"""
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

# --- 3. بدنه اصلی اپلیکیشن (Main Interface) ---

def main():
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Master's Thesis Project | Quantitative Finance & Behavioral Economics_")
    
    # نوار متریک‌های زنده در هدر
    metrics_data = get_global_metrics()
    m_cols = st.columns(len(metrics_data))
    for i, (name, val) in enumerate(metrics_data.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    st.divider()

    # --- سایدبار و ناوبری (Navigation) ---
    st.sidebar.title("🔬 Research Methodology")
    
    with st.sidebar.expander("Academic Framework", expanded=True):
        st.markdown("""
        **1. Quantitative Strategy:**
        Momentum-based Alpha Generation.
        
        **2. Risk Architecture:**
        Monte Carlo GBM Simulations (VaR).
        
        **3. Forecasting Engine:**
        Additive Regression (Prophet).
        """)

    choice = st.sidebar.selectbox("Select Research Module:", 
                                 ["🏠 Home & Documentation", 
                                  "📈 Equity Intelligence", 
                                  "🔮 AI Prediction", 
                                  "💳 Personal Finance AI"])

    # --- MODULE: HOME & DOCUMENTATION ---
    if choice == "🏠 Home & Documentation":
        st.header("📑 Quantitative Research Documentation")
        tab1, tab2, tab3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI vs Traditional"])
        
        with tab1:
            st.subheader("معماری سیستم هوش مصنوعی")
            st.write("""
            - **Prophet Engine:** از یک مدل سری زمانی تجزیه‌پذیر برای جداسازی روند (Trend) و فصلی بودن (Seasonality) استفاده می‌کند.
            - **Stochastic Risk:** شبیه‌سازی‌های مونت‌کارلو بر اساس مدل حرکت براونی هندسی (GBM) پیاده‌سازی شده‌اند:
            """)
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        
        with tab2:
            st.subheader("فرضیات بک‌تست (Backtest Assumptions)")
            st.info("سناریوی عملی: تست روی سهام Nvidia (NVDA) در بازه ۱۲ ماهه.")
            st.write("""
            - سرمایه اولیه: ۱۰,۰۰۰ دلار.
            - هزینه معاملات: صفر (فرضیه نهادی).
            - نرخ بدون ریسک: ۲٪.
            """)
            
        with tab3:
            st.subheader("نوآوری: AI در مقابل روش‌های سنتی")
            compare = {
                "Metric": ["Data Processing", "Trend Detection", "Risk Model", "Integration"],
                "Traditional (Fundamental)": ["Manual Excel", "Linear/Static", "Variance-only", "Siloed"],
                "Diana AI (Quantitative)": ["Automated API", "Non-Linear ML", "Stochastic GBM", "Holistic Portfolio"]
            }
            st.table(compare)

    # --- MODULE: STOCK ANALYSIS ---
    elif choice == "📈 Equity Intelligence":
        st.header("🔍 Backtesting & Alpha Generation")
        ticker = st.text_input("Enter Ticker (e.g., AAPL, NVDA):", "NVDA").upper()
        
        if st.button("Run Institutional Analysis"):
            with st.spinner("Executing..."):
                stock_raw = yf.download(ticker, period="1y")['Close']
                if not stock_raw.empty:
                    stock_data = stock_raw.squeeze()
                    # استراتژی SMA 20
                    signal = np.where(stock_data > stock_data.rolling(20).mean(), 1, 0)
                    returns = stock_data.pct_change() * pd.Series(signal).shift(1).values
                    ai_equity = 10000 * (1 + returns.fillna(0)).cumprod()
                    
                    st.subheader("Performance Analytics")
                    fig_perf = go.Figure()
                    fig_perf.add_trace(go.Scatter(x=ai_equity.index, y=ai_equity, name='AI Strategy', line=dict(color='#FFD700')))
                    fig_perf.update_layout(template="plotly_dark", title="Equity Growth ($10k Start)")
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    st.subheader("Monte Carlo Path Projection")
                    sims = run_monte_carlo(stock_data)
                    fig_mc = px.line(sims, template="plotly_dark", title="Stochastic 30-Day Paths")
                    fig_mc.update_layout(showlegend=False)
                    st.plotly_chart(fig_mc, use_container_width=True)
                else: st.error("Ticker Error.")

    # --- MODULE: AI PREDICTION ---
    elif choice == "🔮 AI Prediction":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Asset to Forecast (e.g., BTC-USD):", "BTC-USD").upper()
        if st.button("Train & Predict"):
            df_raw = yf.download(symbol, period="2y").reset_index()
            if not df_raw.empty:
                df_p = df_raw[['Date', 'Close']].copy()
                df_p.columns = ['ds', 'y']
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                m = Prophet(daily_seasonality=True); m.fit(df_p)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='AI Prediction', line=dict(color='cyan')))
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    # --- MODULE: PERSONAL FINANCE ---
    elif choice == "💳 Personal Finance AI":
        st.header("💳 Intelligent Wealth Management")
        uploaded = st.file_uploader("Upload CSV (Description, Amount)", type="csv")
        if uploaded:
            df_u = pd.read_csv(uploaded)
            if 'Description' in df_u.columns and 'Amount' in df_u.columns:
                def categorize(d):
                    d = str(d).lower()
                    if any(x in d for x in ['shop', 'amazon']): return 'Discretionary'
                    if any(x in d for x in ['rent', 'bill']): return 'Fixed Obligations'
                    return 'Lifestyle/Other'
                
                df_u['Category'] = df_u['Description'].apply(categorize)
                df_u['Amount'] = pd.to_numeric(df_u['Amount']).abs()
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = px.pie(df_u, values='Amount', names='Category', hole=0.5, 
                                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.subheader("Top Expenses")
                    st.table(df_u.sort_values(by='Amount', ascending=False).head(5)[['Description', 'Amount']])
                
                st.divider()
                st.subheader("🤖 AI Behavioral Insight")
                total = df_u['Amount'].sum()
                dis_pct = (df_u[df_u['Category'] == 'Discretionary']['Amount'].sum() / total) * 100
                if dis_pct > 25:
                    st.warning(f"AI Alert: Discretionary spending is high ({dis_pct:.1f}%). Shift capital to AI Wealth modules.")
                else: st.success("Balance: Optimized cash flow detected.")

    st.sidebar.divider()
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
