
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. تنظیمات اولیه سیستم (بسیار حیاتی برای پایداری) ---
st.set_page_config(
    page_title="Diana Finance AI | Institutional Research",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. استایل سفارشی و تصویر پس‌زمینه (CSS) ---
# آدرس تصویر زیر را می‌توانید با لینک مستقیم تصویر خود جایگزین کنید
bg_img = "https://images.unsplash.com/photo-1611974717482-480ce9227694?q=80&w=2070&auto=format&fit=crop"

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{bg_img}");
        background-size: cover;
        background-attachment: fixed;
    }}
    .stApp > header {{
        background: transparent;
    }}
    .main .block-container {{
        background-color: rgba(0, 0, 0, 0.75);
        border-radius: 15px;
        padding: 30px;
        margin-top: 50px;
    }}
    h1, h2, h3, p, span {{
        color: white !important;
    }}
    .stMetric {{
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. توابع محاسباتی (Quantitative Engines) ---

@st.cache_data(ttl=3600)
def get_global_metrics():
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

def calculate_advanced_stats(equity_curve, strategy_returns):
    rf = 0.02 / 252
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    return total_return, sharpe, max_dd

def run_monte_carlo(data, prediction_days=30, simulations=50):
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

# --- 4. رابط کاربری اصلی (UI) ---

def main():
    # هدر سایت با دیتای زنده
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Advanced Quantitative Modeling & Behavioral Economics_")
    
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    # --- ناوبری سایدبار (بدون Anchor برای جلوگیری از رفرش) ---
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.selectbox("Go to:", 
        ["🏠 Documentation", "📈 Equity & Backtest", "🔮 AI Prediction", "💳 Wealth Advisor"])

    # --- بخش 1: مستندات آکادمیک ---
    if page == "🏠 Documentation":
        st.header("📑 Institutional Methodology")
        tab1, tab2 = st.tabs(["Algorithm Logic", "Model Comparison"])
        with tab1:
            st.subheader("Mathematical Framework")
            st.markdown("We utilize the **Geometric Brownian Motion (GBM)** for stochastic risk analysis:")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            
            st.info("The model decomposes time-series into trend, seasonality, and residual components.")
            
        with tab2:
            st.subheader("Traditional vs. Diana AI")
            st.table({
                "Feature": ["Risk Model", "Trend Analysis", "Data Speed"],
                "Traditional": ["Static Variance", "Linear Regression", "Batch (Weekly)"],
                "Diana AI": ["Stochastic GBM", "Non-Linear Prophet", "Real-time API"]
            })

    # --- بخش 2: تحلیل سهام و بک‌تست ---
    elif page == "📈 Equity & Backtest":
        st.header("🔍 Backtesting & Risk Intelligence")
        ticker = st.text_input("Enter Asset Ticker (e.g. NVDA, AAPL):", "NVDA").upper()
        if st.button("Run Quantitative Analysis"):
            with st.spinner("Processing..."):
                stock_raw = yf.download(ticker, period="1y")['Close'].squeeze()
                if not stock_raw.empty:
                    # استراتژی تقاطع میانگین متحرک
                    sma = stock_raw.rolling(20).mean()
                    signal = np.where(stock_raw > sma, 1, 0)
                    returns = stock_raw.pct_change() * pd.Series(signal).shift(1).values
                    equity = 10000 * (1 + returns.fillna(0)).cumprod()
                    
                    ret, sharpe, dd = calculate_advanced_stats(equity, returns.fillna(0))
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Strategy Return", f"{ret:.2f}%")
                    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    c3.metric("Max Drawdown", f"{dd:.2f}%")

                    st.plotly_chart(px.line(equity, title="Equity Growth ($10k Initial)", template="plotly_dark"))
                    
                    st.subheader("Stochastic Stress Test (Monte Carlo)")
                    sims = run_monte_carlo(stock_raw)
                    fig_mc = px.line(sims, template="plotly_dark", title="30-Day Potential Paths")
                    fig_mc.update_layout(showlegend=False)
                    st.plotly_chart(fig_mc, use_container_width=True)

    # --- بخش 3: پیش‌بینی با هوش مصنوعی ---
    elif page == "🔮 AI Prediction":
        st.header("🔮 Neural Time-Series Forecasting")
        symbol = st.text_input("Enter Asset (e.g. BTC-USD):", "BTC-USD").upper()
        if st.button("Generate Forecast"):
            with st.spinner("Training Model..."):
                df = yf.download(symbol, period="2y").reset_index()
                df_p = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=30))
                
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual"))
                fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash')))
                fig_f.update_layout(template="plotly_dark", title=f"Forecast for {symbol}")
                st.plotly_chart(fig_f, use_container_width=True)
                
                upside = ((forecast['yhat'].iloc[-1] / df_p['y'].iloc[-1]) - 1) * 100
                st.metric("30-Day Predicted Move", f"{upside:.2f}%")

    # --- بخش 4: مدیریت مالی رفتاری ---
    elif page == "💳 Wealth Advisor":
        st.header("💳 AI Behavioral Wealth Optimization")
        uploaded = st.file_uploader("Upload CSV (Description, Amount)", type="csv")
        
        # اگر فایل آپلود نشد، دیتای نمونه نشان بده
        if not uploaded:
            st.info("Sample Analysis Mode: Showing standard institutional allocation.")
            df = pd.DataFrame({
                'Description': ['Rent', 'Shop', 'Invest', 'Uber', 'Groceries', 'Netflix'],
                'Amount': [2000, 400, 1000, 100, 300, 20]
            })
        else:
            df = pd.read_csv(uploaded)

        if 'Amount' in df.columns:
            def categorize(d):
                d = str(d).lower()
                if any(x in d for x in ['rent', 'bill']): return 'Fixed Needs'
                if any(x in d for x in ['invest', 'stock', 'save']): return 'Wealth Building'
                return 'Wants/Lifestyle'
            
            df['Category'] = df['Description'].apply(categorize) if 'Description' in df.columns else 'Lifestyle'
            df['Amount'] = pd.to_numeric(df['Amount']).abs()
            total = df['Amount'].sum()
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.pie(df, values='Amount', names='Category', hole=0.5, template="plotly_dark"))
            with c2:
                cat_sums = df.groupby('Category')['Amount'].sum()
                w_pct = (cat_sums.get('Wealth Building', 0) / total) * 100 if total > 0 else 0
                st.subheader("AI Advice")
                st.metric("Wealth Building Allocation", f"{w_pct:.1f}%")
                if w_pct < 20:
                    st.warning("Increase investment by 10% to reach institutional targets.")
                else: st.success("Financial behavior is optimized.")

    st.sidebar.divider()
    st.sidebar.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
