import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# --- 2. CORE ENGINES ---
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

def calculate_metrics(equity_curve, strategy_returns):
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

# --- 3. MAIN APP ---
def main():
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Master's Thesis Project | Quantitative Finance & Behavioral Economics_")
    
    # Global Header
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    # Sidebar Navigation (Anti-Redirect Loop)
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.selectbox("Select Module:", 
                                ["🏠 Home & Documentation", 
                                 "📈 Equity Intelligence", 
                                 "🔮 AI Prediction", 
                                 "💳 Personal Finance AI"])

    if page == "🏠 Home & Documentation":
        st.header("📑 Quantitative Research Documentation")
        t1, t2, t3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI vs Traditional"])
        with t1:
            st.subheader("AI System Architecture")
            st.markdown("**1. Prophet Engine:** Time-series decomposition (Trend/Seasonality).")
                        st.markdown("**2. Stochastic Risk Modeling:** Based on GBM.")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            # اصلاح خط ۱۰۵ (استفاده از فرمت صحیح برای استریم‌لیت)
            st.info("Parameters: S_t = Asset Price, mu = Drift, sigma = Volatility, W_t = Wiener Process.")

        with t2:
            st.subheader("Methodology")
            st.write("- Capital: $10,000\n- Rebalancing: Daily\n- Rate: 2%")

        with t3:
            st.subheader("AI Innovation")
            compare = {"Feature": ["Data", "Trend", "Risk"], "Traditional": ["Manual", "Linear", "Variance"], "Diana AI": ["API", "ML", "GBM"]}
            st.table(compare)

    elif page == "📈 Equity Intelligence":
        st.header("🔍 Backtesting")
        ticker = st.text_input("Enter Ticker:", "NVDA").upper()
        if st.button("Analyze"):
            data = yf.download(ticker, period="1y")['Close'].squeeze()
            if not data.empty:
                # Strategy logic
                signal = np.where(data > data.rolling(20).mean(), 1, 0)
                returns = data.pct_change() * pd.Series(signal).shift(1).values
                equity = 10000 * (1 + returns.fillna(0)).cumprod()
                
                st.plotly_chart(px.line(equity, title="Equity Curve", template="plotly_dark"))
                
                st.subheader("Monte Carlo Simulation")
                sims = run_monte_carlo(data)
                st.plotly_chart(px.line(sims, template="plotly_dark").update_layout(showlegend=False))
                
    elif page == "🔮 AI Prediction":
        st.header("🔮 Forecast")
        symbol = st.text_input("Asset:", "BTC-USD").upper()
        if st.button("Predict"):
            df = yf.download(symbol, period="2y").reset_index()
            df_p = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            m = Prophet().fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=30))
            st.plotly_chart(px.line(forecast, x='ds', y='yhat', template="plotly_dark"))

    elif page == "💳 Personal Finance AI":
        st.header("💳 Intelligent Wealth")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'Description' in df.columns and 'Amount' in df.columns:
                df['Amount'] = pd.to_numeric(df['Amount']).abs()
                st.plotly_chart(px.pie(df, values='Amount', names='Description', hole=0.5, template="plotly_dark"))

    st.sidebar.divider()
    st.sidebar.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
