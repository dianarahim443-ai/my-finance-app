import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --------------------------------------------------
# 1. App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Finance & Research Platform",
    layout="wide"
)

# --------------------------------------------------
# 2. Financial & Quant Functions
# --------------------------------------------------
def calculate_asset_pricing_models(stock_returns, market_returns):
    rf = 0.02 / 252  # Risk-free rate
    combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
    combined.columns = ['stock', 'market']
    excess_stock = combined['stock'] - rf
    excess_market = combined['market'] - rf
    beta = np.cov(excess_stock, excess_market)[0, 1] / np.var(excess_market)
    capm_expected = rf + beta * excess_market.mean()
    return beta, capm_expected

@st.cache_data(ttl=3600)
def get_global_metrics():
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.Ticker(ticker).history(period="2d")
            if len(df) >= 2:
                price = df["Close"].iloc[-1]
                change = (price - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
                data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def run_backtest(prices, signals, initial_capital=10000):
    positions = signals.shift(1).fillna(0)
    returns = prices.pct_change()
    strategy_returns = returns * positions
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()
    return equity_curve.fillna(initial_capital)

def display_backtest_results(equity_curve, benchmark_curve):
    st.subheader("📈 Backtesting & Performance Analysis")
    col1, col2, col3 = st.columns(3)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    benchmark_return = (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1) * 100
    alpha = total_return - benchmark_return
    col1.metric("AI Strategy Return", f"{total_return:.2f}%")
    col2.metric("Market Return", f"{benchmark_return:.2f}%")
    col3.metric("Alpha", f"{alpha:.2f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="AI Strategy", line=dict(color="gold")))
    fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name="Market", line=dict(color="gray", dash="dash")))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

def run_monte_carlo(prices, prediction_days=30, simulations=50):
    returns = prices.pct_change().dropna()
    last_price = prices.iloc[-1]
    daily_vol = returns.std()
    avg_daily_ret = returns.mean()
    simulation_df = pd.DataFrame()
    for i in range(simulations):
        prices_path = [last_price]
        for _ in range(prediction_days):
            next_price = prices_path[-1] * np.exp(avg_daily_ret + daily_vol * np.random.normal())
            prices_path.append(next_price)
        simulation_df[i] = prices_path
    return simulation_df

# --------------------------------------------------
# 3. Main App
# --------------------------------------------------
def main():
    st.title("🏛️ Intelligent Financial Systems & Global Market AI")
    st.markdown("---")

    metrics = get_global_metrics()
    if metrics:
        cols = st.columns(len(metrics))
        for i, (name, val) in enumerate(metrics.items()):
            cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    st.sidebar.title("🔬 Research Methodology")
    page = st.sidebar.radio("Go to Module:", ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI"])

    if page == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        ticker = st.text_input("Enter Ticker:", "NVDA").upper()
        if st.button("Run Full Analysis"):
            df = yf.download(ticker, period="1y")
            if df.empty:
                st.error("Ticker not found.")
            else:
                # اصلاح برای برداشتن ستون Close به صورت Series
                prices = df["Close"]
                if isinstance(prices, pd.DataFrame): # رفع باگ نسخه‌های جدید yfinance
                    prices = prices.iloc[:, 0]
                
                st.subheader(f"Price Chart – {ticker}")
                st.line_chart(prices)

                signals = np.where(prices > prices.rolling(20).mean(), 1, 0)
                signals = pd.Series(signals, index=prices.index)
                
                equity = run_backtest(prices, signals)
                benchmark = 10000 * (1 + prices.pct_change()).cumprod().fillna(10000)

                display_backtest_results(equity, benchmark)

                st.divider()
                st.subheader("🎲 Monte Carlo Risk Simulation")
                sim_results = run_monte_carlo(prices)
                fig_mc = go.Figure()
                for col in sim_results.columns:
                    fig_mc.add_trace(go.Scatter(y=sim_results[col], mode="lines", opacity=0.1, showlegend=False))
                
                expected_price = sim_results.iloc[-1].mean()
                var_5 = np.percentile(sim_results.iloc[-1], 5)
                st.write(f"**Expected Price (30d):** ${expected_price:.2f} | **VaR (5%):** ${var_5:.2f}")
                fig_mc.update_layout(title="Monte Carlo Price Paths", template="plotly_dark", height=400)
                st.plotly_chart(fig_mc, use_container_width=True)

    elif page == "AI Wealth Prediction":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Ticker to Forecast:", "BTC-USD").upper()
        if st.button("Generate AI Prediction"):
            raw = yf.download(symbol, period="2y").reset_index()
            # اطمینان از فرمت Prophet
            df_p = pd.DataFrame()
            df_p['ds'] = raw['Date'].dt.tz_localize(None)
            df_p['y'] = raw['Close'].values
            
            model = Prophet()
            model.fit(df_p)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
            fig_f.update_layout(title="Prophet AI Forecast (30 Days)", template="plotly_dark")
            st.plotly_chart(fig_f, use_container_width=True)

    elif page == "Personal Finance AI":
        st.header("💳 Expense Intelligence")
        st.info("Upload your CSV file to analyze spending patterns.")

    st.sidebar.divider()
    st.sidebar.caption("Thesis Candidate: Master's in Finance/AI")

if __name__ == "__main__":
    main()
