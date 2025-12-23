import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. تنظیمات سیستمی ---
st.set_page_config(page_title="AI Finance & Research Platform", layout="wide")

# --- 2. توابع محاسباتی و منطق مالی ---
def calculate_asset_pricing_models(stock_returns, market_returns):
    # فرض بر نرخ بهره بدون ریسک (Risk-Free Rate) 0.02 برای اروپا
    rf = 0.02 / 252 
    
    # 1. مدل CAPM
    excess_stock = stock_returns - rf
    excess_market = market_returns - rf
    
    # محاسبه Beta با رگرسیون ساده
    beta = np.cov(excess_stock, excess_market)[0, 1] / np.var(excess_market)
    capm_expected = rf + beta * (excess_market.mean())
    
    return beta, capm_expected

# نکته آکادمیک: برای Fama-French باید داده‌های SMB و HML را از سایت Kenneth French بگیرید.
# در اینجا ما Alpha (سود مازاد بر مدل) را به عنوان معیار اصلی نشان می‌دهیم.
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

def run_backtest(data, signals, initial_capital=10000):
    positions = signals.shift(1).fillna(0)
    returns = data.pct_change()
    strategy_returns = returns * positions
    equity_curve = initial_capital * (1 + strategy_returns).cumprod().fillna(initial_capital)
    return equity_curve

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
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name='Diana AI Strategy', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve, name='Market', line=dict(color='gray', dash='dash')))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def run_monte_carlo(data, prediction_days=30, simulations=50):
    returns = data.pct_change()
    last_price = data.iloc[-1]
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

def categorize_expenses(description):
    desc = description.lower()
    if any(word in desc for word in ['amazon', 'shop', 'buy']): return 'Shopping'
    if any(word in desc for word in ['uber', 'gas', 'snapp', 'train']): return 'Transport'
    if any(word in desc for word in ['restaurant', 'food', 'cafe']): return 'Dining'
    return 'Others'

# --- 3. بدنه اصلی برنامه ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Global Market AI")
    st.markdown("---")

    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    st.sidebar.title("🔬 Research Methodology")
    page = st.sidebar.radio("Go to Module:", ["Global Stock 360°", "AI Wealth Prediction", "Personal Finance AI"])

    if page == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        ticker = st.text_input("Enter Ticker:", "NVDA").upper()
        # --- بخش جدید: مقایسه چندگانه و Backtesting واقعی ---
                st.divider()
                st.header("🔬 Institutional Performance Attribution")
                
                with st.spinner("Calculating Academic Benchmarks..."):
                    # ۱. دریافت دیتای بازار برای مقایسه (S&P 500)
                    market_ticker = "^GSPC" 
                    mkt_data = yf.download(market_ticker, period="1y")['Close']
                    
                    # ۲. هم‌راستا سازی داده‌ها
                    combined_df = pd.concat([df['Close'], mkt_data], axis=1).dropna()
                    combined_df.columns = ['Stock', 'Market']
                    
                    stock_rets = combined_df['Stock'].pct_change().dropna()
                    mkt_rets = combined_df['Market'].pct_change().dropna()

                    # ۳. محاسبه استراتژی AI (مثلاً تقاطع میانگین متحرک)
                    # در اینجا سیگنال 1 یعنی خرید و 0 یعنی نقد بودن
                    signals = np.where(combined_df['Stock'] > combined_df['Stock'].rolling(20).mean(), 1, 0)
                    signals = pd.Series(signals, index=combined_df.index).shift(1).fillna(0)
                    
                    # ۴. محاسبه منحنی رشد سرمایه (Equity Curve)
                    initial_investment = 10000
                    ai_returns = stock_rets * signals
                    ai_equity = initial_investment * (1 + ai_returns).cumprod()
                    buy_hold_equity = initial_investment * (1 + stock_rets).cumprod()
                    
                    # ۵. نمایش متریک‌های مقایسه‌ای
                    c1, c2, c3 = st.columns(3)
                    
                    # محاسبه سود نهایی
                    ai_final = ai_equity.iloc[-1]
                    bh_final = buy_hold_equity.iloc[-1]
                    
                    c1.metric("AI Strategy Final", f"${ai_final:,.0f}", f"{(ai_final/initial_investment-1):.2%}")
                    c2.metric("Buy & Hold Final", f"${bh_final:,.0f}", f"{(bh_final/initial_investment-1):.2%}")
                    
                    # محاسبه بتا (Beta) برای CAPM
                    beta = np.cov(stock_rets, mkt_rets)[0, 1] / np.var(mkt_rets)
                    c3.metric("Systematic Risk (Beta)", f"{beta:.2f}")

                    # ۶. نمودار مقایسه‌ای حرفه‌ای
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(x=ai_equity.index, y=ai_equity, name='Diana AI Strategy', line=dict(color='gold', width=3)))
                    fig_comp.add_trace(go.Scatter(x=buy_hold_equity.index, y=buy_hold_equity, name='Market Buy & Hold', line=dict(color='gray', dash='dash')))
                    
                    fig_comp.update_layout(title="Strategic Alpha: AI vs Passive Investing", template="plotly_dark", hovermode="x unified")
                    st.plotly_chart(fig_comp, use_container_width=True)

                    # ۷. تحلیل ریسک (Drawdown)
                    st.subheader("📉 Risk Exposure Control")
                    ai_dd = (ai_equity / ai_equity.cummax() - 1) * 100
                    st.area_chart(ai_dd)
                    st.caption("Max Drawdown shows the potential loss from peak to trough.")
        if st.button("Run Full Analysis"):
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            
            if not df.empty:
                st.subheader(f"Analysis for {ticker}")
                st.line_chart(df['Close'])

                # --- اجرای بک‌تست ---
                df['Signal'] = np.where(df['Close'] > df['Close'].rolling(20).mean(), 1, 0)
                equity = run_backtest(df['Close'], df['Signal'])
                benchmark = 10000 * (1 + df['Close'].pct_change()).cumprod().fillna(10000)
                display_backtest_results(equity, benchmark)

                # --- اجرای مونت‌کارلو ---
                st.divider()
                st.subheader("🎲 Monte Carlo Risk Simulation")
                sim_results = run_monte_carlo(df['Close'])
                
                fig_mc = go.Figure()
                for i in range(sim_results.columns.size):
                    fig_mc.add_trace(go.Scatter(y=sim_results[i], mode='lines', opacity=0.2, showlegend=False))
                
                # تحلیل آماری برای دفاع ارشد
                expected_p = sim_results.iloc[-1].mean()
                var_5 = np.percentile(sim_results.iloc[-1], 5)
                
                st.write(f"**Expected Price (30d):** ${expected_p:.2f} | **Value at Risk (5%):** ${var_5:.2f}")
                fig_mc.update_layout(title="Potential Price Paths (Geometric Brownian Motion)", template="plotly_dark")
                st.plotly_chart(fig_mc, use_container_width=True)
                
            else:
                st.error("Ticker not found.")

    elif page == "AI Wealth Prediction":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Ticker to Forecast:", "BTC-USD").upper()
        if st.button("Generate AI Prediction"):
            raw = yf.download(symbol, period="2y").reset_index()
            df_p = raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            
            model = Prophet()
            model.fit(df_p)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction'))
            fig_f.update_layout(template="plotly_dark")
            st.plotly_chart(fig_f, use_container_width=True)

    elif page == "Personal Finance AI":
        st.header("💳 Expense Intelligence")
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df_user = pd.read_csv(uploaded)
            st.write(df_user.head())

    st.sidebar.divider()
    st.sidebar.caption("Thesis Candidate: Master's in Finance/AI")

if __name__ == "__main__":
    main()
