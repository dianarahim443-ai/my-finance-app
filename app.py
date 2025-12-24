
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# CSS برای پس‌زمینه و زیبایی بصری (بدون ایجاد اختلال در عملکرد)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url('https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(20, 20, 20, 0.85);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid #333;
    }
    h1, h2, h3 { color: #FFD700 !important; }
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. QUANTITATIVE ENGINES ---

@st.cache_data(ttl=3600)
def get_market_pulse():
    tickers = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD"}
    data = {}
    for name, sym in tickers.items():
        try:
            df = yf.Ticker(sym).history(period="2d")
            price = df['Close'].iloc[-1]
            change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def run_backtest_logic(data):
    # استراتژی تقاطع میانگین متحرک 20 و 50 روزه
    fast = data.rolling(20).mean()
    slow = data.rolling(50).mean()
    signal = np.where(fast > slow, 1, 0)
    returns = data.pct_change()
    strat_returns = returns * pd.Series(signal).shift(1).values
    equity_curve = 10000 * (1 + strat_returns.fillna(0)).cumprod()
    
    # محاسبات ریسک
    rf = 0.02 / 252
    excess = strat_returns.fillna(0) - rf
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else 0
    mdd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100
    return equity_curve, sharpe, mdd

def monte_carlo_simulation(last_price, mu, sigma, days=30, sims=100):
    simulation_df = pd.DataFrame()
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
        simulation_df[i] = prices
    return simulation_df

# --- 3. MAIN INTERFACE ---

def main():
    # Header Section
    st.title("🏛️ Diana Finance: AI Institutional Research")
    pulse = get_market_pulse()
    p_cols = st.columns(len(pulse))
    for i, (name, val) in enumerate(pulse.items()):
        p_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    # Navigation (No Anchor Links to prevent Redirect Loop)
    st.sidebar.title("🔬 Core Modules")
    page = st.sidebar.selectbox("Select Research Area:", 
        ["📚 Research & Methodology", 
         "📈 Equity Intelligence", 
         "🔮 AI Forecasting Engine", 
         "💳 Wealth Optimization"])

    # --- PAGE 1: DOCUMENTATION ---
    if page == "📚 Research & Methodology":
        st.header("📑 Quantitative Research Framework")
        t1, t2, t3 = st.tabs(["Mathematical Logic", "AI Architecture", "Project Scope"])
        
        with t1:
            st.subheader("Governing SDE (Geometric Brownian Motion)")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.markdown("""
            این مدل پایه برای شبیه‌سازی مسیرهای تصادفی قیمت (Monte Carlo) استفاده می‌شود که در آن:
            - $\mu$: نرخ بازدهی مورد انتظار (Drift)
            - $\sigma$: نوسانات بازار (Volatility)
            - $W_t$: فرآیند وینر (Wiener Process)
            """)
            
            
        with t2:
            st.subheader("Prophet Decomposable Model")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown("تحلیل سری زمانی شامل سه بخش **Trend ($g$)**، **Seasonality ($s$)** و **Holidays ($h$)** است.")
            

    # --- PAGE 2: EQUITY INTELLIGENCE ---
    elif page == "📈 Equity Intelligence":
        st.header("🔍 Strategy Backtesting & Risk Audit")
        ticker = st.text_input("Institutional Ticker:", "NVDA").upper()
        
        if st.button("Run Quantitative Simulation"):
            df = yf.download(ticker, period="2y")['Close'].squeeze()
            if not df.empty:
                equity, sharpe, mdd = run_backtest_logic(df)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Strategy Return", f"{((equity.iloc[-1]/10000)-1)*100:.2f}%")
                c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                c3.metric("Max Drawdown", f"{mdd:.2f}%")
                
                st.plotly_chart(px.line(equity, title="Alpha Generation Curve", template="plotly_dark"))
                
                # Monte Carlo
                st.subheader("Stochastic Stress Test (Monte Carlo)")
                returns = df.pct_change().dropna()
                sims = monte_carlo_simulation(df.iloc[-1], returns.mean(), returns.std())
                fig_mc = px.line(sims, template="plotly_dark", title="30-Day Potential Paths")
                fig_mc.update_layout(showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)

    # --- PAGE 3: AI FORECASTING ---
    elif page == "🔮 AI Forecasting Engine":
        st.header("🔮 AI Predictive Modeling")
        asset = st.text_input("Forecast Target (e.g. BTC-USD, AAPL):", "BTC-USD").upper()
        
        if st.button("Initialize Neural Forecast"):
            raw = yf.download(asset, period="3y").reset_index()
            df_p = raw[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            df_p['ds'] = df_p['ds'].dt.tz_localize(None)
            
            m = Prophet(daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=60))
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical"))
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash', color='gold')))
            fig_f.update_layout(template="plotly_dark", title=f"60-Day Forward Forecast for {asset}")
            st.plotly_chart(fig_f, use_container_width=True)
            
            st.subheader("Behavioral Cycles Analysis")
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

    # --- PAGE 4: PERSONAL FINANCE ---
    elif page == "💳 Wealth Optimization":
        st.header("💳 AI Behavioral Wealth Management")
        uploaded = st.file_uploader("Upload Transaction Ledger (CSV)", type="csv")
        
        if uploaded or st.button("Show Institutional Sample"):
            if not uploaded:
                df = pd.DataFrame({
                    'Description': ['Salary', 'Rent', 'Amazon', 'ETF Invest', 'Uber', 'Groceries'],
                    'Amount': [5000, -1500, -200, -1000, -50, -300],
                    'Category': ['Income', 'Fixed', 'Wants', 'Wealth', 'Wants', 'Fixed']
                })
            else: df = pd.read_csv(uploaded)

            df['Amount'] = pd.to_numeric(df['Amount'])
            outflow = df[df['Amount'] < 0].abs()
            total_out = outflow['Amount'].sum()
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                fig_pie = px.pie(outflow, values='Amount', names='Category', hole=0.6, template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_b:
                st.subheader("50/30/20 Audit")
                fixed_pct = (outflow[outflow['Category'] == 'Fixed']['Amount'].sum() / total_out) * 100
                wealth_pct = (outflow[outflow['Category'] == 'Wealth']['Amount'].sum() / total_out) * 100
                st.metric("Wealth Building (Target 20%)", f"{wealth_pct:.1f}%")
                
                if wealth_pct < 20:
                    st.warning(f"Optimization Required: Reallocate {(20-wealth_pct):.1f}% to Equity.")
                else: st.success("Behavioral Alignment: Institutional Standard Met.")

    st.sidebar.divider()
    st.sidebar.caption(f"Diana AI Engine v3.1 | 2025 Release")

if __name__ == "__main__":
    main()
