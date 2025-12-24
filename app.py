import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime

# --- 1. SYSTEM & INTERFACE CONFIGURATION ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# Advanced CSS for high-end FinTech UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.92);
        border-radius: 25px;
        padding: 50px;
        border: 1px solid #444;
        box-shadow: 0 10px 40px rgba(0,0,0,0.7);
    }
    h1, h2, h3 { color: #FFD700 !important; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-left: 4px solid #FFD700; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 4px; padding: 10px 20px; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. QUANTITATIVE & ANALYTICAL ENGINES ---

@st.cache_data(ttl=3600)
def get_global_pulse():
    """Live Market Watchlist with MultiIndex Fix"""
    tickers = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period="2d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            price = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2])
            change = ((price - prev) / prev) * 100
            data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def run_institutional_backtest(data):
    """Trend-Following Momentum Strategy"""
    fast_sma = data.rolling(window=20).mean()
    slow_sma = data.rolling(window=50).mean()
    signal = np.where(fast_sma > slow_sma, 1, 0)
    returns = data.pct_change()
    strat_returns = returns * pd.Series(signal).shift(1).values
    equity_curve = 10000 * (1 + strat_returns.fillna(0)).cumprod()
    
    # Statistical Risk Metrics
    rf = 0.02 / 252 
    excess = strat_returns.fillna(0) - rf
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else 0
    max_dd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100
    return equity_curve, sharpe, max_dd

def run_monte_carlo(last_price, mu, sigma, days=30, sims=100):
    """Stochastic Simulation using Geometric Brownian Motion"""
    simulation_df = pd.DataFrame()
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
        simulation_df[i] = prices
    return simulation_df

# --- 3. CORE APPLICATION INTERFACE ---

def main():
    st.title("🏛️ Diana Finance: Institutional AI Research")
    st.markdown("##### *Advanced Quantitative Modeling & Neural Forecasting Platform*")
    
    # Global Header Pulse
    pulse = get_global_pulse()
    p_cols = st.columns(len(pulse))
    for i, (name, val) in enumerate(pulse.items()):
        p_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    # Sidebar Navigation
    st.sidebar.title("🔬 Research Core")
    page = st.sidebar.selectbox("Perspective:", 
        ["📚 Methodology & Proofs", 
         "📈 Equity Intelligence", 
         "🔮 AI Forecasting Engine", 
         "💳 Wealth Optimization"])

    # --- PAGE 1: METHODOLOGY ---
    if page == "📚 Methodology & Proofs":
        st.header("📑 Quantitative Methodology")
        tab1, tab2, tab3 = st.tabs(["Mathematical Logic", "AI Architecture", "Project Scope"])
        
        with tab1:
            st.subheader("Stochastic Process: GBM")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.markdown("""
            The core risk engine utilizes **Geometric Brownian Motion (GBM)** to simulate future price trajectories.
            - **Drift ($\mu$):** Represents deterministic trend based on historical mean.
            - **Diffusion ($\sigma$):** Represents market volatility and uncertainty.
            - **Wiener Process ($dW_t$):** Stochastic component modeled via Brownian motion.
            """)
            
            
        with tab2:
            st.subheader("Decomposable Neural Forecasting")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown("""
            Our AI uses **Prophet Architecture** to decompose time-series into:
            1. **Trend ($g$):** Non-periodic growth logic.
            2. **Seasonality ($s$):** Periodic changes (Weekly, Yearly).
            3. **Holidays ($h$):** Irregular market shocks.
            """)
            

    # --- PAGE 2: EQUITY INTELLIGENCE ---
    elif page == "📈 Equity Intelligence":
        st.header("🔍 Backtesting & Alpha Generation")
        ticker = st.text_input("Institutional Ticker (e.g., NVDA, AAPL, BTC-USD):", "NVDA").upper()
        
        if st.button("Execute Quantitative Run"):
            with st.spinner("Processing Market Dynamics..."):
                raw = yf.download(ticker, period="2y", progress=False)
                if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                
                prices = raw['Close'].squeeze()
                equity, sharpe, mdd = run_institutional_backtest(prices)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Strategy Return", f"{((equity.iloc[-1]/10000)-1)*100:.2f}%")
                c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                c3.metric("Max Drawdown", f"{mdd:.2f}%")
                
                st.plotly_chart(px.line(equity, title="Equity Growth ($10k Initial)", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
                
                st.subheader("Stochastic Stress Test (Monte Carlo)")
                returns = prices.pct_change().dropna()
                sims_df = run_monte_carlo(prices.iloc[-1], returns.mean(), returns.std())
                fig_mc = px.line(sims_df, template="plotly_dark", title="100 Simulated 30-Day Paths")
                fig_mc.update_layout(showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)

    # --- PAGE 3: AI FORECASTING ---
    elif page == "🔮 AI Forecasting Engine":
        st.header("🔮 Neural Time-Series Prediction")
        asset = st.text_input("Forecast Asset (e.g., BTC-USD):", "BTC-USD").upper()
        
        if st.button("Train AI Model"):
            with st.spinner("Deploying Prophet Engine..."):
                raw_data = yf.download(asset, period="3y", progress=False).reset_index()
                if isinstance(raw_data.columns, pd.MultiIndex): raw_data.columns = raw_data.columns.get_level_values(0)
                
                # Robust Data Preprocessing
                df_p = pd.DataFrame()
                df_p['ds'] = pd.to_datetime(raw_data['Date']).dt.tz_localize(None)
                df_p['y'] = pd.to_numeric(raw_data['Close'], errors='coerce')
                df_p = df_p.dropna()

                m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.08)
                m.fit(df_p)
                
                forecast = m.predict(m.make_future_dataframe(periods=60))
                
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical", line=dict(color='#00F2FF')))
                fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Forecast", line=dict(dash='dash', color='#FFD700')))
                fig_forecast.update_layout(template="plotly_dark", title=f"60-Day Forward Outlook: {asset}")
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader("Market Component Breakdown")
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

    # --- PAGE 4: WEALTH OPTIMIZATION ---
    elif page == "💳 Wealth Optimization":
        st.header("💳 AI Behavioral Portfolio Advisor")
        
        # Institutional Transaction Ledger Sample
        df_ledger = pd.DataFrame({
            'Description': ['Monthly Salary', 'Rent Outflow', 'ETF Investment', 'Amazon Purchase', 'Uber/Logistics', 'Fixed Bills', 'Gold/Safe Haven'],
            'Amount': [7000, -2000, -1400, -300, -150, -600, -500],
            'Category': ['Income', 'Fixed', 'Wealth Building', 'Discretionary', 'Discretionary', 'Fixed', 'Wealth Building']
        })
        
        st.subheader("Behavioral Capital Flow")
        st.table(df_ledger)
        
        outflow = df_ledger[df_ledger['Amount'] < 0].copy()
        outflow['Amount'] = outflow['Amount'].abs()
        total_out = outflow['Amount'].sum()
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig_p = px.pie(outflow, values='Amount', names='Category', hole=0.6, 
                           template="plotly_dark", title="Audit: 50/30/20 Capital Allocation Model",
                           color_discrete_sequence=px.colors.sequential.YlOrRd)
            st.plotly_chart(fig_p, use_container_width=True)
            
        with c2:
            st.subheader("Institutional Audit")
            wealth_val = outflow[outflow['Category'] == 'Wealth Building']['Amount'].sum()
            w_pct = (wealth_val / total_out) * 100
            st.metric("Wealth Building Allocation", f"{w_pct:.1f}%", delta=f"{w_pct-20:.1f}% (Target: 20%)")
            
            if w_pct < 20:
                st.error("INSUFFICIENT CAPITAL ALLOCATION: Increase Asset Acquisition.")
            else:
                st.success("INSTITUTIONAL STANDARD MET: Portfolio behavior is optimized.")

    st.sidebar.divider()
    st.sidebar.caption("Diana AI Framework v5.0 | High-Fidelity Research")

if __name__ == "__main__":
    main()
