import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime, timedelta

# --- 1. GLOBAL SYSTEM SETTINGS ---
st.set_page_config(
    page_title="Diana Finance AI | Institutional Quant Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BACKGROUND (IMPORTANT!) ---
# Replace 'YOUR_IMAGE_URL_HERE' with the direct URL of your background image
# Example: 'https://i.imgur.com/your_image.jpg' or 'https://raw.githubusercontent.com/your_user/your_repo/main/background.jpg'
BACKGROUND_IMAGE_URL = "https://images.unsplash.com/photo-1579621970563-ef52a5579344?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" 
# This is a sample Unsplash URL. Replace it with your own direct image URL.

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({BACKGROUND_IMAGE_URL});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed; /* Keeps background fixed when scrolling */
    }}
    /* Optional: Style for better readability on top of background */
    .stSidebar, .stAlert, .stInfo, .stSuccess, .stWarning {{
        background-color: rgba(30, 30, 30, 0.85) !important; /* Darker, semi-transparent background for sidebar/alerts */
        color: white !important;
    }}
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stSelectbox label, .stTextInput label, .stFileUploader label, .stButton button, .stTabs > div > button {{
        color: white !important; /* Ensures text is readable */
    }}
    .metric-value {{ /* Specific class for Streamlit metric values */
        color: #FFD700 !important; /* Gold color for metric values */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. ADVANCED QUANTITATIVE ENGINES ---

@st.cache_data(ttl=600)
def get_institutional_data():
    """Live Market Data for Header & Watchlist"""
    tickers = {
        "S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", 
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "EUR/USD": "EURUSD=X",
        "Crude Oil": "CL=F", "10Y Treasury": "^TNX"
    }
    data = {}
    for name, sym in tickers.items():
        try:
            tick = yf.Ticker(sym)
            hist = tick.history(period="2d")
            if len(hist) >= 2:
                price = hist['Close'].iloc[-1]
                change = ((price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                data[name] = {"price": price, "change": change}
        except: continue
    return data

def calculate_portfolio_metrics(equity_curve, strategy_returns):
    """Calculates Sharpe, Sortino, Max Drawdown and Volatility"""
    rf = 0.02 / 252
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_ret = strategy_returns - rf
    std = strategy_returns.std()
    sharpe = (np.sqrt(252) * excess_ret.mean() / std) if std != 0 else 0
    
    # Drawdown calculation
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    # Volatility (Annualized)
    ann_vol = std * np.sqrt(252) * 100
    return total_ret, sharpe, max_dd, ann_vol

def gbm_monte_carlo(last_price, mu, sigma, days=30, n_sims=100):
    """Geometric Brownian Motion Simulation"""
    dt = 1/252
    simulation_df = pd.DataFrame()
    for i in range(n_sims):
        prices = [last_price]
        for _ in range(days):
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * np.random.normal()
            new_price = prices[-1] * np.exp(drift + diffusion)
            prices.append(new_price)
        simulation_df[i] = prices
    return simulation_df

# --- 3. MAIN INTERFACE ---

def main():
    # --- HEADER & MARKET PULSE ---
    st.title("🏛️ Diana Finance: AI Institutional Research")
    st.markdown("##### *Advanced Quantitative Modeling & Behavioral Portfolio Optimization*")
    
    market_data = get_institutional_data()
    cols = st.columns(len(market_data))
    for i, (name, info) in enumerate(market_data.items()):
        cols[i].metric(name, f"{info['price']:,.2f}", f"{info['change']:.2f}%")
    st.divider()

    # --- SIDEBAR NAVIGATION ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1611/1611154.png", width=100) # Replaced with a financial icon
    st.sidebar.title("Core Modules")
    page = st.sidebar.selectbox("Switch Perspective:", 
        ["🏠 Research Documentation", 
         "📈 Equity Intelligence & Backtest", 
         "🔮 AI Forecasting Engine", 
         "💳 Behavioral Wealth Advisor"])

    # --- PAGE 1: DOCUMENTATION ---
    if page == "🏠 Research Documentation":
        st.header("📑 Quantitative Methodology")
        tab_log, tab_math, tab_bench = st.tabs(["Algorithm Architecture", "Mathematical Logic", "AI Innovation"])
        
        with tab_log:
            st.subheader("System Architecture")
            st.write("The platform operates on a three-tier analytical stack:")
            st.markdown("""
            1. **Ingestion Layer:** Real-time data sync via Yahoo Finance & Federal Reserve (FRED).
            2. **Processing Layer:** Prophet-based additive modeling for time-series decomposition.
            3. **Simulation Layer:** Monte Carlo GBM for stochastic risk assessment.
            """)
            
            
        with tab_math:
            st.subheader("Governing Equations")
            st.markdown("The risk engine utilizes the **Geometric Brownian Motion (GBM)** SDE:")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.markdown("Forecasting is based on **Prophet's Decomposable Model**:")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.info("Where $g(t)$ is trend, $s(t)$ is seasonality, and $h(t)$ is holiday effects.")

        with tab_bench:
            st.subheader("Why Diana AI?")
            comparison = {
                "Metric": ["Alpha Generation", "Risk Management", "Data Processing", "Bias"],
                "Traditional Methods": ["Linear Regression", "Static Variance", "Batch Processing", "High Subjectivity"],
                "Diana Finance AI": ["Non-Linear ML", "Stochastic Simulation", "Streaming API", "Data-Driven/Objective"]
            }
            st.table(comparison)

    # --- PAGE 2: EQUITY INTELLIGENCE ---
    elif page == "📈 Equity Intelligence & Backtest":
        st.header("🔍 Quantitative Strategy Backtesting")
        col_in1, col_in2 = st.columns([2, 1])
        with col_in1:
            ticker = st.text_input("Enter Institutional Ticker (e.g. NVDA, TSLA, AAPL):", "NVDA").upper()
        with col_in2:
            period = st.selectbox("Lookback Period:", ["1y", "2y", "5y"], index=0)

        if st.button("Run Institutional Backtest"):
            with st.spinner("Analyzing Market Dynamics..."):
                raw_data = yf.download(ticker, period=period)['Close'].squeeze()
                if not raw_data.empty:
                    # Strategy: Dual-Moving Average Crossover
                    fast_sma = raw_data.rolling(20).mean()
                    slow_sma = raw_data.rolling(50).mean()
                    signals = np.where(fast_sma > slow_sma, 1, 0)
                    
                    # Performance Metrics
                    returns = raw_data.pct_change()
                    strat_returns = returns * pd.Series(signals).shift(1).values
                    equity_curve = 10000 * (1 + strat_returns.fillna(0)).cumprod()
                    
                    ret, sharpe, mdd, vol = calculate_portfolio_metrics(equity_curve, strat_returns.fillna(0))
                    
                    # Display Stats
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Total Alpha", f"{ret:.2f}%")
                    kpi2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    kpi3.metric("Max Drawdown", f"{mdd:.2f}%")
                    kpi4.metric("Ann. Volatility", f"{vol:.2f}%")

                    # Charts
                    st.plotly_chart(px.line(equity_curve, title="Equity Growth ($10k Base)", template="plotly_dark"), use_container_width=True)
                    
                    st.subheader("Stochastic Stress Test (Monte Carlo)")
                    mu = returns.mean() * 252
                    sigma = returns.std() * np.sqrt(252)
                    sims = gbm_monte_carlo(raw_data.iloc[-1], mu, sigma)
                    
                    fig_sim = px.line(sims, template="plotly_dark", title="30-Day Potential Price Paths (GBM)")
                    fig_sim.update_layout(showlegend=False)
                    
                    st.plotly_chart(fig_sim, use_container_width=True)
                else: st.error("Invalid Ticker or Data Unavailable.")

    # --- PAGE 3: AI FORECASTING ---
    elif page == "🔮 AI Forecasting Engine":
        st.header("🔮 Neural Time-Series Prediction")
        asset = st.text_input("Predict Future of (e.g. BTC-USD, MSFT):", "BTC-USD").upper()
        
        if st.button("Initialize Neural Forecast"):
            with st.spinner("Training Prophet Model..."):
                df = yf.download(asset, period="3y").reset_index()
                df_p = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                
                m = Prophet(changepoint_prior_scale=0.1, daily_seasonality=True)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=60)
                forecast = m.predict(future)
                
                # Main Forecast Plot
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical Price", line=dict(color="#00EEFF")))
                fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(color="#FF0077", dash='dash')))
                fig_f.update_layout(template="plotly_dark", title=f"60-Day Forward Forecast: {asset}", hovermode="x unified")
                st.plotly_chart(fig_f, use_container_width=True)
                
                # Insight Panels
                st.subheader("Model Components & Behavioral Cycles")
                comp_fig = plot_components_plotly(m, forecast)
                st.plotly_chart(comp_fig, use_container_width=True)
                
                # Actionable Intelligence
                last_price = df_p['y'].iloc[-1]
                target_60 = forecast['yhat'].iloc[-1]
                growth = ((target_60 - last_price) / last_price) * 100
                
                res1, res2 = st.columns(2)
                res1.info(f"**AI Price Target (60d):** ${target_60:,.2f}")
                if growth > 0:
                    res2.success(f"**Projected Trajectory:** {growth:+.2f}% (BULLISH)")
                else:
                    res2.error(f"**Projected Trajectory:** {growth:+.2f}% (BEARISH)")

    # --- PAGE 4: PERSONAL FINANCE AI ---
    elif page == "💳 Behavioral Wealth Advisor":
        st.header("💳 Intelligent Capital Allocation")
        st.markdown("##### *Behavioral Audit & Wealth Optimization Engine*")
        
        uploaded = st.file_uploader("Upload Transaction Ledger (CSV with 'Description', 'Amount' columns)", type="csv")
        
        if st.button("Analyze Financial Behavior (or use sample data)") or uploaded:
            if not uploaded:
                # Sample Data for presentation
                df = pd.DataFrame({
                    'Description': ['Monthly Salary', 'Rent Payment', 'Amazon Shopping', 'Netflix Subscription', 
                                    'Equity Investment', 'Uber Ride', 'Groceries', 'Credit Card Bill', 'Savings Transfer', 'Bitcoin Purchase'],
                    'Amount': [5000, -2000, -150, -15, -1000, -40, -300, -500, -500, -200],
                })
            else:
                df = pd.read_csv(uploaded)

            # Auto-categorization (can be enhanced with more keywords)
            def auto_cat(d):
                d = str(d).lower()
                if any(x in d for x in ['salary', 'income']): return 'Income'
                if any(x in d for x in ['rent', 'bill', 'electric', 'insurance', 'loan', 'credit card']): return 'Fixed Needs'
                if any(x in d for x in ['invest', 'stock', 'crypto', 'save', 'bond']): return 'Wealth Building'
                if any(x in d for x in ['shop', 'amazon', 'uber', 'restaurant', 'entertainment']): return 'Discretionary'
                return 'Lifestyle' # Default category

            if 'Description' in df.columns and 'Amount' in df.columns:
                df['Category'] = df['Description'].apply(auto_cat)
                df['Amount'] = pd.to_numeric(df['Amount'])
                
                total_income = df[df['Amount'] > 0]['Amount'].sum()
                total_outflow = df[df['Amount'] < 0]['Amount'].abs().sum()
                net_flow = total_income - total_outflow

                st.subheader("Financial Overview")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                col_sum1.metric("Total Income", f"${total_income:,.2f}")
                col_sum2.metric("Total Outflow", f"${total_outflow:,.2f}")
                col_sum3.metric("Net Cash Flow", f"${net_flow:,.2f}")
                
                # Breakdown of Outflow
                outflow_df = df[df['Amount'] < 0].copy()
                outflow_df['Amount'] = outflow_df['Amount'].abs()
                
                if not outflow_df.empty:
                    fig_pie = px.pie(outflow_df, values='Amount', names='Category', hole=0.6, 
                                     title="Capital Outflow Distribution", template="plotly_dark",
                                     color_discrete_sequence=px.colors.sequential.RdBu)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    st.subheader("📊 AI Behavioral Audit (50/30/20 Rule)")
                    cat_sums = outflow_df.groupby('Category')['Amount'].sum()
                    
                    needs_val = cat_sums.get('Fixed Needs', 0)
                    wants_val = cat_sums.get('Discretionary', 0) + cat_sums.get('Lifestyle', 0)
                    wealth_val = cat_sums.get('Wealth Building', 0)
                    
                    total_allocatable = needs_val + wants_val + wealth_val # Sum of only outflow categories
                    
                    if total_allocatable > 0:
                        needs_pct = (needs_val / total_allocatable) * 100
                        wants_pct = (wants_val / total_allocatable) * 100
                        wealth_pct = (wealth_val / total_allocatable) * 100
                    else: needs_pct, wants_pct, wealth_pct = 0, 0, 0

                    
                    audit1, audit2, audit3 = st.columns(3)
                    audit1.metric("Fixed Needs (Target ~50%)", f"{needs_pct:.1f}%")
                    audit2.metric("Wants/Lifestyle (Target ~30%)", f"{wants_pct:.1f}%")
                    audit3.metric("Wealth Building (Target ~20%)", f"{wealth_pct:.1f}%")

                    st.divider()
                    st.subheader("🤖 AI Advisory Report")
                    if net_flow < 0:
                        st.error("🚨 **Urgent Alert:** Your net cash flow is negative. Review expenses immediately to prevent debt accumulation.")
                    elif wealth_pct < 15: # Lowering threshold slightly for more general advice
                        st.warning(f"**Action Required:** Your Wealth Building ({wealth_pct:.1f}%) is below recommended standards. AI suggests reallocating at least 10% from Discretionary spending to investment strategies.")
                    elif needs_pct > 60:
                        st.warning(f"**Expense Overload:** Fixed Needs ({needs_pct:.1f}%) are high. Consider optimizing core expenses to free up capital for wealth building.")
                    else:
                        st.success("Your financial behavior is optimized. Maintain discipline for sustained wealth growth.")
                else:
                    st.info("No outflow transactions to analyze.")
            else:
                st.error("CSV must contain 'Description' and 'Amount' columns.")

    st.sidebar.divider()
    st.sidebar.caption(f"Diana Finance AI Engine v3.0 | 2025 Release")

if __name__ == "__main__":
    main()
