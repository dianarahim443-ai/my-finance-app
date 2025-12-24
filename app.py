import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm, skew, kurtosis
import warnings
import io

# ==========================================================
# 1. SYSTEM INITIALIZATION & SOVEREIGN STYLING
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Diana Sovereign AI | Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Inter:wght@400;900&display=swap');
    
    :root {
        --gold: #FFD700;
        --dark-bg: #050505;
        --card-bg: rgba(20, 20, 20, 0.95);
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1a1a, #050505);
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 40px 60px;
        max-width: 95%;
        background: rgba(10, 10, 10, 0.9);
        border-radius: 30px;
        border: 1px solid #333;
    }

    /* Sovereign Typography */
    .header-text {
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        font-size: 5.5rem !important;
        background: linear-gradient(to bottom, #FFD700, #B8860B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -4px;
        margin-bottom: 0;
    }

    /* Metric Customization */
    div[data-testid="stMetric"] {
        background: rgba(255, 215, 0, 0.03);
        border: 1px solid rgba(255, 215, 0, 0.15);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    }

    /* Button System */
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important;
        font-weight: 800;
        border: none;
        border-radius: 12px;
        padding: 1rem;
        height: 4em;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 30px rgba(255, 215, 0, 0.4);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.02);
        border-radius: 12px 12px 0 0;
        height: 60px;
        color: white;
    }
    .stTabs [aria-selected="true"] { background-color: var(--gold) !important; color: black !important; }
    
    .upload-container {
        border: 2px dashed var(--gold);
        padding: 40px;
        border-radius: 20px;
        background: rgba(255, 215, 0, 0.02);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE ANALYSIS CORE (CLASS-BASED)
# ==========================================================
class SovereignQuant:
    @staticmethod
    def fetch_data(ticker, period="2y", interval="1d"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception: return None

    @staticmethod
    def get_comprehensive_metrics(df, col='Close'):
        rets = df[col].pct_change().dropna()
        mu, sigma = rets.mean(), rets.std()
        
        return {
            "Last": df[col].iloc[-1],
            "Sharpe": (mu / sigma) * np.sqrt(252) if sigma != 0 else 0,
            "VaR_95": norm.ppf(0.05, mu, sigma) * 100,
            "CVaR_95": rets[rets <= norm.ppf(0.05, mu, sigma)].mean() * 100,
            "Volatility": sigma * np.sqrt(252) * 100,
            "Drawdown": ((df[col] / df[col].cummax()) - 1).min() * 100,
            "Skew": skew(rets),
            "Kurtosis": kurtosis(rets),
            "Returns": rets
        }

    @staticmethod
    def run_monte_carlo(last_price, mu, sigma, days=60, simulations=100):
        paths = np.zeros((days, simulations))
        paths[0] = last_price
        for t in range(1, days):
            random_walk = np.random.normal(loc=mu, scale=sigma, size=simulations)
            paths[t] = paths[t-1] * np.exp(random_walk)
        return paths

# ==========================================================
# 3. INTERFACE MODULES (ENHANCED UX)
# ==========================================================

def render_risk_framework():
    st.markdown('<h1 class="header-text">RISK ARCHITECTURE</h1>', unsafe_allow_html=True)
    st.divider()
    
    t_stoch, t_tail, t_stress = st.tabs(["📐 Stochastic Calculus", "📉 Tail Probability", "🔥 Crisis Stress Test"])
    
    with t_stoch:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("Geometric Brownian Motion (GBM)")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.write("""
                The Sovereign Terminal utilizes GBM to model continuous-time price evolution. 
                This equation assumes prices follow a random walk where the log-returns are normally distributed.
            """)
            
        with c2:
            st.subheader("Implementation Logic")
            st.code("""
def simulate_gbm(S0, mu, sigma, T, N):
    dt = T/N
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W)*np.sqrt(dt)
    X = (mu - 0.5*sigma**2)*np.linspace(0, T, N) + sigma*W
    return S0*np.exp(X)
            """, language='python')

    with t_tail:
        st.subheader("Parametric Value at Risk (VaR)")
        st.latex(r"VaR_{\alpha} = \mu + \sigma \Phi^{-1}(\alpha)")
        
        st.info("The Diana Sovereign model focuses on the 5% left-tail event, ensuring institutional capital preservation.")

    with t_stress:
        st.subheader("Historical Scenario Analysis")
        st.write("Simulating current portfolio performance against historical black-swan events:")
        scenarios = {
            "2008 Financial Crisis": "-50.1%",
            "2020 COVID-19 Crash": "-34.0%",
            "1987 Black Monday": "-22.6%",
            "Dot-com Bubble Burst": "-45.0%"
        }
        cols = st.columns(4)
        for i, (name, impact) in enumerate(scenarios.items()):
            cols[i].metric(name, impact, delta="-Risk Critical", delta_color="inverse")

# ----------------------------------------------------------

def render_equity_intelligence():
    st.markdown('<h1 class="header-text">EQUITY INTELLIGENCE</h1>', unsafe_allow_html=True)
    
    # Advanced Universal Search
    with st.sidebar:
        st.header("🔍 Asset Configuration")
        ticker = st.text_input("Global Ticker Symbol:", "NVDA").upper()
        horizon = st.selectbox("Historical Horizon:", ["1y", "2y", "5y", "10y", "max"], index=1)
        st.divider()

    df = SovereignQuant.fetch_data(ticker, period=horizon)
    
    if df is not None:
        m = SovereignQuant.get_comprehensive_metrics(df)
        
        # Primary KPI Ribbon
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current Value", f"${m['Last']:,.2f}")
        k2.metric("Sharpe Efficiency", f"{m['Sharpe']:.2f}")
        k3.metric("Annual Volatility", f"{m['Volatility']:.2f}%")
        k4.metric("Max Drawdown", f"{m['Drawdown']:.2f}%", delta_color="inverse")
        
        # Main Analytics Canvas
        st.plotly_chart(px.area(df, y='Close', title=f"Audit Trajectory: {ticker}", template="plotly_dark").update_traces(line_color="#FFD700", fillcolor="rgba(255, 215, 0, 0.1)"), use_container_width=True)
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(px.histogram(m['Returns'], nbins=100, title="Daily Return Distribution Density", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
        
        with c_right:
            mc_paths = SovereignQuant.run_monte_carlo(m['Last'], m['Returns'].mean(), m['Returns'].std())
            fig_mc = go.Figure()
            for i in range(mc_paths.shape[1]):
                fig_mc.add_trace(go.Scatter(y=mc_paths[:, i], mode='lines', opacity=0.1, line=dict(color='#FFD700')))
            fig_mc.update_layout(template="plotly_dark", title="Monte Carlo: 100 Stochastic Simulations (60-Day Future)", showlegend=False)
            st.plotly_chart(fig_mc, use_container_width=True)
            
        # Statistical Depth
        with st.expander("🔬 Deep Statistical Audit"):
            s1, s2, s3 = st.columns(3)
            s1.write(f"**Skewness:** {m['Skew']:.4f}")
            s2.write(f"**Kurtosis:** {m['Kurtosis']:.4f}")
            s3.write(f"**Conditional VaR (95%):** {m['CVaR_95']:.2f}%")
    else:
        st.error(f"Global Ticker '{ticker}' not found. Please ensure the symbol is valid for Yahoo Finance (e.g., TSLA, BTC-USD, GC=F).")

# ----------------------------------------------------------

def render_wealth_advisor():
    st.markdown('<h1 class="header-text">WEALTH ADVISOR</h1>', unsafe_allow_html=True)
    
    t_upload, t_ledger = st.tabs(["📥 Smart Document Processing", "📝 Sovereign Ledger"])
    
    final_wealth_df = pd.DataFrame()

    with t_upload:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        up_file = st.file_uploader("Drop Bank Statement or Transaction CSV", type="csv")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if up_file:
            try:
                raw_df = pd.read_csv(up_file)
                st.success("File synchronized successfully.")
                st.dataframe(raw_df.head(5), use_container_width=True)
                
                st.warning("Mapping Intelligence: Assign columns to the Diana Sovereign standard.")
                col_map1, col_map2 = st.columns(2)
                
                cat_col = col_map1.selectbox("Transaction Category Column:", raw_df.columns, key="cat_up")
                amt_col = col_map2.selectbox("Transaction Amount Column:", raw_df.columns, key="amt_up")
                
                if st.button("PROCESS DOCUMENT"):
                    final_wealth_df = raw_df[[cat_col, amt_col]].rename(columns={cat_col: 'Category', amt_col: 'Amount'})
            except Exception as e:
                st.error(f"Document Error: {e}")

    with t_ledger:
        st.write("Manual entry for off-chain or private assets:")
        default_ledger = [
            {"Category": "Primary Income", "Amount": 18000.0},
            {"Category": "Real Estate Costs", "Amount": -4500.0},
            {"Category": "Equity Portfolio", "Amount": -5000.0},
            {"Category": "Technology Expenses", "Amount": -1200.0},
            {"Category": "Luxury/Lifestyle", "Amount": -2000.0}
        ]
        final_wealth_df = st.data_editor(pd.DataFrame(default_ledger), num_rows="dynamic", use_container_width=True)

    if not final_wealth_df.empty:
        try:
            final_wealth_df['Amount'] = pd.to_numeric(final_wealth_df['Amount'], errors='coerce').fillna(0)
            total_income = final_wealth_df[final_wealth_df['Amount'] > 0]['Amount'].sum()
            total_outflow = final_wealth_df[final_wealth_df['Amount'] < 0].copy()
            total_outflow['Abs'] = total_outflow['Amount'].abs()
            
            if total_income > 0:
                # Calculate Wealth Creation Velocity
                wealth_keywords = 'Invest|Wealth|Stock|Gold|Crypto|Save|Portfolio'
                investment_total = total_outflow[total_outflow['Category'].str.contains(wealth_keywords, case=False, na=False)]['Abs'].sum()
                velocity = (investment_total / total_income) * 100
                
                w1, w2, w3 = st.columns(3)
                w1.metric("Gross Capital Inflow", f"${total_income:,.0f}")
                w2.metric("Wealth Creation Velocity", f"{velocity:.1f}%")
                w3.metric("Net Surplus", f"${total_income - total_outflow['Abs'].sum():,.0f}")
                
                st.divider()
                c_pie, c_adv = st.columns([1.5, 1])
                with c_pie:
                    st.plotly_chart(px.pie(total_outflow, values='Abs', names='Category', hole=0.5, template="plotly_dark", title="Capital Allocation Structure"), use_container_width=True)
                with c_adv:
                    st.subheader("🕵️ Sovereign Financial Verdict")
                    if velocity < 20:
                        st.error("CRITICAL: Wealth rate is below institutional benchmark (20%). Your capital decay is exceeding creation.")
                    elif 20 <= velocity < 40:
                        st.warning("STABLE: Moderate wealth creation. Suggest increasing ETF exposure.")
                    else:
                        st.success("SOVEREIGN: Exceptional wealth velocity. Capital is being effectively weaponized.")
                    
                    
        except Exception as e:
            st.error(f"Analysis Fault: {e}")

# ----------------------------------------------------------

def render_neural_prediction():
    st.markdown('<h1 class="header-text">NEURAL PREDICTION</h1>', unsafe_allow_html=True)
    
    target_ticker = st.text_input("Neural Forecast Target:", "BTC-USD").upper()
    
    if st.button("INITIATE NEURAL TRAINING"):
        with st.spinner("Executing Facebook Prophet V3 training..."):
            raw_data = SovereignQuant.fetch_data(target_ticker, period="3y")
            if raw_data is not None:
                df_p = raw_data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                
                model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
                model.fit(df_p)
                
                future = model.make_future_dataframe(periods=90)
                forecast = model.predict(future)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Market Reality", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Neural Path", line=dict(color='#FFD700', dash='dash')))
                
                st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Deep Learning Projection: {target_ticker}"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)
            else:
                st.error("Neural engine failed to fetch data for the specified ticker.")

# ==========================================================
# 4. MAIN TERMINAL CONTROLLER
# ==========================================================
def main():
    # Header Ribbon
    col_logo, col_pulse = st.columns([1, 4])
    with col_logo:
        st.sidebar.markdown("# 🏛️ DIANA SOVEREIGN")
        st.sidebar.markdown("`Institutional Access: Verified`")
    
    # Global Pulse Indicators
    st.sidebar.divider()
    nav = st.sidebar.radio("COMMAND CENTER", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    # Live Clock
    st.sidebar.divider()
    st.sidebar.write(f"**Server Time:** {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.write(f"**Market Status:** {'OPEN' if 9 <= datetime.now().hour < 16 else 'CLOSED'}")
    
    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_prediction()
    elif nav == "Wealth Management": render_wealth_advisor()

if __name__ == "__main__":
    main()

# End of Diana Sovereign AI Terminal
