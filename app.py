import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import io

# ==========================================
# 1. GLOBAL SYSTEM CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Sovereign Magnum Opus",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Premium CSS - Glassmorphism, Gold Accents & Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.96), rgba(0,0,0,0.96)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    
    .main .block-container {
        background: rgba(10, 10, 10, 0.99);
        border-radius: 40px;
        padding: 50px 70px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 40px 120px rgba(0,0,0,1);
    }
    
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.8rem !important; letter-spacing: -3px; line-height: 1; margin-bottom: 20px; }
    h2, h3 { color: #FFFFFF !important; font-weight: 700; border-left: 8px solid #FFD700; padding-left: 20px; margin-top: 45px; margin-bottom: 25px; }
    
    .stMetric { 
        background: rgba(255,255,255,0.02); 
        padding: 30px; 
        border-radius: 20px; 
        border-top: 5px solid #FFD700; 
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
    }
    .stMetric:hover { transform: translateY(-10px); background: rgba(255,215,0,0.05); box-shadow: 0 15px 30px rgba(0,0,0,0.5); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 40px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.3rem; color: #777; font-weight: 600; transition: 0.3s; }
    .stTabs [data-baseweb="tab--active"] { color: #FFD700 !important; border-bottom-color: #FFD700 !important; }
    
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 900; border: none; padding: 15px 40px; border-radius: 12px;
        text-transform: uppercase; letter-spacing: 1px; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. ADVANCED QUANTITATIVE ENGINE
# ==========================================

class SovereignEngine:
    @staticmethod
    def format_yf(df):
        """Standardizes MultiIndex columns for seamless processing."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def audit_performance(returns):
        """Institutional Risk-Adjusted Return Framework."""
        if returns.empty: return None
        rf = 0.045 / 252 # 4.5% Risk-Free Rate (Annualized)
        mu, sigma = returns.mean(), returns.std()
        
        # Performance Ratios
        sharpe = (mu - rf) / sigma * np.sqrt(252) if sigma != 0 else 0
        downside_rets = returns[returns < 0]
        sortino = (mu - rf) / downside_rets.std() * np.sqrt(252) if not downside_rets.empty else 0
        
        # Risk Analytics
        cum_rets = (1 + returns).cumprod()
        peak = cum_rets.cummax()
        drawdown = (cum_rets - peak) / peak
        mdd = drawdown.min() * 100
        
        # Parametric VaR (95%)
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        vol_ann = sigma * np.sqrt(252) * 100
        
        return {"Sharpe": sharpe, "Sortino": sortino, "MDD": mdd, "VaR": var_95, "Vol": vol_ann}

# ==========================================
# 3. INTERFACE PERSPECTIVES
# ==========================================

def render_global_pulse():
    st.title("🏛️ Diana Sovereign")
    st.markdown("##### *Institutional-Grade Quantitative Research Terminal & Capital Management*")
    
    # Real-Time Market Tickers
    indices = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold Spot": "GC=F", "Bitcoin": "BTC-USD", "10Y Treasury": "^TNX"}
    m_cols = st.columns(len(indices))
    for i, (name, sym) in enumerate(indices.items()):
        try:
            d = SovereignEngine.format_yf(yf.download(sym, period="2d", progress=False))
            p, c = d['Close'].iloc[-1], ((d['Close'].iloc[-1]/d['Close'].iloc[-2])-1)*100
            m_cols[i].metric(name, f"{p:,.2f}", f"{c:+.2f}%")
        except: pass
    st.divider()

def render_risk_framework():
    st.header("🔬 Strategic Risk Framework & Academic Methodology")
    
    t1, t2, t3 = st.tabs(["Stochastic Models", "Neural Forecasting", "Portfolio Analytics"])
    
    with t1:
        st.subheader("I. Geometric Brownian Motion (GBM) Theory")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("We model asset price dynamics using the Stochastic Differential Equation above. This allows us to simulate thousands of potential future paths using Wiener Process ($W_t$).")
        
        st.markdown("""
        **Key Components:**
        * **Drift ($\mu$):** Expected rate of return.
        * **Diffusion ($\sigma$):** Market volatility coefficient.
        """)
        
    with t2:
        st.subheader("II. Neural Time-Series Decomposition")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("Using the Prophet Architecture, we decompose market signals into Trend ($g$), Multi-period Seasonality ($s$), and Residual Noise ($\epsilon$).")
        

    with t3:
        st.subheader("III. Institutional Risk Ratios")
        st.write("We employ a three-tier risk audit for every analyzed asset:")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**1. Value at Risk (VaR):**")
            st.latex(r"VaR_{\alpha} = \mu + \sigma \cdot \Phi^{-1}(\alpha)")
            st.write("Calculates the potential maximum loss at a 95% confidence level.")
            
        with col_r2:
            st.markdown("**2. Sortino Ratio:**")
            st.latex(r"Sortino = \frac{R_p - R_f}{\sigma_{downside}}")
            st.write("Focuses only on 'bad' volatility (downside) rather than total volatility.")

def render_equity_intel():
    st.header("📈 Equity Intelligence & Quantitative Audit")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker = st.text_input("Enter Institutional Ticker Symbol (e.g., NVDA, BTC-USD, GC=F):", "NVDA").upper()
    with c2:
        period = st.selectbox("Historical Lookback:", ["1Y", "2Y", "5Y", "Max"])
    
    if st.button("Initialize Deep Quantitative Run"):
        with st.spinner("Processing High-Frequency Market Data..."):
            raw = SovereignEngine.format_yf(yf.download(ticker, period=period.lower(), progress=False))
            if not raw.empty:
                prices = raw['Close'].squeeze()
                returns = prices.pct_change().dropna()
                m = SovereignEngine.audit_performance(returns)
                
                # KPIs Row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
                k2.metric("Ann. Volatility", f"{m['Vol']:.1f}%")
                k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
                k4.metric("Daily VaR (95%)", f"{m['VaR']:.2f}%")
                
                # Dynamic Performance Chart
                fig = px.line(prices, title=f"{ticker} Historical Trajectory", template="plotly_dark")
                fig.update_traces(line_color='#FFD700', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
                
                # Monte Carlo Stress Test
                st.subheader("🎲 Monte Carlo Stochastic Stress Test")
                sim_days, sim_paths = 60, 150
                last_p, mu, sigma = prices.iloc[-1], returns.mean(), returns.std()
                
                fig_mc = go.Figure()
                for _ in range(sim_paths):
                    path = [last_p]
                    for _ in range(sim_days):
                        path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
                    fig_mc.add_trace(go.Scatter(y=path, mode='lines', opacity=0.1, line=dict(color='#FFD700', width=1)))
                
                fig_mc.update_layout(title="150 Forward Paths - 60D Horizon (Geometric Brownian Motion)", template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)

def render_ai_forecast():
    st.header("🔮 Neural Predictive Engine (Prophet V5)")
    target = st.text_input("Forecast Asset Target:", "BTC-USD").upper()
    
    if st.button("Deploy AI Forecast Model"):
        with st.spinner("Optimizing Neural Parameters..."):
            raw = SovereignEngine.format_yf(yf.download(target, period="3y", progress=False).reset_index())
            df_p = pd.DataFrame({
                'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 
                'y': pd.to_numeric(raw['Close'], errors='coerce')
            }).dropna()
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.08).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            # Predictive Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual Price", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Mean Prediction", line=dict(dash='dash', color='#FFD700')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,215,0,0.1)', name='Conf. Upper'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,215,0,0.1)', name='Conf. Lower'))
            
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Forward Forecast: {target}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Behavioral Wealth Management")
    st.markdown("##### *Capital Allocation Strategy & Institutional Behavioral Audit*")
    
    up = st.file_uploader("Upload Transaction Ledger (CSV - Format: Description, Amount, Category)", type="csv")
    
    if up:
        df = pd.read_csv(up)
    else:
        st.info("No File Detected. Interactive Simulation Active.")
        df = pd.DataFrame([
            {"Description": "Institutional Income", "Amount": 15000, "Category": "Income"},
            {"Description": "Mortgage/Rent Outflow", "Amount": -4000, "Category": "Fixed"},
            {"Description": "S&P 500 Equity Buy", "Amount": -3500, "Category": "Wealth"},
            {"Description": "Discretionary Lifestyle", "Amount": -1500, "Category": "Wants"},
            {"Description": "Insurance & Fixed Costs", "Amount": -800, "Category": "Fixed"},
            {"Description": "Venture Capital / Crypto", "Amount": -1200, "Category": "Wealth"}
        ])

    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        outflows = df[df['Amount'] < 0].copy()
        outflows['Abs'] = outflows['Amount'].abs()
        
        if not outflows.empty:
            c_p1, c_p2 = st.columns([1.5, 1])
            with c_p1:
                # 📊 PIE CHART (DOUGHNUT STYLE)
                st.subheader("Capital Outflow Distribution Audit")
                fig_pie = px.pie(
                    outflows, 
                    values='Abs', 
                    names='Category', 
                    hole=0.6, 
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.sequential.YlOrBr
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with c_p2:
                # 📈 WEALTH EFFICIENCY KPI
                st.subheader("Efficiency Metrics")
                w_sum = outflows[outflows['Category'] == 'Wealth']['Abs'].sum()
                w_rate = (w_sum / outflows['Abs'].sum()) * 100
                st.metric("Wealth Building Coefficient", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}% (Target: 20%)")
                
                st.markdown("---")
                if w_rate < 20: 
                    st.error("STRATEGIC ALERT: Under-allocated for capital accumulation. Rebalance from 'Wants'.")
                else: 
                    st.success("AUDIT PASS: High-velocity capital accumulation detected.")

            st.divider()
            st.subheader("Transactional Ledger Intelligence")
            st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['Amount']), use_container_width=True)

# ==========================================
# 4. MASTER NAVIGATOR (MAIN)
# ==========================================

def main():
    render_global_pulse()
    
    # Professional Sidebar Navigation
    st.sidebar.title("💎 Sovereign Menu")
    nav = st.sidebar.radio("Select Research Domain:", 
        ["Strategic Risk Framework", "Equity Intelligence", "Neural AI Forecasting", "Wealth Management Advisor"])
    
    # State Router
    if nav == "Strategic Risk Framework":
        render_risk_framework()
    elif nav == "Equity Intelligence":
        render_equity_intel()
    elif nav == "Neural AI Forecasting":
        render_ai_forecast()
    elif nav == "Wealth Management Advisor":
        render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.markdown("**Terminal Build:** `v16.0.0-Magnum`")
    st.sidebar.caption(f"Sync Time: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.info("Operational: Institutional Data Pipelines Active")

if __name__ == "__main__":
    main()
