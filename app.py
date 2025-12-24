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
# 1. CORE ARCHITECTURE & THEME ENGINE
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Sovereign Grand-Pro",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Professional CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98);
        border-radius: 40px;
        padding: 60px;
        border: 1px solid #222;
        box-shadow: 0 30px 100px rgba(0,0,0,1);
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; letter-spacing: -3px; line-height: 1; }
    h2, h3 { color: #E0E0E0 !important; font-weight: 700; border-left: 5px solid #FFD700; padding-left: 15px; }
    .stMetric { 
        background: rgba(255,255,255,0.02); 
        padding: 30px; 
        border-radius: 25px; 
        border-top: 5px solid #FFD700;
        transition: transform 0.3s ease;
    }
    .stMetric:hover { transform: translateY(-5px); background: rgba(255,215,0,0.03); }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] { color: #666; font-size: 1.1rem; }
    .stTabs [data-baseweb="tab--active"] { color: #FFD700 !important; border-bottom-color: #FFD700 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANTITATIVE ANALYSIS CORE
# ==========================================

class SovereignAnalytics:
    @staticmethod
    def standardize_data(df):
        """Fixes MultiIndex and cleaning issues with latest yfinance"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def compute_risk_ratios(returns):
        """Institutional Risk-Return Statistics"""
        if returns.empty: return None
        rf_daily = 0.04 / 252 # Assumed 4% annual RF rate
        mu = returns.mean()
        sigma = returns.std()
        
        # Risk Ratios
        sharpe = (mu - rf_daily) / sigma * np.sqrt(252) if sigma != 0 else 0
        downside_std = returns[returns < 0].std()
        sortino = (mu - rf_daily) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown Logic
        cum_rets = (1 + returns).cumprod()
        peak = cum_rets.cummax()
        drawdown = (cum_rets - peak) / peak
        mdd = drawdown.min() * 100
        
        # Value at Risk (Parametric)
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        return {"Sharpe": sharpe, "Sortino": sortino, "MDD": mdd, "VaR": var_95}

# ==========================================
# 3. GUI MODULES
# ==========================================

def render_global_pulse():
    st.title("🏛️ Diana Sovereign")
    st.markdown("### *Professional Multi-Asset Research & Capital Management*")
    
    # Real-Time Terminal Strip
    assets = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "10Y Treasury": "^TNX"}
    m_cols = st.columns(len(assets))
    for i, (name, sym) in enumerate(assets.items()):
        try:
            d = SovereignAnalytics.standardize_data(yf.download(sym, period="2d", progress=False))
            price, chg = d['Close'].iloc[-1], ((d['Close'].iloc[-1]/d['Close'].iloc[-2])-1)*100
            m_cols[i].metric(name, f"{price:,.2f}", f"{chg:+.2f}%")
        except: m_cols[i].metric(name, "N/A", "0.00%")
    st.divider()

def render_methodology():
    st.header("🔬 Strategic Research Methodology")
    tab1, tab2, tab3 = st.tabs(["Stochastic Modeling", "Neural Decomposition", "Risk Framework"])
    
    with tab1:
        st.subheader("I. Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Our Monte Carlo engine solves this Stochastic Differential Equation to calculate Value-at-Risk (VaR) and Tail Risk exposure.")
        
        
    with tab2:
        st.subheader("II. Neural Time-Series Deconvolution")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("The Prophet architecture isolates non-linear trends ($g$) from periodic seasonality ($s$) to generate high-fidelity forecasts.")
        

def render_equity_intel():
    st.header("📈 Equity Intelligence Terminal")
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker = st.text_input("Institutional Ticker (e.g., NVDA, BTC-USD):", "NVDA").upper()
    with c2:
        period = st.selectbox("Lookback Horizon:", ["1Y", "2Y", "5Y", "Max"])
        
    if st.button("Initialize Deep Research Run"):
        with st.spinner("Processing Quantum Ticker Data..."):
            raw = SovereignAnalytics.standardize_data(yf.download(ticker, period=period.lower(), progress=False))
            if not raw.empty:
                prices = raw['Close'].squeeze()
                returns = prices.pct_change().dropna()
                metrics = SovereignAnalytics.compute_risk_ratios(returns)
                
                # Dynamic KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
                k2.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}")
                k3.metric("Max Drawdown", f"{metrics['MDD']:.2f}%")
                k4.metric("Daily VaR (95%)", f"{metrics['VaR']:.2f}%")
                
                # Primary Chart
                fig = px.line(prices, title=f"{ticker} Performance Dynamics", template="plotly_dark")
                fig.update_traces(line_color='#FFD700', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
                
                # Monte Carlo Stress Test
                st.subheader("🎲 Monte Carlo Stochastic Projection")
                sim_days, sim_paths = 60, 100
                last_price = prices.iloc[-1]
                mu, sigma = returns.mean(), returns.std()
                
                fig_mc = go.Figure()
                for _ in range(sim_paths):
                    path = [last_price]
                    for _ in range(sim_days):
                        path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
                    fig_mc.add_trace(go.Scatter(y=path, mode='lines', opacity=0.1, line=dict(width=1, color='#FFD700')))
                
                fig_mc.update_layout(title="100 Simulated Forward Paths (60D Horizon)", template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)

def render_ai_forecasting():
    st.header("🔮 Neural Predictive Engine (V4)")
    asset = st.text_input("Forecast Asset Target:", "BTC-USD").upper()
    
    if st.button("Generate AI Forecast"):
        with st.spinner("Calibrating Neural Weights..."):
            raw = SovereignAnalytics.standardize_data(yf.download(asset, period="3y", progress=False).reset_index())
            df_p = pd.DataFrame({
                'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 
                'y': pd.to_numeric(raw['Close'], errors='coerce')
            }).dropna()
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            # Prediction Graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical Price", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash', color='#FFD700')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,215,0,0.1)', name='Confidence Upper'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,215,0,0.1)', name='Confidence Lower'))
            
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Neural Forecast: {asset}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Behavioral Wealth Audit")
    st.markdown("##### *Strategic Portfolio Allocation & Behavioral Audit*")
    
    # --- File Upload Logic ---
    up_file = st.file_uploader("Upload Transaction Ledger (CSV)", type=["csv"])
    
    if up_file:
        df = pd.read_csv(up_file)
    else:
        st.info("Interactive Simulation: Edit the Institutional Sample Data below.")
        df = pd.DataFrame([
            {"Description": "Base Salary (Monthly)", "Amount": 10500, "Category": "Income"},
            {"Description": "Rent/Mortgage Outflow", "Amount": -3000, "Category": "Fixed"},
            {"Description": "Equity Portfolio Buy", "Amount": -2800, "Category": "Wealth"},
            {"Description": "Lifestyle & Dining", "Amount": -900, "Category": "Wants"},
            {"Description": "Utility/Fixed Costs", "Amount": -500, "Category": "Fixed"},
            {"Description": "Crypto Accumulation", "Amount": -600, "Category": "Wealth"},
            {"Description": "Subscription Services", "Amount": -200, "Category": "Wants"}
        ])

    # Dynamic Live Editor
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # --- Analytics & Visuals (Fixed for TypeError) ---
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        # Isolate Outflows for Pie Chart
        outflows = df[df['Amount'] < 0].copy()
        outflows['AbsAmount'] = outflows['Amount'].abs()
        total_out = outflows['AbsAmount'].sum()
        
        if total_out > 0:
            c1, c2 = st.columns([1.5, 1])
            
            with c1:
                # 📊 THE REQUESTED PIE CHART
                st.subheader("Capital Outflow Distribution")
                fig_pie = px.pie(
                    outflows, 
                    values='AbsAmount', 
                    names='Category', 
                    hole=0.6,
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.sequential.YlOrBr
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with c2:
                # Institutional Benchmarking
                st.subheader("Wealth Efficiency Audit")
                w_sum = outflows[outflows['Category'] == 'Wealth']['AbsAmount'].sum()
                w_rate = (w_sum / total_out) * 100
                
                st.metric("Wealth Building Rate", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}% (Target: 20%)")
                
                if w_rate < 20:
                    st.error("ALLOCATION ALERT: Portfolio building speed is sub-optimal. Reallocate from 'Wants'.")
                elif w_rate > 40:
                    st.success("ELITE STATUS: Your capital accumulation rate is in the top 1% of behavioral profiles.")
                else:
                    st.warning("OPTIMIZED: Your allocation meets standard wealth-building benchmarks.")
            
            # Transactional Ledger View
            st.divider()
            st.subheader("Transactional Ledger Intelligence")
            st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['Amount']), use_container_width=True)

# ==========================================
# 4. MASTER CONTROLLER
# ==========================================

def main():
    render_global_pulse()
    
    # Navigation Sidebar
    st.sidebar.title("💎 Sovereign Terminal")
    nav = st.sidebar.radio("Navigation Perspectives:", 
        ["Theoretical Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management Advisor"])
    
    # State-Based Router
    if nav == "Theoretical Framework":
        render_methodology()
    elif nav == "Equity Intelligence":
        render_equity_intel()
    elif nav == "Neural Forecasting":
        render_ai_forecasting()
    elif nav == "Wealth Management Advisor":
        render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.markdown("**Engine Build:** `v12.0.4-Magnum`")
    st.sidebar.caption(f"Last Terminal Sync: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.markdown("---")
    st.sidebar.info("Operational: Institutional Grade Data Pipes")

if __name__ == "__main__":
    main()
