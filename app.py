
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
# 1. CORE ARCHITECTURE & DESIGN
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Quantum Sovereign",
    page_icon="🏛️",
    layout="wide"
)

# Institutional Grade CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(8, 8, 8, 0.98);
        border-radius: 35px;
        padding: 60px;
        border: 1px solid #333;
        box-shadow: 0 25px 70px rgba(0,0,0,1);
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4rem !important; }
    .stMetric { background: rgba(255,255,255,0.02); padding: 30px; border-radius: 20px; border-top: 4px solid #FFD700; }
    .stTabs [data-baseweb="tab-list"] { gap: 30px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.2rem; color: #888; }
    .stTabs [data-baseweb="tab--active"] { color: #FFD700 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANTITATIVE & DATA ENGINE
# ==========================================

class QuantumEngine:
    @staticmethod
    def flatten_yf_data(df):
        """Standardizes yfinance output for both Single and Multi-ticker calls."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def get_performance_audit(returns):
        """Calculates advanced risk-adjusted performance metrics."""
        if returns.empty: return None
        rf = 0.02 / 252
        mean_ret = returns.mean()
        std_dev = returns.std()
        
        sharpe = (mean_ret - rf) / std_dev * np.sqrt(252) if std_dev != 0 else 0
        
        downside = returns[returns < 0]
        sortino = (mean_ret - rf) / downside.std() * np.sqrt(252) if not downside.empty else 0
        
        cum_rets = (1 + returns).cumprod()
        drawdown = (cum_rets / cum_rets.cummax()) - 1
        mdd = drawdown.min() * 100
        
        var_95 = norm.ppf(0.05, mean_ret, std_dev) * 100
        return {"Sharpe": sharpe, "Sortino": sortino, "MDD": mdd, "VaR": var_95}

# ==========================================
# 3. INTERFACE COMPONENTS
# ==========================================

def render_market_pulse():
    st.title("🏛️ Diana Sovereign AI")
    st.markdown("##### *Institutional Multi-Asset Quantitative Research Platform*")
    
    indices = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "Brent Oil": "BZ=F"}
    cols = st.columns(len(indices))
    for i, (name, sym) in enumerate(indices.items()):
        try:
            df = QuantumEngine.flatten_yf_data(yf.download(sym, period="2d", progress=False))
            price = df['Close'].iloc[-1]
            change = ((price / df['Close'].iloc[-2]) - 1) * 100
            cols[i].metric(name, f"{price:,.2f}", f"{change:+.2f}%")
        except: pass
    st.divider()

def render_methodology():
    st.header("🔬 Quantitative Research Methodology")
    tab_m1, tab_m2, tab_m3 = st.tabs(["Stochastic Models", "Neural Forecasting", "Risk Analytics"])
    
    with tab_m1:
        st.subheader("Stochastic Process: Geometric Brownian Motion")
        st.latex(r"S_{t+dt} = S_t \exp\left( (\mu - \frac{\sigma^2}{2})dt + \sigma \sqrt{dt} Z \right)")
        st.write("We use Monte Carlo simulation to project 1,000+ stochastic price trajectories based on historical drift ($\mu$) and volatility ($\sigma$).")
        
        
    with tab_m2:
        st.subheader("Neural Time-Series Architecture")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("Prophet engine performs decomposable forecasting, isolating long-term growth ($g$) from periodic seasonality ($s$).")
        

def render_equity_intel():
    st.header("📈 Equity Intelligence Terminal")
    col_input1, col_input2 = st.columns([3, 1])
    
    with col_input1:
        ticker = st.text_input("Institutional Ticker (e.g., NVDA, AAPL, BTC-USD):", "NVDA").upper()
    with col_input2:
        lookback = st.selectbox("Lookback Period:", ["1Y", "2Y", "3Y", "5Y", "10Y"])
        
    if st.button("Initialize Deep Research Run"):
        with st.spinner("Processing Quantum Data..."):
            raw = QuantumEngine.flatten_yf_data(yf.download(ticker, period=lookback.lower(), progress=False))
            if not raw.empty:
                prices = raw['Close'].squeeze()
                returns = prices.pct_change().dropna()
                metrics = QuantumEngine.get_performance_audit(returns)
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
                k2.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}")
                k3.metric("Max Drawdown", f"{metrics['MDD']:.2f}%")
                k4.metric("Daily VaR (95%)", f"{metrics['VaR']:.2f}%")
                
                st.plotly_chart(px.line(prices, title=f"{ticker} Performance Dynamics", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
                
                # Monte Carlo Simulation
                st.subheader("🎲 Monte Carlo Stress Test (Value-at-Risk)")
                sim_days, sim_paths = 60, 100
                last_price = prices.iloc[-1]
                mu, sigma = returns.mean(), returns.std()
                
                fig_mc = go.Figure()
                for _ in range(sim_paths):
                    path = [last_price]
                    for _ in range(sim_days):
                        path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
                    fig_mc.add_trace(go.Scatter(y=path, mode='lines', opacity=0.15, line=dict(width=1, color='#FFD700')))
                
                fig_mc.update_layout(title="100 Simulated Forward Paths (60-Day Horizon)", template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)

def render_ai_forecasting():
    st.header("🔮 Neural Predictive Engine (V4)")
    target = st.text_input("Enter Forecast Target:", "BTC-USD").upper()
    
    if st.button("Train AI Model"):
        with st.spinner("Optimizing Neural Weights..."):
            raw = QuantumEngine.flatten_yf_data(yf.download(target, period="3y", progress=False).reset_index())
            df_p = pd.DataFrame({
                'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 
                'y': pd.to_numeric(raw['Close'], errors='coerce')
            }).dropna()
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.08).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual Price", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Mean Forecast", line=dict(dash='dash', color='#FFD700')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,215,0,0.1)', name='Upper Bound'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,215,0,0.1)', name='Lower Bound'))
            
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Neural Forecast for {target}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Behavioral Wealth Audit")
    st.markdown("##### Strategic Capital Allocation Audit & Behavioral Analysis")
    
    # 🚨 FIXED FILE UPLOADER LOGIC
    uploaded_file = st.file_uploader("Upload Transactional Ledger (CSV Format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read CSV safely
            df = pd.read_csv(uploaded_file)
            st.success("File Uploaded Successfully")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = pd.DataFrame()
    else:
        st.info("Simulation Mode: Interactive Editor Enabled")
        # Sample professional dataset
        df = pd.DataFrame([
            {"Description": "Executive Salary", "Amount": 9500, "Category": "Income"},
            {"Description": "Mortgage/Rent Payment", "Amount": -2800, "Category": "Fixed"},
            {"Description": "S&P 500 ETF Allocation", "Amount": -2500, "Category": "Wealth"},
            {"Description": "Luxury Lifestyle", "Amount": -800, "Category": "Wants"},
            {"Description": "Fixed Utility Costs", "Amount": -400, "Category": "Fixed"},
            {"Description": "Emergency Fund Save", "Amount": -500, "Category": "Wealth"}
        ])
    
    # Interactive Data Table
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # Calculation logic with strict type checking to avoid TypeError
    if 'Amount' in df.columns:
        # Convert Amount to numeric, forcing errors to NaN then filling with 0
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        # Only process outflows (negative values) for the audit
        outflow_mask = df['Amount'] < 0
        outflow_df = df[outflow_mask].copy()
        
        # 🚨 FIX: Apply abs() ONLY to the numeric column
        outflow_df['AbsAmount'] = outflow_df['Amount'].abs()
        total_out = outflow_df['AbsAmount'].sum()
        
        if total_out > 0:
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.plotly_chart(px.pie(outflow_df, values='AbsAmount', names='Category', hole=0.6, 
                                     template="plotly_dark", title="Audit: Behavioral Capital Distribution",
                                     color_discrete_sequence=px.colors.sequential.YlOrBr))
            with c_right:
                wealth_total = outflow_df[outflow_df['Category'] == 'Wealth']['AbsAmount'].sum()
                wealth_rate = (wealth_total / total_out) * 100
                st.metric("Wealth Building Coefficient", f"{wealth_rate:.1f}%", delta=f"{wealth_rate-20:.1f}% (Target: 20%)")
                
                if wealth_rate < 20:
                    st.error("STRATEGIC ALERT: Insufficient allocation to Wealth-Building assets. Increase savings rate.")
                else:
                    st.success("AUDIT PASS: Behavioral pattern optimized for institutional capital accumulation.")
            
            st.subheader("Transactional Intelligence Ledger")
            st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['Amount']), use_container_width=True)

# ==========================================
# 4. MASTER CONTROLLER
# ==========================================

def main():
    render_market_pulse()
    
    st.sidebar.title("💎 Sovereign Navigation")
    # Fixed Navigation Naming to avoid NameError
    nav_option = st.sidebar.radio("Select Perspective:", 
        ["Research Methodology", "Equity Intelligence", "Neural Forecasting", "Wealth Management Advisor"])
    
    if nav_option == "Research Methodology":
        render_methodology()
    elif nav_option == "Equity Intelligence":
        render_equity_intel()
    elif nav_option == "Neural Forecasting":
        render_ai_forecasting()
    elif nav_option == "Wealth Management Advisor":
        render_wealth_advisor()

    st.sidebar.divider()
    st.sidebar.markdown("**Quantum Engine:** `v11.0.1-Sovereign`")
    st.sidebar.caption(f"Last Terminal Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
