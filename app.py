
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
import warnings

# ==========================================================
# 1. SYSTEM INITIALIZATION & GLOBAL THEME
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Diana Sovereign AI | Institutional Magnum Opus",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Premium CSS with Glassmorphism and Institutional Gold Accents
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: rgba(10, 10, 10, 0.98);
        border-radius: 40px;
        padding: 60px 80px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 40px 150px rgba(0,0,0,1);
    }
    
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 5rem !important; letter-spacing: -3px; margin-bottom: 10px; }
    h2, h3 { color: #FFD700 !important; font-weight: 700; border-left: 10px solid #FFD700; padding-left: 25px; margin-top: 50px; }
    
    .stMetric { 
        background: rgba(255,255,255,0.03); 
        padding: 35px; 
        border-radius: 25px; 
        border-top: 6px solid #FFD700; 
        transition: 0.4s;
    }
    .stMetric:hover { transform: translateY(-10px); background: rgba(255,215,0,0.06); }
    
    .agent-verdict {
        background: rgba(255, 215, 0, 0.05);
        border: 2px dashed #FFD700;
        border-radius: 25px;
        padding: 40px;
        margin: 30px 0;
    }

    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 900; border: none; padding: 20px; border-radius: 15px;
        text-transform: uppercase; letter-spacing: 2px; width: 100%; transition: 0.5s;
    }
    .stButton>button:hover { box-shadow: 0 0 30px rgba(255,215,0,0.4); transform: scale(1.02); }
    
    .sidebar-text { font-size: 0.9rem; color: #888; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE ANALYTICS CORE (SOVEREIGN ENGINE)
# ==========================================================

class SovereignEngine:
    @staticmethod
    def standardize_data(df):
        """Ensures MultiIndex and data types are aligned for processing."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def calculate_risk_ratios(returns):
        """Calculates institutional-grade risk and performance metrics."""
        if returns.empty: return None
        
        # Risk-Free Rate Assumption (4.5% Annualized)
        rf_daily = 0.045 / 252
        mu, sigma = returns.mean(), returns.std()
        
        # Performance Metrics
        sharpe = (mu - rf_daily) / sigma * np.sqrt(252) if sigma > 0 else 0
        downside_sigma = returns[returns < 0].std()
        sortino = (mu - rf_daily) / downside_sigma * np.sqrt(252) if downside_sigma > 0 else 0
        
        # Risk Metrics
        cum_rets = (1 + returns).cumprod()
        peak = cum_rets.cummax()
        drawdown = (cum_rets - peak) / peak
        mdd = drawdown.min() * 100
        
        # Parametric Value at Risk (95% Confidence)
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        vol_ann = sigma * np.sqrt(252) * 100
        
        return {
            "Sharpe": sharpe, "Sortino": sortino, "MDD": mdd, 
            "VaR": var_95, "Vol": vol_ann, "Mean_Daily": mu * 100
        }

# ==========================================================
# 3. INTERFACE PERSPECTIVES (MODULES)
# ==========================================================

def render_market_pulse():
    """Global Real-Time Market Watch."""
    st.title("🏛️ Diana Sovereign")
    st.markdown("##### *Strategic Institutional Multi-Asset Intelligence Terminal*")
    
    watch_list = {
        "S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", 
        "Gold Spot": "GC=F", "Bitcoin": "BTC-USD", 
        "10Y Yield": "^TNX", "Crude Oil": "CL=F"
    }
    
    cols = st.columns(len(watch_list))
    for i, (name, sym) in enumerate(watch_list.items()):
        try:
            data = SovereignEngine.standardize_data(yf.download(sym, period="2d", progress=False))
            price = data['Close'].iloc[-1]
            change = (data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100
            cols[i].metric(name, f"{price:,.2f}", f"{change:+.2f}%")
        except:
            cols[i].metric(name, "N/A", "0.00%")
    st.divider()

# ----------------------------------------------------------

def render_academic_framework():
    """Strategic Risk Methodology and Mathematical Proofs."""
    st.header("🔬 Institutional Risk Framework")
    st.write("Our system architecture integrates stochastic calculus with neural time-series decomposition.")
    
    t1, t2, t3 = st.tabs(["Stochastic Volatility", "Tail-Risk Theory", "Neural Decomposition"])
    
    with t1:
        st.subheader("I. Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("This stochastic differential equation models asset prices as a continuous-time Markov process. We solve this for price trajectory simulations.")
        
        
    with t2:
        st.subheader("II. Parametric Value at Risk (VaR)")
        st.latex(r"VaR_{\alpha} = \mu + \sigma \cdot \Phi^{-1}(\alpha)")
        st.write("We employ Gaussian approximation to calculate the maximum potential drawdown within a 95% confidence interval.")
        

    with t3:
        st.subheader("III. Multi-Factor Neural Forecasting")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("Using Prophet V5, we decompose market signals into Trend (g), Seasonality (s), and Holidays (h) while filtering white noise.")

# ----------------------------------------------------------

def render_equity_intelligence():
    """Live Equity Analysis, Custom Uploads & Monte Carlo."""
    st.header("📈 Equity Intelligence & Custom Audit")
    
    data_source = st.sidebar.selectbox("Data Source:", ["Live Terminal (Yahoo)", "Institutional Upload (CSV)"])
    
    if data_source == "Live Terminal (Yahoo)":
        ticker = st.text_input("Institutional Ticker (e.g., RACE, ENI.MI, NVDA):", "RACE").upper()
        lookback = st.selectbox("Lookback Window:", ["1y", "2y", "5y", "max"])
        if st.button("Initialize Deep Run"):
            with st.spinner("Executing Capital Audit..."):
                tk = yf.Ticker(ticker)
                raw_df = SovereignEngine.standardize_data(tk.history(period=lookback))
                if not raw_df.empty:
                    execute_full_analysis(raw_df, ticker, tk.info)
    else:
        uploaded_file = st.file_uploader("Upload Transaction or Price History (CSV):", type="csv")
        if uploaded_file:
            up_df = pd.read_csv(uploaded_file)
            if 'Date' in up_df.columns and 'Close' in up_df.columns:
                up_df.set_index('Date', inplace=True)
                execute_full_analysis(SovereignEngine.standardize_data(up_df), "Uploaded Dataset", None)
            else:
                st.error("Error: CSV must contain 'Date' and 'Close' columns.")

def execute_full_analysis(df, label, info):
    """Sub-engine to run calculations and plots for both live and uploaded data."""
    prices = df['Close']
    returns = prices.pct_change().dropna()
    metrics = SovereignEngine.calculate_risk_ratios(returns)
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Value", f"${prices.iloc[-1]:,.2f}")
    k2.metric("Sharpe Efficiency", f"{metrics['Sharpe']:.2f}")
    k3.metric("Max Drawdown", f"{metrics['MDD']:.2f}%")
    k4.metric("Daily VaR (95%)", f"{metrics['VaR']:.2f}%")
    
    # Sovereign Agent Verdict (Wall Street Logic)
    if info:
        st.markdown(f"""
        <div class="agent-verdict">
            <h3>🕵️ Sovereign Agent Verdict: {info.get('recommendationKey', 'NEUTRAL').upper()}</h3>
            <p><b>Forward P/E Ratio:</b> {info.get('forwardPE', 'N/A')} | <b>Analyst Target Price:</b> {info.get('targetMeanPrice', 'N/A')}</p>
            <p>Analysis suggests <b>{"High" if metrics['Vol'] > 35 else "Stable"}</b> volatility dynamics. 
            Risk-adjusted performance is <b>{"Superior" if metrics['Sharpe'] > 1 else "Moderate"}</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Performance Chart
    fig_main = px.line(df, y='Close', title=f"{label} - Institutional Performance", template="plotly_dark")
    fig_main.update_traces(line_color="#FFD700", line_width=3)
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Distribution Analysis
    st.subheader("📊 Statistical Density & Volatility Clusters")
    fig_dist = px.histogram(returns, nbins=80, marginal="box", title="Return Distribution Histogram", template="plotly_dark", color_discrete_sequence=['#FFD700'])
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Monte Carlo Stress Test
    st.subheader("🎲 Monte Carlo Stochastic Simulation")
    last_price, mu, sigma = prices.iloc[-1], returns.mean(), returns.std()
    mc_fig = go.Figure()
    for _ in range(70):
        path = [last_price]
        for _ in range(45):
            path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
        mc_fig.add_trace(go.Scatter(y=path, mode='lines', opacity=0.1, line=dict(color='#FFD700')))
    mc_fig.update_layout(title="70 Stochastic Trajectories (45-Day Horizon)", template="plotly_dark", showlegend=False)
    st.plotly_chart(mc_fig, use_container_width=True)

# ----------------------------------------------------------

def render_neural_prediction():
    """High-Performance AI Forecasting."""
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Asset for 90-Day Neural Projection:", "BTC-USD").upper()
    
    if st.button("Deploy Prophet V5 Model"):
        with st.spinner("Training Probabilistic Neural Model..."):
            raw = yf.download(target, period="3y", progress=False).reset_index()
            raw = SovereignEngine.standardize_data(raw)
            df_p = pd.DataFrame({'ds': raw['Date'].dt.tz_localize(None), 'y': raw['Close']}).dropna()
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05).fit(df_p)
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual Price", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Mean Prediction", line=dict(color='#FFD700', dash='dash')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,215,0,0.1)', showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,215,0,0.1)', showlegend=False))
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Forecast Dynamics: {target}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_management():
    """Behavioral Wealth Advisor and Cash Flow Audit."""
    st.header("💳 AI Behavioral Wealth Advisor")
    st.markdown("Automated capital allocation audit based on institutional behavioral standards.")
    
    # Default Matrix
    data = {
        "Description": ["Executive Income", "Mortgage/Fixed Costs", "Equity Portfolio", "Lifestyle/Wants", "Alternative Assets"],
        "Category": ["Income", "Fixed", "Wealth", "Wants", "Wealth"],
        "Amount": [15000, -4500, -3000, -1500, -1000]
    }
    
    df_wealth = pd.DataFrame(data)
    ed_df = st.data_editor(df_wealth, num_rows="dynamic", use_container_width=True)
    
    ed_df['Amount'] = pd.to_numeric(ed_df['Amount'], errors='coerce').fillna(0)
    outflows = ed_df[ed_df['Amount'] < 0].copy()
    outflows['Abs'] = outflows['Amount'].abs()
    
    if not outflows.empty:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig_pie = px.pie(outflows, values='Abs', names='Category', hole=0.6, 
                             title="Capital Allocation Map", template="plotly_dark",
                             color_discrete_sequence=px.colors.sequential.YlOrBr)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("Allocation Efficiency")
            w_sum = outflows[outflows['Category'] == 'Wealth']['Abs'].sum()
            total_out = outflows['Abs'].sum()
            w_rate = (w_sum / total_out) * 100
            
            st.metric("Wealth Creation Velocity", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}% (Target: 20%)")
            if w_rate < 20:
                st.error("Under-allocated. Increase capital velocity towards Wealth assets.")
            else:
                st.success("Target Reached. Optimal behavioral allocation detected.")
    
    st.divider()
    st.subheader("Transactional Ledger Analysis")
    st.dataframe(ed_df.style.background_gradient(cmap='RdYlGn', subset=['Amount']), use_container_width=True)

# ==========================================================
# 4. SYSTEM MASTER CONTROLLER (ROUTER)
# ==========================================================

def main():
    # Sidebar Navigation
    st.sidebar.title("💎 Sovereign Menu")
    navigation = st.sidebar.radio("Navigation Domains:", [
        "Strategic Risk Framework", 
        "Equity Intelligence", 
        "Neural AI Forecasting", 
        "Wealth Management Advisor"
    ])
    
    # Global Header Pulse
    render_market_pulse()
    
    # Main Page Router
    if navigation == "Strategic Risk Framework":
        render_academic_framework()
    elif navigation == "Equity Intelligence":
        render_equity_intelligence()
    elif navigation == "Neural AI Forecasting":
        render_neural_prediction()
    elif navigation == "Wealth Management Advisor":
        render_wealth_management()
    
    # Sidebar Footer
    st.sidebar.divider()
    st.sidebar.markdown(f"<p class='sidebar-text'>Terminal Version: 20.0.0-Magnum</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='sidebar-text'>Sync Time: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
    st.sidebar.info("Operational: Institutional Data Feeds & Neural Engines Active")

if __name__ == "__main__":
    main()
