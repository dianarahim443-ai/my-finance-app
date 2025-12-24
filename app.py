
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
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime
from scipy.stats import norm
import warnings

# ==========================================================
# 1. CORE CONFIGURATION & ELITE STYLING
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Global Terminal", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover; background-attachment: fixed;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98); border-radius: 40px; 
        padding: 60px; border: 1px solid #333; box-shadow: 0 40px 150px rgba(0,0,0,1);
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; letter-spacing: -2px; }
    h2, h3 { color: #FFD700 !important; border-left: 8px solid #FFD700; padding-left: 20px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 25px; border-radius: 20px; border-top: 4px solid #FFD700; }
    .agent-verdict { background: rgba(255, 215, 0, 0.05); border: 1px dashed #FFD700; border-radius: 20px; padding: 30px; margin: 25px 0; }
    .stButton>button { background: linear-gradient(45deg, #FFD700, #B8860B); color: black !important; font-weight: 900; border-radius: 12px; height: 3.5em; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE ANALYTICS ENGINE
# ==========================================================

class SovereignEngine:
    @staticmethod
    def calculate_risk_metrics(returns):
        if returns.empty: return None
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        cum = (1 + returns).cumprod()
        mdd = ((cum / cum.cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "MDD": mdd, "VaR": var_95, "Vol": sigma * np.sqrt(252) * 100}

# ==========================================================
# 3. INTERFACE PERSPECTIVES
# ==========================================================

def render_academic_framework():
    st.header("🔬 Strategic Risk Framework")
    st.write("Academic foundations of our quantitative risk engine.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("I. Geometric Brownian Motion")
        st.latex(r"S_{t+dt} = S_t \exp\left( (\mu - \frac{\sigma^2}{2})dt + \sigma \sqrt{dt} Z \right)")
        st.write("Used for simulating stochastic price trajectories and future probability densities.")
        
        
    with col_b:
        st.subheader("II. Parametric Value at Risk")
        st.latex(r"VaR_{95\%} = \mu + \sigma \cdot 1.645")
        st.write("Determining the 95th percentile exposure for institutional capital preservation.")
        

# ----------------------------------------------------------

def render_equity_intelligence():
    st.header("📈 Universal Equity Intelligence")
    st.markdown("Access 100,000+ global assets including Stocks, ETFs, Crypto, and Forex.")
    
    # Dual Entry System
    col_in1, col_in2 = st.columns([3, 1])
    ticker = col_in1.text_input("Enter Global Ticker (e.g., RACE, NVDA, ENI.MI, 0005.HK, BTC-USD):", "RACE").upper()
    period = col_in2.selectbox("Analysis Horizon:", ["1y", "2y", "5y", "max"])
    
    if st.button("Initialize Sovereign Audit"):
        with st.spinner(f"Agent establishing connection to {ticker} data stream..."):
            tk = yf.Ticker(ticker)
            df = tk.history(period=period)
            
            if not df.empty:
                # Cleaning MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                rets = df['Close'].pct_change().dropna()
                m = SovereignEngine.calculate_risk_metrics(rets)
                
                # KPIs Row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Market Price", f"${df['Close'].iloc[-1]:,.2f}")
                k2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
                k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
                k4.metric("Daily VaR (95%)", f"{m['VaR']:.2f}%")
                
                # AI Agent Verdict
                info = tk.info
                rec = info.get('recommendationKey', 'NEUTRAL').upper()
                st.markdown(f"""
                <div class="agent-verdict">
                    <h3>🕵️ Sovereign Agent Verdict: {rec}</h3>
                    <p><b>Company:</b> {info.get('longName', ticker)} | <b>Sector:</b> {info.get('sector', 'N/A')}</p>
                    <p><b>Analyst Target:</b> {info.get('targetMeanPrice', 'N/A')} | <b>Forward P/E:</b> {info.get('forwardPE', 'N/A')}</p>
                    <p><i>Agent Analysis:</i> This asset exhibits <b>{"High" if m['Vol'] > 30 else "Moderate"}</b> historical volatility. 
                    The risk-adjusted return (Sharpe) of <b>{m['Sharpe']:.2f}</b> indicates 
                    <b>{"Superior" if m['Sharpe'] > 1 else "Standard"}</b> efficiency in capital growth.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Charts
                st.plotly_chart(px.line(df, y='Close', title=f"{ticker} Institutional Trajectory", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
                
                # Statistical Distribution
                st.subheader("📊 Return Density & Kurtosis Audit")
                st.plotly_chart(px.histogram(rets, nbins=100, marginal="violin", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
                
                # Monte Carlo Simulation
                st.subheader("🎲 Monte Carlo Stochastic Stress Test")
                last_p, mu, sigma = df['Close'].iloc[-1], rets.mean(), rets.std()
                mc_fig = go.Figure()
                for _ in range(50):
                    path = [last_p]
                    for _ in range(30): path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
                    mc_fig.add_trace(go.Scatter(y=path, mode='lines', opacity=0.15, line=dict(color='#FFD700')))
                st.plotly_chart(mc_fig.update_layout(title="50 Forward Paths (30-Day Simulation)", template="plotly_dark", showlegend=False), use_container_width=True)
            else:
                st.error("Ticker not found. Please check the symbol (e.g., use 'ENI.MI' for Eni on the Milan exchange).")

# ----------------------------------------------------------

def render_neural_forecast():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Enter Target for 90-Day Forecast:", "BTC-USD").upper()
    if st.button("Deploy Neural Model"):
        with st.spinner("Processing Neural Parameters..."):
            raw = yf.download(target, period="3y", progress=False).reset_index()
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                df_p = pd.DataFrame({'ds': raw['Date'].dt.tz_localize(None), 'y': raw['Close']}).dropna()
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=90))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color='#FFD700', dash='dash')))
                st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Forecast for {target}"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    df = pd.DataFrame([{"Category": "Income", "Amount": 15000}, {"Category": "Fixed", "Amount": -4500}, {"Category": "Wealth", "Amount": -3500}, {"Category": "Wants", "Amount": -1500}])
    ed_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    ed_df['Amount'] = pd.to_numeric(ed_df['Amount'], errors='coerce').fillna(0)
    outflows = ed_df[ed_df['Amount'] < 0].copy()
    outflows['Abs'] = outflows['Amount'].abs()
    
    if not outflows.empty:
        c1, c2 = st.columns([1.5, 1])
        with c1: st.plotly_chart(px.pie(outflows, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
        with c2:
            w_rate = (outflows[outflows['Category'] == 'Wealth']['Abs'].sum() / outflows['Abs'].sum()) * 100
            st.metric("Wealth Creation Rate", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}%")

# ==========================================================
# 4. MASTER CONTROLLER
# ==========================================================

def main():
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Navigation:", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Advisor"])
    
    if nav == "Risk Framework": render_academic_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_prediction()
    elif nav == "Wealth Advisor": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Terminal Build: v21.0 | Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    st.markdown("فایل تراکنش‌ها یا بودجه‌بندی خود را آپلود کنید یا به صورت دستی وارد نمایید.")

    # سیستم دوگانه: آپلود فایل یا ویرایش دستی
    tab_manual, tab_upload_wealth = st.tabs(["📝 ورود دستی داده‌ها", "📥 آپلود لیست مخارج (CSV)"])

    with tab_manual:
        # دیتای پیش‌فرض برای نمایش اولیه
        default_data = [
            {"Category": "Income (درآمد)", "Amount": 15000},
            {"Category": "Fixed Costs (اجاره و قبوض)", "Amount": -4500},
            {"Category": "Investments (پس‌انداز و طلا)", "Amount": -3500},
            {"Category": "Lifestyle (تفریح و خرید)", "Amount": -1500}
        ]
        df_wealth = pd.DataFrame(default_data)
        ed_df = st.data_editor(df_wealth, num_rows="dynamic", use_container_width=True)

    with tab_upload_wealth:
        uploaded_wealth = st.file_uploader("فایل بودجه‌بندی خود را آپلود کنید (CSV)", type=["csv"], key="wealth_up")
        if uploaded_wealth:
            ed_df = pd.read_csv(uploaded_wealth)
            st.write("داده‌های استخراج شده:")
            ed_df = st.data_editor(ed_df, num_rows="dynamic", use_container_width=True)
        else:
            st.info("فایل آپلود نشده است. از جدول 'ورود دستی' استفاده کنید.")
            ed_df = pd.DataFrame(default_data) # در صورت عدم آپلود، همان دیتای دستی را ملاک قرار بده

    # محاسبات هوشمند (Sovereign Analytics)
    try:
        ed_df['Amount'] = pd.to_numeric(ed_df['Amount'], errors='coerce').fillna(0)
        
        total_income = ed_df[ed_df['Amount'] > 0]['Amount'].sum()
        total_outflow = ed_df[ed_df['Amount'] < 0]['Amount'].abs().sum()
        
        # دسته‌بندی مخارج برای نمودار
        outflows = ed_df[ed_df['Amount'] < 0].copy()
        outflows['Abs'] = outflows['Amount'].abs()

        if total_income > 0:
            c1, c2, c3 = st.columns([1, 1, 1])
            
            # محاسبه نرخ ثروت‌سازی (درصد سرمایه‌گذاری نسبت به کل درآمد)
            investment_val = outflows[outflows['Category'].str.contains('Invest|ثروت|پس‌انداز', case=False)]['Abs'].sum()
            wealth_rate = (investment_val / total_income) * 100
            
            c1.metric("Total Income", f"${total_income:,.0f}")
            c2.metric("Wealth Creation Rate", f"{wealth_rate:.1f}%")
            c3.metric("Disposable Income", f"${total_income - total_outflow:,.0f}")

            # نمایش نمودار تخصیص منابع
            st.divider()
            col_chart, col_advise = st.columns([1.5, 1])
            
            with col_chart:
                fig_wealth = px.pie(outflows, values='Abs', names='Category', 
                                  hole=0.6, title="Capital Allocation Structure",
                                  template="plotly_dark",
                                  color_discrete_sequence=px.colors.sequential.YlOrBr)
                st.plotly_chart(fig_wealth, use_container_width=True)
            
            with col_advise:
                st.subheader("🕵️ AI Financial Verdict")
                if wealth_rate < 20:
                    st.warning("هشدار: نرخ ثروت‌سازی شما زیر ۲۰٪ است. پیشنهاد می‌شود هزینه‌های Lifestyle را کاهش دهید.")
                else:
                    st.success("عالی: شما در مسیر استقلال مالی هستید. نرخ ثروت‌سازی شما استاندارد طلایی را رعایت کرده است.")
                
                st.info(f"بر اساس تحلیل هوش مصنوعی، شما می‌توانید ماهانه {total_income * 0.1:,.0f} واحد دیگر به سبد سهام خود اضافه کنید.")
    except Exception as e:
        st.error(f"خطا در پردازش داده‌های مالی: {e}")

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime
from scipy.stats import norm
import warnings

# --- GLOBAL CONFIG & STYLE ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | File Intelligence", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #050505; }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98); border-radius: 35px; 
        padding: 50px 70px; border: 1px solid #2a2a2a;
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4rem !important; }
    h2, h3 { color: #FFD700 !important; border-left: 6px solid #FFD700; padding-left: 20px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 25px; border-radius: 18px; border-top: 4px solid #FFD700; }
    .upload-section {
        border: 2px dashed #FFD700; padding: 40px; border-radius: 25px;
        background: rgba(255, 215, 0, 0.02); text-align: center; margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 800; border-radius: 10px; height: 3.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. THE BRAIN: AUTO-ANALYTICS ENGINE
# ==========================================
class UniversalEngine:
    @staticmethod
    def analyze_uploaded_file(df):
        """بصورت خودکار نوع داده را تشخیص داده و تحلیل می‌کند"""
        st.subheader("🔍 Intelligent Data Inspection")
        
        # شناسایی ستون‌های عددی و زمانی
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        col1, col2 = st.columns(2)
        col1.write(f"**Detected Numeric Columns:** {', '.join(numeric_cols)}")
        col2.write(f"**Detected Time Columns:** {', '.join(date_cols) if date_cols else 'None'}")

        # اگر داده زمانی و عددی باشد (تحلیل سری زمانی مثل بورس)
        if date_cols and numeric_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df = df.sort_values(by=date_cols[0])
            target = numeric_cols[0]
            
            # محاسبات کوانت
            returns = df[target].pct_change().dropna()
            mu, sigma = returns.mean(), returns.std()
            sharpe = (mu / sigma) * np.sqrt(252) if sigma != 0 else 0
            var_95 = norm.ppf(0.05, mu, sigma) * 100
            
            # نمایش KPIها
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current Value", f"{df[target].iloc[-1]:,.2f}")
            k2.metric("Sharpe Efficiency", f"{sharpe:.2f}")
            k3.metric("Volatility (Ann.)", f"{sigma * np.sqrt(252)*100:.1f}%")
            k4.metric("Risk (VaR 95%)", f"{var_95:.2f}%")
            
            # رسم نمودار اصلی
            fig = px.line(df, x=date_cols[0], y=target, title="Custom File Performance Trajectory", template="plotly_dark")
            fig.update_traces(line_color="#FFD700", line_width=2)
            st.plotly_chart(fig, use_container_width=True)
            
            # توزیع بازدهی
            st.subheader("📊 Return Distribution Analysis")
            
            fig_hist = px.histogram(returns, nbins=100, marginal="violin", template="plotly_dark", color_discrete_sequence=['#FFD700'])
            st.plotly_chart(fig_hist, use_container_width=True)

        # اگر داده دسته‌بندی داشته باشد (تحلیل بودجه و مخارج)
        elif 'Category' in df.columns or 'category' in df.columns:
            cat_col = 'Category' if 'Category' in df.columns else 'category'
            val_col = numeric_cols[0]
            
            st.subheader("💰 Wealth Allocation Audit")
            c1, c2 = st.columns([1.5, 1])
            
            with c1:
                fig_pie = px.pie(df, values=np.abs(df[val_col]), names=cat_col, hole=0.5, template="plotly_dark",
                                 color_discrete_sequence=px.colors.sequential.YlOrBr)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                total_in = df[df[val_col] > 0][val_col].sum()
                total_out = df[df[val_col] < 0][val_col].abs().sum()
                st.metric("Total Inflow", f"${total_in:,.0f}")
                st.metric("Total Outflow", f"${total_out:,.0f}")
                wealth_val = df[df[cat_col].str.contains('Invest|Wealth|ثروت', case=False)][val_col].abs().sum()
                if total_in > 0:
                    st.metric("Wealth Creation Rate", f"{(wealth_val/total_in)*100:.1f}%")

# ==========================================
# 2. INTERFACE MODULES
# ==========================================

def render_file_intelligence():
    st.header("📂 Universal File Intelligence")
    st.markdown("این بخش به شما اجازه می‌دهد **هر فایلی** را آپلود کنید تا هوش مصنوعی دایانا آن را تحلیل کند.")
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV file for deep analysis", type="csv")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File synchronized successfully.")
        
        # نمایش پیش‌نمایش دیتا
        with st.expander("👀 View Raw Data Stream"):
            st.dataframe(df.style.background_gradient(cmap='YlOrBr'), use_container_width=True)
        
        # اجرای موتور تحلیل خودکار
        UniversalEngine.analyze_uploaded_file(df)
    else:
        st.info("Waiting for data stream... Please upload a CSV file to begin.")

# ------------------------------------------

def render_risk_framework():
    st.header("🔬 Academic Risk Framework")
    st.write("Mathematical foundations for institutional-grade auditing.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stochastic Differential Equations")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Modeling asset drift and diffusion using Brownian Motion.")
        
        
    with col2:
        st.subheader("Neural Network Decomposition")
        st.latex(r"f(x) = \text{Trend} + \text{Seasonality} + \text{Error}")
        st.write("Prophet-based non-linear time series forecasting.")

# ==========================================
# 3. MAIN APP ROUTER
# ==========================================
def main():
    st.sidebar.title("💎 Diana Sovereign")
    st.sidebar.markdown("Institutional AI Terminal")
    
    menu = st.sidebar.radio("Navigation Domains:", [
        "File Intelligence", 
        "Live Market Audit", 
        "Neural Forecasting", 
        "Risk Framework"
    ])
    
    if menu == "File Intelligence":
        render_file_intelligence()
    elif menu == "Live Market Audit":
        st.title("🌐 Live Terminal")
        ticker = st.text_input("Enter Global Symbol:", "RACE").upper()
        if st.button("Audit Asset"):
            data = yf.download(ticker, period="2y")
            UniversalEngine.analyze_uploaded_file(data)
    elif menu == "Neural Forecasting":
        st.title("🔮 Neural Engine")
        st.write("در حال تحلیل الگوهای عصبی روی دیتای انتخاب شده...")
    elif menu == "Risk Framework":
        render_risk_framework()

    st.sidebar.divider()
    st.sidebar.caption(f"Status: High-Performance Engine Active | {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
