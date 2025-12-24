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
import io

# ==========================================================
# 1. SYSTEM INITIALIZATION & ELITE STYLING
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Diana Sovereign AI | Multi-Asset Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Institutional Theme (Gold & Obsidian)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #E0E0E0; }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.96), rgba(0,0,0,0.96)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover; background-attachment: fixed;
    }
    
    .main .block-container {
        background: rgba(10, 10, 10, 0.98);
        border-radius: 40px; padding: 50px 80px;
        border: 1px solid #2a2a2a; box-shadow: 0 40px 150px rgba(0,0,0,1);
    }
    
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; letter-spacing: -2px; }
    h2, h3 { color: #FFD700 !important; font-weight: 700; border-left: 8px solid #FFD700; padding-left: 20px; margin-top: 40px; }
    
    .stMetric { 
        background: rgba(255,255,255,0.03); padding: 25px; 
        border-radius: 20px; border-top: 4px solid #FFD700; transition: 0.3s;
    }
    .stMetric:hover { transform: translateY(-5px); background: rgba(255,215,0,0.05); }
    
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 900; border: none; padding: 15px; border-radius: 12px;
        text-transform: uppercase; width: 100%; transition: 0.4s;
    }
    
    .upload-zone {
        border: 2px dashed #FFD700; border-radius: 25px; padding: 40px;
        background: rgba(255,215,0,0.02); text-align: center;
    }
    
    .agent-box {
        background: rgba(255, 215, 0, 0.05); border: 1px solid #FFD700;
        border-radius: 20px; padding: 30px; margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE ANALYTICS ENGINE (CORE)
# ==========================================================

class SovereignCore:
    @staticmethod
    def calculate_risk_metrics(prices):
        """Institutional risk calculation engine."""
        returns = prices.pct_change().dropna()
        if returns.empty: return None
        
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        
        cum_rets = (1 + returns).cumprod()
        mdd = ((cum_rets / cum_rets.cummax()) - 1).min() * 100
        
        return {
            "Sharpe": sharpe, "MDD": mdd, "VaR": var_95, 
            "Volatility": sigma * np.sqrt(252) * 100,
            "Returns": returns
        }

    @staticmethod
    def run_monte_carlo(last_price, mu, sigma, days=30, sims=50):
        """Stochastic path simulation."""
        results = []
        for _ in range(sims):
            path = [last_price]
            for _ in range(days):
                path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            results.append(path)
        return results

# ==========================================================
# 3. INTERFACE MODULES
# ==========================================================

def render_header():
    st.title("🏛️ Diana Sovereign")
    st.markdown("##### *Unified Strategic Intelligence & Global Capital Terminal*")
    
    # Live Ticker Ribbon
    ribbon = ["^GSPC", "^IXIC", "GC=F", "BTC-USD", "^TNX"]
    cols = st.columns(len(ribbon))
    for i, sym in enumerate(ribbon):
        try:
            d = yf.download(sym, period="2d", progress=False)
            price = d['Close'].iloc[-1]
            change = (price / d['Close'].iloc[-2] - 1) * 100
            cols[i].metric(sym.replace("^", ""), f"{price:,.2f}", f"{change:+.2f}%")
        except: pass
    st.divider()

# ----------------------------------------------------------

def render_file_intelligence():
    st.header("📂 Universal File Intelligence")
    st.write("Upload any CSV (Stock History, Transaction Logs, or Economics) for automated auditing.")
    
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    up_file = st.file_uploader("Drop your institutional CSV here", type="csv")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if up_file:
        df = pd.read_csv(up_file)
        st.success("Data Stream Synchronized.")
        
        with st.expander("🔍 Raw Data Inspection"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Heuristic Logic to detect data type
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        
        if date_cols and num_cols:
            st.subheader("📈 Time-Series Financial Audit")
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            target = num_cols[0]
            
            m = SovereignCore.calculate_risk_metrics(df[target])
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Latest Val", f"{df[target].iloc[-1]:,.2f}")
            k2.metric("Sharpe", f"{m['Sharpe']:.2f}")
            k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
            k4.metric("Risk (VaR)", f"{m['VaR']:.2f}%")
            
            fig = px.line(df, x=date_cols[0], y=target, title="File Data Trajectory", template="plotly_dark")
            fig.update_traces(line_color="#FFD700")
            st.plotly_chart(fig, use_container_width=True)
            
            
            st.plotly_chart(px.histogram(m['Returns'], nbins=100, title="Return Distribution", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
        else:
            st.warning("Data format recognized as Non-Temporal. Showing General Statistics.")
            st.write(df.describe())

# ----------------------------------------------------------

def render_live_terminal():
    st.header("🌐 Global Live Terminal")
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("Institutional Symbol (e.g. RACE, NVDA, AAPL, BTC-USD):", "RACE").upper()
    horizon = c2.selectbox("Horizon:", ["1y", "2y", "5y", "max"])
    
    if st.button("Initialize Deep Run"):
        with st.spinner("Establishing Connection..."):
            tk = yf.Ticker(ticker)
            df = tk.history(period=horizon)
            
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                m = SovereignCore.calculate_risk_metrics(df['Close'])
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Price", f"${df['Close'].iloc[-1]:,.2f}")
                m2.metric("Sharpe", f"{m['Sharpe']:.2f}")
                m3.metric("Volatility", f"{m['Volatility']:.1f}%")
                m4.metric("VaR (95%)", f"{m['VaR']:.2f}%")
                
                # Agent Verdict
                st.markdown(f"""
                <div class="agent-box">
                    <h3>🕵️ Agent Verdict: {tk.info.get('recommendationKey', 'NEUTRAL').upper()}</h3>
                    <p><b>Asset:</b> {tk.info.get('longName', ticker)} | <b>Sector:</b> {tk.info.get('sector', 'N/A')}</p>
                    <p>Historical efficiency is <b>{"High" if m['Sharpe'] > 1 else "Moderate"}</b>. 
                    Tail-risk exposure is calculated at <b>{abs(m['VaR']):.2f}%</b> daily.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(px.line(df, y='Close', title=f"{ticker} Performance", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
                
                # Monte Carlo Simulation
                st.subheader("🎲 Monte Carlo Stress Test")
                paths = SovereignCore.run_monte_carlo(df['Close'].iloc[-1], m['Returns'].mean(), m['Returns'].std())
                mc_fig = go.Figure()
                for p in paths:
                    mc_fig.add_trace(go.Scatter(y=p, mode='lines', opacity=0.15, line=dict(color='#FFD700')))
                st.plotly_chart(mc_fig.update_layout(template="plotly_dark", title="Stochastic Projections (30D)", showlegend=False), use_container_width=True)
            else:
                st.error("Ticker not found.")

# ----------------------------------------------------------

def render_neural_forecast():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Predict 90-Day Trend for:", "BTC-USD").upper()
    
    if st.button("Deploy Prophet V5"):
        with st.spinner("Neural Training in progress..."):
            raw = yf.download(target, period="3y", progress=False).reset_index()
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                df_p = pd.DataFrame({'ds': raw['Date'].dt.tz_localize(None), 'y': raw['Close']}).dropna()
                
                m = Prophet(daily_seasonality=True).fit(df_p)
                future = m.make_future_dataframe(periods=90)
                forecast = m.predict(future)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color='#FFD700', dash='dash')))
                st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Neural Projection: {target}"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    
    t_man, t_file = st.tabs(["📝 Manual Ledger", "📥 Upload Transactions"])
    
    with t_man:
        df_w = pd.DataFrame([
            {"Category": "Executive Income", "Amount": 15000},
            {"Category": "Fixed Costs", "Amount": -4500},
            {"Category": "Equity Wealth", "Amount": -3500},
            {"Category": "Lifestyle", "Amount": -1500}
        ])
        data = st.data_editor(df_w, num_rows="dynamic", use_container_width=True)
    
    with t_file:
        up_w = st.file_uploader("Upload Wealth CSV", type="csv", key="wealth_up")
        if up_w: data = pd.read_csv(up_w)
        else: st.info("Using Manual Ledger data.")
        
    # Analysis
    try:
        data['Amount'] = pd.to_numeric(data['Amount'])
        income = data[data['Amount'] > 0]['Amount'].sum()
        outflows = data[data['Amount'] < 0].copy()
        outflows['Abs'] = outflows['Amount'].abs()
        
        if income > 0:
            invest = outflows[outflows['Category'].str.contains('Wealth|Invest|Gold', case=False)]['Abs'].sum()
            w_rate = (invest / income) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Income", f"${income:,.0f}")
            c2.metric("Wealth Rate", f"{w_rate:.1f}%")
            c3.metric("Savings", f"${income - outflows['Abs'].sum():,.0f}")
            
            st.divider()
            col_chart, col_verdict = st.columns([1.5, 1])
            with col_chart:
                st.plotly_chart(px.pie(outflows, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
            with col_verdict:
                st.subheader("🕵️ Advisor Verdict")
                if w_rate < 20: st.warning("Wealth rate below 20%. Optimize lifestyle expenses.")
                else: st.success("Standard achieved. Wealth velocity is optimal.")
    except: st.error("Ensure data has 'Category' and 'Amount' columns.")

# ----------------------------------------------------------

def render_framework():
    st.header("🔬 Strategic Risk Framework")
    st.write("Academic foundations of the Sovereign Engine.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("I. Geometric Brownian Motion")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Modeling continuous-time stochastic processes for price evolution.")
        
        
    with c2:
        st.subheader("II. Neural Decomposition")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("Additive decomposition of trend, seasonality, and holidays.")

# ==========================================================
# 4. MASTER CONTROLLER (ROUTER)
# ==========================================================

def main():
    render_header()
    
    st.sidebar.title("💎 Sovereign Menu")
    nav = st.sidebar.radio("Navigation Domains:", [
        "File Intelligence", 
        "Live Market Audit", 
        "Neural Forecasting", 
        "Wealth Management",
        "Risk Framework"
    ])
    
    if nav == "File Intelligence": render_file_intelligence()
    elif nav == "Live Market Audit": render_live_terminal()
    elif nav == "Neural Forecasting": render_neural_forecast()
    elif nav == "Wealth Management": render_wealth_advisor()
    elif nav == "Risk Framework": render_framework()
    
    # Sidebar Metadata
    st.sidebar.divider()
    st.sidebar.caption(f"Diana Sovereign Terminal | Build v25.0")
    st.sidebar.caption(f"Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
