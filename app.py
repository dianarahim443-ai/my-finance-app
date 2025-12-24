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

# ==========================================
# 1. GLOBAL SYSTEM CONFIGURATION & PREMIUM THEME
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Sovereign Magnum Opus",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS - Dark Mode, Gold Accents, and High-Resolution Background
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92)), 
                    url('https://images.unsplash.com/photo-1518544801976-3e159e50e5bb?q=80&w=2069');
        background-size: cover;
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: rgba(10, 10, 10, 0.96);
        border-radius: 40px;
        padding: 50px 70px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 40px 120px rgba(0,0,0,1);
    }
    
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; letter-spacing: -3px; line-height: 1; margin-bottom: 20px; }
    h2, h3 { color: #FFD700 !important; font-weight: 700; border-left: 8px solid #FFD700; padding-left: 20px; margin-top: 45px; margin-bottom: 25px; }
    
    .stMetric { 
        background: rgba(255,255,255,0.03); 
        padding: 30px; 
        border-radius: 20px; 
        border-top: 5px solid #FFD700; 
        transition: all 0.4s ease;
    }
    .stMetric:hover { transform: translateY(-10px); background: rgba(255,215,0,0.08); }
    
    .agent-box { 
        background: rgba(255, 215, 0, 0.04); 
        padding: 30px; 
        border-radius: 24px; 
        border: 1px dashed #FFD700;
        margin: 20px 0;
    }

    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 900; border: none; padding: 15px 40px; border-radius: 12px;
        text-transform: uppercase; letter-spacing: 1px; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANT ENGINE & AGENT LOGIC
# ==========================================

class SovereignEngine:
    @staticmethod
    def format_yf(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def audit_performance(returns):
        if returns.empty: return None
        rf = 0.045 / 252 
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu - rf) / sigma * np.sqrt(252) if sigma != 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        cum_rets = (1 + returns).cumprod()
        mdd = ((cum_rets / cum_rets.cummax()) - 1).min() * 100
        vol_ann = sigma * np.sqrt(252) * 100
        return {"Sharpe": sharpe, "MDD": mdd, "VaR": var_95, "Vol": vol_ann}

    @staticmethod
    def get_agent_verdict(ticker_obj):
        try:
            info = ticker_obj.info
            return {
                "pe": info.get('forwardPE', 'N/A'),
                "rec": info.get('recommendationKey', 'N/A').upper(),
                "target": info.get('targetMeanPrice', 'N/A')
            }
        except: return {"pe": "N/A", "rec": "NEUTRAL", "target": "N/A"}

# ==========================================
# 3. INTERFACE MODULES
# ==========================================

def render_global_pulse():
    st.title("🏛️ Diana Sovereign")
    st.markdown("##### *Institutional-Grade Quantitative Terminal | Magnum Opus Defense Edition*")
    
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
    st.header("🔬 Strategic Risk Framework")
    st.write("Risk analysis in this system is built upon stochastic modeling and modern probability theory.")
    
    t1, t2 = st.tabs(["Stochastic Models", "Tail Risk Analytics"])
    with t1:
        st.subheader("I. Geometric Brownian Motion (GBM)")
        st.latex(r"S_{t+dt} = S_t \exp\left( (\mu - \frac{\sigma^2}{2})dt + \sigma \sqrt{dt} Z \right)")
        st.write("This SDE is solved to generate potential future price paths and calculate tail risk probabilities.")
        
        
    with t2:
        st.subheader("II. Parametric Value at Risk (VaR)")
        st.latex(r"VaR_{95\%} = \mu + \sigma \cdot \Phi^{-1}(0.05)")
        st.write("Calculates the potential maximum loss at a 95% confidence level for a single trading session.")
        

def render_equity_intel():
    st.header("📈 Equity Intelligence & AI Agent")
    ticker_sym = st.text_input("Enter Ticker (e.g., RACE for Ferrari, ENI.MI for Eni, NVDA):", "RACE").upper()
    
    if st.button("Initialize Deep Research"):
        with st.spinner("Agent gathering intelligence..."):
            tk = yf.Ticker(ticker_sym)
            raw = SovereignEngine.format_yf(tk.history(period="2y"))
            if not raw.empty:
                prices = raw['Close']
                returns = prices.pct_change().dropna()
                m = SovereignEngine.audit_performance(returns)
                agent = SovereignEngine.get_agent_verdict(tk)
                
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Market Price", f"${prices.iloc[-1]:,.2f}")
                k2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
                k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
                k4.metric("Daily VaR (95%)", f"{m['VaR']:.2f}%")
                
                st.markdown(f"""<div class="agent-box">
                    <h3>🕵️ Sovereign Agent Verdict: {agent['rec']}</h3>
                    <p><b>Forward P/E:</b> {agent['pe']} | <b>Analyst Mean Target:</b> {agent['target']}</p>
                    <p>Based on volatility clusters and Sharpe efficiency, this asset presents a 
                    <b>{"High" if m['MDD'] < -25 else "Moderate"}</b> risk profile with 
                    <b>{"Optimal" if m['Sharpe'] > 1 else "Sub-optimal"}</b> risk-adjusted returns.</p>
                </div>""", unsafe_allow_html=True)
                
                fig = px.line(prices, title=f"{ticker_sym} Institutional Trajectory", template="plotly_dark")
                fig.update_traces(line_color='#FFD700', line_width=3)
                st.plotly_chart(fig, use_container_width=True)

def render_ai_forecast():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Forecast Asset Target:", "BTC-USD").upper()
    if st.button("Run Neural Training"):
        with st.spinner("Optimizing Neural Parameters..."):
            raw = SovereignEngine.format_yf(yf.download(target, period="3y", progress=False).reset_index())
            df_p = pd.DataFrame({'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 'y': raw['Close']}).dropna()
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.08).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash', color='#FFD700')))
            fig.update_layout(template="plotly_dark", title=f"90-Day Forecast: {target}")
            st.plotly_chart(fig, use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    st.write("Behavioral Capital Audit & Strategic Allocation.")
    
    df = pd.DataFrame([
        {"Description": "Executive Salary", "Amount": 15000, "Category": "Income"},
        {"Description": "Luxury Real Estate", "Amount": -4000, "Category": "Fixed"},
        {"Description": "Equity Portfolio", "Amount": -3500, "Category": "Wealth"},
        {"Description": "Lifestyle/Discretionary", "Amount": -1500, "Category": "Wants"},
        {"Description": "Venture Fund", "Amount": -1000, "Category": "Wealth"}
    ])
    
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    outflows = df[df['Amount'] < 0].copy()
    outflows['Abs'] = outflows['Amount'].abs()
    
    if not outflows.empty:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig = px.pie(outflows, values='Abs', names='Category', hole=0.6, 
                         template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            w_sum = outflows[outflows['Category'] == 'Wealth']['Abs'].sum()
            w_rate = (w_sum / outflows['Abs'].sum()) * 100
            st.metric("Wealth Creation Rate", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}%")
            if w_rate < 20: st.error("ALERT: Capital accumulation velocity below institutional standards.")
            else: st.success("STRATEGIC PASS: Wealth building is highly optimized.")

# ==========================================
# 4. MASTER NAVIGATOR
# ==========================================

def main():
    render_global_pulse()
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Navigation:", 
        ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Advisor"])
    
    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intel()
    elif nav == "Neural Forecasting": render_ai_forecast()
    elif nav == "Wealth Advisor": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Terminal Sync: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.info("Operational Status: Institutional Pipelines Active")

if __name__ == "__main__":
    main()
