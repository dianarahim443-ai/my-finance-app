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

# --- ENV & SETUP ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Diana Sovereign AI | Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. PREMIUM CSS & GLOBAL UI
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700;900&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.94), rgba(0,0,0,0.94)), 
                    url('https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2070');
        background-size: cover;
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: rgba(8, 8, 8, 0.97);
        border-radius: 40px;
        padding: 50px 70px;
        border: 1px solid #2a2a2a;
        box-shadow: 0 40px 120px rgba(0,0,0,1);
    }
    
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.2rem !important; letter-spacing: -2px; }
    h2, h3 { color: #FFD700 !important; font-weight: 700; border-left: 6px solid #FFD700; padding-left: 20px; }
    
    .stMetric { 
        background: rgba(255,255,255,0.03); 
        padding: 25px; 
        border-radius: 18px; 
        border-top: 4px solid #FFD700; 
    }
    
    .agent-box { 
        background: rgba(255, 215, 0, 0.05); 
        padding: 30px; 
        border-radius: 20px; 
        border: 1px dashed #FFD700;
        margin-bottom: 25px;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 30px; }
    .stTabs [data-baseweb="tab--active"] { color: #FFD700 !important; border-bottom-color: #FFD700 !important; }

    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 800; border-radius: 10px;
        height: 3em; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(255,215,0,0.3); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANT ENGINE (THE BRAIN)
# ==========================================
class SovereignEngine:
    @staticmethod
    def fix_data(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def get_stats(returns):
        if returns.empty: return None
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        cum = (1 + returns).cumprod()
        mdd = ((cum / cum.cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "MDD": mdd, "VaR": var_95, "Vol": sigma * np.sqrt(252) * 100}

# ==========================================
# 3. INTERFACE MODULES
# ==========================================

def render_dashboard():
    st.title("🏛️ Diana Sovereign")
    st.markdown("##### *Institutional Multi-Asset Quantitative Research Terminal*")
    
    tickers = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    cols = st.columns(len(tickers))
    for i, (name, sym) in enumerate(tickers.items()):
        try:
            data = SovereignEngine.fix_data(yf.download(sym, period="2d", progress=False))
            price, change = data['Close'].iloc[-1], ((data['Close'].iloc[-1]/data['Close'].iloc[-2])-1)*100
            cols[i].metric(name, f"{price:,.2f}", f"{change:+.2f}%")
        except: pass
    st.divider()

def render_risk_framework():
    st.header("🔬 Strategic Risk Framework")
    st.write("Advanced mathematical modeling for capital preservation and risk decomposition.")
    
    tab1, tab2 = st.tabs(["Stochastic Volatility", "Institutional Metrics"])
    with tab1:
        st.subheader("Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.info("GBM assumes that a constant drift and volatility drive the price evolution through a Wiener process.")
        
        
    with tab2:
        st.subheader("Value at Risk (VaR) Methodology")
        st.latex(r"VaR_{1-\alpha} = \inf \{ l \in \mathbb{R} : P(L > l) \le \alpha \}")
        st.write("We utilize parametric VaR at a 95% confidence interval to stress-test daily liquidity.")
        

def render_equity_intelligence():
    st.header("📈 Equity Intel & AI Agent")
    col1, col2 = st.columns([3, 1])
    ticker = col1.text_input("Enter Asset Symbol (e.g., RACE, ENI.MI, NVDA, AAPL):", "RACE").upper()
    period = col2.selectbox("History:", ["1y", "2y", "5y", "max"])
    
    if st.button("Execute Deep Analysis"):
        with st.spinner("Agent analyzing market structure..."):
            asset = yf.Ticker(ticker)
            hist = SovereignEngine.fix_data(asset.history(period=period))
            
            if not hist.empty:
                rets = hist['Close'].pct_change().dropna()
                stats = SovereignEngine.get_stats(rets)
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sharpe Ratio", f"{stats['Sharpe']:.2f}")
                m2.metric("Ann. Volatility", f"{stats['Vol']:.1f}%")
                m3.metric("Max Drawdown", f"{stats['MDD']:.2f}%")
                m4.metric("Daily VaR (95%)", f"{stats['VaR']:.2f}%")
                
                # Agent Verdict
                info = asset.info
                rec = info.get('recommendationKey', 'N/A').upper()
                st.markdown(f"""<div class="agent-box">
                    <h3>🕵️ Sovereign Agent Verdict: {rec}</h3>
                    <p><b>Forward P/E:</b> {info.get('forwardPE', 'N/A')} | <b>Target Price:</b> {info.get('targetMeanPrice', 'N/A')}</p>
                    <p>The asset shows <b>{"High" if stats['MDD'] < -25 else "Stable"}</b> resilience with 
                    <b>{"High" if stats['Sharpe'] > 1.2 else "Moderate"}</b> reward-to-risk efficiency.</p>
                </div>""", unsafe_allow_html=True)
                
                # Charts
                fig = px.line(hist['Close'], title=f"{ticker} Performance Trajectory", template="plotly_dark")
                fig.update_traces(line_color='#FFD700', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
                
                # Monte Carlo Simulation (The "Deleted" Code restored)
                st.subheader("🎲 Monte Carlo Stress Simulation")
                last_p, mu, sigma = hist['Close'].iloc[-1], rets.mean(), rets.std()
                sim_fig = go.Figure()
                for _ in range(50):
                    path = [last_p]
                    for _ in range(30): path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
                    sim_fig.add_trace(go.Scatter(y=path, mode='lines', opacity=0.15, line=dict(color='#FFD700')))
                sim_fig.update_layout(title="30-Day Forward Stochastic Paths", template="plotly_dark", showlegend=False)
                st.plotly_chart(sim_fig, use_container_width=True)

def render_ai_forecast():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Forecast Target (Symbol):", "BTC-USD").upper()
    if st.button("Run Prophet V5 Prediction"):
        with st.spinner("Training Neural Network..."):
            df = yf.download(target, period="3y", progress=False).reset_index()
            df = SovereignEngine.fix_data(df)
            df_p = pd.DataFrame({'ds': pd.to_datetime(df['Date']).dt.tz_localize(None), 'y': df['Close']}).dropna()
            
            m = Prophet(daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color='#FFD700', dash='dash')))
            fig.update_layout(template="plotly_dark", title=f"90-Day Neural Price Projection: {target}")
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Wealth Management")
    st.write("Behavioral Capital Audit & Automated Allocation.")
    
    data = [
        {"Category": "Fixed", "Amount": -4500},
        {"Category": "Wealth", "Amount": -3200},
        {"Category": "Wants", "Amount": -1800},
        {"Category": "Income", "Amount": 12000}
    ]
    df = st.data_editor(pd.DataFrame(data), num_rows="dynamic", use_container_width=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    outflows = df[df['Amount'] < 0].copy()
    outflows['Abs'] = outflows['Amount'].abs()
    
    if not outflows.empty:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.plotly_chart(px.pie(outflows, values='Abs', names='Category', hole=0.5, 
                                 template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
        with c2:
            w_rate = (outflows[outflows['Category'] == 'Wealth']['Abs'].sum() / outflows['Abs'].sum()) * 100
            st.metric("Wealth Building Velocity", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}%")
            if w_rate < 20: st.error("Strategic Alert: Low Capital Accumulation.")
            else: st.success("Audit Pass: Optimal Wealth Allocation.")

# ==========================================
# 4. MAIN ROUTER
# ==========================================
def main():
    render_dashboard()
    nav = st.sidebar.radio("Navigation", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Advisor"])
    
    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_ai_forecast()
    elif nav == "Wealth Advisor": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Sync: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.info("Operational: High-Performance Multi-Asset Engine Active")

if __name__ == "__main__":
    main()
