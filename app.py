
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime
from scipy.stats import norm, skew
import warnings
import time

# ==========================================================
# 1. CORE SYSTEM CONFIGURATION (Zero-Error Architecture)
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Terminal", page_icon="🏛️", layout="wide")

# Institutional Visual Identity
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&family=Inter:wght@400;900&display=swap');
    .stApp { background-color: #050505; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    .main .block-container { background: rgba(10, 10, 10, 0.98); border-radius: 40px; padding: 40px 60px; border: 1px solid #2a2a2a; }
    .header-gold { font-weight: 900; font-size: 5rem; background: linear-gradient(#FFD700, #B8860B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -3px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-top: 4px solid #FFD700; border-left: 1px solid #333; }
    .stButton>button { background: linear-gradient(45deg, #FFD700, #B8860B); color: black !important; font-weight: 800; border: none; border-radius: 10px; height: 3.5em; transition: 0.3s; }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(255,215,0,0.4); }
    .report-box { background: rgba(255,215,0,0.02); border: 1px dashed #FFD700; border-radius: 15px; padding: 25px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. SOVEREIGN QUANTITATIVE ENGINE
# ==========================================================
class SovereignEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_global_data(ticker, period="2y"):
        try:
            data = yf.download(ticker, period=period, progress=False)
            if data.empty: return None
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            return data
        except: return None

    @staticmethod
    def run_monte_carlo(last_p, mu, sigma, days=60, sims=60):
        results = []
        for _ in range(sims):
            path = [last_p]
            for _ in range(days):
                path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            results.append(path)
        return results

# ==========================================================
# 3. INTERFACE MODULES
# ==========================================================

def render_risk_theory():
    st.markdown('<h1 class="header-gold">RISK FRAMEWORK</h1>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Stochastic Calculus (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Modeling asset drift and volatility via Brownian Motion.")
        
    with c2:
        st.subheader("Tail Risk Analysis")
        st.latex(r"VaR_{\alpha} = \mu + \sigma \cdot \Phi^{-1}(\alpha)")
        st.write("Quantifying the maximum potential loss at 95% confidence.")
        

# ----------------------------------------------------------

def render_equity_audit():
    st.markdown('<h1 class="header-gold">EQUITY AUDIT</h1>', unsafe_allow_html=True)
    
    # Universal Asset Search Engine
    with st.container():
        sc1, sc2 = st.columns([3, 1])
        ticker = sc1.text_input("SEARCH GLOBAL ASSET (Stock, Crypto, FX, ETF):", "TSLA").upper()
        lookback = sc2.selectbox("TIME HORIZON:", ["1y", "2y", "5y", "max"])
        
    if st.button("RUN INSTITUTIONAL AUDIT"):
        df = SovereignEngine.fetch_global_data(ticker, lookback)
        if df is not None:
            # Metrics Logic
            rets = df['Close'].pct_change().dropna()
            mu, sigma = rets.mean(), rets.std()
            sharpe = (mu / sigma) * np.sqrt(252)
            var_95 = norm.ppf(0.05, mu, sigma) * 100
            mdd = ((df['Close'] / df['Close'].cummax()) - 1).min() * 100
            
            # Dashboard
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last Price", f"${df['Close'].iloc[-1]:,.2f}")
            m2.metric("Sharpe Efficiency", f"{sharpe:.2f}")
            m3.metric("Tail Risk (VaR)", f"{var_95:.2f}%")
            m4.metric("Max Drawdown", f"{mdd:.1f}%")
            
            # Visuals
            st.plotly_chart(px.line(df, y='Close', title=f"Historical Trajectory: {ticker}", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.plotly_chart(px.histogram(rets, nbins=100, title="Return Density", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
            with c_right:
                mc_paths = SovereignEngine.run_monte_carlo(df['Close'].iloc[-1], mu, sigma)
                fig_mc = go.Figure()
                for p in mc_paths: fig_mc.add_trace(go.Scatter(y=p, mode='lines', opacity=0.1, line=dict(color='#FFD700')))
                st.plotly_chart(fig_mc.update_layout(template="plotly_dark", title="Monte Carlo Simulations (60D)", showlegend=False), use_container_width=True)
        else:
            st.error("Ticker not found. Ensure correct symbol (e.g., AAPL, BTC-USD).")

# ----------------------------------------------------------

def render_neural_prediction():
    st.markdown('<h1 class="header-gold">NEURAL ENGINE</h1>', unsafe_allow_html=True)
    target = st.text_input("Enter Predictive Target:", "NVDA").upper()
    if st.button("EXECUTE NEURAL FORECAST"):
        with st.spinner("Training Proprietary Neural Model..."):
            df = SovereignEngine.fetch_global_data(target, "3y")
            if df is not None:
                p_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                p_df['ds'] = p_df['ds'].dt.tz_localize(None)
                m = Prophet(daily_seasonality=True).fit(p_df)
                forecast = m.predict(m.make_future_dataframe(periods=90))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name="Actual", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color='#FFD700', dash='dash')))
                st.plotly_chart(fig.update_layout(template="plotly_dark", title="90-Day Neural Projection"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_management():
    st.markdown('<h1 class="header-gold">WEALTH ADVISOR</h1>', unsafe_allow_html=True)
    tab_doc, tab_manual = st.tabs(["📥 Smart Doc Processor", "📝 Sovereign Ledger"])
    
    wealth_data = pd.DataFrame()

    with tab_doc:
        st.write("Upload any financial CSV. The AI will auto-map your structure.")
        up_file = st.file_uploader("Drop Document Here", type="csv")
        if up_file:
            raw_df = pd.read_csv(up_file)
            st.dataframe(raw_df.head(3), use_container_width=True)
            st.info("Mapping: Identify the Category and Amount columns")
            cm1, cm2 = st.columns(2)
            c_col = cm1.selectbox("Category Column:", raw_df.columns)
            a_col = cm2.selectbox("Amount Column:", raw_df.columns)
            if st.button("SYNC DATA"):
                wealth_data = raw_df[[c_col, a_col]].rename(columns={c_col: 'Category', a_col: 'Amount'})

    with tab_manual:
        defaults = [{"Category": "Income", "Amount": 15000}, {"Category": "Rent", "Amount": -4000}, 
                    {"Category": "Investments", "Amount": -3000}, {"Category": "Luxury", "Amount": -2000}]
        wealth_data = st.data_editor(pd.DataFrame(defaults), num_rows="dynamic", use_container_width=True)

    if not wealth_data.empty:
        try:
            wealth_data['Amount'] = pd.to_numeric(wealth_data['Amount'], errors='coerce').fillna(0)
            income = wealth_data[wealth_data['Amount'] > 0]['Amount'].sum()
            outflows = wealth_data[wealth_data['Amount'] < 0].copy()
            outflows['Abs'] = outflows['Amount'].abs()
            
            if income > 0:
                inv_total = outflows[outflows['Category'].str.contains('Invest|Wealth|Stock|Gold|Save', case=False, na=False)]['Abs'].sum()
                w_rate = (inv_total / income) * 100
                
                # Visual Audit
                r1, r2, r3 = st.columns(3)
                r1.metric("Gross Revenue", f"${income:,.0f}")
                r2.metric("Wealth Rate", f"{w_rate:.1f}%")
                r3.metric("Net Surplus", f"${income - outflows['Abs'].sum():,.0f}")
                
                st.divider()
                st.plotly_chart(px.pie(outflows, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
                
                with st.expander("🕵️ AI Strategic Verdict"):
                    if w_rate < 20: st.error("STRATEGIC ALERT: Wealth creation rate is below institutional standards (20%). Reduce lifestyle drift.")
                    else: st.success("SOVEREIGN STATUS: Your capital allocation model is highly efficient.")
        except Exception as e:
            st.error(f"Mapping Error: {e}")

# ==========================================================
# 4. MASTER CONTROLLER
# ==========================================================
def main():
    # Market Pulse Ribbon
    pulse_data = {"S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "Gold": "GC=F"}
    cols = st.columns(len(pulse_data))
    for i, (k, v) in enumerate(pulse_data.items()):
        try:
            d = SovereignEngine.fetch_global_data(v, "2d")
            cols[i].metric(k, f"{d['Close'].iloc[-1]:,.2f}", f"{(d['Close'].iloc[-1]/d['Close'].iloc[-2]-1)*100:+.2f}%")
        except: pass
    st.divider()

    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Command Center:", ["Risk Framework", "Equity Audit", "Neural Prediction", "Wealth Advisor"])
    
    if nav == "Risk Framework": render_risk_theory()
    elif nav == "Equity Audit": render_equity_audit()
    elif nav == "Neural Prediction": render_neural_prediction()
    elif nav == "Wealth Advisor": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"System Operational | UTC: {datetime.utcnow().strftime('%H:%M')}")

if __name__ == "__main__":
    main()
