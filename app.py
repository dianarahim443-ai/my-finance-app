
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
import time

# ==========================================================
# 1. CORE ARCHITECTURE & TERMINAL STYLING
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Terminal", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono&family=Inter:wght@400;900&display=swap');
    .stApp { background-color: #050505; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    .main .block-container { background: rgba(10, 10, 10, 0.98); border-radius: 40px; padding: 40px 60px; border: 1px solid #2a2a2a; }
    .header-gold { font-weight: 900; font-size: 4.5rem; background: linear-gradient(#FFD700, #B8860B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -3px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-top: 4px solid #FFD700; }
    .stButton>button { background: linear-gradient(45deg, #FFD700, #B8860B); color: black !important; font-weight: 800; border-radius: 10px; height: 3.5em; width: 100%; transition: 0.3s; }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(255,215,0,0.4); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. ADVANCED QUANTITATIVE ENGINE
# ==========================================================
class SovereignQuant:
    @staticmethod
    def get_data(ticker, period="2y"):
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df
        except: return None

    @staticmethod
    def compute_risk(df, col='Close'):
        returns = df[col].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        metrics = {
            "Price": df[col].iloc[-1],
            "Sharpe": (mu / sigma) * np.sqrt(252) if sigma != 0 else 0,
            "VaR": norm.ppf(0.05, mu, sigma) * 100,
            "MDD": ((df[col] / df[col].cummax()) - 1).min() * 100,
            "Vol": sigma * np.sqrt(252) * 100,
            "Returns": returns
        }
        return metrics

# ==========================================================
# 3. INTERFACE MODULES
# ==========================================================

def render_risk_framework():
    st.markdown('<h1 class="header-gold">RISK THEORY</h1>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Stochastic Models", "Tail Risk Probability"])
    with t1:
        st.subheader("Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Modeling asset price evolution through drift ($\mu$) and diffusion ($\sigma$).")
        
    with t2:
        st.subheader("Parametric Value at Risk")
        st.latex(r"VaR_{\alpha} = \mu + \sigma \cdot \Phi^{-1}(\alpha)")
        st.write("Institutional standard for measuring the threshold of extreme market losses.")
        

# ----------------------------------------------------------

def render_equity_audit():
    st.markdown('<h1 class="header-gold">EQUITY AUDIT</h1>', unsafe_allow_html=True)
    
    # 1. UNIVERSAL GLOBAL SEARCH
    c_s1, c_s2 = st.columns([3, 1])
    ticker = c_s1.text_input("GLOBAL SEARCH (Search any Stock, Crypto, FX, ETF):", "AAPL").upper()
    timeframe = c_s2.selectbox("HORIZON:", ["1y", "2y", "5y", "max"])
    
    if st.button("INITIALIZE INSTITUTIONAL AUDIT"):
        df = SovereignQuant.get_data(ticker, timeframe)
        if df is not None:
            m = SovereignQuant.compute_risk(df)
            
            # Dashboard Metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current Price", f"${m['Price']:,.2f}")
            k2.metric("Sharpe Efficiency", f"{m['Sharpe']:.2f}")
            k3.metric("Tail Risk (VaR)", f"{m['VaR']:.2f}%")
            k4.metric("Max Drawdown", f"{m['MDD']:.2f}%")
            
            # Charts
            st.plotly_chart(px.line(df, y='Close', title=f"Audit Trajectory: {ticker}", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
            
            c_left, c_right = st.columns(2)
            with c_left:
                st.plotly_chart(px.histogram(m['Returns'], nbins=80, title="Return Distribution Density", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
            with c_right:
                # Monte Carlo Simulations
                sims = []
                for _ in range(50):
                    p = [m['Price']]
                    for _ in range(60): p.append(p[-1] * np.exp((m['Returns'].mean() - 0.5 * m['Returns'].std()**2) + m['Returns'].std() * np.random.normal()))
                    sims.append(p)
                fig_mc = go.Figure()
                for s in sims: fig_mc.add_trace(go.Scatter(y=s, mode='lines', opacity=0.1, line=dict(color='#FFD700')))
                st.plotly_chart(fig_mc.update_layout(template="plotly_dark", title="Monte Carlo: 50 Stochastic Paths (60D)", showlegend=False), use_container_width=True)
        else:
            st.error("Invalid Ticker. Please use Yahoo Finance symbols (e.g., TSLA, BTC-USD, GC=F).")

# ----------------------------------------------------------

def render_neural_prediction():
    st.markdown('<h1 class="header-gold">NEURAL ENGINE</h1>', unsafe_allow_html=True)
    target = st.text_input("Enter Forecast Target (e.g. NVDA):", "BTC-USD").upper()
    if st.button("RUN DEEP LEARNING MODEL"):
        with st.spinner("Training Neural Prophet..."):
            df = SovereignQuant.get_data(target, "3y")
            if df is not None:
                p_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                p_df['ds'] = p_df['ds'].dt.tz_localize(None)
                m = Prophet(daily_seasonality=True).fit(p_df)
                forecast = m.predict(m.make_future_dataframe(periods=90))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name="Actual", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Neural Path", line=dict(color='#FFD700', dash='dash')))
                st.plotly_chart(fig.update_layout(template="plotly_dark", title="90-Day Neural Projection"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_advisor():
    st.markdown('<h1 class="header-gold">WEALTH ADVISOR</h1>', unsafe_allow_html=True)
    t_up, t_man = st.tabs(["📥 Smart Document Processing", "📝 Sovereign Ledger"])
    
    final_df = pd.DataFrame()

    with t_up:
        # 2. BULLETPROOF COLUMN MAPPING
        up_file = st.file_uploader("Upload Transaction CSV:", type="csv")
        if up_file:
            raw = pd.read_csv(up_file)
            st.dataframe(raw.head(3), use_container_width=True)
            st.info("Identify Category and Amount columns to synchronize data:")
            c1, c2 = st.columns(2)
            cat_col = c1.selectbox("Category Column:", raw.columns)
            amt_col = c2.selectbox("Amount Column:", raw.columns)
            if st.button("SYNC DOCUMENT"):
                final_df = raw[[cat_col, amt_col]].rename(columns={cat_col: 'Category', amt_col: 'Amount'})

    with t_man:
        defaults = [{"Category": "Income", "Amount": 15000}, {"Category": "Fixed Costs", "Amount": -4500}, 
                   {"Category": "Wealth/Stocks", "Amount": -3500}, {"Category": "Lifestyle", "Amount": -2000}]
        final_df = st.data_editor(pd.DataFrame(defaults), num_rows="dynamic", use_container_width=True)

    if not final_df.empty:
        try:
            final_df['Amount'] = pd.to_numeric(final_df['Amount'], errors='coerce').fillna(0)
            income = final_df[final_df['Amount'] > 0]['Amount'].sum()
            expenses = final_df[final_df['Amount'] < 0].copy()
            expenses['Abs'] = expenses['Amount'].abs()
            
            if income > 0:
                wealth_inv = expenses[expenses['Category'].str.contains('Wealth|Invest|Stock|Gold|Save', case=False, na=False)]['Abs'].sum()
                rate = (wealth_inv / income) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Income", f"${income:,.0f}")
                m2.metric("Wealth Rate", f"{rate:.1f}%")
                m3.metric("Net Savings", f"${income - expenses['Abs'].sum():,.0f}")
                
                st.divider()
                st.plotly_chart(px.pie(expenses, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
                
                if rate < 20: st.warning("Advisory: Wealth rate below 20%. Capital velocity is suboptimal.")
                else: st.success("Sovereign Grade: Optimal capital allocation detected.")
        except Exception as e: st.error(f"Mapping Failed: {e}")

# ==========================================================
# 4. MASTER CONTROLLER
# ==========================================================
def main():
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Command Center:", ["Risk Framework", "Equity Audit", "Neural Prediction", "Wealth Management"])
    
    # Global Pulse Ribbon
    pulse = {"S&P 500": "^GSPC", "Gold": "GC=F", "Bitcoin": "BTC-USD"}
    cols = st.columns(len(pulse))
    for i, (k, v) in enumerate(pulse.items()):
        try:
            d = SovereignQuant.get_data(v, "2d")
            cols[i].metric(k, f"{d['Close'].iloc[-1]:,.2f}", f"{(d['Close'].iloc[-1]/d['Close'].iloc[-2]-1)*100:+.2f}%")
        except: pass
    st.divider()

    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Audit": render_equity_audit()
    elif nav == "Neural Prediction": render_neural_prediction()
    elif nav == "Wealth Management": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"System Operational | UTC: {datetime.utcnow().strftime('%H:%M')}")

if __name__ == "__main__":
    main()
