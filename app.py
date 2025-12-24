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
import warnings
import io

# ==========================================================
# 1. SYSTEM INITIALIZATION & ULTRA-PREMIUM STYLE
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Magnum Opus", page_icon="🏛️", layout="wide")

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
        padding: 60px 80px; border: 1px solid #2a2a2a; box-shadow: 0 40px 150px rgba(0,0,0,1);
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 5rem !important; letter-spacing: -3px; }
    h2, h3 { color: #FFD700 !important; border-left: 10px solid #FFD700; padding-left: 25px; margin-top: 40px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 30px; border-radius: 20px; border-top: 5px solid #FFD700; }
    .agent-verdict { background: rgba(255, 215, 0, 0.05); border: 1px dashed #FFD700; border-radius: 20px; padding: 30px; }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 900; border-radius: 12px; height: 3.5em; width: 100%; transition: 0.4s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(255,215,0,0.4); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE SOVEREIGN ENGINE
# ==========================================================
class SovereignEngine:
    @staticmethod
    def calculate_risk_ratios(df, col='Close'):
        returns = df[col].pct_change().dropna()
        if returns.empty: return None
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        cum_rets = (1 + returns).cumprod()
        mdd = ((cum_rets / cum_rets.cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "MDD": mdd, "VaR": var_95, "Vol": sigma * np.sqrt(252) * 100, "Returns": returns}

    @staticmethod
    def run_monte_carlo(last_price, mu, sigma, days=45, sims=70):
        mc_fig = go.Figure()
        for _ in range(sims):
            path = [last_price]
            for _ in range(days):
                path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            mc_fig.add_trace(go.Scatter(y=path, mode='lines', opacity=0.1, line=dict(color='#FFD700')))
        return mc_fig

# ==========================================================
# 3. INTERFACE MODULES (FULL COMPLEMENT)
# ==========================================================

def render_market_pulse():
    watch = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD"}
    cols = st.columns(len(watch))
    for i, (name, sym) in enumerate(watch.items()):
        try:
            d = yf.download(sym, period="2d", progress=False)
            p = d['Close'].iloc[-1]
            c = (p / d['Close'].iloc[-2] - 1) * 100
            cols[i].metric(name, f"{p:,.2f}", f"{c:+.2f}%")
        except: pass
    st.divider()

# ----------------------------------------------------------

def render_risk_framework():
    st.header("🔬 Academic Risk Framework")
    t1, t2, t3 = st.tabs(["Stochastic Calculus", "Tail-Risk Theory", "Neural Decomposition"])
    with t1:
        st.subheader("I. Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Modeling asset prices as continuous-time Markov processes.")
        
    with t2:
        st.subheader("II. Parametric Value at Risk")
        st.latex(r"VaR_{\alpha} = \mu + \sigma \cdot \Phi^{-1}(\alpha)")
        st.write("Calculated at 95% confidence interval for institutional capital preservation.")
    with t3:
        st.subheader("III. Multi-Factor Neural Forecasting")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("Decomposing signals into Trend, Seasonality, and Holidays.")

# ----------------------------------------------------------

def render_equity_intelligence():
    st.header("📈 Equity Intelligence & Custom Audit")
    source = st.sidebar.selectbox("Data Source:", ["Live Terminal", "Institutional Upload (CSV)"])
    
    if source == "Live Terminal":
        ticker = st.text_input("Enter Ticker:", "RACE").upper()
        if st.button("Initialize Deep Run"):
            df = yf.download(ticker, period="2y")
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                execute_analysis(df, ticker)
    else:
        up_file = st.file_uploader("Upload Price History CSV:", type="csv")
        if up_file:
            df = pd.read_csv(up_file)
            col_p = st.selectbox("Select Price Column:", df.columns)
            if st.button("Audit Uploaded Data"):
                execute_analysis(df, "Custom Asset", col_p)

def execute_analysis(df, label, col='Close'):
    m = SovereignEngine.calculate_risk_ratios(df, col)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest Value", f"{df[col].iloc[-1]:,.2f}")
    k2.metric("Sharpe Efficiency", f"{m['Sharpe']:.2f}")
    k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
    k4.metric("Daily VaR (95%)", f"{m['VaR']:.2f}%")
    
    st.plotly_chart(px.line(df, y=col, title=f"{label} Trajectory", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
    
    st.subheader("📊 Return Density & Monte Carlo Stress Test")
    c_a, c_b = st.columns(2)
    with c_a:
        
        st.plotly_chart(px.histogram(m['Returns'], nbins=80, title="Return Distribution", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
    with c_b:
        mc_fig = SovereignEngine.run_monte_carlo(df[col].iloc[-1], m['Returns'].mean(), m['Returns'].std())
        st.plotly_chart(mc_fig.update_layout(template="plotly_dark", title="70 Stochastic Trajectories", showlegend=False), use_container_width=True)

# ----------------------------------------------------------

def render_neural_prediction():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("90-Day Neural Forecast Target:", "BTC-USD").upper()
    if st.button("Deploy Prophet Model"):
        with st.spinner("Training Neural Model..."):
            raw = yf.download(target, period="3y", progress=False).reset_index()
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                df_p = pd.DataFrame({'ds': raw['Date'].dt.tz_localize(None), 'y': raw['Close']}).dropna()
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=90))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color='#FFD700', dash='dash')))
                st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Forecast Dynamics: {target}"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    t1, t2 = st.tabs(["📝 Manual Ledger", "📥 Smart CSV Upload"])
    data = pd.DataFrame()
    
    with t1:
        default = [{"Category": "Income", "Amount": 15000}, {"Category": "Fixed", "Amount": -4500}, {"Category": "Wealth", "Amount": -3500}, {"Category": "Wants", "Amount": -1500}]
        data = st.data_editor(pd.DataFrame(default), num_rows="dynamic", use_container_width=True)
        
    with t2:
        up_w = st.file_uploader("Upload Wealth CSV:", type="csv")
        if up_w:
            raw_w = pd.read_csv(up_w)
            c_cat = st.selectbox("Category Column:", raw_w.columns)
            c_amt = st.selectbox("Amount Column:", raw_w.columns)
            if st.button("Process Wealth File"):
                data = raw_w[[c_cat, c_amt]].rename(columns={c_cat: 'Category', c_amt: 'Amount'})

    if not data.empty:
        try:
            data['Amount'] = pd.to_numeric(data['Amount'])
            inc = data[data['Amount'] > 0]['Amount'].sum()
            out = data[data['Amount'] < 0].copy()
            out['Abs'] = out['Amount'].abs()
            if inc > 0:
                w_val = out[out['Category'].str.contains('Wealth|Invest|پس‌انداز', case=False, na=False)]['Abs'].sum()
                w_rate = (w_val / inc) * 100
                st.metric("Wealth Creation Rate", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}% (Target: 20%)")
                st.plotly_chart(px.pie(out, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
        except: pass

# ==========================================================
# 4. MASTER ROUTER
# ==========================================================
def main():
    render_market_pulse()
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Navigation Domains:", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_prediction()
    elif nav == "Wealth Management": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.info(f"Terminal Active | {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
