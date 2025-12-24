
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

# --- SETUP ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Institutional Terminal", page_icon="🏛️", layout="wide")

# ==========================================
# 1. ELITE UI/UX CUSTOMIZATION
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92)), 
                    url('https://images.unsplash.com/photo-1639754390580-2e7437267698?q=80&w=2041');
        background-size: cover; background-attachment: fixed;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98); border-radius: 35px; 
        padding: 50px 70px; border: 1px solid #333; box-shadow: 0 40px 100px rgba(0,0,0,1);
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; letter-spacing: -2px; }
    h2, h3 { color: #FFD700 !important; border-left: 6px solid #FFD700; padding-left: 20px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 25px; border-radius: 18px; border-top: 4px solid #FFD700; }
    .stButton>button { background: linear-gradient(45deg, #FFD700, #B8860B); color: black !important; font-weight: 800; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANTITATIVE ANALYTICS ENGINE
# ==========================================
class SovereignEngine:
    @staticmethod
    def fix_data(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def calculate_metrics(returns):
        if returns.empty: return None
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        cum = (1 + returns).cumprod()
        mdd = ((cum / cum.cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "MDD": mdd, "VaR": var_95, "Vol": sigma * np.sqrt(252) * 100}

# ==========================================
# 3. CORE MODULES
# ==========================================

def render_risk_framework():
    st.header("🔬 Strategic Risk Framework")
    st.markdown("Advanced mathematical models for volatility decomposition and tail-risk assessment.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("I. Geometric Brownian Motion")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Modeling asset dynamics through stochastic differential equations to predict potential tail-events.")
        
        
    with col2:
        st.subheader("II. Parametric Value at Risk (VaR)")
        st.latex(r"VaR_{95\%} = \mu + \sigma \cdot 1.645")
        st.write("Quantifying the maximum expected loss within a 95% confidence interval for institutional liquidity.")
        

def render_equity_and_upload():
    st.header("📈 Equity Intel & Custom Data Analysis")
    
    mode = st.radio("Select Data Source:", ["Live Market (Yahoo Finance)", "Upload Custom Dataset (CSV)"], horizontal=True)
    
    if mode == "Live Market (Yahoo Finance)":
        ticker = st.text_input("Enter Symbol (e.g., RACE, NVDA, BTC-USD):", "RACE").upper()
        if st.button("Initialize Live Audit"):
            with st.spinner("Fetching Market Data..."):
                tk = yf.Ticker(ticker)
                data = SovereignEngine.fix_data(tk.history(period="2y"))
                process_and_plot(data, ticker, tk.info)
                
    else:
        up = st.file_uploader("Upload CSV File (Must contain 'Date' and 'Close' columns):", type="csv")
        if up:
            data = pd.read_csv(up)
            if 'Date' in data.columns and 'Close' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
                process_and_plot(data, "Custom Upload", None)
            else:
                st.error("Invalid Format. Ensure columns 'Date' and 'Close' exist.")

def process_and_plot(df, name, info):
    rets = df['Close'].pct_change().dropna()
    m = SovereignEngine.calculate_metrics(rets)
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Price", f"{df['Close'].iloc[-1]:,.2f}")
    k2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
    k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
    k4.metric("Daily VaR 95%", f"{m['VaR']:.2f}%")
    
    # Live Agent Verdict
    if info:
        st.markdown(f"""<div style="background: rgba(255,215,0,0.05); padding: 25px; border-radius: 20px; border: 1px dashed #FFD700; margin: 20px 0;">
            <h3>🕵️ Sovereign Agent Verdict: {info.get('recommendationKey', 'N/A').upper()}</h3>
            <p><b>Forward P/E:</b> {info.get('forwardPE', 'N/A')} | <b>Analyst Target:</b> {info.get('targetMeanPrice', 'N/A')}</p>
        </div>""", unsafe_allow_html=True)
    
    # Plotting
    st.plotly_chart(px.line(df, y='Close', title=f"{name} Performance", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
    
    # Restored Distribution Plot
    st.subheader("📊 Return Distribution & Volatility Clusters")
    fig_dist = px.histogram(rets, nbins=100, marginal="violin", title="Return Distribution Analysis", template="plotly_dark", color_discrete_sequence=['#FFD700'])
    st.plotly_chart(fig_dist, use_container_width=True)

def render_ai_forecast():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Asset for 90-Day Projection:", "BTC-USD").upper()
    if st.button("Generate Neural Forecast"):
        with st.spinner("Training Model..."):
            raw = yf.download(target, period="3y", progress=False).reset_index()
            df_p = pd.DataFrame({'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 'y': raw['Close']}).dropna()
            m = Prophet(daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color='#FFD700', dash='dash')))
            st.plotly_chart(fig.update_layout(template="plotly_dark"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Behavioral Wealth Advisor")
    df = pd.DataFrame([
        {"Category": "Income", "Amount": 15000}, {"Category": "Fixed", "Amount": -4500},
        {"Category": "Wealth", "Amount": -3500}, {"Category": "Wants", "Amount": -1500}
    ])
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    outflows = df[df['Amount'] < 0].copy()
    outflows['Abs'] = outflows['Amount'].abs()
    
    if not outflows.empty:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.plotly_chart(px.pie(outflows, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
        with c2:
            w_rate = (outflows[outflows['Category'] == 'Wealth']['Abs'].sum() / outflows['Abs'].sum()) * 100
            st.metric("Wealth Creation Rate", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}%")

# ==========================================
# 4. MASTER CONTROLLER
# ==========================================
def main():
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Navigation:", ["Risk Framework", "Equity & Upload", "Neural Forecasting", "Wealth Advisor"])
    
    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity & Upload": render_equity_and_upload()
    elif nav == "Neural Forecasting": render_ai_forecast()
    elif nav == "Wealth Advisor": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
