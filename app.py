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
# 1. GLOBAL CONFIGURATION & SOVEREIGN STYLING
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Terminal", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92)), 
                    url('https://images.unsplash.com/photo-1639762681485-074b7f938ba0?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98); border-radius: 40px; 
        padding: 50px 80px; border: 1px solid #2a2a2a;
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; }
    h2, h3 { color: #FFD700 !important; border-left: 8px solid #FFD700; padding-left: 15px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-top: 4px solid #FFD700; }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 800; border-radius: 10px; height: 3.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE ANALYTICS ENGINE
# ==========================================================
class QuantEngine:
    @staticmethod
    def get_risk_metrics(df, col='Close'):
        rets = df[col].pct_change().dropna()
        mu, sigma = rets.mean(), rets.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma != 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        mdd = ((df[col] / df[col].cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "VaR": var_95, "MDD": mdd, "Returns": rets}

    @staticmethod
    def monte_carlo_sim(last_price, mu, sigma, days=60, sims=50):
        fig = go.Figure()
        for _ in range(sims):
            prices = [last_price]
            for _ in range(days):
                prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            fig.add_trace(go.Scatter(y=prices, mode='lines', opacity=0.15, line=dict(color='#FFD700')))
        return fig

# ==========================================================
# 3. INTERFACE MODULES
# ==========================================================

def render_risk_framework():
    st.header("🔬 Academic Risk Framework")
    col1, col2 = st.tabs(["Stochastic Models", "Tail Risk Theory"])
    with col1:
        st.subheader("Geometric Brownian Motion")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        
        st.write("This SDE models the continuous-time evolution of asset prices used in our Monte Carlo simulations.")
    with col2:
        st.subheader("Parametric Value at Risk (VaR)")
        st.latex(r"VaR_{\alpha} = \mu + \sigma \Phi^{-1}(\alpha)")
        
        st.write("Calculates the maximum potential loss at a 95% confidence interval.")

# ----------------------------------------------------------

def render_equity_intelligence():
    st.header("📈 Equity Intelligence")
    mode = st.sidebar.radio("Data Source:", ["Live Ticker", "Upload CSV"])
    
    if mode == "Live Ticker":
        ticker = st.text_input("Ticker Symbol:", "RACE").upper()
        if st.button("Analyze Live Data"):
            df = yf.download(ticker, period="2y")
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                display_equity_results(df, ticker)
    else:
        up = st.file_uploader("Upload Market Data:", type="csv")
        if up:
            df = pd.read_csv(up)
            p_col = st.selectbox("Select Price Column:", df.columns)
            if st.button("Run Audit"):
                display_equity_results(df, "Custom Asset", p_col)

def display_equity_results(df, name, col='Close'):
    m = QuantEngine.get_risk_metrics(df, col)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Price", f"{df[col].iloc[-1]:,.2f}")
    c2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
    c4.metric("VaR (95%)", f"{m['VaR']:.2f}%")
    
    st.plotly_chart(px.line(df, y=col, title=f"Audit: {name}", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(px.histogram(m['Returns'], nbins=50, title="Return Distribution", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
    with col_b:
        mc = QuantEngine.monte_carlo_sim(df[col].iloc[-1], m['Returns'].mean(), m['Returns'].std())
        st.plotly_chart(mc.update_layout(template="plotly_dark", title="Monte Carlo: 50 Stochastic Paths", showlegend=False), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    t1, t2 = st.tabs(["📝 Manual Ledger", "📥 Document Upload"])
    
    working_df = pd.DataFrame()

    with t1:
        raw_data = [{"Category": "Income", "Amount": 12000}, {"Category": "Fixed Costs", "Amount": -4000}, 
                    {"Category": "Investments", "Amount": -3000}, {"Category": "Lifestyle", "Amount": -1000}]
        working_df = st.data_editor(pd.DataFrame(raw_data), num_rows="dynamic", use_container_width=True)

    with t2:
        up_w = st.file_uploader("Upload Transaction Document (CSV):", type="csv")
        if up_w:
            file_df = pd.read_csv(up_w)
            st.info("Map your CSV columns to System columns:")
            cx, cy = st.columns(2)
            cat_col = cx.selectbox("Category Column:", file_df.columns)
            amt_col = cy.selectbox("Amount Column:", file_df.columns)
            if st.button("Process Document"):
                working_df = file_df[[cat_col, amt_col]].rename(columns={cat_col: 'Category', amt_col: 'Amount'})

    if not working_df.empty:
        try:
            working_df['Amount'] = pd.to_numeric(working_df['Amount'], errors='coerce').fillna(0)
            inc = working_df[working_df['Amount'] > 0]['Amount'].sum()
            out = working_df[working_df['Amount'] < 0].copy()
            out['Abs'] = out['Amount'].abs()
            
            if inc > 0:
                inv = out[out['Category'].str.contains('Invest|Wealth|Stock|Gold|پس‌انداز', case=False, na=False)]['Abs'].sum()
                rate = (inv / inc) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Income", f"${inc:,.0f}")
                m2.metric("Wealth Creation Rate", f"{rate:.1f}%")
                m3.metric("Net Surplus", f"${inc - out['Abs'].sum():,.0f}")
                
                st.divider()
                ca, cb = st.columns([1.5, 1])
                with ca:
                    
                    st.plotly_chart(px.pie(out, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
                with cb:
                    st.subheader("🕵️ AI Financial Verdict")
                    if rate < 20: st.warning("Wealth Rate is below the 20% institutional threshold. Reallocate lifestyle capital.")
                    else: st.success("Optimal Capital Structure detected. High wealth creation velocity.")
        except Exception as e: st.error(f"Processing Error: {e}")

# ----------------------------------------------------------

def render_neural_forecast():
    st.header("🔮 Neural Prediction Engine")
    target = st.text_input("Forecast Asset (e.g., BTC-USD):", "NVDA").upper()
    if st.button("Run Neural Training"):
        df = yf.download(target, period="3y").reset_index()
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            p_df = pd.DataFrame({'ds': df['Date'].dt.tz_localize(None), 'y': df['Close']}).dropna()
            m = Prophet(daily_seasonality=True).fit(p_df)
            future = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], name="Forecast", line=dict(color='#FFD700', dash='dash')))
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Neural Projection: {target}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, future), use_container_width=True)

# ==========================================================
# 4. MAIN CONTROLLER
# ==========================================================
def main():
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Navigation:", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_forecast()
    elif nav == "Wealth Management": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Status: Active | {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
