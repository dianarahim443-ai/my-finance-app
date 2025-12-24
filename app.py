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
# 1. SYSTEM & UI ARCHITECTURE
# ==========================================
st.set_page_config(page_title="Diana Finance AI | Sovereign Ultimate", layout="wide")

# Institutional Dark UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(8, 8, 8, 0.98);
        border-radius: 30px; padding: 50px; border: 1px solid #333;
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4rem !important; letter-spacing: -2px; }
    h2, h3 { color: #E0E0E0 !important; border-left: 5px solid #FFD700; padding-left: 15px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 25px; border-radius: 15px; border-top: 5px solid #FFD700; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANTITATIVE RISK & ANALYTICS ENGINE
# ==========================================

class DianaQuantEngine:
    @staticmethod
    def clean_data(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def get_risk_metrics(returns):
        if returns.empty: return None
        rf = 0.04 / 252 # 4% Annual Risk-Free Rate
        mu, sigma = returns.mean(), returns.std()
        
        # Core Ratios
        sharpe = (mu - rf) / sigma * np.sqrt(252) if sigma != 0 else 0
        downside = returns[returns < 0].std()
        sortino = (mu - rf) / downside * np.sqrt(252) if downside > 0 else 0
        
        # Drawdown & VaR
        cum_rets = (1 + returns).cumprod()
        mdd = ((cum_rets / cum_rets.cummax()) - 1).min() * 100
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        
        return {"Sharpe": sharpe, "Sortino": sortino, "MDD": mdd, "VaR": var_95}

# ==========================================
# 3. INTERFACE MODULES
# ==========================================

def render_market_pulse():
    st.title("🏛️ Diana Sovereign")
    st.markdown("#### *Institutional Multi-Asset Quantitative Research*")
    
    indices = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "10Y Yield": "^TNX"}
    cols = st.columns(len(indices))
    for i, (name, sym) in enumerate(indices.items()):
        try:
            df = DianaQuantEngine.clean_data(yf.download(sym, period="2d", progress=False))
            p, c = df['Close'].iloc[-1], ((df['Close'].iloc[-1]/df['Close'].iloc[-2])-1)*100
            cols[i].metric(name, f"{p:,.2f}", f"{c:+.2f}%")
        except: pass
    st.divider()

def render_methodology():
    st.header("🔬 Academic Framework & Risk Methodology")
    t1, t2, t3 = st.tabs(["Stochastic Logic", "Neural Prediction", "Risk Parity Framework"])
    
    with t1:
        st.subheader("I. Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Used to solve for potential future price paths by integrating historical drift and volatility.")
        
        
    with t2:
        st.subheader("II. Time-Series Deconvolution")
        st.latex(r"f(x) = g(t) + s(t) + h(t) + \epsilon")
        st.write("Prophet Architecture used for decomposing market trends from seasonal noise.")
        

    with t3:
        st.subheader("III. Advanced Risk Framework")
        st.write("Our framework implements the following metrics for institutional-grade auditing:")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Sharpe Ratio:** Measures excess return per unit of total risk.")
            st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")
        with col_r2:
            st.markdown("**Value at Risk (VaR):** Quantifies potential loss at 95% confidence.")
            st.latex(r"VaR_{95\%} = \mu + \sigma \cdot N^{-1}(0.05)")
        

def render_equity_intel():
    st.header("📈 Equity Intelligence & Backtesting")
    ticker = st.text_input("Institutional Ticker:", "NVDA").upper()
    
    if st.button("Initialize Deep Run"):
        data = DianaQuantEngine.clean_data(yf.download(ticker, period="2y", progress=False))
        if not data.empty:
            prices = data['Close'].squeeze()
            returns = prices.pct_change().dropna()
            m = DianaQuantEngine.get_risk_metrics(returns)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
            k2.metric("Sortino Ratio", f"{m['Sortino']:.2f}")
            k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
            k4.metric("Daily VaR (95%)", f"{m['VaR']:.2f}%")
            
            st.plotly_chart(px.line(prices, title=f"{ticker} Performance", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)

def render_ai_forecast():
    st.header("🔮 Neural Predictive Engine")
    asset = st.text_input("Target Asset:", "BTC-USD").upper()
    if st.button("Train AI"):
        raw = DianaQuantEngine.clean_data(yf.download(asset, period="3y", progress=False).reset_index())
        df_p = pd.DataFrame({'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 'y': pd.to_numeric(raw['Close'])}).dropna()
        m = Prophet(daily_seasonality=True).fit(df_p)
        forecast = m.predict(m.make_future_dataframe(periods=90))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(dash='dash', color='#FFD700')))
        st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Neural Outlook"), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    up = st.file_uploader("Upload Transaction CSV", type="csv")
    df = pd.read_csv(up) if up else pd.DataFrame([
        {"Description": "Salary", "Amount": 10000, "Category": "Income"},
        {"Description": "Rent", "Amount": -3000, "Category": "Fixed"},
        {"Description": "Investments", "Amount": -2500, "Category": "Wealth"},
        {"Description": "Lifestyle", "Amount": -1000, "Category": "Wants"}
    ])
    
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    outflows = df[df['Amount'] < 0].copy()
    outflows['Abs'] = outflows['Amount'].abs()
    
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.plotly_chart(px.pie(outflows, values='Abs', names='Category', hole=0.6, template="plotly_dark", title="Outflow Audit", color_discrete_sequence=px.colors.sequential.YlOrBr))
    with c2:
        wealth_pct = (outflows[outflows['Category'] == 'Wealth']['Abs'].sum() / outflows['Abs'].sum()) * 100 if not outflows.empty else 0
        st.metric("Wealth Building Rate", f"{wealth_pct:.1f}%", delta=f"{wealth_pct-20:.1f}%")
        if wealth_pct < 20: st.error("ALERT: Allocation below institutional target.")
        else: st.success("OPTIMIZED: Wealth creation profile is active.")

# ==========================================
# 4. MAIN NAVIGATOR
# ==========================================

def main():
    render_market_pulse()
    nav = st.sidebar.radio("Sovereign Terminal:", ["Methodology", "Equity Intelligence", "Neural Forecasting", "Wealth Advisor"])
    
    if nav == "Methodology": render_methodology()
    elif nav == "Equity Intelligence": render_equity_intel()
    elif nav == "Neural Forecasting": render_ai_forecast()
    elif nav == "Wealth Advisor": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Engine Build: v12.5 | {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
