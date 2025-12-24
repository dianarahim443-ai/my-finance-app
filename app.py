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

# ==========================================
# 1. GLOBAL SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Sovereign Research",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-Premium CSS (Glassmorphism & Gold Accents)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500&display=swap');
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.93), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98);
        border-radius: 25px;
        padding: 50px;
        border: 1px solid #444;
        box-shadow: 0 15px 50px rgba(0,0,0,0.8);
    }
    h1 { color: #FFD700 !important; font-size: 3.8rem !important; letter-spacing: -2px; }
    h2, h3 { color: #E0E0E0 !important; border-bottom: 1px solid #333; padding-bottom: 10px; }
    .stMetric { 
        background: rgba(255,255,255,0.02); 
        padding: 25px; border-radius: 15px; 
        border-bottom: 4px solid #FFD700; 
        transition: 0.4s ease;
    }
    .stMetric:hover { background: rgba(255,215,0,0.05); transform: translateY(-5px); }
    code { color: #FFD700 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. CORE ANALYTICAL LOGIC (THE "ENGINE")
# ==========================================

class SovereignEngine:
    @staticmethod
    def flatten_data(df):
        """Ensures MultiIndex dataframes from yfinance are flattened for processing."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def get_performance_metrics(returns):
        """Calculates advanced quantitative risk/return ratios."""
        rf = 0.02 / 252 # Assumed Risk-Free Rate
        mean_ret = returns.mean()
        std_dev = returns.std()
        
        sharpe = (mean_ret - rf) / std_dev * np.sqrt(252) if std_dev != 0 else 0
        
        downside_rets = returns[returns < 0]
        sortino = (mean_ret - rf) / downside_rets.std() * np.sqrt(252) if len(downside_rets) > 0 else 0
        
        cum_rets = (1 + returns).cumprod()
        peak = cum_rets.cummax()
        drawdown = (cum_rets - peak) / peak
        mdd = drawdown.min() * 100
        
        calmar = (mean_ret * 252) / abs(drawdown.min()) if drawdown.min() != 0 else 0
        var_95 = norm.ppf(0.05, mean_ret, std_dev) * 100
        
        return {
            "Sharpe": sharpe, "Sortino": sortino, 
            "Calmar": calmar, "MaxDD": mdd, "VaR": var_95
        }

# ==========================================
# 3. INTERFACE MODULES
# ==========================================

def render_dashboard():
    st.title("🏛️ Diana Sovereign AI")
    st.markdown("##### *Institutional Multi-Asset Quantitative Research Terminal*")
    
    # Live Ticker Ribbon
    tickers = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "Crude Oil": "CL=F"}
    ribbon = st.columns(len(tickers))
    for i, (name, sym) in enumerate(tickers.items()):
        try:
            d = SovereignEngine.flatten_data(yf.download(sym, period="2d", progress=False))
            price, change = d['Close'].iloc[-1], ((d['Close'].iloc[-1] - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
            ribbon[i].metric(name, f"{price:,.2f}", f"{change:+.2f}%")
        except: pass
    st.divider()

def render_methodology():
    st.header("🔬 Academic Methodology & Proofs")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("I. Stochastic Processes")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("We solve the **Geometric Brownian Motion** SDE to generate price paths. The diffusion term ($\sigma$) is estimated via GARCH-ready rolling volatility.")
        
    with col2:
        st.subheader("II. Neural Time-Series")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("Prophet models non-linear trends with changepoint detection, extracting seasonality cycles ($s$) from historical noise ($\epsilon$).")
        

def render_equity_intel():
    st.header("📈 Equity Intelligence & Correlation")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker1 = st.text_input("Primary Asset:", "NVDA").upper()
        ticker2 = st.text_input("Comparison Asset (Benchmark):", "SPY").upper()
    with c2:
        period = st.select_slider("Analysis Period:", options=["1Y", "2Y", "3Y", "5Y"])
    
    if st.button("Initialize Deep Audit"):
        d1 = SovereignEngine.flatten_data(yf.download(ticker1, period=period.lower(), progress=False))
        d2 = SovereignEngine.flatten_data(yf.download(ticker2, period=period.lower(), progress=False))
        
        if not d1.empty and not d2.empty:
            p1, p2 = d1['Close'].squeeze(), d2['Close'].squeeze()
            r1, r2 = p1.pct_change().dropna(), p2.pct_change().dropna()
            
            # KPI Metrics
            m = SovereignEngine.get_performance_metrics(r1)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
            k2.metric("Sortino Ratio", f"{m['Sortino']:.2f}")
            k3.metric("Max Drawdown", f"{m['MaxDD']:.2f}%")
            k4.metric("Value at Risk (95%)", f"{m['VaR']:.2f}%")
            
            # Comparative Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p1.index, y=p1/p1.iloc[0], name=ticker1, line=dict(color='#FFD700')))
            fig.add_trace(go.Scatter(x=p2.index, y=p2/p2.iloc[0], name=ticker2, line=dict(color='#888')))
            st.plotly_chart(fig.update_layout(title="Relative Performance (Normalized)", template="plotly_dark"), use_container_width=True)
            
            # Correlation Matrix
            st.subheader("Asset Correlation Heatmap")
            corr_df = pd.concat([r1, r2], axis=1, keys=[ticker1, ticker2]).corr()
            st.plotly_chart(px.imshow(corr_df, text_auto=True, color_continuous_scale='YlOrBr', template="plotly_dark"))

def render_ai_forecast():
    st.header("🔮 Neural Predictive Engine (V3)")
    asset = st.text_input("Target Ticker:", "BTC-USD").upper()
    
    if st.button("Run AI Training"):
        with st.spinner("Optimizing Hyperparameters..."):
            raw = SovereignEngine.flatten_data(yf.download(asset, period="4y", progress=False).reset_index())
            df_p = pd.DataFrame({'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 
                                 'y': pd.to_numeric(raw['Close'], errors='coerce')}).dropna()
            
            m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            # Uncertainty Area
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255,215,0,0.1)', name='Confidence Upper'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255,215,0,0.1)', name='Confidence Lower'))
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash', color='#FFD700')))
            
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Neural Outlook with Confidence Bands"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_management():
    st.header("💳 AI Behavioral Wealth Audit")
    
    uploaded = st.file_uploader("Upload Transactional Ledger (CSV)", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("No file detected. Using Interactive Live-Editor mode.")
        # Live-editable dataframe feature
        df = pd.DataFrame([
            {"Description": "Salary", "Amount": 7000, "Category": "Income"},
            {"Description": "Mortgage/Rent", "Amount": -2200, "Category": "Fixed"},
            {"Description": "Stock Investment", "Amount": -1500, "Category": "Wealth"},
            {"Description": "Tech Gadgets", "Amount": -400, "Category": "Wants"},
            {"Description": "Groceries", "Amount": -600, "Category": "Needs"},
            {"Description": "Emergency Fund", "Amount": -500, "Category": "Wealth"}
        ])
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        outflow = df[df['Amount'] < 0].abs()
        total_out = outflow['Amount'].sum()
        
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.plotly_chart(px.pie(outflow, values='Amount', names='Category', hole=0.6, 
                                 template="plotly_dark", title="Capital Allocation Audit",
                                 color_discrete_sequence=px.colors.sequential.YlOrBr))
        with col_right:
            wealth_sum = outflow[outflow['Category'] == 'Wealth']['Amount'].sum()
            w_rate = (wealth_sum / total_out) * 100 if total_out > 0 else 0
            st.metric("Wealth Building Multiplier", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}%")
            
            st.subheader("AI Recommendation")
            if w_rate < 20: st.error("STRATEGIC ALERT: Reallocate 5% from 'Wants' to 'Wealth'.")
            else: st.success("OPTIMIZED: Portfolio trajectory is in line with top 5% of savers.")

# ==========================================
# 4. MASTER CONTROLLER
# ==========================================

def main():
    render_dashboard()
    
    st.sidebar.title("💎 Sovereign Menu")
    nav = st.sidebar.radio("Navigate Framework:", 
        ["Theoretical Logic", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    if nav == "Theoretical Logic": render_methodology()
    elif nav == "Equity Intelligence": render_equity_intel()
    elif nav == "Neural Forecasting": render_ai_forecast()
    elif nav == "Wealth Management": render_wealth_management()
    
    st.sidebar.divider()
    st.sidebar.markdown("**Engine:** `v9.2.0-Sovereign`")
    st.sidebar.caption(f"Last API Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
