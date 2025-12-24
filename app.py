
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
# 1. CORE SYSTEM ARCHITECTURE
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Grand Sovereign",
    page_icon="🏛️",
    layout="wide"
)

# Professional FinTech Aesthetics
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.94), rgba(0,0,0,0.94)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98);
        border-radius: 30px;
        padding: 60px;
        border: 1px solid #333;
        box-shadow: 0 20px 60px rgba(0,0,0,0.9);
    }
    h1 { color: #FFD700 !important; font-weight: 900; letter-spacing: -1px; }
    .stMetric { background: rgba(255,255,255,0.02); padding: 25px; border-radius: 15px; border-top: 3px solid #FFD700; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANTITATIVE ANALYTICS CORE
# ==========================================

class SovereignAnalytics:
    @staticmethod
    def clean_financial_data(df):
        """Advanced MultiIndex flattening for yfinance v0.2.x"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def get_risk_ratios(returns):
        """Institutional Risk-Return Statistics"""
        if returns.empty: return None
        rf = 0.02 / 252
        mu = returns.mean()
        sigma = returns.std()
        
        sharpe = (mu - rf) / sigma * np.sqrt(252) if sigma != 0 else 0
        
        downside = returns[returns < 0]
        sortino = (mu - rf) / downside.std() * np.sqrt(252) if not downside.empty else 0
        
        cum_rets = (1 + returns).cumprod()
        drawdown = (cum_rets / cum_rets.cummax()) - 1
        mdd = drawdown.min() * 100
        
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        return {"Sharpe": sharpe, "Sortino": sortino, "MDD": mdd, "VaR": var_95}

# ==========================================
# 3. INTERFACE MODULES
# ==========================================

def render_market_pulse():
    st.title("🏛️ Diana Grand Sovereign AI")
    st.markdown("##### *Institutional Intelligence & Stochastic Modeling Platform*")
    
    indices = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "Crude Oil": "CL=F"}
    cols = st.columns(len(indices))
    for i, (name, sym) in enumerate(indices.items()):
        try:
            df = SovereignAnalytics.clean_financial_data(yf.download(sym, period="2d", progress=False))
            p, c = df['Close'].iloc[-1], ((df['Close'].iloc[-1]/df['Close'].iloc[-2])-1)*100
            cols[i].metric(name, f"{p:,.2f}", f"{c:+.2f}%")
        except: pass
    st.divider()

def render_methodology():
    st.header("🔬 Academic Research Methodology")
    t1, t2, t3 = st.tabs(["Stochastic Calculation", "Neural Networks", "Risk Parity"])
    
    with t1:
        st.subheader("Geometric Brownian Motion (GBM)")
        st.latex(r"S_t = S_0 \exp\left( (\mu - \frac{1}{2}\sigma^2)t + \sigma W_t \right)")
        st.write("Used to simulate 1,000+ stochastic price trajectories for Value-at-Risk (VaR) estimation.")
        
        
    with t2:
        st.subheader("Additive Time-Series Forecasting")
        st.latex(r"Z_t = Trend(t) + Seasonal(t) + Holiday(t) + \epsilon")
        st.write("Prophet engine extracts cyclicality using Fourier series to predict 90-day forward outlooks.")
        

def render_equity_research():
    st.header("📈 Equity Intelligence Terminal")
    col_a, col_b = st.columns([3, 1])
    
    with col_a:
        ticker = st.text_input("Enter Asset Ticker (e.g., NVDA, AAPL, BTC-USD):", "NVDA").upper()
    with col_b:
        timeframe = st.selectbox("Lookback Period:", ["1Y", "2Y", "3Y", "5Y", "10Y"])
        
    if st.button("Initialize Deep Research"):
        with st.spinner("Crunching Quant Data..."):
            raw = SovereignAnalytics.clean_financial_data(yf.download(ticker, period=timeframe.lower(), progress=False))
            if not raw.empty:
                prices = raw['Close'].squeeze()
                returns = prices.pct_change().dropna()
                metrics = SovereignAnalytics.get_risk_ratios(returns)
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
                k2.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}")
                k3.metric("Max Drawdown", f"{metrics['MDD']:.2f}%")
                k4.metric("Value-at-Risk (95%)", f"{metrics['VaR']:.2f}%")
                
                # Interactive Plot
                st.plotly_chart(px.line(prices, title=f"{ticker} Growth Analysis", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
                
                # Monte Carlo VaR
                st.subheader("🎲 Stochastic Stress Test")
                sim_days = 60
                sim_paths = 100
                last_price = prices.iloc[-1]
                mu, sigma = returns.mean(), returns.std()
                
                sim_results = []
                for _ in range(sim_paths):
                    path = [last_price]
                    for _ in range(sim_days):
                        path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
                    sim_results.append(path)
                
                fig_mc = go.Figure()
                for p in sim_results:
                    fig_mc.add_trace(go.Scatter(y=p, mode='lines', opacity=0.2, line=dict(width=1)))
                st.plotly_chart(fig_mc.update_layout(title="100 Simulated Paths (60-Day)", template="plotly_dark", showlegend=False), use_container_width=True)

def render_ai_forecasting():
    st.header("🔮 Neural Predictive Engine (v4.0)")
    target = st.text_input("Forecasting Target Symbol:", "BTC-USD").upper()
    
    if st.button("Train Neural Model"):
        with st.spinner("Optimizing Fourier Coefficients..."):
            raw = SovereignAnalytics.clean_financial_data(yf.download(target, period="3y", progress=False).reset_index())
            df_p = pd.DataFrame({'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 
                                 'y': pd.to_numeric(raw['Close'], errors='coerce')}).dropna()
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash', color='#FFD700')))
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Forecast for {target}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_management():
    st.header("💳 AI Behavioral Wealth Audit")
    
    # 🚨 FIX: Using st.data_editor with careful column filtering to avoid TypeError
    st.markdown("##### Strategic Capital Allocation Audit")
    
    uploaded = st.file_uploader("Upload Transaction Ledger (CSV)", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.info("Interactive Simulation: Edit the table below to analyze behavior.")
        df = pd.DataFrame([
            {"Description": "Institutional Salary", "Amount": 8500, "Category": "Income"},
            {"Description": "Mortgage Outflow", "Amount": -2500, "Category": "Fixed"},
            {"Description": "ETF / S&P 500 Buy", "Amount": -2000, "Category": "Wealth"},
            {"Description": "Dining & Lifestyle", "Amount": -600, "Category": "Wants"},
            {"Description": "Logistics & Uber", "Amount": -300, "Category": "Fixed"},
            {"Description": "Emergency Fund", "Amount": -500, "Category": "Wealth"}
        ])
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # 🚨 CRITICAL FIX FOR TypeError: Selective absolute value calculation
    if 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        # We only take the absolute value of the NUMERIC 'Amount' column
        outflow_df = df[df['Amount'] < 0].copy()
        outflow_df['AbsAmount'] = outflow_df['Amount'].abs()
        
        total_out = outflow_df['AbsAmount'].sum()
        
        if total_out > 0:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(px.pie(outflow_df, values='AbsAmount', names='Category', hole=0.6, 
                                     template="plotly_dark", title="Audit: Behavioral Distribution",
                                     color_discrete_sequence=px.colors.sequential.YlOrBr))
            with right:
                wealth_sum = outflow_df[outflow_df['Category'] == 'Wealth']['AbsAmount'].sum()
                w_rate = (wealth_sum / total_out) * 100
                st.metric("Wealth Building Factor", f"{w_rate:.1f}%", delta=f"{w_rate-20:.1f}% (Target: 20%)")
                
                if w_rate < 20:
                    st.error("STRATEGIC ALERT: Allocation to Wealth is sub-optimal. Reallocate from 'Wants'.")
                else:
                    st.success("AUDIT PASS: Behavioral pattern aligns with high-net-worth growth.")
            
            st.subheader("Transactional Intelligence Ledger")
            st.dataframe(df, use_container_width=True)

# ==========================================
# 4. MAIN NAVIGATOR
# ==========================================

def main():
    render_market_pulse()
    
    st.sidebar.title("💎 Sovereign Menu")
    nav = st.sidebar.radio("Navigation Perspective:", 
        ["Research Methodology", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    if nav == "Research Methodology": render_methodology()
    elif nav == "Equity Intelligence": render_equity_research()
    elif nav == "Neural Forecasting": render_ai_forecast() # Handled by render_ai_forecasting
    elif nav == "Wealth Management": render_wealth_management()
    
    # Map AI Forecast if radio name differs
    if nav == "AI Forecast" or nav == "Neural Forecasting": render_ai_forecasting()

    st.sidebar.divider()
    st.sidebar.markdown("**Terminal Engine:** `v10.1.5-Sovereign`")
    st.sidebar.caption(f"Last API Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
