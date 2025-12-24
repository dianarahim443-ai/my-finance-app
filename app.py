import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# Institutional Dark UI Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.95);
        border-radius: 20px;
        padding: 50px;
        border: 1px solid #333;
    }
    h1, h2, h3 { color: #FFD700 !important; font-family: 'Inter', sans-serif; font-weight: 800; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; border-left: 5px solid #FFD700; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ANALYTICAL ENGINES ---

@st.cache_data(ttl=3600)
def get_global_pulse():
    tickers = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD"}
    data = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period="2d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            price, prev = float(df['Close'].iloc[-1]), float(df['Close'].iloc[-2])
            data[name] = (price, ((price - prev) / prev) * 100)
        except: data[name] = (0, 0)
    return data

def run_monte_carlo(last_price, mu, sigma, days=30, sims=100):
    simulation_df = pd.DataFrame()
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
        simulation_df[i] = prices
    return simulation_df

# --- 3. MAIN INTERFACE ---

def main():
    st.title("🏛️ Diana Finance: Institutional Research Platform")
    
    # Global Pulse Header
    pulse = get_global_pulse()
    cols = st.columns(len(pulse))
    for i, (name, val) in enumerate(pulse.items()):
        cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    st.sidebar.title("🔬 Research Modules")
    page = st.sidebar.selectbox("Select Module:", 
        ["Documentation & Methodology", "Equity Intelligence", "AI Predictive Engine", "Personal Finance AI"])

    # --- PAGE 1: DOCUMENTATION ---
    if page == "Documentation & Methodology":
        st.header("📑 Quantitative Methodology")
        t1, t2 = st.tabs(["Stochastic Modeling", "Neural Architecture"])
        with t1:
            st.subheader("Geometric Brownian Motion (GBM)")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.write("Used for simulating potential price paths under stochastic market conditions.")
            
        with t2:
            st.subheader("Prophet Decomposable Model")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.write("Our AI decomposes data into Trend, Seasonality, and Holiday effects.")
            

    # --- PAGE 2: EQUITY INTELLIGENCE ---
    elif page == "Equity Intelligence":
        st.header("📈 Backtesting & Risk Intelligence")
        ticker = st.text_input("Institutional Ticker:", "NVDA").upper()
        if st.button("Run Simulation"):
            raw = yf.download(ticker, period="2y", progress=False)
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
            
            prices = raw['Close'].squeeze()
            returns = prices.pct_change().dropna()
            
            # Risk Metrics
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            st.metric("Annualized Sharpe Ratio", f"{sharpe:.2f}")
            
            # Monte Carlo
            st.subheader("Monte Carlo Stress Test")
            sims = run_monte_carlo(prices.iloc[-1], returns.mean(), returns.std())
            st.plotly_chart(px.line(sims, template="plotly_dark", title="100 Simulated Paths (30D)").update_layout(showlegend=False))

    # --- PAGE 3: AI PREDICTION ---
    elif page == "AI Predictive Engine":
        st.header("🔮 Neural Time-Series Forecasting")
        asset = st.text_input("Asset Symbol:", "BTC-USD").upper()
        if st.button("Generate AI Forecast"):
            raw = yf.download(asset, period="3y", progress=False).reset_index()
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
            
            # PREPROCESSING (The Fix)
            df_p = pd.DataFrame({'ds': pd.to_datetime(raw['Date']).dt.tz_localize(None), 
                                 'y': pd.to_numeric(raw['Close'], errors='coerce')}).dropna()

            m = Prophet(daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=60))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash', color='#FFD700')))
            st.plotly_chart(fig.update_layout(template="plotly_dark"), use_container_width=True)
            
            st.subheader("Behavioral Seasonality Breakdown")
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

    # --- PAGE 4: PERSONAL FINANCE (FILE UPLOAD BACK) ---
    elif page == "Personal Finance AI":
        st.header("💳 AI-Driven Capital Allocation")
        st.markdown("Upload your transaction CSV or use the **Institutional Simulation** below.")
        
        uploaded_file = st.file_uploader("Upload Transaction Ledger (CSV)", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Simulation Mode: Visualizing Sample Institutional Data")
            df = pd.DataFrame({
                'Description': ['Salary', 'Rent', 'ETF Investment', 'Amazon', 'Groceries', 'Gold Savings'],
                'Amount': [6000, -2000, -1200, -400, -500, -400],
                'Category': ['Income', 'Fixed', 'Wealth', 'Lifestyle', 'Fixed', 'Wealth']
            })

        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'])
            outflow = df[df['Amount'] < 0].abs()
            total_out = outflow['Amount'].sum()
            
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.plotly_chart(px.pie(outflow, values='Amount', names='Category', hole=0.6, 
                                     template="plotly_dark", title="Audit: Capital Outflow Distribution"))
            with c2:
                wealth_pct = (outflow[outflow['Category'] == 'Wealth']['Amount'].sum() / total_out) * 100
                st.metric("Wealth Building Rate", f"{wealth_pct:.1f}%", delta=f"{wealth_pct-20:.1f}%")
                if wealth_pct < 20: st.error("Audit Fail: Increase Asset Allocation.")
                else: st.success("Audit Pass: Optimized for Wealth Accumulation.")
            
            st.subheader("Transaction Intelligence")
            st.dataframe(df, use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption("Diana AI Framework v6.0 | Total Reintegration")

if __name__ == "__main__":
    main()
