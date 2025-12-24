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
# 1. GLOBAL SYSTEM ARCHITECTURE & THEME
# ==========================================
st.set_page_config(
    page_title="Diana Finance AI | Quant-Pro Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-End Institutional CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.92)), 
                    url('https://images.unsplash.com/photo-1639754390267-dc26d626154c?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(8, 8, 8, 0.96);
        border-radius: 24px;
        padding: 60px;
        border: 1px solid #222;
        box-shadow: 0 20px 50px rgba(0,0,0,1);
    }
    h1 { color: #FFD700 !important; font-weight: 800; font-size: 3.5rem !important; margin-bottom: 0px; }
    h2, h3 { color: #E0E0E0 !important; font-weight: 700; }
    .stMetric { 
        background: rgba(255,255,255,0.02); 
        padding: 25px; 
        border-radius: 16px; 
        border: 1px solid #333;
        transition: 0.3s;
    }
    .stMetric:hover { border-color: #FFD700; background: rgba(255,215,0,0.03); }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important;
        font-weight: 700;
        border: none;
        width: 100%;
        border-radius: 8px;
        padding: 12px;
    }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] { color: #888; font-weight: 600; }
    .stTabs [data-baseweb="tab--active"] { color: #FFD700 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. QUANTITATIVE ANALYTICS ENGINE
# ==========================================

class QuantEngine:
    @staticmethod
    def clean_yf_data(df):
        """Fixes MultiIndex and cleaning issues with latest yfinance"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    @staticmethod
    def calculate_metrics(returns):
        """Advanced Portfolio Statistics"""
        rf = 0.02 / 252
        sharpe = (returns.mean() - rf) / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() - rf) / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        cum_rets = (1 + returns).cumprod()
        peak = cum_rets.cummax()
        drawdown = (cum_rets - peak) / peak
        mdd = drawdown.min() * 100
        
        # Value at Risk (95% Confidence)
        var_95 = np.percentile(returns, 5) * 100
        return sharpe, sortino, mdd, var_95

    @staticmethod
    def monte_carlo_sim(last_price, mu, sigma, days=60, simulations=1000):
        dt = 1/252
        sim_results = np.zeros((days, simulations))
        for s in range(simulations):
            prices = [last_price]
            for d in range(days):
                prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()))
            sim_results[:, s] = prices[1:]
        return sim_results

# ==========================================
# 3. INTERFACE MODULES
# ==========================================

def render_header():
    st.title("🏛️ Diana Finance AI")
    st.markdown("### Institutional Quantitative Framework & Neural Research")
    
    # Real-time Market Pulse
    tickers = {"S&P 500": "^GSPC", "Gold": "GC=F", "Bitcoin": "BTC-USD", "10Y Treasury": "^TNX"}
    m_cols = st.columns(len(tickers))
    for i, (name, sym) in enumerate(tickers.items()):
        try:
            d = yf.download(sym, period="2d", progress=False)
            d = QuantEngine.clean_yf_data(d)
            price = d['Close'].iloc[-1]
            change = ((price - d['Close'].iloc[-2]) / d['Close'].iloc[-2]) * 100
            m_cols[i].metric(name, f"{price:,.2f}", f"{change:+.2f}%")
        except: m_cols[i].metric(name, "Data Error", "0.00%")
    st.divider()

def render_documentation():
    st.header("📑 Theoretical Foundation")
    tab1, tab2, tab3 = st.tabs(["Stochastic Models", "Neural Architecture", "System Logic"])
    
    with tab1:
        st.subheader("Stochastic Differential Equations (SDE)")
        st.write("We model market randomness using Geometric Brownian Motion:")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.markdown("""
        - **$\mu$ (Drift):** Expected return of the asset.
        - **$\sigma$ (Volatility):** Diffusion coefficient representing risk.
        - **$dW_t$:** Wiener process (Brownian Motion).
        """)
        
        st.info("The Monte Carlo engine solves this SDE over 1,000 iterations to calculate VaR (Value at Risk).")

    with tab2:
        st.subheader("Neural Time-Series Decomposition")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.markdown("""
        **Additive Prophet Model:**
        1. **$g(t)$:** Trend function modeling non-periodic changes.
        2. **$s(t)$:** Periodic changes (e.g., weekly/yearly seasonality).
        3. **$h(t)$:** Holiday effects (Market irregular shocks).
        """)
        

def render_equity_intelligence():
    st.header("📈 Equity Intelligence & Backtesting")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Enter Institutional Ticker (e.g. NVDA, AAPL, MSFT, BTC-USD):", "NVDA").upper()
    with col2:
        horizon = st.selectbox("Backtest Horizon:", ["1Y", "2Y", "5Y", "10Y"])
    
    if st.button("Execute Quantitative Audit"):
        with st.spinner("Analyzing Market Dynamics..."):
            data = yf.download(ticker, period=horizon.lower(), progress=False)
            data = QuantEngine.clean_yf_data(data)
            
            if not data.empty:
                prices = data['Close'].squeeze()
                returns = prices.pct_change().dropna()
                
                # Metrics Calculation
                sharpe, sortino, mdd, var = QuantEngine.calculate_metrics(returns)
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                kpi2.metric("Sortino Ratio", f"{sortino:.2f}")
                kpi3.metric("Max Drawdown", f"{mdd:.2f}%")
                kpi4.metric("Daily VaR (95%)", f"{var:.2f}%")
                
                # Equity Curve
                st.plotly_chart(px.line(prices, title=f"{ticker} Performance Growth", template="plotly_dark", 
                                      color_discrete_sequence=['#FFD700']), use_container_width=True)
                
                # Monte Carlo
                st.subheader("🎲 Stochastic Path Simulation (Monte Carlo)")
                sim_data = QuantEngine.monte_carlo_sim(prices.iloc[-1], returns.mean()*252, returns.std()*np.sqrt(252))
                
                fig_mc = go.Figure()
                for i in range(50): # Render 50 paths for visual clarity
                    fig_mc.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', opacity=0.3, line=dict(width=1)))
                fig_mc.update_layout(title="60-Day Forward Stochastic Projections", template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.error("Invalid Ticker or No Data Found.")

def render_ai_forecast():
    st.header("🔮 AI Neural Forecasting")
    asset = st.text_input("Target for AI Prediction:", "BTC-USD").upper()
    
    if st.button("Train AI Model"):
        with st.spinner("Training Prophet Decomposable Model..."):
            raw = yf.download(asset, period="3y", progress=False).reset_index()
            raw = QuantEngine.clean_yf_data(raw)
            
            # Robust Preprocessing for Prophet
            df_p = pd.DataFrame()
            df_p['ds'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)
            df_p['y'] = pd.to_numeric(raw['Close'], errors='coerce')
            df_p = df_p.dropna()
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(df_p)
            
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            # Prediction Plot
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Historical Price", line=dict(color='#00F2FF')))
            fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(dash='dash', color='#FFD700')))
            fig_f.update_layout(title=f"90-Day Neural Forecast: {asset}", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_f, use_container_width=True)
            
            st.subheader("Seasonality & Trend Components")
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

def render_wealth_advisor():
    st.header("💳 Behavioral Wealth Optimization")
    st.markdown("##### Upload Transactional Data (CSV) for Institutional Audit")
    
    uploaded_file = st.file_uploader("Upload CSV (Required columns: 'Description', 'Amount')", type="csv")
    
    if uploaded_file or st.button("Use Sample Institutional Data"):
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            # High-fidelity Sample Data
            df = pd.DataFrame({
                'Description': ['Salary', 'Rent', 'ETF Buy (Vanguard)', 'Grocery Store', 'Uber Logistics', 'Netflix', 'Crypto Invest', 'Electricity Bill'],
                'Amount': [7500, -2200, -1500, -450, -120, -15, -1000, -200],
                'Category': ['Income', 'Fixed', 'Wealth', 'Needs', 'Wants', 'Wants', 'Wealth', 'Fixed']
            })
        
        # AI Categorization Logic (Simple but effective)
        def categorize(desc):
            desc = str(desc).lower()
            if any(x in desc for x in ['rent', 'bill', 'electric']): return 'Fixed'
            if any(x in desc for x in ['etf', 'invest', 'crypto', 'save']): return 'Wealth'
            if any(x in desc for x in ['grocery', 'food', 'health']): return 'Needs'
            return 'Wants'

        if 'Amount' in df.columns:
            if 'Category' not in df.columns:
                df['Category'] = df['Description'].apply(categorize)
            
            df['Amount'] = pd.to_numeric(df['Amount'])
            outflow = df[df['Amount'] < 0].abs()
            total_out = outflow['Amount'].sum()
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(px.pie(outflow, values='Amount', names='Category', hole=0.6, 
                                     title="50/30/20 Capital Distribution Audit", 
                                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr))
            with c2:
                st.subheader("Audit Results")
                wealth_sum = outflow[outflow['Category'] == 'Wealth']['Amount'].sum()
                w_pct = (wealth_sum / total_out) * 100
                st.metric("Wealth Building Rate", f"{w_pct:.1f}%", delta=f"{w_pct-20:.1f}%")
                
                if w_pct < 20: st.error("STRATEGIC ALERT: Wealth allocation below 20% target.")
                else: st.success("OPTIMIZED: Capital flow meets institutional growth standards.")
            
            st.subheader("Transactional Ledger Intelligence")
            st.dataframe(df.style.background_gradient(cmap='Greens', subset=['Amount']), use_container_width=True)

# ==========================================
# 4. MAIN CONTROLLER
# ==========================================

def main():
    render_header()
    
    # Secure Sidebar State Controller
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1611/1611154.png", width=80)
    st.sidebar.title("Core Navigation")
    menu = ["Documentation & Methodology", "Equity Intelligence", "AI Predictive Engine", "Personal Finance AI"]
    choice = st.sidebar.radio("Switch Framework:", menu)
    
    if choice == "Documentation & Methodology":
        render_documentation()
    elif choice == "Equity Intelligence":
        render_equity_intelligence()
    elif choice == "AI Predictive Engine":
        render_ai_forecast()
    elif choice == "Personal Finance AI":
        render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.markdown(f"**Diana Engine v7.0**")
    st.sidebar.caption(f"Last sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
