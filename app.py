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
import io

# ==========================================================
# 1. SYSTEM INITIALIZATION & SOVEREIGN DESIGN SYSTEM
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Diana Sovereign AI | Institutional Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a dark, luxurious, glassmorphic terminal
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Inter:wght@400;900&display=swap');
    
    :root {
        --gold: #FFD700;
        --dark-bg: #050505;
        --card-bg: rgba(20, 20, 20, 0.95);
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1a1a, #050505);
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding: 40px 60px;
        max-width: 95%;
    }

    /* Sovereign Header */
    .header-text {
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        font-size: 5rem !important;
        background: linear-gradient(to bottom, #FFD700, #B8860B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -4px;
        margin-bottom: 0;
    }

    /* Metric Styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 215, 0, 0.03);
        border: 1px solid rgba(255, 215, 0, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Institutional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255,255,255,0.02);
        border-radius: 10px;
        color: white;
        padding: 0 30px;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--gold) !important;
        color: black !important;
        font-weight: bold;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 215, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE ENGINE (CORE MATH)
# ==========================================================
class SovereignQuantEngine:
    @staticmethod
    def get_market_data(ticker, period="2y", interval="1d"):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty: return None
            # Handle Multi-Index columns if yfinance returns them
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data
        except Exception: return None

    @staticmethod
    def compute_advanced_metrics(df, col='Close'):
        returns = df[col].pct_change().dropna()
        avg_ret = returns.mean()
        std_dev = returns.std()
        
        metrics = {
            "last_price": df[col].iloc[-1],
            "sharpe": (avg_ret / std_dev) * np.sqrt(252) if std_dev != 0 else 0,
            "var_95": norm.ppf(0.05, avg_ret, std_dev) * 100,
            "cvar_95": returns[returns <= norm.ppf(0.05, avg_ret, std_dev)].mean() * 100,
            "volatility": std_dev * np.sqrt(252) * 100,
            "mdd": ((df[col] / df[col].cummax()) - 1).min() * 100,
            "skew": skew(returns),
            "kurtosis": kurtosis(returns),
            "returns": returns
        }
        return metrics

    @staticmethod
    def run_monte_carlo(last_price, mu, sigma, days=60, simulations=100):
        simulation_df = pd.DataFrame()
        for i in range(simulations):
            prices = [last_price]
            for _ in range(days):
                prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            simulation_df[f"Sim_{i}"] = prices
        return simulation_df

# ==========================================================
# 3. INTERFACE MODULES
# ==========================================================

def render_dashboard_header():
    st.markdown('<h1 class="header-text">DIANA SOVEREIGN</h1>', unsafe_allow_html=True)
    st.markdown("### Institutional Asset Management & Neural Forecast Terminal")
    
    # Global Pulse Ribbon
    indices = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Bitcoin": "BTC-USD", "Gold": "GC=F", "USD/EUR": "EURUSD=X"}
    cols = st.columns(len(indices))
    for i, (name, ticker) in enumerate(indices.items()):
        data = SovereignQuantEngine.get_market_data(ticker, period="2d")
        if data is not None:
            price = data['Close'].iloc[-1]
            change = ((price / data['Close'].iloc[-2]) - 1) * 100
            cols[i].metric(name, f"{price:,.2f}", f"{change:+.2f}%")
    st.divider()

# ----------------------------------------------------------

def module_risk_framework():
    st.header("🔬 Strategic Risk Architecture")
    tab_theory, tab_math = st.tabs(["Stochastic Models", "Factor Sensitivity"])
    
    with tab_theory:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("Geometric Brownian Motion")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.write("""
                The terminal uses GBM to simulate future price paths. 
                - **$\mu$ (Drift):** Expected return.
                - **$\sigma$ (Diffusion):** Asset volatility.
                - **$dW_t$:** Wiener process (random walk).
            """)
            
        with c2:
            st.subheader("Value at Risk (Parametric)")
            st.latex(r"VaR_{\alpha} = S_0 \cdot (\mu \Delta t + \sigma \sqrt{\Delta t} \Phi^{-1}(\alpha))")
            
            st.info("Institutional Grade Risk Management: We analyze the 5% tail risk to protect capital.")

# ----------------------------------------------------------

def module_equity_intelligence():
    st.header("📈 Equity Intelligence & Search")
    
    # SEARCH ANY ASSET
    st.markdown("#### Universal Asset Search")
    with st.container():
        c_search, c_time, c_bench = st.columns([3, 1, 1])
        ticker = c_search.text_input("Enter Ticker (Any Stock, Crypto, FX, Commodity):", "NVDA").upper()
        horizon = c_time.selectbox("History:", ["1y", "2y", "5y", "max"], index=1)
        analyze_btn = st.button("EXECUTE DEEP AUDIT")

    if analyze_btn:
        with st.spinner(f"Auditing {ticker}..."):
            df = SovereignQuantEngine.get_market_data(ticker, period=horizon)
            if df is not None:
                m = SovereignQuantEngine.compute_advanced_metrics(df)
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Market Price", f"${m['last_price']:,.2f}")
                k2.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
                k3.metric("Annual Volatility", f"{m['volatility']:.2f}%")
                k4.metric("Max Drawdown", f"{m['mdd']:.2f}%", delta_color="inverse")
                
                # Visuals
                fig_price = px.area(df, y='Close', title=f"{ticker} Performance Trajectory", template="plotly_dark")
                fig_price.update_traces(line_color="#FFD700", fillcolor="rgba(255, 215, 0, 0.1)")
                st.plotly_chart(fig_price, use_container_width=True)
                
                col_dist, col_monte = st.columns(2)
                with col_dist:
                    st.plotly_chart(px.histogram(m['returns'], nbins=100, title="Daily Return Distribution", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
                with col_monte:
                    mc_data = SovereignQuantEngine.run_monte_carlo(m['last_price'], m['returns'].mean(), m['returns'].std())
                    fig_mc = px.line(mc_data, title="Monte Carlo: 100 Stochastic Simulations (60D)", template="plotly_dark")
                    fig_mc.update_layout(showlegend=False)
                    st.plotly_chart(fig_mc, use_container_width=True)
            else:
                st.error("Asset not found. Please use Yahoo Finance symbols (e.g. BTC-USD, AAPL, GC=F).")

# ----------------------------------------------------------

def module_neural_forecasting():
    st.header("🔮 Neural Prophet Engine")
    c_in, _ = st.columns([2, 2])
    target = c_in.text_input("Predictive Target:", "TSLA").upper()
    
    if st.button("RUN NEURAL TRAINING"):
        df = SovereignQuantEngine.get_market_data(target, period="3y")
        if df is not None:
            df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
            
            m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name="Actual", line=dict(color="#00F2FF")))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Neural Forecast", line=dict(color="#FFD700", dash='dash')))
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Deep Learning Projection: {target}"), use_container_width=True)
            
            # Components
            st.markdown("#### Signal Decomposition")
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ----------------------------------------------------------

def module_wealth_management():
    st.header("💳 AI Wealth Management & Document Audit")
    
    mode = st.radio("Input Method:", ["Smart Document Upload (CSV)", "Manual Sovereign Ledger"], horizontal=True)
    
    wealth_df = pd.DataFrame()
    
    if mode == "Smart Document Upload (CSV)":
        st.info("Upload any financial export (CSV). The AI will learn your column structure.")
        up_file = st.file_uploader("Choose File", type="csv")
        if up_file:
            raw_df = pd.read_csv(up_file)
            st.write("### Document Preview")
            st.dataframe(raw_df.head(5), use_container_width=True)
            
            st.markdown("#### Column Mapping Intelligence")
            cm1, cm2 = st.columns(2)
            cat_col = cm1.selectbox("Select 'Category' column:", raw_df.columns)
            amt_col = cm2.selectbox("Select 'Amount' column:", raw_df.columns)
            
            if st.button("SYNC & PROCESS"):
                wealth_df = raw_df[[cat_col, amt_col]].rename(columns={cat_col: 'Category', amt_col: 'Amount'})
    else:
        manual_data = [
            {"Category": "Salary", "Amount": 15000},
            {"Category": "Rent", "Amount": -4000},
            {"Category": "Equity Investment", "Amount": -3000},
            {"Category": "Lifestyle", "Amount": -2000}
        ]
        wealth_df = st.data_editor(pd.DataFrame(manual_data), num_rows="dynamic", use_container_width=True)

    if not wealth_df.empty:
        try:
            wealth_df['Amount'] = pd.to_numeric(wealth_df['Amount'], errors='coerce').fillna(0)
            income = wealth_df[wealth_df['Amount'] > 0]['Amount'].sum()
            expenses = wealth_df[wealth_df['Amount'] < 0].copy()
            expenses['Abs'] = expenses['Amount'].abs()
            
            if income > 0:
                inv_keywords = 'Wealth|Invest|Stock|Gold|Save|پس‌انداز|بورس'
                invest_total = expenses[expenses['Category'].str.contains(inv_keywords, case=False, na=False)]['Abs'].sum()
                w_rate = (invest_total / income) * 100
                
                # Reporting
                r1, r2, r3 = st.columns(3)
                r1.metric("Gross Cash Flow", f"${income:,.0f}")
                r2.metric("Wealth Creation Rate", f"{w_rate:.1f}%")
                r3.metric("Monthly Surplus", f"${income - expenses['Abs'].sum():,.0f}")
                
                st.divider()
                graph_col, advise_col = st.columns([1.5, 1])
                with graph_col:
                    
                    st.plotly_chart(px.pie(expenses, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
                with advise_col:
                    st.subheader("🕵️ Sovereign Advisory")
                    if w_rate < 20:
                        st.error("Inadequate Capital Accumulation. Target: 20%.")
                    else:
                        st.success("Optimal Strategic Allocation. Capital Velocity is healthy.")
                    
                    st.info(f"Recommended Next Action: Reinvest ${income*0.1:,.0f} into low-volatility ETFs.")
        except Exception as e:
            st.error(f"Processing Fault: {e}")

# ==========================================================
# 4. MAIN NAVIGATION ROUTER
# ==========================================================
def main():
    render_dashboard_header()
    
    # Sidebar Institutional Branding
    st.sidebar.markdown("# 🏛️ DIANA v30")
    st.sidebar.markdown("`Sovereign Access Level: Tier 1`")
    st.sidebar.divider()
    
    navigation = st.sidebar.radio(
        "TERMINAL MODULES", 
        ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management"]
    )
    
    if navigation == "Risk Framework":
        module_risk_framework()
    elif navigation == "Equity Intelligence":
        module_equity_intelligence()
    elif navigation == "Neural Forecasting":
        module_neural_forecasting()
    elif navigation == "Wealth Management":
        module_wealth_management()

    st.sidebar.divider()
    # Live Clock & Status
    st.sidebar.write(f"**Terminal Status:** Operational")
    st.sidebar.write(f"**UTC Sync:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.caption("Institutional proprietary software. For academic defense only.")

if __name__ == "__main__":
    main()
