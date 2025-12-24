import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime
from scipy.stats import norm, skew, kurtosis
import warnings
import io

# ==========================================================
# 1. GLOBAL CONFIGURATION & SOVEREIGN STYLING
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Terminal", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92)), 
                    url('https://images.unsplash.com/photo-1639762681485-074b7f938ba0?q=80&w=2070');
        background-size: cover;
    }
    
    .main .block-container {
        background: rgba(10, 10, 10, 0.98); border-radius: 40px; 
        padding: 50px 80px; border: 1px solid #2a2a2a;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }
    
    h1 { 
        color: #FFD700 !important; 
        font-weight: 900; 
        font-size: 5rem !important; 
        letter-spacing: -2px;
        margin-bottom: 10px;
    }
    
    h2, h3 { 
        color: #FFD700 !important; 
        border-left: 8px solid #FFD700; 
        padding-left: 15px; 
        font-family: 'Inter', sans-serif;
    }
    
    .stMetric { 
        background: rgba(255,255,255,0.03); 
        padding: 25px; 
        border-radius: 20px; 
        border-top: 5px solid #FFD700;
        transition: 0.3s;
    }
    
    .stMetric:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-5px);
    }

    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; 
        font-weight: 800; 
        border-radius: 12px; 
        height: 4em;
        width: 100%;
        border: none;
        transition: 0.4s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(255,215,0,0.4);
        transform: scale(1.02);
    }

    .upload-section {
        border: 2px dashed #FFD700;
        padding: 30px;
        border-radius: 20px;
        background: rgba(255, 215, 0, 0.02);
        margin-bottom: 20px;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #FFD700; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. QUANTITATIVE ANALYTICS ENGINE
# ==========================================================
class QuantEngine:
    @staticmethod
    def get_risk_metrics(df, col='Close'):
        try:
            # Handle MultiIndex if necessary
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            rets = df[col].pct_change().dropna()
            mu, sigma = rets.mean(), rets.std()
            
            metrics = {
                "Last": df[col].iloc[-1],
                "Sharpe": (mu / sigma) * np.sqrt(252) if sigma != 0 else 0,
                "VaR": norm.ppf(0.05, mu, sigma) * 100,
                "MDD": ((df[col] / df[col].cummax()) - 1).min() * 100,
                "Volatility": sigma * np.sqrt(252) * 100,
                "Skew": skew(rets),
                "Kurtosis": kurtosis(rets),
                "Returns": rets
            }
            return metrics
        except Exception as e:
            st.error(f"Engine Error: {str(e)}")
            return None

    @staticmethod
    def monte_carlo_sim(last_price, mu, sigma, days=60, sims=100):
        fig = go.Figure()
        for _ in range(sims):
            prices = [last_price]
            for _ in range(days):
                prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            fig.add_trace(go.Scatter(y=prices, mode='lines', opacity=0.1, line=dict(color='#FFD700')))
        return fig

# ==========================================================
# 3. INTERFACE MODULES
# ==========================================================

def render_risk_framework():
    st.markdown('<h1 style="font-size:3rem !important;">🔬 Institutional Risk Framework</h1>', unsafe_allow_html=True)
    col1, col2 = st.tabs(["📐 Stochastic Calculus", "📉 Tail Risk Probability"])
    
    with col1:
        st.subheader("Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.info("The terminal utilizes GBM for continuous-time asset price modeling.")
        st.write("This stochastic differential equation is the foundation for our Monte Carlo simulations, where we assume a constant drift and volatility.")
        
        
    with col2:
        st.subheader("Value at Risk (Parametric)")
        st.latex(r"VaR_{\alpha} = S_0 \cdot [\mu \Delta t + \sigma \sqrt{\Delta t} \Phi^{-1}(\alpha)]")
        st.write("We compute the Parametric VaR at 95% confidence to identify the threshold of maximum expected loss under normal market conditions.")
        

# ----------------------------------------------------------

def render_equity_intelligence():
    st.markdown('<h1 style="font-size:3rem !important;">📈 Equity Intelligence</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Search Parameters")
        mode = st.radio("Data Source:", ["Live Market Ticker", "Custom CSV Analysis"])
        st.divider()

    if mode == "Live Market Ticker":
        ticker = st.text_input("Enter Ticker Symbol (e.g., NVDA, BTC-USD, GC=F):", "AAPL").upper()
        lookback = st.selectbox("Period:", ["1y", "2y", "5y", "max"], index=1)
        
        if st.button("RUN INSTITUTIONAL AUDIT"):
            with st.spinner("Fetching data from Global Servers..."):
                df = yf.download(ticker, period=lookback)
                if not df.empty:
                    display_equity_results(df, ticker)
                else:
                    st.error("Ticker not found. Please verify the symbol.")
    else:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        up = st.file_uploader("Upload Market Historical Data:", type="csv")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if up:
            df_up = pd.read_csv(up)
            st.dataframe(df_up.head(5), use_container_width=True)
            p_col = st.selectbox("Assign Price Column (e.g., Close):", df_up.columns)
            if st.button("EXECUTE AUDIT"):
                display_equity_results(df_up, "User Upload", p_col)

def display_equity_results(df, name, col='Close'):
    m = QuantEngine.get_risk_metrics(df, col)
    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${m['Last']:,.2f}")
        c2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
        c3.metric("Tail Risk (VaR 95%)", f"{m['VaR']:.2f}%")
        c4.metric("Max Drawdown", f"{m['MDD']:.2f}%", delta_color="inverse")
        
        st.plotly_chart(px.area(df, y=col, title=f"Audit Trajectory: {name}", template="plotly_dark").update_traces(line_color="#FFD700", fillcolor="rgba(255,215,0,0.1)"), use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(px.histogram(m['Returns'], nbins=100, title="Return Distribution Density", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
        with col_b:
            mc = QuantEngine.monte_carlo_sim(m['Last'], m['Returns'].mean(), m['Returns'].std())
            st.plotly_chart(mc.update_layout(template="plotly_dark", title="Monte Carlo: 100 Stochastic Simulations (60D)", showlegend=False), use_container_width=True)
        
        with st.expander("🔬 Advanced Statistical Breakdown"):
            s1, s2, s3 = st.columns(3)
            s1.write(f"**Annualized Volatility:** {m['Volatility']:.2f}%")
            s2.write(f"**Skewness:** {m['Skew']:.4f}")
            s3.write(f"**Kurtosis:** {m['Kurtosis']:.4f}")

# ----------------------------------------------------------

def render_wealth_advisor():
    st.markdown('<h1 style="font-size:3rem !important;">💳 Wealth Management Advisor</h1>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["📝 Manual Sovereign Ledger", "📥 Smart Document Sync"])
    
    working_df = pd.DataFrame()

    with t1:
        st.write("Manual entry for private assets and lifestyle costs:")
        raw_data = [
            {"Category": "Salary/Inflow", "Amount": 15000}, 
            {"Category": "Real Estate Costs", "Amount": -4000}, 
            {"Category": "Stock Portfolio", "Amount": -3000}, 
            {"Category": "Lifestyle/General", "Amount": -1000}
        ]
        working_df = st.data_editor(pd.DataFrame(raw_data), num_rows="dynamic", use_container_width=True, key="manual_editor")

    with t2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        up_w = st.file_uploader("Upload Transaction Statement (CSV):", type="csv", key="wealth_upload")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if up_w:
            try:
                # --------------------------------------------------
                # FIXED UPLOAD LOGIC (MAPPING INTELLIGENCE)
                # --------------------------------------------------
                file_df = pd.read_csv(up_w)
                st.write("### 🔍 Document Preview")
                st.dataframe(file_df.head(5), use_container_width=True)
                
                st.info("Map your CSV columns to the Sovereign Engine:")
                map_c1, map_c2 = st.columns(2)
                cat_col_map = map_c1.selectbox("Transaction Category Column:", file_df.columns)
                amt_col_map = map_c2.selectbox("Transaction Amount Column:", file_df.columns)
                
                if st.button("PROCESS & SYNC DOCUMENT"):
                    # Create the standardized dataframe from the user's mapping
                    working_df = file_df[[cat_col_map, amt_col_map]].copy()
                    working_df.columns = ['Category', 'Amount']
                    # Store in session state to persist after button click
                    st.session_state['processed_wealth'] = working_df
                    st.success("Document synchronized with Global Advisor.")
            except Exception as e:
                st.error(f"Mapping Error: {str(e)}")

    # Use session state or manual data
    if 'processed_wealth' in st.session_state and not t1:
        data_to_analyze = st.session_state['processed_wealth']
    else:
        data_to_analyze = working_df

    if not data_to_analyze.empty:
        try:
            # Data Cleaning
            data_to_analyze['Amount'] = pd.to_numeric(data_to_analyze['Amount'], errors='coerce').fillna(0)
            inc = data_to_analyze[data_to_analyze['Amount'] > 0]['Amount'].sum()
            out = data_to_analyze[data_to_analyze['Amount'] < 0].copy()
            out['Abs'] = out['Amount'].abs()
            
            if inc > 0:
                # Institutional Wealth Logic
                inv_keywords = 'Invest|Wealth|Stock|Gold|پس‌انداز|Portfolio|Saving'
                inv = out[out['Category'].str.contains(inv_keywords, case=False, na=False)]['Abs'].sum()
                rate = (inv / inc) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Gross Capital Inflow", f"${inc:,.0f}")
                m2.metric("Wealth Creation Rate", f"{rate:.1f}%")
                m3.metric("Net Surplus", f"${inc - out['Abs'].sum():,.0f}")
                
                st.divider()
                ca, cb = st.columns([1.5, 1])
                with ca:
                    st.plotly_chart(px.pie(out, values='Abs', names='Category', hole=0.6, template="plotly_dark", title="Capital Allocation Structure", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
                with cb:
                    st.subheader("🕵️ AI Strategic Verdict")
                    if rate < 20: 
                        st.error("ALERT: Wealth Rate is below institutional threshold (20%). Your capital accumulation is suboptimal.")
                    elif 20 <= rate < 40:
                        st.warning("STABLE: Optimal allocation. Recommend increasing high-yield asset exposure.")
                    else:
                        st.success("SOVEREIGN: Exceptional wealth velocity. Capital is effectively weaponized.")
                    
                    
        except Exception as e: 
            st.error(f"Analysis Error: {e}")

# ----------------------------------------------------------

def render_neural_forecast():
    st.markdown('<h1 style="font-size:3rem !important;">🔮 Neural Prediction Engine</h1>', unsafe_allow_html=True)
    target = st.text_input("Forecast Asset (e.g., TSLA, BTC-USD):", "NVDA").upper()
    
    if st.button("INITIATE NEURAL TRAINING"):
        with st.spinner("Executing Prophet V3 Neural Training..."):
            df = yf.download(target, period="3y").reset_index()
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                p_df = pd.DataFrame({'ds': df['Date'].dt.tz_localize(None), 'y': df['Close']}).dropna()
                
                m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05).fit(p_df)
                future = m.predict(m.make_future_dataframe(periods=90))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name="Actual Price", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], name="Neural Path", line=dict(color='#FFD700', dash='dash')))
                
                st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Deep Learning Projection: {target}"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(m, future), use_container_width=True)
            else:
                st.error("Asset data not found for forecasting.")

# ==========================================================
# 4. MAIN TERMINAL CONTROLLER
# ==========================================================
def main():
    st.sidebar.markdown('<h1 style="font-size:2rem !important; color:#FFD700;">💎 DIANA SOVEREIGN</h1>', unsafe_allow_html=True)
    st.sidebar.markdown("`Institutional Access: Verified`")
    st.sidebar.divider()
    
    nav = st.sidebar.radio("Command Center:", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    # Global Pulse Ribbon
    indices = {"S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "Gold": "GC=F"}
    st.sidebar.divider()
    st.sidebar.write("**Global Market Pulse:**")
    for name, ticker in indices.items():
        try:
            d = yf.download(ticker, period="1d", progress=False)
            price = d['Close'].iloc[-1]
            st.sidebar.caption(f"{name}: ${price:,.2f}")
        except: pass

    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_forecast()
    elif nav == "Wealth Management": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Status: Operational | {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
