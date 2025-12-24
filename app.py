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
# 1. GLOBAL TERMINAL STYLING
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    .stApp { background-color: #050505; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    .main .block-container { background: rgba(10, 10, 10, 0.98); border-radius: 35px; padding: 50px; border: 1px solid #2a2a2a; }
    h1 { color: #FFD700 !important; font-weight: 900; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-top: 4px solid #FFD700; }
    .stButton>button { background: linear-gradient(45deg, #FFD700, #B8860B); color: black !important; font-weight: 800; border-radius: 10px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. CORE ANALYTICS ENGINE
# ==========================================================
class AnalyticsEngine:
    @staticmethod
    def calculate_risk(df, col):
        returns = df[col].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma != 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        mdd = ((df[col] / df[col].cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "VaR": var_95, "MDD": mdd, "Returns": returns}

    @staticmethod
    def monte_carlo(last_price, mu, sigma, days=60):
        mc_fig = go.Figure()
        for _ in range(50):
            path = [last_price]
            for _ in range(days):
                path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            mc_fig.add_trace(go.Scatter(y=path, mode='lines', opacity=0.1, line=dict(color='#FFD700')))
        return mc_fig

# ==========================================================
# 3. INTERFACE MODULES
# ==========================================================

def render_risk_framework():
    st.header("🔬 Academic Risk Framework")
    st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
    
    st.write("Modeling asset behavior using Stochastic Calculus and Parametric VaR.")
    st.latex(r"VaR_{95\%} = \text{Position} \cdot (\mu - 1.65\sigma)")

# ----------------------------------------------------------

def render_equity_intelligence():
    st.header("📈 Universal Equity Intelligence")
    
    # 1. SEARCH ANY TICKER (Global Search)
    st.subheader("🔍 Global Asset Search")
    col_search, col_period = st.columns([3, 1])
    ticker_input = col_search.text_input("Enter any Global Ticker (e.g. AAPL, TSLA, BTC-USD, GC=F):", "AAPL").upper()
    period_input = col_period.selectbox("Time Horizon:", ["1y", "2y", "5y", "max"])
    
    if st.button("Run Institutional Audit"):
        with st.spinner(f"Fetching data for {ticker_input}..."):
            data = yf.download(ticker_input, period=period_input)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                process_and_display(data, ticker_input, 'Close')
            else:
                st.error("Ticker not found. Please check the symbol (e.g., use 'BTC-USD' for Bitcoin).")

def process_and_display(df, label, price_col):
    m = AnalyticsEngine.calculate_risk(df, price_col)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"{df[price_col].iloc[-1]:,.2f}")
    c2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
    c4.metric("Tail Risk (VaR)", f"{m['VaR']:.2f}%")
    
    st.plotly_chart(px.line(df, y=price_col, title=f"Price Trajectory: {label}", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        
        st.plotly_chart(px.histogram(m['Returns'], nbins=100, title="Return Distribution", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
    with c_right:
        mc = AnalyticsEngine.monte_carlo(df[price_col].iloc[-1], m['Returns'].mean(), m['Returns'].std())
        st.plotly_chart(mc.update_layout(template="plotly_dark", title="Monte Carlo (50 Paths)", showlegend=False), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_advisor():
    st.header("💳 AI Wealth Advisor & Document Processing")
    
    # 2. NEW FLEXIBLE UPLOAD SYSTEM
    st.subheader("📥 Intelligent Document Upload")
    up_file = st.file_uploader("Upload your Transaction CSV file:", type="csv")
    
    if up_file:
        df_raw = pd.read_csv(up_file)
        st.write("### Data Preview")
        st.dataframe(df_raw.head(5), use_container_width=True)
        
        st.warning("Please map your CSV columns to the system:")
        c1, c2 = st.columns(2)
        cat_map = c1.selectbox("Which column represents 'Category'?", df_raw.columns)
        amt_map = c2.selectbox("Which column represents 'Amount'?", df_raw.columns)
        
        if st.button("Process & Analyze My Wealth"):
            # Transformation
            processed_df = df_raw[[cat_map, amt_map]].rename(columns={cat_map: 'Category', amt_map: 'Amount'})
            processed_df['Amount'] = pd.to_numeric(processed_df['Amount'], errors='coerce').fillna(0)
            
            # Calculations
            income = processed_df[processed_df['Amount'] > 0]['Amount'].sum()
            outflow_df = processed_df[processed_df['Amount'] < 0].copy()
            outflow_df['Abs'] = outflow_df['Amount'].abs()
            
            if income > 0:
                invest_val = outflow_df[outflow_df['Category'].str.contains('Wealth|Invest|Stock|Saving', case=False, na=False)]['Abs'].sum()
                w_rate = (invest_val / income) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Income", f"${income:,.0f}")
                m2.metric("Wealth Rate", f"{w_rate:.1f}%")
                m3.metric("Net Surplus", f"${income - outflow_df['Abs'].sum():,.0f}")
                
                
                st.plotly_chart(px.pie(outflow_df, values='Abs', names='Category', hole=0.6, template="plotly_dark"), use_container_width=True)
                
                if w_rate < 20: st.warning("Wealth creation rate is below the 20% benchmark.")
                else: st.success("Exceptional capital allocation structure.")

# ----------------------------------------------------------

def render_neural_forecast():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Enter Ticker for 90-Day Forecast:", "BTC-USD").upper()
    if st.button("Execute Neural Training"):
        df = yf.download(target, period="3y").reset_index()
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            p_df = pd.DataFrame({'ds': df['Date'].dt.tz_localize(None), 'y': df['Close']}).dropna()
            m = Prophet(daily_seasonality=True).fit(p_df)
            future = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_df['ds'], y=p_df['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=future['ds'], y=future['yhat'], name="Forecast", line=dict(color='#FFD700', dash='dash')))
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Projection: {target}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, future), use_container_width=True)

# ==========================================================
# 4. MAIN ROUTER
# ==========================================================
def main():
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Navigation:", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_forecast()
    elif nav == "Wealth Management": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Status: Operational | {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
