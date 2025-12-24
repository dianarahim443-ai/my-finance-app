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
# 1. سیستم مرکزی و استایلینگ سلطنتی (Sovereign Style)
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Diana Sovereign AI | Terminal", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #050505; color: #E0E0E0; }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98); border-radius: 35px; 
        padding: 50px 70px; border: 1px solid #2a2a2a;
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4rem !important; }
    h2, h3 { color: #FFD700 !important; border-left: 6px solid #FFD700; padding-left: 20px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-top: 4px solid #FFD700; }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 800; border-radius: 10px; height: 3.5em; width: 100%;
    }
    .upload-box { border: 2px dashed #FFD700; padding: 30px; border-radius: 20px; background: rgba(255, 215, 0, 0.02); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. موتور محاسبات ریسک و تحلیل کوانت
# ==========================================
class SovereignEngine:
    @staticmethod
    def calculate_metrics(df, price_col):
        returns = df[price_col].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma != 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        mdd = ((df[price_col] / df[price_col].cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "VaR": var_95, "MDD": mdd, "Vol": sigma * np.sqrt(252) * 100, "Returns": returns}

# ==========================================
# 3. بخش‌های مختلف رابط کاربری
# ==========================================

def render_equity_intelligence():
    st.header("📈 Equity Intelligence & Private Audit")
    tab1, tab2 = st.tabs(["🌐 Live Global Market", "📥 Private CSV Upload"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        ticker = c1.text_input("Enter Ticker (e.g. RACE, BTC-USD):", "RACE").upper()
        period = c2.selectbox("Horizon:", ["1y", "2y", "5y", "max"])
        if st.button("Start Live Audit"):
            data = yf.download(ticker, period=period)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                display_stock_analysis(data, ticker)

    with tab2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        up_file = st.file_uploader("Upload Stock/Crypto CSV", type="csv", key="stock_up")
        st.markdown('</div>', unsafe_allow_html=True)
        if up_file:
            df = pd.read_csv(up_file)
            st.info("Identify your columns:")
            col_p = st.selectbox("Select Price Column:", df.columns)
            if st.button("Analyze Uploaded Data"):
                display_stock_analysis(df, "Private Data", col_p)

def display_stock_analysis(df, label, price_col='Close'):
    m = SovereignEngine.calculate_metrics(df, price_col)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Price", f"{df[price_col].iloc[-1]:,.2f}")
    c2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
    c4.metric("Risk (VaR 95%)", f"{m['VaR']:.2f}%")
    
    st.plotly_chart(px.line(df, y=price_col, title=f"Trajectory: {label}", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
    st.plotly_chart(px.histogram(m['Returns'], nbins=100, title="Return Distribution", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)

# ------------------------------------------

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    t_man, t_file = st.tabs(["📝 Manual Entry", "📥 Smart CSV Upload"])
    
    final_df = pd.DataFrame()

    with t_man:
        default = [{"Category": "Income", "Amount": 15000}, {"Category": "Fixed Costs", "Amount": -4500}, 
                   {"Category": "Wealth/Invest", "Amount": -3500}, {"Category": "Lifestyle", "Amount": -1500}]
        final_df = st.data_editor(pd.DataFrame(default), num_rows="dynamic", use_container_width=True)

    with t_file:
        up_w = st.file_uploader("Upload Transaction CSV", type="csv")
        if up_w:
            raw_df = pd.read_csv(up_w)
            st.write("📋 Column Mapping:")
            c_cat = st.selectbox("Category Column:", raw_df.columns)
            c_amt = st.selectbox("Amount Column:", raw_df.columns)
            if st.button("Sync & Analyze File"):
                final_df = raw_df[[c_cat, c_amt]].rename(columns={c_cat: 'Category', c_amt: 'Amount'})

    if not final_df.empty:
        try:
            final_df['Amount'] = pd.to_numeric(final_df['Amount'], errors='coerce').fillna(0)
            income = final_df[final_df['Amount'] > 0]['Amount'].sum()
            outflows = final_df[final_df['Amount'] < 0].copy()
            outflows['Abs'] = outflows['Amount'].abs()
            
            if income > 0:
                invest = outflows[outflows['Category'].astype(str).str.contains('Wealth|Invest|بورس|پس‌انداز', case=False, na=False)]['Abs'].sum()
                w_rate = (invest / income) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Income", f"${income:,.0f}")
                m2.metric("Wealth Rate", f"{w_rate:.1f}%")
                m3.metric("Net Savings", f"${income - outflows['Abs'].sum():,.0f}")
                
                st.divider()
                col_c, col_v = st.columns([1.5, 1])
                with col_c:
                    st.plotly_chart(px.pie(outflows, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr), use_container_width=True)
                with col_v:
                    st.subheader("🕵️ AI Financial Verdict")
                    if w_rate < 20: st.warning("Wealth Creation Rate is below 20%. Increase capital velocity.")
                    else: st.success("Optimal Structure Detected. Capital allocation meets Sovereign standards.")
        except Exception as e: st.error(f"Mapping Error: {e}")

# ------------------------------------------

def render_neural_forecast():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Asset for 90-Day Forecast:", "BTC-USD").upper()
    if st.button("Run Neural Training"):
        with st.spinner("Processing..."):
            raw = yf.download(target, period="3y", progress=False).reset_index()
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                df_p = pd.DataFrame({'ds': raw['Date'].dt.tz_localize(None), 'y': raw['Close']}).dropna()
                m = Prophet(daily_seasonality=True).fit(df_p)
                forecast = m.predict(m.make_future_dataframe(periods=90))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color='#FFD700', dash='dash')))
                st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Neural Forecast: {target}"), use_container_width=True)
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ==========================================
# 4. کنترل‌کننده مرکزی (Main Router)
# ==========================================

def main():
    st.sidebar.title("💎 Diana Sovereign")
    st.sidebar.markdown("Institutional AI Terminal")
    nav = st.sidebar.radio("Navigation:", ["Equity Intelligence", "Neural Forecasting", "Wealth Management", "Risk Framework"])
    
    if nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_forecast()
    elif nav == "Wealth Management": render_wealth_advisor()
    elif nav == "Risk Framework":
        st.header("🔬 Strategic Risk Framework")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("Modeling asset drift and diffusion via GBM.")

    st.sidebar.divider()
    st.sidebar.caption(f"Status: Active | {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
