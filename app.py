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
# 1. تنظیمات سیستمی و استایلینگ فوق‌حرفه‌ای (Institutional UI)
# ==========================================================
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Diana Sovereign AI | Terminal V27",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover; background-attachment: fixed;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.98); border-radius: 40px; 
        padding: 50px 80px; border: 1px solid #2a2a2a; box-shadow: 0 40px 150px rgba(0,0,0,1);
    }
    h1 { color: #FFD700 !important; font-weight: 900; font-size: 4.5rem !important; letter-spacing: -3px; }
    h2, h3 { color: #FFD700 !important; border-left: 10px solid #FFD700; padding-left: 20px; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 25px; border-radius: 20px; border-top: 5px solid #FFD700; }
    .stButton>button {
        background: linear-gradient(45deg, #FFD700, #B8860B);
        color: black !important; font-weight: 900; border-radius: 12px; height: 3.5em; width: 100%;
    }
    .upload-box { border: 2px dashed #FFD700; padding: 30px; border-radius: 20px; background: rgba(255, 215, 0, 0.02); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. موتور محاسبات کوانت (Sovereign Engine)
# ==========================================================
class SovereignEngine:
    @staticmethod
    def calculate_risk_metrics(df, col='Close'):
        returns = df[col].pct_change().dropna()
        if returns.empty: return None
        mu, sigma = returns.mean(), returns.std()
        sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
        var_95 = norm.ppf(0.05, mu, sigma) * 100
        cum_rets = (1 + returns).cumprod()
        mdd = ((cum_rets / cum_rets.cummax()) - 1).min() * 100
        return {"Sharpe": sharpe, "MDD": mdd, "VaR": var_95, "Vol": sigma * np.sqrt(252) * 100, "Returns": returns}

    @staticmethod
    def run_monte_carlo(last_price, mu, sigma, days=45, sims=70):
        mc_fig = go.Figure()
        for _ in range(sims):
            path = [last_price]
            for _ in range(days):
                path.append(path[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
            mc_fig.add_trace(go.Scatter(y=path, mode='lines', opacity=0.1, line=dict(color='#FFD700')))
        return mc_fig

# ==========================================================
# 3. ماژول‌های رابط کاربری (بدون هیچ حذفیاتی)
# ==========================================================

def render_risk_framework():
    st.header("🔬 Strategic Risk Framework")
    t1, t2, t3 = st.tabs(["Stochastic Calculus", "Tail-Risk Theory", "Neural Decomposition"])
    with t1:
        st.subheader("I. Geometric Brownian Motion (GBM)")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.write("این معادله دیفرانسیل تصادفی زیربنای مدل‌سازی حرکت قیمت‌ها در سیستم ماست.")
        
    with t2:
        st.subheader("II. Value at Risk (VaR)")
        st.latex(r"VaR_{\alpha} = \mu + \sigma \cdot \Phi^{-1}(\alpha)")
        st.write("محاسبه حداکثر پتانسیل ریزش در بازه اطمینان ۹۵ درصد.")
        
    with t3:
        st.subheader("III. Neural Prophet Model")
        st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
        st.write("تجزیه سیگنال به مولفه‌های روند، فصلی بودن و تعطیلات.")

# ----------------------------------------------------------

def render_equity_intelligence():
    st.header("📈 Equity Intelligence & Custom Audit")
    source = st.sidebar.selectbox("Data Source:", ["Live Terminal", "Institutional Upload (CSV)"])
    
    if source == "Live Terminal":
        ticker = st.text_input("Enter Ticker:", "RACE").upper()
        if st.button("Initialize Deep Run"):
            df = yf.download(ticker, period="2y")
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                display_analysis(df, ticker)
    else:
        up_file = st.file_uploader("آپلود دیتای بورسی (CSV):", type="csv")
        if up_file:
            df = pd.read_csv(up_file)
            col_p = st.selectbox("انتخاب ستون قیمت (Close/Price):", df.columns)
            if st.button("Audit Custom Data"):
                display_analysis(df, "Custom Asset", col_p)

def display_analysis(df, label, col='Close'):
    m = SovereignEngine.calculate_risk_metrics(df, col)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest Price", f"{df[col].iloc[-1]:,.2f}")
    k2.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
    k3.metric("Max Drawdown", f"{m['MDD']:.2f}%")
    k4.metric("Risk VaR", f"{m['VaR']:.2f}%")
    
    st.plotly_chart(px.line(df, y=col, title=f"Historical Audit: {label}", template="plotly_dark").update_traces(line_color="#FFD700"), use_container_width=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(px.histogram(m['Returns'], nbins=80, title="Return Density", template="plotly_dark", color_discrete_sequence=['#FFD700']), use_container_width=True)
    with c_right:
        mc = SovereignEngine.run_monte_carlo(df[col].iloc[-1], m['Returns'].mean(), m['Returns'].std())
        st.plotly_chart(mc.update_layout(template="plotly_dark", title="70 Stochastic Paths", showlegend=False), use_container_width=True)

# ----------------------------------------------------------

def render_neural_prediction():
    st.header("🔮 Neural Predictive Engine")
    target = st.text_input("Asset for 90-Day Neural Forecast:", "BTC-USD").upper()
    if st.button("Deploy Prophet V5 Model"):
        raw = yf.download(target, period="3y", progress=False).reset_index()
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
            df_p = pd.DataFrame({'ds': raw['Date'].dt.tz_localize(None), 'y': raw['Close']}).dropna()
            m = Prophet(daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=90))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual", line=dict(color='#00F2FF')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color='#FFD700', dash='dash')))
            st.plotly_chart(fig.update_layout(template="plotly_dark", title=f"90-Day Forecast: {target}"), use_container_width=True)
            st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

# ----------------------------------------------------------

def render_wealth_advisor():
    st.header("💳 AI Wealth Management Advisor")
    st.write("مدیریت بودجه و آپلود اسناد مالی برای تحلیل هوشمند دارایی‌ها.")
    
    t1, t2 = st.tabs(["📝 Manual Ledger (ورود دستی)", "📥 Smart Document Upload (آپلود مدارک)"])
    final_wealth_df = pd.DataFrame()

    with t1:
        default_data = [
            {"Category": "Income", "Amount": 15000},
            {"Category": "Fixed Costs", "Amount": -4500},
            {"Category": "Investments/Wealth", "Amount": -3500},
            {"Category": "Lifestyle", "Amount": -1500}
        ]
        final_wealth_df = st.data_editor(pd.DataFrame(default_data), num_rows="dynamic", use_container_width=True)

    with t2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        up_w = st.file_uploader("آپلود فایل تراکنش‌ها یا لیست دارایی (CSV):", type="csv")
        st.markdown('</div>', unsafe_allow_html=True)
        if up_w:
            raw_w = pd.read_csv(up_w)
            st.info("نگاشت ستون‌ها (Column Mapping):")
            col_sel1, col_sel2 = st.columns(2)
            cat_col = col_sel1.selectbox("ستون دسته‌بندی:", raw_w.columns, key="cat_up")
            amt_col = col_sel2.selectbox("ستون مبلغ/تراکنش:", raw_w.columns, key="amt_up")
            if st.button("Sync Document Data"):
                final_wealth_df = raw_w[[cat_col, amt_col]].rename(columns={cat_col: 'Category', amt_col: 'Amount'})
        else:
            st.info("فایلی آپلود نشده است؛ از داده‌های جدول دستی استفاده می‌شود.")

    if not final_wealth_df.empty:
        try:
            final_wealth_df['Amount'] = pd.to_numeric(final_wealth_df['Amount'], errors='coerce').fillna(0)
            income = final_wealth_df[final_wealth_df['Amount'] > 0]['Amount'].sum()
            outflows = final_wealth_df[final_wealth_df['Amount'] < 0].copy()
            outflows['Abs'] = outflows['Amount'].abs()
            
            if income > 0:
                invest_keywords = 'Wealth|Invest|Stock|Gold|پس‌انداز|ثروت|سرمایه'
                invest_val = outflows[outflows['Category'].astype(str).str.contains(invest_keywords, case=False, na=False)]['Abs'].sum()
                w_rate = (invest_val / income) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Income", f"${income:,.0f}")
                m2.metric("Wealth Creation Rate", f"{w_rate:.1f}%")
                m3.metric("Net Surplus", f"${income - outflows['Abs'].sum():,.0f}")
                
                st.divider()
                c_chart, c_advise = st.columns([1.5, 1])
                with c_chart:
                    
                    fig = px.pie(outflows, values='Abs', names='Category', hole=0.6, template="plotly_dark", color_discrete_sequence=px.colors.sequential.YlOrBr)
                    st.plotly_chart(fig, use_container_width=True)
                with c_advise:
                    st.subheader("🕵️ AI Financial Verdict")
                    if w_rate < 20:
                        st.warning("هشدار: نرخ ثروت‌سازی شما زیر استاندارد ۲۰٪ است. پیشنهاد می‌شود خروجی‌های غیرضروری را کاهش دهید.")
                    else:
                        st.success("عالی: ساختار توزیع دارایی شما در کلاس استاندارد جهانی قرار دارد.")
        except Exception as e:
            st.error(f"خطا در تحلیل اسناد: {e}")

# ==========================================================
# 4. کنترل‌کننده مرکزی (Master Controller)
# ==========================================================
def main():
    st.sidebar.title("💎 Diana Sovereign")
    nav = st.sidebar.radio("Navigation Domains:", ["Risk Framework", "Equity Intelligence", "Neural Forecasting", "Wealth Management"])
    
    # Market Pulse Ribbon
    watch = {"S&P 500": "^GSPC", "Gold": "GC=F", "Bitcoin": "BTC-USD"}
    cols = st.columns(len(watch))
    for i, (n, s) in enumerate(watch.items()):
        try:
            d = yf.download(s, period="2d", progress=False)
            cols[i].metric(n, f"{d['Close'].iloc[-1]:,.2f}", f"{(d['Close'].iloc[-1]/d['Close'].iloc[-2]-1)*100:+.2f}%")
        except: pass
    st.divider()

    if nav == "Risk Framework": render_risk_framework()
    elif nav == "Equity Intelligence": render_equity_intelligence()
    elif nav == "Neural Forecasting": render_neural_prediction()
    elif nav == "Wealth Management": render_wealth_advisor()
    
    st.sidebar.divider()
    st.sidebar.caption(f"Terminal V27 | Active Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
