import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime

# --- 1. تنظیمات سیستمی و ظاهر فوق حرفه‌ای ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url('https://images.unsplash.com/photo-1611974717483-30510c436662?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(10, 10, 10, 0.9);
        border-radius: 25px;
        padding: 50px;
        border: 1px solid #444;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    h1, h2, h3 { color: #FFD700 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border-bottom: 3px solid #FFD700; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. توابع محاسبات سنگین (کوانت و هوش مصنوعی) ---

@st.cache_data(ttl=3600)
def get_market_pulse():
    tickers = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD"}
    data = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period="2d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            price = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2])
            change = ((price - prev) / prev) * 100
            data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def run_backtest(data):
    # استراتژی تقاطع طلایی (Golden Cross)
    fast = data.rolling(window=20).mean()
    slow = data.rolling(window=50).mean()
    signal = np.where(fast > slow, 1, 0)
    returns = data.pct_change()
    strat_returns = returns * pd.Series(signal).shift(1).values
    equity_curve = 10000 * (1 + strat_returns.fillna(0)).cumprod()
    
    # شاخص‌های ریسک آکادمیک
    rf = 0.02 / 252 # نرخ بدون ریسک فرض شده
    excess = strat_returns.fillna(0) - rf
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() != 0 else 0
    mdd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100
    return equity_curve, sharpe, mdd

def monte_carlo(last_price, mu, sigma, days=30, sims=100):
    simulation_df = pd.DataFrame()
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
        simulation_df[i] = prices
    return simulation_df

# --- 3. بدنه اصلی اپلیکیشن ---

def main():
    st.title("🏛️ Diana Finance: Institutional AI Research")
    st.write("پلتفرم جامع تحلیل بازارهای سرمایه و مدیریت ثروت مبتنی بر هوش مصنوعی")
    
    # نمایش وضعیت بازار در هدر
    pulse = get_market_pulse()
    p_cols = st.columns(len(pulse))
    for i, (name, val) in enumerate(pulse.items()):
        p_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    
    st.divider()

    # منوی ناوبری اصلی (سایدبار)
    st.sidebar.title("🔬 Research Core")
    page = st.sidebar.selectbox("انتخاب ماژول تحقیقاتی:", 
        ["📚 Research & Methodology", 
         "📈 Equity Intelligence (Backtest)", 
         "🔮 AI Predictive Engine", 
         "💳 Wealth Management (Personal)"])

    # --- صفحه ۱: مستندات و فرمول‌ها (برگشت داده شد) ---
    if page == "📚 Research & Methodology":
        st.header("📑 چارچوب متدولوژی کوانت (Quantitative Framework)")
        t1, t2, t3 = st.tabs(["مدل ریاضی بازدهی", "معماری هوش مصنوعی", "اهداف پروژه"])
        
        with t1:
            st.subheader("Governing SDE (Geometric Brownian Motion)")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.markdown("""
            مدل **GBM** ستون اصلی شبیه‌سازی‌های ریسک ماست:
            - **Expected Return ($\mu$):** میانگین بازدهی تاریخی دارایی.
            - **Volatility ($\sigma$):** انحراف معیار که نشان‌دهنده ریسک بازار است.
            - **Wiener Process ($dW_t$):** حرکت براونی که نوسانات تصادفی را شبیه‌سازی می‌کند.
            """)
            
        with t2:
            st.subheader("Prophet Decomposable Model")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.write("ما از مدل افزودنی (Additive Model) برای تفکیک روندها (Trend) از اثرات فصلی (Seasonality) استفاده می‌کنیم.")

    # --- صفحه ۲: تحلیل سهام و بک‌تست (فوق‌کامل) ---
    elif page == "📈 Equity Intelligence (Backtest)":
        st.header("🔍 استراتژی‌های معاملاتی و تحلیل ریسک")
        ticker = st.text_input("نماد بورسی یا کریپتو را وارد کنید:", "NVDA").upper()
        
        if st.button("اجرای تحلیل عمیق"):
            with st.spinner("در حال دریافت داده‌های بازار..."):
                raw = yf.download(ticker, period="2y", progress=False)
                if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
                
                prices = raw['Close'].squeeze()
                equity, sharpe, mdd = run_backtest(prices)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("بازدهی نهایی استراتژی", f"{((equity.iloc[-1]/10000)-1)*100:.2f}%")
                c2.metric("شاخص شارپ (Risk-Adj)", f"{sharpe:.2f}")
                c3.metric("بیشترین افت سرمایه (MDD)", f"{mdd:.2f}%")
                
                st.plotly_chart(px.line(equity, title="منحنی رشد سرمایه (Equity Curve)", template="plotly_dark", color_discrete_sequence=['#FFD700']))
                
                # شبیه‌سازی مونت‌کارلو برای ۳۰ روز آینده
                st.subheader("🎲 تست استرس مونت‌کارلو (Stress Test)")
                returns = prices.pct_change().dropna()
                sims_df = monte_carlo(prices.iloc[-1], returns.mean(), returns.std())
                fig_mc = px.line(sims_df, template="plotly_dark", title="۱۰۰ مسیر احتمالی قیمت در ۳۰ روز آینده")
                fig_mc.update_layout(showlegend=False)
                st.plotly_chart(fig_mc, use_container_width=True)

    # --- صفحه ۳: پیش‌بینی هوش مصنوعی (اصلاح شده و کامل) ---
    elif page == "🔮 AI Predictive Engine":
        st.header("🔮 موتور پیش‌بینی سری زمانی Prophet")
        asset = st.text_input("نماد برای پیش‌بینی (مثلاً BTC-USD):", "BTC-USD").upper()
        
        if st.button("شروع پیش‌بینی عصبی"):
            with st.spinner("در حال آموزش مدل هوش مصنوعی..."):
                raw_data = yf.download(asset, period="3y", progress=False).reset_index()
                if isinstance(raw_data.columns, pd.MultiIndex): raw_data.columns = raw_data.columns.get_level_values(0)
                
                # آماده‌سازی دیتا برای Prophet بدون باگ
                df_p = pd.DataFrame()
                df_p['ds'] = pd.to_datetime(raw_data['Date']).dt.tz_localize(None)
                df_p['y'] = pd.to_numeric(raw_data['Close'], errors='coerce')
                df_p = df_p.dropna()

                m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
                m.fit(df_p)
                
                forecast = m.predict(m.make_future_dataframe(periods=60))
                
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="دیتای واقعی", line=dict(color='#00F2FF')))
                fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="پیش‌بینی AI", line=dict(dash='dash', color='#FFD700')))
                fig_forecast.update_layout(template="plotly_dark", title=f"چشم‌انداز ۶۰ روزه {asset}")
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader("تحلیل چرخه‌های رفتاری (Components)")
                st.plotly_chart(plot_components_plotly(m, forecast), use_container_width=True)

    # --- صفحه ۴: مدیریت مالی شخصی (بازگشت بخش‌های مدیریتی) ---
    elif page == "💳 Wealth Management (Personal)":
        st.header("💳 مدیریت ثروت هوشمند (Wealth Advisor)")
        
        # لجر تراکنش‌ها (دیتاست نمونه کامل)
        df_ledger = pd.DataFrame({
            'شرح': ['حقوق ماهانه', 'اجاره خانه', 'سرمایه‌گذاری ETF', 'خرید آمازون', 'اوبر', 'سوپرمارکت', 'پس‌انداز طلا'],
            'مبلغ': [6500, -1800, -1200, -400, -100, -500, -500],
            'دسته‌بندی': ['Income', 'Fixed', 'Wealth', 'Wants', 'Wants', 'Fixed', 'Wealth']
        })
        
        st.subheader("خلاصه وضعیت جریان نقدینگی")
        st.table(df_ledger)
        
        outflow = df_ledger[df_ledger['مبلغ'] < 0].copy()
        outflow['مبلغ'] = outflow['مبلغ'].abs()
        total_spent = outflow['مبلغ'].sum()
        
        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig_p = px.pie(outflow, values='مبلغ', names='دسته‌بندی', hole=0.5, 
                           template="plotly_dark", title="توزیع مخارج بر اساس مدل ۵۰/۳۰/۲ economic")
            st.plotly_chart(fig_p, use_container_width=True)
            
        with col2:
            st.subheader("بررسی سلامت مالی (50/30/20)")
            w_pct = (outflow[outflow['دسته‌بندی'] == 'Wealth']['مبلغ'].sum() / total_spent) * 100
            st.metric("نرخ ثروت‌سازی (Wealth Building)", f"{w_pct:.1f}%", delta=f"{w_pct-20:.1f}% (هدف ۲۰٪)")
            
            if w_pct < 20:
                st.error("هشدار: نرخ سرمایه‌گذاری شما پایین‌تر از استاندارد است.")
            else:
                st.success("تبریک: رفتار مالی شما با استانداردهای انباشت سرمایه منطبق است.")

    st.sidebar.divider()
    st.sidebar.caption("Diana AI Framework v4.0 | Fully Reintegrated")

if __name__ == "__main__":
    main()
