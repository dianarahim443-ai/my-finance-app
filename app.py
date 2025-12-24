import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_components_plotly
import numpy as np
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url('https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2070');
        background-size: cover;
    }
    .main .block-container {
        background: rgba(15, 15, 15, 0.9);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid #333;
    }
    h1, h2, h3 { color: #FFD700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINES ---
@st.cache_data(ttl=3600)
def get_market_pulse():
    tickers = {"S&P 500": "^GSPC", "Nasdaq 100": "^IXIC", "Gold": "GC=F", "Bitcoin": "BTC-USD"}
    data = {}
    for name, sym in tickers.items():
        try:
            df = yf.download(sym, period="2d", progress=False)
            # رفع مشکل MultiIndex در ورژن جدید yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            price = float(df['Close'].iloc[-1])
            prev_price = float(df['Close'].iloc[-2])
            change = ((price - prev_price) / prev_price) * 100
            data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

def run_monte_carlo(last_price, mu, sigma, days=30, sims=50):
    simulation_df = pd.DataFrame()
    for i in range(sims):
        prices = [last_price]
        for _ in range(days):
            prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal()))
        simulation_df[i] = prices
    return simulation_df

# --- 3. MAIN INTERFACE ---
def main():
    st.title("🏛️ Diana Finance: AI Research Platform")
    
    pulse = get_market_pulse()
    p_cols = st.columns(len(pulse))
    for i, (name, val) in enumerate(pulse.items()):
        p_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    
    st.sidebar.title("🔬 Navigation")
    page = st.sidebar.selectbox("Go to:", 
        ["📚 Methodology", "📈 Equity Intelligence", "🔮 AI Prediction", "💳 Wealth Advisor"])

    if page == "📚 Methodology":
        st.header("📑 Mathematical Logic")
        st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
        st.info("System uses Prophet for Additive Forecasting and GBM for Risk Stress Testing.")
        

    elif page == "📈 Equity Intelligence":
        st.header("🔍 Backtesting Strategy")
        ticker = st.text_input("Ticker:", "NVDA").upper()
        if st.button("Analyze"):
            data_raw = yf.download(ticker, period="2y", progress=False)
            if isinstance(data_raw.columns, pd.MultiIndex):
                data_raw.columns = data_raw.columns.get_level_values(0)
            
            prices = data_raw['Close'].squeeze()
            returns = prices.pct_change().dropna()
            
            # Monte Carlo Simulation
            st.subheader("Monte Carlo Risk Simulation")
            sims = run_monte_carlo(prices.iloc[-1], returns.mean(), returns.std())
            st.plotly_chart(px.line(sims, template="plotly_dark").update_layout(showlegend=False))

    elif page == "🔮 AI Prediction":
        st.header("🔮 AI Forecasting Engine")
        asset = st.text_input("Enter Asset:", "BTC-USD").upper()
        
        if st.button("Predict"):
            with st.spinner("Training Model..."):
                raw = yf.download(asset, period="3y", progress=False).reset_index()
                
                # رفع فیت شدن ستون‌ها (فیکس کردن خطای TypeError)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                
                # آماده‌سازی دقیق برای Prophet
                df_p = pd.DataFrame()
                df_p['ds'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)
                df_p['y'] = pd.to_numeric(raw['Close'], errors='coerce')
                df_p = df_p.dropna()

                m = Prophet(daily_seasonality=True)
                m.fit(df_p)
                
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual"))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(dash='dash')))
                fig.update_layout(template="plotly_dark", title=f"30-Day Outlook: {asset}")
                st.plotly_chart(fig, use_container_width=True)
                

    elif page == "💳 Wealth Advisor":
        st.header("💳 Financial Behavior Audit")
        # دیتای پیش‌فرض برای جلوگیری از ارور در صورت عدم آپلود
        df_sample = pd.DataFrame({
            'Description': ['Salary', 'Rent', 'Invest', 'Food', 'Uber'],
            'Amount': [5000, -1500, -1000, -300, -100]
        })
        st.table(df_sample)
        st.success("AI Recommendation: Maintain 20% Wealth Building Allocation.")

if __name__ == "__main__":
    main()
