import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Diana Finance AI | Institutional Research", layout="wide")

# --- 2. CORE ENGINES ---
@st.cache_data(ttl=3600)
def get_global_metrics():
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            if len(df) >= 2:
                price = float(df['Close'].iloc[-1])
                change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                data[name] = (price, change)
        except: data[name] = (0.0, 0.0)
    return data

def calculate_risk_metrics(equity_curve, strategy_returns):
    rf = 0.02 / 252 
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    excess_returns = strategy_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1)
    max_dd = drawdown.min() * 100
    return total_return, sharpe, max_dd

# --- 3. MAIN APPLICATION ---
def main():
    st.title("🏛️ Diana Finance: AI Research Platform")
    st.markdown("_Advanced Quantitative Modeling & Behavioral Economics_")
    
    metrics = get_global_metrics()
    m_cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")
    st.divider()

    st.sidebar.title("🔬 Research Modules")
    page = st.sidebar.selectbox("Select Module:", 
                                ["🏠 Home & Documentation", 
                                 "📈 Equity Intelligence", 
                                 "🔮 AI Prediction & Forecasting", 
                                 "💳 Behavioral Personal Finance"])

    # --- PAGE 1: DOCUMENTATION ---
    if page == "🏠 Home & Documentation":
        st.header("📑 Academic Documentation")
        t1, t2, t3 = st.tabs(["Algorithm Logic", "Backtest Assumptions", "AI Innovation"])
        with t1:
            st.subheader("System Architecture")
            st.markdown("**Prophet Engine:** Decomposable time-series model handling $y(t) = g(t) + s(t) + h(t) + \epsilon_t$.")
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.info("The GBM model assumes a stochastic process for asset price paths.")

    # --- PAGE 2: EQUITY INTELLIGENCE ---
    elif page == "📈 Equity Intelligence":
        st.header("🔍 Backtesting & Risk Analysis")
        ticker = st.text_input("Ticker:", "NVDA").upper()
        if st.button("Run Analysis"):
            data = yf.download(ticker, period="1y")['Close'].squeeze()
            if not data.empty:
                returns = data.pct_change()
                equity = 10000 * (1 + returns.fillna(0)).cumprod()
                ret, sharpe, dd = calculate_risk_metrics(equity, returns.fillna(0))
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Return", f"{ret:.2f}%")
                c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                c3.metric("Max Drawdown", f"{dd:.2f}%")
                st.plotly_chart(px.line(equity, title="Portfolio Growth", template="plotly_dark"))

    # --- PAGE 3: AI PREDICTION (خیلی کامل شده) ---
    elif page == "🔮 AI Prediction & Forecasting":
        st.header("🔮 AI Forecasting & Time-Series Decomposition")
        symbol = st.text_input("Enter Asset (e.g., BTC-USD, AAPL):", "BTC-USD").upper()
        
        if st.button("Generate Advanced Forecast"):
            with st.spinner("Training Neural Prophecy Model..."):
                df_raw = yf.download(symbol, period="3y").reset_index()
                df_p = df_raw[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                
                # Training the Model
                m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)
                
                # Plot 1: Main Forecast
                st.subheader("30-Day Predictive Price Path")
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual Price", line=dict(color="#636EFA")))
                fig_f.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="AI Prediction", line=dict(color="#00CC96", dash='dash')))
                fig_f.fill_between = True 
                fig_f.update_layout(template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig_f, use_container_width=True)
                
                # Plot 2: Trend & Seasonality Components
                st.subheader("Statistical Decomposition (Trend & Seasonality)")
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_trend = px.line(forecast, x='ds', y='trend', title="Long-term Growth Trend", template="plotly_dark")
                    st.plotly_chart(fig_trend, use_container_width=True)
                with col_b:
                    # Weekly seasonality
                    fig_weekly = px.line(forecast.iloc[:7], y='weekly', title="Weekly Behavioral Pattern", template="plotly_dark")
                    st.plotly_chart(fig_weekly, use_container_width=True)

                # AI Actionable Insights
                current_price = df_p['y'].iloc[-1]
                pred_price = forecast['yhat'].iloc[-1]
                upside = ((pred_price - current_price) / current_price) * 100
                
                st.divider()
                st.subheader("🤖 AI Market Sentiment")
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Predicted Upside/Downside", f"{upside:.2f}%")
                if upside > 0:
                    res_col2.success("Signal: BULLISH - Model expects upward momentum.")
                else:
                    res_col2.error("Signal: BEARISH - Model expects price correction.")

    # --- PAGE 4: PERSONAL FINANCE AI (خیلی کامل شده) ---
    elif page == "💳 Behavioral Personal Finance":
        st.header("💳 AI-Driven Wealth Optimization")
        st.markdown("Upload your transaction history to receive institutional-grade allocation advice.")
        
        uploaded = st.file_uploader("Upload CSV (Columns: Description, Amount)", type="csv")
        
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'Description' in df.columns and 'Amount' in df.columns:
                def categorize_ai(d):
                    d = str(d).lower()
                    if any(x in d for x in ['amazon', 'uber', 'shop', 'netflix', 'starbucks']): return 'Discretionary'
                    if any(x in d for x in ['rent', 'bill', 'electric', 'insurance', 'tax']): return 'Fixed Needs'
                    if any(x in d for x in ['stock', 'crypto', 'invest', 'saving', 'gold']): return 'Wealth Building'
                    return 'Lifestyle'
                
                df['Category'] = df['Description'].apply(categorize_ai)
                df['Amount'] = pd.to_numeric(df['Amount']).abs()
                total_outflow = df['Amount'].sum()
                
                # 50/30/20 Rule Analysis
                cat_sums = df.groupby('Category')['Amount'].sum()
                needs = cat_sums.get('Fixed Needs', 0)
                wants = cat_sums.get('Discretionary', 0) + cat_sums.get('Lifestyle', 0)
                savings = cat_sums.get('Wealth Building', 0)
                
                st.subheader("Portfolio Breakdown")
                fig_pie = px.pie(df, values='Amount', names='Category', hole=0.6, 
                                 color_discrete_sequence=px.colors.sequential.RdBu, template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.divider()
                st.subheader("📊 50/30/20 Institutional Audit")
                
                met1, met2, met3 = st.columns(3)
                met1.metric("Needs (Target 50%)", f"{(needs/total_outflow)*100:.1f}%")
                met2.metric("Wants (Target 30%)", f"{(wants/total_outflow)*100:.1f}%")
                met3.metric("Wealth (Target 20%)", f"{(savings/total_outflow)*100:.1f}%")
                
                # AI Advice Logic
                st.info("💡 **AI Advisor Recommendation:**")
                if savings < (total_outflow * 0.20):
                    st.warning(f"Your 'Wealth Building' is under-allocated. AI suggests reducing 'Wants' by ${wants*0.1:,.2f} and moving it to Equity markets.")
                else:
                    st.success("Excellent discipline. Your capital allocation aligns with high-net-worth behavioral standards.")
                
                with st.expander("View Full Behavioral Audit"):
                    st.table(df.sort_values(by='Amount', ascending=False))

    st.sidebar.divider()
    st.sidebar.caption(f"Diana AI v2.4 | Academic Research")

if __name__ == "__main__":
    main()
