import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from scipy.stats import norm

# --- 1. تنظیمات سیستمی (Academic Presentation Mode) ---
st.set_page_config(page_title="QuantFinance AI | Research Platform", layout="wide")

# --- 2. موتور محاسبات کوانت (Advanced Quant Engine) ---
@st.cache_data(ttl=3600)
def get_advanced_analytics(ticker):
    try:
        data = yf.download(ticker, period="2y")['Close']
        if data.empty: return None
        
        returns = data.pct_change().dropna()
        
        # محاسبات شاخص‌های ریسک (مورد نیاز برای دفاع ارشد)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        var_95 = np.percentile(returns, 5) # Value at Risk
        volatility = returns.std() * np.sqrt(252)
        
        return {
            "data": data,
            "returns": returns,
            "sharpe": sharpe,
            "var": var_95,
            "volatility": volatility
        }
    except: return None

# --- 3. بدنه اصلی اپلیکیشن ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Quantitative Analysis")
    st.markdown("---")

    # سایدبار برای ناوبری متدولوژی
    st.sidebar.title("🔬 Methodology")
    menu = st.sidebar.radio("Select Analysis Module:", 
                           ["Market Intelligence", "Predictive Modeling", "Risk Management"])

    # --- ماژول ۱: هوش بازار و همبستگی ---
    if menu == "Market Intelligence":
        st.header("🌍 Global Asset Correlation & Performance")
        
        tickers = ["^GSPC", "GC=F", "BTC-USD", "EURUSD=X"]
        df_market = yf.download(tickers, period="1y")['Close'].pct_change().dropna()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Asset Correlation")
            corr = df_market.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with col2:
            st.subheader("Performance Metrics")
            st.dataframe(df_market.describe().T[['mean', 'std', 'min', 'max']])

    # --- ماژول ۲: پیش‌بینی سری زمانی با Prophet ---
    elif menu == "Predictive Modeling":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Enter Asset Ticker:", "NVDA").upper()
        
        if st.button("Train AI Model"):
            with st.spinner("Optimizing Hyperparameters..."):
                df_raw = yf.download(symbol, period="5y").reset_index()
                df_prophet = df_raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                
                m = Prophet(daily_seasonality=True, interval_width=0.95)
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=90)
                forecast = m.predict(future)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.1)'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.1)'))
                st.plotly_chart(fig, use_container_width=True)

    # --- ماژول ۳: مدیریت ریسک و توزیع بازدهی ---
    elif menu == "Risk Management":
        st.header("📉 Financial Risk Profiling")
        symbol = st.text_input("Enter Asset:", "AAPL").upper()
        
        analysis = get_advanced_analytics(symbol)
        if analysis:
            c1, c2, c3 = st.columns(3)
            c1.metric("Annualized Volatility", f"{analysis['volatility']:.2%}")
            c2.metric("Sharpe Ratio", f"{analysis['sharpe']:.2f}")
            c3.metric("Daily VaR (95%)", f"{analysis['var']:.2%}")
            
            st.subheader("Returns Distribution Analysis")
            fig_dist = px.histogram(analysis['returns'], nbins=100, marginal="box", 
                                     title=f"Statistical Distribution of {symbol} Returns")
            st.plotly_chart(fig_dist, use_container_width=True)
            
    st.sidebar.divider()
    st.sidebar.caption("Thesis Candidate: [Your Name] | University: [Your University]")

if __name__ == "__main__":
    main()
# --- ماژول ۴: تحلیل ۳۶۰ درجه سهام بین‌المللی ---
    elif menu == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        
        ticker = st.text_input("Enter International Ticker (e.g., TSLA, MSFT, BABA, Ferrari: RACE):", "TSLA").upper()
        
        if ticker:
            stock_obj = yf.Ticker(ticker)
            
            # ۱. استخراج اطلاعات بنیادی (Fundamental Data)
            with st.expander("🏢 Company Profile & Fundamentals", expanded=True):
                info = stock_obj.info
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
                c2.metric("P/E Ratio", info.get('trailingPE', 'N/A'))
                c3.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                c4.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
                st.write(f"**Description:** {info.get('longBusinessSummary', 'No description available.')[:500]}...")

            # ۲. تحلیل تکنیکال (Technical Indicators)
            st.subheader("📈 Technical Strategy Indicators")
            df_tech = stock_obj.history(period="1y")
            
            # محاسبه میانگین‌های متحرک
            df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
            df_tech['MA200'] = df_tech['Close'].rolling(window=200).mean()
            
            # محاسبه RSI (شاخص قدرت نسبی)
            delta = df_tech['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_tech['RSI'] = 100 - (100 / (1 + rs))

            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['Close'], name='Price', line=dict(color='white')))
            fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA50'], name='MA 50 (Short-term)', line=dict(color='orange')))
            fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA200'], name='MA 200 (Long-term)', line=dict(color='red')))
            st.plotly_chart(fig_tech, use_container_width=True)

            # ۳. نمایش RSI در یک نمودار کوچک
            fig_rsi = px.line(df_tech, y='RSI', title="RSI (Relative Strength Index) - Overbought > 70 | Oversold < 30")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True)

            # ۴. تحلیل صورت‌های مالی (Financials)
            st.subheader("📊 Financial Health (Annual Net Income)")
            income_stmt = stock_obj.financials.loc['Net Income']
            st.bar_chart(income_stmt)
