import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime

# --- 1. تنظیمات اولیه سیستم ---
st.set_page_config(page_title="QuantFinance AI | Research Platform", layout="wide")

# --- 2. توابع کمکی (Helper Functions) ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    data = yf.download(tickers, period="1y")['Close']
    return data

# --- 3. بدنه اصلی برنامه ---
def main():
    st.title("🏛️ Intelligent Financial Systems & Quantitative Analysis")
    st.markdown("---")

    # منوی کناری برای ناوبری (بسیار مهم برای ساختار پایان‌نامه)
    st.sidebar.title("🔬 Methodology")
    menu = st.sidebar.radio("Select Analysis Module:", 
                           ["Market Intelligence", "Predictive Modeling", "Global Stock 360°"])

    # --- بخش اول: هوش بازار جهانی ---
    if menu == "Market Intelligence":
        st.header("🌍 Global Asset Correlation")
        tickers = ["^GSPC", "GC=F", "BTC-USD", "EURUSD=X"]
        df_market = get_market_data(tickers)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Correlation Matrix")
            corr = df_market.pct_change().dropna().corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
            
        
        with col2:
            st.subheader("Risk-Return Profile")
            returns = df_market.pct_change().dropna()
            st.dataframe(returns.describe().T[['mean', 'std', 'min', 'max']])

    # --- بخش دوم: پیش‌بینی با هوش مصنوعی ---
    elif menu == "Predictive Modeling":
        st.header("🔮 AI Time-Series Forecasting")
        symbol = st.text_input("Enter Asset Ticker (e.g. NVDA):", "NVDA").upper()
        
        if st.button("Run AI Forecast"):
            df_raw = yf.download(symbol, period="5y").reset_index()
            df_prop = df_raw[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            m = Prophet(daily_seasonality=True)
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.1)'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.1)'))
            st.plotly_chart(fig, use_container_width=True)

    # --- بخش سوم: تحلیل ۳۶۰ درجه (بخش جدید که خطا داشت) ---
    elif menu == "Global Stock 360°":
        st.header("🔍 Comprehensive Equity Intelligence")
        ticker_input = st.text_input("Enter Ticker (e.g., TSLA, AAPL, RACE):", "TSLA").upper()
        
        if ticker_input:
            stock = yf.Ticker(ticker_input)
            
            # نمایش متغیرهای کلیدی
            info = stock.info
            cols = st.columns(4)
            cols[0].metric("Price", f"${info.get('currentPrice', 'N/A')}")
            cols[1].metric("P/E Ratio", info.get('trailingPE', 'N/A'))
            cols[2].metric("Market Cap", f"{info.get('marketCap', 0):,}")
            cols[3].metric("Div. Yield", f"{info.get('dividendYield', 0)*100:.2f}%")

            # نمودار تکنیکال با میانگین متحرک
            df_tech = stock.history(period="1y")
            df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
            
            fig_tech = go.Figure()
            fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['Close'], name='Price'))
            fig_tech.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MA50'], name='MA50 Trend'))
            st.plotly_chart(fig_tech, use_container_width=True)
            

            # تحلیل سودآوری سالانه
            st.subheader("Annual Net Income (Financial Health)")
            try:
                income = stock.financials.loc['Net Income']
                st.bar_chart(income)
            except:
                st.warning("Financial statements not available for this ticker.")

    # فوتر مخصوص پایان‌نامه
    st.sidebar.divider()
    st.sidebar.caption("Project: AI-Driven Financial Analysis\nAcademic Year: 2024-2025")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from datetime import datetime, timedelta

# --- 1. Global Configuration ---
st.set_page_config(page_title="Global AI Finance Intelligence", layout="wide")

# --- 2. Market Data Engine (International Assets) ---
@st.cache_data(ttl=3600)
def get_global_market_data():
    # Ø§Ø³ØªÙØ§Ø¯Ù‡ Ø§Ø² Ø´Ø§Ø®Øµâ€ŒÙ‡Ø§ÛŒ Ø¬Ù‡Ø§Ù†ÛŒ: Ø·Ù„Ø§ØŒ Ø¨ÙˆØ±Ø³ Ø¢Ù…Ø±ÛŒÚ©Ø§ØŒ Ø¨ÛŒØªâ€ŒÚ©ÙˆÛŒÙ† Ùˆ Ø´Ø§Ø®Øµ Ø¯Ù„Ø§Ø±
    tickers = {
        "Gold (Spot)": "GC=F",
        "S&P 500 (US)": "^GSPC",
        "Bitcoin (BTC)": "BTC-USD",
        "Dollar Index (DXY)": "DX-Y.NYB"
    }
    results = {}
    for name, ticker in tickers.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty:
                curr = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                delta = ((curr - prev) / prev) * 100
                results[name] = (round(curr, 2), round(delta, 2))
        except:
            results[name] = (0, 0)
    return results

# --- 3. Synthetic Data Generator (Ø¨Ø±Ø§ÛŒ Ø§ÛŒÙ†Ú©Ù‡ Ø³Ø§ÛŒØª Ø®Ø§Ù„ÛŒ Ù†Ø¨Ø§Ø´Ø¯) ---
def generate_demo_data():
    dates = pd.date_range(start=datetime.now() - timedelta(days=180), periods=180, freq='D')
    # Ø´Ø¨ÛŒÙ‡â€ŒØ³Ø§Ø²ÛŒ Ù‡Ø²ÛŒÙ†Ù‡â€ŒÙ‡Ø§ Ø¨Ø§ Ù†ÙˆØ³Ø§Ù† ØªØµØ§Ø¯ÙÛŒ
    amounts = np.random.normal(loc=50, scale=20, size=180).cumsum() + 1000
    return pd.DataFrame({'Date': dates, 'Amount': amounts})

# --- 4. Main Interface ---
def main():
    st.title("ðŸŒ Global Financial Intelligence & Forecasting System")
    st.markdown("_An advanced AI-driven platform for predictive financial analysis and global market tracking._")
# --- International Stock Analyzer Section ---
    st.divider() # ÛŒÚ© Ø®Ø· Ø¬Ø¯Ø§Ú©Ù†Ù†Ø¯Ù‡ Ø¨Ø±Ø§ÛŒ Ø²ÛŒØ¨Ø§ÛŒÛŒ
    st.subheader("ðŸ” Global Investment Intelligence")
    
    with st.expander("Click to Analyze Specific Stocks", expanded=False):
        ticker_symbol = st.text_input("Enter Ticker (e.g., NVDA, AAPL, TSLA):", "NVDA").upper()
        
        # Ø§Ø³ØªÙØ§Ø¯Ù‡ Ø§Ø² Ø¯Ú©Ù…Ù‡ Ø¨Ø±Ø§ÛŒ Ø¬Ù„ÙˆÚ¯ÛŒØ±ÛŒ Ø§Ø² Ø¯Ø±Ø®ÙˆØ§Ø³Øªâ€ŒÙ‡Ø§ÛŒ Ù…Ú©Ø±Ø± Ùˆ Ø±ÙØ¹ Rate Limit
        if st.button("Run Stock Analysis"):
            try:
                stock_obj = yf.Ticker(ticker_symbol)
                # Ø¯Ø±ÛŒØ§ÙØª ØªØ§Ø±ÛŒØ®Ú†Ù‡ Ù‚ÛŒÙ…Øª (Ø¨Ø³ÛŒØ§Ø± Ù¾Ø§ÛŒØ¯Ø§Ø±ØªØ± Ø§Ø² info)
                stock_hist = stock_obj.history(period="1y")
                
                if not stock_hist.empty:
                    # Ù…Ø­Ø§Ø³Ø¨Ø§Øª Ù¾Ø§ÛŒÙ‡ Ù…Ø§Ù„ÛŒ
                    last_price = stock_hist['Close'].iloc[-1]
                    annual_return = ((last_price - stock_hist['Close'].iloc[0]) / stock_hist['Close'].iloc[0]) * 100
                    
                    c1, c2 = st.columns(2)
                    c1.metric(f"Current {ticker_symbol} Price", f"${last_price:.2f}")
                    c2.metric("Annual Performance", f"{annual_return:.2f}%")
                    
                    # Ø±Ø³Ù… Ù†Ù…ÙˆØ¯Ø§Ø± Ù‚ÛŒÙ…Øª Ø¨Ø§ Ø§Ø³ØªØ§ÛŒÙ„ Ø­Ø±ÙÙ‡â€ŒØ§ÛŒ
                    fig_stock_price = px.line(stock_hist, x=stock_hist.index, y='Close', 
                                            title=f"{ticker_symbol} - Year over Year Analysis",
                                            template="plotly_dark") # Ø§Ø³ØªØ§ÛŒÙ„ ØªÛŒØ±Ù‡ Ø¨Ø±Ø§ÛŒ Ú©Ù„Ø§Ø³ Ù…Ø§Ù„ÛŒ
                    st.plotly_chart(fig_stock_price, use_container_width=True)
                    
                    # ØªØ­Ù„ÛŒÙ„ Ù†ÙˆØ³Ø§Ù†Ø§Øª (Volatility Analysis) - Ø¨Ø±Ø§ÛŒ Ø±Ø²ÙˆÙ…Ù‡ Ø§Ø±Ø´Ø¯ Ø¹Ø§Ù„ÛŒ Ø§Ø³Øª
                    stock_hist['Daily Return'] = stock_hist['Close'].pct_change()
                    volatility = stock_hist['Daily Return'].std() * (252**0.5) # ÙØ±Ù…ÙˆÙ„ Ø³Ø§Ù„Ø§Ù†Ù‡ Ú©Ø±Ø¯Ù† Ù†ÙˆØ³Ø§Ù†
                    st.info(f"ðŸ“Š Annualized Volatility: {volatility:.2%}")
                    
                else:
                    st.error("Invalid Ticker or No Data Found.")
            except Exception as e:
                st.warning("âš ï¸ Market API is busy. Please wait 1 minute and click the button again.")
    # Sidebar
    st.sidebar.header("ðŸ•¹ï¸ Control Panel")
    mode = st.sidebar.radio("Select Data Source:", ["Live Demo (Simulation)", "Upload Personal Data"])
    
    # Live Market Ticker
    market_data = get_global_market_data()
    cols = st.columns(len(market_data))
    for i, (name, val) in enumerate(market_data.items()):
        cols[i].metric(name, f"{val[0]:,}", f"{val[1]}%")

    st.divider()

    # Data Processing
    if mode == "Upload Personal Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV (Required columns: Date, Amount)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            st.info("ðŸ’¡ Please upload a CSV file to begin. Using Demo data for visualization below.")
            df = generate_demo_data()
    else:
        df = generate_demo_data()

    # --- Analysis Tabs ---
    tab1, tab2, tab3 = st.tabs(["ðŸ“ˆ Wealth Forecasting", "ðŸ“Š Expense Analysis", "ðŸŒ Global Correlation"])

    with tab1:
        st.subheader("AI Time-Series Projection")
        df_p = df.rename(columns={'Date': 'ds', 'Amount': 'y'})
        
        m = Prophet(interval_width=0.95)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=60)
        forecast = m.predict(future)

        # Ø±Ø³Ù… Ù†Ù…ÙˆØ¯Ø§Ø± Ù¾ÛŒØ´â€ŒØ¨ÛŒÙ†ÛŒ
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='#00CC96')))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,204,150,0.1)', name='Confidence Upper'))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,204,150,0.1)', name='Confidence Lower'))
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        

    with tab2:
        st.subheader("Spending Pattern Distribution")
        fig_hist = px.histogram(df, x="Amount", nbins=20, marginal="box", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Global Asset Correlation Matrix")
        # ÛŒÚ© Ù…Ø§ØªØ±ÛŒØ³ Ù‡Ù…Ø¨Ø³ØªÚ¯ÛŒ ÙØ±Ø¶ÛŒ Ø¨Ø±Ø§ÛŒ Ù†Ù…Ø§ÛŒØ´ ØªØ®ØµØµ Ù…Ø§Ù„ÛŒ
        corr_data = np.random.rand(4,4)
        labels = ["Spending", "Gold", "S&P 500", "BTC"]
        fig_corr = px.imshow(corr_data, x=labels, y=labels, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
        

    # Footer for Thesis
    st.sidebar.divider()
    st.sidebar.caption("Built with Python, Streamlit & Meta Prophet Model. Target: Predictive Personal Finance Optimization.")

if __name__ == "__main__":
    main()
# --- Ø¨Ø®Ø´ Ø¬Ø¯ÛŒØ¯: ØªØ­Ù„ÛŒÙ„ Ø³Ù‡Ø§Ù… (Stock Intelligence) ---
with st.expander("ðŸ” Global Stock Intelligence"):
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "AAPL").upper()
    
    if ticker_input:
        stock = yf.Ticker(ticker_input)
        
        # Ø¯Ø±ÛŒØ§ÙØª Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ú©Ù„ÛŒ Ø³Ù‡Ø§Ù…
        info = stock.info
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            col1.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            col2.metric("Market Cap", f"{info.get('marketCap', 0):,}")
            col3.metric("P/E Ratio", info.get('forwardPE', 'N/A'))
            col4.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            
            # Ø±Ø³Ù… Ù†Ù…ÙˆØ¯Ø§Ø± Ù‚ÛŒÙ…Øª Ø³Ù‡Ø§Ù…
            hist = stock.history(period="1y")
            fig_stock = px.area(hist, x=hist.index, y='Close', 
                                title=f"{ticker_input} Price Action (Past Year)",
                                color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_stock, use_container_width=True)

            # ØªØ­Ù„ÛŒÙ„ ÙØ§Ù†Ø¯Ø§Ù…Ù†ØªØ§Ù„ (ØªØ­Ù„ÛŒÙ„ Ø³ÙˆØ¯Ø¢ÙˆØ±ÛŒ)
            st.subheader(f"Financial Health: {ticker_input}")
            fundamentals = stock.financials.loc['Net Income'] if 'Net Income' in stock.financials.index else None
            if fundamentals is not None:
                fig_fin = px.bar(fundamentals, title="Annual Net Income Trend")
                st.plotly_chart(fig_fin, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not fetch data for {ticker_input}. Please check the ticker symbol.")
            import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np

# --- 1. Global Configuration ---
st.set_page_config(page_title="AI Finance & Portfolio Intelligence", layout="wide")

# --- 2. Market Data Function ---
@st.cache_data(ttl=3600)
def get_global_metrics():
    tickers = {"Gold": "GC=F", "S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X"}
    data = {}
    for name, tike in tickers.items():
        try:
            df = yf.Ticker(tike).history(period="2d")
            price = df['Close'].iloc[-1]
            change = ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            data[name] = (price, change)
        except: data[name] = (0, 0)
    return data

# --- 3. Expense Categorization Logic (Ø§Ø² Ù„ÛŒÙ†Ú© Ø§Ø±Ø³Ø§Ù„ÛŒ) ---
def categorize_expenses(description):
    description = description.lower()
    if any(word in description for word in ['amazon', 'shop', 'mall', 'buy']):
        return 'Shopping'
    elif any(word in description for word in ['uber', 'gas', 'snapp', 'train', 'flight']):
        return 'Transport'
    elif any(word in description for word in ['restaurant', 'food', 'cafe', 'pizza']):
        return 'Dining'
    elif any(word in description for word in ['rent', 'bill', 'electric', 'water']):
        return 'Bills & Housing'
    else:
        return 'Others'

# --- 4. Main App ---
def main():
    st.title("ðŸ¤– AI Integrated Financial Ecosystem")
    st.markdown("---")

    # Metrics Row
    metrics = get_global_metrics()
    cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        cols[i].metric(name, f"{val[0]:,.2f}", f"{val[1]:.2f}%")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Personal Finance AI", "Global Stock Analyzer", "Wealth Forecasting"])

    # --- PAGE 1: Personal Finance (NLP & Categorization) ---
    if page == "Personal Finance AI":
        st.header("ðŸ’³ Personal Expense Intelligence")
        uploaded_file = st.file_uploader("Upload CSV Statement", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'Description' in df.columns and 'Amount' in df.columns:
                df['Category'] = df['Description'].apply(categorize_expenses)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Spending by Category")
                    fig_pie = px.pie(df, values='Amount', names='Category', hole=0.4)
                    st.plotly_chart(fig_pie)
                
                with col2:
                    st.subheader("AI Analysis")
                    total_spent = df['Amount'].sum()
                    st.write(f"**Total Expenses:** ${total_spent:,.2f}")
                    top_cat = df.groupby('Category')['Amount'].sum().idxmax()
                    st.warning(f"âš ï¸ Your highest spending is in **{top_cat}**. Consider optimizing this area.")

    # --- PAGE 2: Stock Analyzer ---
    elif page == "Global Stock Analyzer":
        st.header("ðŸ” Real-time Equity Analysis")
        ticker = st.text_input("Enter Ticker (e.g. NVDA, AAPL):", "NVDA").upper()
        if st.button("Analyze Stock"):
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if not hist.empty:
                fig = px.line(hist, y='Close', title=f"{ticker} Performance")
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility Check
                returns = hist['Close'].pct_change()
                vol = (returns.std() * np.sqrt(252)) * 100
                st.info(f"Annualized Volatility: {vol:.2f}%")

    # --- PAGE 3: Forecasting ---
    elif page == "Wealth Forecasting":
        st.header("ðŸ”® AI Wealth Projection")
        st.write("Projecting your future net worth based on current trends...")
        # Ø´Ø¨ÛŒÙ‡â€ŒØ³Ø§Ø²ÛŒ Ø¯Ø§Ø¯Ù‡ Ø¨Ø±Ø§ÛŒ Ù†Ù…Ø§ÛŒØ´ Ù‚Ø§Ø¨Ù„ÛŒØª Ù¾ÛŒØ´â€ŒØ¨ÛŒÙ†ÛŒ
        df_demo = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=100),
            'y': np.random.normal(100, 10, 100).cumsum()
        })
        m = Prophet()
        m.fit(df_demo)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        fig_f = px.line(forecast, x='ds', y='yhat', title="Next 30 Days Forecast")
        st.plotly_chart(fig_f)

    # Global Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Global Finance AI - v2.0 | Scientific Research Project")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from scipy.stats import norm

# --- CONFIGURATION ---
st.set_page_config(page_title="QuantFinance AI | Research Platform", layout="wide")

# --- ADVANCED QUANT FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_advanced_metrics(tickers):
    data = yf.download(tickers, period="1y")['Close']
    returns = data.pct_change().dropna()
    
    # Correlation Matrix
    corr_matrix = returns.corr()
    
    # Sharpe Ratio (Assuming Risk-Free Rate = 0.02)
    rf = 0.02 / 252
    excess_returns = returns - rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    return returns, corr_matrix, sharpe

# --- MAIN APP ---
def main():
    st.title("ðŸ›ï¸ Quantitative Financial Intelligence Platform")
    st.markdown("---")

    st.sidebar.title("ðŸ” Analytical Engines")
    engine = st.sidebar.selectbox("Select Methodology:", 
                                 ["Global Market Pulse", "Portfolio Risk Analysis", "AI Wealth Prediction", "Smart Categorizer"])

    # 1. GLOBAL MARKET PULSE
    if engine == "Global Market Pulse":
        st.header("ðŸŒ Macro-Economic Indicators")
        tickers = ["^GSPC", "GC=F", "BTC-USD", "EURUSD=X"]
        returns, corr, sharpe = get_advanced_metrics(tickers)
        
        # Heatmap of Correlations
        st.subheader("Asset Correlation Matrix")
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
        

        # Sharpe Ratio Comparison
        st.subheader("Risk-Adjusted Performance (Sharpe Ratio)")
        st.bar_chart(sharpe)
        st.info("The Sharpe Ratio shows how much excess return you are receiving for the extra volatility endured.")

    # 2. PORTFOLIO RISK ANALYSIS
    elif engine == "Portfolio Risk Analysis":
        st.header("ðŸ“‰ Risk Management & Volatility")
        symbol = st.text_input("Enter Asset for Risk Profile:", "NVDA").upper()
        if symbol:
            asset_data = yf.download(symbol, period="1y")['Close']
            asset_ret = asset_data.pct_change().dropna()
            
            # Histogram of Returns (Normal Distribution Overlap)
            st.subheader("Returns Distribution & Kurtosis")
            fig_dist = px.histogram(asset_ret, nbins=50, marginal="box", title=f"Distribution of {symbol} Daily Returns")
            st.plotly_chart(fig_dist, use_container_width=True)
            

    # 3. AI WEALTH PREDICTION
    elif engine == "AI Wealth Prediction":
        st.header("ðŸ”® Time-Series Forecasting (Prophet)")
        # Simulation for Demo
        df_p = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=200),
            'y': np.random.normal(105, 12, 200).cumsum()
        })
        m = Prophet(daily_seasonality=True)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Expected Value'))
        fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,176,246,0.2)', name='Upper Bound'))
        fig_fore.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.2)', name='Lower Bound'))
        st.plotly_chart(fig_fore, use_container_width=True)
        

    # 4. SMART CATEGORIZER
    elif engine == "Smart Categorizer":
        st.header("ðŸ’³ Intelligent Expense Classification")
        st.write("Upload your bank statement to apply NLP-based classification.")
        # ... (Ú©Ø¯ Ø¯Ø³ØªÙ‡â€ŒØ¨Ù†Ø¯ÛŒ Ú©Ù‡ Ù‚Ø¨Ù„Ø§Ù‹ Ø¯Ø§Ø´ØªÛŒÙ…)

    # FOOTER
    st.sidebar.divider()
    st.sidebar.caption("Research Methodology: Quant Finance & ML (MSc Level)")

if __name__ == "__main__":
    main()
import streamlit as st
import yfinance as yf
import pandas as pd

# --- Û±. Ø§Ø³ØªÙØ§Ø¯Ù‡ Ø§Ø² Ú©Ø´ Ø¨Ø±Ø§ÛŒ Ø¬Ù„ÙˆÚ¯ÛŒØ±ÛŒ Ø§Ø² Ø¨Ù„Ø§Ú© Ø´Ø¯Ù† ---
@st.cache_data(ttl=3600)  # Ø¯Ø§Ø¯Ù‡â€ŒÙ‡Ø§ Ø±Ø§ Û± Ø³Ø§Ø¹Øª Ø¯Ø± Ø­Ø§ÙØ¸Ù‡ Ù†Ú¯Ù‡ Ù…ÛŒâ€ŒØ¯Ø§Ø±Ø¯
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Ø§Ø³ØªÙØ§Ø¯Ù‡ Ø§Ø² history Ø¨Ù‡ Ø¬Ø§ÛŒ info Ø¨Ø±Ø§ÛŒ Ù¾Ø§ÛŒØ¯Ø§Ø±ÛŒ Ø¨ÛŒØ´ØªØ±
        df = stock.history(period="1y")
        if df.empty:
            return None, None
        
        # Ø§Ø³ØªØ®Ø±Ø§Ø¬ Ø§Ø·Ù„Ø§Ø¹Ø§Øª Ù¾Ø§ÛŒÙ‡ Ú©Ù‡ Ù‚Ø¨Ù„Ø§Ù‹ Ø§Ø² info Ù…ÛŒâ€ŒÚ¯Ø±ÙØªÛŒÙ…
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        change = ((last_price - prev_price) / prev_price) * 100
        
        return df, {"price": last_price, "change": change}
    except Exception as e:
        return None, str(e)

# --- Û². Ù¾ÛŒØ§Ø¯Ù‡â€ŒØ³Ø§Ø²ÛŒ Ø¯Ø± Ø±Ø§Ø¨Ø· Ú©Ø§Ø±Ø¨Ø±ÛŒ ---
st.subheader("ðŸ” Advanced Asset Intelligence")
ticker_input = st.text_input("Enter Ticker (e.g. AAPL, NVDA):", "AAPL").upper()

if st.button("Analyze Asset"):
    with st.spinner("Fetching Data..."):
        hist_data, stats = fetch_stock_data(ticker_input)
        
        if hist_data is not None and isinstance(stats, dict):
            col1, col2 = st.columns(2)
            col1.metric(f"{ticker_input} Price", f"${stats['price']:.2f}", f"{stats['change']:.2f}%")
            
            # ØªØ­Ù„ÛŒÙ„ Ø±ÛŒØ³Ú© (Ø¨Ø³ÛŒØ§Ø± Ù…Ù‡Ù… Ø¨Ø±Ø§ÛŒ Ø±Ø²ÙˆÙ…Ù‡ Finance)
            daily_returns = hist_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * (252**0.5)
            col2.metric("Annual Volatility", f"{volatility:.2%}")
            
            st.plotly_chart(px.line(hist_data, y='Close', title=f"{ticker_input} Historical Trend"))
        else:
            st.error("âš ï¸ Rate limit hit or invalid ticker. Please wait or try a different symbol.")
# Ø§Ø¶Ø§ÙÙ‡ Ú©Ø±Ø¯Ù† ØªØ­Ù„ÛŒÙ„ ØªÙˆØ²ÛŒØ¹ Ø¨Ø§Ø²Ø¯Ù‡ÛŒ
st.subheader("Distribution of Returns (Risk Profile)")
fig_dist = px.histogram(daily_returns, nbins=50, marginal="box", 
                         title="Daily Returns Frequency",
                         color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig_dist, use_container_width=True)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.ex
