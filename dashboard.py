import streamlit as st
import pandas as pd
import time
from stock_anomaly_detector import MarketDataFetcher, SimpleStatAnomaly, VolumeSpikeDetector, SequencePatternDetector, DataSanitizer
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go

st.set_page_config(page_title="Stock Anomaly Dashboard", layout="wide")
st.title("📈 Real-Time Stock Anomaly Detection Dashboard")

symbol = st.sidebar.selectbox("Select Stock Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"])
window = st.sidebar.slider("Moving Average Window", 5, 60, 20)
multiplier = st.sidebar.slider("Std Multiplier", 1.0, 4.0, 2.0)
refresh_interval = st.sidebar.slider("Auto-refresh interval (seconds)", 10, 300, 60)

# Add option for multiple anomaly detection methods
anomaly_methods = st.sidebar.multiselect(
    "Anomaly Detection Methods",
    ["Statistical (Moving Avg)", "LSTM Pattern", "Volume Spike"],
    default=["Statistical (Moving Avg)"]
)

# Real-time streaming: use Streamlit's autorefresh widget for true live updates
st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")

fetcher = MarketDataFetcher(delay=0.5)
data = fetcher.fetch_history(symbol, span="3mo")

if data.empty:
    st.error(f"No data found for {symbol}")
    st.stop()

clean_data = DataSanitizer.clean(data)

# Set current_price only if any anomaly method is selected
current_price = None
if anomaly_methods:
    current_price = clean_data['Close'].iloc[-1]

# Statistical anomaly detection
anomalies = []
if "Statistical (Moving Avg)" in anomaly_methods and current_price is not None:
    stat_detector = SimpleStatAnomaly(window=window, multiplier=multiplier)
    stat_detector.train(clean_data)
    anomalies = stat_detector.detect_anomalies(clean_data, current_price)

# Volume anomaly detection
volume_anomalies = []
if "Volume Spike" in anomaly_methods and current_price is not None:
    volume_detector = VolumeSpikeDetector(window=window, multiplier=3.0)
    volume_detector.train(clean_data)
    recent_volume = clean_data['Volume'].iloc[-1]
    volume_anomalies = volume_detector.detect_anomalies(clean_data, recent_volume)

# LSTM anomaly detection
lstm_anomalies = []
if "LSTM Pattern" in anomaly_methods and current_price is not None:
    if len(clean_data) >= 100:
        lstm_detector = SequencePatternDetector(seq_len=30, threshold_pct=95)
        lstm_detector.train(clean_data)
        lstm_anomalies = lstm_detector.detect_anomalies(clean_data, current_price)
    else:
        st.warning("Not enough data for LSTM anomaly detection (need at least 100 data points)")

if current_price is not None:
    st.subheader(f"{symbol} - Last Price: ${current_price:.2f}")
    st.line_chart(clean_data['Close'], use_container_width=True)

def explain_anomaly(anomaly):
    if hasattr(anomaly, 'anomaly_type'):
        if anomaly.anomaly_type == 'STATISTICAL':
            return f"Price deviated from the moving average by more than {multiplier} standard deviations. Z-score: {anomaly.details.get('z', 'N/A')}"
        elif anomaly.anomaly_type == 'LSTM_PATTERN':
            return f"LSTM predicted price: {anomaly.details.get('predicted', 'N/A')}, Actual: {anomaly.details.get('actual', 'N/A')}, Error: {anomaly.details.get('error', 'N/A')}"
        elif anomaly.anomaly_type == 'VOLUME_SPIKE':
            return f"Volume spiked above normal range. Z-score: {anomaly.details.get('z', 'N/A')}"
        else:
            return str(anomaly.details)
    return "No explanation available."

if "Statistical (Moving Avg)" in anomaly_methods:
    if anomalies:
        st.error(f"🚨 {len(anomalies)} statistical anomaly(s) detected!")
        for anomaly in anomalies:
            st.write(f"**Type:** {anomaly.anomaly_type} | **Severity:** {anomaly.severity} | **Confidence:** {anomaly.confidence:.2%}")
            st.write(f"Details: {anomaly.details}")
            with st.expander("Why is this an anomaly? (Explainable AI)"):
                st.info(explain_anomaly(anomaly))
    else:
        st.success("No statistical anomalies detected - price is within normal range.")

if volume_anomalies:
    st.error(f"🚨 {len(volume_anomalies)} volume anomaly(s) detected!")
    for anomaly in volume_anomalies:
        st.write(f"**Type:** {anomaly.anomaly_type} | **Severity:** {anomaly.severity} | **Confidence:** {anomaly.confidence:.2%}")
        st.write(f"Details: {anomaly.details}")
        with st.expander("Why is this an anomaly? (Explainable AI)"):
            st.info(explain_anomaly(anomaly))
elif "Volume Spike" in anomaly_methods:
    st.success("No volume anomalies detected.")

if lstm_anomalies:
    st.error(f"🚨 {len(lstm_anomalies)} LSTM anomaly(s) detected!")
    for anomaly in lstm_anomalies:
        st.write(f"**Type:** {anomaly.anomaly_type} | **Severity:** {anomaly.severity} | **Confidence:** {anomaly.confidence:.2%}")
        st.write(f"Details: {anomaly.details}")
        with st.expander("Why is this an anomaly? (Explainable AI)"):
            st.info(explain_anomaly(anomaly))
elif "LSTM Pattern" in anomaly_methods and len(clean_data) >= 100:
    st.success("No LSTM pattern anomalies detected.")

# Advanced visualization: Candlestick chart with anomaly markers
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=clean_data.index,
    open=clean_data['Open'],
    high=clean_data['High'],
    low=clean_data['Low'],
    close=clean_data['Close'],
    name='Candlestick',
    increasing_line_color='green', decreasing_line_color='red',
    showlegend=True
))

# Add moving average
ma_20 = clean_data['Close'].rolling(window=window).mean()
fig.add_trace(go.Scatter(
    x=clean_data.index,
    y=ma_20,
    mode='lines',
    name=f'{window}-day MA',
    line=dict(color='blue', width=2, dash='dash')
))

# Add anomaly markers
if anomalies:
    anomaly_indices = [a.index for a in anomalies if hasattr(a, 'index') and a.index in clean_data.index]
    anomaly_prices = [clean_data.loc[idx, 'Close'] for idx in anomaly_indices]
    fig.add_trace(go.Scatter(
        x=anomaly_indices,
        y=anomaly_prices,
        mode='markers',
        marker=dict(color='orange', size=12, symbol='x'),
        name='Statistical Anomaly'
    ))
if lstm_anomalies:
    lstm_indices = [a.index for a in lstm_anomalies if hasattr(a, 'index') and a.index in clean_data.index]
    lstm_prices = [clean_data.loc[idx, 'Close'] for idx in lstm_indices]
    fig.add_trace(go.Scatter(
        x=lstm_indices,
        y=lstm_prices,
        mode='markers',
        marker=dict(color='purple', size=12, symbol='star'),
        name='LSTM Anomaly'
    ))
if volume_anomalies:
    vol_indices = [a.index for a in volume_anomalies if hasattr(a, 'index') and a.index in clean_data.index]
    vol_prices = [clean_data.loc[idx, 'Close'] for idx in vol_indices]
    fig.add_trace(go.Scatter(
        x=vol_indices,
        y=vol_prices,
        mode='markers',
        marker=dict(color='red', size=12, symbol='triangle-up'),
        name='Volume Anomaly'
    ))

fig.update_layout(
    title=f"{symbol} Candlestick Chart with Anomaly Markers",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Show statistics
tab1, tab2 = st.tabs(["Statistics", "Raw Data"])
with tab1:
    if "Statistical (Moving Avg)" in anomaly_methods and 'stat_detector' in locals():
        st.metric("Moving Average", f"${stat_detector.mean:.2f}")
        st.metric("Std Deviation", f"${stat_detector.stdev:.2f}")
    st.metric("Volume Avg", f"{clean_data['Volume'].mean():,.0f}")
with tab2:
    st.dataframe(clean_data.tail(30), use_container_width=True)
