import streamlit as st
import pandas as pd
import time
from sqlalchemy import create_engine

# --- CONFIGURATION ---
DB_PATH = "sqlite:///warehouse.db"
engine = create_engine(DB_PATH)

# Configure the web page
st.set_page_config(page_title="BE PRESS MLOps Dashboard", layout="wide")
st.title("🏭 BE PRESS: Live Telemetry & AI Anomaly Detection")
st.markdown("Real-time industrial sensor stream with Isolation Forest inference.")

# Create permanent UI placeholders that we will update in a loop
metric_placeholder = st.empty()
chart_placeholder = st.empty()
log_placeholder = st.empty()

def fetch_latest_data():
    """Pulls the last 60 seconds of telemetry from the SQL warehouse."""
    query = "SELECT * FROM telemetry_stream ORDER BY id DESC LIMIT 60"
    try:
        df = pd.read_sql(query, engine)
        # Reverse the dataframe so chronological order goes left-to-right on the chart
        return df.iloc[::-1] 
    except Exception:
        return pd.DataFrame()

# --- REAL-TIME DASHBOARD LOOP ---
while True:
    df = fetch_latest_data()
    
    if not df.empty:
        latest = df.iloc[-1]
        anomalies = df[df['is_anomaly'] == True]
        
        # 1. Update the Top Metric Cards
        with metric_placeholder.container():
            c1, c2, c3 = st.columns(3)
            status = "🔴 AI INTERCEPT: CRITICAL" if latest['is_anomaly'] else "🟢 NORMAL OPERATION"
            
            c1.metric("System Status", status)
            c2.metric("Roller Temp (°C)", f"{latest['roller_temp_c']:.1f}")
            c3.metric("Vibration (Hz)", f"{latest['vibration_hz']:.1f}")
        
        # 2. Update the Live Time-Series Chart
        with chart_placeholder.container():
            # Standardize index for charting
            chart_data = df.set_index("timestamp")[["roller_temp_c", "vibration_hz"]]
            st.line_chart(chart_data, color=["#3498db", "#e74c3c"])
            
        # 3. Update the Anomaly Database Log
        with log_placeholder.container():
            st.subheader("🛑 Recent AI Intercepts")
            if not anomalies.empty:
                st.dataframe(
                    anomalies[['timestamp', 'roller_temp_c', 'vibration_hz']].tail(5), 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No anomalies detected in the current window.")

    # Pause for 1 second, then force Streamlit to refresh the UI
    time.sleep(1)
    st.rerun()