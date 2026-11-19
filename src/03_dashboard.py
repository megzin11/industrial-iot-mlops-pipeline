import time
import pandas as pd
import streamlit as st

from src.models import DEFAULT_DB_URL, create_warehouse_engine

st.set_page_config(page_title="Industrial IoT MLOps Monitor", layout="wide")
st.title("Industrial IoT Streaming & Explainable Anomaly Monitor")
st.caption("Live MQTT telemetry scored by EmpiricalCovariance Mahalanobis detector with Z-score attribution (SQLite WAL).")
engine = create_warehouse_engine(DEFAULT_DB_URL)

with st.sidebar:
    window_size = st.slider("Rolling Window (frames)", min_value=30, max_value=300, value=90, step=30)
    auto_refresh = st.checkbox("Auto-refresh (2s)", value=True)


def fetch_latest(limit: int) -> pd.DataFrame:
    query = f"SELECT * FROM telemetry_stream ORDER BY id DESC LIMIT {int(limit)}"
    try:
        df = pd.read_sql(query, engine)
        return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


df = fetch_latest(window_size)

if df.empty:
    st.info("Waiting for telemetry frames in SQLite warehouse. Start `python -m src.02_mlops_consumer` and `python -m src.01_edge_sensor`.")
else:
    latest = df.iloc[-1]
    anomalies = df[df["is_anomaly"] == True]  # noqa: E712

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("System Status", "ALERT" if latest["is_anomaly"] else "NORMAL", latest["primary_driver"])
    c2.metric("Roller Temp (C)", f"{latest['roller_temp_c']:.2f}")
    c3.metric("Vibration (Hz)", f"{latest['vibration_hz']:.2f}")
    c4.metric("Motor Current (A)", f"{latest['motor_current_a']:.2f}")
    c5.metric("Avg Inference", f"{df['inference_ms'].mean():.2f} ms")

    st.subheader("Live Multi-Sensor Telemetry Stream")
    chart_df = df.set_index("timestamp")[["roller_temp_c", "vibration_hz", "motor_current_a"]]
    st.line_chart(chart_df)

    st.subheader(f"Auditable Anomaly Log ({len(anomalies)} alerts in last {len(df)} frames)")
    if not anomalies.empty:
        cols = [
            "timestamp",
            "machine_id",
            "roller_temp_c",
            "vibration_hz",
            "motor_current_a",
            "anomaly_score",
            "primary_driver",
            "model_version",
        ]
        st.dataframe(anomalies[cols].tail(10), use_container_width=True, hide_index=True)
    else:
        st.success("Zero anomalies detected in the active window.")

if auto_refresh:
    time.sleep(2)
    st.rerun()