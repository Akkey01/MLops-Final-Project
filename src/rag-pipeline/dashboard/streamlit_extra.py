import streamlit as st
import glob, json, pathlib
import pandas as pd

def show_dashboard():
    st.header("📊 Live Streaming Metrics")
    metric_dir = "/mnt/block/metrics"
    files = sorted(pathlib.Path(metric_dir).glob("*.jsonl"))
    records = []
    for mf in files:
        for line in mf.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except:
                continue

    if not records:
        st.info("No metrics yet. Run the simulator!")
        return

    df = pd.DataFrame(records)
    # Convert epoch → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")

    # Latency over time
    st.subheader("Latency (sec)")
    st.line_chart(df["latency"])

    # Success rate
    success_rate = (df["status"] == 200).mean() * 100
    st.metric("Success rate", f"{success_rate:.1f}%")
