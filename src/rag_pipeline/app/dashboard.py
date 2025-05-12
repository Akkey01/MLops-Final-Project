# dashboard.py
import streamlit as st, pandas as pd, json, os
from datetime import datetime

LOG_PATH = "/mnt/block/stream_logs.jsonl"

@st.cache(ttl=60)
def load_logs():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                rec["ts"] = datetime.fromisoformat(rec["ts"])
                rows.append(rec)
            except:
                pass
    df = pd.DataFrame(rows)
    return df

st.set_page_config(page_title="📊 Streaming Dashboard", layout="wide")
st.title("📊 Streaming Dashboard")

df = load_logs()
if df.empty:
    st.info("No streaming logs yet.")
else:
    st.metric("Total requests", len(df))
    df = df.set_index("ts")
    st.subheader("Requests over time")
    st.line_chart(df.resample("1T").size())

    st.subheader("Response lengths (last 20)")
    df["len_friend"]   = df["friend_model"].str.len()
    df["len_together"] = df["together"].str.len()
    st.bar_chart(df[["len_friend","len_together"]].tail(20))
