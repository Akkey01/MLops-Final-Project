#!/usr/bin/env python3
import os
import sys

import streamlit as st
import requests

# ── Make project root importable ─────────────────────────────────────
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from dashboard.streamlit_extra import show_dashboard

# ── Page config & ENV ───────────────────────────────────────────────
st.set_page_config(page_title="IMPS.AI", layout="wide")
RAG_URL = os.getenv("RAG_ENDPOINT_URL", "http://127.0.0.1:8000/predict")

# ── Sidebar navigation ──────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Dashboard"])

# ── Chat page ───────────────────────────────────────────────────────
if page == "Chat":
    st.title("💬 RAG Chat")
    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Your question:")
    if st.button("Submit") and query:
        try:
            resp = requests.post(RAG_URL, json={"query": query})
            resp.raise_for_status()
            answer = resp.json()
        except Exception as e:
            answer = {"error": str(e)}

        st.session_state.history.append((query, answer))

    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**RAG:** {a}")

# ── Dashboard page ─────────────────────────────────────────────────
else:
    show_dashboard()
