# app.py
import os, pickle, faiss, requests, streamlit as st
from sentence_transformers import SentenceTransformer

# ─── Config ─────────────────────────────────────────────────────────────────────
TOGETHER_KEY    = os.getenv("TOGETHER_API_KEY")
MODEL_ENDPOINT  = os.getenv("MODEL_ENDPOINT")
DATA_ROOT       = "/mnt/block/faiss_base"

# ─── Load index, chunks, model ─────────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def load_resources():
    idx   = faiss.read_index(os.path.join(DATA_ROOT, "index.bin"))
    chunks= pickle.load(open(os.path.join(DATA_ROOT, "chunks.pkl"), "rb"))
    mod   = SentenceTransformer("all-MiniLM-L6-v2")
    return idx, chunks, mod

idx, chunks, model = load_resources()

# ─── UI setup ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="💬 RAG Chat", layout="wide")
st.title("💬 RAG Chat")

k         = st.sidebar.slider("Context chunks (k)", 1, 10, 5)
show_src  = st.sidebar.checkbox("Show sources", True)

if "history" not in st.session_state:
    st.session_state.history = []

def ask_rag(q):
    qv = model.encode([q]).astype("float32")
    D, I = idx.search(qv, k)
    ctx = [ chunks[i]["chunk"] for i in I[0] ]
    src = [ chunks[i]["fn"]    for i in I[0] ]
    prompt = "Context:\n" + "\n\n".join(ctx) + f"\n\nQuestion: {q}\nAnswer:"

    # Friend’s model
    try:
        r1 = requests.post(MODEL_ENDPOINT, json={"prompt": prompt}).json()
        a1 = r1.get("text", "")
    except:
        a1 = "⚠️ Error contacting model."

    # Together.ai
    try:
        r2 = requests.post(
            "https://api.together.xyz/v1/completions",
            headers={"Authorization": f"Bearer {TOGETHER_KEY}"},
            json={"model":"mistralai/Mixtral-8x7B-Instruct-v0.1", "prompt": prompt}
        ).json()
        a2 = r2.get("choices",[{}])[0].get("text","")
    except:
        a2 = "⚠️ Error contacting Together API."

    return a1, a2, src

query = st.text_input("Ask a question", "")
if query:
    ans1, ans2, sources = ask_rag(query)
    st.session_state.history.append((query, ans1, ans2, sources))
    st.experimental_rerun()

for q, a1, a2, src in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Friend Model:** {a1}")
    st.markdown(f"**Together AI:** {a2}")
    if show_src:
        st.markdown("Sources: " + ", ".join(src))
    st.markdown("---")
