import os
import streamlit as st
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast
import nbformat

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Hard‑coded Together.ai API key
TOG_KEY = "cc4b628095c0531f06fe08ff20e1f0bad8cf4e6c39ed2b3c70744a6278a7faab"
# Default data directory env fallback
DATA_ROOT = os.getenv("DATA_DIR", "ami_public_manual_1.6.2")

# ── UI SETUP ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IMPS.AI", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
      body { background-color: #f7f9fc; }
      .stApp { color: #333; font-family: 'Segoe UI', Tahoma, sans-serif; }
      .title { font-size: 3rem; color: #1f77b4; margin-bottom: 0.2em; }
      .sidebar .stButton>button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 class='title'>📚 Hello, Lets Decode the Meeting", unsafe_allow_html=True)

# ── FILE READERS ────────────────────────────────────────────────────────────────
def extract_text_from_pdf(p): return "\n".join(page.extract_text() or "" for page in pdfplumber.open(p).pages)
def extract_text_from_docx(p): return "\n".join(para.text for para in Document(p).paragraphs)
def extract_text_from_html(p):
    soup = BeautifulSoup(open(p, encoding="utf-8"), "html.parser")
    return html2text.html2text(soup.prettify())
def extract_text_from_txt(p): return open(p, encoding="utf-8").read()
def extract_text_from_xml(p):
    tree = ET.parse(p)
    return " ".join(e.text.strip() for e in tree.getroot().iter() if e.text)
def extract_text_from_ipynb(p):
    nb = nbformat.read(p, as_version=4)
    parts = []
    for cell in nb.cells:
        if cell.cell_type in ("markdown", "code"):
            parts.append(cell.source)
    return "\n".join(parts)

def read_any_file(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".pdf": return extract_text_from_pdf(fp)
    if ext == ".docx": return extract_text_from_docx(fp)
    if ext == ".html": return extract_text_from_html(fp)
    if ext in (".txt", ".md", ".css"): return extract_text_from_txt(fp)
    if ext == ".xml": return extract_text_from_xml(fp)
    if ext == ".ipynb": return extract_text_from_ipynb(fp)
    return ""

# ── CHUNKING ────────────────────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500):
    words, chunks, cur, ct = text.split(), [], [], 0
    for w in words:
        l = len(tokenizer.encode(w, add_special_tokens=False))
        if ct + l > max_tokens:
            chunks.append(" ".join(cur)); cur, ct = [w], l
        else:
            cur.append(w); ct += l
    if cur: chunks.append(" ".join(cur))
    return chunks

# ── BUILD INDEX ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_index(root):
    docs = []
    for dp, _, files in os.walk(root):
        for f in files:
            fp = os.path.join(dp, f)
            txt = read_any_file(fp)
            if txt:
                docs.append({"fn": os.path.relpath(fp, root), "txt": txt})
    # include uploads
    upl = "uploads"
    if os.path.isdir(upl):
        for f in os.listdir(upl):
            fp = os.path.join(upl, f)
            txt = read_any_file(fp)
            if txt:
                docs.append({"fn": f, "txt": txt})
    if not docs:
        return None, None, None, "No documents found."
    chunked = []
    for d in docs:
        for c in chunk_text(d["txt"]):
            chunked.append({"fn": d["fn"], "chunk": c})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([c["chunk"] for c in chunked]).astype('float32')
    idx = faiss.IndexFlatL2(embs.shape[1]); idx.add(embs)
    return idx, chunked, model, None

# ── GENERATION ─────────────────────────────────────────────────────────────────
def generate_answer(prompt):
    r = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOG_KEY}", "Content-Type": "application/json"},
        json={"model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "prompt": prompt, "max_tokens": 512, "temperature": 0.3, "top_p": 0.9}
    ).json()
    if "choices" in r:
        return r["choices"][0].get("text", "")
    if r.get("output") and "choices" in r["output"]:
        return r["output"]["choices"][0].get("text", "")
    return ""

def ask_rag(q, idx, chunked, mod, k=5):
    qv = mod.encode([q]).astype('float32')
    D, I = idx.search(qv, k)
    ctx = "\n\n".join(chunked[i]["chunk"] for i in I[0])
    return generate_answer(f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

# ── SESSION & UPLOAD ───────────────────────────────────────────────────────────
if "idx" not in st.session_state:
    st.session_state.idx, st.session_state.chunked, st.session_state.mod, st.session_state.err = build_index(DATA_ROOT)
if st.session_state.err:
    st.error(st.session_state.err)
st.sidebar.header("Upload documents")
ufs = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)
if ufs:
    os.makedirs("uploads", exist_ok=True)
    for f in ufs:
        with open(os.path.join("uploads", f.name), "wb") as out: out.write(f.getbuffer())
    st.sidebar.success(f"Uploaded {len(ufs)} files.")
    st.session_state.idx, st.session_state.chunked, st.session_state.mod, st.session_state.err = build_index(DATA_ROOT)

# ── QUERY & HISTORY ─────────────────────────────────────────────────────────────
st.write("---")
with st.form("query_form", clear_on_submit=True):
    question = st.text_input("💬 Ask a question", placeholder="Type and press Enter")
    submitted = st.form_submit_button("Ask")
if submitted and question:
    if st.session_state.err:
        st.error("Cannot query: no index.")
    else:
        ans = ask_rag(question, st.session_state.idx, st.session_state.chunked, st.session_state.mod)
        st.session_state.history = st.session_state.get("history", []) + [(question, ans)]
if st.session_state.get("history"):
    st.write("### Conversation History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")