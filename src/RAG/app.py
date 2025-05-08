# app.py

import os
import streamlit as st
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast
import nbformat
import nltk

# ── Ensure NLTK model ───────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)

# ── CONFIG ──────────────────────────────────────────────────────────────────────
TOG_KEY = "cc4b628095c0531f06fe08ff20e1f0bad8cf4e6c39ed2b3c70744a6278a7faab"
DATA_ROOT = os.getenv("DATA_DIR", "ami_data")  # your base corpus path

# ── PAGE SETUP & CSS ──────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat+", layout="wide")
st.markdown("""
    <style>
      body { background-color: #f5f7fa; }
      .block-container { padding: 1.5rem; }
      .title { font-size: 2.2rem; font-weight: 600; color: #2c3e50; margin-bottom: 1rem; }
      .user-query, .bot-response { 
        padding: 0.75rem; margin: 0.5rem 0; border-radius: 8px; color: #000 !important;
      }
      .user-query { background-color: #e8f0fe !important; }
      .bot-response { background-color: #fff8dc !important; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='title'>💬 RAG Chat — Ask Anything from Your Docs</div>", unsafe_allow_html=True)

# ── FILE READERS ────────────────────────────────────────────────────────────────
def extract_text_from_pdf(p): 
    with pdfplumber.open(p) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
def extract_text_from_docx(p): 
    return "\n".join(para.text for para in Document(p).paragraphs)
def extract_text_from_html(p):
    soup = BeautifulSoup(open(p, encoding="utf-8"), "html.parser")
    return html2text.html2text(soup.prettify())
def extract_text_from_txt(p): 
    return open(p, encoding="utf-8").read()
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
    if ext == ".pdf":   return extract_text_from_pdf(fp)
    if ext == ".docx":  return extract_text_from_docx(fp)
    if ext in (".txt", ".md", ".css"): return extract_text_from_txt(fp)
    if ext == ".html":  return extract_text_from_html(fp)
    if ext == ".xml":   return extract_text_from_xml(fp)
    if ext == ".ipynb": return extract_text_from_ipynb(fp)
    raise ValueError(f"Unsupported file type: {ext}")

# ── SMART CHUNKING ─────────────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500):
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    chunks, cur, ct = [], [], 0
    for s in sents:
        tl = len(tokenizer.encode(s, add_special_tokens=False))
        if ct + tl > max_tokens:
            chunks.append(" ".join(cur))
            cur, ct = [s], tl
        else:
            cur.append(s)
            ct += tl
    if cur: chunks.append(" ".join(cur))
    return chunks

# ── BUILD BASE INDEX ───────────────────────────────────────────────────────────
@st.cache_data
def build_base_index(root=DATA_ROOT):
    docs = []
    for dp, _, files in os.walk(root):
        for f in files:
            fp = os.path.join(dp, f)
            try: docs.append({"fn": f, "txt": read_any_file(fp)})
            except: pass
    if not docs:
        return None, None, None, "No base documents found."
    # chunk + embed
    base_chunks = []
    for d in docs:
        for c in chunk_text(d["txt"]):
            base_chunks.append({"fn": d["fn"], "chunk": c})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([c["chunk"] for c in base_chunks]).astype("float32")
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    return idx, base_chunks, model, None

# ── INIT STATE ─────────────────────────────────────────────────────────────────
if "base_idx" not in st.session_state:
    st.session_state.base_idx, st.session_state.base_chunks, st.session_state.model, st.session_state.err = build_base_index()

# create empty upload index
if "upload_idx" not in st.session_state and not st.session_state.err:
    dim = st.session_state.base_idx.d  # dimension
    st.session_state.upload_idx = faiss.IndexFlatL2(dim)
    st.session_state.upload_chunks = []

# error on base
if st.session_state.err:
    st.error(st.session_state.err)

# ── UPLOAD & INDEX ─────────────────────────────────────────────────────────────
st.sidebar.header("📤 Upload & Index Files")
ufs = st.sidebar.file_uploader(
    "Supported: pdf, docx, txt, md, html, xml, ipynb", 
    accept_multiple_files=True,
    type=["pdf","docx","txt","md","html","xml","ipynb"]
)
if ufs:
    os.makedirs("uploads", exist_ok=True)
    new_count = 0
    for f in ufs:
        dst = os.path.join("uploads", f.name)
        with open(dst, "wb") as out: out.write(f.getbuffer())
        try:
            txt = read_any_file(dst)
            new_chunks = [{"fn": f.name, "chunk": c} for c in chunk_text(txt)]
            embs = st.session_state.model.encode([c["chunk"] for c in new_chunks]).astype("float32")
            st.session_state.upload_idx.add(embs)
            st.session_state.upload_chunks.extend(new_chunks)
            new_count += len(new_chunks)
        except Exception as e:
            st.sidebar.error(f"Error indexing {f.name}: {e}")
    st.sidebar.success(f"Indexed {new_count} new chunks from uploads.")

# ── GENERATOR ──────────────────────────────────────────────────────────────────
def generate_answer(prompt):
    r = requests.post("https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOG_KEY}", "Content-Type":"application/json"},
        json={"model":"mistralai/Mixtral-8x7B-Instruct-v0.1","prompt":prompt,
              "max_tokens":512,"temperature":0.3,"top_p":0.9}
    ).json()
    if "choices" in r:
        return r["choices"][0].get("text","")
    if r.get("output") and "choices" in r["output"]:
        return r["output"]["choices"][0].get("text","")
    return "⚠️ Unexpected response."

def ask_rag(query, k=5):
    # greeting?
    if query.strip().lower() in ("hi","hello","hey"):
        return "Hello! How can I help you today?"

    base_idx = st.session_state.base_idx
    upload_idx = st.session_state.upload_idx
    base_chunks = st.session_state.base_chunks
    upload_chunks = st.session_state.upload_chunks
    model = st.session_state.model

    # search uploads first
    ctx_list = []
    if upload_idx.ntotal > 0:
        _, Iu = upload_idx.search(model.encode([query]).astype("float32"), min(k, upload_idx.ntotal))
        for i in Iu[0]:
            ctx_list.append(upload_chunks[i]["chunk"])
    # then fill from base
    remaining = k - len(ctx_list)
    if remaining > 0:
        _, Ib = base_idx.search(model.encode([query]).astype("float32"), remaining)
        for i in Ib[0]:
            ctx_list.append(base_chunks[i]["chunk"])

    context = "\n\n".join(ctx_list)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_answer(prompt)

# ── CHAT HISTORY & INPUT ──────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# display history
for q, a in st.session_state.history:
    st.markdown(f"<div class='user-query'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-response'><b>RAG:</b> {a}</div>", unsafe_allow_html=True)

# input form
with st.form("chat", clear_on_submit=True):
    query = st.text_input("Ask a question", placeholder="Type and press Enter")
    submitted = st.form_submit_button("Ask")
    if submitted and query:
        answer = ask_rag(query)
        st.session_state.history.append((query, answer))
