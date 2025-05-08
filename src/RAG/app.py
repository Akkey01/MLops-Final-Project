# app.py

import os, tempfile
import streamlit as st
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast
import nbformat

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Your Together.ai API key
TOG_KEY = "cc4b628095c0531f06fe08ff20e1f0bad8cf4e6c39ed2b3c70744a6278a7faab"

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
    return "\n".join(page.extract_text() or "" for page in pdfplumber.open(p).pages)
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
    # chunk by sentences then pack to ~max_tokens
    from nltk import sent_tokenize
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

# ── BUILD & CACHE INDEX ────────────────────────────────────────────────────────
@st.cache_data
def build_index(root="ami_data"):
    # load all files on disk + uploads
    docs = []
    for dp, _, files in os.walk(root):
        for f in files:
            fp = os.path.join(dp, f)
            try: docs.append({"fn": f, "txt": read_any_file(fp)})
            except: pass
    # initial error?
    if not docs: return None, None, None, "No documents found in DATA_DIR."
    # chunk & embed
    chunked = []
    for d in docs:
        for c in chunk_text(d["txt"]):
            chunked.append({"fn": d["fn"], "chunk": c})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([c["chunk"] for c in chunked]).astype("float32")
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    return idx, chunked, model, None

# ── GENERATION ─────────────────────────────────────────────────────────────────
def generate_answer(prompt):
    r = requests.post("https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOG_KEY}", "Content-Type": "application/json"},
        json={"model":"mistralai/Mixtral-8x7B-Instruct-v0.1","prompt":prompt,
              "max_tokens":512,"temperature":0.3,"top_p":0.9}
    ).json()
    if "choices" in r: return r["choices"][0].get("text","")
    if r.get("output") and "choices" in r["output"]:
        return r["output"]["choices"][0].get("text","")
    st.error(f"Unexpected response: {r}")
    return ""

def ask_rag(q, idx, chunked, model, k=5):
    qv = model.encode([q]).astype("float32")
    D, I = idx.search(qv, k)
    ctx = "\n\n".join(chunked[i]["chunk"] for i in I[0])
    return generate_answer(f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

# ── INITIAL INDEX BUILT ────────────────────────────────────────────────────────
if "idx" not in st.session_state:
    st.session_state.idx, st.session_state.chunked, st.session_state.model, st.session_state.err = build_index()

if st.session_state.err:
    st.error(st.session_state.err)

# ── UPLOAD & INCREMENTAL INDEXING ─────────────────────────────────────────────
st.sidebar.header("📤 Upload & Index Files")
ufs = st.sidebar.file_uploader(
    "Supported: pdf, docx, txt, md, html, xml, ipynb", 
    accept_multiple_files=True,
    type=["pdf","docx","txt","md","html","xml","ipynb"]
)
if ufs:
    os.makedirs("uploads", exist_ok=True)
    new_total = 0
    for f in ufs:
        dst = os.path.join("uploads", f.name)
        with open(dst, "wb") as out: out.write(f.getbuffer())
        try:
            txt = read_any_file(dst)
            new_chunks = [{"fn": f.name, "chunk": c} for c in chunk_text(txt)]
            embs = st.session_state.model.encode([c["chunk"] for c in new_chunks]).astype("float32")
            st.session_state.idx.add(embs)
            st.session_state.chunked.extend(new_chunks)
            new_total += len(new_chunks)
        except Exception as e:
            st.sidebar.error(f"Error indexing {f.name}: {e}")
    st.sidebar.success(f"Indexed {new_total} new chunks.")

# ── CHAT HISTORY UI ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# display history
for q,a in st.session_state.history:
    st.markdown(f"<div class='user-query'><b>You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-response'><b>RAG:</b> {a}</div>", unsafe_allow_html=True)

# form at bottom
with st.form("chat_form", clear_on_submit=True):
    query = st.text_input("Ask a question", placeholder="Type and press Enter")
    submitted = st.form_submit_button("Ask")
    if submitted and query:
        if st.session_state.err:
            st.error("No documents indexed.")
        else:
            ans = ask_rag(query,
                          st.session_state.idx,
                          st.session_state.chunked,
                          st.session_state.model)
            st.session_state.history.append((query, ans))
            st.experimental_rerun()
