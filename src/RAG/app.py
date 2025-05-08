import os, json, tempfile
import streamlit as st
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast
import nbformat

TOG_KEY = "cc4b628095c0531f06fe08ff20e1f0bad8cf4e6c39ed2b3c70744a6278a7faab"

st.set_page_config(page_title="RAG Chat", layout="wide")

st.markdown("""
    <style>
        body { background-color: #f5f7fa; }
        .block-container { padding: 2rem; }
        .title { font-size: 2.2rem; font-weight: 600; color: #2c3e50; }
        .user-query, .bot-response {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 10px;
        }
        .user-query { background-color: #e8f0fe; }
        .bot-response { background-color: #fdf6e3; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>💬 RAG Chat - Ask Anything From Your Docs</h1>", unsafe_allow_html=True)

# ------------------ File Extractors ------------------ #
def extract_text_from_pdf(p): return "\n".join(page.extract_text() or "" for page in pdfplumber.open(p).pages)
def extract_text_from_docx(p): return "\n".join(para.text for para in Document(p).paragraphs)
def extract_text_from_html(p):
    soup=BeautifulSoup(open(p, encoding="utf-8"),"html.parser")
    return html2text.html2text(soup.prettify())
def extract_text_from_txt(p): return open(p, encoding="utf-8").read()
def extract_text_from_xml(p):
    tree=ET.parse(p)
    return " ".join(e.text.strip() for e in tree.getroot().iter() if e.text)
def extract_text_from_ipynb(p):
    nb=nbformat.read(p, as_version=4)
    parts=[]
    for cell in nb.cells:
        if cell.cell_type in ("markdown","code"):
            parts.append(cell.source)
    return "\n".join(parts)

def read_any_file(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".pdf": return extract_text_from_pdf(fp)
    if ext == ".docx": return extract_text_from_docx(fp)
    if ext in (".txt", ".md", ".css"): return extract_text_from_txt(fp)
    if ext == ".xml": return extract_text_from_xml(fp)
    if ext == ".html": return extract_text_from_html(fp)
    if ext == ".ipynb": return extract_text_from_ipynb(fp)
    raise ValueError(f"Unsupported file type: {ext}")

# ------------------ Chunking ------------------ #
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks, cur, tokens = [], [], 0
    for w in words:
        tl = len(tokenizer.encode(w, add_special_tokens=False))
        if tokens + tl > max_tokens:
            chunks.append(" ".join(cur))
            cur, tokens = [w], tl
        else:
            cur.append(w)
            tokens += tl
    if cur: chunks.append(" ".join(cur))
    return chunks

# ------------------ FAISS Index ------------------ #
@st.cache_resource
def build_index(root="ami_data"):
    docs = []
    for dp, _, files in os.walk(root):
        for f in files:
            fp = os.path.join(dp, f)
            try:
                docs.append({"fn": f, "txt": read_any_file(fp)})
            except: pass
    if not docs:
        return None, None, None, "No documents found."
    chunked = []
    for d in docs:
        for c in chunk_text(d["txt"]):
            chunked.append({"fn": d["fn"], "chunk": c})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([c["chunk"] for c in chunked]).astype("float32")
    idx = faiss.IndexFlatL2(embs.shape[1]); idx.add(embs)
    return idx, chunked, model, None

# ------------------ Generator ------------------ #
def generate_answer(prompt):
    r = requests.post("https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOG_KEY}", "Content-Type": "application/json"},
        json={"model":"mistralai/Mixtral-8x7B-Instruct-v0.1","prompt":prompt,"max_tokens":512,"temperature":0.3,"top_p":0.9}
    ).json()
    if "choices" in r: return r["choices"][0].get("text","")
    if r.get("output") and "choices" in r["output"]: return r["output"]["choices"][0].get("text","")
    st.error(f"Unexpected response: {r}")
    return ""

def ask_rag(q, idx, chunked, model, k=5):
    qv = model.encode([q]).astype("float32")
    D, I = idx.search(qv, k)
    ctx = "\n\n".join(chunked[i]["chunk"] for i in I[0])
    return generate_answer(f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

# ------------------ Session Init ------------------ #
if "idx" not in st.session_state:
    st.session_state.idx, st.session_state.chunked, st.session_state.model, st.session_state.err = build_index()

# ------------------ File Upload & Embedding ------------------ #
st.sidebar.header("📤 Upload Files")
ufs = st.sidebar.file_uploader("Supported: pdf, docx, txt, ipynb, html, xml", accept_multiple_files=True, type=["pdf", "docx", "txt", "md", "css", "xml", "html", "ipynb"])
if ufs:
    os.makedirs("uploads", exist_ok=True)
    for f in ufs:
        fname = os.path.join("uploads", f.name)
        with open(fname, "wb") as out: out.write(f.getbuffer())
        try:
            txt = read_any_file(fname)
            new_chunks = [{"fn": f.name, "chunk": c} for c in chunk_text(txt)]
            new_embs = st.session_state.model.encode([c["chunk"] for c in new_chunks]).astype("float32")
            st.session_state.idx.add(new_embs)
            st.session_state.chunked.extend(new_chunks)
        except Exception as e:
            st.error(f"Error processing {f.name}: {e}")
    st.sidebar.success("Uploaded & indexed.")

# ------------------ Chat UI ------------------ #
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("---")
for q, a in st.session_state.history:
    st.markdown(f"<div class='user-query'><b>👤 You:</b> {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-response'><b>🤖 RAG:</b> {a}</div>", unsafe_allow_html=True)

with st.form("query_box", clear_on_submit=True):
    user_input = st.text_input("Ask a question", placeholder="Type and press Enter")
    submitted = st.form_submit_button("Ask")
if submitted and user_input:
    if st.session_state.err:
        st.error("No documents indexed.")
    else:
        answer = ask_rag(user_input, st.session_state.idx, st.session_state.chunked, st.session_state.model)
        st.session_state.history.append((user_input, answer))
