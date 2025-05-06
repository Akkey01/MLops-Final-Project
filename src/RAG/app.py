import streamlit as st
import os

DATA_ROOT = os.getenv("DATA_DIR", "ami_public_manual_1.6.2")
# ── 1) Must be very first Streamlit call ────────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="🤖")

# ── 2) Now your imports ─────────────────────────────────────────────────────────
import os, pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast

# ── 3) Document readers (unchanged) ──────────────────────────────────────────────
def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
def extract_text_from_docx(path):
    return "\n".join(p.text for p in Document(path).paragraphs)
def extract_text_from_html(path):
    soup = BeautifulSoup(open(path, encoding="utf-8"), "html.parser")
    return html2text.html2text(soup.prettify())
def extract_text_from_txt(path):
    return open(path, encoding="utf-8").read()
def extract_text_from_xml(path):
    tree = ET.parse(path)
    return " ".join(e.text.strip() for e in tree.getroot().iter() if e.text)
def read_any_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext==".pdf":   return extract_text_from_pdf(filepath)
    if ext==".docx":  return extract_text_from_docx(filepath)
    if ext==".html":  return extract_text_from_html(filepath)
    if ext in (".txt",".md",".css"): return extract_text_from_txt(filepath)
    if ext==".xml":   return extract_text_from_xml(filepath)
    raise ValueError(f"Unsupported file type: {ext}")

def load_all_documents(root_folder):
    docs=[]
    for dp,_,files in os.walk(root_folder):
        for f in files:
            full=os.path.join(dp,f)
            try:
                docs.append({"filename":os.path.relpath(full,root_folder),"text":read_any_file(full)})
            except: pass
    return docs

# ── 4) Chunking ───────────────────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500):
    words=text.split(); chunks=[]; cur=[]; cur_t=0
    for w in words:
        tl=len(tokenizer.encode(w, add_special_tokens=False))
        if cur_t+tl>max_tokens:
            chunks.append(" ".join(cur)); cur=[w]; cur_t=tl
        else:
            cur.append(w); cur_t+=tl
    if cur: chunks.append(" ".join(cur))
    return chunks

# ── 5) Build & cache FAISS index ─────────────────────────────────────────────────
@st.cache_data
def build_index(root_folder):
    # 1) load everything
    docs = load_all_documents(root_folder)
    if not docs:
        return None, None, None, "No documents found. Check folder path."

    # 2) chunk
    chunked = []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunked.append({"filename": d["filename"], "chunk": c})
    if not chunked:
        return None, None, None, "No text chunks created (files may be empty)."

    # 3) embed & index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([c["chunk"] for c in chunked]).astype('float32')
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)

    # 4) no error
    return idx, chunked, model, None

# — 4) Together.ai generation —
TOGETHER_API_KEY = "cc4b628095c0531f06fe08ff20e1f0bad8cf4e6c39ed2b3c70744a6278a7faab"
def generate_answer(prompt, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    r=requests.post(
      "https://api.together.xyz/v1/completions",
      headers={"Authorization":f"Bearer {TOGETHER_API_KEY}","Content-Type":"application/json"},
      json={"model":model_name,"prompt":prompt,"max_tokens":512,"temperature":0.3,"top_p":0.9}
    ).json()
    return r["choices"][0]["text"]

def ask_rag(query, idx, chunked, emb_model, k=5):
    q_emb=emb_model.encode([query]).astype('float32')
    D,I=idx.search(q_emb,k)
    context="\n\n".join(chunked[i]["chunk"] for i in I[0])
    prompt=f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_answer(prompt)

# ── Streamlit UI ────────────────────────────────────────────────────────────

st.title("📚 RAG-Powered Chat")

# 1) Input for root folder & build index
root = st.text_input("📂 Root folder for docs", DATA_ROOT)
if st.button("🔄 Build/Refresh Index"):
    idx, chunked, emb_model, error = build_index(root)
    if error:
        st.error(error)
    else:
        st.success(f"Index built with {idx.ntotal} chunks")
        st.session_state.idx = idx
        st.session_state.chunked = chunked
        st.session_state.emb_model = emb_model

st.markdown("---")

# 2) Question input and Ask button (always shown)
question = st.text_input("💬 Your question", key="question_input")
ask = st.button("Ask")

# 3) Only when user clicks “Ask” AND index exists in session
if ask:
    if "idx" not in st.session_state:
        st.warning("Please build the index first.")
    elif not question:
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking…"):
            answer = ask_rag(
                question,
                st.session_state.idx,
                st.session_state.chunked,
                st.session_state.emb_model
            )
        st.markdown("**Answer:**")
        st.write(answer)
