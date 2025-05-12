# app.py

import os
import pickle
import streamlit as st
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast
import nbformat, nltk, time
from nltk.tokenize import sent_tokenize

# ─── Configuration ─────────────────────────────────────────────────────────────
TOG_KEY = "c17ba696b3c19a15d9f3dd2b933c4627bfd2eeea848d62755ecce77123059a96"
DATA_ROOT = "ami_data"

# ─── Ensure NLTK punkt for chunking ─────────────────────────────────────────────
nltk.download("punkt", quiet=True)

# ─── Streamlit page config & CSS ────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat+", layout="wide")
st.markdown("""
  <style>
    body { background: #f5f7fa; }
    .block-container { padding: 1rem 2rem; padding-bottom: 6rem; } /* extra bottom padding */
    .title { font-size: 2rem; color: #2c3e50; margin-bottom: 1rem; font-weight:600; }
    .history { max-height: 70vh; overflow-y: auto; }
    .user-query, .bot-response {
      padding: 0.75rem; margin: 0.5rem 0; border-radius: 8px; color: #000;
    }
    .user-query { background: #e8f0fe; }
    .bot-response { background: #fff8dc; }
    .sources { font-size:0.8rem; color:#666; margin-top:0.25rem; }
    .input-container {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: #f5f7fa; padding: 1rem 2rem; box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
      z-index: 100;
    }
  </style>
""", unsafe_allow_html=True)
st.markdown("<div class='title'>💬 RAG Chat — Ask Anything from Your Docs</div>", unsafe_allow_html=True)

# ─── Utility: read any file type ────────────────────────────────────────────────
def extract_text_from_pdf(p):
    with pdfplumber.open(p) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
def extract_text_from_docx(p):
    return "\n".join(para.text for para in Document(p).paragraphs)
def extract_text_from_html(p):
    soup = BeautifulSoup(open(p, encoding="utf-8"), "html.parser")
    return html2text.html2text(soup.prettify())
def extract_text_from_txt(p):
    for enc in ("utf-8","latin-1"):
        try: return open(p, encoding=enc).read()
        except: pass
    return ""
def extract_text_from_xml(p):
    tree = ET.parse(p)
    return " ".join(e.text.strip() for e in tree.getroot().iter() if e.text)
def extract_text_from_ipynb(p):
    nb = nbformat.read(p, as_version=4)
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type in ("markdown","code"))

def read_any_file(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext==".pdf":   return extract_text_from_pdf(fp)
    if ext==".docx":  return extract_text_from_docx(fp)
    if ext in (".txt",".md",".css"): return extract_text_from_txt(fp)
    if ext==".html":  return extract_text_from_html(fp)
    if ext==".xml":   return extract_text_from_xml(fp)
    if ext==".ipynb": return extract_text_from_ipynb(fp)
    return ""

# ─── Chunking with overlap ─────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500, overlap_sents=3):
    if not text.strip(): return []
    sents = sent_tokenize(text)
    chunks, cur, ct = [], [], 0
    for s in sents:
        tl = len(tokenizer.encode(s, add_special_tokens=False))
        if ct + tl > max_tokens:
            chunks.append(" ".join(cur))
            keep = cur[-overlap_sents:] if len(cur)>overlap_sents else cur
            cur, ct = keep.copy(), sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in keep)
            cur.append(s); ct+=tl
        else:
            cur.append(s); ct+=tl
    if cur: chunks.append(" ".join(cur))
    return chunks

# ─── Persistent Base Index ─────────────────────────────────────────────────────
INDEX_PATH, CHUNKS_PATH = "index.faiss", "chunks.pkl"
@st.cache_data
def build_or_load_index(root=DATA_ROOT):
    # load existing
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        idx = faiss.read_index(INDEX_PATH)
        chunks = pickle.load(open(CHUNKS_PATH,"rb"))
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return idx, chunks, model, None
    # build fresh
    docs=[]
    if os.path.isdir(root):
        for dp,_,files in os.walk(root):
            for f in files:
                fp=os.path.join(dp,f)
                txt=read_any_file(fp)
                if txt.strip(): docs.append({"fn":f,"txt":txt})
    if not docs: return None,None,None,"No base documents."
    all_chunks=[]
    for d in docs:
        all_chunks += [{"fn":d["fn"],"chunk":c} for c in chunk_text(d["txt"])]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([c["chunk"] for c in all_chunks]).astype("float32")
    dim = embs.shape[1]
    if len(all_chunks)>1000:
        nlist=min(len(all_chunks)//10,100)
        quant=faiss.IndexFlatL2(dim)
        idx=faiss.IndexIVFFlat(quant,dim,nlist,faiss.METRIC_L2)
        idx.train(embs); idx.add(embs)
    else:
        idx=faiss.IndexFlatL2(dim); idx.add(embs)
    faiss.write_index(idx, INDEX_PATH)
    pickle.dump(all_chunks, open(CHUNKS_PATH,"wb"))
    return idx, all_chunks, model, None

with st.spinner("Building/loading base index…"):
    base_idx, base_chunks, model, err = build_or_load_index()
if err: st.error(err)

# ─── Setup Upload Index & History ─────────────────────────────────────────────
if "upload_idx" not in st.session_state:
    dim = model.get_sentence_embedding_dimension()
    st.session_state.upload_idx = faiss.IndexFlatL2(dim)
    st.session_state.upload_chunks = []
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar: Upload & Settings ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    k_value = st.slider("Context chunks (k)",1,10,5)
    temperature = st.slider("Temperature",0.0,1.0,0.3,0.1)
    show_sources = st.checkbox("Show sources",True)
    st.markdown("---")
    st.header("📤 Upload Documents")
    ufs = st.file_uploader("pdf,docx,txt,md,html,xml,ipynb",accept_multiple_files=True)
    if ufs:
        total=0; prog=st.progress(0)
        for i,f in enumerate(ufs):
            dst=os.path.join("uploads",f.name)
            os.makedirs("uploads",exist_ok=True)
            with open(dst,"wb") as out: out.write(f.getbuffer())
            txt=read_any_file(dst)
            chunks=chunk_text(txt)
            if chunks:
                embs=model.encode(chunks).astype("float32")
                st.session_state.upload_idx.add(embs)
                st.session_state.upload_chunks += [{"fn":f.name,"chunk":c} for c in chunks]
                total += len(chunks)
            prog.progress((i+1)/len(ufs))
        st.success(f"Indexed {total} chunks.")
    if st.button("Clear Uploads"):
        st.session_state.upload_idx = faiss.IndexFlatL2(dim)
        st.session_state.upload_chunks = []
        st.success("Cleared uploads.")
    if st.button("New Chat"):
        st.session_state.history = []
        st.experimental_rerun()

# ─── RAG & Generation ─────────────────────────────────────────────────────────
def generate_answer(prompt):
    for _ in range(3):
        try:
            r=requests.post(
              "https://api.together.xyz/v1/completions",
              headers={"Authorization":f"Bearer {TOG_KEY}","Content-Type":"application/json"},
              json={"model":"mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "prompt":prompt,"max_tokens":512,
                    "temperature":temperature,"top_p":0.9}
            )
            r.raise_for_status()
            data=r.json()
            if "choices" in data: return data["choices"][0].get("text","")
            if data.get("output","") and "choices" in data["output"]:
                return data["output"]["choices"][0].get("text","")
            return "⚠️ Bad response."
        except:
            time.sleep(1)
    return "⚠️ API error."

def ask_rag(query):
    ql=query.strip().lower()
    if ql in ("hi","hello","hey"):
        return "Hello! How can I help you today?", []
    qv=model.encode([query]).astype("float32")
    ctx,sources=[],[]
    # uploads first
    if st.session_state.upload_idx.ntotal>0:
        D,I=st.session_state.upload_idx.search(qv,k_value)
        for dist,i in zip(D[0],I[0]):
            ch=st.session_state.upload_chunks[i]
            ctx.append(ch["chunk"]); sources.append((ch["fn"],dist))
    # then base
    rem=k_value-len(ctx)
    if rem>0:
        D,I=base_idx.search(qv,rem)
        for dist,i in zip(D[0],I[0]):
            ch=base_chunks[i]
            ctx.append(ch["chunk"]); sources.append((ch["fn"],dist))
    if not ctx: return "❓ No relevant context found.", []
    prompt = f"Context:\n{'\n\n'.join(ctx)}\n\nQuestion: {query}\nAnswer:"
    return generate_answer(prompt), sources

# ─── Chat History & Fixed Input ────────────────────────────────────────────────
st.markdown("<div class='history'>", unsafe_allow_html=True)
for turn in st.session_state.history:
    st.markdown(f"<div class='user-query'><b>You:</b> {turn['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-response'><b>RAG:</b> {turn['a']}</div>", unsafe_allow_html=True)
    if show_sources and turn.get("sources"):
        srcs=", ".join(f"{fn} ({dist:.2f})" for fn,dist in turn["sources"])
        st.markdown(f"<div class='sources'>Sources: {srcs}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# fixed input at bottom
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
query = st.text_input("Ask a question", key="user_input", placeholder="Type here and press Enter")
if query:
    ans, srcs = ask_rag(query)
    st.session_state.history.append({"q":query,"a":ans,"sources":srcs})
    st.session_state.user_input = ""  # clear input
st.markdown("</div>", unsafe_allow_html=True)
