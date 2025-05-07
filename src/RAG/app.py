import os, json, tempfile
import streamlit as st
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast
import nbformat

# Hard‑code Together.ai API key (overwrite any env‑var)
TOG_KEY = "c17ba696b3c19a15d9f3dd2b933c4627bfd2eeea848d62755ecce77123059a96"

# ── UI SETUP ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat+", page_icon="🤖")
st.markdown("""
    <style>
      body { background-color: #f0f8ff; }
      .stApp { color: #333; font-family: Arial; }
      .title { font-size: 2.5rem; color: #2a4d69; }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h1 class='title'>📚 Enhanced RAG-Powered Chat</h1>", unsafe_allow_html=True)

# ── FILE READERS ────────────────────────────────────────────────────────────────
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
    ext=os.path.splitext(fp)[1].lower()
    if ext==".pdf": return extract_text_from_pdf(fp)
    if ext==".docx": return extract_text_from_docx(fp)
    if ext==".html": return extract_text_from_html(fp)
    if ext in (".txt",".md",".css"): return extract_text_from_txt(fp)
    if ext==".xml": return extract_text_from_xml(fp)
    if ext==".ipynb": return extract_text_from_ipynb(fp)
    raise ValueError(f"Unsupported file type: {ext}")

# ── CHUNKING ────────────────────────────────────────────────────────────────────
tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500):
    words=text.split(); chunks=[]; cur=[]; ct=0
    for w in words:
        l=len(tokenizer.encode(w, add_special_tokens=False))
        if ct+l>max_tokens:
            chunks.append(" ".join(cur)); cur=[w]; ct=l
        else:
            cur.append(w); ct+=l
    if cur: chunks.append(" ".join(cur))
    return chunks

# ── BUILD INDEX ─────────────────────────────────────────────────────────────────
@st.cache_data
def build_index(root):
    docs=[]
    # load on-disk data
    for dp,_,files in os.walk(root):
        for f in files:
            fp=os.path.join(dp,f)
            try:
                docs.append({"fn":os.path.relpath(fp,root),"txt":read_any_file(fp)})
            except Exception:
                pass
    # include uploads
    upl="uploads"
    if os.path.isdir(upl):
        for f in os.listdir(upl):
            fp=os.path.join(upl,f)
            try: docs.append({"fn":f,"txt":read_any_file(fp)})
            except: pass

    if not docs: return None,None,None,"No documents found."

    chunked=[]
    for d in docs:
        for c in chunk_text(d["txt"]):
            chunked.append({"fn":d["fn"],"chunk":c})

    model=SentenceTransformer('all-MiniLM-L6-v2')
    embs=model.encode([c["chunk"] for c in chunked]).astype('float32')
    idx=faiss.IndexFlatL2(embs.shape[1]); idx.add(embs)
    return idx,chunked,model,None

# ── TOGETHER.AI GENERATION ─────────────────────────────────────────────────────
TOG_KEY=os.getenv("TOGETHER_API_KEY","<your_key>")
def generate_answer(prompt):
    r=requests.post("https://api.together.xyz/v1/completions",
        headers={"Authorization":f"Bearer {TOG_KEY}","Content-Type":"application/json"},
        json={"model":"mistralai/Mixtral-8x7B-Instruct-v0.1","prompt":prompt,"max_tokens":512,"temperature":0.3,"top_p":0.9}
    ).json()
    # support both formats
    if "choices" in r:
        return r["choices"][0].get("text","")
    if r.get("output") and "choices" in r["output"]:
        return r["output"]["choices"][0].get("text","")
    st.error(f"Unexpected response structure: {r}")
    return ""

def ask_rag(q,idx,chunked,mod,k=5):
    qv=mod.encode([q]).astype('float32')
    D,I=idx.search(qv,k)
    ctx="\n\n".join(chunked[i]["chunk"] for i in I[0])
    return generate_answer(f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "idx" not in st.session_state:
    st.session_state.idx,st.session_state.chunked,st.session_state.mod,st.session_state.err=build_index(os.getenv("DATA_DIR","ami_public_manual_1.6.2"))

if st.session_state.err:
    st.error(st.session_state.err)

# ── UPLOAD WIDGET ──────────────────────────────────────────────────────────────
st.sidebar.header("Upload docs")
ufs=st.sidebar.file_uploader("Choose files",accept_multiple_files=True)
if ufs:
    os.makedirs("uploads",exist_ok=True)
    for f in ufs:
        with open(os.path.join("uploads",f.name),"wb") as out: out.write(f.getbuffer())
    st.sidebar.success("Uploaded and added to index.")
    # rebuild index to include uploads
    st.session_state.idx,st.session_state.chunked,st.session_state.mod,st.session_state.err=build_index(os.getenv("DATA_DIR","ami_public_manual_1.6.2"))

# ── QUERY FORM ─────────────────────────────────────────────────────────────────
st.write("---")
with st.form("query_form", clear_on_submit=False):
    question=st.text_input("💬 Ask a question", placeholder="Type and press Enter")
    submitted=st.form_submit_button("Ask")
if submitted and question:
    if st.session_state.err:
        st.error("Cannot query: no index.")
    else:
        ans=ask_rag(question,st.session_state.idx,st.session_state.chunked,st.session_state.mod)
        st.session_state.history=st.session_state.get("history",[])+[(question,ans)]

# ── HISTORY ────────────────────────────────────────────────────────────────────
if st.session_state.get("history"):
    st.write("### Conversation History")
    for q,a in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
