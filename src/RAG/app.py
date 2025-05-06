import os
import streamlit as st
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, requests
from transformers import GPT2TokenizerFast
import tempfile

# ── UI: page config & custom CSS ─────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat+", page_icon="🤖")
st.markdown(
    """
    <style>
      body { background-color: #f0f8ff; }       /* AliceBlue background */
      .stApp { color: #333333; font-family: 'Arial'; }
      .title { font-size: 2.5rem; color: #2a4d69; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 class='title'>📚 Enhanced RAG-Powered Chat</h1>", unsafe_allow_html=True)

# ── Document readers ───────────────────────────────────────────────────────────
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
    if ext == ".pdf":      return extract_text_from_pdf(filepath)
    if ext == ".docx":     return extract_text_from_docx(filepath)
    if ext == ".html":     return extract_text_from_html(filepath)
    if ext in (".txt",".md",".css"): return extract_text_from_txt(filepath)
    if ext == ".xml":      return extract_text_from_xml(filepath)
    raise ValueError(f"Unsupported file type: {ext}")

# ── Chunking ───────────────────────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500):
    words, chunks, cur, cur_t = text.split(), [], [], 0
    for w in words:
        tl = len(tokenizer.encode(w, add_special_tokens=False))
        if cur_t + tl > max_tokens:
            chunks.append(" ".join(cur)); cur, cur_t = [w], tl
        else:
            cur.append(w); cur_t += tl
    if cur: chunks.append(" ".join(cur))
    return chunks

# ── Build & cache FAISS index ─────────────────────────────────────────────────
@st.cache_data
def build_index(root_folder):
    # load all existing files on disk
    docs = []
    for dp,_,files in os.walk(root_folder):
        for f in files:
            full = os.path.join(dp, f)
            try:
                docs.append({"filename": os.path.relpath(full, root_folder), "text": read_any_file(full)})
            except: pass

    if not docs:
        return None, None, None, "No documents found – check DATA_DIR."

    # chunk
    chunked = []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunked.append({"filename": d["filename"], "chunk": c})

    # embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode([c["chunk"] for c in chunked]).astype('float32')
    idx = faiss.IndexFlatL2(embs.shape[1]); idx.add(embs)
    return idx, chunked, model, None

# ── Together.ai generation ────────────────────────────────────────────────────
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "<paste-your-key>")
def generate_answer(prompt, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    r = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
        json={"model": model_name, "prompt": prompt, "max_tokens": 512, "temperature": 0.3, "top_p": 0.9}
    ).json()
    return r["choices"][0]["text"]

def ask_rag(query, idx, chunked, emb_model, k=5):
    q_emb = emb_model.encode([query]).astype('float32')
    D, I = idx.search(q_emb, k)
    context = "\n\n".join(chunked[i]["chunk"] for i in I[0])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_answer(prompt)

# ── Session & State Initialization ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar: Upload new files ───────────────────────────────────────────────────
st.sidebar.header("📤 Upload Documents")
uploaded = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)
if uploaded:
    # save to uploads/ and rebuild index to include them
    upl_dir = "uploads"
    os.makedirs(upl_dir, exist_ok=True)
    for f in uploaded:
        dest = os.path.join(upl_dir, f.name)
        with open(dest, "wb") as out: out.write(f.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded)} files.")
    # rebuild index including uploads
    root = os.getenv("DATA_DIR", "ami_public_manual_1.6.2")
    idx, chunked, emb_model, err = build_index(root)  # include root data
    # now add uploaded docs
    for fname in os.listdir(upl_dir):
        full = os.path.join(upl_dir, fname)
        txt = read_any_file(full)
        for c in chunk_text(txt):
            chunked.append({"filename": fname, "chunk": c})
            emb = emb_model.encode([c]).astype('float32')
            idx.add(emb)

# ── Main: Build index button ─────────────────────────────────────────────────────
st.write("---")
root = st.text_input("📂 Data root folder", os.getenv("DATA_DIR", "ami_public_manual_1.6.202"))
if st.button("🔄 Build/Refresh Index"):
    idx, chunked, emb_model, error = build_index(root)
    if error: st.error(error)
    else: st.success(f"Index built with {idx.ntotal} chunks")

# ── Chat input & history ─────────────────────────────────────────────────────────
st.write("---")
question = st.text_input("💬 Ask a question", key="q")
if st.button("Ask") and question:
    if 'idx' not in locals() and 'idx' not in st.session_state:
        st.warning("Build the index first.")
    else:
        idx_obj = st.session_state.get("idx", idx)
        chunks = st.session_state.get("chunked", chunked)
        model_inst = st.session_state.get("emb_model", emb_model)
        answer = ask_rag(question, idx_obj, chunks, model_inst)
        st.session_state.history.append((question, answer))

# ── Display history ──────────────────────────────────────────────────────────────
if st.session_state.history:
    st.write("### 🕘 Conversation History")
    for q,a in st.session_state.history[::-1]:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
