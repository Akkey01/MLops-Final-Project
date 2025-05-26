import os, time, pickle, nbformat, faiss
import streamlit as st
from streamlit_chat import message
import pdfplumber, html2text, xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np, requests
from transformers import GPT2TokenizerFast
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from prometheus_client import start_http_server, Counter, Histogram, REGISTRY

# ─── NLTK setup (punkt only) ────────────────────────────────────────────────
os.environ["NLTK_DATA"] = "/usr/share/nltk_data"
nltk.data.path.append("/usr/share/nltk_data")
pparam = PunktParameters()
punkt_tokenizer = PunktSentenceTokenizer(pparam)
def sent_tokenize(text: str):
    return punkt_tokenizer.tokenize(text.strip())

# ─── Prometheus instrumentation ───────────────────────────────────────────────
try:
    REQUEST_COUNT = Counter('rag_requests_total', 'Total RAG queries')
    LATENCY       = Histogram('rag_query_latency_seconds', 'RAG query latency')
    start_http_server(8000)
except ValueError:
    REQUEST_COUNT = REGISTRY._names_to_collectors['rag_requests_total']
    LATENCY       = REGISTRY._names_to_collectors['rag_query_latency_seconds']

def timed_ask_rag(**kwargs):
    REQUEST_COUNT.inc()
    with LATENCY.time():
        return ask_rag(**kwargs)

# ─── RAG + LLM call ───────────────────────────────────────────────────────────
TOG_KEY    = "c17ba696b3c19a15d9f3dd2b933c4627bfd2eeea848d62755ecce77123059a96"
DATA_ROOT  = "ami_data"
INDEX_PATH = "index.faiss"
CHUNKS_PATH= "chunks.pkl"

def ask_rag(
    query: str,
    model: SentenceTransformer,
    base_idx: faiss.Index,
    base_chunks: list,
    k: int,
    upload_idx: faiss.Index,
    upload_chunks: list,
    temperature: float
):
    vec = model.encode([query]).astype("float32")
    ctx, srcs = [], []

    # 1) uploaded docs
    if upload_idx.ntotal>0:
        D,I = upload_idx.search(vec, k)
        for dist,i in zip(D[0], I[0]):
            c = upload_chunks[i]
            ctx.append(c["chunk"])
            srcs.append((c["fn"], dist))

    # 2) base docs
    if len(ctx)<k:
        D,I = base_idx.search(vec, k-len(ctx))
        for dist,i in zip(D[0], I[0]):
            c = base_chunks[i]
            ctx.append(c["chunk"])
            srcs.append((c["fn"], dist))

    if not ctx:
        return "❓ No relevant context found.", []

    prompt = f"Context:\n\n{chr(10).join(ctx)}\n\nQuestion: {query}\nAnswer:"
    r = requests.post(
        "https://api.together.xyz/v1/completions",
        headers={
            "Authorization":f"Bearer {TOG_KEY}",
            "Content-Type":"application/json"
        },
        json={
            "model":"mistralai/Mixtral-8x7B-Instruct-v0.1",
            "prompt":prompt,
            "max_tokens":512,
            "temperature":temperature,
            "top_p":0.9
        }
    )
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or data.get("output",{}).get("choices",[])
    text = choices[0].get("text","⚠️ No response.") if choices else "⚠️ No choices."
    return text.strip(), srcs

# ─── File→text utilities ──────────────────────────────────────────────────────
def extract_text_from_pdf(p):
    with pdfplumber.open(p) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
def extract_text_from_docx(p):
    return "\n".join(para.text for para in Document(p).paragraphs)
def extract_text_from_html(p):
    soup = BeautifulSoup(open(p, encoding="utf-8"), "html.parser")
    return html2text.html2text(soup.prettify())
def extract_text_from_txt(p):
    for e in ("utf-8","latin-1"):
        try: return open(p, encoding=e).read()
        except: pass
    return ""
def extract_text_from_xml(p):
    tree = ET.parse(p)
    return " ".join(e.text.strip() for e in tree.getroot().iter() if e.text)
def extract_text_from_ipynb(p):
    nb = nbformat.read(p, as_version=4)
    return "\n".join(c.source for c in nb.cells if c.cell_type in ("markdown","code"))

def read_any_file(fp):
    ext = os.path.splitext(fp)[1].lower()
    return {
      ".pdf":   extract_text_from_pdf,
      ".docx":  extract_text_from_docx,
      ".html":  extract_text_from_html,
      ".xml":   extract_text_from_xml,
      ".ipynb": extract_text_from_ipynb
    }.get(ext, extract_text_from_txt)(fp)

# ─── Chunking ────────────────────────────────────────────────────────────────
tok = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500, overlap=3):
    if not text.strip(): return []
    sents = sent_tokenize(text)
    chunks, cur, ct = [], [], 0
    for s in sents:
        l = len(tok.encode(s, add_special_tokens=False))
        if ct + l > max_tokens:
            chunks.append(" ".join(cur))
            keep = cur[-overlap:] if len(cur)>overlap else cur
            cur = keep.copy()
            ct  = sum(len(tok.encode(x, add_special_tokens=False)) for x in keep)
        cur.append(s); ct += l
    if cur: chunks.append(" ".join(cur))
    return chunks

# ─── Build/load FAISS index ───────────────────────────────────────────────────
@st.cache_data
def build_or_load_index(root=DATA_ROOT):
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        idx    = faiss.read_index(INDEX_PATH)
        chunks = pickle.load(open(CHUNKS_PATH,"rb"))
        mdl    = SentenceTransformer('all-MiniLM-L6-v2')
        return idx, chunks, mdl

    docs=[]
    for dp,_,files in os.walk(root):
        for f in files:
            txt = read_any_file(os.path.join(dp,f))
            if txt.strip():
                docs.append({"fn":f,"txt":txt})

    all_chunks=[]
    for d in docs:
        all_chunks += [{"fn":d["fn"], "chunk":c} for c in chunk_text(d["txt"])]

    mdl   = SentenceTransformer('all-MiniLM-L6-v2')
    embs  = mdl.encode([c["chunk"] for c in all_chunks]).astype("float32")
    dim   = embs.shape[1]

    if len(all_chunks)>1000:
        nlist = min(len(all_chunks)//10,100)
        quant = faiss.IndexFlatL2(dim)
        idx   = faiss.IndexIVFFlat(quant,dim,nlist,faiss.METRIC_L2)
        idx.train(embs); idx.add(embs)
    else:
        idx = faiss.IndexFlatL2(dim); idx.add(embs)

    faiss.write_index(idx, INDEX_PATH)
    pickle.dump(all_chunks, open(CHUNKS_PATH,"wb"))
    return idx, all_chunks, mdl

# ─── Layout & CSS ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="IMPS.AI", layout="wide")
st.markdown("""
<style>
:root {
  --bg: #1e1e2f; --fg: #e0e0e0;
  --user-bg: #2a2a3b; --bot-bg: #33334a;
}
body, .stApp { background:var(--bg); color:var(--fg); }
.stApp .block-container { padding-bottom:5rem; position:relative; }
.chat-container {
  position:absolute; top:4rem; bottom:4rem; left:2rem; right:2rem;
  overflow-y:auto; border:1px solid #444; border-radius:8px; padding:1rem;
}
.input-area {
  position:fixed; bottom:0; left:2rem; right:2rem;
  background:var(--bg); padding:1rem; border-top:1px solid #444;
}
.source { font-size:0.75rem; color:#888; margin-left:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Build index ───────────────────────────────────────────────────────────────
with st.spinner("Loading documents…"):
    base_idx, base_chunks, model = build_or_load_index()

# ─── Session init ─────────────────────────────────────────────────────────────
if "upload_idx" not in st.session_state:
    d = model.get_sentence_embedding_dimension()
    st.session_state.upload_idx    = faiss.IndexFlatL2(d)
    st.session_state.upload_chunks = []
if "history" not in st.session_state:
    st.session_state.history   = []
if "latencies" not in st.session_state:
    st.session_state.latencies = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Page", ["Chat","Metrics"])
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
k_value     = st.sidebar.slider("Context chunks (k)",1,10,5)
temperature = st.sidebar.slider("Temperature",0.0,1.0,0.3,0.1)
st.sidebar.markdown("---")
st.sidebar.header("📤 Upload Documents")
ufs = st.sidebar.file_uploader("", accept_multiple_files=True)
if ufs:
    prog, total = st.sidebar.progress(0), 0
    os.makedirs("uploads", exist_ok=True)
    for i,f in enumerate(ufs):
        dst = os.path.join("uploads", f.name)
        with open(dst,"wb") as out: out.write(f.getbuffer())
        ch = chunk_text(read_any_file(dst))
        if ch:
            embs = model.encode(ch).astype("float32")
            st.session_state.upload_idx.add(embs)
            st.session_state.upload_chunks += [{"fn":f.name,"chunk":c} for c in ch]
            total += len(ch)
        prog.progress((i+1)/len(ufs))
    st.sidebar.success(f"Indexed {total} chunks.")

# ─── Chat page ─────────────────────────────────────────────────────────────────
if page == "Chat":
    st.title("💬 IMPS.AI — Ask Anything from Your Docs")

    # History (scrollable)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for idx, turn in enumerate(st.session_state.history):
        # give each user bubble a unique key
        message(
            turn["q"],
            is_user=True,
            key=f"user_{idx}",
            avatar_style="pixel-art",
            seed="user"
        )
        # and each bot bubble one too
        message(
            turn["a"],
            is_user=False,
            key=f"bot_{idx}",
            avatar_style="pixel-art",
            seed="bot"
        )
        if turn.get("sources"):
            src = ", ".join(f"{fn} ({d:.1f})" for fn,d in turn["sources"])
            st.markdown(f"<div class='source'>Sources: {src}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input bar (fixed)
    st.markdown("<div class='input-area'>", unsafe_allow_html=True)
    with st.form("query_form", clear_on_submit=True):
        query  = st.text_input("Your question…", "")
        submit = st.form_submit_button("Send")
    if submit and query:
        start = time.time()
        ans, srcs = timed_ask_rag(
            query=query,
            model=model,
            base_idx=base_idx,
            base_chunks=base_chunks,
            k=k_value,
            upload_idx=st.session_state.upload_idx,
            upload_chunks=st.session_state.upload_chunks,
            temperature=temperature
        )
        st.session_state.latencies.append(time.time() - start)
        st.session_state.history.append({"q":query,"a":ans,"sources":srcs})
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Metrics page ─────────────────────────────────────────────────────────────
else:
    st.title("📊 IMPS.AI Metrics")
    st.metric("Total Queries", len(st.session_state.latencies),
              delta=f"{st.session_state.latencies[-1]:.2f}s last" 
                    if st.session_state.latencies else "")
    st.subheader("Latency Over Time (s)")
    st.line_chart(st.session_state.latencies or [0])
    st.subheader("Prometheus Endpoint")
    st.write("[http://localhost:8000/metrics](http://localhost:8000/metrics)")
