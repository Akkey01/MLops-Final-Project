import faiss, pickle, os, numpy as np, sentence_transformers
ROOT = "/data/raw/icsi"
chunks, filenames = [], []
model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
for txt in sorted(Path(ROOT).rglob("*.txt")):
    content = Path(txt).read_text(encoding="utf-8", errors="ignore")
    if content.strip():
        chunks.extend(content.splitlines())
        filenames.extend([txt.name]*len(content.splitlines()))
emb = model.encode(chunks).astype("float32")
index = faiss.IndexFlatL2(emb.shape[1]); index.add(emb)
os.makedirs("/data/faiss_base", exist_ok=True)
faiss.write_index(index, "/data/faiss_base/index.bin")
pickle.dump(filenames, open("/data/faiss_base/meta.pkl", "wb"))
print("✅ FAISS built: ", index.ntotal)
