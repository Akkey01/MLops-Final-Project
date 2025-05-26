#!/usr/bin/env python3
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────
ROOT      = "/data/raw/icsi"
OUT_DIR   = "/data/faiss_base"
MODEL_NAME= "all-MiniLM-L6-v2"

def main():
    # 1. Load all transcripts
    docs = []
    for txt_path in sorted(Path(ROOT).rglob("*.txt")):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append({"fn": txt_path.name, "txt": text})

    if not docs:
        print(f"No transcripts found under {ROOT}")
        return

    # 2. Chunk by non-empty lines
    chunks = []
    for doc in docs:
        for line in doc["txt"].splitlines():
            line = line.strip()
            if line:
                chunks.append({"fn": doc["fn"], "chunk": line})

    # 3. Embed chunks
    print("🔍 Encoding", len(chunks), "chunks with", MODEL_NAME, "…")
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["chunk"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).astype("float32")

    # 4. Build FAISS index
    dim = embeddings.shape[1]
    print("⚙️  Building FAISS index (dim =", dim, ", ntotal =", len(embeddings), ")…")
    if len(embeddings) > 1000:
        nlist = min(len(embeddings) // 10, 100)
        quant = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    # 5. Save index + chunks
    os.makedirs(OUT_DIR, exist_ok=True)
    idx_path    = os.path.join(OUT_DIR, "index.bin")
    chunks_path = os.path.join(OUT_DIR, "chunks.pkl")

    print(f"💾 Saving index to {idx_path} and metadata to {chunks_path}…")
    faiss.write_index(index, idx_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ FAISS built: {index.ntotal} vectors saved in {OUT_DIR}")

if __name__ == "__main__":
    main()