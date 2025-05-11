#!/usr/bin/env python3
import json, argparse, pathlib, pickle
import faiss
from sentence_transformers import SentenceTransformer

p = argparse.ArgumentParser()
p.add_argument("--chunks", required=True,
               help="comma-sep train_chunks.jsonl,val_chunks.jsonl")
p.add_argument("--index_dir", required=True)
args = p.parse_args()

# load chunks
cts = []
for fn in args.chunks.split(","):
    for line in pathlib.Path(fn).read_text().splitlines():
        cts.append(json.loads(line))

# embed
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = model.encode([c["chunk"] for c in cts]).astype("float32")

# build Faiss
dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)

# save
out = pathlib.Path(args.index_dir)
out.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(out/"index.faiss"))
pickle.dump(cts, open(out/"chunks.pkl","wb"))
print(f"Indexed {len(cts)} chunks → {out}")
