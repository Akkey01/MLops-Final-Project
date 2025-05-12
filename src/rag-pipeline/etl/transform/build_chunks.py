#!/usr/bin/env python3
import json, yaml, argparse, pathlib
from nltk.tokenize import sent_tokenize
from transformers import GPT2TokenizerFast

# ── Chunking fn ────────────────────────────────────────────────
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def chunk_text(text, max_tokens=500, overlap_sents=3):
    sents = sent_tokenize(text)
    chunks, cur, ct = [], [], 0
    for s in sents:
        tl = len(tokenizer.encode(s, add_special_tokens=False))
        if ct + tl > max_tokens:
            chunks.append(" ".join(cur))
            keep = cur[-overlap_sents:] if len(cur)>overlap_sents else cur
            cur, ct = keep.copy(), sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in keep)
        cur.append(s); ct += tl
    if cur: chunks.append(" ".join(cur))
    return chunks

# ── Main ────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--transcripts", required=True,
               help="comma-sep JSONL names in /data/processed")
p.add_argument("--splits_cfg", required=True)
p.add_argument("--out_dir", required=True)
args = p.parse_args()

# load splits
splits = yaml.safe_load(pathlib.Path(args.splits_cfg).read_text())

# load all transcripts
records = []
for fn in args.transcripts.split(","):
    path = pathlib.Path("/data/processed")/fn
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        rec["meeting_id"] = rec["file"].split("_")[0]
        records.append(rec)

# OPTIONAL: parse NXT annotations here (TODO)

# build chunks
all_chunks = []
for rec in records:
    for c in chunk_text(rec["text"]):
        all_chunks.append({
            "meeting_id": rec["meeting_id"],
            "chunk": c
        })

# write per-split JSONL
out_base = pathlib.Path(args.out_dir)
out_base.mkdir(parents=True, exist_ok=True)

for split, info in splits.items():
    mids = set(info["meetings"])
    split_chunks = [c for c in all_chunks if c["meeting_id"] in mids]
    outp = out_base / f"{split}_chunks.jsonl"
    with open(outp, "w") as f:
        for c in split_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(split_chunks)} → {outp}")
