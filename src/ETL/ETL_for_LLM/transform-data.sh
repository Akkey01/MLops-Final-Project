#!/usr/bin/env bash
set -eux

RAW=/data/raw/newami
OUT=/data/processed

# skip if already transformed
if [ -n "$(ls -A "${OUT}" 2>/dev/null)" ]; then
  echo "▶ /data/processed is non‐empty, skipping transform."
  exit 0
fi

# install OCR+PDF+DOCX support
apt-get update && \
apt-get install -y tesseract-ocr poppler-utils python3-pip
pip3 install --no-cache-dir pdfminer.six python-docx pillow pytesseract

mkdir -p "${OUT}"

python3 << 'PYCODE'
import os, json
from pdfminer.high_level import extract_text as pdf2text
from docx import Document
from PIL import Image
import pytesseract

RAW = os.environ['RAW']
OUT = os.environ['OUT']

handlers = {
    '.txt': lambda p: open(p,encoding='utf8',errors='ignore').read(),
    '.trs': lambda p: open(p,encoding='utf8',errors='ignore').read(),
    '.xml': lambda p: open(p,encoding='utf8',errors='ignore').read(),
    '.json': lambda p: open(p,encoding='utf8',errors='ignore').read(),
    '.pdf': pdf2text,
    '.docx': lambda p: "\n".join(p.text for p in Document(p).paragraphs),
    '.png': lambda p: pytesseract.image_to_string(Image.open(p)),
    '.jpg': lambda p: pytesseract.image_to_string(Image.open(p)),
    '.jpeg': lambda p: pytesseract.image_to_string(Image.open(p)),
}

outputs = {
    'transcripts.jsonl': open(os.path.join(OUT,'transcripts.jsonl'),'w',encoding='utf8'),
    'manual.jsonl':      open(os.path.join(OUT,'manual.jsonl')     ,'w',encoding='utf8'),
    'automatic.jsonl':   open(os.path.join(OUT,'automatic.jsonl'),'w',encoding='utf8'),
}

for root,_,files in os.walk(RAW):
    for f in files:
        ext = os.path.splitext(f.lower())[1]
        rdr = handlers.get(ext)
        if not rdr: continue
        path = os.path.join(root,f)
        try:
            text = rdr(path).strip()
        except:
            continue
        if not text: continue

        rec = {'file': os.path.relpath(path, RAW), 'text': text}
        if '/amicorpus/' in root:
            out = outputs['transcripts.jsonl']
        elif '/manual_annotations/' in root:
            out = outputs['manual.jsonl']
        elif '/automatic_annotations/' in root:
            out = outputs['automatic.jsonl']
        else:
            continue

        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

for name, fh in outputs.items():
    fh.close()
    size = os.path.getsize(os.path.join(OUT, name))
    print(f"{name}: {size} bytes")

PYCODE


echo "▶ JSONL file sizes:"
du -h "${OUT}"/*.jsonl

echo "▶ Processed-dir listing:"
ls -lh "${OUT}"
