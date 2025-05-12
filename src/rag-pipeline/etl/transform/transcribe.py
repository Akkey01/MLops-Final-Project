#!/usr/bin/env python3
import os, json, time, pathlib, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

# If no real endpoint, write dummy transcripts
if not os.getenv("WHISPER_ENDPOINT_URL"):
    print("⚠️  Whisper stub: generating dummy transcripts")
    out = []
    for wav in pathlib.Path(args.audio_dir).glob("*.wav"):
        out.append({"file": wav.name,
                    "text": f"[DUMMY] {wav.stem}",
                    "ts": time.time()})
    pathlib.Path(args.out).write_text(
        "\n".join(json.dumps(x) for x in out)
    )
    exit(0)

# Real call path
import requests, tqdm
endpoint = os.getenv("WHISPER_ENDPOINT_URL")

results = []
for wav in tqdm.tqdm(sorted(pathlib.Path(args.audio_dir).glob("*.wav"))):
    with open(wav, "rb") as f:
        r = requests.post(endpoint, files={"file": f})
    r.raise_for_status()
    text = r.json().get("text","")
    results.append({"file": wav.name, "text": text, "ts": time.time()})

pathlib.Path(args.out).write_text(
    "\n".join(json.dumps(x) for x in results)
)
print(f"Wrote {len(results)} transcripts to {args.out}")
