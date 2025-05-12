#!/usr/bin/env python3
import argparse, pathlib, sys
import whisper

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Directory to cache Whisper model")
    p.add_argument("--audio_dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Load Whisper model (cached in model_dir)
    model = whisper.load_model("base", download_root=args.model_dir)

    out = []
    for wav in pathlib.Path(args.audio_dir).rglob("*.wav"):
        print(f"Transcribing {wav}…")
        result = model.transcribe(str(wav))
        text = result.get("text", "").strip()
        out.append({"file": wav.name, "text": text})

    # Write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(f"{rec}\n")
    print(f"Wrote {len(out)} transcripts to {args.out}")

if __name__ == "__main__":
    main()
