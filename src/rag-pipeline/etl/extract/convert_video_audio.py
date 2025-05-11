#!/usr/bin/env python3
import argparse, pathlib, subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="path to video files")
    p.add_argument("--dst", required=True, help="output WAV dir")
    args = p.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for vid in src.rglob("*"):
        if vid.suffix.lower() not in (".mp4", ".mkv", ".mov", ".avi"):
            continue
        out = dst / f"{vid.stem}.wav"
        subprocess.run([
            "ffmpeg", "-i", str(vid),
            "-ar", "16000", "-ac", "1", str(out)
        ], check=True)
        print(f"Converted {vid} → {out}")

if __name__ == "__main__":
    main()
