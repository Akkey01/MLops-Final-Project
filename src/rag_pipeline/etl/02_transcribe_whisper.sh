#!/usr/bin/env bash
set -e

# ─── Install dependencies including ffmpeg ───────────────────────────────────────
apt-get update && apt-get install -y ffmpeg
pip install --no-cache-dir openai-whisper

# ─── Transcription ─────────────────────────────────────────────────────────────
python3 - << PYCODE
import whisper
import os
import glob

print("🔄 Loading Whisper model...")
# Use the tiny.en model which is small but works well for English
model = whisper.load_model("tiny.en")
print("✅ Model loaded successfully!")

print("🔍 Looking for WAV files...")
wav_files = glob.glob("/data/raw/icsi/audio/**/*.wav", recursive=True)
print(f"📊 Found {len(wav_files)} WAV files")

if not wav_files:
    print("⚠️ No WAV files found. Check the data path.")
    exit(0)

for wav in wav_files:
    print(f"🔉 Transcribing {wav}")
    try:
        # Transcribe audio
        result = model.transcribe(wav)
        text = result["text"]

        # Save transcription to text file
        txt_path = wav.rsplit(".", 1)[0] + ".txt"
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved transcription to {txt_path}")
    except Exception as e:
        print(f"❌ Error transcribing {wav}: {str(e)}")

print("✅ All transcriptions complete.")
PYCODE