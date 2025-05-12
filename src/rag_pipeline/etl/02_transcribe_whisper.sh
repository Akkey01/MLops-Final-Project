#!/usr/bin/env bash
set -e
MODEL="ggml-tiny.en.bin"   # 23 MB
test -f "/root/.cache/whisper.cpp/$MODEL" || \
  whisper-cpp --model $MODEL --language en --download-only
find /data/raw/icsi/audio -name '*.wav' | while read -r WAV; do
  OUT="${WAV%.wav}.txt"
  whisper-cpp --model $MODEL --language en --output-txt "$WAV" --output-file "$OUT"
done
