#!/bin/bash

# Just re-confirm ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg not found."
else
    echo "ffmpeg is installed."
fi

# Run vosk-transcriber (ensure the audio and model files exist)
# Replace the following with your actual input file paths and model names
# Example: assume you have mounted or copied the model and audio into /app
if [ -f "demovideo.MP4" ]; then
    vosk-transcriber -n vosk-model-en-us-0.42-gigaspeech -i demovideo.MP4 -o test.txt
else
    echo "demovideo.MP4 not found. Skipping transcription."
fi
