brew install pyenv
pyenv install 3.12.3
pyenv local 3.12.3
python3 -m venv venv
source venv/bin/activate

# Clear pip cache
pip cache purge

# Install Vosk and dependencies
pip install --upgrade pip wheel setuptools
pip install vosk

# Install FFMPEG if not already installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg not found. Installing ffmpeg..."
    # For macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    # For Ubuntu/Debian Linux
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    fi
else
    echo "ffmpeg is already installed."
fi

# Run vosk-transcriber
vosk-transcriber -n vosk-model-en-us-0.42-gigaspeech -i audio/youtube.mp4 -o test.txt
