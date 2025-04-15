FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --quiet git+https://github.com/openai/whisper.git
