FROM python:3.10-alpine3.20

WORKDIR /app

RUN apk add --no-cache git

RUN pip install --quiet git+https://github.com/openai/whisper.git
