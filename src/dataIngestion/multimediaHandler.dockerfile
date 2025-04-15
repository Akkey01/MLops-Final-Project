FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY ./multimediaHandler.py .
COPY ./model.py .
COPY ./main.py .
COPY ./demovideo.MP4 .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python","main.py","demovideo.MP4"]