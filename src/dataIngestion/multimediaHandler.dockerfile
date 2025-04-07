FROM python:3.10-alpine3.20

WORKDIR /app

COPY ./multimediaHandler.py .
COPY ./whisper.py .
COPY ./main.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python","main.py"]