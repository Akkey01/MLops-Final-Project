FROM python:3.10-slim-buster

WORKDIR /app

COPY ./backend /app/backend  
COPY ./dataIngestion /app/dataIngestion
COPY ./app.py /app/app.py

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install -r /app/backend/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]