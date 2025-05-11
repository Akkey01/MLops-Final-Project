FROM python:3.10-slim-buster

WORKDIR /app

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y git unzip wget && \
    pip install gdown && \
    rm -rf /var/lib/apt/lists/*

# Copy your app files (excluding model zip initially)
COPY ./backend /app/backend
COPY ./app.py /app/app.py

# Download and extract model zip to a temp location
RUN gdown "https://drive.google.com/uc?id=16gr9ItkcsAZNRJPDBnmHf2N_7DFyu2Gl" -O /tmp/model.zip && \
    unzip -o /tmp/model.zip -d /tmp/ && \
    rm /tmp/model.zip

# Move extracted model to target path
RUN mkdir -p /app/backend/tinyllama11b_chat_ft1 && \
    mv /tmp/tinyllama11b_chat_ft1/* /app/backend/tinyllama11b_chat_ft1/ && \
    rm -rf /tmp/__MACOSX /tmp/tinyllama11b_chat_ft1

# Install Python dependencies
RUN pip install -r /app/backend/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]