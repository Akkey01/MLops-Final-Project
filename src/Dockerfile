# ───────── Dockerfile ─────────────────────────────────────────────────────────
FROM python:3.10-slim-buster

# 1) Install system dependencies for:
#    • building C/C++ extensions (llama‑cpp‑python, etc.)
#    • PDF parsing (pdfplumber → poppler-utils)
#    • XML/HTML parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    cmake \
    pkg-config \
    git \
    wget \
    poppler-utils \
    libpoppler-cpp-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy & install Python dependencies
COPY ./backend/requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# 3) Copy your ingestion code & application entrypoint
COPY ./dataIngestion     ./dataIngestion
COPY ./app.py            .

# 4) Download & extract AMI corpus from OpenSLR into /app/ami_data
ARG DATA_URL=https://openslr.org/resources/16/ami_manual_1.6.1.tar.gz
ENV DATA_URL=${DATA_URL}
ENV DATA_DIR=/app/ami_data
RUN mkdir -p ${DATA_DIR} \
 && wget "${DATA_URL}" -O /tmp/ami.tar.gz \
 && tar -xzf /tmp/ami.tar.gz -C ${DATA_DIR} --strip-components=1 \
 && rm /tmp/ami.tar.gz

# 5) Expose the port your app listens on
EXPOSE 8000

# 6) Run via Uvicorn (for FastAPI) or Streamlit (if you renamed entrypoint)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
