# Dockerfile

# 1. Base image: Python slim
FROM python:3.12-slim

# 2. Install system dependencies for building extensions & PDF parsing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      tar \
      build-essential \
      cmake \
      python3-dev \
      git \
      poppler-utils \
      libpoppler-cpp-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 5. Download & extract AMI data
ENV DATA_URL=https://openslr.org/resources/16/ami_manual_1.6.1.tar.gz \
    DATA_DIR=/app/ami_data
RUN mkdir -p ${DATA_DIR} && \
    wget "${DATA_URL}" -O /tmp/ami.tar.gz && \
    tar -xzf /tmp/ami.tar.gz -C ${DATA_DIR} --strip-components=1 && \
    rm /tmp/ami.tar.gz

# 6. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt

# 7. Copy your app code and any other files
COPY . .

# 8. Expose Streamlit port
EXPOSE 8501

# 9. Entrypoint: run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
