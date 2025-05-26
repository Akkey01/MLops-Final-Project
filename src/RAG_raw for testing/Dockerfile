# 1. Base image
FROM python:3.12-slim

# 2. System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget tar build-essential cmake python3-dev git \
      poppler-utils libpoppler-cpp-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. Workdir
WORKDIR /app

# 4. NLTK_DATA path
ENV NLTK_DATA=/usr/share/nltk_data

# 5. Download & extract AMI data
ENV DATA_URL=https://openslr.org/resources/16/ami_manual_1.6.1.tar.gz \
    DATA_DIR=/app/ami_data
RUN mkdir -p ${DATA_DIR} && \
    wget "${DATA_URL}" -O /tmp/ami.tar.gz && \
    tar -xzf /tmp/ami.tar.gz -C ${DATA_DIR} --strip-components=1 && \
    rm /tmp/ami.tar.gz

# 6. Install Python deps & punkt
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /usr/share/nltk_data && \
    python -m nltk.downloader -d /usr/share/nltk_data punkt

# 7. Copy app code
COPY . .

# 8. Expose ports for UI + metrics
EXPOSE 8501
EXPOSE 8000

# 9. Entrypoint
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
