# Use the official Python image with slim buster
FROM python:3.12-slim-buster

# 1. Base image
FROM python:3.11-slim

# 2. Install system dependencies (wget, tar) 
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget tar && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working dir
WORKDIR /app

# 4. Build args & env for data URL and data dir
ARG DATA_URL=https://openslr.org/resources/16/ami_manual_1.6.1.tar.gz
ENV DATA_URL=${DATA_URL}
ENV DATA_DIR=/app/ami_data

# 5. Download & extract AMI data
RUN mkdir -p ${DATA_DIR} \
    && wget "${DATA_URL}" -O /tmp/ami.tar.gz \
    && tar -xzf /tmp/ami.tar.gz -C ${DATA_DIR} --strip-components=1 \
    && rm /tmp/ami.tar.gz

# 6. Copy requirements & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code
COPY . .

# 8. Expose port & entrypoint
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
