# Use official Python slim image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    tar \
    build-essential \
    cmake \
    python3-dev \
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV DATA_URL=https://openslr.org/resources/16/ami_manual_1.6.1.tar.gz \
    DATA_DIR=/app/ami_data

# Download and extract AMI data
RUN mkdir -p ${DATA_DIR} && \
    wget "${DATA_URL}" -O /tmp/ami.tar.gz && \
    tar -xzf /tmp/ami.tar.gz -C ${DATA_DIR} --strip-components=1 && \
    rm /tmp/ami.tar.gz

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Entrypoint to run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
