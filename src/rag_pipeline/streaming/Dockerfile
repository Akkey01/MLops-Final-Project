# streaming/Dockerfile
FROM python:3.11-slim

WORKDIR /code

# Install system dependencies and diagnostic tools
RUN apt-get update && apt-get install -y \
    curl \
    iputils-ping \
    dnsutils \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies with specific versions for stability
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    aiohttp==3.9.1 \
    asyncio==3.4.3 \
    aiohttp_retry==2.8.3

# Copy application code
COPY stream_simulator.py .

# Make the script executable
RUN chmod +x stream_simulator.py

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import os; exit(0 if os.path.exists('/mnt/block/stream_logs.jsonl') else 1)"

# Entry point with proper error handling
ENTRYPOINT ["python", "-u", "stream_simulator.py"]