FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system and Python dependencies
RUN apt-get update && \
    apt-get install -y git wget unzip && \
    pip install torch transformers onnx onnxruntime numpy && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install optimum[onnxruntime]

# Copy all files into container
COPY export_to_onnx.py /app/backend/export_to_onnx.py
COPY benchmark_onnx.py /app/backend/benchmark_onnx.py
COPY tinyllama11b_chat_ft1 /app/backend/tinyllama11b_chat_ft1

# Run export and benchmark sequentially
CMD ["sh", "-c", "cd backend && python export_to_onnx.py && python benchmark_onnx.py"]

