# Use a minimal Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY app_main.py .

# Expose the port the app will run on
EXPOSE 8000

# Start the Uvicorn server
CMD ["uvicorn", "app_main:app", "--host", "0.0.0.0", "--port", "8000"]