import os
import time
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Model version for canary/stable labeling
MODEL_VERSION = os.getenv("MODEL_VERSION", "stable")

# Prometheus metrics with version labeling
REQUEST_COUNTER = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "endpoint", "http_status", "version"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency in seconds",
    ["method", "endpoint", "version"]
)

app = FastAPI(
    title="MLOPS Final Project (ONNX)",
    description="API for consuming Intelligent Multimedia Processing using ONNX Runtime",
    version="1.0.0"
)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    method = request.method
    endpoint = request.url.path
    status = str(response.status_code)

    REQUEST_COUNTER.labels(method, endpoint, status, MODEL_VERSION).inc()
    REQUEST_LATENCY.labels(method, endpoint, MODEL_VERSION).observe(duration)

    return response

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
MAX_FILE_SIZE_MB = 50
MODEL_PATH = Path("./backend/tinyllama11b_chat_ft1")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH.as_posix(), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH.as_posix(), local_files_only=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

class Prompt(BaseModel):
    text: str
    max_new_tokens: int = 50

@app.get("/")
def read_root():
    return {"status": "API is up and running!"}

@app.post("/generate")
def generate_text(prompt: Prompt):
    try:
        response = generator(
            prompt.text,
            max_new_tokens=prompt.max_new_tokens,
            do_sample=True,
            temperature=0.7
        )
        return {"output": response[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/version")
def get_model_version():
    return {"model_version": MODEL_VERSION}
