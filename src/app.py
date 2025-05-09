import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
#from backend.orchestrator import route_inference
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pathlib import Path

app = FastAPI(
    title="MLOPS Final Project (ONNX)",
    description="API for consuming Intelligent Multimedia Processing using ONNX Runtime",
    version="1.0.0"
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
MAX_FILE_SIZE_MB = 50  # Adjusted to match the logic
MODEL_PATH = Path("./backend/tinyllama11b_chat_ft1")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH.as_posix(), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH.as_posix(), local_files_only=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Request body schema
class Prompt(BaseModel):
    text: str
    max_new_tokens: int = 50

# Health check
@app.get("/")
def read_root():
    return {"status": "API is up and running!"}

# Inference route
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
# @app.post("/downloadTextAudioFile") 
# async def predict(
#     useGpu: bool = Form(...), 
#     file: UploadFile = File(...)
# ):
#     # Validate file extension
#     ext = os.path.splitext(file.filename)[-1].lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")

#     # Read bytes and check file size
#     contents = await file.read()
#     if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
#         raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")

#     try:

#         # Call your audio file processor
#         output = await route_inference(useGpu, file)

#         # Optionally call orchestrator if needed
#         # result = await route_inference(output, useGpu)

#         return {"audio_text": output}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")
