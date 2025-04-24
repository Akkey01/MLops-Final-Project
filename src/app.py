import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dataIngestion.model import getAudioFileFromVideo
from backend.orchestrator import route_inference

app = FastAPI(
    title="MLOPS Final Project (ONNX)",
    description="API for consuming Intelligent Multimedia Processing using ONNX Runtime",
    version="1.0.0"
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
MAX_FILE_SIZE_MB = 50  # Adjusted to match the logic

@app.get("/", status_code=200)
def read_root():
    return {"message": "Server healthcheck!"}

@app.post("/downloadTextAudioFile") 
async def predict(
    useGpu: bool = Form(...), 
    file: UploadFile = File(...)
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")

    # Read bytes and check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")

    try:
        # Call your audio file processor
        output = getAudioFileFromVideo(file)

        # Optionally call orchestrator if needed
        # result = await route_inference(output, useGpu)

        return {"audio_text": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")
