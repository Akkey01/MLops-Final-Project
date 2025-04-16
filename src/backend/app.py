from fastapi import FastAPI, UploadFile, File
from dataIngestion.model import getAudioFileFromVideo

app = FastAPI(
    title="MLOPS Final Project (ONNX)",
    description="API for consuming Intelligent Multimedia Processing using ONNX Runtime",
    version="1.0.0"
)

# Remove BaseModel for file upload and use FastAPI's UploadFile directly in the endpoint
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
MAX_FILE_SIZE_MB = 50

@app.post("/downloadTextAudioFile")
async def predict(file: UploadFile = File(...)):
    # Validate file extension
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")

    # Read bytes and check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    try:
        output = getAudioFileFromVideo(file)

        return {"Audio File": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")

    
