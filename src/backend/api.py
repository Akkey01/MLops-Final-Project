from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import base64
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI(
    title="MLOPS Final Project (ONNX)",
    description="API for consuming Intelligent Multimedia Processing using ONNX Runtime",
    version="1.0.0"
)

# Remove BaseModel for file upload and use FastAPI's UploadFile directly in the endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Example transform (modify as needed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).numpy()

    # Run inference (example assumes a session is already created)
    outputs = session.run(None, {"input": input_tensor})

    return {"outputs": outputs}
