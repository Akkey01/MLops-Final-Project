#!/usr/bin/env python3
import os
import time
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Audio→RAG Inference",
    description="Wraps Whisper (optional) and your RAG model endpoint",
)


RAG_URL     = os.getenv("RAG_ENDPOINT_URL")        

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an audio file, transcribes via Whisper (if configured),
    then calls your RAG model and returns the JSON response.
    """
    data = await file.read()

    # 1) Transcribe
    if WHISPER_URL:
        try:
            r_wh = requests.post(WHISPER_URL, files={"file": (file.filename, data)})
            r_wh.raise_for_status()
            text = r_wh.json().get("text", "")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper error: {e}")
    else:
        # dummy fallback
        text = f"[DUMMY] {file.filename}"

    # 2) Query RAG
    if not RAG_URL:
        raise HTTPException(status_code=500, detail="RAG_ENDPOINT_URL not set")
    try:
        r_rag = requests.post(RAG_URL, json={"query": text})
        r_rag.raise_for_status()
        answer = r_rag.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return JSONResponse(content={
        "query": text,
        "answer": answer
    })
