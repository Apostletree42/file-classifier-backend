import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

try:
    from .model import load_model, predict_file, predict_block, CLASS_LABELS, NUM_CLASSES, BLOCK_SIZE
except ImportError:
    from model import load_model, predict_file, predict_block, CLASS_LABELS, NUM_CLASSES, BLOCK_SIZE

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="LFCNN File Fragment Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BytePredictionRequest(BaseModel):
    bytes: List[int]
    filename: str


@app.get("/")
def root():
    return {
        "status": "File Fragment Classifier API",
        "model": "LFCNN-512",
        "classes": NUM_CLASSES
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "LFCNN-512", "classes": NUM_CLASSES}


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_LABELS, "count": NUM_CLASSES}


@app.post("/extract-bytes")
async def extract_bytes(file: UploadFile = File(...)):
    """Extract a 512-byte fragment from the uploaded file for frontend editing."""
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")

    file_size = len(contents)
    if file_size <= BLOCK_SIZE:
        fragment = list(contents) + [0] * (BLOCK_SIZE - file_size)
        offset = 0
    else:
        max_offset = file_size - BLOCK_SIZE
        offset = random.randint(0, max_offset)
        fragment = list(contents[offset:offset + BLOCK_SIZE])

    return {
        "filename": file.filename,
        "original_size": file_size,
        "fragment_size": len(fragment),
        "offset": offset,
        "bytes": fragment,
    }


@app.post("/predict-bytes")
async def predict_bytes(request: BytePredictionRequest):
    """Predict file type from a (potentially edited) byte array."""
    if len(request.bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty byte array")
    if any(b < 0 or b > 255 for b in request.bytes):
        raise HTTPException(status_code=400, detail="Invalid byte values (must be 0-255)")

    # Treat the submitted bytes as the entire file content and run full prediction
    file_bytes = bytes(request.bytes[:BLOCK_SIZE])
    result = predict_file(file_bytes)

    predictions = [
        {
            "class": p["class_name"],
            "confidence": p["confidence"],
            "class_id": p["class_id"],
        }
        for p in result["aggregate_top5"]
    ]

    return {
        "filename": request.filename,
        "fragment_size": len(request.bytes),
        "predictions": predictions,
    }


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
):
    """Full-file prediction endpoint (same as majorProjectB interface)."""
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 100MB limit")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    result = predict_file(file_bytes)
    return {
        "filename": file.filename,
        "file_size": len(file_bytes),
        **result,
    }


@app.get("/debug")
def debug():
    """Test endpoint: predict PNG magic bytes."""
    png_bytes = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b'\x00' * 504
    top5 = predict_block(png_bytes)
    return {
        "test": "PNG magic bytes",
        "expected": "PNG (index 16)",
        "predictions": [(p["class_name"], p["confidence"]) for p in top5],
    }