import random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

app = FastAPI(title="File Fragment Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== Depthwise Separable Convolution ======
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_ch,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)


# ====== Inception Block (512 optimized) ======
class InceptionBlock512(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b7 = DepthwiseSeparableConv(channels, channels, 7, 3)
        self.b11 = DepthwiseSeparableConv(channels, channels, 11, 5)

        self.pool = nn.MaxPool1d(2, 2)
        self.skip = nn.Conv1d(
            channels, channels,
            kernel_size=1,
            stride=2,
            bias=False
        )

        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.Hardswish()

    def forward(self, x):
        y = self.pool(
            self.act(self.b7(x)) +
            self.act(self.b11(x))
        )
        s = self.skip(x)
        return self.act(self.bn(y + s))


# ====== LFCNN-512 (Improved Model) ======
class LFCNN_512(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(256, 48)

        self.conv1 = nn.Conv1d(48, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.act = nn.Hardswish()

        self.inc1 = InceptionBlock512(64)
        self.inc2 = InceptionBlock512(64)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Conv1d(64, num_classes, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.drop(self.gap(x))
        return self.fc(x).squeeze(-1)


# Class labels (75 classes from FFT-75)
CLASS_LABELS = [
    "jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf",
    "rw2", "3fr", "tiff", "heic", "bmp", "gif", "png", "ai", "eps", "psd",
    "mov", "mp4", "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar", "msi",
    "dmg", "7z", "bz2", "deb", "gz", "pkg", "rar", "rpm", "xz", "zip",
    "exe", "mach-o", "elf", "dll", "doc", "docx", "key", "ppt", "pptx", "xls",
    "xlsx", "djvu", "epub", "mobi", "pdf", "md", "rtf", "txt", "tex", "json",
    "html", "xml", "log", "csv", "aiff", "flac", "m4a", "mp3", "ogg", "wav",
    "wma", "pcap", "ttf", "dwg", "sqlite"
]

NUM_CLASSES = 75
idx_to_class = {i: cls for i, cls in enumerate(CLASS_LABELS)}

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LFCNN_512(num_classes=NUM_CLASSES)

# Load weights - direct state_dict (not wrapped in checkpoint dict)
MODEL_PATH = "lfcnn1_512.pth"
try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"✅ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ Model file {MODEL_PATH} not found. Place it in the same directory.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

model.eval()
model.to(device)


def extract_fragment(file_bytes: bytes) -> list:
    """Extract a 512-byte fragment from file"""
    if len(file_bytes) <= 512:
        fragment = file_bytes
    else:
        # Random offset, avoiding first 512 bytes (header)
        max_offset = len(file_bytes) - 512
        offset = random.randint(min(512, max_offset), max_offset)
        fragment = file_bytes[offset:offset+512]
    
    arr = list(fragment)
    # Pad if needed
    if len(arr) < 512:
        arr = arr + [0] * (512 - len(arr))
    
    return arr


def bytes_to_tensor(byte_list: List[int]) -> torch.Tensor:
    """Convert byte list to model input tensor"""
    arr = np.array(byte_list[:512], dtype=np.uint8)
    if len(arr) < 512:
        arr = np.pad(arr, (0, 512 - len(arr)), 'constant')
    return torch.tensor(arr, dtype=torch.long).unsqueeze(0)


# Request model for byte-based prediction
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


@app.post("/extract-bytes")
async def extract_bytes(file: UploadFile = File(...)):
    """
    Extract 512-byte fragment from uploaded file.
    Returns the byte array for frontend manipulation.
    """
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        fragment = extract_fragment(contents)
        
        return {
            "filename": file.filename,
            "original_size": len(contents),
            "fragment_size": len(fragment),
            "bytes": fragment
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-bytes")
async def predict_bytes(request: BytePredictionRequest):
    """
    Predict file type from manipulated byte array.
    Accepts the byte array directly (after user modifications).
    """
    try:
        if len(request.bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty byte array")
        
        # Validate bytes are in valid range
        if any(b < 0 or b > 255 for b in request.bytes):
            raise HTTPException(status_code=400, detail="Invalid byte values (must be 0-255)")
        
        input_tensor = bytes_to_tensor(request.bytes).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Top 5 predictions
        top_k = 5
        top_indices = probs.argsort()[-top_k:][::-1]
        
        predictions = [
            {
                "class": idx_to_class[int(idx)],
                "confidence": float(probs[idx]),
                "class_id": int(idx)
            }
            for idx in top_indices
        ]
        
        return {
            "filename": request.filename,
            "fragment_size": min(len(request.bytes), 512),
            "predictions": predictions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Keep original predict endpoint for backward compatibility
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        fragment = extract_fragment(contents)
        input_tensor = bytes_to_tensor(fragment).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Top 5 predictions
        top_k = 5
        top_indices = probs.argsort()[-top_k:][::-1]
        
        predictions = [
            {
                "class": idx_to_class[int(idx)],
                "confidence": float(probs[idx]),
                "class_id": int(idx)
            }
            for idx in top_indices
        ]
        
        return {
            "filename": file.filename,
            "file_size": len(contents),
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "device": device}


@app.get("/classes")
def get_classes():
    return {"classes": CLASS_LABELS, "count": NUM_CLASSES}


@app.get("/debug")
def debug():
    # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    png_bytes = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b'\x00' * 504
    fragment = list(png_bytes)
    input_tensor = bytes_to_tensor(fragment).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    top_idx = probs.argsort()[-5:][::-1]
    return {
        "test": "PNG magic bytes",
        "expected": "png (index 16)",
        "predictions": [(idx_to_class[int(i)], float(probs[i])) for i in top_idx]
    }