from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import io

app = FastAPI(title="File Fragment Classifier")

# CORS for Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Architecture (copy from your notebook)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding):
        super().__init__()
        self.depth = nn.Conv1d(
            in_ch, in_ch, k,
            padding=padding,
            groups=in_ch,
            bias=False
        )
        self.point = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return self.bn(x)


class InceptionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b11 = DepthwiseSeparableConv(ch, ch, 11, 5)
        self.b19 = DepthwiseSeparableConv(ch, ch, 19, 9)
        self.b27 = DepthwiseSeparableConv(ch, ch, 27, 13)

        self.pool = nn.MaxPool1d(4)
        self.conv1x1 = nn.Conv1d(ch, ch, 1, stride=4)

    def forward(self, x):
        a = F.relu(self.b11(x))
        b = F.relu(self.b19(x))
        c = F.relu(self.b27(x))
        y = a + b + c
        y = self.pool(y)
        s = self.conv1x1(x)
        return F.relu(y + s)


class LFCNN1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.embed = nn.Embedding(256, 32)
        self.conv = nn.Conv1d(32, 64, 3, padding=1)
        self.act = nn.Hardswish()

        self.inc1 = InceptionBlock(64)
        self.inc2 = InceptionBlock(64)
        self.inc3 = InceptionBlock(64)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(64, num_classes, 1)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.conv(x))
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)
        x = self.gap(x)
        x = self.fc(x)
        return x.squeeze(-1)


# Class labels (you need to provide these)
CLASS_LABELS = [f"Class_{i}" for i in range(75)]  # REPLACE with actual labels

# Load model
device = 'cpu'  # Railway uses CPU
model = LFCNN1(num_classes=75)
checkpoint = torch.load('lfcnn1_512.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)


def preprocess_file(file_bytes: bytes) -> torch.Tensor:
    """Convert file bytes to model input format"""
    # Take first 512 bytes
    fragment = np.frombuffer(file_bytes[:512], dtype=np.uint8)
    
    # Pad if less than 512 bytes
    if len(fragment) < 512:
        fragment = np.pad(fragment, (0, 512 - len(fragment)), 'constant')
    
    # Convert to tensor
    tensor = torch.tensor(fragment, dtype=torch.long).unsqueeze(0)
    return tensor


@app.get("/")
def root():
    return {"status": "File Fragment Classifier API", "model": "LFCNN1"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Preprocess
        input_tensor = preprocess_file(contents).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
        
        # Get top 3 predictions
        top_k = 3
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = [
            {
                "class": CLASS_LABELS[idx.item()],
                "confidence": prob.item(),
                "class_id": idx.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            "filename": file.filename,
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}