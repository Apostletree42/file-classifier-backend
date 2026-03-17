import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random


def _load_class_config():
    """Load CLASS_LABELS from classes.json key '1' and CATEGORY_MAP from tags.txt."""
    base = os.path.dirname(os.path.abspath(__file__))

    # Ordered class labels (model training order) from classes.json
    with open(os.path.join(base, "classes.json"), encoding="utf-8") as f:
        classes_data = json.load(f)
    labels = [lbl.upper() for lbl in classes_data["1"]]

    # Label -> category mapping from tags.txt  (format: index:LABEL:Category)
    category_map = {}
    with open(os.path.join(base, "tags.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 2)
            if len(parts) == 3:
                category_map[parts[1].upper()] = parts[2]

    return labels, category_map


# Loaded once at import time
CLASS_LABELS, CATEGORY_MAP = _load_class_config()

# Build reverse lookup: label name -> index in CLASS_LABELS
_LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}

NUM_CLASSES = len(CLASS_LABELS)
BLOCK_SIZE = 512

# ── File signature (magic byte) detection ──────────────────────
_SIGNATURES: list[tuple[int, bytes, str]] = [
    # Images
    (0, b'\x89PNG\r\n\x1a\n', "PNG"),
    (0, b'\xff\xd8\xff', "JPG"),
    (0, b'GIF87a', "GIF"),
    (0, b'GIF89a', "GIF"),
    (0, b'BM', "BMP"),
    (0, b'\x00\x00\x01\x00', "BMP"),
    (0, b'II\x2a\x00', "TIFF"),
    (0, b'MM\x00\x2a', "TIFF"),
    # HEIC (ftyp box)
    (4, b'ftyp', "HEIC"),
    # RAW camera
    (0, b'II\x55\x00', "ARW"),
    # Video / ftyp-based
    (4, b'ftypmp4', "MP4"),
    (4, b'ftypisom', "MP4"),
    (4, b'ftypMSNV', "MP4"),
    (4, b'ftypqt', "MOV"),
    (4, b'ftyp3gp', "3GP"),
    (0, b'\x1a\x45\xdf\xa3', "MKV"),
    (0, b'RIFF', "AVI"),
    (0, b'\x00\x00\x00\x1c\x66\x74\x79\x70', "MP4"),
    # Audio
    (0, b'FORM', "AIFF"),
    (0, b'fLaC', "FLAC"),
    (0, b'\xff\xfb', "MP3"),
    (0, b'\xff\xf3', "MP3"),
    (0, b'\xff\xf2', "MP3"),
    (0, b'ID3', "MP3"),
    (0, b'OggS', "OGG"),
    # Archives
    (0, b'PK\x03\x04', "ZIP"),
    (0, b'Rar!\x1a\x07', "RAR"),
    (0, b'\x1f\x8b', "GZ"),
    (0, b'7z\xbc\xaf\x27\x1c', "7Z"),
    (0, b'BZh', "BZ2"),
    (0, b'\xfd7zXZ\x00', "XZ"),
    # Executables
    (0, b'MZ', "EXE"),
    (0, b'\x7fELF', "ELF"),
    (0, b'\xfe\xed\xfa', "MACH-O"),
    (0, b'\xca\xfe\xba\xbe', "MACH-O"),
    # Office / compound docs
    (0, b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1', "DOC"),
    # PDF
    (0, b'%PDF', "PDF"),
    # Published
    (0, b'AT&TFORM', "DJVU"),
    # Fonts
    (0, b'\x00\x01\x00\x00', "TTF"),
    # Database
    (0, b'SQLite format 3', "SQLITE"),
    # PCAP
    (0, b'\xd4\xc3\xb2\xa1', "PCAP"),
    (0, b'\xa1\xb2\xc3\xd4', "PCAP"),
]


def detect_signature(file_bytes: bytes) -> str | None:
    """Detect file type from magic bytes. Returns label or None."""
    for offset, magic, label in _SIGNATURES:
        end = offset + len(magic)
        if len(file_bytes) >= end and file_bytes[offset:end] == magic:
            if magic == b'RIFF' and len(file_bytes) >= 12:
                sub = file_bytes[8:12]
                if sub == b'AVI ':
                    return "AVI"
                if sub == b'WAVE':
                    return "WAV"
                return "AVI"
            if label == "ZIP" and len(file_bytes) > 30:
                window = file_bytes[:min(len(file_bytes), 8192)]
                if b'AndroidManifest' in window or b'classes.dex' in window:
                    return "APK"
                if b'word/' in window or b'word\\' in window:
                    return "DOCX"
                if b'xl/' in window or b'xl\\' in window:
                    return "XLSX"
                if b'ppt/' in window or b'ppt\\' in window:
                    return "PPTX"
                if b'META-INF/' in window and b'.class' in window:
                    return "JAR"
                if b'mimetype' in window and b'epub' in window:
                    return "EPUB"
            if magic == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                header = file_bytes[:4096] if len(file_bytes) >= 4096 else file_bytes
                if b'PowerPoint' in header or b'P\x00o\x00w\x00e\x00r' in header:
                    return "PPT"
                if b'Excel' in header or b'Workbook' in header:
                    return "XLS"
                return "DOC"
            if magic == b'ftyp':
                ftyp_brand = file_bytes[4:12]
                if b'heic' in ftyp_brand or b'heix' in ftyp_brand or b'mif1' in ftyp_brand:
                    return "HEIC"
                if b'mp4' in ftyp_brand or b'isom' in ftyp_brand:
                    return "MP4"
                if b'qt' in ftyp_brand:
                    return "MOV"
                if b'3gp' in ftyp_brand:
                    return "3GP"
                return "MP4"
            return label

    # Text-based heuristics
    head = file_bytes[:1024]
    try:
        text = head.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        return None
    text_stripped = text.lstrip()
    if text_stripped.startswith('{') or text_stripped.startswith('['):
        return "JSON"
    if text_stripped.startswith('<!') or text_stripped.lower().startswith('<html'):
        return "HTML"
    if text_stripped.startswith('<?xml') or text_stripped.startswith('<'):
        return "XML"
    if text_stripped.startswith('{\\rtf'):
        return "RTF"
    return None


# ── Model definition ──────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)


class InceptionBlock512(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b7 = DepthwiseSeparableConv(channels, channels, 7, 3)
        self.b11 = DepthwiseSeparableConv(channels, channels, 11, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.Hardswish()

    def forward(self, x):
        y = self.pool(self.act(self.b7(x)) + self.act(self.b11(x)))
        s = self.skip(x)
        return self.act(self.bn(y + s))


class LFCNN_512(nn.Module):
    def __init__(self, num_classes=75):
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


_model = None


def load_model():
    global _model
    model_path = os.path.join(os.path.dirname(__file__), "best_lfcnn_512.pth")
    _model = LFCNN_512(num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    _model.load_state_dict(state_dict)
    _model.eval()
    return _model


def get_model():
    if _model is None:
        load_model()
    return _model


def _extract_top5(probs: torch.Tensor) -> list[dict]:
    """Extract top-5 predictions from a probability vector."""
    top5_values, top5_indices = torch.topk(probs, 5)
    results = []
    for val, cls_idx in zip(top5_values.tolist(), top5_indices.tolist()):
        label = CLASS_LABELS[cls_idx]
        results.append({
            "class_name": label,
            "category": CATEGORY_MAP[label],
            "confidence": round(val, 4),
            "class_id": cls_idx,
        })
    return results


def predict_block(block: bytes) -> list[dict]:
    """Predict file type for a single 512-byte block. Returns top-5 predictions."""
    model = get_model()
    block_bytes = block[:BLOCK_SIZE]
    if len(block_bytes) < BLOCK_SIZE:
        block_bytes = block_bytes + b'\x00' * (BLOCK_SIZE - len(block_bytes))

    tensor = torch.tensor(list(block_bytes), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0)
    return _extract_top5(probs)


def predict_file(file_bytes: bytes, block_indices: list[int] | None = None,
                 n_samples: int = 128) -> dict:
    """Predict file type using FFT-75 compatible random-offset sampling."""
    model = get_model()
    file_size = len(file_bytes)
    total_blocks = max(1, file_size // BLOCK_SIZE)

    # --- Sequential block-aligned mode (single / range) ---
    if block_indices is not None:
        indices_to_analyze = [i for i in block_indices if 0 <= i < total_blocks]
        blocks_data = []
        for idx in indices_to_analyze:
            start = idx * BLOCK_SIZE
            block = file_bytes[start:start + BLOCK_SIZE]
            if len(block) < BLOCK_SIZE:
                block = block + b'\x00' * (BLOCK_SIZE - len(block))
            blocks_data.append(list(block))

        if not blocks_data:
            return {
                "total_blocks": total_blocks,
                "analyzed_blocks": 0,
                "sampling_mode": "sequential",
                "aggregate_top5": [],
                "blocks": []
            }

        tensor = torch.tensor(blocks_data, dtype=torch.long)
        with torch.no_grad():
            all_probs = F.softmax(model(tensor), dim=1)

        block_results = []
        for i, idx in enumerate(indices_to_analyze):
            block_results.append({"index": idx, "top5": _extract_top5(all_probs[i])})

        avg_probs = all_probs.mean(dim=0)
        return {
            "total_blocks": total_blocks,
            "analyzed_blocks": len(indices_to_analyze),
            "sampling_mode": "sequential",
            "aggregate_top5": _extract_top5(avg_probs),
            "blocks": block_results
        }

    # --- FFT-75 random-offset sampling mode (default) ---
    max_offset = file_size - BLOCK_SIZE
    if max_offset < 0:
        # File smaller than one block — pad and predict once
        block = file_bytes + b'\x00' * (BLOCK_SIZE - file_size)
        tensor = torch.tensor([list(block)], dtype=torch.long)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze(0)
        sig_label = detect_signature(file_bytes)
        if sig_label and sig_label in _LABEL_TO_IDX:
            sig_one_hot = torch.zeros_like(probs)
            sig_one_hot[_LABEL_TO_IDX[sig_label]] = 1.0
            probs = 0.35 * probs + 0.65 * sig_one_hot
        return {
            "total_blocks": 1,
            "analyzed_blocks": 1,
            "sampling_mode": "random",
            "signature_detected": sig_label,
            "aggregate_top5": _extract_top5(probs),
            "blocks": [{"offset": 0, "top5": _extract_top5(probs)}]
        }

    actual_samples = min(n_samples, max_offset + 1)
    offsets = sorted(random.sample(range(max_offset + 1), actual_samples))

    blocks_data = []
    for off in offsets:
        block = file_bytes[off:off + BLOCK_SIZE]
        blocks_data.append(list(block))

    tensor = torch.tensor(blocks_data, dtype=torch.long)
    with torch.no_grad():
        all_probs = F.softmax(model(tensor), dim=1)

    block_results = []
    for i, off in enumerate(offsets):
        block_results.append({"offset": off, "top5": _extract_top5(all_probs[i])})

    avg_probs = all_probs.mean(dim=0)

    # Combine with file signature detection
    sig_label = detect_signature(file_bytes)
    if sig_label and sig_label in _LABEL_TO_IDX:
        sig_idx = _LABEL_TO_IDX[sig_label]
        sig_one_hot = torch.zeros_like(avg_probs)
        sig_one_hot[sig_idx] = 1.0
        avg_probs = 0.35 * avg_probs + 0.65 * sig_one_hot

    return {
        "total_blocks": total_blocks,
        "analyzed_blocks": actual_samples,
        "sampling_mode": "random",
        "signature_detected": sig_label,
        "aggregate_top5": _extract_top5(avg_probs),
        "blocks": block_results
    }
