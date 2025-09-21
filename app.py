import os
import glob
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "neutral", "positive"]
CHECKPOINT_DIR = "bert_out"

# Temperature for probability calibration (softmax overconfidence reducer)
# Set via env: TEMPERATURE=2.0 uvicorn app:app --port 8000
try:
    TEMPERATURE = float(os.getenv("TEMPERATURE", "1.8"))
    if TEMPERATURE < 1.0:
        TEMPERATURE = 1.0
except Exception:
    TEMPERATURE = 1.8

app = FastAPI(title="SvaraAI Reply Classifier API")

def get_latest_checkpoint(dir_path: str) -> Optional[str]:
    if not os.path.isdir(dir_path):
        return None
    checkpoints = sorted(glob.glob(os.path.join(dir_path, "checkpoint-*")))
    if not checkpoints:
        return dir_path if os.path.exists(os.path.join(dir_path, "config.json")) else None
    return checkpoints[-1]

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

@app.on_event("startup")
def load_model():
    global tokenizer, model
    ckpt = get_latest_checkpoint(CHECKPOINT_DIR) or "distilbert-base-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {ckpt}: {e}")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' must be a non-empty string")
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model(**inputs)
        # Temperature-scaled softmax to reduce overconfidence
        logits = outputs.logits / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)[0]
        conf, pred_id = torch.max(probs, dim=-1)
        label = LABELS[pred_id.item()] if pred_id.item() < len(LABELS) else str(pred_id.item())
        return PredictResponse(label=label, confidence=float(conf.item()))

@app.get("/")
def health():
    return {"status": "ok", "temperature": TEMPERATURE}
