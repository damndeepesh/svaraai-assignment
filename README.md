## SvaraAI Reply Classification – How to Run

This project classifies email replies into three categories: negative, neutral, positive. It includes:
- A training notebook to build and compare models
- A FastAPI service to serve predictions at /predict

### What you need
- macOS or Linux (Windows WSL works too)
- Python 3.11 or newer
- Git (optional)
- Docker (optional, only if you want to use the container)

### Project files
- `reply_classification_dataset.csv` – training data (already included)
- `reply_classification_pipeline.ipynb` – notebook to train/evaluate models
- `bert_out/` – saved checkpoints from DistilBERT fine-tuning (created after training)
- `app.py` – FastAPI service
- `requirements.txt` – Python dependencies
- `Dockerfile` – container setup

---

[Google Colab](https://colab.research.google.com/drive/1ZqBbRr7giVhSEvZUH0F-2Ea-fpo7P7l6?usp=sharing)


## Option 1: Run locally (recommended)

### 1) Create and activate a virtual environment
From the project folder:
```bash
cd /Users/damndeepesh/Documents/internship/SvaraAI
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

### 3) (Optional) Train models in the notebook
Open the notebook and run all cells:
```bash
open reply_classification_pipeline.ipynb
```
Notes:
- The notebook loads `reply_classification_dataset.csv` and cleans text/labels.
- It trains baseline models (TF‑IDF + Logistic Regression, LinearSVC, etc., plus LightGBM), then fine‑tunes DistilBERT.
- Checkpoints are saved under `bert_out/`. The API will load the latest checkpoint automatically.

### 4) Start the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Optional: control confidence softness with temperature (higher = softer probabilities):
```bash
TEMPERATURE=2.0 uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5) Test the API
Health check:
```bash
curl -s http://localhost:8000/ | jq
```
Prediction:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Let’s schedule a demo next week."}' | jq
```
You should see a JSON response like:
```json
{"label":"positive","confidence":0.87}
```

---

## Option 2: Run with Docker

### 1) Build the image
```bash
cd /Users/damndeepesh/Documents/internship/SvaraAI
docker build -t svara-reply-api:latest .
```

### 2) Run the container
```bash
docker run --rm -p 8000:8000 -e TEMPERATURE=1.8 svara-reply-api:latest
```

### 3) Test the API (same as above)
```bash
curl -s http://localhost:8000/ | jq
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"Let’s schedule a demo next week."}' | jq
```

---

## Tips and troubleshooting
- Externally managed Python (PEP 668) error: use a virtual environment as shown above.
- LightGBM on macOS: Dockerfile already installs `libgomp1` for OpenMP. Locally, if you hit build errors, try `brew install libomp` then `pip install lightgbm`.
- PyTorch: If install fails, try the official CPU wheels: `python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Confidence too high on borderline text: increase `TEMPERATURE` (e.g., 2.0–2.5) when running the API to soften probabilities.


## What the API returns
- Endpoint: `POST /predict`
- Input JSON: `{ "text": "Some email reply" }`
- Output JSON: `{ "label": "negative|neutral|positive", "confidence": 0.00-1.00 }`

The API loads the latest model from `bert_out/checkpoint-*`. If no checkpoint is found, it falls back to `distilbert-base-uncased`.
