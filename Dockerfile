FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# LightGBM runtime needs OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV TEMPERATURE=1.8
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
