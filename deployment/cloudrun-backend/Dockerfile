FROM python:3.13-slim

# Faster, quieter pip; no .pyc clutter
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM needs libgomp at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first so this layer caches across code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# App + model artifacts (model.skops is ~1.4MB)
COPY app.py champion_features.json model.skops ./

# Dashboard (served at / by app.py)
COPY static ./static

# Cloud Run sends traffic to $PORT (defaults to 8080). Bind to it.
ENV PORT=8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
