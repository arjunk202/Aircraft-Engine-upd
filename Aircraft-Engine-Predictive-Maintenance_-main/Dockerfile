# ---------------------------------------------------------------------------
# Serving container for the Aircraft Engine RUL inference API.
#
# Builds a self-contained image that runs serving_api.py with uvicorn. The
# image is used by the CD workflow, which builds it on every push to main and
# (optionally) deploys it. LightGBM needs libgomp1 at runtime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime system dependency for LightGBM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code and the MLflow store the serving API loads the champion from.
# (In a cloud deployment the model would instead be pulled from a remote
# registry; here the local mlflow.db + mlruns are copied in for a self-contained
# image.)
COPY serving_api.py .
COPY mlflow.db .
COPY mlruns ./mlruns
COPY feature_repo ./feature_repo

# Cloud Run / container platforms route traffic to $PORT (default 8080).
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn serving_api:app --host 0.0.0.0 --port ${PORT}
