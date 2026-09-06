"""
Slim RUL serving API for Cloud Run.

Loads the champion LightGBM model straight from model.skops (no MLflow, no
mlruns) and serves predictions. Behaviourally identical to the team's
serving_api.py for the endpoints the dashboard uses, but with zero MLflow
dependency so the container stays tiny and cold-starts fast.

The champion here is aircraft_engine_rul_model v4 (LightGBM), val RMSE 15.55.
It expects EXACTLY the 67 features in champion_features.json, in that order.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import skops.io as sio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.skops"
FEATURES_PATH = ROOT / "champion_features.json"

MODEL_VERSION = "4"          # matches the MLflow champion alias -> v4
MODEL_NAME = "aircraft_engine_rul_model"

# Risk thresholds — identical to the team's serving_api.py
RISK_CRITICAL_MAX = 20       # RUL <= 20 -> critical
RISK_MODERATE_MAX = 60       # RUL < 60  -> moderate; RUL >= 60 -> safe

# Columns the model signature declares as integer
INT_COLS = {"cycle", "s17"}


def _risk_category(rul: float) -> str:
    if rul <= RISK_CRITICAL_MAX:
        return "critical"
    if rul < RISK_MODERATE_MAX:
        return "moderate"
    return "safe"


FEATURE_NAMES: list[str] = json.loads(FEATURES_PATH.read_text())

# skops needs the trusted types declared; get them from the file itself.
_untrusted = sio.get_untrusted_types(file=str(MODEL_PATH))
MODEL = sio.load(str(MODEL_PATH), trusted=_untrusted)

app = FastAPI(title="Aircraft Engine RUL Inference API (slim)", version="1.0")

# CORS so the Netlify frontend (different origin) can call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    id: int = Field(..., description="Engine id — for correlating responses")
    features: dict[str, float] = Field(..., description="67 engineered features")
    ground_truth_rul: Optional[float] = Field(None, description="Optional true RUL")


class PredictResponse(BaseModel):
    id: int
    predicted_rul: float
    predicted_risk_category: str
    model_version: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "alias": "champion"}


@app.get("/model_info")
def model_info():
    return {
        "registered_model_name": MODEL_NAME,  # dashboard reads this for the Model banner
        "version": MODEL_VERSION,
        "alias": "champion",
        "expected_features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    missing = set(FEATURE_NAMES) - set(req.features.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {sorted(missing)}",
        )
    # Build the row in the exact training order, cast int columns.
    row = pd.DataFrame([{name: req.features[name] for name in FEATURE_NAMES}])
    for c in INT_COLS:
        if c in row.columns:
            row[c] = row[c].astype("int64")

    predicted = float(MODEL.predict(row)[0])
    predicted = max(0.0, min(predicted, 125.0))   # clip like training
    latency = (time.perf_counter() - start) * 1000

    return PredictResponse(
        id=req.id,
        predicted_rul=predicted,
        predicted_risk_category=_risk_category(predicted),
        model_version=MODEL_VERSION,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------
# Serve the dashboard (static files) at the site root.
# Mounted LAST so the API routes above (/health, /model_info, /predict)
# take priority; everything else ("/", "/fleet-data.js", ...) is served
# from ./static. The dashboard's API_BASE is "" so it calls these same
# endpoints on its own origin — one URL serves page + model.
# ---------------------------------------------------------------------
STATIC_DIR = ROOT / "static"
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
