"""
Stage 4: Serving — FastAPI inference endpoint for the champion RUL model
registered in the MLflow Model Registry.

Loads whichever model version currently carries the `champion` alias on
`aircraft_engine_rul_model` (set by modeling.py's register_best_model()) via
`models:/aircraft_engine_rul_model@champion` — a serving client never needs
to know a run id or version number, only the alias, so promoting a newly
retrained model to production is a single `set_registered_model_alias` call,
not a code change here.

Exposes:
    GET  /health       liveness + which model version is loaded
    GET  /model_info   registry metadata + expected feature schema
    POST /predict       {"id": ..., "features": {...}} -> predicted_rul
    GET  /metrics       Prometheus exposition format (serving-time metrics)

Serving-time metrics vs. MLflow's training-time metrics are deliberately
different concerns and are NOT wired together: MLflow's val_rmse/val_r2 etc.
describe how the model performed once, offline, at training time. The
Prometheus metrics below describe the live service — request volume, latency,
the distribution of predictions, and (only when a caller supplies
`ground_truth_rul`, as simulate_traffic.py does) live prediction error. That
is the correct signal to alert and dashboard on in production; re-exporting
MLflow's static training metrics into Prometheus would just be a second copy
of a number that never changes after training.

RISK CATEGORIES / PRECISION & RECALL
-------------------------------------
Precision and recall aren't native to a regression problem — there's no
"positive class" for a continuous RUL estimate. They become meaningful once
the RUL is thresholded into a discrete risk category, which is standard
practice for predictive maintenance: keep the regressor for the continuous
estimate, but derive a critical/moderate/safe label from it purely for
monitoring and alerting. Both the prediction and (when known) the ground
truth get bucketed the same way, so "recall on critical" answers a concrete
operational question — of the engines that were actually about to fail, how
many did the model actually flag as critical?

    critical  RUL <= 20
    moderate  20 < RUL < 60
    safe      RUL >= 60

Every prediction updates `rul_engine_predicted_rul{engine_id}` (a Gauge, so
Grafana always reflects each engine's *latest* known status, not a running
count of every historical prediction for it). When ground_truth_rul is also
supplied, `rul_risk_classification_total{predicted_category,actual_category}`
records one cell of a confusion matrix, from which Grafana derives precision/
recall per category via PromQL (see monitoring/README.md).

Run:
    uvicorn serving_api:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from mlflow import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serving_api")

ROOT_DIR = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
REGISTERED_MODEL_NAME = "aircraft_engine_rul_model"
MODEL_ALIAS = "champion"

# ---------------------------------------------------------------------
# Prometheus metrics (scraped by Prometheus, visualized in Grafana)
# ---------------------------------------------------------------------
PREDICTION_REQUESTS = Counter(
    "rul_prediction_requests_total", "Total prediction requests received", ["status"]
)
PREDICTION_LATENCY = Histogram(
    "rul_prediction_latency_seconds", "End-to-end /predict request latency"
)
PREDICTED_RUL = Histogram(
    "rul_predicted_value",
    "Distribution of predicted RUL values (cycles)",
    buckets=(0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150),
)
PREDICTION_ABS_ERROR = Histogram(
    "rul_prediction_absolute_error",
    "abs(predicted_rul - ground_truth_rul), only recorded when a caller supplies ground_truth_rul",
    buckets=(0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100),
)
MODEL_INFO = Gauge(
    "rul_model_info",
    "Static info about the currently loaded champion model (value is always 1; info is in the labels)",
    ["model_name", "model_version", "alias", "run_id"],
)

RISK_CRITICAL_MAX = 20   # RUL <= this  -> critical
RISK_MODERATE_MAX = 60   # RUL < this   -> moderate; RUL >= this -> safe


def _risk_category(rul: float) -> str:
    if rul <= RISK_CRITICAL_MAX:
        return "critical"
    if rul < RISK_MODERATE_MAX:
        return "moderate"
    return "safe"


ENGINE_PREDICTED_RUL = Gauge(
    "rul_engine_predicted_rul",
    "Latest predicted RUL for a given engine id — a Gauge (not a Counter/Histogram) "
    "so it reflects each engine's current status rather than a history of every call",
    ["engine_id"],
)
RISK_CLASSIFICATION = Counter(
    "rul_risk_classification_total",
    "Confusion matrix cell for predicted vs. actual risk category; only recorded when ground_truth_rul is supplied",
    ["predicted_category", "actual_category"],
)


class ModelState:
    model = None
    version: Optional[str] = None
    run_id: Optional[str] = None
    feature_names: Optional[list[str]] = None
    feature_dtypes: Optional[dict[str, str]] = None


state = ModelState()


def _load_champion() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    state.model = mlflow.pyfunc.load_model(model_uri)
    state.version = mv.version
    state.run_id = mv.run_id

    signature = state.model.metadata.get_input_schema()
    if signature is not None:
        state.feature_names = [c.name for c in signature.inputs]
        # The API accepts every feature as JSON float, but the model's logged
        # signature can declare some columns as int64 (e.g. `cycle`, or any
        # sensor whose head(3) input_example happened to be whole numbers) —
        # mlflow's schema enforcement rejects a float64->int64 downcast
        # outright, so each column must be cast to its declared dtype before
        # predict() rather than left as whatever pandas inferred from JSON.
        state.feature_dtypes = {c.name: c.type.to_pandas() for c in signature.inputs}
    else:
        state.feature_names = None
        state.feature_dtypes = None

    MODEL_INFO.labels(
        model_name=REGISTERED_MODEL_NAME, model_version=str(mv.version), alias=MODEL_ALIAS, run_id=mv.run_id
    ).set(1)
    logger.info(
        "Loaded champion model: %s v%s (run_id=%s, %d expected features)",
        REGISTERED_MODEL_NAME, mv.version, mv.run_id, len(state.feature_names or []),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_champion()
    yield


app = FastAPI(title="Aircraft Engine RUL Inference API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    id: int = Field(..., description="Engine id — for correlating responses, not used by the model")
    features: dict[str, float] = Field(
        ..., description="Engineered feature name -> value, matching the engine_engineered_features_* Feast schema"
    )
    ground_truth_rul: Optional[float] = Field(
        None,
        description="Optional true RUL, when known (e.g. simulated traffic against the val split). "
        "Used only to emit a live prediction-error metric — never passed to the model.",
    )


class PredictResponse(BaseModel):
    id: int
    predicted_rul: float
    predicted_risk_category: str
    model_version: str
    latency_ms: float


@app.get("/health")
def health():
    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": state.version, "alias": MODEL_ALIAS}


@app.get("/model_info")
def model_info():
    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": state.version,
        "alias": MODEL_ALIAS,
        "run_id": state.run_id,
        "expected_features": state.feature_names,
        "feature_count": len(state.feature_names) if state.feature_names else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    try:
        if state.feature_names is not None:
            missing = set(state.feature_names) - set(req.features.keys())
            if missing:
                PREDICTION_REQUESTS.labels(status="error").inc()
                raise HTTPException(status_code=422, detail=f"Missing required features: {sorted(missing)}")
            row = pd.DataFrame([{name: req.features[name] for name in state.feature_names}])
            if state.feature_dtypes:
                row = row.astype(state.feature_dtypes)
        else:
            row = pd.DataFrame([req.features])

        predicted_rul = float(state.model.predict(row)[0])
        latency = time.perf_counter() - start
        predicted_category = _risk_category(predicted_rul)

        PREDICTION_REQUESTS.labels(status="success").inc()
        PREDICTION_LATENCY.observe(latency)
        PREDICTED_RUL.observe(predicted_rul)
        ENGINE_PREDICTED_RUL.labels(engine_id=str(req.id)).set(predicted_rul)
        if req.ground_truth_rul is not None:
            PREDICTION_ABS_ERROR.observe(abs(predicted_rul - req.ground_truth_rul))
            actual_category = _risk_category(req.ground_truth_rul)
            RISK_CLASSIFICATION.labels(predicted_category=predicted_category, actual_category=actual_category).inc()

        return PredictResponse(
            id=req.id, predicted_rul=predicted_rul, predicted_risk_category=predicted_category,
            model_version=str(state.version), latency_ms=latency * 1000,
        )
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_REQUESTS.labels(status="error").inc()
        logger.exception("Prediction failed for id=%s", req.id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
