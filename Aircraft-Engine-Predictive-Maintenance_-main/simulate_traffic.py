"""
Simulates real inference traffic against the serving_api.py endpoint, so the
Prometheus metrics it exposes have something to show in Grafana.

Rows are drawn from the Feast-served val split (engine_engineered_features_val)
rather than test, specifically because val still carries a ground-truth RUL
column (feature_engineering.py only formalizes RUL on train/val — test is the
real FD001 held-out set, where RUL is intentionally unknown except at each
engine's final cycle via PM_truth.csv). Val gives every simulated call a
ground_truth_rul to send along, which serving_api.py uses to emit a live
rul_prediction_absolute_error metric — without that, Grafana would only ever
show request volume/latency, not accuracy drift.

Run (serving_api.py must already be running, e.g. `uvicorn serving_api:app
--host 127.0.0.1 --port 8001`):

    python simulate_traffic.py --requests 150
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from feast import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("simulate_traffic")

ROOT_DIR = Path(__file__).resolve().parent
REPO_PATH = ROOT_DIR / "feature_repo"
NON_FEATURE_COLS = {"id", "event_timestamp", "created_timestamp", "RUL"}  # cycle IS a model feature


def load_val_rows() -> pd.DataFrame:
    """Pulls the val split through Feast's offline store, same as modeling.py."""
    fs = FeatureStore(repo_path=str(REPO_PATH))
    view_name = "engine_engineered_features_val"
    feature_view = fs.get_feature_view(view_name)
    feature_refs = [f"{view_name}:{f.name}" for f in feature_view.features]

    source_path = REPO_PATH / "data" / "engineered_val.parquet"
    spine = pq.read_table(source_path, columns=["id", "event_timestamp"]).to_pandas()

    df = fs.get_historical_features(entity_df=spine, features=feature_refs).to_df()
    logger.info("Loaded %d val rows via Feast offline store", len(df))
    return df


def simulate(api_url: str, n_requests: int, min_delay: float, max_delay: float) -> None:
    df = load_val_rows()
    n_requests = min(n_requests, len(df))
    sample = df.sample(n=n_requests, random_state=random.randint(0, 1_000_000)).reset_index(drop=True)
    feature_cols = [c for c in sample.columns if c not in NON_FEATURE_COLS]

    successes, failures, latencies, abs_errors = 0, 0, [], []

    for i, row in sample.iterrows():
        payload = {
            "id": int(row["id"]),
            "features": {c: float(row[c]) for c in feature_cols},
            "ground_truth_rul": float(row["RUL"]),
        }
        start = time.perf_counter()
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            elapsed = time.perf_counter() - start
            resp.raise_for_status()
            body = resp.json()
            predicted = body["predicted_rul"]
            category = body.get("predicted_risk_category", "?")
            abs_error = abs(predicted - row["RUL"])
            latencies.append(elapsed)
            abs_errors.append(abs_error)
            successes += 1
            logger.info(
                "[%3d/%d] engine=%-4d predicted=%.1f (%-8s) actual=%.1f abs_error=%.1f latency=%.1fms",
                i + 1, n_requests, int(row["id"]), predicted, category, row["RUL"], abs_error, elapsed * 1000,
            )
        except requests.RequestException as exc:
            failures += 1
            logger.warning("[%3d/%d] request failed: %s", i + 1, n_requests, exc)

        time.sleep(random.uniform(min_delay, max_delay))

    logger.info("=" * 60)
    logger.info("Simulation complete: %d succeeded, %d failed", successes, failures)
    if latencies:
        logger.info("Latency  avg=%.1fms  p95=%.1fms", 1000 * sum(latencies) / len(latencies),
                    1000 * sorted(latencies)[int(0.95 * len(latencies)) - 1])
    if abs_errors:
        logger.info("Abs error  avg=%.2f  p95=%.2f", sum(abs_errors) / len(abs_errors),
                    sorted(abs_errors)[int(0.95 * len(abs_errors)) - 1])
    logger.info("Check Grafana / Prometheus / GET %s/metrics for the live view.", api_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8001", help="Base URL of serving_api.py")
    parser.add_argument("--requests", type=int, default=150, help="Number of simulated calls (100-200 suggested)")
    parser.add_argument("--min-delay", type=float, default=0.05, help="Min seconds between requests")
    parser.add_argument("--max-delay", type=float, default=0.4, help="Max seconds between requests")
    args = parser.parse_args()

    simulate(args.api_url, args.requests, args.min_delay, args.max_delay)


if __name__ == "__main__":
    main()
