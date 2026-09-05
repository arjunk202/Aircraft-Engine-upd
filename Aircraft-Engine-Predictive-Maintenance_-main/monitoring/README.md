# Serving monitoring stack (Prometheus + Grafana)

Answers "do we need Prometheus for Grafana?" — **yes**. Grafana has no
built-in ability to scrape an application's `/metrics` endpoint itself; it
only *queries* a datasource. Prometheus is that datasource here: it scrapes
`serving_api.py`'s `/metrics` endpoint on a timer and stores the time series,
and Grafana queries Prometheus (PromQL) to draw the dashboard.

MLflow's tracking metrics (`val_rmse`, `val_r2`, ...) are **not** pushed into
Prometheus, and don't need to be — they're one-off training-time numbers
recorded once per run. Prometheus/Grafana here monitor the live *serving*
path instead: request rate, latency, the distribution of predictions, and
(since `simulate_traffic.py` supplies `ground_truth_rul` from the val split)
live prediction error. That's the metric surface that actually changes while
the service is running, which is what Prometheus/Grafana are for.

```
serving_api.py (FastAPI, port 8001)
    -> exposes /metrics (Prometheus exposition format, via prometheus_client)
Prometheus (docker, port 9090)
    -> scrapes serving_api.py:/metrics every 5s
Grafana (docker, port 3000)
    -> queries Prometheus, renders the provisioned "RUL Model Serving" dashboard
```

## 1. Start the model server (on the host, not in Docker)

```
cd ..
uvicorn serving_api:app --host 127.0.0.1 --port 8001
```

Verify: `curl http://127.0.0.1:8001/health` and `curl http://127.0.0.1:8001/metrics`.

## 2. Start Prometheus + Grafana

Requires Docker Desktop running.

```
cd monitoring
docker compose up -d
```

- Prometheus: http://localhost:9090 — check **Status > Targets**, the
  `rul_serving_api` job should show as `UP`. It reaches the host API via
  `host.docker.internal:8001` (Docker Desktop's route from a container back
  to the Windows host), not `localhost`, because `localhost` inside the
  Prometheus container means the container itself.
- Grafana: http://localhost:3000 (anonymous viewer access is enabled; login
  `admin`/`admin` if you need to edit). The **"Aircraft Engine RUL - Model
  Serving"** dashboard is auto-provisioned from
  `grafana/dashboards/rul_serving_dashboard.json` — no manual datasource or
  import steps needed.

## 3. Generate traffic to visualize

```
cd ..
python simulate_traffic.py --requests 150
```

Pulls 150 rows from the Feast-served val split, POSTs each to `/predict`
with a small randomized delay between calls (so the dashboard shows a spread
of points over time rather than one instant spike), and logs a running
summary (latency, prediction error) to the console as it goes. Watch the
Grafana dashboard update in real time (5s refresh).

## Dashboard panels

- Request rate (success vs. error)
- Prediction latency (p50/p95/p99)
- Predicted RUL distribution (median/p95)
- Prediction absolute error vs. ground truth (median/p95) — only populated
  when callers send `ground_truth_rul`, which `simulate_traffic.py` does
- Total requests / error rate (last 30m)
- Loaded champion model (name, version, run id — from `rul_model_info`)

## Tear down

```
cd monitoring
docker compose down
```
