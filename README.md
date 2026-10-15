# Aircraft Engine Predictive Maintenance — RUL MLOps Pipeline

Predicts Remaining Useful Life (RUL) for turbofan engines from the NASA
C-MAPSS FD001 dataset, built end-to-end as an MLOps pipeline following a
formal Data Science Process Model (DSPM — see `Materials/DSPM_material.pdf`):
data validation → EDA → feature engineering → a Feast feature store →
MLflow-tracked model training/registry → a FastAPI serving endpoint →
Prometheus/Grafana monitoring.

Every stage is a plain, independently-runnable Python script that reads the
previous stage's artifacts and writes its own — nothing is a notebook-only
one-off. `simulate_pipeline.py` runs the whole thing end-to-end in one command.

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running everything (quick path)](#running-everything-quick-path)
- [Running stage-by-stage](#running-stage-by-stage)
- [Serving + monitoring](#serving--monitoring)
- [Moving this project to another machine](#moving-this-project-to-another-machine)
- [Troubleshooting](#troubleshooting)

## Architecture

```
PM_train.csv, PM_test.csv, PM_truth.csv   (raw NASA C-MAPSS FD001 data)
        |
        v
data_validation.py        Stage 2a  -> Data_Report_Artifacts/data_validation/
        |                             (validated, leakage-free engine-wise train/val/test split)
        v
eda.py                     Stage 2b  -> Data_Report_Artifacts/eda/
        |                             (train-only exploration; drives feature_engineering.py's decisions)
        v
feature_engineering.py     Stage 2c  -> Data_Report_Artifacts/feature_engineering/
        |                             (drop constant/redundant cols, Yeo-Johnson, rolling stats, clip RUL)
        v
build_feature_store_sources.py  Stage 2d  -> feature_repo/data/*.parquet
        |                             (bridges the above into Feast-ingestible sources)
        v
feast apply
feast materialize-incremental   Stage 2e  -> feature_repo/data/registry.db, online_store.db
        |                             (Feast feature store: offline point-in-time + online retrieval)
        v
modeling.py                Stage 3   -> mlflow.db, mlruns/
        |                             (5 regressors trained via Feast; best one registered as "champion"
        |                              in the MLflow Model Registry)
        v
serving_api.py              Stage 4   FastAPI inference endpoint, loads the "champion" model,
        |                             exposes /predict + Prometheus /metrics
        v
simulate_traffic.py         Stage 6   replays the val split through /predict to generate live traffic
        |
        v
Prometheus + Grafana        monitoring/   scrapes /metrics, renders the "Aircraft Engine RUL" dashboard
```

Design choices worth knowing before you read the code:

- **Nothing later re-derives the train/val/test split.** `data_validation.py`
  locks it in once (`engine_split_manifest.json`); every later stage reads
  the same partition so there is no risk of a different split sneaking in.
- **EDA only looks at TRAIN.** The validation partition is held out even at
  the exploration stage, so no downstream decision (which sensors look
  redundant, which are skewed, ...) can leak information from held-out engines.
- **Feature engineering decisions are read from EDA's report, not
  hardcoded** — re-running `eda.py` on different data changes what
  `feature_engineering.py` does automatically.
- **`modeling.py` reads training data through Feast's offline store**
  (`get_historical_features`), not `pd.read_csv`, so the feature store is
  provably the single source of truth for what a model was trained on.
- **Every registered model (Linear Regression, Random Forest, XGBoost,
  LightGBM, HistGradientBoosting) accepts an identical raw-feature schema** —
  Linear Regression bundles its `StandardScaler` into the same
  `sklearn.Pipeline` that gets logged, so `serving_api.py` never needs to
  special-case scaling per model type regardless of which one is "champion".
- **The MLflow `champion` alias, not a hardcoded run id/version**, is what
  `serving_api.py` loads (`models:/aircraft_engine_rul_model@champion`) —
  promoting a newly retrained model to production is one
  `set_registered_model_alias` call, not a code change.

## Repository layout

```
PM_train.csv / PM_test.csv / PM_truth.csv   Raw NASA C-MAPSS FD001 data (source of truth — never edit)
data_cleaning_notebook.ipynb                Early exploratory notebook (superseded by the scripts below)
Data_modeling.ipynb                         Early modeling notebook (superseded by modeling.py)
Materials/                                  DSPM reference material (UML diagrams, spec docs)
MLOPS_PIPELINE/                             Legacy pre-Feast/MLflow pipeline snapshot — kept for reference,
                                             nothing in the current pipeline reads or writes it

data_validation.py                          Stage 2a
eda.py                                      Stage 2b
feature_engineering.py                      Stage 2c
build_feature_store_sources.py              Stage 2d
feature_repo/                               Feast feature repository (Stage 2e) — see feature_repo/README.md
modeling.py                                 Stage 3
serving_api.py                              Stage 4 — FastAPI inference endpoint
simulate_traffic.py                         Stage 6 — traffic simulator for demo/monitoring
monitoring/                                 Prometheus + Grafana stack — see monitoring/README.md

Data_Report_Artifacts/                      Generated — every stage's reports/plots/split files
mlflow.db, mlruns/                          Generated — MLflow tracking store + artifact store
feature_repo/data/*.parquet, registry.db,
  online_store.db                           Generated — Feast sources + registry + online store

simulate_pipeline.py                        Runs Stages 2a-6 end-to-end in one command
fix_mlflow_paths.py                         One-time fix if this project folder was moved/copied (see below)
run_mlflow_ui.py                            Workaround launcher for `mlflow ui` (see Troubleshooting)
requirements.txt                            Pinned/unpinned Python dependencies (see Setup)
```

Everything under "Generated" is a build artifact: safe to delete and
regenerate by re-running the relevant stage(s), never hand-edited.

## Setup

Requires Python 3.12+ and, for the monitoring stack, Docker Desktop.

```
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

`requirements.txt` pins `mlflow==3.15.1` and `protobuf>=7.35.1,<8`
deliberately — see [Troubleshooting](#troubleshooting) for why.

## Running everything (quick path)

`simulate_pipeline.py` runs the full pipeline (Stages 2a through 6,
including starting `serving_api.py`, health-checking it, and replaying 150
requests through it) in one command, writing every stage's console output to
`ci_logs/`.

It does **not** create a venv or install dependencies for you — activate
yours first (see Setup above). Run it from the folder that **contains** this
project folder (i.e. its parent), not from inside it:

```
venv\Scripts\activate
cd ..
python Aircraft-Engine-Predictive-Maintenance_-main\simulate_pipeline.py --clean
```

- `--clean` deletes previously generated artifacts first, so the run is a
  true from-scratch reproduction instead of silently reusing whatever's
  already sitting in `Data_Report_Artifacts/`, `mlruns/`, etc.
- `--with-monitoring` additionally brings up Prometheus + Grafana via
  `docker compose` once the pipeline succeeds (requires Docker Desktop running).

```
python Aircraft-Engine-Predictive-Maintenance_-main\simulate_pipeline.py --clean --with-monitoring
```

## Running stage-by-stage

Equivalent to the above, run manually from inside this project folder
(useful when iterating on a single stage):

```
python data_validation.py
python eda.py
python feature_engineering.py
python build_feature_store_sources.py

cd feature_repo
feast apply
feast materialize-incremental 2026-01-01T00:00:00   # any current UTC timestamp
cd ..

python modeling.py
```

Check `modeling.py`'s console output (or
`Data_Report_Artifacts/modeling/modeling_report.json`) for which model was
registered as `champion` and its validation metrics.

To browse experiment runs / the model registry:

```
python run_mlflow_ui.py --backend-store-uri sqlite:///mlflow.db --workers 1
```

then open http://127.0.0.1:5000. Use `run_mlflow_ui.py` instead of the raw
`mlflow ui` command — see [Troubleshooting](#troubleshooting) for why.

To browse the Feast registry (entities, feature views, lineage):

```
cd feature_repo
feast ui
```

then open http://localhost:8888. Full retrieval examples (offline
point-in-time, online) are in `feature_repo/README.md`.

## Serving + monitoring

```
uvicorn serving_api:app --host 127.0.0.1 --port 8001
```

Endpoints: `GET /health`, `GET /model_info`, `POST /predict`,
`GET /metrics` (Prometheus exposition format).

```
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/model_info
```

Generate demo traffic (pulls rows from the Feast-served val split, since it
carries ground-truth RUL for live error metrics — test intentionally doesn't):

```
python simulate_traffic.py --requests 150
```

Bring up Prometheus + Grafana:

```
cd monitoring
docker compose up -d
```

Grafana: http://localhost:3000 (dashboard auto-provisioned, no manual
import). Prometheus: http://localhost:9090. Full panel descriptions, the
precision/recall-via-risk-category design, and teardown are in
`monitoring/README.md`.

## Moving this project to another machine

Both MLflow's tracking store (`mlflow.db`) and Feast's local file-based
artifact/online stores bake in **absolute filesystem paths** at the moment
they're created. Copying the project folder to a different drive, username,
or OS doesn't move those paths with it. Two ways to handle this:

**Preferred — regenerate on the target machine** (no path-patching needed):
copy over only the source files (raw CSVs + all `.py` files + `feature_repo/definitions.py`
+ `feature_repo/feature_store.yaml` + `monitoring/`), skip everything listed
under "Generated" above, and run the pipeline fresh there
(`simulate_pipeline.py --clean`, or the manual stage-by-stage steps). Every
path gets baked in correctly for that machine from the start.

**If you've already copied a folder that includes the generated artifacts**
(e.g. a zip with `mlruns/`/`mlflow.db` already in it) and don't want to
retrain: run `python fix_mlflow_paths.py` once from the project root on the
new machine. It rewrites `mlflow.db`'s stale absolute paths (across
`experiments`, `runs`, `logged_models`, and `model_versions`) to the current
project root. Safe to re-run. Note this only fixes MLflow — Feast's
`registry.db`/`online_store.db` store their stale path inside a
length-prefixed protobuf field, which can't be safely string-patched the
same way, so if `feast apply`/`materialize-incremental` fails with a path
error after a move, delete `feature_repo/data/registry.db` and
`online_store.db` and re-run `feast apply` + `materialize-incremental` —
they're cheap, regenerable build artifacts.

## Troubleshooting

**`mlflow.exceptions.MlflowException: Detected out-of-date database schema`**
— the installed MLflow version is newer than the one that created
`mlflow.db`. `requirements.txt` pins `mlflow==3.15.1` specifically to avoid
this; if you still hit it, `pip install mlflow==3.15.1` to match, or (only if
you specifically need a newer MLflow) back up `mlflow.db` and run
`mlflow db upgrade sqlite:///mlflow.db`.

**`mlflow.exceptions.MlflowException: No such artifact: ''`** when
`serving_api.py` loads the champion model — stale absolute paths in
`mlflow.db` from a copied/moved project folder. See
[Moving this project to another machine](#moving-this-project-to-another-machine).

**`feast apply` fails with `FileNotFoundError: ... feature_repo\data`** (or
any other path under a drive/username that doesn't exist on this machine) —
same root cause, but in Feast's `registry.db`/`online_store.db` instead of
MLflow's. Delete both files and re-run `feast apply` +
`feast materialize-incremental` (see above).

**`AttributeError: module 'anyio' has no attribute 'from_thread'`** when
running `mlflow ui` — a known upstream bug in Starlette's deprecated WSGI
bridge (which MLflow's UI server uses to wrap its legacy Flask app): it
accesses `anyio.from_thread` as a bare attribute, which only resolves if
something already explicitly imported that submodule earlier in the
process. Use `python run_mlflow_ui.py ...` instead of `mlflow ui ...` — it
pre-imports the submodule so the bridge never breaks. Same flags as `mlflow ui`.

**`feast ui` fails with `VersionError: Detected incompatible Protobuf
Gencode/Runtime versions`** — installing `mlflow` after `feast[grpcio]`
silently downgrades `protobuf` below what `grpcio-health-checking` needs.
`requirements.txt` already pins `protobuf>=7.35.1,<8` after `mlflow` to
prevent this; if you still hit it, `pip install "protobuf>=7.35.1,<8"`.

**`mlflow ui` fails on startup with `OSError: [WinError 10022] An invalid
argument was supplied`** (Windows only) — MLflow 3's server defaults to
multiple worker processes, and Windows can't share a listening socket across
`multiprocessing`-spawned workers the way Linux/macOS can. Always pass
`--workers 1` (already the case in the commands above).

**Grafana dashboard doesn't reflect a JSON edit** — could be browser
caching (hard refresh, Ctrl+Shift+R) or Windows Docker Desktop's bind-mount
file-change detection missing the edit. `docker compose restart grafana`
from `monitoring/` forces a fresh read; no re-import needed, the dashboard
is provisioned from the mounted file.

**Grafana panels show 0 or "No Data" after a while** — see
`monitoring/README.md` and the note above about `rate()`/`increase()`
windowing, panels that only populate when `ground_truth_rul` is supplied
(the val split has it, test doesn't by design), and that Prometheus's data
isn't in a named Docker volume — `docker compose down` (not `stop`) wipes
its history, so a fresh `docker compose up -d` starts empty until new
traffic runs.

**Predicted RUL values are fractional (e.g. 11.5) even though the ground
truth is all integers** — expected. Nothing in RMSE/MAE-minimizing
regression training constrains outputs to integers; a fractional prediction
represents estimate precision, not a literal claim about fractional cycles.
Round only at display time if you need integer-looking output — rounding
before computing metrics would inject needless quantization error.
