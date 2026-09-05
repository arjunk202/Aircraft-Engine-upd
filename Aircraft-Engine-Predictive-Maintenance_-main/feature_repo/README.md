# Feature Store (Feast)

Local Feast feature repository for the aircraft engine predictive maintenance
project. Corresponds to the "Feature Store" component in the DSPM UML
information-flow diagrams (`Materials/DSPM_material.pdf`).

## Workflow

1. `python data_validation.py` (repo root) — validates the raw data and writes
   the leakage-free engine-wise train/val/test splits to
   `Data_Report_Artifacts/data_validation/validated_splits/`.
2. `python eda.py` (repo root) — explores the TRAIN partition and writes
   `Data_Report_Artifacts/eda/data_exploration.json`.
3. `python feature_engineering.py` (repo root) — acts on the EDA findings
   (drops constant/redundant columns, Yeo-Johnson-transforms skewed ones,
   adds rolling-window stats, formalizes the clipped RUL target) and writes
   `Data_Report_Artifacts/feature_engineering/engineered_splits/`.
4. `python build_feature_store_sources.py` (repo root) — converts both the
   validated raw splits and the engineered splits into Feast-ingestible
   parquet under `feature_repo/data/`, adding a synthetic `event_timestamp`
   (the raw data has no wall-clock time, only a per-engine `cycle` counter).
5. From `feature_repo/`, register the definitions and materialize:
   ```
   feast apply
   feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
   ```

## Current contents

- Entity: `engine` (join key `id`)
- `engine_raw_sensors_{train,val,test}` — raw validated sensor/setting
  columns, one view per split.
- `engine_engineered_features_{train,val,test}` — post-EDA engineered
  features (constant/redundant columns dropped, skew-transformed, rolling
  mean/std, clipped RUL target on train/val). Schema is **inferred directly
  from the parquet file** at `feast apply` time rather than hardcoded, since
  which columns exist depends on the upstream EDA/feature-engineering
  reports — re-run step 4 before `feast apply` whenever those change.

Views are split per partition (rather than one view with a `split` column) so
a downstream consumer can never accidentally join across train/val/test.

## Retrieval

```python
from feast import FeatureStore
import pandas as pd

fs = FeatureStore(repo_path="feature_repo")

# offline, point-in-time correct
entity_df = pd.DataFrame({"id": [1, 2], "event_timestamp": pd.to_datetime(["2020-01-05"] * 2)})
fs.get_historical_features(
    entity_df=entity_df,
    features=["engine_engineered_features_train:RUL", "engine_engineered_features_train:s4_mean_20"],
).to_df()

# online (after materialize)
fs.get_online_features(
    features=["engine_engineered_features_train:RUL"],
    entity_rows=[{"id": 1}],
).to_dict()
```

## Feast UI

```
cd feature_repo
feast ui
```

Then open `http://localhost:8888` — it browses the registry (entities,
feature views, data sources, lineage) built by `feast apply`. Needs the
`grpcio` extra (`feast[grpcio]`, already in `requirements.txt`) — without it
`feast ui` fails with `ModuleNotFoundError: No module named 'grpc'`.

`grpcio-health-checking`/`grpcio-reflection`/`grpcio-tools` (pulled in by the
`grpcio` extra) require `protobuf>=7.35.1,<8`. If `mlflow` gets installed
afterward it silently downgrades `protobuf` to a pre-7 release to satisfy its
own (looser) pin, which then breaks `feast ui` with:
`VersionError: Detected incompatible Protobuf Gencode/Runtime versions ...`.
`requirements.txt` now pins `protobuf>=7.35.1,<8` after `mlflow` so a plain
`pip install -r requirements.txt` lands on a version both packages accept.
If you ever see that error again, `pip install "protobuf>=7.35.1,<8"` fixes
it (harmless `databricks-sdk` conflict warning is expected and unused here —
nothing in this project talks to Databricks).

## MLflow UI (experiment tracking / model registry)

`modeling.py` (repo root) logs every training run to a local SQLite-backed
MLflow tracking store (`../mlflow.db`), which also enables the Model
Registry. To browse it:

```
cd ..
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

Then open `http://localhost:5000`. `--workers 1` is required on Windows —
MLflow 3's server defaults to multiple worker processes, and Windows can't
share a listening socket across `multiprocessing`-spawned workers the way
Linux/macOS can, which fails with `OSError: [WinError 10022] An invalid
argument was supplied` on startup. A single worker is fine for local/solo use.

## Extending further

Once modeling exists, either read training data straight from
`engine_engineered_features_train`/`_val` via `get_historical_features`, or
add new FeatureViews for anything computed later (e.g. model-specific
features). Reuse the existing `engine` entity and the same train/val/test
partitioning already locked in by
`Data_Report_Artifacts/data_validation/engine_split_manifest.json` — do not
re-derive the split.
