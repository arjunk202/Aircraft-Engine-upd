"""
Stage 3: Modeling — trains Remaining Useful Life (RUL) regressors on top of
the engineered feature views registered in the Feast feature store, and
tracks every run (params, metrics, artifacts, model) in MLflow, including
registering the best model in the MLflow Model Registry.

Feature retrieval is intentionally NOT `pd.read_csv(engineered_splits/...)`.
Training reads through Feast's offline store
(`FeatureStore.get_historical_features`), the same point-in-time-correct API
a real-time or batch inference client would use, so the feature store is
provably the single source of truth for what a model was trained on — not
just a mirror of files a model happens to also be trained from. The feature
list itself is discovered dynamically from the `engine_engineered_features_*`
FeatureView definitions (`feature_repo/definitions.py`), so this script never
goes stale if feature_engineering.py's output columns change.

Models trained (see MODEL SELECTION note below):
    - Linear Regression   (scaled)   — interpretable baseline
    - Random Forest                  — notebook baseline, handles nonlinearity
    - XGBoost              (+ early stopping on val)
    - LightGBM             (+ early stopping on val)   <- added
    - HistGradientBoosting (+ early stopping on val)    <- added

Run:
    python modeling.py

Prerequisites: data_validation.py -> eda.py -> feature_engineering.py ->
build_feature_store_sources.py -> `feast apply` (+ `feast materialize-incremental`
if you also want online-store parity), all already wired in feature_repo/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from feast import FeatureStore
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("modeling")

ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class ModelingConfig:
    repo_path: str = str(ROOT_DIR / "feature_repo")
    feature_view_prefix: str = "engine_engineered_features"
    target_col: str = "RUL"
    id_col: str = "id"
    cycle_col: str = "cycle"
    timestamp_col: str = "event_timestamp"

    truth_path: Path = ROOT_DIR / "PM_truth.csv"
    rul_clip: int = 125  # must match feature_engineering.py's clip so test-set scoring is apples-to-apples

    random_state: int = 42

    mlflow_tracking_uri: str = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
    mlflow_experiment_name: str = "aircraft_engine_rul_prediction"
    registered_model_name: str = "aircraft_engine_rul_model"

    artifacts_dir: Path = ROOT_DIR / "Data_Report_Artifacts" / "modeling"

    rf_params: dict[str, Any] = field(
        default_factory=lambda: dict(n_estimators=200, max_depth=15, min_samples_split=5, n_jobs=-1)
    )
    xgb_params: dict[str, Any] = field(
        default_factory=lambda: dict(
            n_estimators=800, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10, n_jobs=-1,
        )
    )
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: dict(
            n_estimators=800, max_depth=-1, num_leaves=31, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1,
        )
    )
    hgb_params: dict[str, Any] = field(
        default_factory=lambda: dict(max_iter=500, max_depth=8, learning_rate=0.05, l2_regularization=0.1)
    )


class RULModelTrainer:
    """
    MODEL SELECTION
    ----------------
    Data_modeling.ipynb trained Linear Regression, Random Forest and XGBoost.
    Two gradient-boosting models are added here:

    - LightGBM: histogram-based boosting, generally matches or beats XGBoost
      on this class of tabular problem while training noticeably faster —
      relevant here because the engineered feature set is wide (rolling-window
      mean/std roughly doubles the sensor column count) and LightGBM's
      leaf-wise growth tends to find interactions across those correlated
      rolling features more efficiently than level-wise XGBoost.
    - HistGradientBoostingRegressor: scikit-learn's native histogram boosting
      implementation (no extra dependency beyond scikit-learn, which is
      already required). Included as a second, independently-implemented
      gradient boosting baseline — useful for sanity-checking that XGBoost/
      LightGBM's edge over Random Forest is a real boosting effect and not an
      artifact of one specific library's defaults.

    Both get an early-stopping validation set, unlike the notebook's fixed
    n_estimators — this is a training-time robustness improvement, not a
    dataset change.

    SCALING
    -------
    The notebook scaled every model's inputs uniformly. Only Linear
    Regression is scale-sensitive; tree ensembles (Random Forest, XGBoost,
    LightGBM, HistGradientBoosting) split on raw feature values and are
    invariant to monotonic rescaling, so they train on the Yeo-Johnson/
    rolling-window features straight out of Feast. Linear Regression's
    StandardScaler is fit train-only and bundled into a `sklearn.Pipeline`
    with the model itself (rather than pre-scaling X and logging a bare
    LinearRegression), so every model registered from this script — whichever
    one ends up champion — accepts the identical raw engineered-feature
    schema. That matters once a model is served: the inference API shouldn't
    need to know per-model-type whether to scale first.
    """

    def __init__(self, config: ModelingConfig):
        self.config = config
        self.fs = FeatureStore(repo_path=config.repo_path)
        self.raw: dict[str, pd.DataFrame] = {}
        self.X: dict[str, pd.DataFrame] = {}
        self.y: dict[str, pd.Series] = {}
        self.results: list[dict[str, Any]] = []
        self.parent_run_id: str | None = None

    # ------------------------------------------------------------------
    def load_features_from_feast(self) -> "RULModelTrainer":
        cfg = self.config
        for split in ("train", "val", "test"):
            view_name = f"{cfg.feature_view_prefix}_{split}"
            feature_view = self.fs.get_feature_view(view_name)
            feature_refs = [f"{view_name}:{f.name}" for f in feature_view.features]

            # The entity spine (id + event_timestamp) is the minimal key set a
            # Feast caller must supply to identify *which* rows to fetch — it is
            # not a feature payload, so reading it off the source parquet
            # doesn't bypass the feature store; get_historical_features() below
            # is what actually resolves it to feature values.
            source_path = Path(cfg.repo_path) / "data" / f"engineered_{split}.parquet"
            spine = pq.read_table(source_path, columns=[cfg.id_col, cfg.timestamp_col]).to_pandas()

            df = self.fs.get_historical_features(entity_df=spine, features=feature_refs).to_df()
            self.raw[split] = df
            logger.info("Loaded %s via Feast offline store: %s rows, %s columns", split, *df.shape)
        return self

    # ------------------------------------------------------------------
    def prepare_xy(self) -> "RULModelTrainer":
        cfg = self.config
        non_feature_cols = {cfg.id_col, cfg.timestamp_col, "created_timestamp", cfg.target_col}

        for split in ("train", "val", "test"):
            df = self.raw[split]
            feature_cols = [c for c in df.columns if c not in non_feature_cols]
            assert cfg.target_col not in feature_cols, f"{cfg.target_col} leaked into {split} features"
            assert cfg.id_col not in feature_cols, f"{cfg.id_col} leaked into {split} features"
            self.X[split] = df[feature_cols].reset_index(drop=True)
            if cfg.target_col in df.columns:
                self.y[split] = df[cfg.target_col].reset_index(drop=True)

        logger.info(
            "Feature matrix ready: %d features | train=%s val=%s test=%s",
            self.X["train"].shape[1], self.X["train"].shape, self.X["val"].shape, self.X["test"].shape,
        )
        return self

    # ------------------------------------------------------------------
    def setup_mlflow(self) -> "RULModelTrainer":
        cfg = self.config
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment_name)
        return self

    # ------------------------------------------------------------------
    def _log_and_evaluate(
        self, run_name: str, model, X_train, y_train, X_val, y_val, params: dict, log_fn, flavor: str
    ) -> dict:
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)

            metrics = {
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, preds_val))),
                "val_mae": float(mean_absolute_error(y_val, preds_val)),
                "val_r2": float(r2_score(y_val, preds_val)),
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("feature_source", "feast_offline_store")
            mlflow.set_tag("stage", "modeling")
            # input_example lets MLflow infer and store a signature (feature
            # names + dtypes) with the model, so a serving client can discover
            # the expected schema from the registry instead of hardcoding it.
            log_fn(model, name="model", input_example=X_train.head(3))

            logger.info("%-22s val_rmse=%.4f val_mae=%.4f val_r2=%.4f", run_name, *metrics.values())

            return {
                "run_name": run_name,
                "run_id": run.info.run_id,
                "flavor": flavor,
                "params": params,
                **metrics,
            }

    # ------------------------------------------------------------------
    def train_linear_regression(self) -> "RULModelTrainer":
        # Bundled as a Pipeline (scaler + model) rather than scaling X upfront
        # and logging a bare LinearRegression: every registered model — whichever
        # one ends up champion — must accept the same raw engineered-feature
        # schema at serving time, so scaling can't be a step the inference API
        # has to know to apply conditionally per model type. The scaler is
        # still fit train-only, just inside the pipeline instead of outside it.
        model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        result = self._log_and_evaluate(
            "linear_regression", model,
            self.X["train"], self.y["train"], self.X["val"], self.y["val"],
            params={"fit_intercept": True, "scaler": "StandardScaler(train-only, bundled in pipeline)"},
            log_fn=mlflow.sklearn.log_model, flavor="sklearn",
        )
        self.results.append(result)
        return self

    def train_random_forest(self) -> "RULModelTrainer":
        cfg = self.config
        model = RandomForestRegressor(random_state=cfg.random_state, **cfg.rf_params)
        result = self._log_and_evaluate(
            "random_forest", model,
            self.X["train"], self.y["train"], self.X["val"], self.y["val"],
            params={**cfg.rf_params, "random_state": cfg.random_state},
            log_fn=mlflow.sklearn.log_model, flavor="sklearn",
        )
        self.results.append(result)
        return self

    def train_xgboost(self) -> "RULModelTrainer":
        cfg = self.config
        model = XGBRegressor(
            random_state=cfg.random_state, objective="reg:squarederror",
            eval_metric="rmse", early_stopping_rounds=30, **cfg.xgb_params,
        )
        with mlflow.start_run(run_name="xgboost", nested=True) as run:
            model.fit(
                self.X["train"], self.y["train"],
                eval_set=[(self.X["val"], self.y["val"])], verbose=False,
            )
            preds_val = model.predict(self.X["val"])
            metrics = {
                "val_rmse": float(np.sqrt(mean_squared_error(self.y["val"], preds_val))),
                "val_mae": float(mean_absolute_error(self.y["val"], preds_val)),
                "val_r2": float(r2_score(self.y["val"], preds_val)),
            }
            params = {**cfg.xgb_params, "random_state": cfg.random_state, "early_stopping_rounds": 30}
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("best_iteration", model.best_iteration)
            mlflow.set_tag("feature_source", "feast_offline_store")
            mlflow.set_tag("stage", "modeling")
            mlflow.xgboost.log_model(model, name="model", input_example=self.X["train"].head(3))
            logger.info("%-22s val_rmse=%.4f val_mae=%.4f val_r2=%.4f (best_iter=%d)",
                        "xgboost", *metrics.values(), model.best_iteration)
            self.results.append({"run_name": "xgboost", "run_id": run.info.run_id, "flavor": "xgboost",
                                  "params": params, **metrics})
        return self

    def train_lightgbm(self) -> "RULModelTrainer":
        from lightgbm import early_stopping, log_evaluation

        cfg = self.config
        model = LGBMRegressor(random_state=cfg.random_state, **cfg.lgbm_params)
        with mlflow.start_run(run_name="lightgbm", nested=True) as run:
            model.fit(
                self.X["train"], self.y["train"],
                eval_set=[(self.X["val"], self.y["val"])],
                callbacks=[early_stopping(stopping_rounds=30, verbose=False), log_evaluation(period=0)],
            )
            preds_val = model.predict(self.X["val"])
            metrics = {
                "val_rmse": float(np.sqrt(mean_squared_error(self.y["val"], preds_val))),
                "val_mae": float(mean_absolute_error(self.y["val"], preds_val)),
                "val_r2": float(r2_score(self.y["val"], preds_val)),
            }
            params = {**cfg.lgbm_params, "random_state": cfg.random_state, "early_stopping_rounds": 30}
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("best_iteration", model.best_iteration_)
            mlflow.set_tag("feature_source", "feast_offline_store")
            mlflow.set_tag("stage", "modeling")
            mlflow.lightgbm.log_model(model, name="model", input_example=self.X["train"].head(3))
            logger.info("%-22s val_rmse=%.4f val_mae=%.4f val_r2=%.4f (best_iter=%d)",
                        "lightgbm", *metrics.values(), model.best_iteration_)
            self.results.append({"run_name": "lightgbm", "run_id": run.info.run_id, "flavor": "lightgbm",
                                  "params": params, **metrics})
        return self

    def train_hist_gradient_boosting(self) -> "RULModelTrainer":
        cfg = self.config
        model = HistGradientBoostingRegressor(
            random_state=cfg.random_state, early_stopping=True, n_iter_no_change=30,
            validation_fraction=None, **cfg.hgb_params,
        )
        # HistGradientBoostingRegressor's early_stopping uses its own internal
        # validation carve-out from the training data (validation_fraction) by
        # default; passing explicit val data isn't supported the way
        # XGBoost/LightGBM do it, so we still evaluate on our own held-out
        # val split afterward for a metric that's comparable to the other models.
        result = self._log_and_evaluate(
            "hist_gradient_boosting", model,
            self.X["train"], self.y["train"], self.X["val"], self.y["val"],
            params={**cfg.hgb_params, "random_state": cfg.random_state, "early_stopping": True, "n_iter_no_change": 30},
            log_fn=mlflow.sklearn.log_model, flavor="sklearn",
        )
        self.results.append(result)
        return self

    # ------------------------------------------------------------------
    def evaluate_best_on_test(self) -> "RULModelTrainer":
        """
        Scores the best model against PM_truth.csv — the real FD001 test-set
        RUL at the truncation point of each test engine's trajectory. Only the
        *last* observed cycle per engine is scored, mirroring how the model
        would actually be queried in deployment (predict RUL now, from the
        latest available reading).
        """
        cfg = self.config
        best = max(self.results, key=lambda r: r["val_r2"])
        self.best_result = best

        model_uri = f"runs:/{best['run_id']}/model"
        loaded = mlflow.pyfunc.load_model(model_uri)

        X_test = self.X["test"]  # every registered model (incl. linear_regression's Pipeline) accepts raw features
        test_meta = self.raw["test"][[cfg.id_col, cfg.cycle_col]].reset_index(drop=True)
        last_cycle_idx = test_meta.groupby(cfg.id_col)[cfg.cycle_col].idxmax()

        preds_all = loaded.predict(X_test)
        preds_last = pd.Series(preds_all, name="predicted_rul").loc[last_cycle_idx].reset_index(drop=True)
        ids_last = test_meta.loc[last_cycle_idx, cfg.id_col].reset_index(drop=True)

        truth = pd.read_csv(cfg.truth_path).rename(columns={"cycle": "true_rul"})  # mislabeled header, see data_validation.py
        scored = pd.DataFrame({cfg.id_col: ids_last, "predicted_rul": preds_last}).merge(truth, on=cfg.id_col, how="inner")

        raw_rmse = float(np.sqrt(mean_squared_error(scored["true_rul"], scored["predicted_rul"])))
        raw_mae = float(mean_absolute_error(scored["true_rul"], scored["predicted_rul"]))
        clipped_truth = scored["true_rul"].clip(upper=cfg.rul_clip)
        clipped_rmse = float(np.sqrt(mean_squared_error(clipped_truth, scored["predicted_rul"])))
        clipped_mae = float(mean_absolute_error(clipped_truth, scored["predicted_rul"]))

        self.test_metrics = {
            "engines_scored": int(len(scored)),
            "raw_test_rmse": raw_rmse,
            "raw_test_mae": raw_mae,
            "clipped_test_rmse": clipped_rmse,  # truth clipped at rul_clip, matching the training target's clip
            "clipped_test_mae": clipped_mae,
        }
        logger.info("Best model (%s) on held-out FD001 test set: raw_rmse=%.4f clipped_rmse=%.4f",
                    best["run_name"], raw_rmse, clipped_rmse)

        # Already inside the parent run's context (see run()) — log directly
        # rather than start_run(run_id=...), which would error on an already-active run.
        mlflow.log_metrics({f"test_{k}": v for k, v in self.test_metrics.items() if isinstance(v, float)})
        return self

    # ------------------------------------------------------------------
    def register_best_model(self) -> "RULModelTrainer":
        cfg = self.config
        client = MlflowClient()
        model_uri = f"runs:/{self.best_result['run_id']}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=cfg.registered_model_name)
        client.set_registered_model_alias(cfg.registered_model_name, "champion", mv.version)
        client.set_model_version_tag(cfg.registered_model_name, mv.version, "val_rmse", str(self.best_result["val_rmse"]))
        client.set_model_version_tag(cfg.registered_model_name, mv.version, "val_r2", str(self.best_result["val_r2"]))
        client.update_registered_model(
            cfg.registered_model_name,
            description="RUL regressor for NASA C-MAPSS FD001 turbofan engines, trained on Feast-served engineered features.",
        )
        self.registered_version = mv.version
        logger.info(
            "Registered '%s' as %s v%s and aliased it 'champion'",
            self.best_result["run_name"], cfg.registered_model_name, mv.version,
        )
        return self

    # ------------------------------------------------------------------
    def save_artifacts(self) -> dict:
        cfg = self.config
        cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

        leaderboard = sorted(self.results, key=lambda r: r["val_rmse"])
        report = {
            "mlflow_tracking_uri": cfg.mlflow_tracking_uri,
            "mlflow_experiment": cfg.mlflow_experiment_name,
            "parent_run_id": self.parent_run_id,
            "feature_count": self.X["train"].shape[1],
            "train_rows": int(self.X["train"].shape[0]),
            "val_rows": int(self.X["val"].shape[0]),
            "test_rows": int(self.X["test"].shape[0]),
            "leaderboard": [
                {k: v for k, v in r.items() if k != "params"} | {"params": r["params"]} for r in leaderboard
            ],
            "best_model": self.best_result["run_name"],
            "best_model_run_id": self.best_result["run_id"],
            "test_set_evaluation": self.test_metrics,
            "registered_model_name": cfg.registered_model_name,
            "registered_model_version": self.registered_version,
            "registered_model_alias": "champion",
        }
        with open(cfg.artifacts_dir / "modeling_report.json", "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Saved modeling report -> %s", cfg.artifacts_dir / "modeling_report.json")
        return report

    # ------------------------------------------------------------------
    def run(self) -> dict:
        (
            self.load_features_from_feast()
            .prepare_xy()
            .setup_mlflow()
        )
        with mlflow.start_run(run_name="rul_modeling_pipeline") as parent_run:
            self.parent_run_id = parent_run.info.run_id
            mlflow.log_param("feature_count", self.X["train"].shape[1])
            mlflow.log_param("train_rows", self.X["train"].shape[0])
            mlflow.log_param("val_rows", self.X["val"].shape[0])
            (
                self.train_linear_regression()
                .train_random_forest()
                .train_xgboost()
                .train_lightgbm()
                .train_hist_gradient_boosting()
                .evaluate_best_on_test()
                .register_best_model()
            )
        return self.save_artifacts()


def main() -> None:
    trainer = RULModelTrainer(ModelingConfig())
    report = trainer.run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
