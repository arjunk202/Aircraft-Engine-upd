"""
Feature Engineering — Stage 2 (Data Collection, Exploration & Preparation)
of the MLOps-based Data Science Process Model (DSPM).

Turns the validated, engine-wise TRAIN/VAL/TEST partitions from
data_validation.py into modeling-ready features, acting directly on the
findings from eda.py's Data Exploration Report
(Data_Report_Artifacts/eda/data_exploration.json) — nothing here is
hardcoded from a one-off look at the data; every decision below is read from
the upstream validation/EDA artifacts so re-running earlier stages on new
data automatically changes what this stage does.

How each EDA recommendation is handled (see inline comments at each step
for the reasoning):
  1. Constant features (data_quality.json)      -> dropped outright.
  2. Highly-correlated redundant pairs (EDA)     -> one feature per pair dropped.
  3. Skewed features (EDA)                       -> Yeo-Johnson power transform.
  4. Monotonic degradation-trend features (EDA)  -> rolling mean/std windows.

The engine-wise split itself is NOT re-derived here — it was locked in by
data_validation.py's engine_split_manifest.json and is inherited as-is by
reading the already-partitioned validated_{train,val,test}_raw.csv files.

Run after data_validation.py and eda.py:

    python feature_engineering.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("feature_engineering")

ROOT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = ROOT_DIR / "Data_Report_Artifacts" / "data_validation" / "validated_splits"
QUALITY_REPORT_PATH = ROOT_DIR / "Data_Report_Artifacts" / "data_validation" / "data_quality.json"
EXPLORATION_REPORT_PATH = ROOT_DIR / "Data_Report_Artifacts" / "eda" / "data_exploration.json"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "Data_Report_Artifacts" / "feature_engineering"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "engineered_splits"


@dataclass
class FeatureEngineeringConfig:
    id_col: str = "id"
    cycle_col: str = "cycle"
    rolling_windows: list[int] = field(default_factory=lambda: [5, 20])
    rul_clip: int = 125
    split_dir: Path = field(default_factory=lambda: SPLIT_DIR)
    quality_report_path: Path = field(default_factory=lambda: QUALITY_REPORT_PATH)
    exploration_report_path: Path = field(default_factory=lambda: EXPLORATION_REPORT_PATH)
    artifact_dir: Path = field(default_factory=lambda: DEFAULT_ARTIFACT_DIR)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)


class FeatureEngineer:
    """Applies EDA-driven feature engineering to the validated engine-wise splits."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.quality_report: dict = {}
        self.exploration_report: dict = {}
        self.dropped_constant: list[str] = []
        self.dropped_redundant: list[dict] = []
        self.skew_transformed: list[str] = []
        self.rolling_source_features: list[str] = []
        self.power_transformer: Optional[PowerTransformer] = None
        self.log: list[str] = []

    # ------------------------------------------------------------------
    def load_inputs(self) -> "FeatureEngineer":
        cfg = self.config
        train_path = cfg.split_dir / "validated_train_raw.csv"
        val_path = cfg.split_dir / "validated_val_raw.csv"
        test_path = cfg.split_dir / "validated_test_raw.csv"
        for p in (train_path, val_path, test_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found — run data_validation.py first")
        if not cfg.exploration_report_path.exists():
            raise FileNotFoundError(f"{cfg.exploration_report_path} not found — run eda.py first")

        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        self.quality_report = json.loads(cfg.quality_report_path.read_text(encoding="utf-8"))
        self.exploration_report = json.loads(cfg.exploration_report_path.read_text(encoding="utf-8"))

        logger.info(
            "Loaded splits: train=%s val=%s test=%s (engine-wise partitioning inherited from data_validation.py)",
            self.df_train.shape, self.df_val.shape, self.df_test.shape,
        )
        return self

    # ------------------------------------------------------------------
    # Step 1 — drop constant features (data_validation's data_quality.json)
    # ------------------------------------------------------------------
    def drop_constant_features(self) -> "FeatureEngineer":
        # Zero-variance columns carry no information for any model — they were
        # only *flagged* (not removed) at validation time because removal is a
        # preparation-stage decision. This is that stage, so we act on the flag.
        constant_cols = [c for c in self.quality_report.get("constant_features", []) if c in self.df_train.columns]
        self.dropped_constant = constant_cols
        for df in (self.df_train, self.df_val, self.df_test):
            df.drop(columns=constant_cols, inplace=True, errors="ignore")
        self.log.append(f"Dropped {len(constant_cols)} constant feature(s): {constant_cols}")
        logger.info("Dropped constant features: %s", constant_cols)
        return self

    # ------------------------------------------------------------------
    # Step 2 — drop one feature from each highly-correlated redundant pair
    # ------------------------------------------------------------------
    def drop_redundant_features(self) -> "FeatureEngineer":
        # EDA flagged pairs with |correlation| >= 0.9 on the TRAIN partition.
        # Keeping both adds multicollinearity without adding information, so we
        # drop one side of each pair. We keep `feature_a` and drop `feature_b`
        # (EDA always orders pairs by the itertools.combinations() column order,
        # so this is a deterministic, reproducible choice — not an arbitrary one
        # made per run) and skip a pair if one side was already removed as a
        # constant feature or by an earlier pair, so we never drop more than
        # one feature per redundant group.
        pairs = self.exploration_report.get("correlation_analysis", {}).get("high_correlation_pairs", [])
        already_dropped = set(self.dropped_constant)
        redundant_dropped = []
        for pair in pairs:
            a, b = pair["feature_a"], pair["feature_b"]
            if a in already_dropped or b in already_dropped:
                continue
            if b in self.df_train.columns:
                redundant_dropped.append(pair)
                already_dropped.add(b)

        drop_cols = [p["feature_b"] for p in redundant_dropped]
        self.dropped_redundant = redundant_dropped
        for df in (self.df_train, self.df_val, self.df_test):
            df.drop(columns=drop_cols, inplace=True, errors="ignore")
        self.log.append(f"Dropped {len(drop_cols)} redundant feature(s): {redundant_dropped}")
        logger.info("Dropped redundant features (kept feature_a, dropped feature_b): %s", redundant_dropped)
        return self

    # ------------------------------------------------------------------
    # Step 3 — Yeo-Johnson power transform for skewed features
    # ------------------------------------------------------------------
    def transform_skewed_features(self) -> "FeatureEngineer":
        # Technique: sklearn's PowerTransformer(method="yeo-johnson").
        #
        # Why Yeo-Johnson specifically (over log1p/sqrt/Box-Cox):
        #   - Box-Cox and log1p only work on strictly positive data; several
        #     sensor/setting columns here are near zero or can be <= 0, which
        #     would break them outright.
        #   - Yeo-Johnson handles zero and negative values natively.
        #   - It fits one optimal power parameter (lambda) per column via
        #     maximum likelihood instead of us guessing a transform per
        #     feature by hand, and is the direct sklearn analogue of Box-Cox
        #     for mixed-sign data.
        #
        # Leakage control: the transformer is fit on TRAIN ONLY, then applied
        # (not re-fit) to VAL/TEST — the same convention already used for
        # scaling in this project's earlier modeling pipeline
        # (see MLOPS_PIPELINE/feature_store/feature_store_metadata.json:
        # "scaler_fitted_on": "train only").
        skewed = [
            f["feature"] for f in self.exploration_report.get("skewed_features", [])
            if f["feature"] in self.df_train.columns
        ]
        self.skew_transformed = skewed
        if not skewed:
            self.log.append("No skewed features to transform")
            return self

        transformer = PowerTransformer(method="yeo-johnson")
        transformer.fit(self.df_train[skewed])
        for df in (self.df_train, self.df_val, self.df_test):
            df[skewed] = transformer.transform(df[skewed])
        self.power_transformer = transformer

        self.log.append(f"Applied Yeo-Johnson power transform (train-fit) to {len(skewed)} skewed feature(s): {skewed}")
        logger.info("Yeo-Johnson transform applied to skewed features: %s", skewed)
        return self

    # ------------------------------------------------------------------
    # Step 4 — rolling-window features for monotonic degradation-trend sensors
    # ------------------------------------------------------------------
    def add_rolling_features(self) -> "FeatureEngineer":
        # Technique: per-engine rolling mean + rolling std, computed only over
        # past/current cycles (pandas .rolling(), no centering) so no future
        # information ever leaks into a row — required for a time-series
        # target like RUL.
        #
        # Restricted to the features EDA flagged as having a consistent
        # monotonic trend vs. cycle (mean |Spearman correlation| >= threshold,
        # averaged per engine). Blanket-applying rolling stats to every sensor
        # would double the feature count for columns that showed no
        # degradation signal in EDA, adding noise/dimensionality without
        # informative content — the EDA report's explicit recommendation was
        # to prioritize the trending sensors, so only those get rolling
        # features. Non-trending (but still non-redundant, non-constant) raw
        # sensors are kept as-is: shape/level can still carry signal even
        # without a monotonic slope.
        cfg = self.config
        trend_features = [
            f["feature"] for f in self.exploration_report.get("degradation_trend_analysis", {}).get("monotonic_trend_features", [])
            if f["feature"] in self.df_train.columns
        ]
        self.rolling_source_features = trend_features

        def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
            df = df.sort_values([cfg.id_col, cfg.cycle_col]).copy()
            grouped = df.groupby(cfg.id_col)
            for w in cfg.rolling_windows:
                for col in trend_features:
                    df[f"{col}_mean_{w}"] = grouped[col].transform(lambda s: s.rolling(window=w, min_periods=w).mean())
                    df[f"{col}_std_{w}"] = grouped[col].transform(lambda s: s.rolling(window=w, min_periods=w).std())
            return df

        self.df_train = add_rolling(self.df_train)
        self.df_val = add_rolling(self.df_val)
        self.df_test = add_rolling(self.df_test)

        self.log.append(
            f"Added rolling mean/std over windows {cfg.rolling_windows} for {len(trend_features)} "
            f"monotonic-trend feature(s): {trend_features}"
        )
        logger.info("Rolling features added for windows=%s on trend features: %s", cfg.rolling_windows, trend_features)
        return self

    # ------------------------------------------------------------------
    # Step 5 — formalize the RUL target (train/val only; clip is a
    # preparation-stage decision, deliberately not made back in eda.py)
    # ------------------------------------------------------------------
    def compute_target(self) -> "FeatureEngineer":
        cfg = self.config
        for name, df in (("train", self.df_train), ("val", self.df_val)):
            max_cycle = df.groupby(cfg.id_col)[cfg.cycle_col].transform("max")
            df["RUL"] = (max_cycle - df[cfg.cycle_col]).clip(upper=cfg.rul_clip)
        self.log.append(f"Computed RUL = max_cycle - cycle, clipped at {cfg.rul_clip} cycles (train/val only)")
        logger.info("RUL target computed and clipped at %d for train/val", cfg.rul_clip)
        return self

    # ------------------------------------------------------------------
    # Step 6 — drop rows with incomplete rolling windows
    # ------------------------------------------------------------------
    def drop_incomplete_rows(self) -> "FeatureEngineer":
        # Early cycles of each engine don't have `window` prior cycles yet, so
        # rolling_mean/rolling_std are NaN there by construction (min_periods=w).
        # Those rows carry no valid engineered features and are dropped, same
        # as the established convention in this project's earlier pipeline.
        before = {"train": len(self.df_train), "val": len(self.df_val), "test": len(self.df_test)}
        self.df_train = self.df_train.dropna().reset_index(drop=True)
        self.df_val = self.df_val.dropna().reset_index(drop=True)
        self.df_test = self.df_test.dropna().reset_index(drop=True)
        after = {"train": len(self.df_train), "val": len(self.df_val), "test": len(self.df_test)}
        self.log.append(f"Dropped rows with incomplete rolling windows: {before} -> {after}")
        logger.info("Row counts after dropping incomplete rolling windows: %s -> %s", before, after)
        return self

    # ------------------------------------------------------------------
    def save_artifacts(self) -> dict:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self.df_train.to_csv(cfg.output_dir / "train_features.csv", index=False)
        self.df_val.to_csv(cfg.output_dir / "val_features.csv", index=False)
        self.df_test.to_csv(cfg.output_dir / "test_features.csv", index=False)

        if self.power_transformer is not None:
            joblib.dump(self.power_transformer, cfg.artifact_dir / "power_transformer.pkl")

        feature_cols = [c for c in self.df_train.columns if c not in (cfg.id_col, "RUL")]
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "steps_applied": self.log,
            "dropped_constant_features": self.dropped_constant,
            "dropped_redundant_features": self.dropped_redundant,
            "skew_transformed_features": self.skew_transformed,
            "skew_transform_technique": "sklearn.preprocessing.PowerTransformer(method='yeo-johnson'), fit on TRAIN only",
            "rolling_window_features": {
                "source_features": self.rolling_source_features,
                "windows": self.config.rolling_windows,
                "statistics": ["mean", "std"],
            },
            "rul_clip": self.config.rul_clip,
            "final_feature_count": len(feature_cols),
            "final_feature_columns": feature_cols,
            "output_shapes": {
                "train": list(self.df_train.shape),
                "val": list(self.df_val.shape),
                "test": list(self.df_test.shape),
            },
        }
        with open(cfg.artifact_dir / "feature_engineering_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Engineered splits + report written to %s", cfg.artifact_dir)
        return report

    # ------------------------------------------------------------------
    def run(self) -> dict:
        logger.info("=" * 70)
        logger.info("STAGE 2 — FEATURE ENGINEERING")
        logger.info("=" * 70)
        (
            self.load_inputs()
            .drop_constant_features()
            .drop_redundant_features()
            .transform_skewed_features()
            .add_rolling_features()
            .compute_target()
            .drop_incomplete_rows()
        )
        report = self.save_artifacts()
        logger.info("-" * 70)
        logger.info("Final feature count: %d", report["final_feature_count"])
        logger.info("=" * 70)
        return report


def main() -> int:
    FeatureEngineer(FeatureEngineeringConfig()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
