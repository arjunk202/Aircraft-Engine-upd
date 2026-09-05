"""
Data Validation — Stage 2 (Data Collection, Exploration & Preparation)
of the MLOps-based Data Science Process Model (DSPM).

Validates the raw NASA C-MAPSS FD001 engine sensor data (PM_train.csv,
PM_test.csv, PM_truth.csv) before it is handed off to EDA / feature
engineering. Produces the Stage 2 documentation artifacts referenced in the
DSPM UML information-flow diagram (Data Description Report, Data Quality
Report, Schema/Business-rule Validation Report, Data Dictionary, Data
Requirements) and locks in a deterministic, leakage-free engine-wise
train/validation split so every later stage (EDA, feature engineering,
modeling) partitions engines identically instead of re-deriving its own split.

Run directly to validate the datasets in the project root and write all
artifacts under Data_Report_Artifacts/data_validation/:

    python data_validation.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("data_validation")

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "Data_Report_Artifacts" / "data_validation"
DEFAULT_SPLIT_DIR = DEFAULT_ARTIFACT_DIR / "validated_splits"


def _resolve_path(candidates: list[str]) -> Optional[Path]:
    """Resolve the first existing file among case-variant filename candidates."""
    for name in candidates:
        path = ROOT_DIR / name
        if path.exists():
            return path
    return None


@dataclass
class ValidationConfig:
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    truth_path: Optional[Path] = None
    id_col: str = "id"
    cycle_col: str = "cycle"
    val_ratio: float = 0.2
    random_seed: int = 42
    min_engines_required: int = 50
    near_zero_variance_threshold: float = 1e-6
    rul_clip: int = 125  # documented for the downstream preparation stage
    artifact_dir: Path = field(default_factory=lambda: DEFAULT_ARTIFACT_DIR)
    split_dir: Path = field(default_factory=lambda: DEFAULT_SPLIT_DIR)

    def __post_init__(self) -> None:
        self.train_path = self.train_path or _resolve_path(["PM_train.csv", "pm_train.csv"])
        self.test_path = self.test_path or _resolve_path(["PM_test.csv", "pm_test.csv"])
        self.truth_path = self.truth_path or _resolve_path(["PM_truth.csv", "pm_truth.csv"])


class DataValidator:
    """Runs the Stage 2 data validation pipeline and emits DSPM artifacts."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.df_truth: Optional[pd.DataFrame] = None
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.artifacts: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_data(self) -> "DataValidator":
        cfg = self.config
        missing = [
            name
            for name, path in [("train", cfg.train_path), ("test", cfg.test_path), ("truth", cfg.truth_path)]
            if path is None
        ]
        if missing:
            raise FileNotFoundError(
                f"Could not locate raw data file(s) for: {', '.join(missing)}. "
                "Expected PM_train.csv, PM_test.csv, PM_truth.csv in the project root."
            )

        self.df_train = pd.read_csv(cfg.train_path)
        self.df_test = pd.read_csv(cfg.test_path)
        self.df_truth = pd.read_csv(cfg.truth_path)

        # PM_truth.csv ships with a misleading header ("id,cycle") — the second
        # column is actually the true RUL at the end of each test engine's
        # trajectory, not a cycle count. Normalize it here for downstream use.
        if list(self.df_truth.columns) == [cfg.id_col, cfg.cycle_col]:
            self.df_truth = self.df_truth.rename(columns={cfg.cycle_col: "RUL"})

        logger.info("Loaded train=%s test=%s truth=%s", self.df_train.shape, self.df_test.shape, self.df_truth.shape)
        return self

    # ------------------------------------------------------------------
    # Data Description Report
    # ------------------------------------------------------------------
    def describe_data(self) -> "DataValidator":
        cfg = self.config

        def describe_df(name: str, df: pd.DataFrame) -> dict:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            desc = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "column_statistics": {
                    col: {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()) if len(df) > 1 else 0.0,
                        "unique": int(df[col].nunique()),
                    }
                    for col in numeric_cols
                },
            }
            if cfg.id_col in df.columns:
                desc["unique_engines"] = int(df[cfg.id_col].nunique())
            return desc

        engine_lifetimes = self.df_train.groupby(self.config.id_col)[self.config.cycle_col].max()

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "train": str(self.config.train_path.name),
                "test": str(self.config.test_path.name),
                "truth": str(self.config.truth_path.name),
            },
            "train": describe_df("train", self.df_train),
            "test": describe_df("test", self.df_test),
            "truth": describe_df("truth", self.df_truth),
            "engine_lifetime_cycles": {
                "min": int(engine_lifetimes.min()),
                "max": int(engine_lifetimes.max()),
                "mean": round(float(engine_lifetimes.mean()), 2),
                "median": float(engine_lifetimes.median()),
            },
        }
        self.artifacts["data_description"] = report
        logger.info("Data description report generated")
        return self

    # ------------------------------------------------------------------
    # Schema Validation Report
    # ------------------------------------------------------------------
    def validate_schema(self) -> "DataValidator":
        cfg = self.config
        checks = []

        expected_cols = set(self.df_train.columns)
        actual_test_cols = set(self.df_test.columns)

        def add_check(check_type: str, name: str, passed: bool, details: str) -> None:
            checks.append({"type": check_type, "check": name, "status": "PASS" if passed else "FAIL", "details": details})
            if not passed:
                self.errors.append(f"[schema] {name}: {details}")

        add_check(
            "schema",
            "train/test column parity",
            expected_cols == actual_test_cols,
            f"missing_in_test={sorted(expected_cols - actual_test_cols)}, extra_in_test={sorted(actual_test_cols - expected_cols)}",
        )
        add_check("schema", f"'{cfg.id_col}' column present", cfg.id_col in self.df_train.columns, "required entity/engine id column")
        add_check("schema", f"'{cfg.cycle_col}' column present", cfg.cycle_col in self.df_train.columns, "required time-index column")

        for col in (cfg.id_col, cfg.cycle_col):
            if col in self.df_train.columns:
                is_int_like = pd.api.types.is_integer_dtype(self.df_train[col]) or (
                    pd.api.types.is_float_dtype(self.df_train[col]) and (self.df_train[col] % 1 == 0).all()
                )
                add_check("schema", f"'{col}' is integer-valued", bool(is_int_like), f"dtype={self.df_train[col].dtype}")

        sensor_setting_cols = [c for c in self.df_train.columns if c.startswith("s") or c.startswith("setting")]
        non_numeric = [c for c in sensor_setting_cols if not pd.api.types.is_numeric_dtype(self.df_train[c])]
        add_check("schema", "sensor/setting columns are numeric", len(non_numeric) == 0, f"non_numeric_columns={non_numeric}")

        add_check(
            "schema",
            "truth engine ids match test engine ids",
            set(self.df_truth[cfg.id_col]) == set(self.df_test[cfg.id_col]),
            f"truth_engines={self.df_truth[cfg.id_col].nunique()}, test_engines={self.df_test[cfg.id_col].nunique()}",
        )

        self.artifacts["schema_validation"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "passed": sum(1 for c in checks if c["status"] == "PASS"),
            "failed": sum(1 for c in checks if c["status"] == "FAIL"),
        }
        logger.info("Schema validation: %d/%d checks passed", self.artifacts["schema_validation"]["passed"], len(checks))
        return self

    # ------------------------------------------------------------------
    # Data Quality Report
    # ------------------------------------------------------------------
    def assess_quality(self) -> "DataValidator":
        cfg = self.config
        quality = {}

        for name, df in [("train", self.df_train), ("test", self.df_test), ("truth", self.df_truth)]:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = int(df.isnull().sum().sum())
            completeness = round(((total_cells - missing_cells) / total_cells) * 100, 4) if total_cells else 100.0

            duplicate_rows = int(df.duplicated().sum())
            id_cycle_dups = (
                int(df.duplicated(subset=[cfg.id_col, cfg.cycle_col]).sum())
                if {cfg.id_col, cfg.cycle_col}.issubset(df.columns)
                else None
            )

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_finite = int(np.isinf(df[numeric_cols]).sum().sum()) if len(numeric_cols) else 0

            entry = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "completeness_pct": completeness,
                "missing_values_total": missing_cells,
                "missing_values_by_column": {c: int(v) for c, v in df.isnull().sum().items() if v > 0},
                "duplicate_rows": duplicate_rows,
                "duplicate_id_cycle_pairs": id_cycle_dups,
                "non_finite_values": non_finite,
            }

            if cfg.cycle_col in df.columns:
                entry["invalid_cycles_le_zero"] = int((df[cfg.cycle_col] <= 0).sum())

            if cfg.id_col in df.columns and cfg.cycle_col in df.columns:
                gap_engines = self._count_engines_with_cycle_gaps(df)
                entry["unique_engines"] = int(df[cfg.id_col].nunique())
                entry["engines_with_cycle_gaps"] = gap_engines

            quality[name] = entry
            if duplicate_rows or (id_cycle_dups or 0) > 0:
                self.warnings.append(f"[quality:{name}] {duplicate_rows} duplicate rows, {id_cycle_dups} duplicate (id,cycle) pairs")
            if missing_cells:
                self.warnings.append(f"[quality:{name}] {missing_cells} missing cells")

        # near-zero-variance / constant feature detection (reported only —
        # removal is a Preparation-stage decision, not performed here)
        feature_cols = [c for c in self.df_train.columns if c.startswith("s") or c.startswith("setting")]
        variances = self.df_train[feature_cols].var()
        constant_cols = variances[variances == 0].index.tolist()
        near_zero_cols = variances[(variances > 0) & (variances < cfg.near_zero_variance_threshold)].index.tolist()

        self.artifacts["data_quality"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "datasets": quality,
            "constant_features": constant_cols,
            "near_zero_variance_features": near_zero_cols,
            "near_zero_variance_threshold": cfg.near_zero_variance_threshold,
        }
        logger.info("Data quality report generated (%d constant, %d near-zero-variance features flagged)", len(constant_cols), len(near_zero_cols))
        return self

    def _count_engines_with_cycle_gaps(self, df: pd.DataFrame) -> int:
        cfg = self.config
        gap_count = 0
        for _, group in df.groupby(cfg.id_col)[cfg.cycle_col]:
            cycles = group.sort_values().to_numpy()
            expected = np.arange(cycles[0], cycles[0] + len(cycles))
            if not np.array_equal(cycles, expected):
                gap_count += 1
        return gap_count

    # ------------------------------------------------------------------
    # Business rule validation
    # ------------------------------------------------------------------
    def validate_business_rules(self) -> "DataValidator":
        cfg = self.config
        checks = []

        def add_check(name: str, passed: bool, details: str) -> None:
            checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})
            if not passed:
                self.errors.append(f"[business_rule] {name}: {details}")

        add_check(
            "engine ids are positive integers",
            bool((self.df_train[cfg.id_col] > 0).all() and pd.api.types.is_integer_dtype(self.df_train[cfg.id_col])),
            f"min_id={self.df_train[cfg.id_col].min()}",
        )
        add_check(
            "every engine's cycle sequence starts at 1 and has no gaps",
            self._count_engines_with_cycle_gaps(self.df_train) == 0,
            f"engines_with_gaps={self._count_engines_with_cycle_gaps(self.df_train)}",
        )
        add_check(
            "truth RUL values are non-negative",
            bool((self.df_truth["RUL"] >= 0).all()) if "RUL" in self.df_truth.columns else False,
            f"min_truth_rul={self.df_truth.get('RUL', pd.Series(dtype=float)).min()}",
        )
        add_check(
            f"at least {cfg.min_engines_required} training engines available for a leakage-free engine-wise split",
            self.df_train[cfg.id_col].nunique() >= cfg.min_engines_required,
            f"train_engines={self.df_train[cfg.id_col].nunique()}",
        )
        min_val_engines = max(1, int(cfg.val_ratio * self.df_train[cfg.id_col].nunique()))
        add_check(
            "validation split ratio yields at least one held-out engine",
            min_val_engines >= 1,
            f"val_ratio={cfg.val_ratio}, projected_val_engines={min_val_engines}",
        )

        self.artifacts["business_rules"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "passed": sum(1 for c in checks if c["status"] == "PASS"),
            "failed": sum(1 for c in checks if c["status"] == "FAIL"),
        }
        logger.info("Business rule validation: %d/%d checks passed", self.artifacts["business_rules"]["passed"], len(checks))
        return self

    # ------------------------------------------------------------------
    # Data Dictionary
    # ------------------------------------------------------------------
    def build_data_dictionary(self) -> "DataValidator":
        cfg = self.config
        rows = []
        for col in self.df_train.columns:
            if col == cfg.id_col:
                role, description = "Entity / Engine ID", "Unique engine identifier used for engine-wise grouping and splitting"
            elif col == cfg.cycle_col:
                role, description = "Time index", "Operating cycle number since engine start (1-based, contiguous per engine)"
            elif col.startswith("setting"):
                role, description = "Operating setting", f"Engine operating condition ({col})"
            elif col.startswith("s"):
                role, description = "Sensor measurement", f"Raw sensor reading ({col}); potential physical degradation proxy"
            else:
                role, description = "Other", "Unclassified column"

            series = self.df_train[col]
            rows.append(
                {
                    "feature": col,
                    "data_type": str(series.dtype),
                    "role": role,
                    "description": description,
                    "min": float(series.min()) if pd.api.types.is_numeric_dtype(series) else None,
                    "max": float(series.max()) if pd.api.types.is_numeric_dtype(series) else None,
                    "constant": bool(pd.api.types.is_numeric_dtype(series) and series.std() == 0),
                    "source": "Raw data",
                }
            )
        self.artifacts["data_dictionary"] = {"generated_at": datetime.now(timezone.utc).isoformat(), "features": rows}
        logger.info("Data dictionary generated for %d columns", len(rows))
        return self

    # ------------------------------------------------------------------
    # Engine-wise split (locked in for every downstream stage)
    # ------------------------------------------------------------------
    def generate_engine_split(self) -> "DataValidator":
        cfg = self.config
        engine_ids = np.sort(self.df_train[cfg.id_col].unique())
        rng = np.random.default_rng(cfg.random_seed)
        shuffled = rng.permutation(engine_ids)

        n_val = max(1, int(cfg.val_ratio * len(engine_ids)))
        val_ids = sorted(int(i) for i in shuffled[:n_val])
        train_ids = sorted(int(i) for i in shuffled[n_val:])
        overlap = set(train_ids) & set(val_ids)

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "strategy": "engine_wise_group_split",
            "random_seed": cfg.random_seed,
            "val_ratio": cfg.val_ratio,
            "train_engine_ids": train_ids,
            "val_engine_ids": val_ids,
            "test_engine_ids": sorted(int(i) for i in self.df_test[cfg.id_col].unique()),
            "train_engine_count": len(train_ids),
            "val_engine_count": len(val_ids),
            "engine_overlap": sorted(overlap),
            "leakage_free": len(overlap) == 0,
            "note": (
                "This manifest is the single source of truth for engine-wise splitting. "
                "EDA, feature engineering, and modeling stages must reuse these engine id "
                "sets (not re-derive their own split) to guarantee no engine ever appears "
                "in more than one partition."
            ),
        }
        if overlap:
            self.errors.append(f"[split] engine overlap between train/val: {sorted(overlap)}")

        self.artifacts["engine_split_manifest"] = manifest

        # Materialize the raw (unengineered) engine-wise partitions so later
        # stages load a ready-made, leakage-free split instead of re-splitting.
        cfg.split_dir.mkdir(parents=True, exist_ok=True)
        train_df = self.df_train[self.df_train[cfg.id_col].isin(train_ids)].reset_index(drop=True)
        val_df = self.df_train[self.df_train[cfg.id_col].isin(val_ids)].reset_index(drop=True)
        train_df.to_csv(cfg.split_dir / "validated_train_raw.csv", index=False)
        val_df.to_csv(cfg.split_dir / "validated_val_raw.csv", index=False)
        self.df_test.to_csv(cfg.split_dir / "validated_test_raw.csv", index=False)

        logger.info(
            "Engine split locked: %d train engines / %d val engines (seed=%d, overlap=%s)",
            len(train_ids), len(val_ids), cfg.random_seed, sorted(overlap) or "none",
        )
        return self

    # ------------------------------------------------------------------
    # Data Requirements (restated dynamically from what was actually checked)
    # ------------------------------------------------------------------
    def compile_data_requirements(self) -> "DataValidator":
        cfg = self.config
        self.artifacts["data_requirements"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stage": "Data Collection, Exploration & Preparation — Data Validation",
            "non_functional": [
                "Validation must be reproducible from the raw CSV files alone",
                "No engine may appear in more than one split partition",
                "All validation artifacts are written to disk for downstream stages",
            ],
            "functional": [
                "Load NASA C-MAPSS FD001 raw files (PM_train.csv, PM_test.csv, PM_truth.csv)",
                "Validate schema (required columns, dtypes, train/test parity)",
                "Assess completeness, duplicates, and cycle-sequence integrity",
                "Flag constant / near-zero-variance sensors for the preparation stage",
                f"Lock a deterministic engine-wise train/val split (seed={cfg.random_seed}, val_ratio={cfg.val_ratio})",
            ],
            "analytical": [
                f"Minimum {cfg.min_engines_required} training engines required",
                "Zero missing values tolerated without an explicit downstream imputation step",
                "Zero engine overlap between train and validation partitions",
            ],
        }
        return self

    # ------------------------------------------------------------------
    # Summary / go-no-go gate
    # ------------------------------------------------------------------
    def compile_summary(self) -> dict:
        data_sufficient = len(self.errors) == 0
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_sufficient": data_sufficient,
            "decision": "PROCEED to Analysis / Modeling" if data_sufficient else "RETURN to Data Collection, Exploration & Preparation",
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "artifacts_generated": sorted(self.artifacts.keys()),
        }
        self.artifacts["validation_summary"] = summary
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_artifacts(self) -> None:
        out_dir = self.config.artifact_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for name, content in self.artifacts.items():
            if name == "data_dictionary":
                pd.DataFrame(content["features"]).to_csv(out_dir / "data_dictionary.csv", index=False)
            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, default=str)

        logger.info("Artifacts written to %s", out_dir)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self) -> dict:
        logger.info("=" * 70)
        logger.info("STAGE 2 — DATA VALIDATION")
        logger.info("=" * 70)
        (
            self.load_data()
            .describe_data()
            .validate_schema()
            .assess_quality()
            .validate_business_rules()
            .build_data_dictionary()
            .generate_engine_split()
            .compile_data_requirements()
        )
        summary = self.compile_summary()
        self.save_artifacts()

        logger.info("-" * 70)
        logger.info("Decision: %s (%d error(s), %d warning(s))", summary["decision"], summary["error_count"], summary["warning_count"])
        logger.info("=" * 70)
        return summary


def main() -> int:
    validator = DataValidator(ValidationConfig())
    summary = validator.run()
    return 0 if summary["data_sufficient"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
