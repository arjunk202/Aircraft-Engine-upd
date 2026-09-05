"""
Exploratory Data Analysis — Stage 2 (Data Collection, Exploration & Preparation)
of the MLOps-based Data Science Process Model (DSPM).

Explores the validated, engine-wise TRAIN partition produced by
data_validation.py (Data_Report_Artifacts/data_validation/validated_splits/
validated_train_raw.csv). EDA intentionally only looks at TRAIN — the
validation partition is held out even at this stage so no exploratory
decision (which sensors look predictive, which look redundant, ...) can leak
information from the engines reserved for evaluation.

A temporary, exploratory-only RUL column (max_cycle - cycle, per engine) is
derived purely to characterize sensor-degradation relationships. It is not
clipped, not persisted as a modeling target, and not written back into any
split file — formalizing the target (including the RUL clip) is a Data
Preparation-stage decision, not an EDA one.

Produces the Data Exploration Report referenced in the DSPM UML
information-flow diagram plus supporting plots, both dynamically computed
from whatever is in validated_train_raw.csv (no hardcoded figures).

Run after data_validation.py:

    python eda.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eda")

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = ROOT_DIR / "Data_Report_Artifacts" / "data_validation" / "validated_splits" / "validated_train_raw.csv"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "Data_Report_Artifacts" / "eda"
DEFAULT_PLOT_DIR = DEFAULT_ARTIFACT_DIR / "plots"


@dataclass
class EDAConfig:
    train_path: Path = DEFAULT_TRAIN_PATH
    id_col: str = "id"
    cycle_col: str = "cycle"
    high_corr_threshold: float = 0.9
    skew_threshold: float = 1.0
    monotonic_trend_threshold: float = 0.5
    artifact_dir: Path = field(default_factory=lambda: DEFAULT_ARTIFACT_DIR)
    plot_dir: Path = field(default_factory=lambda: DEFAULT_PLOT_DIR)


class EDAAnalyzer:
    """Explores the validated TRAIN partition and emits the Data Exploration Report."""

    def __init__(self, config: EDAConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: list[str] = []
        self.report: dict = {}

    # ------------------------------------------------------------------
    def load_data(self) -> "EDAAnalyzer":
        cfg = self.config
        if not cfg.train_path.exists():
            raise FileNotFoundError(f"{cfg.train_path} not found — run data_validation.py first")

        self.df = pd.read_csv(cfg.train_path)
        self.feature_cols = [c for c in self.df.columns if c.startswith("s") or c.startswith("setting")]

        # exploratory-only RUL, never persisted
        max_cycle = self.df.groupby(cfg.id_col)[cfg.cycle_col].transform("max")
        self.df["_exploratory_rul"] = max_cycle - self.df[cfg.cycle_col]

        cfg.plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loaded TRAIN partition for EDA: %s (%d engines)", self.df.shape, self.df[cfg.id_col].nunique())
        return self

    # ------------------------------------------------------------------
    # Univariate distribution summary
    # ------------------------------------------------------------------
    def summarize_distributions(self) -> "EDAAnalyzer":
        summary = {}
        skewed = []
        for col in self.feature_cols:
            series = self.df[col]
            skew = float(series.skew())
            entry = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "skew": skew,
                "kurtosis": float(series.kurtosis()),
            }
            summary[col] = entry
            if abs(skew) >= self.config.skew_threshold:
                skewed.append({"feature": col, "skew": round(skew, 3)})

        self.report["distribution_summary"] = summary
        self.report["skewed_features"] = sorted(skewed, key=lambda x: abs(x["skew"]), reverse=True)
        logger.info("Distribution summary computed for %d features (%d skewed)", len(self.feature_cols), len(skewed))
        return self

    # ------------------------------------------------------------------
    # Engine lifecycle characterization
    # ------------------------------------------------------------------
    def analyze_engine_lifecycle(self) -> "EDAAnalyzer":
        cfg = self.config
        lifetimes = self.df.groupby(cfg.id_col)[cfg.cycle_col].max()
        self.report["engine_lifecycle"] = {
            "engine_count": int(lifetimes.shape[0]),
            "min_cycles": int(lifetimes.min()),
            "max_cycles": int(lifetimes.max()),
            "mean_cycles": round(float(lifetimes.mean()), 2),
            "median_cycles": float(lifetimes.median()),
            "std_cycles": round(float(lifetimes.std()), 2),
        }

        plt.figure(figsize=(10, 4))
        sns.histplot(lifetimes, bins=30, kde=True, color="steelblue")
        plt.title("Engine Lifetime Distribution (TRAIN)")
        plt.xlabel("Total Cycles Before End of Trajectory")
        plt.ylabel("Number of Engines")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "engine_lifetime_distribution.png", dpi=120)
        plt.close()

        plt.figure(figsize=(10, 4))
        sns.histplot(self.df["_exploratory_rul"], bins=40, kde=True, color="steelblue")
        plt.title("Exploratory RUL Distribution (TRAIN, unclipped)")
        plt.xlabel("max_cycle - cycle")
        plt.ylabel("Row Count")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "exploratory_rul_distribution.png", dpi=120)
        plt.close()

        logger.info("Engine lifecycle analysis complete (plots saved)")
        return self

    # ------------------------------------------------------------------
    # Correlation structure: feature-feature redundancy + feature-RUL signal
    # ------------------------------------------------------------------
    def analyze_correlations(self) -> "EDAAnalyzer":
        cfg = self.config
        corr_matrix = self.df[self.feature_cols].corr()

        high_corr_pairs = []
        for col_a, col_b in combinations(self.feature_cols, 2):
            value = corr_matrix.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= cfg.high_corr_threshold:
                high_corr_pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": round(float(value), 4)})
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        rul_corr = self.df[self.feature_cols + ["_exploratory_rul"]].corr()["_exploratory_rul"].drop("_exploratory_rul")
        top_rul_corr = (
            rul_corr.abs().sort_values(ascending=False).head(10).index.tolist()
        )
        top_rul_corr_detail = [{"feature": f, "correlation": round(float(rul_corr[f]), 4)} for f in top_rul_corr]

        self.report["correlation_analysis"] = {
            "high_correlation_pairs": high_corr_pairs,
            "high_correlation_threshold": cfg.high_corr_threshold,
            "top_rul_correlated_features": top_rul_corr_detail,
        }

        plt.figure(figsize=(14, 11))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, linewidths=0.4, cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Heatmap (TRAIN)")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "correlation_heatmap.png", dpi=120)
        plt.close()

        plt.figure(figsize=(10, 6))
        rul_corr.sort_values().plot(kind="barh", color="steelblue")
        plt.title("Feature Correlation with Exploratory RUL")
        plt.xlabel("Correlation")
        plt.axvline(x=0, color="red", linestyle="--")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "rul_correlation_barh.png", dpi=120)
        plt.close()

        logger.info(
            "Correlation analysis complete: %d high-correlation pairs (|r|>=%.2f)",
            len(high_corr_pairs), cfg.high_corr_threshold,
        )
        return self

    # ------------------------------------------------------------------
    # Degradation trend detection (per-engine monotonicity vs cycle)
    # ------------------------------------------------------------------
    def analyze_degradation_trends(self) -> "EDAAnalyzer":
        cfg = self.config

        def spearman_vs_cycle(g: pd.DataFrame, col: str) -> float:
            # a constant cycle or constant feature within an engine makes the
            # correlation undefined (scipy would just warn and return NaN) —
            # skip the call entirely instead of triggering that warning
            if g[cfg.cycle_col].nunique() <= 1 or g[col].nunique() <= 1:
                return np.nan
            return g[cfg.cycle_col].corr(g[col], method="spearman")

        trend_scores = {}
        for col in self.feature_cols:
            per_engine_corr = self.df.groupby(cfg.id_col)[[cfg.cycle_col, col]].apply(
                lambda g, col=col: spearman_vs_cycle(g, col)
            )
            trend_scores[col] = float(per_engine_corr.mean(skipna=True))

        monotonic_features = [
            {"feature": col, "mean_spearman_vs_cycle": round(score, 4)}
            for col, score in trend_scores.items()
            if pd.notna(score) and abs(score) >= cfg.monotonic_trend_threshold
        ]
        monotonic_features.sort(key=lambda x: abs(x["mean_spearman_vs_cycle"]), reverse=True)

        self.report["degradation_trend_analysis"] = {
            "mean_spearman_vs_cycle": {k: round(v, 4) if pd.notna(v) else None for k, v in trend_scores.items()},
            "monotonic_trend_features": monotonic_features,
            "monotonic_trend_threshold": cfg.monotonic_trend_threshold,
        }

        top_features = [f["feature"] for f in monotonic_features[:6]] or self.feature_cols[:6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        sample_engines = self.df[cfg.id_col].drop_duplicates().sample(min(15, self.df[cfg.id_col].nunique()), random_state=42)
        sample = self.df[self.df[cfg.id_col].isin(sample_engines)]
        for i, col in enumerate(top_features):
            for engine_id, group in sample.groupby(cfg.id_col):
                axes[i].plot(group[cfg.cycle_col], group[col], alpha=0.4, linewidth=0.8)
            axes[i].set_title(col)
            axes[i].set_xlabel(cfg.cycle_col)
            axes[i].set_ylabel(col)
        plt.suptitle("Sensor Trends vs Cycle — Strongest Monotonic Signals (sampled engines)")
        plt.tight_layout()
        plt.savefig(cfg.plot_dir / "degradation_trends.png", dpi=120)
        plt.close()

        logger.info("Degradation trend analysis complete: %d features with |spearman|>=%.2f", len(monotonic_features), cfg.monotonic_trend_threshold)
        return self

    # ------------------------------------------------------------------
    # Recommendations for the next (Feature Engineering) stage
    # ------------------------------------------------------------------
    def compile_recommendations(self) -> "EDAAnalyzer":
        redundant = {p["feature_b"] for p in self.report["correlation_analysis"]["high_correlation_pairs"]}
        trend_features = [f["feature"] for f in self.report["degradation_trend_analysis"]["monotonic_trend_features"]]

        recommendations = []
        if trend_features:
            recommendations.append(
                f"{len(trend_features)} feature(s) show a consistent monotonic trend with cycle "
                f"({', '.join(trend_features[:8])}{'...' if len(trend_features) > 8 else ''}); "
                "prioritize these for rolling-window (mean/std/trend) feature engineering."
            )
        if redundant:
            recommendations.append(
                f"{len(redundant)} feature(s) are highly correlated (|r|>={self.config.high_corr_threshold}) with another "
                f"feature already in the set ({', '.join(sorted(redundant)[:8])}{'...' if len(redundant) > 8 else ''}); "
                "consider dropping one of each redundant pair to reduce dimensionality."
            )
        if self.report["skewed_features"]:
            skew_names = [f["feature"] for f in self.report["skewed_features"][:8]]
            recommendations.append(
                f"{len(self.report['skewed_features'])} feature(s) are notably skewed ({', '.join(skew_names)}"
                f"{'...' if len(self.report['skewed_features']) > 8 else ''}); consider scaling method robust to skew."
            )
        recommendations.append(
            "Target formalization (RUL = max_cycle - cycle, with a clip threshold) belongs to the "
            "Data Preparation stage, not EDA — this report's RUL figures are exploratory only."
        )

        self.report["recommendations"] = recommendations
        return self

    # ------------------------------------------------------------------
    def save_artifacts(self) -> dict:
        cfg = self.config
        cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(cfg.train_path.relative_to(ROOT_DIR)),
            "engines_analyzed": int(self.df[self.config.id_col].nunique()),
            "rows_analyzed": int(self.df.shape[0]),
            **self.report,
            "plots_generated": sorted(p.name for p in cfg.plot_dir.glob("*.png")),
        }

        with open(cfg.artifact_dir / "data_exploration.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Data Exploration Report written to %s", cfg.artifact_dir / "data_exploration.json")
        return report

    # ------------------------------------------------------------------
    def run(self) -> dict:
        logger.info("=" * 70)
        logger.info("STAGE 2 — EXPLORATORY DATA ANALYSIS (TRAIN partition only)")
        logger.info("=" * 70)
        (
            self.load_data()
            .summarize_distributions()
            .analyze_engine_lifecycle()
            .analyze_correlations()
            .analyze_degradation_trends()
            .compile_recommendations()
        )
        report = self.save_artifacts()
        logger.info("-" * 70)
        for rec in report["recommendations"]:
            logger.info("- %s", rec)
        logger.info("=" * 70)
        return report


def main() -> int:
    EDAAnalyzer(EDAConfig()).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
