"""
Bridges Stage 2 pipeline outputs into the Feast feature store.

Feast requires an event timestamp per row. The NASA C-MAPSS data has no real
wall-clock timestamps, only an engine-relative `cycle` counter, so a synthetic
`event_timestamp` is derived from it (one synthetic hour per cycle). This is
sufficient for offline point-in-time feature retrieval; nothing about it
implies real elapsed time.

Converts two source families, one parquet file per split each:
  - validated_{train,val,test}_raw.csv   (data_validation.py)   -> engine_{split}.parquet
  - {train,val,test}_features.csv        (feature_engineering.py) -> engineered_{split}.parquet

Run after data_validation.py (and, for the engineered sources, after
feature_engineering.py too):

    python build_feature_store_sources.py

Then, from feature_repo/:

    feast apply
    feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
VALIDATED_SPLIT_DIR = ROOT_DIR / "Data_Report_Artifacts" / "data_validation" / "validated_splits"
ENGINEERED_SPLIT_DIR = ROOT_DIR / "Data_Report_Artifacts" / "feature_engineering" / "engineered_splits"
FEATURE_REPO_DATA_DIR = ROOT_DIR / "feature_repo" / "data"

SYNTHETIC_EPOCH = pd.Timestamp("2020-01-01")


def to_feast_source(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    df["event_timestamp"] = SYNTHETIC_EPOCH + pd.to_timedelta(df["cycle"], unit="h")
    df["created_timestamp"] = df["event_timestamp"]
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({df.shape[0]} rows, {df.shape[1]} columns)")


def build_from(split_dir: Path, filename_template: str, output_prefix: str, required: bool) -> None:
    if not split_dir.exists():
        if required:
            raise FileNotFoundError(f"{split_dir} not found — run data_validation.py first")
        print(f"skipping {output_prefix}_* sources: {split_dir} not found (run feature_engineering.py first)")
        return

    FEATURE_REPO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        csv_path = split_dir / filename_template.format(split=split)
        if csv_path.exists():
            to_feast_source(csv_path, FEATURE_REPO_DATA_DIR / f"{output_prefix}_{split}.parquet")


def main() -> None:
    build_from(VALIDATED_SPLIT_DIR, "validated_{split}_raw.csv", "engine", required=True)
    build_from(ENGINEERED_SPLIT_DIR, "{split}_features.csv", "engineered", required=False)


if __name__ == "__main__":
    main()
