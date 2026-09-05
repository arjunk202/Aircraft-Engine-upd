"""
One-time fix for moving this project to a new machine/folder.

mlflow.db's SQLite backend stores ABSOLUTE file:// paths to mlruns/ at the
time each experiment/run/model was created (baked in for whatever machine
first ran modeling.py). If the project folder is copied somewhere else —
different drive, different username, different OS path — those old absolute
paths point nowhere, and mlflow.pyfunc.load_model() fails with
"No such artifact: ''".

Five columns across four tables carry these absolute paths (all four must be
fixed together — ModelsArtifactRepo resolves models:/name@alias through
model_versions.storage_location specifically, not through logged_models):
    experiments.artifact_location
    runs.artifact_uri
    logged_models.artifact_location
    model_versions.storage_location
    model_versions.source

Safe to re-run: each column is fixed independently by finding its own stale
prefix, so it's fine if some columns were already fixed by a previous run
(e.g. a partial version of this script) and others weren't.

Usage:
    python fix_mlflow_paths.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "mlflow.db"
NEW_ROOT = Path(__file__).resolve().parent.as_posix()  # e.g. C:/Users/arjun/.../Aircraft-Engine-Predictive-Maintenance_-main

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

total = 0
for table, col in [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "storage_location"),
    ("model_versions", "source"),
]:
    # Find one stale row in THIS column to learn its old root — independent
    # per column, since some may already be fixed from a prior partial run.
    row = cur.execute(
        f"SELECT {col} FROM {table} WHERE {col} LIKE 'file:%mlruns%' AND {col} NOT LIKE ?",
        (f"%{NEW_ROOT}%",),
    ).fetchone()
    if not row:
        print(f"{table}.{col}: already up to date (or no file:// paths) — skipped")
        continue

    old_root = row[0].split("/mlruns/")[0].removeprefix("file:")
    cur.execute(
        f"UPDATE {table} SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ?",
        (old_root, NEW_ROOT, f"%{old_root}%"),
    )
    print(f"{table}.{col}: rewrote {old_root} -> {NEW_ROOT}  ({cur.rowcount} rows)")
    total += cur.rowcount

conn.commit()
conn.close()
print(f"\nDone. {total} rows updated. Re-run: uvicorn serving_api:app --host 127.0.0.1 --port 8001")
