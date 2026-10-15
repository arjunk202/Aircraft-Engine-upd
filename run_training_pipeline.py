#!/usr/bin/env python3
"""
run_training_pipeline.py

TRAINING half of the Aircraft Engine RUL pipeline (split out of the former
simulate_pipeline.py). Produces everything the inference pipeline needs:

  Stage 0:  (optional) clean previously generated artifacts
  Stage 2:  data_validation -> eda -> feature_engineering -> Feast sources
            -> feast apply / materialize-incremental
  Stage 3:  modeling (trains 5 regressors, logs to MLflow, registers "champion")

Run run_inference_pipeline.py afterwards to serve the champion model, replay
traffic and (optionally) bring up Prometheus/Grafana.

Pure Python, cross-platform (Windows/macOS/Linux) -- no bash required. It
does NOT create or install anything: activate your venv (with
requirements.txt already installed) yourself before running this, and every
step below runs through whatever `python`/`feast` that activated venv puts
on PATH.

Usage (run from the repo root, i.e. the folder that CONTAINS
"Aircraft-Engine-Predictive-Maintenance_-main/", with your venv active):

    python run_training_pipeline.py [--clean] [--with-ui]

    --clean     Delete previously generated artifacts first, so the run is a
                true from-scratch reproduction instead of appending to
                whatever is already checked in.
    --with-ui   Also start the Feast UI and the MLflow UI in the background
                and leave them running when the script exits. Local,
                interactive use only -- not meant for a CI job that wants a
                clean exit with nothing left running. If something is
                already listening on one of these ports (e.g. left over from
                a previous --with-ui run) it is reused as-is instead of
                being restarted.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_DIR_NAME = "Aircraft-Engine-Predictive-Maintenance_-main"
FEAST_UI_PORT = 8888
MLFLOW_UI_PORT = 5000
HEALTH_TIMEOUT_S = 60
HEALTH_POLL_S = 2


# --------------------------------------------------------------------------
# Small console helpers
# --------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"\n\033[1;34m==> {msg}\033[0m")


def fail(msg: str) -> None:
    print(f"\033[1;31mERROR: {msg}\033[0m", file=sys.stderr)
    sys.exit(1)


def run_and_tee(cmd: list[str], log_path: Path, cwd: Path, env: dict | None = None) -> None:
    """Run a command, streaming its output to the console AND a log file
    (the Python equivalent of `cmd | tee log_path`). Exits on non-zero exit,
    mirroring `set -e` in a bash version."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fh.write(line)
        proc.wait()
    if proc.returncode != 0:
        fail(f"Command failed ({proc.returncode}): {' '.join(cmd)}  -- see {log_path}")


def require_on_path(name: str) -> str:
    """Resolves a console-script from PATH -- i.e. from whatever venv the
    caller already activated -- and fails loudly if it's missing, rather than
    letting subprocess raise an opaque FileNotFoundError deep in Stage 2e."""
    exe = shutil.which(name)
    if exe is None:
        fail(f"'{name}' not found on PATH -- activate your venv (with requirements.txt installed) before running this script.")
    return exe


def is_responding(url: str, timeout: float = 2.0) -> bool:
    """One-shot check -- used to detect a service already left running by an
    earlier --with-ui invocation, so it can be reused instead of fighting it
    for the port (or masking a fresh instance's bind failure)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def wait_for_health(url: str, timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_responding(url, timeout=5):
            return True
        time.sleep(HEALTH_POLL_S)
    return False


def ensure_background_service(
    name: str, check_url: str, cmd: list[str], cwd: Path, log_path: Path,
    env: dict | None, ready_timeout_s: float = 60,
) -> subprocess.Popen | None:
    """Starts a long-running UI/server in the background unless one is
    already answering at check_url, in which case it's reused as-is. Returns
    the Popen handle for a process THIS call started, or None if an existing
    one was reused."""
    if is_responding(check_url):
        log(f"{name} already running at {check_url} -- reusing it")
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    if not wait_for_health(check_url, timeout_s=ready_timeout_s):
        log_fh.flush()
        print(log_path.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
        fail(f"{name} did not become ready within {ready_timeout_s}s -- see {log_path}")
    log(f"{name} is up at {check_url}")
    return proc


# --------------------------------------------------------------------------
def clean_generated_artifacts(pipeline_dir: Path) -> None:
    log("Stage 0: Cleaning previously generated pipeline artifacts")
    # Only remove what the CURRENT scripts regenerate on every run.
    # MLOPS_PIPELINE/ is a legacy snapshot from an earlier (pre-Feast/MLflow)
    # version of the pipeline that nothing today rewrites -- left alone.
    targets = [
        pipeline_dir / "Data_Report_Artifacts",
        pipeline_dir / "mlruns",
        pipeline_dir / "mlflow.db",
        pipeline_dir / "feature_repo" / "data" / "registry.db",
        pipeline_dir / "feature_repo" / "data" / "online_store.db",
        pipeline_dir / "feature_repo" / "data" / "engine_train.parquet",
        pipeline_dir / "feature_repo" / "data" / "engine_val.parquet",
        pipeline_dir / "feature_repo" / "data" / "engine_test.parquet",
        pipeline_dir / "feature_repo" / "data" / "engineered_train.parquet",
        pipeline_dir / "feature_repo" / "data" / "engineered_val.parquet",
        pipeline_dir / "feature_repo" / "data" / "engineered_test.parquet",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TRAINING half of the RUL pipeline.")
    parser.add_argument("--clean", action="store_true", help="Delete previously generated artifacts first.")
    parser.add_argument("--with-ui", action="store_true", help="Also start the Feast UI and MLflow UI and leave them running.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    pipeline_dir = repo_root / PIPELINE_DIR_NAME
    log_dir = repo_root / "ci_logs"

    if not pipeline_dir.is_dir():
        fail(f"Expected to run from the repo root (missing {PIPELINE_DIR_NAME}/). cd into the repo root first.")

    py = sys.executable  # whatever venv is already active -- see module docstring
    env = None            # None -> subprocess inherits the current (activated-venv) environment
    feast_exe = require_on_path("feast")

    if args.clean:
        clean_generated_artifacts(pipeline_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    log("Stage 2a / Data Validation")
    run_and_tee([py, "data_validation.py"], log_dir / "01_data_validation.log", cwd=pipeline_dir, env=env)

    log("Stage 2b / EDA")
    run_and_tee([py, "eda.py"], log_dir / "02_eda.log", cwd=pipeline_dir, env=env)

    log("Stage 2c / Feature Engineering")
    run_and_tee([py, "feature_engineering.py"], log_dir / "03_feature_engineering.log", cwd=pipeline_dir, env=env)

    log("Stage 2d / Bridging pipeline outputs into Feast sources")
    run_and_tee([py, "build_feature_store_sources.py"], log_dir / "04_feature_store_sources.log", cwd=pipeline_dir, env=env)

    log("Stage 2e / feast apply + materialize-incremental")
    feature_repo_dir = pipeline_dir / "feature_repo"
    run_and_tee([feast_exe, "apply"], log_dir / "05a_feast_apply.log", cwd=feature_repo_dir, env=env)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    run_and_tee([feast_exe, "materialize-incremental", ts], log_dir / "05b_feast_materialize.log", cwd=feature_repo_dir, env=env)

    feast_ui_url = f"http://127.0.0.1:{FEAST_UI_PORT}/"
    if args.with_ui:
        log("Stage 2f / Starting Feast UI")
        ensure_background_service(
            "Feast UI", feast_ui_url, [feast_exe, "ui"], feature_repo_dir,
            log_dir / "05c_feast_ui.log", env,
        )

    log("Stage 3 / Modeling")
    run_and_tee([py, "modeling.py"], log_dir / "06_modeling.log", cwd=pipeline_dir, env=env)

    mlflow_ui_url = f"http://127.0.0.1:{MLFLOW_UI_PORT}/"
    if args.with_ui:
        log("Stage 3b / Starting MLflow UI")
        # run_mlflow_ui.py, not a bare `mlflow ui` -- see its own docstring:
        # works around a Starlette/anyio bug in MLflow's UI server that
        # otherwise 500s every request.
        ensure_background_service(
            "MLflow UI", mlflow_ui_url,
            [py, "run_mlflow_ui.py", "--backend-store-uri", "sqlite:///mlflow.db", "--workers", "1"],
            pipeline_dir, log_dir / "07_mlflow_ui.log", env,
        )

    log(f"Training pipeline complete. Logs and artifacts are in {log_dir}/")

    print("\n\033[1;34m==> Next step\033[0m")
    print("  python run_inference_pipeline.py [--with-monitoring]")
    if args.with_ui:
        print("\n\033[1;34m==> Service URLs\033[0m")
        print(f"  Feast UI      {feast_ui_url}")
        print(f"  MLflow UI     {mlflow_ui_url}")
    else:
        print("  (Feast UI / MLflow UI were not started -- rerun with --with-ui to bring them up.)")


if __name__ == "__main__":
    main()
