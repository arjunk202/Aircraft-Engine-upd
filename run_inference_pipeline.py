#!/usr/bin/env python3
"""
run_inference_pipeline.py

INFERENCE half of the Aircraft Engine RUL pipeline (split out of the former
simulate_pipeline.py). Expects run_training_pipeline.py to have been run
first, so the "champion" model is registered in MLflow and the Feast
registry/data exist.

  Stage 4:   start serving_api.py, smoke-test /model_info
  Stage 5b:  (optional) bring up Prometheus + Grafana via `docker compose`
  Stage 6:   replay traffic through the API (simulate_traffic.py), then
             scrape /metrics for the artifact bundle

Pure Python, cross-platform (Windows/macOS/Linux) -- no bash required. It
does NOT create or install anything: activate your venv (with
requirements.txt already installed) yourself before running this, and every
step below runs through whatever `python`/`uvicorn` that activated venv puts
on PATH.

Usage (run from the repo root, i.e. the folder that CONTAINS
"Aircraft-Engine-Predictive-Maintenance_-main/", with your venv active):

    python run_inference_pipeline.py [--with-monitoring]

    --with-monitoring   Also brings up Prometheus/Grafana (via `docker
                        compose`) and leaves them AND serving_api.py running
                        when the script exits, instead of tearing
                        serving_api.py down. Requires a local Docker daemon.
                        Not meant for a CI container job (which wants a clean
                        exit with nothing left running) -- only for local,
                        interactive runs. If something is already listening
                        on the API port (e.g. left over from a previous
                        --with-monitoring run), it's reused as-is instead of
                        being restarted.

Why --with-monitoring also changes serving_api.py's teardown: Prometheus
marks a scraped series stale ~5 minutes after its target stops responding,
and a restarted serving_api.py resets every in-process Counter/Gauge back to
zero. Killing serving_api.py right after each run guarantees the Grafana
dashboard goes blank shortly after every single run, independent of any
dashboard/time-range setting. Leaving it running (only under
--with-monitoring) is what actually fixes that, not a PromQL or time-range
change.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PIPELINE_DIR_NAME = "Aircraft-Engine-Predictive-Maintenance_-main"
API_HOST = "127.0.0.1"
API_PORT = 8001
PROMETHEUS_PORT = 9090
GRAFANA_PORT = 3000
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
    letting subprocess raise an opaque FileNotFoundError later."""
    exe = shutil.which(name)
    if exe is None:
        fail(f"'{name}' not found on PATH -- activate your venv (with requirements.txt installed) before running this script.")
    return exe


def is_responding(url: str, timeout: float = 2.0) -> bool:
    """One-shot check -- used to detect a service already left running by an
    earlier --with-monitoring invocation, so it can be reused instead of
    fighting it for the port (or masking a fresh instance's bind failure)."""
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


def http_get_to_file(url: str, out_path: Path) -> None:
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(data.decode("utf-8", errors="replace"))


def ensure_background_service(
    name: str, check_url: str, cmd: list[str], cwd: Path, log_path: Path,
    env: dict | None, ready_timeout_s: float = 60,
) -> subprocess.Popen | None:
    """Starts a long-running server in the background unless one is already
    answering at check_url (left over from a previous --with-monitoring run),
    in which case it's reused as-is. Returns the Popen handle for a process
    THIS call started, or None if an existing one was reused (so the caller
    knows not to claim ownership of it)."""
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


def require_training_outputs(pipeline_dir: Path) -> None:
    """The inference half depends on artifacts written by the training half.
    Fail early with a clear message instead of a deep MLflow/Feast error."""
    required = [
        pipeline_dir / "mlflow.db",                                   # champion alias lives here
        pipeline_dir / "feature_repo" / "data" / "registry.db",       # Feast registry
    ]
    missing = [str(p.relative_to(pipeline_dir)) for p in required if not p.exists()]
    if missing:
        fail(
            "Missing training outputs: " + ", ".join(missing)
            + " -- run `python run_training_pipeline.py` first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the INFERENCE half of the RUL pipeline.")
    parser.add_argument("--with-monitoring", action="store_true", help="Also bring up Prometheus/Grafana via docker compose and leave everything running.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    pipeline_dir = repo_root / PIPELINE_DIR_NAME
    log_dir = repo_root / "ci_logs"

    if not pipeline_dir.is_dir():
        fail(f"Expected to run from the repo root (missing {PIPELINE_DIR_NAME}/). cd into the repo root first.")

    py = sys.executable  # whatever venv is already active -- see module docstring
    env = None            # None -> subprocess inherits the current (activated-venv) environment
    uvicorn_exe = require_on_path("uvicorn")
    if args.with_monitoring:
        require_on_path("docker")

    require_training_outputs(pipeline_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log("Stage 4 / Starting serving_api.py")
    health_url = f"http://{API_HOST}:{API_PORT}/health"
    api_proc = ensure_background_service(
        "serving_api", health_url,
        [uvicorn_exe, "serving_api:app", "--host", API_HOST, "--port", str(API_PORT)],
        pipeline_dir, log_dir / "08_serving_api.log", env,
    )
    # api_proc is None when an already-running instance (e.g. left up by a
    # previous --with-monitoring run) was reused instead -- never torn down
    # below either way, since this run doesn't own it.
    keep_api_running = args.with_monitoring or api_proc is None

    try:
        log("Stage 4b / model_info smoke check")
        http_get_to_file(f"http://{API_HOST}:{API_PORT}/model_info", log_dir / "09_model_info.json")

        if args.with_monitoring:
            # Brought up BEFORE the traffic replay so Prometheus is already
            # scraping serving_api while simulate_traffic.py runs, instead of
            # coming online only after serving_api has already been killed.
            log("Stage 5b / Bringing up Prometheus + Grafana (local Docker only)")
            monitoring_dir = pipeline_dir / "monitoring"
            subprocess.run(["docker", "compose", "up", "-d"], cwd=str(monitoring_dir), check=True)

        log("Stage 6 / Replaying traffic through the API (Utilization)")
        run_and_tee([py, "simulate_traffic.py", "--requests", "150"], log_dir / "10_simulate_traffic.log", cwd=pipeline_dir, env=env)

        log("Scraping /metrics for the artifact bundle")
        http_get_to_file(f"http://{API_HOST}:{API_PORT}/metrics", log_dir / "11_prometheus_metrics.txt")

    finally:
        if api_proc is not None and not keep_api_running:
            log(f"Stopping serving_api (pid {api_proc.pid})")
            api_proc.terminate()
            try:
                api_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_proc.kill()

    log(f"Inference pipeline complete. Logs and artifacts are in {log_dir}/")

    print("\n\033[1;34m==> Service URLs\033[0m")
    if keep_api_running:
        print(f"  Serving API   {health_url.rsplit('/', 1)[0]}  (docs: /docs, metrics: /metrics)")
    else:
        print("  Serving API   stopped after this run  (rerun with --with-monitoring to leave it up)")
    if args.with_monitoring:
        print(f"  Prometheus    http://localhost:{PROMETHEUS_PORT}")
        print(f"  Grafana       http://localhost:{GRAFANA_PORT}")
    else:
        print("  Prometheus / Grafana were not started -- rerun with --with-monitoring to bring them up.")


if __name__ == "__main__":
    main()
