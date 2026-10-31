"""
Retraining trigger (Continuous Training).

Implements the model-side counterpart to the CI/CD pipeline: where CI/CD reacts
to CODE changes, this reacts to MODEL DEGRADATION. It reads the live prediction
history that serving_api.py writes to `predictions.db`, computes the champion's
recent accuracy on calls for which the true RUL is known, and decides whether the
model has drifted far enough to warrant retraining.

This closes the loop that the prediction-history store made possible:

    serving_api.py  --writes-->  predictions.db  --read by-->  this trigger
                                                                   |
                                        drift above threshold?     |
                                             |                     |
                                            yes  --------> retrain + promote
                                             no   --------> do nothing

How degradation is measured
---------------------------
For every prediction whose `ground_truth_rul` is known, the absolute error
|predicted_rul - ground_truth_rul| is already stored. The trigger looks at the
most recent N such predictions for the current champion version and computes
their mean absolute error (MAE). If that live MAE exceeds a threshold, the model
is considered degraded. The threshold is expressed relative to the model's
training-time performance (its validation MAE), so "degraded" means "materially
worse in production than it was in validation", not an arbitrary absolute number.

Optionally the trigger also checks recall on the CRITICAL category, because in a
safety context, failing to flag an engine that is genuinely about to fail is the
most costly error, and a drop in critical-recall is a more operationally
meaningful alarm than overall MAE alone.

What it does on a breach
------------------------
By default the trigger runs in report-only mode: it prints/logs its decision and
exits with a non-zero status if retraining is warranted (so it can gate a
scheduled CI job). With --retrain it additionally invokes the existing training
pipeline (modeling.py), which retrains all candidate models, registers the best
one as a new version, and re-points the `champion` alias -- after which the
running serving service picks up the new model with no code change.

Usage
-----
    python retraining_trigger.py                 # report only
    python retraining_trigger.py --window 100    # look at last 100 labelled calls
    python retraining_trigger.py --mae-threshold 20
    python retraining_trigger.py --retrain       # actually retrain+promote on breach

Exit codes: 0 = healthy (no retraining needed), 2 = retraining warranted.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retraining_trigger")

ROOT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DB_PATH = ROOT_DIR / "predictions.db"
MODELING_SCRIPT = ROOT_DIR / "modeling.py"

# The champion's validation MAE is ~11 cycles. A live MAE meaningfully above that
# signals the model is performing worse in production than it did in validation.
# Default threshold leaves headroom for normal variance; tune per project.
DEFAULT_MAE_THRESHOLD = 18.0
# Minimum critical-recall before an alarm: of engines that were truly critical,
# at least this fraction must have been predicted critical.
DEFAULT_MIN_CRITICAL_RECALL = 0.70
# Minimum number of labelled predictions required to make a decision at all;
# too few and the estimate is noise.
DEFAULT_MIN_SAMPLES = 30


def _latest_champion_version(con: sqlite3.Connection) -> str | None:
    row = con.execute(
        "SELECT model_version FROM predictions "
        "WHERE ground_truth_rul IS NOT NULL ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def _recent_labelled(con: sqlite3.Connection, version: str, window: int) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT predicted_rul, ground_truth_rul, predicted_category, actual_category, abs_error "
        "FROM predictions "
        "WHERE model_version = ? AND ground_truth_rul IS NOT NULL "
        "ORDER BY ts DESC LIMIT ?",
        (version, window),
    ).fetchall()
    return rows


def _mean_abs_error(rows: list[sqlite3.Row]) -> float:
    errs = [r["abs_error"] for r in rows if r["abs_error"] is not None]
    return sum(errs) / len(errs) if errs else float("nan")


def _critical_recall(rows: list[sqlite3.Row]) -> float | None:
    # recall = (predicted critical AND truly critical) / (truly critical)
    truly_critical = [r for r in rows if r["actual_category"] == "critical"]
    if not truly_critical:
        return None  # no truly-critical engines in the window; recall undefined
    caught = [r for r in truly_critical if r["predicted_category"] == "critical"]
    return len(caught) / len(truly_critical)


def evaluate(window: int, mae_threshold: float, min_recall: float, min_samples: int) -> bool:
    """Returns True if retraining is warranted."""
    if not PREDICTIONS_DB_PATH.exists():
        logger.error("Prediction history not found at %s; cannot evaluate drift.", PREDICTIONS_DB_PATH)
        raise SystemExit(1)

    with sqlite3.connect(PREDICTIONS_DB_PATH) as con:
        version = _latest_champion_version(con)
        if version is None:
            logger.warning("No labelled predictions in history yet; nothing to evaluate.")
            return False

        rows = _recent_labelled(con, version, window)

    if len(rows) < min_samples:
        logger.info(
            "Only %d labelled predictions for champion v%s (need %d); skipping decision.",
            len(rows), version, min_samples,
        )
        return False

    mae = _mean_abs_error(rows)
    recall = _critical_recall(rows)
    recall_str = f"{recall:.2f}" if recall is not None else "n/a"

    logger.info(
        "Champion v%s over last %d labelled predictions: live MAE=%.2f, critical-recall=%s",
        version, len(rows), mae, recall_str,
    )

    mae_breach = mae > mae_threshold
    recall_breach = recall is not None and recall < min_recall

    if mae_breach:
        logger.warning("Live MAE %.2f exceeds threshold %.2f.", mae, mae_threshold)
    if recall_breach:
        logger.warning("Critical-recall %.2f below minimum %.2f.", recall, min_recall)

    warranted = mae_breach or recall_breach
    if warranted:
        logger.warning("RETRAINING WARRANTED for champion v%s.", version)
    else:
        logger.info("Champion v%s within thresholds; no retraining needed.", version)
    return warranted


def retrain() -> None:
    """Invoke the existing training pipeline, which retrains, registers the best
    model, and re-points the champion alias (see modeling.py)."""
    if not MODELING_SCRIPT.exists():
        logger.error("modeling.py not found at %s; cannot retrain.", MODELING_SCRIPT)
        raise SystemExit(1)
    logger.info("Starting retraining via %s ...", MODELING_SCRIPT)
    # -I: isolated mode, so the script's own directory logic is predictable.
    result = subprocess.run([sys.executable, "-I", str(MODELING_SCRIPT)], cwd=str(ROOT_DIR))
    if result.returncode != 0:
        logger.error("Retraining failed with exit code %d.", result.returncode)
        raise SystemExit(result.returncode)
    logger.info("Retraining complete; champion alias now points to the new best model.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RUL model retraining trigger (drift check).")
    parser.add_argument("--window", type=int, default=100,
                        help="Number of most recent labelled predictions to assess (default 100).")
    parser.add_argument("--mae-threshold", type=float, default=DEFAULT_MAE_THRESHOLD,
                        help=f"Live MAE above which retraining is warranted (default {DEFAULT_MAE_THRESHOLD}).")
    parser.add_argument("--min-critical-recall", type=float, default=DEFAULT_MIN_CRITICAL_RECALL,
                        help=f"Critical-recall below which retraining is warranted (default {DEFAULT_MIN_CRITICAL_RECALL}).")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help=f"Minimum labelled predictions required to decide (default {DEFAULT_MIN_SAMPLES}).")
    parser.add_argument("--retrain", action="store_true",
                        help="Actually retrain and promote on a breach (default: report only).")
    args = parser.parse_args()

    warranted = evaluate(args.window, args.mae_threshold, args.min_critical_recall, args.min_samples)

    if warranted and args.retrain:
        retrain()
        sys.exit(0)
    # Non-zero exit on breach lets a scheduled CI job detect "retraining needed".
    sys.exit(2 if warranted else 0)


if __name__ == "__main__":
    main()
