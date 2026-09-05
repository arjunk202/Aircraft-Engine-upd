"""
Workaround launcher for `mlflow ui` on this anyio/starlette combo.

MLflow's UI server wraps its legacy Flask app in Starlette's WSGI bridge
(starlette.middleware.wsgi), which is deprecated upstream and calls
`anyio.from_thread.run(...)` as a bare attribute access. That only works if
`anyio.from_thread` was already imported as a submodule somewhere earlier in
the process — normally something else does this as a side effect, but here
nothing does, so every request 500s with:
    AttributeError: module 'anyio' has no attribute 'from_thread'

Fix: explicitly import anyio.from_thread before MLflow's server starts, which
registers it as an attribute on the anyio package for the rest of the
process. Nothing else changes.

Usage (same flags as `mlflow ui`):
    python run_mlflow_ui.py --backend-store-uri sqlite:///mlflow.db --workers 1
"""

import anyio.from_thread  # noqa: F401 (import side effect is the fix — see module docstring)
import sys

from mlflow.cli import cli

if __name__ == "__main__":
    sys.argv = ["mlflow", "ui", *sys.argv[1:]]
    cli()
