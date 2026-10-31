"""
Shared fixtures/helpers for the serving tests.

serving_api.py registers Prometheus metrics at import time, which may only
happen once per process. This module loads serving_api.py exactly once and
exposes it to all tests, so importing it from several test files does not
trigger a duplicate-metrics registration error.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVING_API = REPO_ROOT / "serving_api.py"

_MODULE_NAME = "serving_api_under_test"


def load_serving_api():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SERVING_API)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module
