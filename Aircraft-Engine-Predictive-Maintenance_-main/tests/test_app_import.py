"""
Import-level smoke tests: confirm serving_api.py is valid and its request schema
is well-formed, without starting the service or loading a model.
"""

import pydantic
import pytest

from conftest import load_serving_api

MODULE = load_serving_api()


def test_module_imports():
    assert MODULE.app is not None
    assert MODULE.app.title == "Aircraft Engine RUL Inference API"


def test_request_schema_requires_features():
    with pytest.raises(pydantic.ValidationError):
        MODULE.PredictRequest()  # type: ignore[call-arg]


def test_valid_request_parses():
    req = MODULE.PredictRequest(id=7, features={"cycle": 100.0})
    assert req.id == 7
    assert req.features["cycle"] == 100.0
