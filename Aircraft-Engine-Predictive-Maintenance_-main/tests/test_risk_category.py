"""
Unit tests for the risk-categorisation logic in serving_api.py.

These exercise the pure business logic that maps a continuous RUL estimate to a
discrete critical/moderate/safe category. They need no model, no MLflow and no
network, so they run quickly and deterministically in CI.

Thresholds under test (defined in serving_api.py):
    critical  RUL <= 20
    moderate  20 < RUL < 60
    safe      RUL >= 60
"""

import pytest

from conftest import load_serving_api

risk_category = load_serving_api()._risk_category


@pytest.mark.parametrize(
    "rul, expected",
    [
        (0, "critical"),
        (10, "critical"),
        (20, "critical"),      # boundary: <= 20 is critical
        (20.5, "moderate"),
        (40, "moderate"),
        (59.9, "moderate"),
        (60, "safe"),          # boundary: >= 60 is safe
        (100, "safe"),
        (125, "safe"),
    ],
)
def test_risk_category_mapping(rul, expected):
    assert risk_category(rul) == expected


def test_boundaries_are_consistent():
    assert risk_category(20) == "critical"
    assert risk_category(20.0001) == "moderate"
    assert risk_category(59.9999) == "moderate"
    assert risk_category(60) == "safe"
