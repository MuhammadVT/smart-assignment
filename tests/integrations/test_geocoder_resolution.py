"""
Geocoder resolution: the SMART_ASSIGNMENT_GEOCODER knob (mock | census, default
mock) so every surface picks the same provider and dev/demo runs are
deterministic and consistent across processes.
"""

from __future__ import annotations

import pytest

from smart_assignment.integrations.census_geocoder import CensusGeocoder
from smart_assignment.integrations.geocoding_client import MockGeocoder, resolve_geocoder

_ENV = "SMART_ASSIGNMENT_GEOCODER"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    yield


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, MockGeocoder),        # unset -> deterministic mock (default)
        ("mock", MockGeocoder),
        ("MOCK", MockGeocoder),
        ("bogus", MockGeocoder),     # unknown -> mock
        ("census", CensusGeocoder),
        ("Census", CensusGeocoder),
    ],
)
def test_resolve_geocoder(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(_ENV, value)
    assert isinstance(resolve_geocoder(), expected)


def test_default_mock_is_deterministic_across_calls(monkeypatch):
    # The whole point: same address -> same coordinates every time, so adk web
    # and the web app (separate processes) agree.
    addr = "1200 McKinney St, Houston, TX 77010"
    a = resolve_geocoder().geocode(addr)
    b = resolve_geocoder().geocode(addr)
    assert (a.latitude, a.longitude) == (b.latitude, b.longitude)
