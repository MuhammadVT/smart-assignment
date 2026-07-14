"""
Geocoder resolution: the SMART_ASSIGNMENT_GEOCODER knob (census | mock, default
census) so every surface picks the same provider from the same config.
Constructing a geocoder makes no network call, so these stay offline.
"""

from __future__ import annotations

import pytest

from smart_assignment.integrations.census_geocoder import CensusGeocoder
from smart_assignment.integrations.geocoding_client import MockGeocoder, resolve_geocoder

_ENV = "SMART_ASSIGNMENT_GEOCODER"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # The suite-wide conftest pins this to "mock"; drop it so we can exercise the
    # real default (census) and explicit values here.
    monkeypatch.delenv(_ENV, raising=False)
    yield


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, CensusGeocoder),      # unset -> live census (default)
        ("census", CensusGeocoder),
        ("Census", CensusGeocoder),
        ("bogus", CensusGeocoder),   # unknown -> census
        ("mock", MockGeocoder),
        ("MOCK", MockGeocoder),
    ],
)
def test_resolve_geocoder(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(_ENV, value)
    assert isinstance(resolve_geocoder(), expected)


def test_mock_is_deterministic_across_calls(monkeypatch):
    monkeypatch.setenv(_ENV, "mock")
    addr = "1200 McKinney St, Houston, TX 77010"
    a = resolve_geocoder().geocode(addr)
    b = resolve_geocoder().geocode(addr)
    assert (a.latitude, a.longitude) == (b.latitude, b.longitude)
