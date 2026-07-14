"""
Data-source resolution for route_capacity_client: the SMART_ASSIGNMENT_DATA_SOURCE
knob (mock | cache | live_sql, default cache), the deprecated ROUTE_SOURCE alias,
and the graceful mock fallback when a requested cache snapshot is absent.
"""

from __future__ import annotations

import pytest

from smart_assignment.integrations import route_capacity_client as rc

_DATA = "SMART_ASSIGNMENT_DATA_SOURCE"
_LEGACY = "SMART_ASSIGNMENT_ROUTE_SOURCE"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_DATA, raising=False)
    monkeypatch.delenv(_LEGACY, raising=False)
    yield


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, rc.SOURCE_CACHE),        # unset -> default cache
        ("mock", rc.SOURCE_MOCK),
        ("cache", rc.SOURCE_CACHE),
        ("cached", rc.SOURCE_CACHE),    # synonym
        ("live_sql", rc.SOURCE_LIVE_SQL),
        ("LIVE", rc.SOURCE_LIVE_SQL),   # case-insensitive synonym
        ("nonsense", rc.SOURCE_CACHE),  # unknown -> cache
    ],
)
def test_data_source_resolution(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(_DATA, value)
    assert rc._data_source() == expected


def test_legacy_route_source_is_honored_and_maps_prepared_to_live_sql(monkeypatch):
    monkeypatch.setenv(_LEGACY, "prepared")
    assert rc._data_source() == rc.SOURCE_LIVE_SQL
    monkeypatch.setenv(_LEGACY, "mock")
    assert rc._data_source() == rc.SOURCE_MOCK


def test_data_source_takes_precedence_over_legacy(monkeypatch):
    monkeypatch.setenv(_DATA, "mock")
    monkeypatch.setenv(_LEGACY, "prepared")
    assert rc._data_source() == rc.SOURCE_MOCK


def test_mock_source_returns_the_demo_routes(monkeypatch):
    monkeypatch.setenv(_DATA, "mock")
    routes = rc.fetch_candidate_routes()
    assert {r.route_id for r in routes} >= {"RTE-4100"}


def test_cache_source_falls_back_to_mock_when_snapshot_missing(monkeypatch):
    # No cache snapshot exists in the repo, so a cache request must degrade to
    # the mock demo routes rather than crash.
    monkeypatch.setenv(_DATA, "cache")
    routes = rc.fetch_candidate_routes()
    assert routes  # did not raise
    assert {r.route_id for r in routes} >= {"RTE-4100"}
