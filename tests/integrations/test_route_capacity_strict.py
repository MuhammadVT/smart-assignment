"""Strict-mode data source: a declared source that can't load must FAIL loudly
rather than silently fall back to the mock demo routes.

Off by default -- the existing fall-back-to-mock behavior is preserved for every
normal surface -- and turned on by the eval harness (see ``eval/dataset.py``) so
an eval never scores against silently-substituted data. Hermetic: the prepared
table loader is monkeypatched to raise, so no snapshot or SQL is touched.
"""

from __future__ import annotations

import pytest

from smart_assignment.integrations import route_capacity_client as rc


def _raise_no_snapshot():
    raise RuntimeError("no cache snapshot")


def test_strict_off_falls_back_to_mock(monkeypatch):
    # Default behavior: a cache load failure quietly yields the mock routes.
    monkeypatch.delenv(rc._STRICT_ENV, raising=False)
    monkeypatch.setattr(rc, "_load_prepared_route_tables", _raise_no_snapshot)

    routes = rc._fetch_candidate_routes_uncached(rc.SOURCE_CACHE)

    assert routes == rc._mock_routes()  # silent fallback preserved when not strict


def test_strict_on_raises_instead_of_falling_back(monkeypatch):
    # Strict mode: the same failure is a loud error, never a swap to mock.
    monkeypatch.setenv(rc._STRICT_ENV, "1")
    monkeypatch.setattr(rc, "_load_prepared_route_tables", _raise_no_snapshot)

    with pytest.raises(RuntimeError, match="strict mode is on"):
        rc._fetch_candidate_routes_uncached(rc.SOURCE_CACHE)


def test_strict_mode_flag_parsing(monkeypatch):
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(rc._STRICT_ENV, truthy)
        assert rc._strict_data_source() is True
    for falsy in ("", "0", "false", "no", "off"):
        monkeypatch.setenv(rc._STRICT_ENV, falsy)
        assert rc._strict_data_source() is False
    monkeypatch.delenv(rc._STRICT_ENV, raising=False)
    assert rc._strict_data_source() is False
