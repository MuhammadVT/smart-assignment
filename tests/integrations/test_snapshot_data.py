"""Tests for the self-contained snapshot substrate: serialization round-trip, the
`snapshot` route data source, and the SnapshotGeocoder."""

from __future__ import annotations

from datetime import time

import pytest

from smart_assignment.integrations import route_capacity_client as rcc
from smart_assignment.integrations import snapshot_data
from smart_assignment.integrations.geocoding_client import SnapshotGeocoder
from smart_assignment.shared.geo import AddressNotFoundError
from smart_assignment.shared.models import DayOfWeek, GeoPoint, Route, RouteStop


def _sample_routes():
    return [
        Route(
            route_id="RTE-1",
            name="Central",
            day=DayOfWeek.TUE,
            service_center=GeoPoint(29.76, -95.37),
            service_radius_miles=12.0,
            vehicle_capacity_cases=1000.0,
            avg_load_cases=580.0,
            available_windows=[(time(7, 0), time(10, 0))],
            committed_stops=[
                RouteStop(
                    customer_number="STOP-001",
                    location=GeoPoint(29.75, -95.36),
                    delivery_time_window=(time(8, 0), time(9, 0)),
                    customer_tier="5",
                ),
            ],
        ),
    ]


def test_route_round_trip_is_lossless():
    routes = _sample_routes()
    restored = snapshot_data.deserialize_routes(snapshot_data.serialize_routes(routes))
    assert restored == routes  # dataclass equality across every nested field


def test_geocode_round_trip():
    mapping = {"1 Main St": GeoPoint(29.7, -95.3)}
    restored = snapshot_data.deserialize_geocode(snapshot_data.serialize_geocode(mapping))
    assert restored == mapping


def test_write_and_load_bundle(tmp_path):
    routes = _sample_routes()
    geocode = {"1 Main St": GeoPoint(29.7, -95.3)}
    cases = [{"eval_id": "c1", "context": {"address": "1 Main St"}}]
    manifest = {"source": "test", "count": 1}
    snapshot_data.write_bundle(
        str(tmp_path), routes=routes, geocode=geocode, cases=cases, manifest=manifest
    )
    assert snapshot_data.load_routes(str(tmp_path)) == routes
    assert snapshot_data.load_geocode(str(tmp_path)) == geocode
    assert snapshot_data.load_cases(str(tmp_path)) == cases
    assert snapshot_data.load_manifest(str(tmp_path))["source"] == "test"


def test_snapshot_data_source_serves_bundle_routes(tmp_path, monkeypatch):
    snapshot_data.write_bundle(
        str(tmp_path), routes=_sample_routes(), geocode={}, cases=[], manifest={"source": "t"}
    )
    monkeypatch.setenv("SMART_ASSIGNMENT_DATA_SOURCE", "snapshot")
    monkeypatch.setenv(snapshot_data.SNAPSHOT_DIR_ENV, str(tmp_path))
    rcc.clear_route_cache()
    try:
        routes = rcc.fetch_candidate_routes()
        assert [r.route_id for r in routes] == ["RTE-1"]
        assert routes[0].committed_stops[0].customer_tier == "5"
    finally:
        rcc.clear_route_cache()


def test_snapshot_source_without_dir_is_loud(monkeypatch):
    monkeypatch.setenv("SMART_ASSIGNMENT_DATA_SOURCE", "snapshot")
    monkeypatch.delenv(snapshot_data.SNAPSHOT_DIR_ENV, raising=False)
    rcc.clear_route_cache()
    with pytest.raises(RuntimeError, match="SNAPSHOT_DIR"):
        rcc.fetch_candidate_routes()


def test_snapshot_geocoder_replays_and_is_loud_on_miss(tmp_path):
    snapshot_data.write_bundle(
        str(tmp_path),
        routes=[],
        geocode={"1200 McKinney St": GeoPoint(29.757, -95.367)},
        cases=[],
        manifest={},
    )
    geocoder = SnapshotGeocoder(str(tmp_path))
    assert geocoder.geocode("1200 McKinney St") == GeoPoint(29.757, -95.367)
    with pytest.raises(AddressNotFoundError, match="not in the snapshot"):
        geocoder.geocode("999 Nowhere Rd")
