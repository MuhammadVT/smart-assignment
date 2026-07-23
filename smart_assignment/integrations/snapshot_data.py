"""
Self-contained snapshot bundle: (de)serialize the *world* a decision runs against.

A snapshot dataset is the file-backed analogue of the code-defined ``mock`` world:
a committed directory that carries everything a replay needs, so an eval runs
fully offline and deterministically with no live data source.

    <dir>/
      routes.json     the candidate routes (capacity, committed stops, windows, tiers)
      geocode.json    {address -> {lat, lon}} for every case address
      cases.json      the curated eval cases (intake + expected outcome)
      manifest.json   provenance for visibility (model, version, source, counts)

This module owns *only* the encoding of the world half (``routes.json`` /
``geocode.json``) plus the shared path/filename/env constants -- the pure,
dependency-light substrate that the ``snapshot`` data source
(``route_capacity_client``), the ``SnapshotGeocoder`` (``geocoding_client``), and
the freeze/synthetic authors all share. It imports only stdlib + the domain
models, so importing it never pulls a backend, credentials, or pandas.

The JSON is intentionally human-readable and diffable (times as ``HH:MM:SS``,
day as its code, coordinates as ``lat``/``lon``) -- a golden dataset you can read
in a PR, exactly like the mock routes are readable in Python.
"""

from __future__ import annotations

import json
import os
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from smart_assignment.shared.models import (
    DayOfWeek,
    GeoPoint,
    Route,
    RouteStop,
    Window,
)

# The active snapshot directory (set by eval.dataset.apply_eval_dataset when a
# snapshot dataset is pinned). The snapshot data source and geocoder both read it.
SNAPSHOT_DIR_ENV = "SMART_ASSIGNMENT_SNAPSHOT_DIR"

# Bundle file names (one directory per dataset).
ROUTES_FILE = "routes.json"
GEOCODE_FILE = "geocode.json"
CASES_FILE = "cases.json"
MANIFEST_FILE = "manifest.json"


# ---------------------------------------------------------------------------
# Primitive encoders (kept tiny and symmetric so a round-trip is obvious)
# ---------------------------------------------------------------------------


def _enc_geo(point: GeoPoint) -> Dict[str, float]:
    return {"lat": point.latitude, "lon": point.longitude}


def _dec_geo(data: Dict[str, Any]) -> GeoPoint:
    return GeoPoint(latitude=float(data["lat"]), longitude=float(data["lon"]))


def _enc_window(window: Optional[Window]) -> Optional[List[str]]:
    if window is None:
        return None
    return [window[0].isoformat(), window[1].isoformat()]


def _dec_window(data: Optional[List[str]]) -> Optional[Window]:
    if not data:
        return None
    return (time.fromisoformat(data[0]), time.fromisoformat(data[1]))


def _dec_windows(data: Optional[List[Any]]) -> List[Window]:
    """A list of windows, dropping any that fail to decode (defensive)."""
    return [w for w in (_dec_window(item) for item in (data or [])) if w is not None]


def _enc_stop(stop: RouteStop) -> Dict[str, Any]:
    return {
        "customer_number": stop.customer_number,
        "location": _enc_geo(stop.location),
        "delivery_time_window": _enc_window(stop.delivery_time_window),
        "customer_tier": stop.customer_tier,
    }


def _dec_stop(data: Dict[str, Any]) -> RouteStop:
    return RouteStop(
        customer_number=data["customer_number"],
        location=_dec_geo(data["location"]),
        delivery_time_window=_dec_window(data.get("delivery_time_window")),
        customer_tier=data.get("customer_tier"),
    )


def serialize_route(route: Route) -> Dict[str, Any]:
    """One ``Route`` as a JSON-ready dict (every capacity/load field preserved)."""
    return {
        "route_id": route.route_id,
        "name": route.name,
        "day": route.day.value,
        "service_center": _enc_geo(route.service_center),
        "service_radius_miles": route.service_radius_miles,
        "vehicle_capacity_weight": route.vehicle_capacity_weight,
        "vehicle_capacity_cases": route.vehicle_capacity_cases,
        "vehicle_capacity_cubes": route.vehicle_capacity_cubes,
        "avg_load_weight": route.avg_load_weight,
        "avg_load_cases": route.avg_load_cases,
        "avg_load_cubes": route.avg_load_cubes,
        "available_windows": [_enc_window(w) for w in route.available_windows],
        "committed_stops": [_enc_stop(s) for s in route.committed_stops],
    }


def deserialize_route(data: Dict[str, Any]) -> Route:
    """Inverse of :func:`serialize_route`."""
    return Route(
        route_id=data["route_id"],
        name=data["name"],
        day=DayOfWeek(data["day"]),
        service_center=_dec_geo(data["service_center"]),
        service_radius_miles=data.get("service_radius_miles"),
        vehicle_capacity_weight=data.get("vehicle_capacity_weight", 0.0),
        vehicle_capacity_cases=data.get("vehicle_capacity_cases", 0.0),
        vehicle_capacity_cubes=data.get("vehicle_capacity_cubes", 0.0),
        avg_load_weight=data.get("avg_load_weight", 0.0),
        avg_load_cases=data.get("avg_load_cases", 0.0),
        avg_load_cubes=data.get("avg_load_cubes", 0.0),
        available_windows=_dec_windows(data.get("available_windows")),
        committed_stops=[_dec_stop(s) for s in data.get("committed_stops", [])],
    )


def serialize_routes(routes: List[Route]) -> List[Dict[str, Any]]:
    return [serialize_route(r) for r in routes]


def deserialize_routes(data: List[Dict[str, Any]]) -> List[Route]:
    return [deserialize_route(d) for d in data]


def serialize_geocode(mapping: Dict[str, GeoPoint]) -> Dict[str, Dict[str, float]]:
    return {address: _enc_geo(point) for address, point in mapping.items()}


def deserialize_geocode(data: Dict[str, Any]) -> Dict[str, GeoPoint]:
    return {address: _dec_geo(point) for address, point in data.items()}


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_routes(snapshot_dir: str) -> List[Route]:
    """The routes for a snapshot dataset directory (raises if the file is absent
    -- an eval must fail loudly rather than replay against nothing)."""
    return deserialize_routes(_read_json(Path(snapshot_dir) / ROUTES_FILE))


def load_geocode(snapshot_dir: str) -> Dict[str, GeoPoint]:
    """The address->coordinate map for a snapshot dataset directory."""
    return deserialize_geocode(_read_json(Path(snapshot_dir) / GEOCODE_FILE))


def load_cases(snapshot_dir: str) -> List[Dict[str, Any]]:
    """The curated eval cases (same schema ``eval/case_source.py`` loads)."""
    return _read_json(Path(snapshot_dir) / CASES_FILE)


def load_manifest(snapshot_dir: str) -> Dict[str, Any]:
    path = Path(snapshot_dir) / MANIFEST_FILE
    return _read_json(path) if path.exists() else {}


def write_bundle(
    snapshot_dir: str,
    *,
    routes: List[Route],
    geocode: Dict[str, GeoPoint],
    cases: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> None:
    """Write a complete, self-contained snapshot bundle to ``snapshot_dir``."""
    base = Path(snapshot_dir)
    _write_json(base / ROUTES_FILE, serialize_routes(routes))
    _write_json(base / GEOCODE_FILE, serialize_geocode(geocode))
    _write_json(base / CASES_FILE, cases)
    _write_json(base / MANIFEST_FILE, manifest)


def active_snapshot_dir() -> Optional[str]:
    """The pinned snapshot directory (``SMART_ASSIGNMENT_SNAPSHOT_DIR``), or None."""
    value = os.environ.get(SNAPSHOT_DIR_ENV)
    return value.strip() if value and value.strip() else None
