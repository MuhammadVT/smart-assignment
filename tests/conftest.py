"""Shared pytest fixtures for smart_assignment tests."""

from __future__ import annotations

import os

import dotenv

_DEVELOPER_ENV_PREFIXES = ("SMART_ASSIGNMENT_", "SAGE_", "LLM_GATEWAY_")


def _isolate_from_developer_env() -> None:
    """Run the hermetic suite against code defaults, never a developer's .env.

    ``smart_assignment/__init__.py`` loads the repo-root ``.env`` deliberately --
    by an absolute, cwd-independent path, so every entry point sees one
    configuration. That is right for the app and wrong for this suite: on a
    configured machine it hands the tests real credentials and flipped feature
    flags, so tests that are green in CI (which has no .env) fail locally for
    reasons that have nothing to do with the code under test.

    Stubbing the loader is the part that matters -- scrubbing alone would simply
    be undone by that import. .env loading itself stays covered, in subprocesses
    this cannot reach: see tests/webapp/test_dotenv.py.

    A test that wants a flag on sets it explicitly (``monkeypatch.setenv`` or a
    ``Config(...)`` argument), which is where that intent belongs anyway.
    """
    dotenv.load_dotenv = lambda *args, **kwargs: False  # type: ignore[assignment]
    for name in list(os.environ):
        if name.startswith(_DEVELOPER_ENV_PREFIXES):
            del os.environ[name]


_isolate_from_developer_env()

# Pin offline, deterministic providers for the whole test session BEFORE any
# smart_assignment import below resolves them. The code defaults are now live
# (census geocoder; cache/live data source), so without this a test that uses
# the default geocoder would make a network call. Must follow the scrub above,
# which would otherwise drop these.
os.environ["SMART_ASSIGNMENT_GEOCODER"] = "mock"
os.environ["SMART_ASSIGNMENT_DATA_SOURCE"] = "mock"

from datetime import time  # noqa: E402

import pytest  # noqa: E402

from smart_assignment.shared.config import Config  # noqa: E402
from smart_assignment.shared.models import (  # noqa: E402
    CustomerProfile,
    DayOfWeek,
    GeoPoint,
    PreferredSlot,
    Route,
    RouteStop,
)


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def sample_customer() -> CustomerProfile:
    """A geocoded downtown-Houston customer preferring Tuesday mornings."""
    return CustomerProfile(
        customer_number="067-100001",
        name="Riverside Diner",
        address="1200 McKinney St, Houston, TX 77010",
        order_quantity_cases=90,
        preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(7, 0), time(10, 0))),
        location=GeoPoint(29.7570, -95.3670),
    )


@pytest.fixture
def open_route() -> Route:
    """Nearby route with plenty of capacity and an overlapping window."""
    return Route(
        route_id="RTE-TEST-1",
        name="Test Open Route",
        day=DayOfWeek.TUE,
        service_center=GeoPoint(29.7589, -95.3677),
        service_radius_miles=12.0,
        vehicle_capacity_cases=1000,
        avg_load_cases=120,
        available_windows=[(time(7, 0), time(10, 0))],
        committed_stops=[
            RouteStop("067-090001", GeoPoint(29.7560, -95.3650)),
        ],
    )


@pytest.fixture
def full_route() -> Route:
    """Nearby route already near the 90% capacity ceiling."""
    return Route(
        route_id="RTE-TEST-2",
        name="Test Full Route",
        day=DayOfWeek.WED,
        service_center=GeoPoint(29.7589, -95.3677),
        service_radius_miles=12.0,
        vehicle_capacity_cases=500,
        avg_load_cases=470,
        available_windows=[(time(7, 0), time(10, 0))],
        committed_stops=[
            RouteStop("067-090002", GeoPoint(29.7560, -95.3650)),
        ],
    )


@pytest.fixture
def far_route() -> Route:
    """Route whose service center is well outside serviceable range."""
    return Route(
        route_id="RTE-TEST-3",
        name="Test Far Route",
        day=DayOfWeek.THU,
        service_center=GeoPoint(30.1658, -95.4613),  # The Woodlands, ~30 mi away
        service_radius_miles=12.0,
        vehicle_capacity_cases=1000,
        available_windows=[(time(7, 0), time(10, 0))],
        committed_stops=[],
    )
