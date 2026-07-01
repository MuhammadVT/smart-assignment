"""Shared pytest fixtures for smart_assignment tests."""

from __future__ import annotations

from datetime import time

import pytest

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import (
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
        available_windows=[(time(7, 0), time(10, 0))],
        committed_stops=[
            RouteStop("067-090001", GeoPoint(29.7560, -95.3650), 120),
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
        available_windows=[(time(7, 0), time(10, 0))],
        committed_stops=[
            RouteStop("067-090002", GeoPoint(29.7560, -95.3650), 470),
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
