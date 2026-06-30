"""Shared pytest fixtures for smart_assignment tests."""

from __future__ import annotations

from datetime import time

import pytest

from smart_assignment.shared.models import CommittedStop, CustomerProfile, DayOfWeek, RouteSlot


@pytest.fixture
def sample_customer() -> CustomerProfile:
    return CustomerProfile(
        customer_id="CUST-NEW-9001",
        name="Riverside Diner",
        address="123 Example St",
        latitude=37.77,
        longitude=-122.41,
        weekly_order_volume_cases=150,
        product_temp_zone="mixed",
        requested_days=[DayOfWeek.TUE, DayOfWeek.WED],
        requested_time_window=(time(7, 0), time(11, 0)),
    )


@pytest.fixture
def open_route() -> RouteSlot:
    """A route with plenty of capacity and an open window."""
    return RouteSlot(
        route_id="RTE-TEST-1",
        day=DayOfWeek.TUE,
        vehicle_id="TRK-TEST",
        vehicle_capacity_cases=1000,
        vehicle_temp_zone="mixed",
        driver_id="DRV-TEST",
        driver_shift_start=time(6, 0),
        driver_shift_end=time(14, 0),
        service_zone_ids=["ZONE_TEST"],
        committed_stops=[],
        available_arrival_windows=[(time(8, 0), time(10, 0))],
    )


@pytest.fixture
def full_route() -> RouteSlot:
    """A route already at capacity — should be filtered out."""
    return RouteSlot(
        route_id="RTE-TEST-2",
        day=DayOfWeek.WED,
        vehicle_id="TRK-TEST-2",
        vehicle_capacity_cases=500,
        vehicle_temp_zone="mixed",
        driver_id="DRV-TEST-2",
        driver_shift_start=time(6, 0),
        driver_shift_end=time(14, 0),
        service_zone_ids=["ZONE_TEST"],
        committed_stops=[CommittedStop("CUST-EXISTING", (time(8, 0), time(8, 30)), 480)],
        available_arrival_windows=[(time(9, 0), time(10, 0))],
    )
