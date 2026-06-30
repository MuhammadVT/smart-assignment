"""
External system client(s) — code that actually talks to outside systems,
kept separate from `shared/tools.py` so workflows depend on stable,
testable function signatures rather than on integration details.

[ASSUMPTION BLOCK]
This client is entirely mocked. It stands in for whatever real system
holds Sysco's route capacity / TMS data (no real schema was provided).
This is the single highest-priority file to replace with a real
integration — everything downstream (constraint filtering in
shared/tools.py, the recommendation workflow) should not need to change
as long as the real system's response can populate the `RouteSlot`
dataclass in shared/models.py.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.models import (
    CommittedStop,
    CustomerProfile,
    DayOfWeek,
    RouteSlot,
)


def fetch_candidate_route_slots(zone_id: str, customer: CustomerProfile) -> list[RouteSlot]:
    """
    [STUB] Query the route capacity system for RouteSlots (route+day
    instances) that already serve, or could serve, this zone.

    [ASSUMPTION] Real version should call into Sysco's TMS/routing system
    (e.g. via an internal API or DB query) filtering by:
      - service_zone_ids containing `zone_id`
      - vehicle_temp_zone compatible with customer.product_temp_zone
      - route status = active

    Below is mock data standing in for that call so workflows are
    runnable end-to-end. DO NOT treat this data as representative of real
    Sysco route density, shift patterns, or vehicle capacities.
    """
    mock_routes = [
        RouteSlot(
            route_id="RTE-114",
            day=DayOfWeek.TUE,
            vehicle_id="TRK-22",
            vehicle_capacity_cases=850,
            vehicle_temp_zone="mixed",
            driver_id="DRV-09",
            driver_shift_start=time(5, 0),
            driver_shift_end=time(14, 0),
            service_zone_ids=[zone_id, "ZONE_ADJACENT_1"],
            committed_stops=[
                CommittedStop("CUST-001", (time(7, 0), time(7, 30)), 120),
                CommittedStop("CUST-002", (time(9, 0), time(9, 30)), 200),
            ],
            available_arrival_windows=[
                (time(7, 30), time(9, 0)),
                (time(9, 30), time(11, 0)),
                (time(12, 0), time(13, 30)),
            ],
        ),
        RouteSlot(
            route_id="RTE-114",
            day=DayOfWeek.FRI,
            vehicle_id="TRK-22",
            vehicle_capacity_cases=850,
            vehicle_temp_zone="mixed",
            driver_id="DRV-09",
            driver_shift_start=time(5, 0),
            driver_shift_end=time(14, 0),
            service_zone_ids=[zone_id],
            committed_stops=[
                CommittedStop("CUST-003", (time(6, 30), time(7, 0)), 600),
                CommittedStop("CUST-004", (time(8, 0), time(8, 30)), 220),
            ],
            available_arrival_windows=[
                (time(11, 0), time(13, 30)),
            ],
        ),
        RouteSlot(
            route_id="RTE-208",
            day=DayOfWeek.WED,
            vehicle_id="TRK-31",
            vehicle_capacity_cases=600,
            vehicle_temp_zone="refrigerated",
            driver_id="DRV-15",
            driver_shift_start=time(6, 0),
            driver_shift_end=time(13, 0),
            service_zone_ids=[zone_id],
            committed_stops=[],
            available_arrival_windows=[
                (time(6, 0), time(13, 0)),  # lightly booked route, wide open
            ],
        ),
    ]
    return mock_routes
