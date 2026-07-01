"""
[MOCK] Route/capacity data source — the single highest-priority integration
point to replace with a real system (Sysco's TMS / routing engine, e.g.
Roadnet, Descartes, or an internal service).

`fetch_candidate_routes()` returns `Route` objects for the Houston metro
(Sysco is headquartered in Houston, TX). As long as a real system's response
can populate the `Route` / `RouteStop` dataclasses in shared/models.py, no
downstream code (geo-lookup, constraints, scoring) needs to change.

DO NOT treat these capacities, stop densities, or shift windows as
representative of real Sysco operations — they exist only to exercise the
workflow end-to-end.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.shared.models import (
    DayOfWeek,
    GeoPoint,
    Route,
    RouteStop,
)


def _mock_routes() -> list[Route]:
    return [
        # Dense downtown/Midtown route — lots of tightly-clustered stops,
        # comfortable capacity headroom, morning windows.
        Route(
            route_id="RTE-4100",
            name="Central Houston",
            day=DayOfWeek.TUE,
            service_center=GeoPoint(29.7589, -95.3677),
            service_radius_miles=12.0,
            vehicle_capacity_cases=900,
            available_windows=[(time(7, 0), time(10, 0)), (time(10, 30), time(12, 30))],
            committed_stops=[
                RouteStop("067-011011", GeoPoint(29.7550, -95.3650), 140),
                RouteStop("067-011012", GeoPoint(29.7620, -95.3720), 120),
                RouteStop("067-011013", GeoPoint(29.7480, -95.3810), 160),
                RouteStop("067-011014", GeoPoint(29.7700, -95.3900), 100),
            ],
        ),
        # West Houston / Energy Corridor — moderate load, wide capacity,
        # stops trend toward the Galleria/west side.
        Route(
            route_id="RTE-4200",
            name="West Houston / Energy Corridor",
            day=DayOfWeek.WED,
            service_center=GeoPoint(29.7836, -95.6100),
            service_radius_miles=12.0,
            vehicle_capacity_cases=950,
            available_windows=[(time(7, 30), time(11, 0)), (time(12, 0), time(14, 0))],
            committed_stops=[
                RouteStop("067-022021", GeoPoint(29.7450, -95.4700), 130),
                RouteStop("067-022022", GeoPoint(29.7600, -95.5200), 150),
                RouteStop("067-022023", GeoPoint(29.7830, -95.6350), 120),
            ],
        ),
        # North / The Woodlands — lightly booked, plenty of room, later windows.
        Route(
            route_id="RTE-4300",
            name="North Houston / The Woodlands",
            day=DayOfWeek.THU,
            service_center=GeoPoint(30.1658, -95.4613),
            service_radius_miles=16.0,
            vehicle_capacity_cases=800,
            available_windows=[(time(8, 0), time(12, 0)), (time(13, 0), time(15, 0))],
            committed_stops=[
                RouteStop("067-033031", GeoPoint(30.1600, -95.4550), 110),
                RouteStop("067-033032", GeoPoint(30.1720, -95.4700), 90),
            ],
        ),
        # Southwest / Sugar Land — nearly full (near the 90% ceiling),
        # so most new volume won't fit.
        Route(
            route_id="RTE-4400",
            name="Southwest / Sugar Land",
            day=DayOfWeek.TUE,
            service_center=GeoPoint(29.6197, -95.6349),
            service_radius_miles=12.0,
            vehicle_capacity_cases=700,
            available_windows=[(time(6, 0), time(9, 0)), (time(9, 30), time(12, 0))],
            committed_stops=[
                RouteStop("067-044041", GeoPoint(29.6200, -95.6300), 320),
                RouteStop("067-044042", GeoPoint(29.6100, -95.6500), 300),
            ],
        ),
    ]


def fetch_candidate_routes() -> list[Route]:
    """[STUB] Return all active route+day instances known to the system."""
    return _mock_routes()
