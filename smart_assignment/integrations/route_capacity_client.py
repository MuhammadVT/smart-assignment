"""
Route/capacity data source — mock routes for tests/demo, or prepared ODI data.

`fetch_candidate_routes()` returns `Route` objects. Set
`SMART_ASSIGNMENT_ROUTE_SOURCE=prepared` to build routes from live SQL (with
cache fallback); default is `mock`.
"""

from __future__ import annotations

import logging
import os
from datetime import time

import pandas as pd

import ds_utils
from smart_assignment.data_prep.prep_dlvry_tw_data import (
    DATA_LOCATION,
    ROUTES_CACHE_PATH,
    build_route_summary_tables,
    fetch_route_stop_records,
)
from smart_assignment.shared.models import (
    DayOfWeek,
    GeoPoint,
    Route,
    RouteStop,
)

logger = logging.getLogger(__name__)

_ROUTE_SOURCE_ENV = "SMART_ASSIGNMENT_ROUTE_SOURCE"

_DELIVERY_DAY_NAME_TO_ENUM = {
    "monday": DayOfWeek.MON,
    "tuesday": DayOfWeek.TUE,
    "wednesday": DayOfWeek.WED,
    "thursday": DayOfWeek.THU,
    "friday": DayOfWeek.FRI,
    "saturday": DayOfWeek.SAT,
}


def _use_mock_routes() -> bool:
    return os.environ.get(_ROUTE_SOURCE_ENV, "mock").lower() == "mock"


def _parse_delivery_day_name(day_name: str) -> DayOfWeek:
    normalized = day_name.strip().lower()
    try:
        return _DELIVERY_DAY_NAME_TO_ENUM[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown delivery day name: {day_name!r}") from exc


def _safe_int(value) -> int:
    if pd.isna(value):
        return 0
    return int(round(float(value)))


def _safe_float(value) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _build_committed_stops(
    route_id: str,
    delivery_day_name: str,
    stop_locations: pd.DataFrame,
) -> list[RouteStop]:
    route_stops = stop_locations[
        (stop_locations["route_id"] == route_id)
        & (stop_locations["dlvry_day_nm"] == delivery_day_name)
    ]
    committed_stops: list[RouteStop] = []
    for stop in route_stops.itertuples(index=False):
        if pd.isna(stop.latitude) or pd.isna(stop.longitude):
            continue
        committed_stops.append(
            RouteStop(
                customer_number=str(stop.co_cust_nbr),
                location=GeoPoint(float(stop.latitude), float(stop.longitude)),
            )
        )
    return committed_stops


def routes_from_summary_tables(
    route_summary: pd.DataFrame,
    stop_locations: pd.DataFrame,
) -> list[Route]:
    routes: list[Route] = []
    for route_row in route_summary.itertuples(index=False):
        routes.append(
            Route(
                route_id=str(route_row.route_id),
                name=str(route_row.route_nm),
                day=_parse_delivery_day_name(route_row.dlvry_day_nm),
                service_center=GeoPoint(
                    float(route_row.service_center_latitude),
                    float(route_row.service_center_longitude),
                ),
                service_radius_miles=None,
                vehicle_capacity_cases=_safe_int(route_row.route_case_capacity),
                vehicle_capacity_weight=_safe_float(route_row.route_weight_capacity),
                vehicle_capacity_cubes=_safe_float(route_row.route_cube_capacity),
                avg_load_cases=_safe_int(route_row.cases_sum),
                avg_load_weight=_safe_float(route_row.weight_sum),
                avg_load_cubes=_safe_float(route_row.cubes_sum),
                available_windows=[],
                committed_stops=_build_committed_stops(
                    str(route_row.route_id),
                    str(route_row.dlvry_day_nm),
                    stop_locations,
                ),
            )
        )
    return routes


def _fetch_live_route_stop_records() -> pd.DataFrame:
    run_mode = ds_utils.Mode("dev")
    cachey = ds_utils.Data(
        rm=run_mode,
        data_location=DATA_LOCATION,
        session_date="",
        ignore_cache=False,
        default_cache_extension=".csv.gz",
    )
    sql = ds_utils.SQLAccess(run_mode, data=cachey)
    return fetch_route_stop_records(sql)


def _load_cached_route_stop_records() -> pd.DataFrame:
    return pd.read_csv(ROUTES_CACHE_PATH)


def _load_prepared_route_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        raw_df = _fetch_live_route_stop_records()
        logger.info("Loaded route stop records from live SQL.")
    except Exception as exc:
        logger.warning("Live SQL route pull failed (%s); falling back to cache.", exc)
        raw_df = _load_cached_route_stop_records()
        logger.info("Loaded route stop records from cache: %s", ROUTES_CACHE_PATH)

    return build_route_summary_tables(raw_df)


def _mock_routes() -> list[Route]:
    return [
        Route(
            route_id="RTE-4100",
            name="Central Houston",
            day=DayOfWeek.TUE,
            service_center=GeoPoint(29.7589, -95.3677),
            service_radius_miles=12.0,
            vehicle_capacity_cases=900,
            vehicle_capacity_weight=18000.0,
            vehicle_capacity_cubes=1200.0,
            avg_load_cases=520,
            avg_load_weight=10400.0,
            avg_load_cubes=693.0,
            available_windows=[(time(7, 0), time(10, 0)), (time(10, 30), time(12, 30))],
            committed_stops=[
                RouteStop("067-011011", GeoPoint(29.7550, -95.3650)),
                RouteStop("067-011012", GeoPoint(29.7620, -95.3720)),
                RouteStop("067-011013", GeoPoint(29.7480, -95.3810)),
                RouteStop("067-011014", GeoPoint(29.7700, -95.3900)),
            ],
        ),
        Route(
            route_id="RTE-4200",
            name="West Houston / Energy Corridor",
            day=DayOfWeek.WED,
            service_center=GeoPoint(29.7836, -95.6100),
            service_radius_miles=12.0,
            vehicle_capacity_cases=950,
            vehicle_capacity_weight=19000.0,
            vehicle_capacity_cubes=1300.0,
            avg_load_cases=400,
            avg_load_weight=8000.0,
            avg_load_cubes=547.0,
            available_windows=[(time(7, 30), time(11, 0)), (time(12, 0), time(14, 0))],
            committed_stops=[
                RouteStop("067-022021", GeoPoint(29.7450, -95.4700)),
                RouteStop("067-022022", GeoPoint(29.7600, -95.5200)),
                RouteStop("067-022023", GeoPoint(29.7830, -95.6350)),
            ],
        ),
        Route(
            route_id="RTE-4300",
            name="North Houston / The Woodlands",
            day=DayOfWeek.THU,
            service_center=GeoPoint(30.1658, -95.4613),
            service_radius_miles=16.0,
            vehicle_capacity_cases=800,
            vehicle_capacity_weight=16000.0,
            vehicle_capacity_cubes=1100.0,
            avg_load_cases=500,
            avg_load_weight=10000.0,
            avg_load_cubes=688.0,
            available_windows=[(time(8, 0), time(12, 0)), (time(13, 0), time(15, 0))],
            committed_stops=[
                RouteStop("067-033031", GeoPoint(30.1600, -95.4550)),
                RouteStop("067-033032", GeoPoint(30.1720, -95.4700)),
            ],
        ),
        Route(
            route_id="RTE-4400",
            name="Southwest / Sugar Land",
            day=DayOfWeek.TUE,
            service_center=GeoPoint(29.6197, -95.6349),
            service_radius_miles=12.0,
            vehicle_capacity_cases=700,
            vehicle_capacity_weight=14000.0,
            vehicle_capacity_cubes=1000.0,
            avg_load_cases=620,
            avg_load_weight=12400.0,
            avg_load_cubes=886.0,
            available_windows=[(time(6, 0), time(9, 0)), (time(9, 30), time(12, 0))],
            committed_stops=[
                RouteStop("067-044041", GeoPoint(29.6200, -95.6300)),
                RouteStop("067-044042", GeoPoint(29.6100, -95.6500)),
            ],
        ),
    ]


def fetch_candidate_routes() -> list[Route]:
    """Return all active route+day instances known to the system."""
    if _use_mock_routes():
        return _mock_routes()

    route_summary, stop_locations = _load_prepared_route_tables()
    return routes_from_summary_tables(route_summary, stop_locations)
