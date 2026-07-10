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
    CUST_TIER_CACHE_PATH,
    DATA_LOCATION,
    DEFAULT_CACHE_EXTENSION,
    DEFAULT_CUST_TIER,
    DLVR_WINDOW_CACHE_PATH,
    ROUTES_CACHE_PATH,
    attach_cust_tier_to_stop_locations,
    build_route_summary_tables,
    fetch_cust_tier_records,
    fetch_dlvr_window_records,
    fetch_route_stop_records,
    read_cached_dataframe,
    summarize_committed_tw1_slots,
)
from smart_assignment.shared.models import (
    DayOfWeek,
    GeoPoint,
    Route,
    RouteStop,
    Window,
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

_ROUTE_ID_DAY_DIGIT_TO_ENUM = {
    1: DayOfWeek.MON,
    2: DayOfWeek.TUE,
    3: DayOfWeek.WED,
    4: DayOfWeek.THU,
    5: DayOfWeek.FRI,
    6: DayOfWeek.SAT,
}


def _delivery_day_from_route_id(route_id: str) -> DayOfWeek | None:
    """Map the first digit of a 4-digit route_id to weekday (7 = not applicable)."""
    route_text = str(route_id).strip()
    if len(route_text) != 4 or not route_text.isdigit():
        return None

    day_digit = int(route_text[0])
    if day_digit == 7:
        return None
    return _ROUTE_ID_DAY_DIGIT_TO_ENUM.get(day_digit)


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


def _parse_time_window(open_tm, close_tm) -> "Window | None":
    """Return a Window tuple from tw1opentime/tw1closetime values, or None if either is missing."""
    if open_tm is None or close_tm is None:
        return None
    try:
        if pd.isna(open_tm) or pd.isna(close_tm):
            return None
    except (TypeError, ValueError):
        pass

    def _to_time(val) -> time:
        if isinstance(val, time):
            return val
        return time.fromisoformat(str(val))

    return (_to_time(open_tm), _to_time(close_tm))


def _derive_available_windows(committed_stops: list[RouteStop]) -> list[Window]:
    """
    Derive the set of distinct delivery windows offered on a route from the
    TW1 windows already committed to its existing stops.

    Uniqueness is determined by the (open, close) pair; the result is sorted
    ascending by window open time so callers always see earliest-first order.
    """
    seen: set[Window] = set()
    windows: list[Window] = []
    for stop in committed_stops:
        w = stop.delivery_time_window
        if w is not None and w not in seen:
            seen.add(w)
            windows.append(w)
    windows.sort(key=lambda w: w[0])
    return windows


def _build_committed_stops(
    route_id: str,
    stop_locations: pd.DataFrame,
) -> list[RouteStop]:
    route_stops = stop_locations[stop_locations["route_id"].astype(str) == str(route_id)]
    committed_stops: list[RouteStop] = []
    for stop in route_stops.itertuples(index=False):
        if pd.isna(stop.latitude) or pd.isna(stop.longitude):
            continue
        open_tm = getattr(stop, "tw1opentime", None)
        close_tm = getattr(stop, "tw1closetime", None)
        committed_stops.append(
            RouteStop(
                customer_number=str(stop.co_cust_nbr),
                location=GeoPoint(float(stop.latitude), float(stop.longitude)),
                delivery_time_window=_parse_time_window(open_tm, close_tm),
            )
        )
    return committed_stops


def routes_from_summary_tables(
    route_summary: pd.DataFrame,
    stop_locations: pd.DataFrame,
) -> list[Route]:
    routes: list[Route] = []
    for route_row in route_summary.itertuples(index=False):
        route_id = str(route_row.route_id)
        day = _delivery_day_from_route_id(route_id)
        if day is None:
            logger.debug(
                "Skipping route %s: weekday not encoded in route_id (expected first digit 1-6).",
                route_id,
            )
            continue
        routes.append(
            Route(
                route_id=route_id,
                name=str(route_row.route_nm),
                day=day,
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
                committed_stops=_build_committed_stops(route_id, stop_locations),
            )
        )
        route = routes[-1]
        route.available_windows = _derive_available_windows(route.committed_stops)
    return routes


def _sql_access() -> ds_utils.SQLAccess:
    run_mode = ds_utils.Mode("dev")
    cachey = ds_utils.Data(
        rm=run_mode,
        data_location=DATA_LOCATION,
        session_date="",
        ignore_cache=False,
        default_cache_extension=DEFAULT_CACHE_EXTENSION,
    )
    return ds_utils.SQLAccess(run_mode, data=cachey)


def _fetch_live_route_stop_records() -> pd.DataFrame:
    return fetch_route_stop_records(_sql_access())


def _load_cached_route_stop_records() -> pd.DataFrame:
    return read_cached_dataframe(ROUTES_CACHE_PATH)


def _fetch_live_cust_tier_records() -> pd.DataFrame:
    return fetch_cust_tier_records(_sql_access())


def _load_cached_cust_tier_records() -> pd.DataFrame:
    return read_cached_dataframe(CUST_TIER_CACHE_PATH)


def _fetch_live_dlvr_window_records() -> pd.DataFrame:
    return fetch_dlvr_window_records(_sql_access())


def _load_cached_dlvr_window_records() -> pd.DataFrame:
    return read_cached_dataframe(DLVR_WINDOW_CACHE_PATH)


def _load_route_capacity_raw_df() -> pd.DataFrame:
    try:
        route_capacity_raw_df = _fetch_live_route_stop_records()
        logger.info("Loaded route stop records from live SQL.")
        return route_capacity_raw_df
    except Exception as exc:
        logger.warning("Live SQL route pull failed (%s); falling back to cache.", exc)
        route_capacity_raw_df = _load_cached_route_stop_records()
        logger.info("Loaded route stop records from cache: %s", ROUTES_CACHE_PATH)
        return route_capacity_raw_df


def _load_cust_tier_records() -> pd.DataFrame | None:
    try:
        cust_tier_df = _fetch_live_cust_tier_records()
        logger.info("Loaded cust tier records from live SQL.")
        return cust_tier_df
    except Exception as exc:
        logger.warning("Live SQL cust tier pull failed (%s); falling back to cache.", exc)
        try:
            cust_tier_df = _load_cached_cust_tier_records()
            logger.info("Loaded cust tier records from cache: %s", CUST_TIER_CACHE_PATH)
            return cust_tier_df
        except Exception as cache_exc:
            logger.warning(
                "Cust tier cache unavailable (%s); defaulting stop tiers to Other.",
                cache_exc,
            )
            return None


def _load_dlvr_window_records() -> pd.DataFrame:
    try:
        dlvr_window_df = _fetch_live_dlvr_window_records()
        logger.info("Loaded delivery-window records from live SQL.")
        return dlvr_window_df
    except Exception as exc:
        logger.warning("Live SQL delivery-window pull failed (%s); falling back to cache.", exc)
        dlvr_window_df = _load_cached_dlvr_window_records()
        logger.info("Loaded delivery-window records from cache: %s", DLVR_WINDOW_CACHE_PATH)
        return dlvr_window_df


def _load_committed_tw1_slots_df() -> pd.DataFrame:
    committed_tw1_slots_df = summarize_committed_tw1_slots(_load_dlvr_window_records())
    cust_tier_df = _load_cust_tier_records()
    if cust_tier_df is not None:
        committed_tw1_slots_df = attach_cust_tier_to_stop_locations(
            committed_tw1_slots_df,
            cust_tier_df,
        )
    else:
        committed_tw1_slots_df = committed_tw1_slots_df.copy()
        committed_tw1_slots_df["cust_tier"] = DEFAULT_CUST_TIER
    return committed_tw1_slots_df


def _load_prepared_route_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_capacity_raw_df = _load_route_capacity_raw_df()
    committed_tw1_slots_df = _load_committed_tw1_slots_df()
    route_summary, stop_locations = build_route_summary_tables(
        route_capacity_raw_df,
        committed_tw1_slots_df,
    )
    return route_summary, stop_locations, committed_tw1_slots_df


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
                RouteStop("067-011011", GeoPoint(29.7550, -95.3650), delivery_time_window=(time(7, 0), time(10, 0))),
                RouteStop("067-011012", GeoPoint(29.7620, -95.3720), delivery_time_window=(time(7, 0), time(10, 0))),
                RouteStop("067-011013", GeoPoint(29.7480, -95.3810), delivery_time_window=(time(10, 30), time(12, 30))),
                RouteStop("067-011014", GeoPoint(29.7700, -95.3900), delivery_time_window=(time(10, 30), time(12, 30))),
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
                RouteStop("067-022021", GeoPoint(29.7450, -95.4700), delivery_time_window=(time(7, 30), time(11, 0))),
                RouteStop("067-022022", GeoPoint(29.7600, -95.5200), delivery_time_window=(time(7, 30), time(11, 0))),
                RouteStop("067-022023", GeoPoint(29.7830, -95.6350), delivery_time_window=(time(12, 0), time(14, 0))),
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
                RouteStop("067-033031", GeoPoint(30.1600, -95.4550), delivery_time_window=(time(8, 0), time(12, 0))),
                RouteStop("067-033032", GeoPoint(30.1720, -95.4700), delivery_time_window=(time(13, 0), time(15, 0))),
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
                RouteStop("067-044041", GeoPoint(29.6200, -95.6300), delivery_time_window=(time(6, 0), time(9, 0))),
                RouteStop("067-044042", GeoPoint(29.6100, -95.6500), delivery_time_window=(time(9, 30), time(12, 0))),
            ],
        ),
    ]


def fetch_candidate_routes() -> list[Route]:
    """Return all active route+day instances known to the system."""
    if _use_mock_routes():
        return _mock_routes()

    route_summary, stop_locations, _committed_tw1_slots_df = _load_prepared_route_tables()
    return routes_from_summary_tables(route_summary, stop_locations)
