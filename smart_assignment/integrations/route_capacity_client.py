"""
Route/capacity data source. `fetch_candidate_routes()` returns `Route` objects.

The source is chosen by `SMART_ASSIGNMENT_DATA_SOURCE`, one of:

  - "mock"     — built-in Houston demo routes (fully offline, deterministic).
  - "cache"    — prepared ODI tables read from the on-disk snapshot ONLY
                 (deterministic; no SQL). **This is the default**, so a given
                 machine returns the same routes on every call and across
                 surfaces (adk web, the web app), which is what you want for
                 development and demos.
  - "live_sql" — pull from live SQL, falling back to the cache per table if a
                 pull fails (the old "prepared" behavior). Data can change
                 between runs, so two surfaces may disagree — use it only when
                 you actually want fresh data.

If the cache is requested (or defaulted to) but the snapshot files are missing
(e.g. a fresh checkout that never built one), we fall back to "mock" with a
loud warning rather than crash. The legacy `SMART_ASSIGNMENT_ROUTE_SOURCE`
(values mock|prepared) is still honored with a deprecation warning:
"prepared" maps to "live_sql".
"""

from __future__ import annotations

import logging
import os
from datetime import time

import pandas as pd

import ds_utils
from smart_assignment.data_prep.prep_dlvry_tw_data import (
    CUST_TIER_CACHE_PATH,
    DEFAULT_CUST_TIER,
    create_sql_access,
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

_DATA_SOURCE_ENV = "SMART_ASSIGNMENT_DATA_SOURCE"
_LEGACY_ROUTE_SOURCE_ENV = "SMART_ASSIGNMENT_ROUTE_SOURCE"

SOURCE_MOCK = "mock"
SOURCE_CACHE = "cache"
SOURCE_LIVE_SQL = "live_sql"
_VALID_SOURCES = (SOURCE_MOCK, SOURCE_CACHE, SOURCE_LIVE_SQL)

# Synonyms accepted for each source (incl. the legacy ROUTE_SOURCE values).
_SOURCE_ALIASES = {
    "mock": SOURCE_MOCK,
    "cache": SOURCE_CACHE,
    "cached": SOURCE_CACHE,
    "live_sql": SOURCE_LIVE_SQL,
    "live": SOURCE_LIVE_SQL,
    "sql": SOURCE_LIVE_SQL,
    "prepared": SOURCE_LIVE_SQL,  # legacy ROUTE_SOURCE value
}


def _data_source() -> str:
    """Resolve the active data source (default "cache"). Honors the legacy
    SMART_ASSIGNMENT_ROUTE_SOURCE with a deprecation warning; an unrecognized
    value falls back to "cache" with a warning."""
    raw = os.environ.get(_DATA_SOURCE_ENV)
    if raw is None:
        legacy = os.environ.get(_LEGACY_ROUTE_SOURCE_ENV)
        if legacy is not None and legacy.strip():
            logger.warning(
                "%s is deprecated; use %s (mock|cache|live_sql). Honoring %r for now.",
                _LEGACY_ROUTE_SOURCE_ENV,
                _DATA_SOURCE_ENV,
                legacy,
            )
            raw = legacy
    value = (raw or SOURCE_CACHE).strip().lower()
    resolved = _SOURCE_ALIASES.get(value)
    if resolved is None:
        logger.warning("Unknown data source %r; defaulting to %r.", value, SOURCE_CACHE)
        return SOURCE_CACHE
    return resolved


def _live_first() -> bool:
    """True when the active source should try live SQL before the cache."""
    return _data_source() == SOURCE_LIVE_SQL


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
        tier = getattr(stop, "cust_tier", None)
        committed_stops.append(
            RouteStop(
                customer_number=str(stop.co_cust_nbr),
                location=GeoPoint(float(stop.latitude), float(stop.longitude)),
                delivery_time_window=_parse_time_window(open_tm, close_tm),
                customer_tier=(str(tier) if tier is not None and not pd.isna(tier) else None),
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
    return create_sql_access()


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
    if _live_first():
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
    if _live_first():
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
    if _live_first():
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
    cust_tier_df = _load_cust_tier_records()
    if cust_tier_df is not None:
        committed_tw1_slots_df = summarize_committed_tw1_slots(
            _load_dlvr_window_records(),
            cust_tier_df,
        )
    else:
        committed_tw1_slots_df = summarize_committed_tw1_slots(_load_dlvr_window_records())
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
            # Sized so the large Galleria order is capacity-feasible here too (a
            # second feasible route), but at a poor buffer -- it still scores below
            # the auto-assign bar, so the case escalates while demonstrating a
            # route choice (and the map switching) on the Frontend tab.
            vehicle_capacity_cases=1050,
            vehicle_capacity_weight=18000.0,
            vehicle_capacity_cubes=1200.0,
            avg_load_cases=520,
            avg_load_weight=10400.0,
            avg_load_cubes=693.0,
            available_windows=[(time(7, 0), time(10, 0)), (time(10, 30), time(12, 30))],
            committed_stops=[
                RouteStop("067-011011", GeoPoint(29.7550, -95.3650), delivery_time_window=(time(7, 0), time(10, 0)), customer_tier="5"),
                RouteStop("067-011012", GeoPoint(29.7620, -95.3720), delivery_time_window=(time(7, 0), time(10, 0)), customer_tier="Perks"),
                RouteStop("067-011013", GeoPoint(29.7480, -95.3810), delivery_time_window=(time(10, 30), time(12, 30)), customer_tier="4"),
                RouteStop("067-011014", GeoPoint(29.7700, -95.3900), delivery_time_window=(time(10, 30), time(12, 30)), customer_tier="5"),
                # Additional downtown stops (all >1.6 mi from the demo prospect, so
                # its nearest-3 -- and thus its slot windows/clustering -- are
                # unchanged; they just fill out the route's delivery cluster).
                RouteStop("067-011015", GeoPoint(29.7850, -95.3450), delivery_time_window=(time(7, 0), time(10, 0)), customer_tier="4"),
                RouteStop("067-011016", GeoPoint(29.7350, -95.3480), delivery_time_window=(time(7, 0), time(10, 0)), customer_tier="Other"),
                RouteStop("067-011017", GeoPoint(29.7400, -95.3980), delivery_time_window=(time(10, 30), time(12, 30)), customer_tier="5"),
                RouteStop("067-011018", GeoPoint(29.7880, -95.3950), delivery_time_window=(time(10, 30), time(12, 30)), customer_tier="Perks"),
                RouteStop("067-011019", GeoPoint(29.7720, -95.3380), delivery_time_window=(time(7, 0), time(10, 0)), customer_tier="Other"),
            ],
        ),
        Route(
            # A SECOND downtown route on a different day (WED). It gives a
            # close-in prospect like Bayou a real cross-route slot option (so the
            # Frontend map can switch routes), but its day mismatch keeps its
            # window score at 0 for a TUE-preferring prospect, so RTE-4100 stays
            # the recommended winner. Sized so a small order (Bayou, 90) fits with
            # comfortable headroom while a large one (Galleria, 400) does NOT --
            # it stays capacity-infeasible there, leaving Galleria's escalation
            # unchanged.
            route_id="RTE-4110",
            name="Downtown / Midtown",
            day=DayOfWeek.WED,
            service_center=GeoPoint(29.7480, -95.3560),
            service_radius_miles=12.0,
            vehicle_capacity_cases=850,
            vehicle_capacity_weight=16000.0,
            vehicle_capacity_cubes=1100.0,
            avg_load_cases=430,
            avg_load_weight=8600.0,
            avg_load_cubes=573.0,
            available_windows=[(time(7, 30), time(10, 30)), (time(11, 0), time(13, 0))],
            committed_stops=[
                RouteStop("067-055051", GeoPoint(29.7420, -95.3300), delivery_time_window=(time(7, 30), time(10, 30)), customer_tier="4"),
                RouteStop("067-055052", GeoPoint(29.7350, -95.3370), delivery_time_window=(time(7, 30), time(10, 30)), customer_tier="5"),
                RouteStop("067-055053", GeoPoint(29.7500, -95.3250), delivery_time_window=(time(11, 0), time(13, 0)), customer_tier="Other"),
                RouteStop("067-055054", GeoPoint(29.7280, -95.3450), delivery_time_window=(time(11, 0), time(13, 0)), customer_tier="Perks"),
                RouteStop("067-055055", GeoPoint(29.7450, -95.3230), delivery_time_window=(time(7, 30), time(10, 30)), customer_tier="4"),
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
                RouteStop("067-022021", GeoPoint(29.7450, -95.4700), delivery_time_window=(time(7, 30), time(11, 0)), customer_tier="Perks"),
                RouteStop("067-022022", GeoPoint(29.7600, -95.5200), delivery_time_window=(time(7, 30), time(11, 0)), customer_tier="4"),
                RouteStop("067-022023", GeoPoint(29.7830, -95.6350), delivery_time_window=(time(12, 0), time(14, 0)), customer_tier="5"),
                # Additional Energy-Corridor stops clustered around the route's
                # western service center (far from the demo prospect on the eastern
                # edge, so its clustering/escalation are unchanged).
                RouteStop("067-022024", GeoPoint(29.7950, -95.6250), delivery_time_window=(time(7, 30), time(11, 0)), customer_tier="4"),
                RouteStop("067-022025", GeoPoint(29.7720, -95.6300), delivery_time_window=(time(7, 30), time(11, 0)), customer_tier="5"),
                RouteStop("067-022026", GeoPoint(29.8050, -95.6050), delivery_time_window=(time(12, 0), time(14, 0)), customer_tier="Perks"),
                RouteStop("067-022027", GeoPoint(29.7900, -95.6500), delivery_time_window=(time(12, 0), time(14, 0)), customer_tier="Other"),
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
                RouteStop("067-033031", GeoPoint(30.1600, -95.4550), delivery_time_window=(time(8, 0), time(12, 0)), customer_tier="4"),
                RouteStop("067-033032", GeoPoint(30.1720, -95.4700), delivery_time_window=(time(13, 0), time(15, 0)), customer_tier="Other"),
                # More Woodlands-area stops so the cluster is representative.
                RouteStop("067-033033", GeoPoint(30.1780, -95.4460), delivery_time_window=(time(8, 0), time(12, 0)), customer_tier="5"),
                RouteStop("067-033034", GeoPoint(30.1490, -95.4720), delivery_time_window=(time(8, 0), time(12, 0)), customer_tier="Perks"),
                RouteStop("067-033035", GeoPoint(30.1850, -95.4820), delivery_time_window=(time(13, 0), time(15, 0)), customer_tier="4"),
                RouteStop("067-033036", GeoPoint(30.1440, -95.4500), delivery_time_window=(time(8, 0), time(12, 0)), customer_tier="Other"),
                RouteStop("067-033037", GeoPoint(30.1910, -95.4560), delivery_time_window=(time(13, 0), time(15, 0)), customer_tier="5"),
            ],
        ),
        Route(
            # A SECOND Woodlands-area route on a different day (FRI). It gives a
            # Woodlands prospect a real cross-route slot option (so the Frontend
            # map can switch routes), but its FRI day mismatch keeps its window
            # score at 0 for a THU-preferring prospect, so the well-fitting
            # RTE-4300 stays the recommended winner while this appears as a
            # lower-ranked, still-feasible alternative.
            route_id="RTE-4310",
            name="Spring / South Woodlands",
            day=DayOfWeek.FRI,
            service_center=GeoPoint(30.1470, -95.4730),
            service_radius_miles=16.0,
            vehicle_capacity_cases=800,
            vehicle_capacity_weight=16000.0,
            vehicle_capacity_cubes=1080.0,
            avg_load_cases=470,
            avg_load_weight=9400.0,
            avg_load_cubes=627.0,
            # All stops sit in the single morning window, so this route offers
            # Woodlands one clean cross-route alternative slot (keeping the demo
            # to ~3 option cards) rather than a second afternoon one.
            available_windows=[(time(8, 0), time(11, 0))],
            committed_stops=[
                RouteStop("067-066061", GeoPoint(30.1400, -95.4680), delivery_time_window=(time(8, 0), time(11, 0)), customer_tier="4"),
                RouteStop("067-066062", GeoPoint(30.1350, -95.4800), delivery_time_window=(time(8, 0), time(11, 0)), customer_tier="5"),
                RouteStop("067-066063", GeoPoint(30.1520, -95.4650), delivery_time_window=(time(8, 0), time(11, 0)), customer_tier="Other"),
                RouteStop("067-066064", GeoPoint(30.1300, -95.4900), delivery_time_window=(time(8, 0), time(11, 0)), customer_tier="Perks"),
                RouteStop("067-066065", GeoPoint(30.1560, -95.4780), delivery_time_window=(time(8, 0), time(11, 0)), customer_tier="4"),
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
                RouteStop("067-044041", GeoPoint(29.6200, -95.6300), delivery_time_window=(time(6, 0), time(9, 0)), customer_tier="5"),
                RouteStop("067-044042", GeoPoint(29.6100, -95.6500), delivery_time_window=(time(9, 30), time(12, 0)), customer_tier="Perks"),
                # More Sugar Land stops for a representative cluster.
                RouteStop("067-044043", GeoPoint(29.6320, -95.6180), delivery_time_window=(time(6, 0), time(9, 0)), customer_tier="4"),
                RouteStop("067-044044", GeoPoint(29.5980, -95.6470), delivery_time_window=(time(9, 30), time(12, 0)), customer_tier="5"),
                RouteStop("067-044045", GeoPoint(29.6380, -95.6520), delivery_time_window=(time(6, 0), time(9, 0)), customer_tier="Other"),
                RouteStop("067-044046", GeoPoint(29.6030, -95.6220), delivery_time_window=(time(9, 30), time(12, 0)), customer_tier="Perks"),
            ],
        ),
    ]


def fetch_candidate_routes() -> list[Route]:
    """Return all active route+day instances from the configured data source
    (see the module docstring). "cache" (the default) and "live_sql" build from
    the prepared ODI tables; "mock" returns the demo routes. If the prepared
    tables can't be loaded (e.g. no cache snapshot on a fresh checkout), fall
    back to mock with a loud warning rather than crash."""
    source = _data_source()
    if source == SOURCE_MOCK:
        return _mock_routes()

    logger.info("Loading candidate routes from data source: %s", source)
    try:
        route_summary, stop_locations, _committed_tw1_slots_df = _load_prepared_route_tables()
        return routes_from_summary_tables(route_summary, stop_locations)
    except Exception as exc:
        logger.warning(
            "Data source %r requested but its data could not be loaded (%s); using the "
            "mock demo routes instead. Build the cache snapshot (see data_prep/) or check "
            "SQL access.",
            source,
            exc,
        )
        return _mock_routes()
