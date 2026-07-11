"""Load and filter committed TW1 slot data for analysis views."""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

LoadSource = Literal["auto", "cache", "sql", "sample"]


def _prep_sql_access():
    import ds_utils
    from smart_assignment.data_prep.prep_dlvry_tw_data import DATA_LOCATION, DEFAULT_CACHE_EXTENSION

    run_mode = ds_utils.Mode("dev")
    cachey = ds_utils.Data(
        rm=run_mode,
        data_location=DATA_LOCATION,
        session_date="",
        ignore_cache=False,
        default_cache_extension=DEFAULT_CACHE_EXTENSION,
    )
    sql = ds_utils.SQLAccess(run_mode, data=cachey)
    return sql


def _to_time(value) -> time | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().time().replace(microsecond=0)
    text = str(value).strip()
    if not text or text.lower() in {"none", "nat", "nan"}:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def _normalize_committed(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in ["tw1opentime", "tw1closetime"]:
        if col in normalized.columns:
            normalized[col] = normalized[col].apply(_to_time)
    for col in ["latitude", "longitude"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    if "cust_tier" in normalized.columns:
        normalized["cust_tier"] = normalized["cust_tier"].fillna("Other").astype(str)
    else:
        normalized["cust_tier"] = "Other"
    return normalized


def _load_raw_dlvr_window(source: LoadSource) -> pd.DataFrame:
    from smart_assignment.data_prep.prep_dlvry_tw_data import (
        DLVR_WINDOW_CACHE_PATH,
        fetch_dlvr_window_records,
        read_cached_dataframe,
    )

    if source in ("auto", "cache"):
        try:
            return read_cached_dataframe(DLVR_WINDOW_CACHE_PATH)
        except Exception as exc:
            if source == "cache":
                raise FileNotFoundError(
                    "Delivery window cache not found. Run prep_dlvry_tw_data.py first or use source='sample'."
                ) from exc
            logger.warning("Cache load failed (%s); trying SQL.", exc)

    sql = _prep_sql_access()
    return fetch_dlvr_window_records(sql)


def _load_cust_tier_df(source: LoadSource) -> pd.DataFrame:
    from smart_assignment.data_prep.prep_dlvry_tw_data import (
        CUST_TIER_CACHE_PATH,
        fetch_cust_tier_records,
        read_cached_dataframe,
    )

    if source in ("auto", "cache"):
        try:
            return read_cached_dataframe(CUST_TIER_CACHE_PATH)
        except Exception as exc:
            if source == "cache":
                raise FileNotFoundError(
                    "Customer tier cache not found. Run prep_dlvry_tw_data.py first or use source='sample'."
                ) from exc
            logger.warning("Customer tier cache load failed (%s); trying SQL.", exc)

    return fetch_cust_tier_records(_prep_sql_access())


def load_sample_committed_tw1_slots_df() -> pd.DataFrame:
    """Synthetic committed TW1 rows for offline UI development and demos."""
    sample = pd.DataFrame(
        [
            {
                "route_id": "4101",
                "co_cust_nbr": "067-100001",
                "tw1opendate": "2025-05-01",
                "tw1closedate": "2025-05-01",
                "tw1opentime": "08:00:00",
                "tw1closetime": "10:00:00",
                "latitude": 29.756,
                "longitude": -95.366,
                "cust_tier": "Perks",
            },
            {
                "route_id": "4101",
                "co_cust_nbr": "067-200002",
                "tw1opendate": "2025-05-01",
                "tw1closedate": "2025-05-01",
                "tw1opentime": "09:15:00",
                "tw1closetime": "11:30:00",
                "latitude": 29.763,
                "longitude": -95.372,
                "cust_tier": "4",
            },
            {
                "route_id": "4209",
                "co_cust_nbr": "067-100001",
                "tw1opendate": "2025-05-02",
                "tw1closedate": "2025-05-02",
                "tw1opentime": "07:30:00",
                "tw1closetime": "09:00:00",
                "latitude": 29.745,
                "longitude": -95.470,
                "cust_tier": "Perks",
            },
            {
                "route_id": "4209",
                "co_cust_nbr": "067-300003",
                "tw1opendate": "2025-05-02",
                "tw1closedate": "2025-05-02",
                "tw1opentime": "10:00:00",
                "tw1closetime": "12:00:00",
                "latitude": 29.733,
                "longitude": -95.460,
                "cust_tier": "Non-Perks",
            },
        ]
    )
    return _normalize_committed(sample)


def load_committed_tw1_slots_df(source: LoadSource = "auto") -> pd.DataFrame:
    """Load committed TW1 slots derived from delivery-window facts."""
    if source == "sample":
        return load_sample_committed_tw1_slots_df()

    from smart_assignment.data_prep.prep_dlvry_tw_data import (
        summarize_committed_tw1_slots,
    )

    dlvr_window_df = _load_raw_dlvr_window(source)
    tier_df = _load_cust_tier_df(source)
    committed = summarize_committed_tw1_slots(dlvr_window_df, tier_df)
    committed = _normalize_committed(committed)
    committed = committed.sort_values(["co_cust_nbr", "route_id"]).reset_index(drop=True)
    return committed


def _window_minutes(df: pd.DataFrame) -> pd.Series:
    start = pd.to_datetime(df["tw1opentime"].astype(str), format="%H:%M:%S", errors="coerce")
    end = pd.to_datetime(df["tw1closetime"].astype(str), format="%H:%M:%S", errors="coerce")
    return ((end - start).dt.total_seconds() / 60.0).fillna(0)


def list_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct customers with labels and quick summary stats."""
    if df.empty:
        return pd.DataFrame(columns=["co_cust_nbr", "cust_tier", "route_count", "label"])
    summary = (
        df.groupby("co_cust_nbr", as_index=False)
        .agg(
            cust_tier=("cust_tier", "first"),
            route_count=("route_id", "nunique"),
        )
        .sort_values("co_cust_nbr")
    )
    summary["label"] = (
        summary["co_cust_nbr"].astype(str)
        + " ["
        + summary["cust_tier"].fillna("Other").astype(str)
        + "] · routes: "
        + summary["route_count"].astype(int).astype(str)
    )
    return summary


def list_routes(df: pd.DataFrame) -> pd.DataFrame:
    """Distinct routes with stop counts and display labels."""
    if df.empty:
        return pd.DataFrame(columns=["route_id", "stop_count", "label"])
    summary = (
        df.groupby("route_id", as_index=False)
        .agg(stop_count=("co_cust_nbr", "nunique"), tier_count=("cust_tier", "nunique"))
        .sort_values("route_id")
    )
    summary["label"] = (
        summary["route_id"].astype(str)
        + " · stops: "
        + summary["stop_count"].astype(int).astype(str)
        + " · tiers: "
        + summary["tier_count"].astype(int).astype(str)
    )
    return summary


def filter_customer(df: pd.DataFrame, co_cust_nbr: str, cust_tiers: list[str] | None = None) -> pd.DataFrame:
    """Committed TW1 rows for one customer, with optional tier filter."""
    subset = df[df["co_cust_nbr"].astype(str) == str(co_cust_nbr)].copy()
    if cust_tiers:
        subset = subset[subset["cust_tier"].astype(str).isin([str(t) for t in cust_tiers])]
    subset["window_minutes"] = _window_minutes(subset)
    return subset.sort_values(["route_id", "tw1opentime"]).reset_index(drop=True)


def filter_route(df: pd.DataFrame, route_id: str, cust_tiers: list[str] | None = None) -> pd.DataFrame:
    """Committed TW1 rows for one route, with optional tier filter."""
    subset = df[df["route_id"].astype(str) == str(route_id)].copy()
    if cust_tiers:
        subset = subset[subset["cust_tier"].astype(str).isin([str(t) for t in cust_tiers])]
    subset["window_minutes"] = _window_minutes(subset)
    return subset.sort_values(["co_cust_nbr", "tw1opentime"]).reset_index(drop=True)
