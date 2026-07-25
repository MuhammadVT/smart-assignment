"""Prepare the delivery datasets the assignment pipeline reads.

Three sources are pulled over SQL (see `QUERIES`) and cached under
`data/<run mode>/`, where the run mode comes from DS_UTILS_RUN_MODE:

- ``routes``      -- route stop facts, used for route capacity and geography.
- ``cust_tier``   -- customer tier lookup (Perks / 4 / 5 / ...).
- ``dlvr_window`` -- historical delivery-window facts, used to derive each
  customer's committed TW1 slot.

Run this module as a script to refresh the cache; `integrations/` and
`analysis/` read the cached files rather than hitting SQL.
"""

import logging
import os

import pandas as pd

import ds_utils

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.realpath(os.path.join(BASE_DIR, '..', 'queries'))
DATA_LOCATION = os.path.realpath(os.path.join(BASE_DIR, '..', '..', 'data'))
DEFAULT_CREDENTIALS_PATH = os.path.realpath(os.path.join(BASE_DIR, '..', '..', 'creds.json'))
CREDENTIALS_LOCATION_ENV = 'DATABASE_CREDENTIALS_LOCATION'
DEFAULT_CACHE_EXTENSION = '.parquet'  # '.csv.gz'
DS_UTILS_RUN_MODE_ENV = 'DS_UTILS_RUN_MODE'
DEFAULT_DS_UTILS_RUN_MODE = 'dev'
IGNORE_CACHE = False


def get_ds_utils_run_mode() -> ds_utils.Mode:
    name = os.environ.get(DS_UTILS_RUN_MODE_ENV, DEFAULT_DS_UTILS_RUN_MODE).strip().lower()
    return ds_utils.Mode(name)


def create_sql_access(*, ignore_cache: bool = False) -> ds_utils.SQLAccess:
    """Build a ds_utils SQL handle backed by the cache for the current run mode.

    Everything with a side effect lives here rather than at module scope, so
    importing this module stays inert -- `integrations/` imports it on the live
    request path just to read cached parquet, and that path must not touch
    credentials, the environment, or the filesystem.
    """
    # ds_utils reads the credentials path from the environment. Default to the
    # repo-root creds.json but let an already-set value win.
    os.environ.setdefault(CREDENTIALS_LOCATION_ENV, DEFAULT_CREDENTIALS_PATH)
    os.makedirs(cache_dir(), exist_ok=True)

    run_mode = get_ds_utils_run_mode()
    cachey = ds_utils.Data(
        rm=run_mode,
        data_location=DATA_LOCATION,
        session_date='',
        ignore_cache=ignore_cache,
        default_cache_extension=DEFAULT_CACHE_EXTENSION,
    )
    return ds_utils.SQLAccess(run_mode, data=cachey)


def cache_dir() -> str:
    """Cache directory for the current run mode, e.g. `data/dev/`.

    Mirrors how ds_utils lays out its cache (`<data_location>/<run_mode>/`), so
    readers and `create_sql_access()`'s writer agree on where files land. The
    run mode is read per call rather than frozen at import, so changing
    DS_UTILS_RUN_MODE takes effect without reloading the module.
    """
    return os.path.join(DATA_LOCATION, get_ds_utils_run_mode().run_mode)


def cache_path(cache_name: str) -> str:
    """Build a cache file path for the current run mode."""
    extension = DEFAULT_CACHE_EXTENSION.lstrip('.')
    return os.path.join(cache_dir(), f'{cache_name}.{extension}')


def read_cached_dataframe(path: str) -> pd.DataFrame:
    """Read a cached dataframe using the configured DEFAULT_CACHE_EXTENSION."""
    extension = DEFAULT_CACHE_EXTENSION.lstrip('.').lower()
    if extension == 'parquet':
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            # pandas raises ImportError when no parquet engine is installed. Turn
            # its generic "Unable to find a usable engine" into an actionable
            # remedy -- otherwise this bubbles up as an opaque reason in
            # fetch_candidate_routes' "using the mock demo routes instead"
            # warning, which reads like a data problem rather than a missing dep.
            raise ImportError(
                "Reading the parquet cache under data/<run mode>/ needs a parquet engine "
                "(pyarrow), which isn't installed. Add it with the project's "
                "'cache' extra: pip install -e \".[cache]\" (or uv pip install "
                "-e \".[cache]\" / uv pip install pyarrow)."
            ) from exc
    return pd.read_csv(path)


def routes_cache_path() -> str:
    return cache_path('routes')


def cust_tier_cache_path() -> str:
    return cache_path('cust_tier')


def dlvr_window_cache_path() -> str:
    return cache_path('dlvr_window')


QUERIES = {
    'routes': {
        'path': os.path.join(QUERY_DIR, 'routes.sql'),
        'clusternm': 'ODI_PROD',
        'params': {},
        'cache_name': 'routes'
    },
    'cust_tier': {
        'path': os.path.join(QUERY_DIR, 'cust_tier.sql'),
        'clusternm': 'SEED_PROD',
        'params': {},
        'cache_name': 'cust_tier'
    },
    'dlvr_window': {
        'path': os.path.join(QUERY_DIR, 'dlvr_window_fact.sql'),
        'clusternm': 'ODI_PROD',
        'params': {},
        'cache_name': 'dlvr_window'
    },
}


def fetch_route_stop_records(sql, qry=QUERIES['routes']):
    df = sql.select_sql(qry)

    assert len(df) > 0

    return df.drop_duplicates()


def fetch_cust_tier_records(sql, qry=QUERIES['cust_tier']):
    df = sql.select_sql(qry)

    assert len(df) > 0

    return df.drop_duplicates(subset=['co_cust_nbr'])


def fetch_dlvr_window_records(sql, qry=QUERIES['dlvr_window']):
    df = sql.select_sql(qry)

    assert len(df) > 0

    return df.drop_duplicates()


def summarize_committed_tw1_slots(
    dlvr_window_df: pd.DataFrame,
    cust_tier_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive committed TW1 slot per route/customer from historical delivery-window facts."""
    df = dlvr_window_df.copy()
    df['tw1opendatetime'] = pd.to_datetime(df['tw1opendatetime'])
    df['tw1closedatetime'] = pd.to_datetime(df['tw1closedatetime'])
    df['tw1opendate'] = df['tw1opendatetime'].dt.date
    df['tw1opentime'] = df['tw1opendatetime'].dt.time
    df['tw1closedate'] = df['tw1closedatetime'].dt.date
    df['tw1closetime'] = df['tw1closedatetime'].dt.time

    # Latest route start date first, so the 'first' aggregates below pick the
    # most recently committed window per route/customer.
    df = df.sort_values(
        by=['route_id', 'co_cust_nbr', 'rte_strt_dt'],
        ascending=[True, True, False],
    )

    committed = df.groupby(['route_id', 'co_cust_nbr']).agg(
        tw1opendate=('tw1opendate', 'first'),
        tw1closedate=('tw1closedate', 'first'),
        tw1opentime=('tw1opentime', 'first'),
        tw1closetime=('tw1closetime', 'first'),
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
    ).reset_index()

    if cust_tier_df is not None:
        # Add customer tier when available and keep higher-priority tiers.
        committed = attach_cust_tier_to_stop_locations(committed, cust_tier_df)
        committed = committed[committed['cust_tier'].isin(['4', '5', 'Perks'])].copy()

    return committed


def prepare_route_capacity_raw_data(route_capacity_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filter and normalize route stop facts used for route-capacity calculation."""
    df = route_capacity_raw_df.copy()
    route_ids = df["route_id"].astype(str)
    return df[route_ids.str.len() == 4].copy()


def summarize_route_capacity(df):
    route_capacity_daily = df.groupby(['route_id', 'route_nm', 'route_start_date']).agg(
        route_weight_capacity=('route_weight_capacity', 'mean'),
        route_cube_capacity=('route_cube_capacity', 'mean'),
        weight_sum=('weight', 'sum'),
        cubes_sum=('cubes', 'sum'),
        cases_sum=('cases', 'sum'),
    ).reset_index()

    route_capacity_summary = route_capacity_daily.groupby(['route_id', 'route_nm']).agg(
        route_weight_capacity=('route_weight_capacity', 'mean'),
        route_cube_capacity=('route_cube_capacity', 'mean'),
        weight_sum=('weight_sum', 'mean'),
        cubes_sum=('cubes_sum', 'mean'),
        cases_sum=('cases_sum', 'mean'),
    ).reset_index()
    route_capacity_summary['route_weigh_capacity_pct'] = (
        route_capacity_summary['weight_sum'] / route_capacity_summary['route_weight_capacity']
    )
    route_capacity_summary['route_cube_capacity_pct'] = (
        route_capacity_summary['cubes_sum'] / route_capacity_summary['route_cube_capacity']
    )
    route_capacity_summary['route_case_capacity'] = (
        route_capacity_summary['cases_sum'] / route_capacity_summary['route_cube_capacity_pct']
    )  # TODO: validate this assumption

    return route_capacity_summary


DEFAULT_CUST_TIER = "Other"


def attach_cust_tier_to_stop_locations(
    stop_locations: pd.DataFrame,
    cust_tier_df: pd.DataFrame,
) -> pd.DataFrame:
    tier_lookup = cust_tier_df[["co_cust_nbr", "cust_tier"]].drop_duplicates(subset=["co_cust_nbr"])
    stop_locations = stop_locations.merge(tier_lookup, on="co_cust_nbr", how="left")
    stop_locations["cust_tier"] = stop_locations["cust_tier"].fillna(DEFAULT_CUST_TIER)
    return stop_locations


def summarize_stop_geographies(
    committed_tw1_slots_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stop coordinates and route service centers keyed by route_id and dlvry_day_nm."""
    stop_locations = committed_tw1_slots_df.groupby(['route_id', 'co_cust_nbr']).agg(
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
        tw1opentime=('tw1opentime', 'first'),
        tw1closetime=('tw1closetime', 'first'),
        cust_tier=('cust_tier', 'first'),
    ).reset_index()

    # TODO: find the center point in a more accurate way
    service_centers = stop_locations.groupby(['route_id']).agg(
        service_center_latitude=('latitude', 'mean'),
        service_center_longitude=('longitude', 'mean'),
    ).reset_index()

    return stop_locations, service_centers


def build_route_summary_tables(
    route_capacity_raw_df: pd.DataFrame,
    committed_tw1_slots_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_df = prepare_route_capacity_raw_data(route_capacity_raw_df)
    route_capacity_summary = summarize_route_capacity(prepared_df)
    stop_locations, service_centers = summarize_stop_geographies(committed_tw1_slots_df)
    route_summary = route_capacity_summary.merge(
        service_centers,
        on=['route_id'],
        how='inner',
        validate='1:1',
    )
    return route_summary, stop_locations


if __name__ == '__main__':

    sql = create_sql_access(ignore_cache=IGNORE_CACHE)

    route_capacity_raw_df = fetch_route_stop_records(sql)
    cust_tier_df = fetch_cust_tier_records(sql)
    raw_dlvr_window_df = fetch_dlvr_window_records(sql)

    # Find committed time slot from historical data.
    committed_tw1_slots_df = summarize_committed_tw1_slots(raw_dlvr_window_df, cust_tier_df)

    df_routes, df_stop_locations = build_route_summary_tables(
        route_capacity_raw_df,
        committed_tw1_slots_df,
    )
