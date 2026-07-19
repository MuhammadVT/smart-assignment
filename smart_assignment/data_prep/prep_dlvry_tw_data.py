import logging
import os
import pandas as pd
import ds_utils

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)

# Config setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['DATABASE_CREDENTIALS_LOCATION'] = os.path.realpath(os.path.join(BASE_DIR, '..', '..', 'creds.json'))

# Directory setup
QUERY_DIR = os.path.realpath(os.path.join(BASE_DIR, '..', 'queries'))
DATA_LOCATION = os.path.realpath(os.path.join(BASE_DIR, '..', '..', 'data'))
DEFAULT_CACHE_EXTENSION = '.parquet' # '.csv.gz'
DS_UTILS_RUN_MODE_ENV = 'DS_UTILS_RUN_MODE'
DEFAULT_DS_UTILS_RUN_MODE = 'dev'


def get_ds_utils_run_mode() -> ds_utils.Mode:
    name = os.environ.get(DS_UTILS_RUN_MODE_ENV, DEFAULT_DS_UTILS_RUN_MODE).strip().lower()
    return ds_utils.Mode(name)


def create_sql_access(*, ignore_cache: bool = False) -> ds_utils.SQLAccess:
    run_mode = get_ds_utils_run_mode()
    cachey = ds_utils.Data(
        rm=run_mode,
        data_location=DATA_LOCATION,
        session_date='',
        ignore_cache=ignore_cache,
        default_cache_extension=DEFAULT_CACHE_EXTENSION,
    )
    return ds_utils.SQLAccess(run_mode, data=cachey)


DEV_CACHE_DIR = os.path.join(DATA_LOCATION, 'dev')
os.makedirs(DEV_CACHE_DIR, exist_ok=True)


def cache_path(cache_name: str) -> str:
    """Build a dev-cache file path using DEFAULT_CACHE_EXTENSION."""
    extension = DEFAULT_CACHE_EXTENSION.lstrip('.')
    return os.path.join(DEV_CACHE_DIR, f'{cache_name}.{extension}')


def read_cached_dataframe(cache_path: str) -> pd.DataFrame:
    """Read a cached dataframe using the configured DEFAULT_CACHE_EXTENSION."""
    extension = DEFAULT_CACHE_EXTENSION.lstrip('.').lower()
    if extension == 'parquet':
        try:
            return pd.read_parquet(cache_path)
        except ImportError as exc:
            # pandas raises ImportError when no parquet engine is installed. Turn
            # its generic "Unable to find a usable engine" into an actionable
            # remedy -- otherwise this bubbles up as an opaque reason in
            # fetch_candidate_routes' "using the mock demo routes instead"
            # warning, which reads like a data problem rather than a missing dep.
            raise ImportError(
                "Reading the parquet cache under data/dev/ needs a parquet engine "
                "(pyarrow), which isn't installed. Add it with the project's "
                "'cache' extra: pip install -e \".[cache]\" (or uv pip install "
                "-e \".[cache]\" / uv pip install pyarrow)."
            ) from exc
    return pd.read_csv(cache_path)


ROUTES_CACHE_PATH = cache_path('routes')
CUST_TIER_CACHE_PATH = cache_path('cust_tier')
DLVR_WINDOW_CACHE_PATH = cache_path('dlvr_window')
OUTPUT_DIR = os.path.realpath(os.path.join(DATA_LOCATION, 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_DIR = os.path.realpath(os.path.join(DATA_LOCATION, 'input'))
os.makedirs(INPUT_DIR, exist_ok=True)


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
    # 'dot_base': {
    #     'path': os.path.join(QUERY_DIR, 'dot_base.sql'),
    #     'clusternm': 'SEED_PROD',
    #     'params': {},
    #     'cache_name': 'dot_base'
    # },
    # 'co_itm_dim': {
    #     'path': os.path.join(QUERY_DIR, 'co_itm_dim.sql'),
    #     'clusternm': 'SEED_PROD',
    #     'params': {},
    #     'cache_name': 'co_itm_dim'
    # },
}


def fillin_qry_params(qry, qry_params):
    for key, val in qry_params.items():
        qry['params'][key] = val

    return qry


def fetch_route_stop_records(sql, qry=QUERIES['routes'], cache_nm='routes'):
    df = sql.select_sql(qry)

    assert len(df) > 0
    df = df.drop_duplicates()
    # assert df.duplicated(subset=['co_nbr', 'itm_nbr', 'fisc_wk_id']).sum() == 0

    return df


def fetch_cust_tier_records(sql, qry=QUERIES['cust_tier'], cache_nm='cust_tier'):
    df = sql.select_sql(qry)

    assert len(df) > 0
    df = df.drop_duplicates(subset=['co_cust_nbr'])

    return df


def fetch_dlvr_window_records(sql, qry=QUERIES['dlvr_window'], cache_nm='dlvr_window'):
    df = sql.select_sql(qry)

    assert len(df) > 0
    df = df.drop_duplicates()

    return df


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

    df = df.sort_values(by=['route_id', 'co_cust_nbr', 'rte_strt_dt'], ascending=[True, True, False]) 

    # cols_to_keep = ['route_id', 'rte_strt_dt', 'co_cust_nbr', 'latitude', 'longitude', 
    # 'tw1opendate', 'tw1opentime', 'tw1closedate', 'tw1closetime']
    # committed = df.drop_duplicates(subset=['route_id', 'co_cust_nbr'])
    committed = df.groupby(['route_id', 'co_cust_nbr']).agg(
        tw1opendate=('tw1opendate', 'first'),
        tw1closedate=('tw1closedate', 'first'),
        tw1opentime=('tw1opentime', 'first'),
        tw1closetime=('tw1closetime', 'first'),
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
    ).reset_index()

    # committed = committed[committed['tw1opendatetime'] < committed['tw1closedatetime']].copy()
    
    if cust_tier_df is not None:
        # Add customer tier when available and keep higher-priority tiers.
        committed = attach_cust_tier_to_stop_locations(committed, cust_tier_df)
        committed = committed[committed['cust_tier'].isin(['4', '5', 'Perks'])].copy()

    return committed


def normalize_dlvry_day_nm(series: pd.Series) -> pd.Series:
    """Canonical weekday name for joins (routes SQL uses padded TO_CHAR 'Day' labels)."""
    return series.astype(str).str.strip().str.title()


def merge_stop_locations_with_dlvr_window(
    stop_locations: pd.DataFrame,
    dlvr_window_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach delivery-window attributes to route stops by customer and weekday."""
    stops = stop_locations.copy()
    window_df = dlvr_window_df.drop(columns=["cust_tier"], errors="ignore").copy()
    stops["dlvry_day_nm"] = normalize_dlvry_day_nm(stops["dlvry_day_nm"])
    window_df["tw_dlvry_day_nm"] = normalize_dlvry_day_nm(window_df["tw_dlvry_day_nm"])
    merged = stops.merge(
        window_df,
        left_on=["co_cust_nbr", "dlvry_day_nm"],
        right_on=["co_cust_nbr", "tw_dlvry_day_nm"],
        how="left",
    )
    return merged


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


def summarize_stop_geographies(committed_tw1_slots_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


# Backward-compatible aliases for existing callers/tests. # TODO: remove these later
pull_routes_data = fetch_route_stop_records
pull_cust_tier_data = fetch_cust_tier_records
pull_dlvr_window_data = fetch_dlvr_window_records
calculate_route_capacity = summarize_route_capacity
get_route_stops_locations = summarize_stop_geographies

# def pull_dot_base_data(sql, strt_wk, end_wk, markets, qry=QUERIES['dot_base'], cache_nm=None):
#
#     qry = copy.deepcopy(qry)
#     if cache_nm is None:
#         qry['cache_name'] = qry['cache_name'] + '_mkt_' + '_'.join(markets) + '_fw_' + '_'.join(map(str, [strt_wk, end_wk]))
#     markets_str = ','.join(f"'{x}'" for x in markets)
#     qry_params = {'START_WEEK': strt_wk, 'END_WEEK': end_wk, 'MARKET': markets_str}
#     # df = PrepException.pull_data(sql, qry, qry_params=qry_params)
#     df = PrepException.pull_data(sql, qry, qry_params=qry_params,
#                                  data_grain=['co_nbr', 'true_vndr_nbr', 'src_vndr_nbr', 'vndr_ship_pt_nbr',
#                                              'vndr_sb_grp_id', 'itm_nbr'])
#
#     return df


if __name__ == '__main__':

    IGNORE_CACHE = False
    sql = create_sql_access(ignore_cache=IGNORE_CACHE)

    route_capacity_raw_df = fetch_route_stop_records(sql)
    cust_tier_df = fetch_cust_tier_records(sql)
    raw_dlvr_window_df = fetch_dlvr_window_records(sql)

    # Find committed time slot from historical data.
    committed_tw1_slots_df = summarize_committed_tw1_slots(raw_dlvr_window_df, cust_tier_df)

    df_routes, df_stop_locations = build_route_summary_tables(route_capacity_raw_df, committed_tw1_slots_df)




