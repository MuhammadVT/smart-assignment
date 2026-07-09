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
ROUTES_CACHE_PATH = os.path.join(DATA_LOCATION, 'dev', 'routes.csv.gz')
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


def summarize_route_capacity_by_weekday(df):
    route_capacity_daily = df.groupby(['route_id', 'route_start_date', 'dlvry_day_nm']).agg(
        route_weight_capacity=('route_weight_capacity', 'mean'),
        route_cube_capacity=('route_cube_capacity', 'mean'),
        weight_sum=('weight', 'sum'),
        cubes_sum=('cubes', 'sum'),
        cases_sum=('cases', 'sum'),
    ).reset_index()

    route_capacity_summary = route_capacity_daily.groupby(['route_id', 'dlvry_day_nm']).agg(
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


def summarize_stop_geographies(df):
    """Stop coordinates and route service centers keyed by route_id and dlvry_day_nm."""

    stop_locations = df.groupby(['route_id', 'route_nm', 'dlvry_day_nm', 'co_cust_nbr']).agg(
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
    ).reset_index()

    # TODO: find the center point in a more accurate way
    service_centers = stop_locations.groupby(['route_id', 'route_nm', 'dlvry_day_nm']).agg(
        service_center_latitude=('latitude', 'mean'),
        service_center_longitude=('longitude', 'mean'),
    ).reset_index()

    return stop_locations, service_centers


def build_route_summary_tables(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_capacity_summary = summarize_route_capacity_by_weekday(raw_df)
    stop_locations, service_centers = summarize_stop_geographies(raw_df)
    route_summary = route_capacity_summary.merge(
        service_centers,
        on=['route_id', 'dlvry_day_nm'],
        how='inner',
        validate='1:1',
    )
    return route_summary, stop_locations


# Backward-compatible aliases for existing callers/tests.
pull_routes_data = fetch_route_stop_records
calculate_route_capacity = summarize_route_capacity_by_weekday
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

    # ds_utils related params
    RUN_MODE = ds_utils.Mode('dev')
    IGNORE_CACHE = False
    DEFAULT_CACHE_EXTENSION = '.csv.gz' # 'parquet'

    # Set ds_utils cachey and sql
    cachey = ds_utils.Data(rm=RUN_MODE, data_location=DATA_LOCATION, session_date='',
                           ignore_cache=IGNORE_CACHE, default_cache_extension=DEFAULT_CACHE_EXTENSION)
    sql = ds_utils.SQLAccess(RUN_MODE, data=cachey)

    raw_df = fetch_route_stop_records(sql)
    df_routes, df_stop_locations = build_route_summary_tables(raw_df)
