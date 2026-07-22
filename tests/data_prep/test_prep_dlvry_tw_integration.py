"""Verify prep_dlvry_tw_data works against the vendored ds-utils-lite dependency."""

from unittest.mock import MagicMock

import pandas as pd

import ds_utils
from ds_utils import Data
from ds_utils.deploy.mode import Mode
from ds_utils.sql import SQLAccess

from smart_assignment.data_prep import prep_dlvry_tw_data as prep


def test_ds_utils_lite_imports():
    assert ds_utils.__version__ == "0.3.6-beta"
    assert hasattr(ds_utils, "SQLAccess")
    assert hasattr(ds_utils, "Mode")
    assert hasattr(ds_utils, "Data")


def test_fetch_route_stop_records_with_mock_sql():
    raw = pd.DataFrame(
        {
            "co_nbr": ["067", "067"],
            "cust_nbr": ["100001", "100002"],
            "route_id": ["R1", "R2"],
        }
    )
    duplicate_row = raw.iloc[[0]]
    mock_df = pd.concat([raw, duplicate_row], ignore_index=True)

    mock_sql = MagicMock()
    mock_sql.select_sql.return_value = mock_df

    result = prep.fetch_route_stop_records(mock_sql, qry=prep.QUERIES["routes"])

    mock_sql.select_sql.assert_called_once_with(prep.QUERIES["routes"])
    assert len(result) == 2
    assert result.duplicated().sum() == 0


def test_sql_access_select_sql_dict_path(tmp_path):
    sql_file = tmp_path / "sample.sql"
    sql_file.write_text("SELECT 1 AS co_nbr, 2 AS cust_nbr")

    creds = MagicMock()
    creds.has_credential.return_value = True

    access = SQLAccess(Mode(Mode.LOCAL), credentials=creds)
    access._read_sql_query_and_enforce_types = MagicMock(
        return_value=pd.DataFrame({"co_nbr": ["067"], "cust_nbr": ["100001"]})
    )

    qry = {
        "path": str(sql_file),
        "clusternm": "ODI_PROD",
        "params": {},
        "cache_name": "routes",
    }
    df = access.select_sql(qry)

    assert len(df) == 1
    assert list(df.columns) == ["co_nbr", "cust_nbr"]


def test_data_and_sql_access_main_block_pattern(tmp_path):
    run_mode = Mode(Mode.DEV)
    cachey = Data(
        rm=run_mode,
        data_location=str(tmp_path),
        session_date="",
        ignore_cache=False,
        default_cache_extension=".csv.gz",
    )
    creds = MagicMock()
    creds.has_credential.return_value = True
    sql = SQLAccess(run_mode, data=cachey, credentials=creds)

    expected = pd.DataFrame({"co_nbr": ["067"], "cust_nbr": ["100001"], "route_id": ["R1"]})
    sql.select_sql = MagicMock(return_value=expected.copy())

    result = prep.fetch_route_stop_records(sql)

    assert len(result) == 1
    sql.select_sql.assert_called_once()


def test_data_cache_hit(tmp_path):
    run_mode = Mode(Mode.DEV)
    cachey = Data(
        rm=run_mode,
        data_location=str(tmp_path),
        session_date="",
        ignore_cache=False,
        default_cache_extension=".csv.gz",
    )

    df = pd.DataFrame({"co_nbr": ["067"]})
    cachey.write(df, "routes")

    call_count = {"n": 0}

    def expensive():
        call_count["n"] += 1
        return pd.DataFrame({"co_nbr": ["999"]})

    first = cachey.check_into_cache("routes", expensive)
    second = cachey.check_into_cache("routes", expensive)

    pd.testing.assert_frame_equal(first, df)
    pd.testing.assert_frame_equal(second, df)
    assert call_count["n"] == 0
