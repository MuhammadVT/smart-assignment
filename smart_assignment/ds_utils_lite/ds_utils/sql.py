"""SQL cluster access — lite subset of ds_utils.sql."""

import json
import os
import re
import time
from contextlib import contextmanager
from multiprocessing import Pool
from typing import Union

import pandas as pd
from pandas.io.sql import DatabaseError

import psycopg2
import psycopg2.errors

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - optional [aws] extra
    boto3 = None
    ClientError = Exception

try:
    import pyodbc
except ImportError:  # pragma: no cover - optional [mssql] extra
    pyodbc = None

from ds_utils.base_abc import DataABC, S3ABC
from ds_utils.deploy.mode import Mode
from ds_utils.formatting import ColumnFormatter
from ds_utils.logger import Logger
from ds_utils import __version__

logger = Logger(__name__)


class Credentials:
    """Stores SQL cluster credentials."""

    CREDENTIALS_ENV_NAME = "DATABASE_CREDENTIALS_LOCATION"
    SEED_INDICATOR_ENV_VARIABLE = "SEED_IMPLEMENTATION"
    CREDS_AWS_PARAMETER_PATHS = {
        "SEED_PROD": "/SEED/PROD/EDW/REDSHIFT/EAT/",
        "CATMAN_PROD": "/SEED/PROD/CATMAN/REDSHIFT/EAT/",
        "SDR_DEV": "/SEED/NONPROD/SDR/REDSHIFT/EAT/",
        "ODI_PROD": "/SEED/PROD/ODI/REDSHIFT/EAT/",
    }

    PARAM_LABEL_USERNAME = "USERNAME"
    PARAM_LABEL_PASSWORD = "PASSWORD"

    def __init__(self, credentials_json_path: str = None, credentials_param_store: dict = None):
        self._credentials = {}
        logger.info("Loading SQL database credentials.")

        if credentials_json_path is not None:
            self._resolve_creds_from_file(credentials_json_path)
            logger.info("Loaded credentials from user specified path.")
        elif os.environ.get(Credentials.SEED_INDICATOR_ENV_VARIABLE) or credentials_param_store is not None:
            logger.info("Detected running on SEED managed cluster.")
            self._resolve_creds_from_aws_parameter_store(credentials_param_store)
            logger.info("Credentials Loaded from AWS Parameter Store")
        else:
            self._resolve_creds_from_file_from_environment_variable()
            logger.info("Credentials Loaded from environment location.")

    def add_credential(self, clusternm, user, password):
        if clusternm not in self._credentials:
            self._credentials[clusternm] = {"user": user, "password": password}
        else:
            logger.warning(
                f"Credentials for {clusternm} already exists. "
                f"If you want to overwrite it, use add_or_replace_credential()"
            )

    def add_or_replace_credential(self, clusternm, user, password):
        self._credentials[clusternm] = {"user": user, "password": password}

    def get_user(self, clusternm):
        try:
            return self._credentials[clusternm]["user"]
        except KeyError as badkey:
            raise KeyError(f"You do not have credentials for clusternm {badkey} in your credentials file.")

    def get_password(self, clusternm):
        try:
            return self._credentials[clusternm]["password"]
        except KeyError as badkey:
            raise KeyError(f"You do not have credentials for clusternm {badkey} in your credentials file.")

    def has_credential(self, clusternm) -> bool:
        cred = self._credentials.get(clusternm)
        if cred is None:
            return False
        if cred.get("user") in (None, ""):
            return False
        if cred.get("password") in (None, ""):
            return False
        return True

    def _resolve_creds_from_aws_parameter_store(self, parameter_store: dict = None):
        if boto3 is None:
            raise ImportError(
                "boto3 is required for AWS Parameter Store credentials. "
                'Install with: pip install "ds-utils-lite[aws]"'
            )

        ssm_client = boto3.client("ssm", region_name="us-east-1")
        if parameter_store is None:
            parameter_store = Credentials.CREDS_AWS_PARAMETER_PATHS
        for clusternm, path in parameter_store.items():
            response = ssm_client.get_parameters_by_path(Path=path, WithDecryption=True)

            user, password = None, None
            for param in response["Parameters"]:
                if param["Name"] == str(path) + Credentials.PARAM_LABEL_USERNAME:
                    user = param["Value"]
                elif param["Name"] == path + Credentials.PARAM_LABEL_PASSWORD:
                    password = param["Value"]
            self.add_or_replace_credential(clusternm, user, password)

    def _resolve_creds_from_file_from_environment_variable(self, environment_variable_name: str = None):
        if environment_variable_name is None:
            environment_variable_name = Credentials.CREDENTIALS_ENV_NAME

        try:
            with open(os.environ[environment_variable_name]) as file:
                loaded_json_dict = json.loads(file.read())
        except KeyError as error:
            logger.error(
                "You need to have your credentials specified as json in a file "
                "referenced by the environment variable:" + Credentials.CREDENTIALS_ENV_NAME
            )
            raise error
        except FileNotFoundError as error:
            logger.error(
                f"You have specified a credentials location as an environment variable, "
                f"but we cannot find the file in that location. \n"
                f"os.environ['{Credentials.CREDENTIALS_ENV_NAME}'] = {os.environ[Credentials.CREDENTIALS_ENV_NAME]} \n"
                f"which resolves to {os.path.realpath(os.environ[Credentials.CREDENTIALS_ENV_NAME])}"
            )
            raise error

        self._resolve_creds_from_loaded_json(loaded_json_dict)

    def _resolve_creds_from_file(self, filepath: str):
        try:
            with open(filepath) as file:
                loaded_json_dict = json.loads(file.read())
        except FileNotFoundError as error:
            logger.error(f"Could not find SQL credentials file at: {filepath}")
            raise error

        self._resolve_creds_from_loaded_json(loaded_json_dict)

    def _resolve_creds_from_loaded_json(self, loaded_json_dict: dict):
        try:
            for cred in loaded_json_dict["credentials"]:
                try:
                    clusternm = cred["clusternm"]
                except KeyError as bad_key:
                    raise KeyError(
                        f"Could not find {bad_key} in your database credentials file.  "
                        f"See ds_utils documentation for new credentials file example."
                    )
                user = cred.get("user")
                password = cred.get("pswd")
                self.add_or_replace_credential(clusternm, user, password)
        except KeyError as bad_key:
            if bad_key == "credentials":
                raise KeyError(
                    f"Your credentials file does not contain the key: {bad_key}. "
                    f"See the database_creds_template in ds_utils."
                )
            raise KeyError(bad_key)


class SQLQuery:
    """Container for a SQL query."""

    FDP_PROD_HOST = "seed-catman-prod.cloud.sysco.net"
    FDP_PROD_DATABASE = "fdp_pro"
    FDP_DEV_HOST = "seed-catman-nonprod.cloud.sysco.net"
    FDP_DEV_DATABASE = "fdp_pro"
    SEED_PROD_HOST = "seed-edw-prod.cloud.sysco.net"
    SEED_PROD_DATABASE = "seedpro"
    SEED_NP_HOST = "seed-edw-nonprod.cloud.sysco.net"
    SEED_NP_DATABASE = "seedpro"
    ODI_PROD_HOST = "seed-odi-prod.cloud.sysco.net"
    ODI_PROD_DATABASE = "odi"
    ODI_DEV_HOST = "seed-odi-dev.cloud.sysco.net"
    ODI_DEV_DATABASE = "odi"
    SDR_PROD_HOST = "seed-prod-sdr-cluster.cpkt18asiyoc.us-east-1.redshift.amazonaws.com"
    SDR_PROD_DATABASE = "seedanalysisprod"
    SDR_DEV_HOST = "seed-repo-analysis.czc8orfanam8.us-east-1.redshift.amazonaws.com"
    SDR_DEV_DATABASE = "seedanalysis"

    PORT_DEFAULTS = {"redshift": 5439, "postgres": 5432, "sql server": 1433}

    CLUSTER_DEFAULTS = {
        "SEED_PROD": {"host": SEED_PROD_HOST, "database": SEED_PROD_DATABASE, "driver": "redshift"},
        "SEED_DEV": {"host": SEED_NP_HOST, "database": SEED_NP_DATABASE, "driver": "redshift"},
        "CATMAN_PROD": {"host": FDP_PROD_HOST, "database": FDP_PROD_DATABASE, "driver": "redshift"},
        "CATMAN_DEV": {"host": FDP_DEV_HOST, "database": FDP_DEV_DATABASE, "driver": "redshift"},
        "ODI_PROD": {"host": ODI_PROD_HOST, "database": ODI_PROD_DATABASE, "driver": "redshift"},
        "ODI_DEV": {"host": ODI_DEV_HOST, "database": ODI_DEV_DATABASE, "driver": "redshift"},
        "SDR_PROD": {"host": SDR_PROD_HOST, "database": SDR_PROD_DATABASE, "driver": "redshift"},
        "SDR_DEV": {"host": SDR_DEV_HOST, "database": SDR_DEV_DATABASE, "driver": "redshift"},
    }

    def __init__(
        self,
        path: str,
        clusternm: str,
        host: str = None,
        database: str = None,
        driver: str = None,
        port: int = None,
        params: dict = None,
        data_types: dict = None,
        chunksize=100000,
        **kwargs,
    ):
        self.path = path
        self.clusternm = clusternm

        try:
            self.host = host if host else SQLQuery.CLUSTER_DEFAULTS[clusternm]["host"]
            self.database = database if database else SQLQuery.CLUSTER_DEFAULTS[clusternm]["database"]
            self.driver = driver if driver else SQLQuery.CLUSTER_DEFAULTS[clusternm]["driver"]
        except KeyError as bad_key:
            raise KeyError(
                f"You specified an unknown clusternm: {bad_key}.  You should manually "
                "specify the host, database, and driver to avoid this error."
            )

        self.params = params
        self.data_types = data_types
        self.port = port

        for key in self.PORT_DEFAULTS:
            if self.driver.lower().find(key) != -1:
                if not self.port:
                    self.port = self.PORT_DEFAULTS[key]
                break
        else:
            logger.error(f"Unsupported Driver Type {self.driver}! Currently supporting - {self.PORT_DEFAULTS.keys()}")

        self.chunksize = chunksize
        self.kwargs = kwargs

    def __repr__(self):
        return (
            f"SQLQuery Object \n"
            f"Path: {self.path} \n"
            f"Cluster: {self.clusternm} \n"
            f"Host: {self.host} \n"
            f"DB: {self.database} \n"
        )

    @staticmethod
    def from_dict(query_dict: dict):
        return SQLQuery(**query_dict)


class SQLAccess:
    """Instanced class for interacting with SQL databases."""

    CREDENTIALS_ENV_NAME = "DATABASE_CREDENTIALS_LOCATION"

    def __init__(
        self,
        run_mode: Mode,
        repo_dir: str = "",
        data: DataABC = None,
        s3: S3ABC = None,
        formatter=ColumnFormatter(),
        **kwargs,
    ):
        logger.info(f"Using ds_utils version {__version__} with pandas version {pd.__version__}.")

        self._run_mode = run_mode
        self._repo_dir = repo_dir
        self._enforce_data_types = not self._run_mode.is_local()
        if kwargs.get("credentials") is not None:
            self._credentials = kwargs["credentials"]
        else:
            self._credentials = Credentials()

        if s3 is not None:
            assert data is not None, (
                "UNACCEPTABLE: you have specified an S3 bucket without specifying a "
                "local_data object, please specify both if you would like me to upload to S3"
            )

        self.data = data
        self._s3 = s3
        self.formatter = formatter

    def set_ignore_cache(self, value: bool = True):
        logger.warning("SQLAccess.data is public, so there is no need for SQLAccess.set_ignore_cache.")
        self.data.set_ignore_cache(value)

    @contextmanager
    def disable_cache(self):
        if isinstance(self.data, DataABC):
            prev_cache_val = self.data.ignore_cache
            self.data.ignore_cache = True
            try:
                yield
            finally:
                self.data.ignore_cache = prev_cache_val
        else:
            try:
                yield
            finally:
                pass

    def assert_creds_exist_for_queries(self, queries) -> bool:
        if isinstance(queries, SQLQuery):
            assert self._credentials.has_credential(queries.clusternm), (
                f"You have no creds for {queries.clusternm}. (Raised by SQLQuery object for {queries.path}"
            )

        if isinstance(queries, dict):
            if "clusternm" in queries:
                assert self._credentials.has_credential(queries["clusternm"]), (
                    f"You have no creds for {queries['clusternm']}. "
                    f"(Raised by dictionary object for {queries.get('path')}"
                )
            else:
                for _, value in queries.items():
                    if isinstance(value, SQLQuery):
                        assert self._credentials.has_credential(value.clusternm), (
                            f"You have no creds for {value.clusternm}. "
                            f"(Raised by SQLQuery object for {value.path}"
                        )
                    elif isinstance(value, dict):
                        assert self._credentials.has_credential(value["clusternm"]), (
                            f"You have no creds for {value['clusternm']}. "
                            f"(Raised by dictionary object for {value.get('path')}"
                        )

        return True

    def select_sql_from_dict(self, query_dict: dict) -> pd.DataFrame:
        qry_obj = SQLQuery.from_dict(query_dict)
        return self.select_sql(qry_obj)

    def select_sql(self, query: SQLQuery, cache_name: str = None) -> pd.DataFrame:
        if isinstance(query, dict):
            query = SQLQuery.from_dict(query)

        self._pre_select_checks(query)

        if cache_name is None:
            cache_name = self._create_cache_filename(query)

        if self.data is not None:
            result = self.data.cic(cache_name, self._read_sql_query_and_enforce_types, query)
        else:
            result = self._read_sql_query_and_enforce_types(query)

        if self._s3 is not None and hasattr(self._s3, "upload"):
            try:
                self._s3.upload(self.data.get_handle(cache_name))
                logger.info(f"Results of {query.path} uploaded to S3.")
            except ClientError as cer:
                if self._run_mode.is_dangerous() or self._run_mode.is_dev():
                    raise cer
                logger.warning(
                    "There was an error uploading to S3, probably due to an expired security token. "
                    "No files have been uploaded to S3 but the program was allowed to continue due "
                    "to running in local mode."
                )

        return result

    def execute(self, query: Union[SQLQuery, dict]):
        if isinstance(query, dict):
            query = SQLQuery.from_dict(query)
        self._pre_select_checks(query)
        query_text = self._read_format_query(query)
        with self._create_connection(query) as conn:
            cur = conn.cursor()
            cur.execute(query_text)
            conn.commit()

    def _pre_select_checks(self, query: SQLQuery) -> bool:
        if self._run_mode.is_dangerous() and query.data_types is None:
            raise ValueError(
                "UNACCEPTABLE: in production you must specify data types for each query.  "
                "Add the type dictionary to your query object and try again."
            )
        if self._run_mode.is_dev() and query.data_types is None:
            logger.warning(
                query.path
                + "is being executed in dev mode but no data types have been specified. "
                "Please add data types to each query before pushing to qa, as the code "
                "will cease to function."
            )

        if os.path.exists(os.path.join(self._repo_dir, "queries", query.path)):
            query.path = os.path.join(self._repo_dir, "queries", query.path)
        elif os.path.exists(query.path):
            pass
        else:
            raise FileNotFoundError(query.path)

        return True

    @staticmethod
    def _read_format_query(query: SQLQuery) -> str:
        def is_wrapped_with_quotes(string) -> bool:
            if string[0] == "'" and string[-1] == "'":
                return True
            if string[0] == '"' and string[-1] == '"':
                return True
            return False

        all_params = {}
        if query.params is not None:
            for key, value in query.params.items():
                if isinstance(value, tuple):
                    if len(value) == 1:
                        if isinstance(value[0], str):
                            if is_wrapped_with_quotes(value[0]):
                                value = "(" + str(value[0]) + ")"
                            else:
                                value = "(" + "'" + value[0] + "'" + ")"
                        else:
                            value = "(" + str(value[0]) + ")"
                elif not isinstance(value, (str, float, int)):
                    logger.warning("Your params values should probably of types [str, float, int, tuple].")
                all_params[key] = value

        with open(query.path) as file:
            query_text = file.read()

        query_text = re.sub("(?<=--)(.*)({)", r"\1", query_text, flags=re.MULTILINE)
        query_text = re.sub("(?<=--)(.*)(})", r"\1", query_text, flags=re.MULTILINE)
        if all_params:
            query_text = query_text.format(**all_params)

        return query_text

    def _create_cache_filename(self, query: SQLQuery) -> str:
        cache_name_from_kwargs = query.kwargs.get("cache_name")
        if cache_name_from_kwargs is not None:
            return cache_name_from_kwargs

        params_string = ""
        if query.params:
            params_string_list = []
            for key, value in query.params.items():
                params_string_list.append(str(key) + "-" + str(value))
            params_string_list = sorted(params_string_list, key=len)
            params_string = params_string + "_" + "_".join(params_string_list)

        cache_name = re.split(r"[/\\]", query.path)[-1].split(".")[0] + params_string
        if len(cache_name) > 150:
            cache_name = cache_name[:150]

        return cache_name

    def _read_sql_query_and_enforce_types(self, query: SQLQuery) -> pd.DataFrame:
        start_time = time.time()
        qry_name = os.path.basename(query.path)
        logger.info(f"Querying {query.clusternm}: {qry_name}")

        if query.kwargs.get("multiprocess"):
            logger.info("Using Multiprocess module")
            with Pool(1) as pool:
                results = pool.starmap(read_sql_helper, [(self, query)])[0]
        else:
            query_text = self._read_format_query(query)
            with self._create_connection(query) as conn:
                if query.chunksize is None:
                    results = pd.read_sql(query_text, conn)
                else:
                    try:
                        results = pd.concat(
                            list(pd.read_sql(query_text, conn, chunksize=query.chunksize)),
                            ignore_index=True,
                        )
                    except ValueError:
                        results = pd.read_sql(query_text, conn)

        if query.data_types is not None:
            if not self._run_mode.is_local():
                if list(sorted(query.data_types.keys())) != list(sorted(results.columns)):
                    raise AssertionError(
                        "You specified data_types for your query, "
                        "but the column names in the query did not match."
                    )
            for column in query.data_types:
                results[column] = results[column].astype(query.data_types[column])

        logger.info(f"Completed query in {round(time.time() - start_time, 2)} s")
        return results

    @contextmanager
    def _create_connection(self, query: SQLQuery):
        driver = str(query.driver).lower()

        if driver.find("postgres") != -1:
            conn = self._create_postgresql_connection(query=query)
        elif driver.find("redshift") != -1:
            conn = self._create_postgresql_connection(query=query)
        elif driver.find("sql server") != -1:
            conn = self._create_mssql_connection(query=query)
        else:
            raise ValueError("ds_utils does not know how to connect with driver type:" + query.driver)

        try:
            yield conn
        except DatabaseError as error:
            logger.error(error)
            raise error
        except psycopg2.errors.SyntaxError as error:
            logger.error(error)
            raise error
        finally:
            conn.close()

    def _create_postgresql_connection(self, query: SQLQuery):
        if not self._credentials.has_credential(query.clusternm):
            raise KeyError(f"You do not have credentials for {query.clusternm} in your credentials file.")
        user = self._credentials.get_user(query.clusternm)
        password = self._credentials.get_password(query.clusternm)

        connection_string = "".join(
            [
                "dbname=",
                str(query.database),
                " host=",
                str(query.host),
                " port=",
                str(query.port),
                " user=",
                str(user),
                " password=",
                str(password),
            ]
        )
        return psycopg2.connect(connection_string)

    def _create_mssql_connection(self, query: SQLQuery):
        if pyodbc is None:
            raise ImportError(
                "pyodbc is required for SQL Server connections. "
                'Install with: pip install "ds-utils-lite[mssql]"'
            )

        connection_string = [
            "Driver={",
            str(query.driver),
            "};Server=",
            str(query.host),
            ";Database=",
            str(query.database),
            ";Trusted_Connection=yes",
            ";Port=",
            str(query.port),
        ]

        if self._credentials.has_credential(query.clusternm):
            user = self._credentials.get_user(query.clusternm)
            password = self._credentials.get_password(query.clusternm)
            connection_string += [";UID=", str(user), ";PWD=", str(password)]

        return pyodbc.connect("".join(connection_string))


def read_sql_helper(access: SQLAccess, query: SQLQuery) -> pd.DataFrame:
    with access._create_connection(query) as conn:
        query_text = access._read_format_query(query)
        return pd.read_sql(query_text, conn)
