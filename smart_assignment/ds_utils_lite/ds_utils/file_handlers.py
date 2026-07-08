"""File readers and writers for Data cache files."""

import gzip
import json
import os
import pickle
from abc import ABC, abstractmethod

import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover - optional [parquet] extra
    pyarrow = None


class FileHandler(ABC):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def make_file_path(directory: str, file_name: str):
        return os.path.join(directory, file_name)

    @abstractmethod
    def write(self, object_to_write, file_path):
        pass

    @abstractmethod
    def read(self, file_path):
        pass

    @abstractmethod
    def uses_header(self) -> bool:
        pass

    def write_header(self, object_to_write: pd.DataFrame, file_path: str):
        header_path = str(file_path) + ".header"
        data_types = object_to_write.dtypes.to_dict()
        data_types = {k: str(v) if str(v) != "object" else "str" for k, v in data_types.items()}
        with open(header_path, "w") as header_file:
            header_file.write(json.dumps(data_types))

    def read_header(self, file_path: str):
        header_path = str(file_path) + ".header"
        with open(header_path, "r") as header_file:
            header = json.loads(header_file.read())
        date_cols = []
        for col_name in header:
            if header[col_name].find("datetime") != -1:
                header[col_name] = "str"
                date_cols.append(col_name)
        return header, date_cols


class CSVHandler(FileHandler):
    def write(self, object_to_write: pd.DataFrame, file_path: str):
        object_to_write.to_csv(file_path, index=False, compression="infer")
        self.write_header(object_to_write, file_path)

    def read(self, file_path) -> pd.DataFrame:
        try:
            header, date_cols = self.read_header(file_path)
            return pd.read_csv(file_path, dtype=header, parse_dates=date_cols)
        except FileNotFoundError:
            pass
        return pd.read_csv(file_path)

    def uses_header(self) -> bool:
        return True


class ParquetHandler(FileHandler):
    def write(self, object_to_write: pd.DataFrame, file_path: str):
        if pyarrow is None:
            raise ImportError(
                'pyarrow is required for parquet files. Install with: pip install "ds-utils-lite[parquet]"'
            )
        object_to_write.to_parquet(file_path)
        self.write_header(object_to_write, file_path)

    def read(self, file_path) -> pd.DataFrame:
        if pyarrow is None:
            raise ImportError(
                'pyarrow is required for parquet files. Install with: pip install "ds-utils-lite[parquet]"'
            )
        return pd.read_parquet(file_path)

    def uses_header(self) -> bool:
        return False


class JSONHandler(FileHandler):
    def write(self, object_to_write, file_path):
        with open(file_path, "w") as file_cache:
            file_cache.write(json.dumps(object_to_write))

    def read(self, file_path):
        with open(file_path, "r") as file_cache:
            return json.loads(file_cache.read())

    def uses_header(self) -> bool:
        return False


class PickleHandler(FileHandler):
    def write(self, object_to_write, file_path):
        if hasattr(object_to_write, "to_pickle"):
            object_to_write.to_pickle(file_path)
        elif str(file_path).endswith("gz") or str(file_path).endswith("gzip"):
            with gzip.open(file_path, "wb") as file_cache:
                pickle.dump(object_to_write, file_cache)
        else:
            with open(file_path, "wb") as file_cache:
                pickle.dump(object_to_write, file_cache)

    def read(self, file_path):
        if str(file_path).endswith("gz") or str(file_path).endswith("gzip"):
            with gzip.open(file_path, "rb") as file_cache:
                return pickle.load(file_cache)

        with open(file_path, "rb") as file_cache:
            return pickle.load(file_cache)

    def uses_header(self) -> bool:
        return False


def file_handler_factory(cache_extension: str, *args, **kwargs) -> FileHandler:
    known_cache_extensions = [
        "csv",
        "csv.gz",
        "csv.zip",
        "pkl",
        "pkl.gz",
        "pickle",
        "pickle.gz",
        "parquet",
        "json",
    ]

    if "csv" in cache_extension:
        return CSVHandler(*args, **kwargs)
    if "pkl" in cache_extension or "pickle" in cache_extension:
        return PickleHandler(*args, **kwargs)
    if "json" in cache_extension:
        return JSONHandler(*args, **kwargs)
    if "parquet" in cache_extension:
        return ParquetHandler(*args, **kwargs)

    raise ValueError(
        f"Invalid cache extension:: {cache_extension} :: "
        f"Acceptable cache extensions: {known_cache_extensions}"
    )
