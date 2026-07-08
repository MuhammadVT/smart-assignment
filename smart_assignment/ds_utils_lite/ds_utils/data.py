"""Local data caching for ds_utils-lite."""

import datetime
import os

import pandas as pd

from ds_utils.base_abc import DataABC
from ds_utils.deploy.enviro import Environment
from ds_utils.deploy.mode import Mode
from ds_utils.file_handlers import file_handler_factory
from ds_utils.formatting import ColumnFormatter
from ds_utils.logger import Logger

logger = Logger(__name__)


class Data(DataABC):
    """Handles local caching of dataframes and other objects."""

    def __init__(
        self,
        rm: Mode,
        data_location=None,
        session_date=datetime.datetime.today().date(),
        ignore_cache=False,
        default_cache_extension="csv.gz",
        repo_dir="",
        formatter=ColumnFormatter(),
    ):
        self._run_mode = rm
        self.ref_data_dir = os.path.join(repo_dir, "ref")
        if os.path.exists(os.path.join(repo_dir, "reference")):
            self.ref_data_dir = os.path.join(repo_dir, "reference")
        self.session_date = session_date
        if data_location is None:
            data_location = "/dev/shm" if Environment.on_ec2() else "data"

        self.default_cache_extension = default_cache_extension.lstrip(". ")
        _ = file_handler_factory(self.default_cache_extension)

        self.data_dir = data_location

        if self._run_mode is not None:
            self.data_dir = os.path.join(self.data_dir, self._run_mode.run_mode)

        if self.session_date is not None:
            self.data_dir = os.path.join(self.data_dir, str(self.session_date))

        os.makedirs(self.data_dir, exist_ok=True)

        try:
            self.ignore_cache = ignore_cache if self._run_mode.is_local() or self._run_mode.is_dev() else True
        except AttributeError:
            self.ignore_cache = ignore_cache

        self.formatter = formatter

    def _validate_extension(self, file_name: str) -> (str, str):
        file_name_split = file_name.split(".")

        if len(file_name_split) == 1:
            file_name = file_name + "." + self.default_cache_extension
            extension = self.default_cache_extension
        elif len(file_name_split) == 2:
            extension = file_name_split[-1]
        elif len(file_name_split) == 3:
            extension = ".".join(file_name_split[-2:])
        else:
            extension = ".".join(file_name_split[1:])

        return file_name, extension

    def _write(self, object_to_write: (pd.DataFrame, any), directory: str, file_name: str):
        file_name, extension = self._validate_extension(file_name)
        file_path = os.path.join(directory, file_name)
        handler = file_handler_factory(extension)
        handler.write(object_to_write, file_path)

    def write(self, object_to_write: (pd.DataFrame, any), file_name: str):
        self._write(object_to_write, self.data_dir, file_name)

    def write_ref(self, object_to_write: (pd.DataFrame, any), file_name: str):
        os.makedirs(self.ref_data_dir, exist_ok=True)
        self._write(object_to_write, self.ref_data_dir, file_name)

    def _read(self, path, file_name):
        file_name, extension = self._validate_extension(file_name)
        file_path = os.path.join(path, file_name)
        handler = file_handler_factory(extension)
        return handler.read(file_path)

    def read(self, file_name):
        return self._read(self.data_dir, file_name)

    def read_ref(self, file_name):
        return self._read(self.ref_data_dir, file_name)

    def get_handle(self, file_name: str, validate_extension=True):
        if validate_extension:
            file_name, _ = self._validate_extension(file_name)
        return os.path.join(self.data_dir, file_name)

    def get_ref_handle(self, file_name: str, validate_extension=True):
        if validate_extension:
            file_name, _ = self._validate_extension(file_name)
        return os.path.join(self.ref_data_dir, file_name)

    def check_into_cache(self, cache_name, function_to_execute, *args, **kwargs):
        if self.ignore_cache:
            logger.info(f"Ignoring cache. Running function: {function_to_execute.__name__}")
            cache_object = function_to_execute(*args, **kwargs)
            self.write(cache_object, cache_name)
        else:
            try:
                cache_object = self.read(cache_name)
                logger.info("Found cache: " + cache_name)
            except FileNotFoundError:
                logger.info(f"No cache found. Running function: {function_to_execute.__name__}")
                cache_object = function_to_execute(*args, **kwargs)
                self.write(cache_object, cache_name)

        return cache_object

    def set_ignore_cache(self, value: bool = True):
        self.ignore_cache = value

    def ignore_cache_exists(self, file_name):
        tmp_path = os.path.join(self.data_dir, file_name)
        return self.ignore_cache | (not os.path.exists(tmp_path))

    def cic(self, cache_name, function_to_execute, *args, **kwargs):
        return self.check_into_cache(cache_name, function_to_execute, *args, **kwargs)
