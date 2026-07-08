"""Abstractions for ds_utils classes."""

from abc import ABC, abstractmethod

import pandas as pd

from ds_utils.logger import Logger

logger = Logger(__name__)


class DataABC(ABC):
    """Abstract Data class for cache integration."""

    data_dir: str
    ignore_cache: bool

    @abstractmethod
    def get_handle(self, file_name: str, validate_extension=True):
        pass

    @abstractmethod
    def write(self, object_to_write: (pd.DataFrame, any), file_name: str):
        pass

    @abstractmethod
    def read(self, file_name):
        pass

    @abstractmethod
    def check_into_cache(self, cache_name, function_to_execute, *args, **kwargs):
        pass

    def cic(self, cache_name, function_to_execute, *args, **kwargs):
        return self.check_into_cache(cache_name, function_to_execute, *args, **kwargs)


class S3ABC(ABC):
    """Abstract S3 class."""

    @abstractmethod
    def upload(self, file_path: str, **kwargs):
        pass

    @abstractmethod
    def download(self, file_name: str, **kwargs):
        pass

    @abstractmethod
    def get(self, file_name: str, **kwargs):
        pass
