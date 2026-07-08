"""Stdlib logging wrapper compatible with ds_utils.Logger."""

import logging
import sys


class Logger:
    """Lightweight logger using the standard library instead of daiquiri."""

    _is_setup = False

    def __init__(self, logger_name=__name__, level=logging.INFO, log_format=None):
        if log_format is not None:
            formatter = logging.Formatter(fmt=log_format["str"], datefmt=log_format["tm"])
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(formatter)
            logging.basicConfig(level=level, handlers=[handler], force=True)
            Logger._is_setup = True
        elif not Logger._is_setup:
            Logger.setup()

        self._log = logging.getLogger(logger_name)
        self.set_level(level)

    def set_level(self, level):
        if level == "DEBUG":
            level = logging.DEBUG
        elif level == "INFO":
            level = logging.INFO
        elif level == "WARN":
            level = logging.WARN
        elif level == "ERROR":
            level = logging.ERROR
        elif level == "CRITICAL":
            level = logging.CRITICAL
        elif level is None:
            level = logging.INFO

        self._log.setLevel(level)

    def info(self, message):
        self._log.info(message)

    def debug(self, message):
        self._log.debug(message)

    def warning(self, message):
        self._log.warning(message)

    def critical(self, message):
        self._log.critical(message)

    def error(self, message):
        self._log.error(message)

    @classmethod
    def setup(cls, log_file: str = None, formatter: logging.Formatter = None, default_level: int = logging.INFO):
        handlers = []
        stream_handler = logging.StreamHandler(sys.stderr)
        if formatter is not None:
            stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            if formatter is not None:
                file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        logging.basicConfig(level=default_level, handlers=handlers, force=True)
        cls._is_setup = True
