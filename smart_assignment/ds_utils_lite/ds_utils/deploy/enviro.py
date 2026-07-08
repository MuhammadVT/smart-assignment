"""Environment detection helpers."""

import os


class Environment:
    """Functions for detecting the runtime environment."""

    @staticmethod
    def on_ec2():
        return os.environ.get("USER") == "ec2-user" or os.environ.get("USER") == "hadoop"
