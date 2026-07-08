"""Run mode configuration for ds_utils-lite."""


class Mode:
    """Store run configurations and share with other ds_utils classes."""

    LOCAL = "local"
    DEV = "dev"
    QA = "qa"
    PERF = "perf"
    PREPROD = "preprod"
    PROD = "prod"

    def __init__(self, name):
        self.run_mode = name
        assert self.run_mode in {
            Mode.LOCAL,
            Mode.DEV,
            Mode.QA,
            Mode.PERF,
            Mode.PREPROD,
            Mode.PROD,
        }, 'Requested mode: "' + name + '", is UNACCEPTABLE.'

    def is_prod(self):
        return self.run_mode == Mode.PROD or self.run_mode == Mode.PREPROD

    def is_preprod(self):
        return self.run_mode == Mode.PREPROD

    def is_nonprod(self):
        return self.run_mode == Mode.DEV or self.run_mode == Mode.QA or self.run_mode == self.PERF

    def is_dangerous(self):
        return self.run_mode == Mode.PROD or self.run_mode == Mode.QA

    def is_qa(self):
        return self.run_mode == Mode.QA

    def is_dev(self):
        return self.run_mode == Mode.DEV

    def is_local(self):
        return self.run_mode == Mode.LOCAL

    def __str__(self):
        return self.run_mode

    def __repr__(self):
        return f"Mode('{self.run_mode}')"

    def get_mode_value(self):
        return self.run_mode
