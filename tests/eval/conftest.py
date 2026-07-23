"""Test isolation for eval tests.

Some eval tests pin a dataset (``apply_eval_dataset`` / ``pin_dataset``), which
writes the data-source / geocoder / snapshot-dir env vars directly to
``os.environ`` -- outside pytest's ``monkeypatch`` bookkeeping. Left unrestored,
a pinned (and often now-deleted tmp) snapshot would leak into unrelated tests
that run later in the same process. This autouse fixture snapshots and restores
those vars around every test here, and drops the route cache, so pinning is
always local to the test that did it.
"""

from __future__ import annotations

import os

import pytest

_PINNED_ENV = (
    "SMART_ASSIGNMENT_DATA_SOURCE",
    "SMART_ASSIGNMENT_GEOCODER",
    "SMART_ASSIGNMENT_DATA_SOURCE_STRICT",
    "SMART_ASSIGNMENT_SNAPSHOT_DIR",
    "SMART_ASSIGNMENT_EVAL_DATASET",
    "SMART_ASSIGNMENT_SNAPSHOTS_ROOT",
)


@pytest.fixture(autouse=True)
def _restore_dataset_env():
    saved = {key: os.environ.get(key) for key in _PINNED_ENV}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            from smart_assignment.integrations.route_capacity_client import clear_route_cache

            clear_route_cache()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
