"""Pytest configuration for the eval suite.

Locks every ``pytest eval/...`` run onto the *declared* eval dataset
(``SMART_ASSIGNMENT_EVAL_DATASET``, default ``mock``) before any test collects,
so eval measures against a reproducible, pinned world and can never silently
inherit a developer's ambient ``SMART_ASSIGNMENT_DATA_SOURCE`` (which points at
an *uncommitted* ``data/dev/`` snapshot). See ``eval/dataset.py`` for the why.

Applied at import time -- conftest is imported before collection -- so the pin
precedes any agent/route import a test triggers. It sets process env vars, so it
is a no-op for the hermetic ``tests/`` suite, which does not load this conftest
(pytest scopes conftest by directory; the canonical ``python3 -m pytest -q`` run
has ``testpaths = ["tests"]``).
"""

from __future__ import annotations

from eval.dataset import apply_eval_dataset

# Lock the dataset for the whole eval session (loud + idempotent; see
# eval.dataset.apply_eval_dataset).
apply_eval_dataset()
