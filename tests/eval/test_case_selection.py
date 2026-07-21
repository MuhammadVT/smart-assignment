"""Hermetic tests for eval/case_selection.py -- the SMART_ASSIGNMENT_EVAL_IDS
knob's guardrails: local-only, rejected under CI, loud when it narrows, and the
explicit filter_cases_by_ids primitive that capture uses instead of the env var.
"""

from __future__ import annotations

import logging

import pytest

from eval.case_selection import (
    EVAL_IDS_ENV,
    filter_cases_by_ids,
    parse_eval_ids,
    select_cases,
)
from eval.golden_cases import GOLDEN_CASES

_IDS = [c.eval_id for c in GOLDEN_CASES]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Isolate every test from the ambient environment (including the CI=true that
    # GitHub Actions sets while this hermetic suite runs) so behavior is explicit.
    monkeypatch.delenv(EVAL_IDS_ENV, raising=False)
    monkeypatch.delenv("CI", raising=False)


def test_parse_eval_ids_drops_blanks():
    assert parse_eval_ids(" a , ,b,") == ["a", "b"]


def test_unset_returns_all_cases():
    assert [c.eval_id for c in select_cases(GOLDEN_CASES)] == _IDS


def test_local_subset_selects_named_in_order_and_warns(monkeypatch, caplog):
    monkeypatch.setenv(EVAL_IDS_ENV, f"{_IDS[1]},{_IDS[0]}")
    with caplog.at_level(logging.WARNING, logger="eval.case_selection"):
        selected = select_cases(GOLDEN_CASES)
    assert [c.eval_id for c in selected] == [_IDS[1], _IDS[0]]  # order preserved
    assert any("is set: running" in r.message for r in caplog.records)  # loud


def test_unknown_id_raises_naming_the_env_var(monkeypatch):
    monkeypatch.setenv(EVAL_IDS_ENV, "no_such_case")
    with pytest.raises(ValueError, match=EVAL_IDS_ENV):
        select_cases(GOLDEN_CASES)


def test_rejected_under_ci(monkeypatch):
    # The core guard: a subset filter must never silently narrow a CI run.
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv(EVAL_IDS_ENV, _IDS[0])
    with pytest.raises(ValueError, match="must not narrow CI"):
        select_cases(GOLDEN_CASES)


def test_ci_without_filter_still_runs_all(monkeypatch):
    # CI=true with the var UNSET is the normal CI path -- returns all, no error.
    monkeypatch.setenv("CI", "true")
    assert [c.eval_id for c in select_cases(GOLDEN_CASES)] == _IDS


def test_filter_cases_by_ids_is_explicit_and_ignores_env(monkeypatch):
    # capture's primitive: reads no environment, so an ambient EVAL_IDS/CI can't
    # change what it selects.
    monkeypatch.setenv(EVAL_IDS_ENV, _IDS[0])
    monkeypatch.setenv("CI", "true")
    selected = filter_cases_by_ids(GOLDEN_CASES, [_IDS[2]], source="--ids")
    assert [c.eval_id for c in selected] == [_IDS[2]]


def test_filter_cases_by_ids_unknown_raises_naming_source():
    with pytest.raises(ValueError, match="--ids names unknown"):
        filter_cases_by_ids(GOLDEN_CASES, ["nope"], source="--ids")
