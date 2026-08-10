"""Hermetic tests for eval/case_set.py -- the declared eval CASE set.

No LLM backend and no ``google-adk[eval]``: this resolves cases from the
built-in fixtures or a curated candidates file, both of which are plain data.
"""

from __future__ import annotations

import json

import pytest

from eval.case_set import (
    CASE_SET_ENV,
    DEFAULT_CASE_SET,
    resolve_case_set,
    resolve_cases,
)
from eval.golden_cases import GOLDEN_CASES

# One curated candidate in the shape scripts/curate_feedback.py emits, carrying
# the production decision_id that makes a judge verdict joinable to a human label.
_CANDIDATE = {
    "eval_id": "curated_1210bd3e",
    "context": {
        "name": "Curated Bistro",
        "address": "1200 McKinney St, Houston, TX 77010",
        "order_quantity_cases": 90,
        "preferred_day": "TUE",
        "preferred_window": "07:00-10:00",
    },
    "observed_outcome": "recommend",
    "provenance": {"decision_id": "1210bd3e4a984ff0bfb72a5426af3ed6"},
}


def _write_candidates(tmp_path, candidates):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(candidates), encoding="utf-8")
    return str(path)


def test_unset_resolves_to_the_built_in_golden_fixtures(monkeypatch):
    monkeypatch.delenv(CASE_SET_ENV, raising=False)
    case_set = resolve_case_set()
    assert case_set.name == DEFAULT_CASE_SET
    assert case_set.kind == "code"
    assert case_set.is_default
    assert [c.eval_id for c in case_set.cases] == [c.eval_id for c in GOLDEN_CASES]


def test_blank_value_is_treated_as_unset(monkeypatch):
    # A commented-out .env line re-added empty, or an unset CI variable expanding
    # to nothing, means "not configured" -- not a path named "".
    monkeypatch.setenv(CASE_SET_ENV, "   ")
    assert resolve_case_set().is_default


def test_explicit_golden_is_the_same_set(monkeypatch):
    monkeypatch.setenv(CASE_SET_ENV, DEFAULT_CASE_SET)
    assert resolve_case_set().is_default


def test_resolve_cases_returns_a_plain_mutable_list(monkeypatch):
    # Callers pass this to select_cases/filter_cases_by_ids, which index and
    # slice it; the CaseSet itself holds a tuple so it can stay frozen.
    monkeypatch.delenv(CASE_SET_ENV, raising=False)
    cases = resolve_cases()
    assert isinstance(cases, list)
    assert len(cases) == len(GOLDEN_CASES)


def test_a_curated_file_becomes_a_file_backed_case_set(monkeypatch, tmp_path):
    path = _write_candidates(tmp_path, [_CANDIDATE])
    monkeypatch.setenv(CASE_SET_ENV, path)

    case_set = resolve_case_set()
    assert case_set.kind == "file"
    assert not case_set.is_default
    assert case_set.path == path
    assert case_set.name.startswith("file:")
    assert [c.eval_id for c in case_set.cases] == ["curated_1210bd3e"]


def test_a_curated_case_carries_its_decision_id(monkeypatch, tmp_path):
    # The whole point of running curated cases: decision_id is what a judge
    # verdict joins to a human label on (eval/judge_calibration.py). A built-in
    # fixture has none, so without this the join has nothing to work with.
    monkeypatch.setenv(CASE_SET_ENV, _write_candidates(tmp_path, [_CANDIDATE]))
    (case,) = resolve_case_set().cases
    assert case.decision_id == "1210bd3e4a984ff0bfb72a5426af3ed6"
    assert all(c.decision_id is None for c in GOLDEN_CASES)


def test_unreplayable_candidates_are_reported_not_silently_dropped(monkeypatch, tmp_path):
    # A PII-redacted address can't be geocoded, so case_source skips it. The set
    # must still say so, or a run scores fewer cases than the file holds and
    # nobody knows.
    redacted = {**_CANDIDATE, "eval_id": "curated_redacted"}
    redacted["context"] = {**_CANDIDATE["context"], "address": "[redacted]"}
    monkeypatch.setenv(CASE_SET_ENV, _write_candidates(tmp_path, [_CANDIDATE, redacted]))

    case_set = resolve_case_set()
    assert [c.eval_id for c in case_set.cases] == ["curated_1210bd3e"]
    assert [entry["eval_id"] for entry in case_set.skipped] == ["curated_redacted"]


def test_a_path_that_does_not_exist_raises_naming_the_valid_forms(monkeypatch, tmp_path):
    monkeypatch.setenv(CASE_SET_ENV, str(tmp_path / "nope.json"))
    with pytest.raises(ValueError) as excinfo:
        resolve_case_set()
    message = str(excinfo.value)
    assert CASE_SET_ENV in message
    assert DEFAULT_CASE_SET in message


def test_an_unparseable_file_raises_rather_than_yielding_nothing(monkeypatch, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv(CASE_SET_ENV, str(path))
    with pytest.raises(ValueError):
        resolve_case_set()


def test_a_file_with_no_replayable_cases_raises(monkeypatch, tmp_path):
    # Scoring zero cases would report success over nothing -- the same silent
    # green eval/run_config.py rejects for num_runs=0.
    redacted = {**_CANDIDATE}
    redacted["context"] = {**_CANDIDATE["context"], "address": "[redacted]"}
    monkeypatch.setenv(CASE_SET_ENV, _write_candidates(tmp_path, [redacted]))
    with pytest.raises(ValueError) as excinfo:
        resolve_case_set()
    assert "no replayable eval cases" in str(excinfo.value)


def test_a_non_default_selection_is_announced(monkeypatch, tmp_path, caplog):
    # Never invisible: same discipline as case_selection.select_cases warning
    # when SMART_ASSIGNMENT_EVAL_IDS narrows a run.
    monkeypatch.setenv(CASE_SET_ENV, _write_candidates(tmp_path, [_CANDIDATE]))
    with caplog.at_level("WARNING"):
        resolve_case_set()
    assert CASE_SET_ENV in caplog.text
