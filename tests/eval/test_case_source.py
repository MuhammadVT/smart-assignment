"""Tests for the file-backed eval case loader (eval/case_source.py) — curated
candidates become runnable eval cases with no hand-copying into golden_cases.py."""

from __future__ import annotations

import json

from datetime import time

from eval.build_evalset import build_eval_set
from eval.case_source import candidate_to_case, load_curated_cases
from eval.golden_cases import intake_args
from smart_assignment.shared.models import DayOfWeek


def _candidate(**over):
    base = {
        "eval_id": "phoenix_ab12cd34_negative",
        "human_verdict": "negative",
        "human_label": "thumbs_down",
        "suggested_expected_outcome": None,
        "observed_outcome": "recommend",
        "note": "VIP — should have been reviewed",
        "context": {
            "name": "Woodlands Fresh Cafe",
            "address": "1201 Lake Woodlands Dr, The Woodlands, TX 77380",
            "order_quantity_cases": 150,
            "preferred_day": "THU",
            "preferred_window": "09:00-12:00",
            "outcome": "recommend",
        },
    }
    base.update(over)
    return base


def test_reconstructs_profile_with_preference():
    case = candidate_to_case(_candidate())
    assert case.customer.name == "Woodlands Fresh Cafe"
    assert case.customer.order_quantity_cases == 150
    slot = case.customer.preferred_slot
    assert slot.day == DayOfWeek.THU
    assert slot.window == (time(9, 0), time(12, 0))
    # The query is consistent with the profile, so the trajectory expectation holds.
    assert "prefers THU 09:00-12:00" in case.query
    args = intake_args(case.customer)
    assert args["preferred_day"] == "THU" and args["preferred_window_start"] == "09:00"


def test_suggested_outcome_overrides_observed():
    case = candidate_to_case(_candidate(suggested_expected_outcome="escalate"))
    assert case.expected_outcome == "escalate"


def test_no_preference_is_fine():
    ctx = {"name": "X", "address": "5 Main St, Houston, TX", "order_quantity_cases": 10}
    case = candidate_to_case(_candidate(context=ctx))
    assert case.customer.preferred_slot is None
    assert "prefers" not in case.query


def test_load_skips_redacted_and_missing(tmp_path):
    good = _candidate(eval_id="good")
    redacted = _candidate(
        eval_id="redacted",
        context={"name": "Y", "address": "[redacted], Houston, TX", "order_quantity_cases": 20},
    )
    no_qty = _candidate(eval_id="no_qty", context={"name": "Z", "address": "9 Elm St, Houston"})
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([good, redacted, no_qty]), encoding="utf-8")

    cases, skipped = load_curated_cases(str(path))
    assert [c.eval_id for c in cases] == ["good"]
    reasons = {s["eval_id"]: s["reason"] for s in skipped}
    assert "redacted" in reasons["redacted"] or "PII" in reasons["redacted"]
    assert "order_quantity_cases" in reasons["no_qty"]


def test_load_dedupes_eval_ids(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps([_candidate(eval_id="dup"), _candidate(eval_id="dup")]), "utf-8")
    cases, skipped = load_curated_cases(str(path))
    assert len(cases) == 1
    assert any(s["reason"] == "duplicate eval_id" for s in skipped)


def test_curated_cases_build_a_valid_evalset(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps([_candidate()]), encoding="utf-8")
    cases, _ = load_curated_cases(str(path))
    evalset = build_eval_set(cases, captured={})
    assert len(evalset["eval_cases"]) == 1
    invocation = evalset["eval_cases"][0]["conversation"][0]
    # The reconstructed intake message + the expected tool trajectory are present.
    assert "Woodlands Fresh Cafe" in invocation["user_content"]["parts"][0]["text"]
    tool_names = [t["name"] for t in invocation["intermediate_data"]["tool_uses"]]
    assert tool_names[0] == "intake_customer"
