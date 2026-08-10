"""Tests for the file-backed eval case loader (eval/case_source.py) — curated
candidates become runnable eval cases with no hand-copying into golden_cases.py."""

from __future__ import annotations

import json

from datetime import time

from eval.build_evalset import build_eval_set
from eval.case_source import candidate_to_case, load_curated_cases
from eval.golden_cases import intake_args
from smart_assignment.shared.models import PROSPECT_PLACEHOLDER_NAME, DayOfWeek


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


def test_carries_the_originating_decision_id():
    """The id a judge verdict and the human's label on the SAME decision join on
    (eval/judge_calibration.py). The minted eval_id only encodes 8 characters of
    it, so it has to travel as its own field."""
    candidate = _candidate(provenance={"decision_id": "1210bd3e4a984ff0bfb72a5426af3ed6"})
    assert candidate_to_case(candidate).decision_id == "1210bd3e4a984ff0bfb72a5426af3ed6"


def test_decision_id_is_none_without_provenance():
    """A hand-written or provenance-less candidate has nothing to join to, and
    says so rather than inventing an id."""
    assert candidate_to_case(_candidate()).decision_id is None
    assert candidate_to_case(_candidate(provenance={})).decision_id is None


def test_suggested_outcome_overrides_observed():
    case = candidate_to_case(_candidate(suggested_expected_outcome="escalate"))
    assert case.expected_outcome == "escalate"


def test_no_preference_is_fine():
    ctx = {"name": "X", "address": "5 Main St, Houston, TX", "order_quantity_cases": 10}
    case = candidate_to_case(_candidate(context=ctx))
    assert case.customer.preferred_slot is None
    assert "prefers" not in case.query


def test_the_unnamed_prospect_placeholder_is_not_replayed_as_a_name():
    """THE regression that made every curated case unusable.

    Production stores ``PROSPECT_PLACEHOLDER_NAME`` when a prospect has no
    business name. Replaying it as a name put "New prospect" at the head of the
    query and demanded ``name="New prospect"`` back from the agent -- which
    correctly reads it as a descriptor and omits the field. Result: tool
    trajectory 0.0 on every curated case, which looked like a broken agent."""
    ctx = {
        "name": PROSPECT_PLACEHOLDER_NAME,
        "address": "5 Main St, Houston, TX",
        "order_quantity_cases": 10,
    }
    case = candidate_to_case(_candidate(context=ctx))

    assert case.customer.name == ""
    assert "name" not in intake_args(case.customer)
    assert not case.query.startswith(PROSPECT_PLACEHOLDER_NAME)
    assert case.query.startswith("5 Main St")


def test_a_missing_name_is_treated_the_same_as_the_placeholder():
    """It used to substitute "Curated prospect", which had the identical
    problem -- an invented name the agent has no way to produce."""
    ctx = {"address": "5 Main St, Houston, TX", "order_quantity_cases": 10}
    case = candidate_to_case(_candidate(context=ctx))

    assert case.customer.name == ""
    assert "name" not in intake_args(case.customer)


def test_a_real_business_name_is_still_replayed():
    """The fix must not throw away a genuine name -- only the placeholder."""
    case = candidate_to_case(_candidate())

    assert case.customer.name == "Woodlands Fresh Cafe"
    assert intake_args(case.customer)["name"] == "Woodlands Fresh Cafe"
    assert case.query.startswith("Woodlands Fresh Cafe")


def test_the_query_and_the_expected_intake_args_always_agree_on_the_name():
    """The invariant behind all three: the trajectory metric compares intake args
    exactly, so a name may appear in the expectation only if it appears in the
    message the agent is given."""
    for name in (PROSPECT_PLACEHOLDER_NAME, "", "Real Cafe Ltd"):
        ctx = {
            "name": name,
            "address": "5 Main St, Houston, TX",
            "order_quantity_cases": 10,
        }
        case = candidate_to_case(_candidate(context=ctx))
        in_args = "name" in intake_args(case.customer)
        in_query = case.query.startswith(name) if name else False
        assert in_args == in_query, f"disagreement for {name!r}"


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
