"""Tests for the pure join/transform in scripts/phoenix_curate.py (the Phoenix
I/O is not exercised here -- only the vendor-neutral transform that turns span
rows into the shared candidate-cases schema)."""

from __future__ import annotations

import importlib.util
import json
import pathlib

# scripts/ isn't a package; load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "phoenix_curate",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "phoenix_curate.py",
)
phoenix_curate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(phoenix_curate)


def _decision_row(trace_id="t-abc12345", **over):
    intake = {
        "name": "Woodlands Fresh Cafe",
        "address": "1201 Lake Woodlands Dr, The Woodlands, TX 77380",
        "order_quantity_cases": 150,
        "preferred_day": "THU",
        "preferred_window": "09:00-12:00",
    }
    output = {"decision": "RECOMMENDED", "recommended_route_id": "RTE-4300",
              "recommended_window": "08:30-11:30"}
    row = {
        "trace_id": trace_id,
        "span_id": "s-1",
        "input": json.dumps(intake),
        "output": json.dumps(output),
        "outcome": "recommend",
    }
    row.update(over)
    return row


def _feedback_row(target="t-abc12345", label="thumbs_down", **over):
    row = {"label": label, "target_trace_id": target, "note": "VIP",
           "score": None, "session_id": "s9", "annotator_id": "customer_web"}
    row.update(over)
    return row


def test_join_builds_candidate_matching_case_source_schema():
    cands = phoenix_curate.spans_to_candidates([_feedback_row()], [_decision_row()])
    assert len(cands) == 1
    c = cands[0]
    assert c["human_verdict"] == "negative"
    assert c["observed_outcome"] == "recommend"
    assert c["suggested_expected_outcome"] is None
    ctx = c["context"]
    assert ctx["address"].startswith("1201 Lake Woodlands")
    assert ctx["preferred_day"] == "THU"
    assert ctx["order_quantity_cases"] == 150
    assert c["provenance"]["trace_id"] == "t-abc12345"
    # The candidate is loadable by the file-backed eval loader.
    from eval.case_source import candidate_to_case
    case = candidate_to_case(c)
    assert case.customer.order_quantity_cases == 150


def test_only_matching_label_and_joinable_rows():
    fb = [
        _feedback_row(target="t-1", label="thumbs_up"),          # wrong label
        _feedback_row(target="t-missing", label="thumbs_down"),  # no decision span
        _feedback_row(target="t-2", label="thumbs_down"),        # joins
    ]
    dec = [_decision_row(trace_id="t-2")]
    cands = phoenix_curate.spans_to_candidates(fb, dec)
    assert [c["provenance"]["trace_id"] for c in cands] == ["t-2"]


def test_decision_without_input_payload_is_skipped():
    # A decision span with no replay payload (scrub on / flag off) can't be curated.
    dec = [_decision_row(input=None)]
    assert phoenix_curate.spans_to_candidates([_feedback_row()], dec) == []


def test_as_obj_and_query_helpers():
    assert phoenix_curate._as_obj('{"a": 1}') == {"a": 1}
    assert phoenix_curate._as_obj("not json") == {}
    assert phoenix_curate._as_obj({"b": 2}) == {"b": 2}
    q = phoenix_curate._query_from_intake(
        {"name": "N", "address": "A", "order_quantity_cases": 5,
         "preferred_day": "TUE", "preferred_window": "07:00-10:00"}
    )
    assert "N" in q and "5 cases" in q and "prefers TUE 07:00-10:00" in q
