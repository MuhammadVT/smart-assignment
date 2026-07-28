"""Tests for batch result records and the JSONL sink (batch/sink.py)."""

from __future__ import annotations

import json

from smart_assignment.batch.sink import (
    OUTCOME_ESCALATE,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_RECOMMEND,
    BatchRecord,
    JsonlResultSink,
)


def test_to_json_includes_only_set_optional_fields():
    recommend = BatchRecord(
        prospect_id="MOCK-001",
        generated_at="t0",
        outcome=OUTCOME_RECOMMEND,
        payload={"frontendHtml": "<div/>"},
    )
    out = recommend.to_json()
    assert out["outcome"] == OUTCOME_RECOMMEND
    assert out["payload"] == {"frontendHtml": "<div/>"}
    assert "error" not in out and "triage_brief" not in out and "review_reason" not in out

    attention = BatchRecord(
        prospect_id="MOCK-002",
        generated_at="t0",
        outcome=OUTCOME_NEEDS_ATTENTION,
        error="address not found: nowhere",
    )
    out = attention.to_json()
    assert out["error"] == "address not found: nowhere"
    assert "payload" not in out  # no decision was produced

    escalate = BatchRecord(
        prospect_id="MOCK-003",
        generated_at="t0",
        outcome=OUTCOME_ESCALATE,
        payload={"frontendHtml": "<div/>"},
        review_reason="thin margin",
        triage_brief="SITUATION\n...",
    )
    out = escalate.to_json()
    assert out["review_reason"] == "thin margin"
    assert out["triage_brief"].startswith("SITUATION")
    assert out["payload"]


def test_jsonl_sink_round_trips(tmp_path):
    path = tmp_path / "nested" / "results.jsonl"  # parent dir is created
    records = [
        BatchRecord("MOCK-001", "t0", OUTCOME_RECOMMEND, payload={"frontendHtml": "<a/>"}),
        BatchRecord("MOCK-002", "t0", OUTCOME_NEEDS_ATTENTION, error="bad address"),
    ]

    with JsonlResultSink(path) as sink:
        for record in records:
            sink.emit(record)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    reloaded = [json.loads(line) for line in lines]
    assert reloaded[0]["prospect_id"] == "MOCK-001"
    assert reloaded[0]["payload"] == {"frontendHtml": "<a/>"}
    assert reloaded[1]["outcome"] == OUTCOME_NEEDS_ATTENTION
    assert reloaded[1]["error"] == "bad address"
