"""Unit tests for the durable JSONL feedback store."""

from __future__ import annotations

from smart_assignment.feedback.schema import FeedbackRecord, FeedbackTarget
from smart_assignment.feedback.store import append_record, iter_records, read_records


def _rec(decision_id, label="thumbs_down", **over):
    return FeedbackRecord(
        target=FeedbackTarget(decision_id=decision_id, session_id="s1"),
        label=label,
        **over,
    )


def test_append_and_read_round_trip(tmp_path):
    path = str(tmp_path / "nested" / "annotations.jsonl")  # parent dir auto-created
    assert append_record(_rec("d1", score=0.0, note="wrong route"), path) is True
    assert append_record(_rec("d2", label="thumbs_up", score=1.0), path) is True

    records = read_records(path)
    assert [r.target.decision_id for r in records] == ["d1", "d2"]
    assert records[0].note == "wrong route"
    assert records[1].label == "thumbs_up"
    assert records[0].context == {}


def test_missing_file_yields_nothing(tmp_path):
    assert read_records(str(tmp_path / "absent.jsonl")) == []


def test_malformed_lines_skipped(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"target": {"decision_id": "ok"}, "label": "thumbs_up"}\n'
                    "not json at all\n"
                    "\n", encoding="utf-8")
    records = list(iter_records(str(path)))
    assert len(records) == 1
    assert records[0].target.decision_id == "ok"


def test_append_failure_returns_false_not_raise(tmp_path):
    # A directory where the file should be -> open() fails, but we must not raise.
    clash = tmp_path / "log.jsonl"
    clash.mkdir()
    assert append_record(_rec("d1"), str(clash)) is False
