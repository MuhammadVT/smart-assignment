"""Unit tests for the offline curation of feedback into candidate eval cases."""

from __future__ import annotations

import json

from smart_assignment.feedback.curate import curate_feedback, write_curation
from smart_assignment.feedback.schema import (
    ANNOTATOR_CODE,
    FeedbackRecord,
    FeedbackTarget,
)
from smart_assignment.feedback.store import append_record


def _write(path, decision_id, label, *, created_at, context, annotator="HUMAN", score=None):
    append_record(
        FeedbackRecord(
            target=FeedbackTarget(decision_id=decision_id, session_id="s1"),
            label=label,
            annotator_kind=annotator,
            score=score,
            created_at=created_at,
            context=context,
        ),
        str(path),
    )


def test_negative_on_recommend_suggests_escalate(tmp_path):
    log = tmp_path / "log.jsonl"
    _write(log, "d1", "thumbs_down", created_at="2026-01-01T00:00:00Z",
           context={"outcome": "recommend", "name": "Galleria Grill", "address": "5085 Westheimer"})
    cases = curate_feedback(str(log))
    assert len(cases) == 1
    c = cases[0]
    assert c.human_verdict == "negative"
    assert c.suggested_expected_outcome == "escalate"
    assert "Galleria Grill" in c.query


def test_positive_confirms_observed_outcome(tmp_path):
    log = tmp_path / "log.jsonl"
    _write(log, "d2", "thumbs_up", created_at="2026-01-01T00:00:00Z",
           context={"outcome": "recommend"})
    c = curate_feedback(str(log))[0]
    assert c.human_verdict == "positive"
    assert c.suggested_expected_outcome == "recommend"


def test_latest_annotation_per_decision_wins(tmp_path):
    log = tmp_path / "log.jsonl"
    _write(log, "d3", "thumbs_up", created_at="2026-01-01T00:00:00Z",
           context={"outcome": "recommend"})
    _write(log, "d3", "thumbs_down", created_at="2026-02-01T00:00:00Z",
           context={"outcome": "recommend"})
    cases = curate_feedback(str(log))
    assert len(cases) == 1
    assert cases[0].human_verdict == "negative"


def test_only_negative_filter(tmp_path):
    log = tmp_path / "log.jsonl"
    _write(log, "d4", "thumbs_up", created_at="2026-01-01T00:00:00Z",
           context={"outcome": "recommend"})
    _write(log, "d5", "thumbs_down", created_at="2026-01-01T00:00:00Z",
           context={"outcome": "recommend"})
    cases = curate_feedback(str(log), only_negative=True)
    assert [c.provenance["decision_id"] for c in cases] == ["d5"]


def test_non_human_annotations_skipped(tmp_path):
    log = tmp_path / "log.jsonl"
    _write(log, "d6", "bad", created_at="2026-01-01T00:00:00Z",
           context={"outcome": "recommend"}, annotator=ANNOTATOR_CODE)
    assert curate_feedback(str(log)) == []


def test_write_curation_emits_json_array(tmp_path):
    log = tmp_path / "log.jsonl"
    _write(log, "d7", "thumbs_down", created_at="2026-01-01T00:00:00Z",
           context={"outcome": "recommend"})
    out = tmp_path / "out" / "candidates.json"
    write_curation(curate_feedback(str(log)), str(out))
    data = json.loads(out.read_text())
    assert isinstance(data, list) and data[0]["human_verdict"] == "negative"
