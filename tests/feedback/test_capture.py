"""Unit tests for the capture orchestrator (flag gate, scrub toggle, persist)."""

from __future__ import annotations

import pytest

from smart_assignment.feedback.capture import record_feedback
from smart_assignment.feedback.schema import (
    FeedbackRecord,
    FeedbackTarget,
    FeedbackValidationError,
)
from smart_assignment.feedback.store import read_records
from smart_assignment.shared.config import Config


def _rec(**over):
    target = over.pop("target", FeedbackTarget(decision_id="d1", session_id="s1"))
    base = dict(target=target, label="thumbs_down", note="wrong 1200 McKinney St order")
    base.update(over)
    return FeedbackRecord(**base)


def _config(tmp_path, **over):
    return Config(
        use_human_feedback=True,
        feedback_log_path=str(tmp_path / "annotations.jsonl"),
        use_tracing=False,  # no OTLP emit in the offline test env
        **over,
    )


def test_flag_off_is_noop(tmp_path):
    cfg = _config(tmp_path)
    cfg = Config(use_human_feedback=False, feedback_log_path=cfg.feedback_log_path)
    status = record_feedback(cfg, _rec())
    assert status["disabled"] is True
    assert read_records(cfg.feedback_log_path) == []


def test_invalid_record_raises(tmp_path):
    with pytest.raises(FeedbackValidationError):
        record_feedback(_config(tmp_path), _rec(label=""))


def test_persists_and_scrubs_by_default(tmp_path):
    cfg = _config(tmp_path, feedback_scrub_pii=True)
    status = record_feedback(cfg, _rec(context={"address": "1200 McKinney St, Houston"}))
    assert status["persisted"] is True
    assert status["disabled"] is False

    stored = read_records(cfg.feedback_log_path)
    assert len(stored) == 1
    # Note + context PII were scrubbed before persistence.
    assert "McKinney" not in (stored[0].note or "")
    assert "McKinney" not in stored[0].context.get("address", "")


def test_scrub_off_retains_pii(tmp_path):
    cfg = _config(tmp_path, feedback_scrub_pii=False)
    record_feedback(cfg, _rec(context={"address": "1200 McKinney St, Houston"}))
    stored = read_records(cfg.feedback_log_path)
    # On a trusted network the real PII is kept on purpose.
    assert "McKinney" in (stored[0].note or "")
    assert "McKinney" in stored[0].context.get("address", "")
