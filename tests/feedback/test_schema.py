"""Unit tests for the feedback record schema + deterministic validation."""

from __future__ import annotations

import pytest

from smart_assignment.feedback.schema import (
    ANNOTATOR_HUMAN,
    FeedbackRecord,
    FeedbackTarget,
    FeedbackValidationError,
    validate_feedback,
)


def _rec(**over):
    target = over.pop("target", FeedbackTarget(decision_id="d1"))
    base = dict(target=target, label="thumbs_up", annotator_kind=ANNOTATOR_HUMAN)
    base.update(over)
    return FeedbackRecord(**base)


def test_valid_record_passes():
    validate_feedback(_rec(score=1.0, note="looks right"))


def test_to_dict_is_json_shaped():
    d = _rec(score=0.0).to_dict()
    assert d["label"] == "thumbs_up"
    assert d["target"]["decision_id"] == "d1"
    assert d["target"]["decision_kind"] == "final_response"


def test_missing_decision_id_rejected():
    with pytest.raises(FeedbackValidationError):
        validate_feedback(_rec(target=FeedbackTarget(decision_id="")))


def test_blank_label_rejected():
    with pytest.raises(FeedbackValidationError):
        validate_feedback(_rec(label="  "))


def test_unknown_annotator_kind_rejected():
    with pytest.raises(FeedbackValidationError):
        validate_feedback(_rec(annotator_kind="ROBOT"))


def test_unknown_decision_kind_rejected():
    with pytest.raises(FeedbackValidationError):
        validate_feedback(_rec(target=FeedbackTarget(decision_id="d1", decision_kind="nope")))


@pytest.mark.parametrize("score", [1.5, -2.0, float("nan"), float("inf"), True])
def test_bad_scores_rejected(score):
    with pytest.raises(FeedbackValidationError):
        validate_feedback(_rec(score=score))


@pytest.mark.parametrize("score", [-1.0, 0.0, 0.5, 1.0, None])
def test_good_scores_accepted(score):
    validate_feedback(_rec(score=score))
