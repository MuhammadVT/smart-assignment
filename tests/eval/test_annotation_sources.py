"""Tests for Tier-3 annotation sources (eval/annotation_sources.py) — the pure
verdict/label parsing, vendor-free dimensional merge, and the shared backend-row
normalizer used by the Phoenix/Langfuse adapters (no live backend)."""

from __future__ import annotations

from eval.annotation_sources import (
    _rows_to_labels,
    load_labels,
    parse_dimension_label,
    verdict_to_bool,
    vendorfree_labels,
)
from eval.judge_calibration import DIM_BRIEF_QUALITY, DIM_RESPONSE_CLARITY


def test_verdict_to_bool_words_and_scores():
    assert verdict_to_bool("good") is True
    assert verdict_to_bool("incorrect") is False
    assert verdict_to_bool("4") is True          # 1-5 scale
    assert verdict_to_bool("2") is False
    assert verdict_to_bool("0.9") is True         # 0-1 scale
    assert verdict_to_bool("0.3") is False
    assert verdict_to_bool("banana") is None


def test_parse_dimension_label():
    assert parse_dimension_label("response_clarity:bad") == (DIM_RESPONSE_CLARITY, False)
    assert parse_dimension_label("brief_quality:5") == (DIM_BRIEF_QUALITY, True)
    assert parse_dimension_label("thumbs_down") is None       # not a dimension label
    assert parse_dimension_label("not_a_dim:good") is None


def test_vendorfree_merges_holistic_and_dimensional(tmp_path):
    from smart_assignment.feedback.schema import FeedbackRecord, FeedbackTarget
    from smart_assignment.feedback.store import append_record

    path = str(tmp_path / "log.jsonl")
    # A holistic end-user thumb...
    append_record(
        FeedbackRecord(
            target=FeedbackTarget(decision_id="d1"),
            label="thumbs_down", note="confusing", context={"outcome": "recommend"},
        ),
        path,
    )
    # ...and a later SME dimensional annotation on the same decision.
    append_record(
        FeedbackRecord(
            target=FeedbackTarget(decision_id="d1"),
            label="response_clarity:bad", annotator_id="sme_amy",
        ),
        path,
    )
    labels = vendorfree_labels(path)
    assert len(labels) == 1
    label = labels[0]
    assert label.thumb == "down"
    assert label.outcome == "recommend"
    assert label.dimensions == {DIM_RESPONSE_CLARITY: False}


def test_rows_to_labels_normalizes_backend_annotations():
    rows = [
        {"key": "trace-1", "dimension": "response_clarity", "value": "good"},
        {"key": "trace-1", "dimension": "brief_quality", "value": 2},
        {"key": "trace-2", "thumb": "thumbs_up", "outcome": "recommend"},
    ]
    labels = {x.decision_id: x for x in _rows_to_labels(rows)}
    assert labels["trace-1"].dimensions == {
        DIM_RESPONSE_CLARITY: True,
        DIM_BRIEF_QUALITY: False,
    }
    assert labels["trace-2"].thumb == "up"


def test_load_labels_dispatch_and_errors():
    import pytest

    with pytest.raises(ValueError):
        load_labels("vendorfree")  # missing log
    with pytest.raises(ValueError):
        load_labels("bogus", log="x")
