"""Tests for the judge-calibration core (eval/judge_calibration.py) — pure metrics
+ the holistic/dimensional tiering. No judge model or backend needed."""

from __future__ import annotations

from eval.judge_calibration import (
    COMPOSITE,
    DIM_BRIEF_QUALITY,
    DIM_RESPONSE_CLARITY,
    HumanLabel,
    JudgeVerdict,
    Pair,
    build_pairs,
    calibrate,
    cohen_kappa,
    composite_pairs,
    confusion,
    dangerous_cell_rate,
    human_dimension_signals,
    human_labels_from_feedback,
    route_by_outcome,
    trust_band,
    verdicts_from_mapping,
)


def _pairs(spec):
    # spec: list of (human_positive, judge_positive)
    return [Pair("d", str(i), h, j, "holistic") for i, (h, j) in enumerate(spec)]


# --- pure metrics ---------------------------------------------------------------


def test_confusion_counts_and_dangerous_cell():
    pairs = _pairs([(True, True), (False, True), (True, False), (False, False)])
    c = confusion(pairs)
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    # dangerous = judge passed of the human-negatives: fp/(fp+tn) = 1/2
    assert dangerous_cell_rate(pairs) == 0.5


def test_kappa_perfect_and_none():
    assert cohen_kappa(_pairs([(True, True), (False, False)])) == 1.0
    assert cohen_kappa([]) is None


def test_rubber_stamp_judge_on_skewed_labels_scores_near_zero():
    # 9 human 👍, 1 👎; judge always passes -> 90% raw accuracy but no real skill.
    pairs = _pairs([(True, True)] * 9 + [(False, True)])
    assert confusion(pairs)["fp"] == 1
    kappa = cohen_kappa(pairs)
    assert kappa is not None and kappa <= 0.05  # ~0, not ~0.9


def test_trust_band():
    assert trust_band(0.8, 30) == "gate"
    assert trust_band(0.4, 30) == "advisory"
    assert trust_band(0.1, 30) == "distrust"
    assert trust_band(0.8, 5) == "insufficient"   # too few labels
    assert trust_band(None, 100) == "insufficient"


# --- tiering / routing ----------------------------------------------------------


def test_outcome_routing():
    assert route_by_outcome("escalate") == DIM_BRIEF_QUALITY
    assert route_by_outcome("recommend") == DIM_RESPONSE_CLARITY
    assert route_by_outcome(None) is None


def test_holistic_thumb_routes_by_outcome():
    label = HumanLabel(decision_id="d1", outcome="recommend", thumb="down")
    signals = list(human_dimension_signals(label))
    assert signals == [(DIM_RESPONSE_CLARITY, False, "holistic")]


def test_explicit_dimension_wins_over_routing():
    label = HumanLabel(
        decision_id="d1", outcome="recommend", thumb="down",
        dimensions={DIM_BRIEF_QUALITY: False},
    )
    signals = list(human_dimension_signals(label))
    assert signals == [(DIM_BRIEF_QUALITY, False, "dimensional")]


def test_note_tagger_tier():
    label = HumanLabel(
        decision_id="d1", outcome="recommend", thumb="down", note="the message was confusing"
    )

    def tagger(note):
        return {DIM_RESPONSE_CLARITY} if "confus" in note else set()

    signals = list(human_dimension_signals(label, note_tagger=tagger))
    assert signals == [(DIM_RESPONSE_CLARITY, False, "note")]


def test_unknown_outcome_holistic_yields_nothing_dimensional():
    label = HumanLabel(decision_id="d1", outcome=None, thumb="up")
    assert list(human_dimension_signals(label)) == []


# --- join + composite -----------------------------------------------------------


def test_build_pairs_joins_and_filters_non_human():
    labels = [
        HumanLabel("d1", outcome="recommend", thumb="up"),
        HumanLabel("d2", outcome="recommend", thumb="down"),
        HumanLabel("d3", outcome="recommend", thumb="up", annotator_kind="LLM"),  # skipped
    ]
    verdicts = [
        JudgeVerdict("d1", DIM_RESPONSE_CLARITY, True),
        JudgeVerdict("d2", DIM_RESPONSE_CLARITY, True),   # judge passed, human 👎 -> dangerous
    ]
    pairs = build_pairs(labels, verdicts)
    assert len(pairs) == 2
    assert confusion(pairs)["fp"] == 1


def test_composite_predicts_thumb_from_all_verdicts():
    labels = [HumanLabel("d1", thumb="up"), HumanLabel("d2", thumb="up")]
    verdicts = [
        JudgeVerdict("d1", DIM_RESPONSE_CLARITY, True),
        JudgeVerdict("d1", DIM_BRIEF_QUALITY, True),      # all pass -> predicted 👍 (agrees)
        JudgeVerdict("d2", DIM_RESPONSE_CLARITY, False),  # one fails -> predicted 👎 (disagrees)
    ]
    pairs = composite_pairs(labels, verdicts)
    assert {p.decision_id: p.judge_positive for p in pairs} == {"d1": True, "d2": False}
    assert all(p.dimension == COMPOSITE for p in pairs)


def test_calibrate_end_to_end_report_shape():
    labels = [HumanLabel(f"d{i}", outcome="recommend", thumb="up") for i in range(3)]
    verdicts = [JudgeVerdict(f"d{i}", DIM_RESPONSE_CLARITY, True) for i in range(3)]
    report = calibrate(labels, verdicts, min_n=2)
    assert report["totals"]["aligned_pairs"] == 3
    assert report["dimensions"][DIM_RESPONSE_CLARITY]["trust"] == "gate"
    assert "composite" in report


# --- vendor-free label reader ---------------------------------------------------


def test_human_labels_from_feedback(tmp_path):
    from smart_assignment.feedback.schema import FeedbackRecord, FeedbackTarget
    from smart_assignment.feedback.store import append_record

    path = str(tmp_path / "log.jsonl")
    append_record(
        FeedbackRecord(
            target=FeedbackTarget(decision_id="d1"),
            label="thumbs_down", note="confusing",
            context={"outcome": "recommend"},
        ),
        path,
    )
    append_record(
        FeedbackRecord(
            target=FeedbackTarget(decision_id="d2"),
            label="thumbs_up", annotator_kind="LLM",  # auto-judge, not ground truth
        ),
        path,
    )
    labels = human_labels_from_feedback(path)
    assert [x.decision_id for x in labels] == ["d1"]  # LLM record filtered out
    assert labels[0].thumb == "down" and labels[0].outcome == "recommend"


def test_verdicts_from_mapping():
    verdicts = verdicts_from_mapping({"d1": {"response_clarity": {"passed": True, "score": 0.9}}})
    assert verdicts[0] == JudgeVerdict("d1", "response_clarity", True, 0.9)
