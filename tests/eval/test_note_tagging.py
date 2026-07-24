"""Tests for the Tier-2 note tagger (eval/note_tagging.py) — keyword map (pure)
and the opt-in LLM suggestion (monkeypatched, never a real backend call)."""

from __future__ import annotations

from eval import note_tagging
from eval.judge_calibration import (
    DIM_BRIEF_QUALITY,
    DIM_DECISION_CORRECT,
    DIM_RESPONSE_CLARITY,
    DIM_SLOT_REASONABLE,
    HumanLabel,
    human_dimension_signals,
)
from eval.note_tagging import keyword_tags, make_note_tagger


def test_keyword_tags_clarity():
    assert keyword_tags("the message was really confusing") == {DIM_RESPONSE_CLARITY}


def test_keyword_tags_decision_and_slot():
    assert DIM_DECISION_CORRECT in keyword_tags("wrong route, should have escalated")
    assert DIM_SLOT_REASONABLE in keyword_tags("the delivery window was wrong")


def test_keyword_tags_brief():
    assert keyword_tags("the escalation brief was incomplete") == {DIM_BRIEF_QUALITY}


def test_keyword_tags_none():
    assert keyword_tags("thanks, looks great") == set()


def test_parse_dimension_list_constrained():
    parsed = note_tagging._parse_dimension_list("response_clarity, slot_reasonable, banana")
    assert parsed == {DIM_RESPONSE_CLARITY, DIM_SLOT_REASONABLE}
    assert note_tagging._parse_dimension_list("none") == set()


def test_make_note_tagger_keyword_only():
    tagger = make_note_tagger()  # no config, no LLM
    assert tagger("confusing wording") == {DIM_RESPONSE_CLARITY}


def test_llm_tagging_unions_and_is_optional(monkeypatch):
    # Keyword finds clarity; LLM additionally suggests slot_reasonable.
    monkeypatch.setattr(
        note_tagging, "llm_suggest_tags", lambda note, config: {DIM_SLOT_REASONABLE}
    )
    from smart_assignment.shared.config import Config

    tagger = make_note_tagger(Config(), use_llm=True)
    assert tagger("confusing") == {DIM_RESPONSE_CLARITY, DIM_SLOT_REASONABLE}


def test_llm_suggest_tags_degrades_to_empty_on_failure(monkeypatch):
    # No backend/credentials -> generate_text raises -> empty set (keyword-only).
    import smart_assignment.shared.llm as llm_mod

    def boom(*a, **k):
        raise RuntimeError("no backend")

    monkeypatch.setattr(llm_mod, "generate_text", boom)
    from smart_assignment.shared.config import Config

    assert note_tagging.llm_suggest_tags("confusing", Config()) == set()


def test_tagger_routes_note_to_tagged_dimension():
    # End-to-end with the real keyword tagger through the calibration tiering:
    # a recommend 👎 whose note is about the brief routes to brief_quality, not
    # the outcome-default response_clarity.
    tagger = make_note_tagger()
    label = HumanLabel(
        decision_id="d1", outcome="recommend", thumb="down",
        note="the escalation brief was incomplete",
    )
    signals = list(human_dimension_signals(label, note_tagger=tagger))
    assert signals == [(DIM_BRIEF_QUALITY, False, "note")]
