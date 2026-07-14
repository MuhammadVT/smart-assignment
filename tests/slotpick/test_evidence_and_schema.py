"""Slot evidence packet + slot-choice schema parsing."""

from __future__ import annotations

from datetime import time

import pytest

from smart_assignment.shared.config import Config
from smart_assignment.slotpick.evidence import NUMERIC_SLOT_FIELDS, build_slot_packet
from smart_assignment.slotpick.schema import SlotChoiceParseError, parse_slot_choice

from .conftest import AFTERNOON, MORNING, customer, evaluation


def test_packet_enumerates_candidates_with_facts_and_preference_overlap():
    cust = customer(pref=(time(13, 0), time(15, 30)))  # afternoon preference
    packet = build_slot_packet(cust, evaluation([MORNING, AFTERNOON]), Config())

    assert packet.n == 2
    assert packet.preferred_window_minutes == 150
    for i, cand in enumerate(packet.candidates):
        assert cand["index"] == i
        for k in NUMERIC_SLOT_FIELDS:
            assert k in cand["facts"]
    # The afternoon candidate overlaps the afternoon preference; the morning one
    # does not.
    assert packet.candidates[0]["facts"]["preference_overlap_minutes"] == 0
    assert packet.candidates[1]["facts"]["preference_overlap_minutes"] > 0


def test_no_preference_means_zero_overlap_everywhere():
    packet = build_slot_packet(customer(pref=None), evaluation([MORNING, AFTERNOON]), Config())
    assert packet.preferred_window_minutes is None
    assert all(c["facts"]["preference_overlap_minutes"] == 0 for c in packet.candidates)


def test_packet_surfaces_the_deterministic_blend_as_reference():
    # The blend's own score is a fact on every candidate, and the index it would
    # pick by itself (chosen_index=0 -> MORNING) is named -- reference, not gospel.
    packet = build_slot_packet(customer(), evaluation([MORNING, AFTERNOON], chosen_index=0),
                               Config())
    assert packet.deterministic_choice_index == 0
    assert all("blended_score" in c["facts"] for c in packet.candidates)
    # blended_score is citable/verifiable like any other fact.
    assert "blended_score" in NUMERIC_SLOT_FIELDS


def test_parse_valid_slot_choice():
    out = parse_slot_choice(
        {
            "chosen_index": 1,
            "rationale": "Best preference overlap.",
            "citations": [{"index": 1, "field": "preference_overlap_minutes", "value": 150}],
        }
    )
    assert out.chosen_index == 1
    assert len(out.citations) == 1 and out.citations[0].field == "preference_overlap_minutes"


@pytest.mark.parametrize(
    "raw",
    [
        {"rationale": "x"},  # missing chosen_index
        {"chosen_index": "two", "rationale": "x"},  # non-integer index
        {"chosen_index": 0, "rationale": ""},  # empty rationale
    ],
)
def test_parse_rejects_malformed(raw):
    with pytest.raises(SlotChoiceParseError):
        parse_slot_choice(raw)
