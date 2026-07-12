"""Deterministic verification of a slot choice against its packet."""

from __future__ import annotations

from smart_assignment.shared.config import Config
from smart_assignment.slotpick.evidence import build_slot_packet
from smart_assignment.slotpick.schema import parse_slot_choice
from smart_assignment.slotpick.verifier import verify_choice

from .conftest import AFTERNOON, MORNING, customer, evaluation


def _packet():
    return build_slot_packet(customer(), evaluation([MORNING, AFTERNOON]), Config())


def test_valid_index_and_grounded_citation_passes():
    packet = _packet()
    out = parse_slot_choice(
        {
            "chosen_index": 0,
            "rationale": "Tight fit.",
            "citations": [{"index": 0, "field": "fit_score", "value": 0.7}],
        }
    )
    assert verify_choice(out, packet).ok


def test_index_out_of_range_fails():
    packet = _packet()
    out = parse_slot_choice({"chosen_index": 5, "rationale": "Invented slot.", "citations": []})
    result = verify_choice(out, packet)
    assert not result.ok
    assert any("valid candidate" in r for r in result.reasons)


def test_fabricated_citation_value_fails():
    packet = _packet()
    out = parse_slot_choice(
        {
            "chosen_index": 0,
            "rationale": "Looks empty.",
            "citations": [{"index": 0, "field": "committed_overlap", "value": 99}],
        }
    )
    result = verify_choice(out, packet)
    assert not result.ok
    assert any("committed_overlap" in r for r in result.reasons)


def test_citation_to_unknown_field_fails():
    packet = _packet()
    out = parse_slot_choice(
        {
            "chosen_index": 0,
            "rationale": "x",
            "citations": [{"index": 0, "field": "made_up_field", "value": 1}],
        }
    )
    assert not verify_choice(out, packet).ok
