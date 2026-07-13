"""Route-slot evidence packet, choice schema, and deterministic verifier."""

from __future__ import annotations

import pytest

from smart_assignment.shared.config import Config
from smart_assignment.routeslot.evidence import NUMERIC_FACT_KEYS, build_route_slot_packet
from smart_assignment.routeslot.schema import (
    RouteSlotChoiceParseError,
    parse_route_slot_choice,
)
from smart_assignment.routeslot.verifier import verify_choice

from .conftest import AFTERNOON, MORNING, customer, scored_eval, scored_slot


def _evals():
    # RTE-A: a strong morning (0.80) and a weaker afternoon (0.60).
    # RTE-B: one open-but-lower option (0.66).
    a = scored_eval("RTE-A", "Alpha", [
        scored_slot(MORNING, avail=0.33, total=0.80),
        scored_slot(AFTERNOON, avail=0.91, total=0.60),
    ])
    b = scored_eval("RTE-B", "Bravo", [scored_slot(MORNING, avail=0.90, total=0.66)])
    return [a, b]


def test_packet_enumerates_all_feasible_route_slots_sorted_by_total():
    packet = build_route_slot_packet(customer(), _evals(), Config())
    assert packet.n == 3
    totals = [o["facts"]["reference_weighted_score"] for o in packet.options]
    assert totals == sorted(totals, reverse=True)      # descending
    assert packet.deterministic_best_index == 0        # the 0.80 morning on RTE-A
    for o in packet.options:
        for k in NUMERIC_FACT_KEYS:
            if k == "window_match":
                continue  # only present when a preference was scored in
            assert k in o["facts"]


def test_parse_valid_and_malformed_choices():
    ok = parse_route_slot_choice({
        "chosen_index": 2,
        "rationale": "Most open slot.",
        "citations": [{"index": 2, "field": "slot_availability", "value": 0.9}],
    })
    assert ok.chosen_index == 2 and ok.citations[0].field == "slot_availability"
    for bad in ({"rationale": "x"}, {"chosen_index": "two", "rationale": "x"},
                {"chosen_index": 0, "rationale": ""}):
        with pytest.raises(RouteSlotChoiceParseError):
            parse_route_slot_choice(bad)


def _packet():
    from smart_assignment.shared.config import Config
    return build_route_slot_packet(customer(), _evals(), Config())


def test_verifier_accepts_grounded_choice():
    packet = _packet()
    val = packet.options[0]["facts"]["reference_weighted_score"]
    choice = parse_route_slot_choice({
        "chosen_index": 0, "rationale": "best",
        "citations": [{"index": 0, "field": "reference_weighted_score", "value": val}],
    })
    assert verify_choice(choice, packet).ok


def test_verifier_rejects_out_of_range_and_fabricated_and_unknown_field():
    packet = _packet()
    assert not verify_choice(
        parse_route_slot_choice({"chosen_index": 9, "rationale": "x", "citations": []}), packet
    ).ok
    fabricated = parse_route_slot_choice({
        "chosen_index": 0, "rationale": "x",
        "citations": [{"index": 0, "field": "slot_availability", "value": 0.99}],
    })
    assert not verify_choice(fabricated, packet).ok
    unknown = parse_route_slot_choice({
        "chosen_index": 0, "rationale": "x",
        "citations": [{"index": 0, "field": "made_up", "value": 1}],
    })
    assert not verify_choice(unknown, packet).ok
