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

from .conftest import AFTERNOON, MORNING, choice_dict, customer, scored_eval, scored_slot


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


def test_min_score_filters_the_menu_to_above_threshold_options():
    # Totals present: 0.80, 0.66, 0.60. A 0.65 floor keeps only the first two.
    full = build_route_slot_packet(customer(), _evals(), Config())
    assert full.n == 3
    filtered = build_route_slot_packet(customer(), _evals(), Config(), min_score=0.65)
    assert filtered.n == 2
    assert all(o["facts"]["reference_weighted_score"] >= 0.65 for o in filtered.options)


def test_parse_valid_and_malformed_choices():
    ok = parse_route_slot_choice(choice_dict(
        2,
        citations=[{"index": 2, "field": "slot_availability", "value": 0.9}],
    ))
    assert ok.chosen_index == 2 and ok.citations[0].field == "slot_availability"
    assert ok.decision_summary and ok.primary_reasons
    assert ok.runner_up is not None and ok.vs_deterministic_default.verdict == "DIVERGE"
    for bad in (
        {"decision_summary": "x", "primary_reasons": ["y"]},       # no chosen_index
        {"chosen_index": "two", "decision_summary": "x", "primary_reasons": ["y"]},
        {"chosen_index": 0, "decision_summary": "", "primary_reasons": ["y"]},  # empty summary
        {"chosen_index": 0, "decision_summary": "x", "primary_reasons": []},    # empty reasons
        {"chosen_index": 0, "decision_summary": "x", "primary_reasons": ["y"],  # bad verdict
         "vs_deterministic_default": {"verdict": "MAYBE"}},
    ):
        with pytest.raises(RouteSlotChoiceParseError):
            parse_route_slot_choice(bad)


def _packet():
    from smart_assignment.shared.config import Config
    return build_route_slot_packet(customer(), _evals(), Config())


def test_verifier_accepts_grounded_choice():
    packet = _packet()
    val = packet.options[0]["facts"]["reference_weighted_score"]
    choice = parse_route_slot_choice(choice_dict(
        0,  # == deterministic default -> AGREE
        runner_up_index=1,
        citations=[{"index": 0, "field": "reference_weighted_score", "value": val}],
    ))
    assert verify_choice(choice, packet).ok


def test_verifier_rejects_out_of_range_and_fabricated_and_unknown_field():
    packet = _packet()
    assert not verify_choice(
        parse_route_slot_choice(choice_dict(9)), packet
    ).ok
    fabricated = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        citations=[{"index": 0, "field": "slot_availability", "value": 0.99}],
    ))
    assert not verify_choice(fabricated, packet).ok
    unknown = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        citations=[{"index": 0, "field": "made_up", "value": 1}],
    ))
    assert not verify_choice(unknown, packet).ok


def test_verifier_rejects_dishonest_verdict_and_missing_tradeoff():
    packet = _packet()  # n == 3
    # Picks the non-default option but claims AGREE -> inconsistent.
    dishonest = parse_route_slot_choice(choice_dict(
        1, runner_up_index=0,
        vs_deterministic_default={"verdict": "AGREE", "note": ""},
    ))
    assert not verify_choice(dishonest, packet).ok
    # More than one option, but no runner_up / trade-off named.
    incomplete = parse_route_slot_choice(choice_dict(0, runner_up=None, key_tradeoff=""))
    assert not verify_choice(incomplete, packet).ok


def test_verifier_rejects_ungrounded_number_in_prose():
    packet = _packet()
    liar = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        primary_reasons=["Fits within 42.7 miles of every stop."],  # 42.7 is nowhere in the packet
    ))
    result = verify_choice(liar, packet)
    assert not result.ok
    assert "42.7" in result.as_feedback()


def test_verifier_allows_single_option_without_runner_up():
    # One eligible route-slot: no runner-up is possible, and that's fine.
    evals = [scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.7, total=0.80)])]
    packet = build_route_slot_packet(customer(), evals, Config())
    assert packet.n == 1
    choice = parse_route_slot_choice(choice_dict(0, runner_up=None, key_tradeoff=""))
    assert verify_choice(choice, packet).ok


# ---------------------------------------------------------------------------
# Adversarial-hardening regressions: cases that slipped the original checks.
# ---------------------------------------------------------------------------


def test_percent_form_citation_passes_but_magnitude_laundering_fails():
    packet = _packet()
    avail = packet.options[0]["facts"]["slot_availability"]  # 0.33
    pct = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        citations=[{"index": 0, "field": "slot_availability", "value": round(avail * 100, 2)}],
    ))
    assert verify_choice(pct, packet).ok  # "33%" for a stored 0.33
    laundered = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        citations=[{"index": 0, "field": "slot_availability", "value": round(avail / 100, 4)}],
    ))
    assert not verify_choice(laundered, packet).ok  # 100x down must not pass


def test_near_tolerance_but_different_citation_value_fails():
    # 0.35 is not a faithful paraphrase of a stored 0.33 -- the old 0.02
    # tolerance accepted it.
    packet = _packet()
    off = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        citations=[{"index": 0, "field": "slot_availability", "value": 0.35}],
    ))
    assert not verify_choice(off, packet).ok


def test_unit_bearing_figure_cannot_launder_through_percent_normalization():
    # A stored 0.33 availability must not ground "33 miles"; percent form stays fine.
    packet = _packet()
    bad = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        primary_reasons=["The stop sits 33 miles from the cluster core."],
    ))
    result = verify_choice(bad, packet)
    assert not result.ok and "33" in result.as_feedback()
    good = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        primary_reasons=["Scores 80% overall; the morning slot is only 33% open."],
    ))
    assert verify_choice(good, packet).ok


def test_comma_grouped_and_small_unit_figures_are_checked():
    packet = _packet()
    for reason in ("Leaves 8,000 cases of headroom.", "Only 7 cases of headroom remain."):
        bad = parse_route_slot_choice(choice_dict(0, runner_up_index=1,
                                                  primary_reasons=[reason]))
        assert not verify_choice(bad, packet).ok, reason
    fine = parse_route_slot_choice(
        choice_dict(0, runner_up_index=1, primary_reasons=["Better than the other 2 options."])
    )
    assert verify_choice(fine, packet).ok


def test_wrong_day_and_invented_window_are_flagged():
    packet = _packet()
    wrong_day = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1, primary_reasons=["The Friday run has the most open slot."],
    ))
    result = verify_choice(wrong_day, packet)
    assert not result.ok and "Friday" in result.as_feedback()
    real_day = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1, primary_reasons=["The Tuesday run has the most open slot."],
    ))
    assert verify_choice(real_day, packet).ok

    fake_window = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1, primary_reasons=["Deliver in the 13:00-15:00 window."],
    ))
    result = verify_choice(fake_window, packet)
    assert not result.ok and "13:00" in result.as_feedback()
    real_window = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1,
        primary_reasons=[f"The {packet.options[0]['window']} window clusters well."],
    ))
    assert verify_choice(real_window, packet).ok


def test_hallucinated_route_mentions_are_flagged():
    packet = _packet()
    hyphenated = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1, primary_reasons=["RTE-Z-99 could also absorb it."],
    ))
    assert not verify_choice(hyphenated, packet).ok
    # "Route 90" names no candidate; 90 equalling the order size must not save it.
    bare = parse_route_slot_choice(choice_dict(
        0, runner_up_index=1, primary_reasons=["Route 90 can absorb this order."],
    ))
    result = verify_choice(bare, packet)
    assert not result.ok and "'90'" in result.as_feedback()
