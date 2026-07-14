"""Deterministic groundedness verification."""

from __future__ import annotations

from smart_assignment.judgment.evidence import build_evidence_packet
from smart_assignment.judgment.schema import parse_judgment
from smart_assignment.judgment.verifier import verify
from smart_assignment.shared.config import Config

from .conftest import LOW_SCORE, evaluations_for, feasible_ids


def _packet_and_winner():
    config = Config()
    customer, evals = evaluations_for(LOW_SCORE, config)
    packet = build_evidence_packet(customer, evals, config)
    winner_id = feasible_ids(evals)[0]
    facts = packet.candidate_dict(winner_id)["facts"]
    return packet, winner_id, facts


def test_grounded_fact_citation_passes():
    packet, winner_id, facts = _packet_and_winner()
    out = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": winner_id,
            "rationale": "Acceptable headroom for this order.",
            "citations": [
                {
                    "route_id": winner_id,
                    "field": "utilization_after",
                    "value": facts["utilization_after"],
                },
            ],
        }
    )
    assert verify(out, packet).ok


def test_fabricated_fact_value_fails():
    packet, winner_id, facts = _packet_and_winner()
    bogus = float(facts["utilization_after"]) + 0.25
    out = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": winner_id,
            "rationale": "Looks fine.",
            "citations": [
                {"route_id": winner_id, "field": "utilization_after", "value": round(bogus, 4)},
            ],
        }
    )
    result = verify(out, packet)
    assert not result.ok
    assert any("utilization_after" in r for r in result.reasons)


def test_pick_outside_feasible_set_fails():
    packet, _winner_id, _facts = _packet_and_winner()
    out = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": "RTE-DOES-NOT-EXIST",
            "rationale": "Invented route.",
            "citations": [],
        }
    )
    result = verify(out, packet)
    assert not result.ok
    assert any("feasible set" in r for r in result.reasons)


def test_recommend_with_no_pick_fails():
    packet, _winner_id, _facts = _packet_and_winner()
    out = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": None,
            "rationale": "No pick.",
            "citations": [],
        }
    )
    result = verify(out, packet)
    assert not result.ok
    assert any("no recommended_route_id" in r for r in result.reasons)


def test_percent_paraphrase_in_prose_is_grounded():
    packet, winner_id, facts = _packet_and_winner()
    pct = round(float(facts["utilization_after"]) * 100)
    out = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": winner_id,
            "rationale": f"After this order the truck sits at roughly {pct}% utilization.",
            "citations": [],
        }
    )
    assert verify(out, packet).ok


def test_hallucinated_number_in_prose_fails():
    packet, winner_id, _facts = _packet_and_winner()
    out = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": winner_id,
            "rationale": "This route leaves 9999 cases of headroom, plenty of room.",
            "citations": [],
        }
    )
    result = verify(out, packet)
    assert not result.ok
    assert any("9999" in r for r in result.reasons)


def test_route_id_in_prose_does_not_trip_the_number_scan():
    # The digits inside a real route-id must not be read as an ungrounded number.
    packet, winner_id, _facts = _packet_and_winner()
    out = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": winner_id,
            "rationale": f"Route {winner_id} is the only feasible option and clusters well.",
            "citations": [],
        }
    )
    assert verify(out, packet).ok


def test_true_and_false_comparisons():
    config = Config()
    customer, evals = evaluations_for(LOW_SCORE, config)
    packet = build_evidence_packet(customer, evals, config)
    ids = [c["route_id"] for c in packet.feasible_candidates + packet.infeasible_candidates]
    # Need two routes to compare distance against.
    if len(ids) < 2:
        return
    a, b = ids[0], ids[1]
    fa = packet.candidate_dict(a)["facts"]["distance_miles"]
    fb = packet.candidate_dict(b)["facts"]["distance_miles"]
    truth = "greater" if fa > fb else "less"
    lie = "less" if fa > fb else "greater"
    winner_id = feasible_ids(evals)[0]

    good = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": winner_id,
            "rationale": "ok",
            "citations": [
                {
                    "kind": "comparison",
                    "field": "distance_miles",
                    "route_id_a": a,
                    "route_id_b": b,
                    "relation": truth,
                }
            ],
        }
    )
    assert verify(good, packet).ok

    bad = parse_judgment(
        {
            "decision": "RECOMMEND",
            "confidence": "HIGH",
            "recommended_route_id": winner_id,
            "rationale": "ok",
            "citations": [
                {
                    "kind": "comparison",
                    "field": "distance_miles",
                    "route_id_a": a,
                    "route_id_b": b,
                    "relation": lie,
                }
            ],
        }
    )
    assert not verify(bad, packet).ok
