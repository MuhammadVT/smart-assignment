"""Evidence-packet construction and judgment-schema parsing."""

from __future__ import annotations

import pytest

from smart_assignment.judgment.evidence import NUMERIC_FACT_KEYS, build_evidence_packet
from smart_assignment.judgment.schema import (
    ComparisonCitation,
    Confidence,
    FactCitation,
    JudgmentDecision,
    JudgmentParseError,
    parse_judgment,
)
from smart_assignment.shared.config import Config

from .conftest import CLEAR_RECOMMEND, LOW_SCORE, NO_FEASIBLE, evaluations_for, feasible_ids


def test_packet_splits_feasible_and_infeasible_and_exposes_raw_facts():
    config = Config()
    customer, evals = evaluations_for(LOW_SCORE, config)
    packet = build_evidence_packet(customer, evals, config)

    assert packet.feasible_candidates, "Galleria should have at least one feasible route"
    assert packet.infeasible_candidates, "Galleria should have infeasible routes too"

    for cand in packet.feasible_candidates:
        facts = cand["facts"]
        for key in NUMERIC_FACT_KEYS:
            assert key in facts
        # The legacy weighted score is present but explicitly labelled reference-only.
        assert facts["reference_weighted_score"] is not None
        assert "context only" in cand["reference_only_note"].lower()

    for cand in packet.infeasible_candidates:
        assert cand["failed_constraints"], "an infeasible candidate must say why it failed"


def test_packet_facts_match_the_evaluation_numbers():
    config = Config()
    customer, evals = evaluations_for(LOW_SCORE, config)
    packet = build_evidence_packet(customer, evals, config)
    winner_id = feasible_ids(evals)[0]
    ev = packet.evaluation_for(winner_id)
    facts = packet.candidate_dict(winner_id)["facts"]
    assert facts["utilization_after"] == round(ev.utilization_after, 4)
    assert facts["order_quantity_cases"] == customer.order_quantity_cases


def test_no_feasible_packet_has_empty_feasible_list():
    config = Config()
    customer, evals = evaluations_for(NO_FEASIBLE, config)
    packet = build_evidence_packet(customer, evals, config)
    assert packet.feasible_candidates == []
    assert packet.feasible_route_ids == []


def test_parse_valid_judgment_with_both_citation_kinds():
    raw = {
        "decision": "recommend",
        "confidence": "high",
        "recommended_route_id": "RTE-4200",
        "rationale": "Best clustering with acceptable headroom.",
        "candidate_notes": [{"route_id": "RTE-4200", "note": "tight fit"}],
        "citations": [
            {"kind": "fact", "route_id": "RTE-4200", "field": "utilization_after", "value": "87%"},
            {
                "kind": "comparison",
                "field": "remaining_capacity_after",
                "route_id_a": "RTE-4200",
                "route_id_b": "RTE-4400",
                "relation": "greater",
            },
        ],
    }
    out = parse_judgment(raw)
    assert out.decision is JudgmentDecision.RECOMMEND
    assert out.confidence is Confidence.HIGH
    assert out.is_confident_recommend
    assert len(out.fact_citations) == 1 and isinstance(out.fact_citations[0], FactCitation)
    # "87%" was parsed to 87.0 (verifier normalizes percent-vs-fraction later).
    assert out.fact_citations[0].value == 87.0
    assert len(out.comparison_citations) == 1
    assert isinstance(out.comparison_citations[0], ComparisonCitation)


@pytest.mark.parametrize(
    "raw",
    [
        {"confidence": "HIGH", "rationale": "x"},  # missing decision
        {"decision": "MAYBE", "confidence": "HIGH", "rationale": "x"},  # bad decision
        {"decision": "RECOMMEND", "confidence": "SURE", "rationale": "x"},  # bad confidence
        {"decision": "RECOMMEND", "confidence": "HIGH", "rationale": ""},  # empty rationale
    ],
)
def test_parse_rejects_malformed_output(raw):
    with pytest.raises(JudgmentParseError):
        parse_judgment(raw)


def test_clear_recommend_customer_has_a_feasible_winner():
    # Sanity: the "clear" scenario really does produce a feasible route to pick.
    config = Config()
    _, evals = evaluations_for(CLEAR_RECOMMEND, config)
    assert feasible_ids(evals), "Bayou City Bistro should have a feasible route"
