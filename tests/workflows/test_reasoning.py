"""
Unit tests for the natural-language reasoning layer
(workflows/slot_recommendation/reasoning.py). The DeterministicReasoner is
meant to read like a colleague explaining their own decision -- verbose,
first-person, no underscored identifiers leaking into the prose.
"""

from __future__ import annotations

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.config import Config
from smart_assignment.workflows.slot_recommendation.pipeline import run_slot_recommendation
from smart_assignment.workflows.slot_recommendation.reasoning import DeterministicReasoner


def _run_all():
    reasoner = DeterministicReasoner()
    config = Config()
    return [run_slot_recommendation(c, config=config, reasoner=reasoner) for c in SAMPLE_CUSTOMERS]


def test_reasoning_has_no_underscored_words():
    for result in _run_all():
        reasoning = result.recommendation.reasoning
        tokens = reasoning.replace(",", " ").replace(".", " ").replace(";", " ").split()
        assert not any("_" in t for t in tokens), reasoning
        for alt in result.recommendation.rejected_alternatives:
            assert "_" not in alt, alt


def test_reasoning_is_verbose_and_first_person():
    for result in _run_all():
        reasoning = result.recommendation.reasoning
        # Reads like a short explanatory paragraph, not a terse log line.
        assert len(reasoning) > 200
        assert reasoning.count(".") >= 3
        assert "I " in reasoning or "I'" in reasoning


def test_all_three_narrative_branches_are_covered_by_the_mock_set():
    reasonings = [r.recommendation.reasoning for r in _run_all()]
    joined = " ".join(reasonings)
    # Confident, auto-assigned recommendation.
    assert "comfortable moving ahead without a specialist review" in joined
    # Low-total-score escalation (a proposed slot, but flagged for review).
    assert "I'd like a specialist to take a quick look" in joined
    # No feasible slot at all (nothing to propose).
    assert "escalating this to a routing specialist" in joined
