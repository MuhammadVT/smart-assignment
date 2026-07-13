"""The route-slot decision: deterministic best, escalation gate, grounded pick,
and fallback. Offline -- the LLM is an injected fake."""

from __future__ import annotations

from smart_assignment.routeslot import decide_route_slot
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision

from .conftest import AFTERNOON, MORNING, customer, scored_eval, scored_slot


def _evals():
    a = scored_eval("RTE-A", "Alpha", [
        scored_slot(MORNING, avail=0.33, total=0.80),     # deterministic best
        scored_slot(AFTERNOON, avail=0.91, total=0.60),
    ])
    b = scored_eval("RTE-B", "Bravo", [scored_slot(MORNING, avail=0.90, total=0.66)])
    return [a, b]


def test_deterministic_picks_the_highest_total_route_slot():
    rec = decide_route_slot(customer(), _evals(), Config(use_route_slot_scoring=True))
    assert rec.decision is Decision.RECOMMENDED
    assert rec.recommended_route_id == "RTE-A"
    assert rec.total_score == 0.80


def test_escalates_when_best_route_slot_is_below_threshold():
    evals = [scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.3, total=0.40)])]
    cfg = Config(use_route_slot_scoring=True, route_slot_score_threshold=0.55)
    rec = decide_route_slot(customer(), evals, cfg)
    assert rec.decision is Decision.ESCALATED_LOW_SCORE


def test_no_feasible_route_escalates():
    infeasible = scored_eval("RTE-X", "X", [], feasible=False)
    rec = decide_route_slot(customer(), [infeasible], Config(use_route_slot_scoring=True))
    assert rec.decision is Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None


def test_grounded_pick_diverges_to_a_more_open_slot():
    cfg = Config(use_route_slot_scoring=True, use_grounded_judgment=True)

    # Options are sorted by descending total: idx0=RTE-A morning (0.80),
    # idx1=RTE-B morning (0.66, openness 0.90), idx2=RTE-A afternoon (0.60).
    def stub(config, prompt):
        return {
            "chosen_index": 1,
            "rationale": "RTE-B's slot is far more open (0.90), protecting valued incumbents.",
            "citations": [{"index": 1, "field": "slot_availability", "value": 0.90}],
        }

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=stub)
    assert rec.recommended_route_id == "RTE-B"
    assert "more open" in rec.recommended_window_rationale
    assert rec.decision is Decision.RECOMMENDED       # 0.66 >= 0.55


def test_grounded_falls_back_to_deterministic_on_backend_error():
    cfg = Config(use_route_slot_scoring=True, use_grounded_judgment=True)

    def boom(config, prompt):
        raise RuntimeError("SAGE_CLIENT_ID missing")

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=boom)
    assert rec.recommended_route_id == "RTE-A"        # deterministic best
    assert rec.grounded_fallback is True
    assert rec.recommended_window_rationale is None


def test_grounded_falls_back_on_persistently_ungrounded_choice():
    cfg = Config(use_route_slot_scoring=True, use_grounded_judgment=True)

    def liar(config, prompt):
        return {"chosen_index": 1, "rationale": "x",
                "citations": [{"index": 1, "field": "slot_availability", "value": 0.99}]}

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=liar)
    assert rec.recommended_route_id == "RTE-A"        # fell back
    assert rec.grounded_fallback is True


def test_llm_menu_excludes_below_threshold_route_slots():
    # One route with an above-bar (0.80) and a below-bar (0.50) slot; only the
    # above-bar one should reach the LLM.
    evals = [scored_eval("RTE-A", "Alpha", [
        scored_slot(MORNING, avail=0.7, total=0.80),
        scored_slot(AFTERNOON, avail=0.3, total=0.50),
    ])]
    cfg = Config(use_route_slot_scoring=True, use_grounded_judgment=True,
                 route_slot_score_threshold=0.55)
    seen = {}

    def capture(config, prompt):
        seen["prompt"] = prompt
        return {"chosen_index": 0, "rationale": "the open morning slot",
                "citations": [{"index": 0, "field": "reference_weighted_score", "value": 0.80}]}

    rec = decide_route_slot(customer(), evals, cfg, choice_fn=capture)
    assert rec.decision is Decision.RECOMMENDED
    assert rec.recommended_window == "08:30-11:30"        # the 0.80 morning slot
    # The below-bar slot's score never appears in the menu the LLM saw.
    assert '"reference_weighted_score": 0.8' in seen["prompt"]
    assert '"reference_weighted_score": 0.5' not in seen["prompt"]


def test_low_score_escalation_never_calls_the_llm():
    evals = [scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.3, total=0.40)])]
    cfg = Config(use_route_slot_scoring=True, use_grounded_judgment=True,
                 route_slot_score_threshold=0.55)

    def boom(config, prompt):
        raise AssertionError("LLM must not be consulted when nothing clears the bar")

    rec = decide_route_slot(customer(), evals, cfg, choice_fn=boom)
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert "auto-assign bar" in rec.review_reason


def test_feasible_route_with_no_slots_has_its_own_reason():
    # Feasible on hard constraints, but zero candidate slots could be built.
    empty = scored_eval("RTE-A", "Alpha", [], feasible=True)
    rec = decide_route_slot(customer(), [empty], Config(use_route_slot_scoring=True))
    assert rec.decision is Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None
    # Distinct from the no-feasible-route reason.
    assert "no delivery window" in rec.review_reason.lower()
    assert "hard constraint" not in rec.review_reason.lower()
