"""The route-slot decision: deterministic best, escalation gate, grounded pick,
and fallback. Offline -- the LLM is an injected fake."""

from __future__ import annotations

from smart_assignment.routeslot import decide_route_slot
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision

from .conftest import AFTERNOON, MORNING, choice_dict, customer, scored_eval, scored_slot


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
    # The flat reasoning line is unchanged (compat / page fallback)...
    assert "strongest route-slot overall" in rec.reasoning
    # ...but the deterministic structured floor is always populated, so a user
    # (or the agent narration) gets the reasons + trade-off, not a one-liner.
    # Routes are named as "<route id> - <route name>" -- both id and name together.
    assert rec.decision_summary and "RTE-A - Alpha" in rec.decision_summary
    assert rec.recommended_route_name == "Alpha"
    assert "strongest route-slot overall" in rec.reasoning and "RTE-A - Alpha" in rec.reasoning
    assert len(rec.primary_reasons) == 2
    assert rec.runner_up and rec.key_tradeoff
    assert rec.default_comparison is None  # only the grounded self-assessment sets this


def test_deterministic_floor_tradeoff_names_the_runner_up_advantage():
    # RTE-A morning wins overall (0.80) but its slot is tight (avail 0.33); the
    # runner-up RTE-B is more open (0.90) -> the trade-off should call that out.
    rec = decide_route_slot(customer(), _evals(), Config(use_route_slot_scoring=True))
    assert "0.80 vs" in rec.key_tradeoff        # the winner's score edge
    assert "slot openness" in rec.key_tradeoff  # the factor the runner-up leads on
    assert "RTE-B - Bravo" in rec.runner_up     # runner-up named as <id> - <name>


def test_escalates_when_best_route_slot_is_below_threshold():
    evals = [scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.3, total=0.40)])]
    cfg = Config(use_route_slot_scoring=True, route_slot_score_threshold=0.55)
    rec = decide_route_slot(customer(), evals, cfg)
    assert rec.decision is Decision.ESCALATED_LOW_SCORE


def test_no_feasible_route_escalates():
    infeasible = scored_eval("RTE-X", "Xavier", [], feasible=False)
    rec = decide_route_slot(customer(), [infeasible], Config(use_route_slot_scoring=True))
    assert rec.decision is Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None
    # Infeasible routes are named as "<route id> - <route name>" too.
    assert any("RTE-X - Xavier" in line for line in rec.rejected_alternatives)


def test_grounded_pick_diverges_to_a_more_open_slot():
    cfg = Config(use_route_slot_scoring=True, use_grounded_judgment=True)

    # Options are sorted by descending total: idx0=RTE-A morning (0.80),
    # idx1=RTE-B morning (0.66, openness 0.90), idx2=RTE-A afternoon (0.60).
    def stub(config, prompt):
        return choice_dict(
            1,  # diverges from the deterministic default (idx0)
            runner_up_index=0,
            primary_reasons=["RTE-B's slot is far more open (0.90), protecting incumbents."],
            citations=[{"index": 1, "field": "slot_availability", "value": 0.90}],
        )

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=stub)
    assert rec.recommended_route_id == "RTE-B"
    assert rec.decision is Decision.RECOMMENDED       # 0.66 >= 0.55
    # The structured explanation is surfaced, and folded into reasoning/rationale.
    assert rec.decision_summary and rec.key_tradeoff
    assert rec.primary_reasons and "more open" in rec.primary_reasons[0]
    assert "more open" in rec.recommended_window_rationale
    assert "Diverged" in rec.default_comparison
    assert rec.runner_up and "Alpha" in rec.runner_up  # runner-up rendered with its route name


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
        # Well-formed shape, but a fabricated citation (idx1 openness is 0.90).
        return choice_dict(
            1, runner_up_index=0,
            citations=[{"index": 1, "field": "slot_availability", "value": 0.99}],
        )

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=liar)
    assert rec.recommended_route_id == "RTE-A"        # fell back
    assert rec.grounded_fallback is True
    # On fallback the deterministic structured floor still stands (never a
    # one-liner), but the grounded-only self-assessment is absent.
    assert rec.decision_summary and rec.primary_reasons and rec.key_tradeoff
    assert rec.default_comparison is None
    assert "Trade-off:" not in rec.reasoning          # reasoning stays the flat line


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
        # Only one option clears the bar -> a single-option menu, so no runner_up.
        return choice_dict(
            0, runner_up=None, key_tradeoff="",
            citations=[{"index": 0, "field": "reference_weighted_score", "value": 0.80}],
        )

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
