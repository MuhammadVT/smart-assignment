"""The route-slot decision on the THRESHOLD path (the rollback: the 0.55 bar gates
recommend-vs-escalate and the LLM only picks among above-bar options). These pin
``use_grounded_route_slot_escalation=False``; the grounded-escalation path (the
default, where the LLM decides recommend-vs-escalate itself) is covered in
test_grounded_escalation.py. Offline -- the LLM is an injected fake."""

from __future__ import annotations

import json

from smart_assignment.routeslot import decide_route_slot
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision

from .conftest import AFTERNOON, MORNING, choice_dict, customer, scored_eval, scored_slot


def _cfg(**kw):
    """A route-slot config pinned to the THRESHOLD (flag-off) rollback path."""
    return Config(use_grounded_route_slot_escalation=False, **kw)


def _evals():
    a = scored_eval("RTE-A", "Alpha", [
        scored_slot(MORNING, avail=0.33, total=0.80),     # deterministic best
        scored_slot(AFTERNOON, avail=0.91, total=0.60),
    ])
    b = scored_eval("RTE-B", "Bravo", [scored_slot(MORNING, avail=0.90, total=0.66)])
    return [a, b]


def test_total_score_is_the_winners_own_score_untouched_by_the_runner_up():
    # A near-tie between two GOOD options is not penalized -- the winning
    # route-slot's own score stands on its own, regardless of how close the
    # runner-up scored. It is intentionally NOT a margin over the runner-up.
    close = [
        scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.5, total=0.75)]),
        scored_eval("RTE-B", "Bravo", [scored_slot(MORNING, avail=0.5, total=0.74)]),
    ]
    rec = decide_route_slot(customer(), close, _cfg())
    assert rec.decision is Decision.RECOMMENDED
    assert rec.total_score == 0.75

    # A near-tie between two MEDIOCRE options stays mediocre -- still below the
    # bar, so it correctly escalates rather than being rescued by the tie.
    weak = [
        scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.5, total=0.54)]),
        scored_eval("RTE-B", "Bravo", [scored_slot(MORNING, avail=0.5, total=0.53)]),
    ]
    rec = decide_route_slot(customer(), weak, _cfg())
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert rec.total_score == 0.54


def test_deterministic_picks_the_highest_total_route_slot():
    rec = decide_route_slot(customer(), _evals(), _cfg())
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
    # primary_reasons comprehensively covers EVERY scored factor (here geo,
    # capacity, window-match, slot-openness), in the breakdown's canonical order --
    # so slot openness and window match are never dropped.
    assert len(rec.primary_reasons) == len(_evals()[0].scored_slots[0].factor_scores)
    joined = " ".join(rec.primary_reasons)
    assert "Slot openness" in joined and "Preferred-window match" in joined
    assert "Geographic fit" in joined and "Capacity headroom" in joined
    assert rec.runner_up and rec.key_tradeoff
    assert rec.default_comparison is None  # only the grounded self-assessment sets this


def test_deterministic_floor_tradeoff_names_the_runner_up_advantage():
    # RTE-A morning wins overall (0.80) but its slot is tight (avail 0.33); the
    # runner-up RTE-B is more open (0.90) -> the trade-off should call that out.
    rec = decide_route_slot(customer(), _evals(), _cfg())
    assert "0.80 vs" in rec.key_tradeoff        # the winner's score edge
    assert "slot openness" in rec.key_tradeoff  # the factor the runner-up leads on
    assert "RTE-B - Bravo" in rec.runner_up     # runner-up named as <id> - <name>


def test_escalates_when_best_route_slot_is_below_threshold():
    evals = [scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.3, total=0.40)])]
    cfg = _cfg(route_slot_score_threshold=0.55)
    rec = decide_route_slot(customer(), evals, cfg)
    assert rec.decision is Decision.ESCALATED_LOW_SCORE


def test_no_feasible_route_escalates():
    infeasible = scored_eval("RTE-X", "Xavier", [], feasible=False)
    rec = decide_route_slot(customer(), [infeasible], _cfg())
    assert rec.decision is Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None
    # Infeasible routes are named as "<route id> - <route name>" too.
    assert any("RTE-X - Xavier" in line for line in rec.rejected_alternatives)


def test_grounded_pick_diverges_to_a_more_open_slot():
    cfg = _cfg(use_grounded_route_slot_pick=True)

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


def test_grounded_pick_retries_once_on_a_non_json_reply():
    # The observed failure: the model reasons correctly but replies in PROSE, so
    # the reply won't parse. That is recoverable -- it earns one corrective retry
    # (JSON only), not an immediate deterministic fallback.
    cfg = _cfg(use_grounded_route_slot_pick=True)
    calls = {"n": 0, "retry_prompt": ""}

    def prose_then_json(config, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            # Exactly what the sage agent did: prose -> JSONDecodeError at char 0.
            raise json.JSONDecodeError("Expecting value", "Based on the options...", 0)
        calls["retry_prompt"] = prompt
        return choice_dict(
            1,  # diverges to the more-open slot
            runner_up_index=0,
            primary_reasons=["RTE-B's slot is far more open (0.90), protecting incumbents."],
            citations=[{"index": 1, "field": "slot_availability", "value": 0.90}],
        )

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=prose_then_json)
    assert calls["n"] == 2                             # retried once, didn't give up
    assert rec.recommended_route_id == "RTE-B"         # the recovered grounded pick shipped
    assert rec.grounded_fallback is not True
    assert "more open" in rec.recommended_window_rationale
    # The retry told the model exactly what to fix: return JSON only.
    assert "JSON object" in calls["retry_prompt"] and "REJECTED" in calls["retry_prompt"]


def test_grounded_pick_retries_once_on_a_malformed_choice_dict():
    # The other shape failure: valid JSON that doesn't fit the schema (here it's
    # missing chosen_index) -> RouteSlotChoiceParseError -> same one retry.
    cfg = _cfg(use_grounded_route_slot_pick=True)
    calls = {"n": 0}

    def bad_then_good(config, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"primary_reasons": ["no chosen_index here"]}  # fails parse
        return choice_dict(0, runner_up_index=1)

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=bad_then_good)
    assert calls["n"] == 2
    assert rec.recommended_route_id == "RTE-A"
    assert rec.grounded_fallback is not True


def test_grounded_pick_falls_back_when_the_retry_is_also_non_json():
    # One retry, not infinite: if the reply is STILL unparseable, fall back to the
    # deterministic best -- never worse than the baseline.
    cfg = _cfg(use_grounded_route_slot_pick=True)
    calls = {"n": 0}

    def always_prose(config, prompt):
        calls["n"] += 1
        raise json.JSONDecodeError("Expecting value", "still prose", 0)

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=always_prose)
    assert calls["n"] == 2                             # tried exactly twice, then gave up
    assert rec.recommended_route_id == "RTE-A"         # deterministic best
    assert rec.grounded_fallback is True
    assert rec.recommended_window_rationale is None


def test_grounded_falls_back_to_deterministic_on_backend_error():
    cfg = _cfg(use_grounded_route_slot_pick=True)

    def boom(config, prompt):
        raise RuntimeError("SAGE_CLIENT_ID missing")

    rec = decide_route_slot(customer(), _evals(), cfg, choice_fn=boom)
    assert rec.recommended_route_id == "RTE-A"        # deterministic best
    assert rec.grounded_fallback is True
    assert rec.recommended_window_rationale is None


def test_grounded_falls_back_on_persistently_ungrounded_choice():
    cfg = _cfg(use_grounded_route_slot_pick=True)

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
    cfg = _cfg(use_grounded_route_slot_pick=True, route_slot_score_threshold=0.55)
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
    cfg = _cfg(use_grounded_route_slot_pick=True, route_slot_score_threshold=0.55)

    def boom(config, prompt):
        raise AssertionError("LLM must not be consulted when nothing clears the bar")

    rec = decide_route_slot(customer(), evals, cfg, choice_fn=boom)
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert "auto-assign bar" in rec.review_reason


def test_feasible_route_with_no_slots_has_its_own_reason():
    # Feasible on hard constraints, but zero candidate slots could be built.
    empty = scored_eval("RTE-A", "Alpha", [], feasible=True)
    rec = decide_route_slot(customer(), [empty], _cfg())
    assert rec.decision is Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None
    # Distinct from the no-feasible-route reason.
    assert "no delivery window" in rec.review_reason.lower()
    assert "hard constraint" not in rec.review_reason.lower()
