"""
GroundedSlotSelector orchestration + refine_slot wiring. Offline: every LLM
interaction is a fake choice_fn.
"""

from __future__ import annotations

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision, SlotRecommendation
from smart_assignment.slotpick import (
    DeterministicSlotSelector,
    GroundedSlotSelector,
    default_slot_selector,
    refine_slot,
)

from .conftest import AFTERNOON, MORNING, customer, evaluation


def _pick_index(idx, field="fit_score"):
    """A fake choice_fn that picks candidate `idx` with a grounded citation."""

    def fn(config, prompt):
        # We cite the known fact value from conftest's slots rather than parsing
        # it back out of the prompt.
        vals = {0: {"fit_score": 0.7, "committed_overlap": 2},
                1: {"fit_score": 0.3, "committed_overlap": 1}}
        return {
            "chosen_index": idx,
            "rationale": f"Candidate {idx} chosen.",
            "citations": [{"index": idx, "field": field, "value": vals[idx][field]}],
        }

    return fn


def test_deterministic_selector_returns_the_blended_pick():
    ev = evaluation([MORNING, AFTERNOON], chosen_index=0)
    pick = DeterministicSlotSelector().select(customer(), ev, Config())
    assert pick.window == MORNING.window
    assert pick.rationale is None


def test_grounded_selector_picks_the_models_candidate():
    ev = evaluation([MORNING, AFTERNOON], chosen_index=0)  # deterministic = morning
    sel = GroundedSlotSelector(choice_fn=_pick_index(1))   # model prefers afternoon
    pick = sel.select(customer(), ev, Config())
    assert pick.window == AFTERNOON.window
    assert "Candidate 1" in pick.rationale


def test_grounded_selector_falls_back_on_persistently_invalid_index():
    ev = evaluation([MORNING, AFTERNOON], chosen_index=0)

    def bad(config, prompt):
        return {"chosen_index": 9, "rationale": "invalid", "citations": []}

    pick = GroundedSlotSelector(choice_fn=bad).select(customer(), ev, Config())
    assert pick.window == MORNING.window       # fell back to the deterministic slot
    assert pick.rationale is None


def test_grounded_selector_falls_back_on_backend_error():
    ev = evaluation([MORNING, AFTERNOON], chosen_index=0)

    def boom(config, prompt):
        raise RuntimeError("SAGE_CLIENT_ID missing")

    pick = GroundedSlotSelector(choice_fn=boom).select(customer(), ev, Config())
    assert pick.window == MORNING.window
    assert pick.rationale is None


def test_default_slot_selector_gating():
    assert default_slot_selector(Config(use_grounded_slot_selection=False)) is None
    assert isinstance(default_slot_selector(Config(use_grounded_slot_selection=True)),
                      GroundedSlotSelector)


# --- refine_slot wiring ------------------------------------------------------


def _recommendation(route_id="RTE-4100"):
    return SlotRecommendation(
        customer_name="Bayou City Bistro",
        decision=Decision.RECOMMENDED,
        total_score=0.9,
        reasoning="ok",
        recommended_route_id=route_id,
        recommended_route_name="Central Houston",
        recommended_day="TUE",
        recommended_window="07:20-10:20",
        recommended_window_basis="between_adjacent_stops",
    )


def test_refine_slot_updates_the_winner_window_and_rationale():
    ev = evaluation([MORNING, AFTERNOON], chosen_index=0)
    rec = _recommendation()
    refine_slot(rec, [ev], customer(), Config(),
                selector=GroundedSlotSelector(choice_fn=_pick_index(1)))
    assert rec.recommended_window == "13:00-16:00"      # the model's afternoon pick
    assert rec.recommended_window_rationale is not None


def test_refine_slot_is_a_noop_when_selection_disabled():
    ev = evaluation([MORNING, AFTERNOON], chosen_index=0)
    rec = _recommendation()
    before = rec.recommended_window
    refine_slot(rec, [ev], customer(), Config(use_grounded_slot_selection=False))
    assert rec.recommended_window == before
    assert rec.recommended_window_rationale is None


def test_refine_slot_is_a_noop_without_a_recommended_route():
    rec = _recommendation(route_id=None)  # e.g. no-feasible escalation
    refine_slot(rec, [], customer(), Config(),
                selector=GroundedSlotSelector(choice_fn=_pick_index(0)))
    assert rec.recommended_window_rationale is None
