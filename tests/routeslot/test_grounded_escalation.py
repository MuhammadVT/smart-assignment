"""
The grounded route-slot ESCALATION path (Config.use_grounded_route_slot_escalation,
the default): the LLM reasons over ALL feasible route-slots and decides
recommend-vs-escalate itself -- NOT gated by the 0.55 bar -- with k-try consensus,
grounding + verification, and a deterministic threshold fallback. Offline: the LLM
is an injected fake.
"""

from __future__ import annotations

import json

from smart_assignment.routeslot import decide_route_slot
from smart_assignment.routeslot.evidence import build_route_slot_packet
from smart_assignment.routeslot.prompts import build_route_slot_decision_prompt
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision

from .conftest import AFTERNOON, MORNING, choice_dict, customer, scored_eval, scored_slot


def _cfg(**kw):
    base = dict(use_route_slot_scoring=True, use_grounded_route_slot_escalation=True)
    base.update(kw)
    return Config(**base)


def _evals():
    a = scored_eval("RTE-A", "Alpha", [
        scored_slot(MORNING, avail=0.33, total=0.80),     # deterministic best, but tight slot
        scored_slot(AFTERNOON, avail=0.91, total=0.60),
    ])
    b = scored_eval("RTE-B", "Bravo", [scored_slot(MORNING, avail=0.90, total=0.66)])
    return [a, b]


def _recommend(index, **kw):
    kw.setdefault("decision", "RECOMMEND")
    kw.setdefault("confidence", "HIGH")
    return choice_dict(index, **kw)


def _escalate(index=0, **kw):
    kw.setdefault("decision", "ESCALATE")
    kw.setdefault("confidence", "LOW")
    kw.setdefault(
        "citations", [{"index": index, "field": "slot_availability", "value": 0.33}]
    )
    return choice_dict(index, **kw)


# --- packet reference ---------------------------------------------------------


def test_packet_includes_all_feasible_and_the_bar_as_reference():
    packet = build_route_slot_packet(
        customer(), _evals(), _cfg(), auto_assign_threshold=0.55
    )
    # All three feasible route-slots are present (nothing pre-filtered by the bar).
    assert packet.n == 3
    assert packet.as_dict()["auto_assign_threshold"] == 0.55
    # Each option carries the reference "would auto-assign?" flag.
    assert packet.options[0]["meets_auto_assign_bar"] is True   # 0.80
    assert packet.options[2]["meets_auto_assign_bar"] is True   # 0.60
    # Pick-only packet (no threshold) omits both -- unchanged shape.
    plain = build_route_slot_packet(customer(), _evals(), _cfg())
    assert "auto_assign_threshold" not in plain.as_dict()
    assert "meets_auto_assign_bar" not in plain.options[0]


# --- recommend / escalate over feasible options -------------------------------


def test_confident_recommend_ships_on_first_call():
    fn = lambda c, p: _recommend(  # noqa: E731
        0, runner_up_index=1,
        citations=[{"index": 0, "field": "reference_weighted_score", "value": 0.80}],
    )
    rec = decide_route_slot(customer(), _evals(), _cfg(), choice_fn=fn)
    assert rec.decision is Decision.RECOMMENDED
    assert rec.recommended_route_id == "RTE-A"
    assert rec.alternative_takes == []  # single call, no resample


def test_llm_escalates_an_above_bar_option():
    """The whole point: the LLM may escalate even though the best option (0.80)
    clears the 0.55 bar -- the bar does not gate the decision here."""
    calls = {"n": 0}

    def fn(config, prompt):
        calls["n"] += 1
        return _escalate(0, runner_up_index=1)

    rec = decide_route_slot(customer(), _evals(), _cfg(judgment_sample_count=3), choice_fn=fn)
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert rec.total_score == 0.80  # above the bar, still escalated
    assert calls["n"] == 3          # an escalate resamples the full budget
    assert len(rec.alternative_takes) == 3
    assert "no feasible route-slot strong enough" in rec.review_reason
    # The strongest option is still proposed for the specialist, with a structured floor.
    assert rec.recommended_route_id == "RTE-A" and rec.decision_summary


def test_llm_recommends_a_below_bar_option():
    """Symmetric: with a single feasible option below the bar, the LLM may still
    RECOMMEND it -- the bar doesn't block an auto-assign here."""
    evals = [scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.7, total=0.40)])]

    def fn(config, prompt):
        return _recommend(
            0, runner_up=None, key_tradeoff="Only workable option and its slot is open.",
            citations=[{"index": 0, "field": "reference_weighted_score", "value": 0.40}],
        )

    rec = decide_route_slot(customer(), evals, _cfg(route_slot_score_threshold=0.55), choice_fn=fn)
    assert rec.decision is Decision.RECOMMENDED  # 0.40 < 0.55, but the LLM recommended it
    assert rec.recommended_route_id == "RTE-A"


# --- k-try consensus ----------------------------------------------------------


def test_low_confidence_recommend_resamples_then_ships_on_unanimous():
    calls = {"n": 0}

    def fn(config, prompt):
        calls["n"] += 1
        conf = "LOW" if calls["n"] == 1 else "HIGH"  # first low -> resample; all recommend
        return _recommend(
            0, runner_up_index=1, confidence=conf,
            citations=[{"index": 0, "field": "reference_weighted_score", "value": 0.80}],
        )

    rec = decide_route_slot(
        customer(), _evals(), _cfg(judgment_consensus="unanimous", judgment_sample_count=3),
        choice_fn=fn,
    )
    assert rec.decision is Decision.RECOMMENDED
    assert calls["n"] == 3
    assert len(rec.alternative_takes) == 3  # divided takes surfaced


def test_majority_consensus_clears_a_mixed_batch():
    calls = {"n": 0}

    def fn(config, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return _escalate(0, runner_up_index=1)     # triggers resampling
        return _recommend(
            0, runner_up_index=1,
            citations=[{"index": 0, "field": "reference_weighted_score", "value": 0.80}],
        )

    rec = decide_route_slot(
        customer(), _evals(), _cfg(judgment_consensus="majority", judgment_sample_count=3),
        choice_fn=fn,
    )
    assert rec.decision is Decision.RECOMMENDED  # 2 recommend / 1 escalate -> majority


def test_unanimous_consensus_escalates_a_mixed_batch():
    calls = {"n": 0}

    def fn(config, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return _escalate(0, runner_up_index=1)
        return _recommend(
            0, runner_up_index=1,
            citations=[{"index": 0, "field": "reference_weighted_score", "value": 0.80}],
        )

    rec = decide_route_slot(
        customer(), _evals(), _cfg(judgment_consensus="unanimous", judgment_sample_count=3),
        choice_fn=fn,
    )
    assert rec.decision is Decision.ESCALATED_LOW_SCORE  # not all recommended


# --- fallbacks (never worse than the deterministic threshold baseline) --------


def test_backend_error_falls_back_to_deterministic_threshold():
    def boom(config, prompt):
        raise RuntimeError("SAGE_CLIENT_ID missing")

    rec = decide_route_slot(customer(), _evals(), _cfg(), choice_fn=boom)
    assert rec.decision is Decision.RECOMMENDED       # deterministic best clears 0.55
    assert rec.recommended_route_id == "RTE-A"
    assert rec.grounded_fallback is True


def test_fallback_below_bar_escalates_like_the_threshold_baseline():
    """On an LLM failure with nothing above the bar, the fallback is the
    deterministic threshold escalation -- never worse than the bar-gated path."""
    evals = [scored_eval("RTE-A", "Alpha", [scored_slot(MORNING, avail=0.3, total=0.40)])]

    def boom(config, prompt):
        raise RuntimeError("no creds")

    rec = decide_route_slot(
        customer(), evals, _cfg(route_slot_score_threshold=0.55), choice_fn=boom
    )
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert rec.grounded_fallback is True


def test_persistently_ungrounded_choice_falls_back():
    def liar(config, prompt):
        # Fabricated citation (idx0 openness is 0.33, not 0.99) -> fails both tries.
        return _recommend(
            0, runner_up_index=1,
            citations=[{"index": 0, "field": "slot_availability", "value": 0.99}],
        )

    rec = decide_route_slot(customer(), _evals(), _cfg(), choice_fn=liar)
    assert rec.grounded_fallback is True
    assert rec.recommended_route_id == "RTE-A"  # deterministic best


def test_no_feasible_route_always_escalates_without_the_llm():
    def boom(config, prompt):
        raise AssertionError("LLM must not be consulted with no feasible route")

    infeasible = scored_eval("RTE-X", "Xavier", [], feasible=False)
    rec = decide_route_slot(customer(), [infeasible], _cfg(), choice_fn=boom)
    assert rec.decision is Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None


# --- non-JSON reply (prose / tool call): one corrective JSON-only retry ------
#
# The sage generic agent tends to answer with prose ("I have analyzed...") or a
# tool call, so the first reply raises JSONDecodeError before verification. A single
# corrective retry demanding JSON-only should recover; two non-JSON replies fall back.


def test_json_only_directive_is_in_the_prompt():
    packet = build_route_slot_packet(customer(), _evals(), _cfg(), auto_assign_threshold=0.55)
    prompt = build_route_slot_decision_prompt(packet)
    # The forceful directive is present, and it forbids prose and tool calls.
    assert prompt.startswith("CRITICAL OUTPUT FORMAT")
    assert "do not call" in prompt.lower()
    assert prompt.rstrip().endswith("discarded.")  # repeated at the end too


def test_prose_first_reply_recovers_on_the_json_retry():
    calls = {"n": 0}

    def fn(config, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            # Mimics generate_route_slot_choice raising on a prose reply.
            raise json.JSONDecodeError("Expecting value", "I have analyzed...", 0)
        return _recommend(0, runner_up_index=1)

    rec = decide_route_slot(customer(), _evals(), _cfg(), choice_fn=fn)
    assert rec.decision is Decision.RECOMMENDED
    assert calls["n"] == 2               # prose, then the JSON-only retry
    assert not rec.grounded_fallback     # grounded pick shipped, not the fallback


def test_two_non_json_replies_fall_back_deterministically():
    def always_prose(config, prompt):
        raise json.JSONDecodeError("Expecting value", "prose", 0)

    rec = decide_route_slot(customer(), _evals(), _cfg(), choice_fn=always_prose)
    assert rec.grounded_fallback is True
    # Still a valid deterministic recommendation, never worse than the baseline.
    assert rec.recommended_route_id is not None
