"""
GroundedJudge orchestration: sampling, consensus, the config knobs, and the
deterministic fallback that guarantees "never worse than current."

All offline -- every LLM interaction is a `FakeJudgmentFn`.
"""

from __future__ import annotations

from dataclasses import replace

from smart_assignment.judgment import GroundedJudge, WeightedScoreJudge, default_judge
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision

from .conftest import (
    CLEAR_RECOMMEND,
    LOW_SCORE,
    NO_FEASIBLE,
    FakeJudgmentFn,
    escalate,
    evaluations_for,
    feasible_ids,
    grounded_citation,
    recommend,
)

_FALLBACK = DeterministicReasoner()


def _judge(fake):
    return GroundedJudge(judgment_fn=fake, fallback_reasoner=_FALLBACK)


# --- the confident happy path ships on ONE call -----------------------------


def test_confident_recommend_ships_on_a_single_call():
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    winner = feasible_ids(evals)[0]
    cite = grounded_citation(customer, evals, winner, config)
    fake = FakeJudgmentFn(recommend(winner, confidence="HIGH", citations=[cite]))

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 1  # no resampling for a confident recommend
    assert rec.decision is Decision.RECOMMENDED
    assert rec.recommended_route_id == winner
    assert rec.alternative_takes == []  # single call -> no surfaced takes


# --- escalation-side cases resample up to k ---------------------------------


def test_hard_escalate_resamples_k_times_and_escalates():
    config = Config(use_grounded_judgment=True, judgment_sample_count=3)
    customer, evals = evaluations_for(LOW_SCORE, config)
    fake = FakeJudgmentFn(escalate())  # every sample escalates

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 3  # spent the full budget on an escalation-side case
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert rec.requires_human_review is True
    assert len(rec.alternative_takes) == 3  # all takes surfaced to the specialist
    assert rec.recommended_route_id is not None  # a slot is still proposed


def test_k_of_one_disables_resampling():
    config = Config(use_grounded_judgment=True, judgment_sample_count=1)
    customer, evals = evaluations_for(LOW_SCORE, config)
    fake = FakeJudgmentFn(escalate())

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 1
    assert rec.decision is Decision.ESCALATED_LOW_SCORE


# --- the low-confidence-recommend config knob -------------------------------


def test_low_conf_recommend_resamples_when_knob_on():
    config = Config(
        use_grounded_judgment=True,
        judgment_sample_count=3,
        judgment_retry_on_low_confidence_recommend=True,
    )
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    winner = feasible_ids(evals)[0]
    cite = grounded_citation(customer, evals, winner, config)
    fake = FakeJudgmentFn(recommend(winner, confidence="LOW", citations=[cite]))

    rec = _judge(fake).decide(customer, evals, config)

    # Hedged recommend is treated as escalation-side -> resampled. All 3 agree
    # to recommend, so under unanimous consensus it clears back to RECOMMENDED.
    assert fake.calls == 3
    assert rec.decision is Decision.RECOMMENDED
    assert len(rec.alternative_takes) == 3  # but the split is surfaced


def test_low_conf_recommend_ships_immediately_when_knob_off():
    config = Config(
        use_grounded_judgment=True,
        judgment_sample_count=3,
        judgment_retry_on_low_confidence_recommend=False,
    )
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    winner = feasible_ids(evals)[0]
    cite = grounded_citation(customer, evals, winner, config)
    fake = FakeJudgmentFn(recommend(winner, confidence="LOW", citations=[cite]))

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 1  # a pick is a pick
    assert rec.decision is Decision.RECOMMENDED
    assert rec.alternative_takes == []


# --- consensus rule: unanimous vs majority ----------------------------------


def _mixed_first_escalate_then_two_recommend(winner, cite):
    # First sample escalates (triggers resample); next two recommend.
    return FakeJudgmentFn(
        [escalate(), recommend(winner, citations=[cite]), recommend(winner, citations=[cite])]
    )


def test_unanimous_consensus_escalates_on_a_split():
    config = Config(
        use_grounded_judgment=True, judgment_sample_count=3, judgment_consensus="unanimous"
    )
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    winner = feasible_ids(evals)[0]
    fake = _mixed_first_escalate_then_two_recommend(
        winner, grounded_citation(customer, evals, winner, config)
    )

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 3
    # 2 of 3 recommend -> not unanimous -> escalate (precautionary).
    assert rec.decision is Decision.ESCALATED_LOW_SCORE


def test_majority_consensus_clears_the_same_split():
    config = Config(
        use_grounded_judgment=True, judgment_sample_count=3, judgment_consensus="majority"
    )
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    winner = feasible_ids(evals)[0]
    fake = _mixed_first_escalate_then_two_recommend(
        winner, grounded_citation(customer, evals, winner, config)
    )

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 3
    # 2 of 3 recommend -> majority -> recommend.
    assert rec.decision is Decision.RECOMMENDED
    assert rec.recommended_route_id == winner


def test_differing_good_picks_are_not_treated_as_disagreement():
    # Two samples recommend, but different routes. That's agreement on the
    # DECISION axis, so it should clear (even under unanimous consensus).
    config = Config(
        use_grounded_judgment=True, judgment_sample_count=2, judgment_consensus="unanimous"
    )
    customer, evals = evaluations_for(LOW_SCORE, config)
    ids = feasible_ids(evals)
    a = ids[0]
    b = ids[1] if len(ids) > 1 else ids[0]
    # First is a LOW-confidence recommend (escalation-side -> resample), second
    # recommends a different (or same) route.
    fake = FakeJudgmentFn(
        [
            recommend(
                a, confidence="LOW", citations=[grounded_citation(customer, evals, a, config)]
            ),
            recommend(
                b, confidence="HIGH", citations=[grounded_citation(customer, evals, b, config)]
            ),
        ]
    )

    rec = _judge(fake).decide(customer, evals, config)

    assert rec.decision is Decision.RECOMMENDED  # both recommended -> cleared


# --- the "never worse than current" guarantees ------------------------------


def test_mechanical_failure_falls_back_to_the_deterministic_pick():
    # If the model errors on every call, we must land on EXACTLY today's
    # deterministic weighted result -- never something worse.
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    fake = FakeJudgmentFn(RuntimeError("backend down"))

    grounded = _judge(fake).decide(customer, evals, config)
    weighted = WeightedScoreJudge(reasoner=_FALLBACK).decide(customer, evals, config)

    assert grounded.decision == weighted.decision == Decision.RECOMMENDED
    assert grounded.recommended_route_id == weighted.recommended_route_id
    assert grounded.reasoning == weighted.reasoning  # identical deterministic output


def test_mechanical_failure_marks_grounded_fallback_for_the_ui():
    # A backend/credentials failure must flag the result so the UI can show a
    # "grounded reasoning unavailable" notice.
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    fake = FakeJudgmentFn(RuntimeError("SAGE_CLIENT_ID missing"))

    rec = _judge(fake).decide(customer, evals, config)

    assert rec.grounded_fallback is True
    assert rec.grounded_fallback_reason and "unavailable" in rec.grounded_fallback_reason.lower()


def test_successful_grounded_recommend_does_not_flag_fallback():
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    winner = feasible_ids(evals)[0]
    cite = grounded_citation(customer, evals, winner, config)
    fake = FakeJudgmentFn(recommend(winner, confidence="HIGH", citations=[cite]))

    rec = _judge(fake).decide(customer, evals, config)

    assert rec.grounded_fallback is False
    assert rec.grounded_fallback_reason is None


def test_no_feasible_route_is_not_a_grounded_fallback():
    # No feasible route is a legitimate deterministic outcome, not a grounded
    # failure -- it must NOT trigger the "reasoning unavailable" notice.
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(NO_FEASIBLE, config)
    fake = FakeJudgmentFn(recommend("RTE-ANYTHING"))

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 0
    assert rec.grounded_fallback is False


def test_mechanical_failure_is_logged_not_silent(caplog):
    # The fallback must be observable: a backend/credentials failure that lands
    # on the deterministic text should leave a WARNING explaining why, so an
    # identical-looking output is never a silent mystery.
    import logging

    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    fake = FakeJudgmentFn(RuntimeError("SAGE_CLIENT_ID missing"))

    with caplog.at_level(logging.WARNING, logger="smart_assignment.judgment.judge"):
        _judge(fake).decide(customer, evals, config)

    text = caplog.text
    assert "SAGE_CLIENT_ID missing" in text  # the underlying cause is surfaced
    assert "falling back" in text.lower()


def test_no_feasible_route_never_calls_the_model_and_escalates():
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(NO_FEASIBLE, config)
    fake = FakeJudgmentFn(recommend("RTE-ANYTHING"))

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 0  # deterministic no-feasible path, no LLM needed
    assert rec.decision is Decision.ESCALATED_NO_FEASIBLE_SLOT
    assert rec.recommended_route_id is None


def test_persistently_infeasible_pick_falls_back_not_ships():
    # The model keeps trying to pick an infeasible route. The verifier rejects
    # it (initial + one retry), then we fall back to the deterministic pick --
    # an infeasible route can never be auto-assigned.
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(LOW_SCORE, config)
    fake = FakeJudgmentFn(recommend("RTE-INFEASIBLE-XYZ"))

    rec = _judge(fake).decide(customer, evals, config)

    assert fake.calls == 2  # initial + one corrective retry, both rejected
    # Falls back to today's deterministic Galleria outcome.
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert rec.recommended_route_id in set(feasible_ids(evals))


# --- the factory + default (off) behavior -----------------------------------


def test_default_judge_off_is_the_weighted_strategy():
    config = Config(use_grounded_judgment=False)
    assert isinstance(default_judge(config, reasoner=_FALLBACK), WeightedScoreJudge)


def test_default_judge_on_is_the_grounded_strategy():
    config = Config(use_grounded_judgment=True)
    assert isinstance(default_judge(config), GroundedJudge)


def test_weighted_strategy_matches_pipeline_decide_default():
    # WeightedScoreJudge must reproduce today's decision exactly.
    config = replace(Config(), use_grounded_judgment=False)
    customer, evals = evaluations_for(LOW_SCORE, config)
    rec = WeightedScoreJudge(reasoner=_FALLBACK).decide(customer, evals, config)
    assert rec.decision is Decision.ESCALATED_LOW_SCORE
    assert rec.total_score < config.total_score_threshold
