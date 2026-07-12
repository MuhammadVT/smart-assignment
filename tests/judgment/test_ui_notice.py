"""
The grounded-fallback flag must reach the web-app payload as a structured
notice, so the UI can warn the user that the reasoning shown is the
deterministic fallback (not grounded output).
"""

from __future__ import annotations

from smart_assignment.pipeline import rank_feasible
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.reporting.page import build_workflow_payload
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import RecommendationResult

from .conftest import CLEAR_RECOMMEND, FakeJudgmentFn, evaluations_for
from smart_assignment.judgment import GroundedJudge


def _result_for(recommendation, customer, evals) -> RecommendationResult:
    return RecommendationResult(
        customer=customer,
        candidates_considered=evals,
        ranked_feasible=rank_feasible(evals),
        recommendation=recommendation,
    )


def test_fallback_surfaces_a_warning_notice_in_the_payload():
    config = Config(use_grounded_judgment=True)
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    judge = GroundedJudge(
        judgment_fn=FakeJudgmentFn(RuntimeError("no creds")),
        fallback_reasoner=DeterministicReasoner(),
    )
    rec = judge.decide(customer, evals, config)
    assert rec.grounded_fallback is True  # precondition

    payload = build_workflow_payload(_result_for(rec, customer, evals), config)
    notices = payload["notices"]
    assert len(notices) == 1
    assert notices[0]["kind"] == "warning"
    assert "unavailable" in notices[0]["text"].lower()


def test_no_notice_on_the_normal_deterministic_path():
    # The default weighted/deterministic result carries no fallback flag, so the
    # payload (and the published static page) shows no banner.
    config = Config()  # grounded off
    customer, evals = evaluations_for(CLEAR_RECOMMEND, config)
    from smart_assignment.pipeline import decide

    rec = decide(customer, evals, DeterministicReasoner(), config)
    payload = build_workflow_payload(_result_for(rec, customer, evals), config)
    assert payload["notices"] == []
