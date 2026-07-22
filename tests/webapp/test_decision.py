"""
Unit tests for webapp/decision.py -- the shared feedback context extractor and
the traced-decision span helper (increments A + B).
"""

from __future__ import annotations

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import (
    CustomerProfile,
    Decision,
    RecommendationResult,
    SlotRecommendation,
)
from smart_assignment.webapp.decision import (
    decision_outcome,
    feedback_context,
    traced_decision,
)


def _result(decision, route_id="RTE-4100", window="07:00-10:00", cases=90):
    rec = SlotRecommendation(
        customer_name="Bayou City Bistro",
        decision=decision,
        total_score=0.8,
        reasoning="because",
        recommended_route_id=route_id,
        recommended_window=window,
    )
    customer = CustomerProfile(
        name="Bayou City Bistro",
        address="1200 McKinney St, Houston, TX",
        order_quantity_cases=cases,
    )
    return RecommendationResult(
        customer=customer,
        candidates_considered=[],
        ranked_feasible=[],
        recommendation=rec,
    )


def test_decision_outcome_recommend_and_escalate():
    assert decision_outcome(_result(Decision.RECOMMENDED).recommendation) == "recommend"
    assert decision_outcome(_result(Decision.ESCALATED_LOW_SCORE).recommendation) == "escalate"
    no_slot = _result(Decision.ESCALATED_NO_FEASIBLE_SLOT).recommendation
    assert decision_outcome(no_slot) == "escalate"
    assert decision_outcome(None) is None


def test_feedback_context_structured_facts_only():
    ctx = feedback_context(_result(Decision.RECOMMENDED, route_id="RTE-9", cases=120))
    assert ctx["outcome"] == "recommend"
    assert ctx["recommended_route_id"] == "RTE-9"
    assert ctx["recommended_window"] == "07:00-10:00"
    assert ctx["order_quantity_cases"] == 120
    # No customer name/address here -- that PII is added by the caller downstream.
    assert "name" not in ctx and "address" not in ctx


def test_feedback_context_drops_nones():
    ctx = feedback_context(_result(Decision.RECOMMENDED, route_id=None, window=None))
    assert "recommended_route_id" not in ctx
    assert "recommended_window" not in ctx
    assert ctx["outcome"] == "recommend"


def test_feedback_context_empty_for_none():
    assert feedback_context(None) == {}


def test_traced_decision_is_noop_without_tracing():
    cfg = Config(use_tracing=False)
    with traced_decision(cfg) as decision:
        pass
    # No SDK / tracing off -> empty coordinates, and the body ran fine.
    assert decision.coords == {}
    assert decision.context == {}


def test_record_attaches_decision_facts_to_span(monkeypatch):
    # Bullet 6: the webapp.recommendation span carries the decision facts, not
    # just a role label -- so it's informative in Phoenix. Inject a fake span.
    import contextlib

    from smart_assignment.webapp import decision as decision_mod

    captured = {}

    class _FakeSpan:
        def set_attribute(self, key, value):
            captured[key] = value

    @contextlib.contextmanager
    def _fake_llm_span(config, label, **attrs):
        yield _FakeSpan()

    monkeypatch.setattr(decision_mod, "llm_span", _fake_llm_span)
    with traced_decision(Config(use_tracing=True)) as d:
        d.record(_result(Decision.ESCALATED_LOW_SCORE, route_id="RTE-4200", cases=400))

    assert captured["smart_assignment.decision.outcome"] == "escalate"
    assert captured["smart_assignment.decision.recommended_route_id"] == "RTE-4200"
    assert captured["smart_assignment.decision.order_quantity_cases"] == 400


def test_traced_decision_record_populates_context_body_runs():
    ran = {}
    with traced_decision(Config(use_tracing=False)) as decision:
        ran["did"] = True
        # record() on a no-op span must not raise and must stash the context.
        ctx = decision.record(_result(Decision.RECOMMENDED, route_id="RTE-7"))
        assert ctx["recommended_route_id"] == "RTE-7"
    assert ran["did"] is True
    assert decision.context["outcome"] == "recommend"
    assert decision.coords == {}
