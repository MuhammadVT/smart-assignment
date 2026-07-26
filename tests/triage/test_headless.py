"""
Composing the escalation brief without a conversation
(`smart_assignment.triage.headless`).

The point of this module is reuse: the *existing* `escalation_triage` agent
writes the brief, so there is no second prompt and no second context builder to
drift. These tests therefore protect the seam rather than the prose:

* the throwaway session is seeded with exactly what `get_escalation_context`
  reads, built from `to_state_dict()` so a new field cannot silently go missing;
* a stub agent is driven through the real ADK `Runner`, so the session seeding,
  the event filtering, and the text extraction are genuinely exercised;
* every failure path returns ``None`` -- a specialist keeps the structured
  escalation facts, and the decision is never affected.

No model and no credentials: the agent is stubbed, except where a test asserts
what happens when the backend is unavailable.
"""

from __future__ import annotations

from datetime import time
from typing import AsyncGenerator

import pytest
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import (
    CustomerProfile,
    Decision,
    DayOfWeek,
    PreferredSlot,
    SlotRecommendation,
)
from smart_assignment.triage import headless
from smart_assignment.tools.slot_recommendation import (
    _STATE_LAST_RECOMMENDATION_KEY,
    _STATE_PROFILE_KEY,
)

_BRIEF = (
    "SITUATION\nGalleria Grill, 400 cases, escalated for review.\n\n"
    "ROOT CAUSE\nNo route-slot cleared the 55% bar (best 40%).\n\n"
    "OPTIONS (most workable first)\n"
    "1) RTE-4200 - West Houston · WED - 84% utilized\n"
    "Action: split the order across two days\n"
    "Trade-off: two deliveries instead of one\n\n"
    "RECOMMENDATION\nStart with option 1.\n\n"
    "DECISION NEEDED\nApprove the split, or hold for capacity?"
)


def _customer() -> CustomerProfile:
    return CustomerProfile(
        name="Galleria Grill & Catering",
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=400,
        customer_number="067-100002",
        preferred_slot=PreferredSlot(DayOfWeek.WED, (time(9, 0), time(12, 0))),
    )


def _escalation(decision: Decision = Decision.ESCALATED_LOW_SCORE) -> SlotRecommendation:
    return SlotRecommendation(
        customer_name="Galleria Grill & Catering",
        decision=decision,
        total_score=0.40,
        reasoning="No route-slot cleared the auto-assign bar.",
        recommended_route_id="RTE-4200",
        recommended_route_name="West Houston / Energy Corridor",
        recommended_day="WED",
        recommended_window="11:30-14:30",
        review_reason="No route-slot cleared the 55% auto-assign bar (best 40%).",
        alternative_takes=["[ESCALATE/LOW] RTE-4200: too tight."],
    )


class _StubAgent(BaseAgent):
    """Stands in for the triage LlmAgent, driven by the real ADK Runner.

    It records the session state it was handed, which is what proves the seeding
    contract without needing a model.
    """

    def __init__(self, text: str = _BRIEF, **kwargs):
        super().__init__(name="stub_triage", **kwargs)
        # BaseAgent is a pydantic model, so stash on the instance dict directly.
        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "seen_state", {})

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        object.__setattr__(self, "seen_state", dict(ctx.session.state))
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=self._text)]),
        )


@pytest.fixture
def stub_agent(monkeypatch):
    agent = _StubAgent()
    monkeypatch.setattr(headless, "_triage_agent", lambda config: agent)
    return agent


# --- the seam: what the agent is handed --------------------------------------


def test_session_is_seeded_with_what_get_escalation_context_reads(stub_agent):
    headless.compose_brief(_customer(), _escalation(), Config())
    state = stub_agent.seen_state

    profile = state[_STATE_PROFILE_KEY]
    assert profile["address"] == "5085 Westheimer Rd, Houston, TX 77056"
    assert profile["order_quantity_cases"] == 400
    assert profile["preferred_day"] == "WED"
    assert profile["preferred_window_start"] == "09:00"

    last = state[_STATE_LAST_RECOMMENDATION_KEY]
    # The four keys get_escalation_context gates on or reads directly.
    assert last["requires_human_review"] is True
    assert last["decision"] == Decision.ESCALATED_LOW_SCORE.value
    assert last["recommended_route_id"] == "RTE-4200"
    assert last["review_reason"].startswith("No route-slot cleared")
    assert last["alternative_takes"] == ["[ESCALATE/LOW] RTE-4200: too tight."]


def test_seeded_recommendation_carries_every_declared_field(stub_agent):
    """Built from to_state_dict() rather than hand-picked keys, so a field added
    to SlotRecommendation reaches triage automatically."""
    from dataclasses import fields

    headless.compose_brief(_customer(), _escalation(), Config())
    seeded = stub_agent.seen_state[_STATE_LAST_RECOMMENDATION_KEY]

    declared = {f.name for f in fields(SlotRecommendation)}
    assert declared.issubset(set(seeded))


@pytest.mark.parametrize(
    "decision",
    [Decision.ESCALATED_LOW_SCORE, Decision.ESCALATED_NO_FEASIBLE_SLOT],
)
def test_every_escalation_kind_gets_a_brief(stub_agent, decision):
    """Low-score and no-feasible-route both escalate, and a specialist needs the
    handoff in both cases -- the brief is not only for the ones with a proposed
    route."""
    brief = headless.compose_brief(_customer(), _escalation(decision), Config())
    assert brief is not None and "SITUATION" in brief


def test_brief_text_is_returned_and_normalized(stub_agent):
    brief = headless.compose_brief(_customer(), _escalation(), Config())
    assert brief is not None
    for header in ("SITUATION", "ROOT CAUSE", "OPTIONS", "RECOMMENDATION", "DECISION NEEDED"):
        assert header in brief
    # normalize_brief is idempotent, so a well-formed brief survives unchanged.
    from smart_assignment.triage.formatting import normalize_brief

    assert normalize_brief(brief) == brief


# --- every failure is None, never an exception -------------------------------


def test_auto_approved_recommendation_has_nothing_to_triage(stub_agent):
    approved = SlotRecommendation(
        customer_name="Bayou City Bistro",
        decision=Decision.RECOMMENDED,
        total_score=0.79,
        reasoning="Clear winner.",
        recommended_route_id="RTE-4100",
    )
    assert headless.compose_brief(_customer(), approved, Config()) is None


def test_agent_failure_returns_none_rather_than_raising(monkeypatch, caplog):
    def _boom(config):
        raise RuntimeError("no backend configured")

    monkeypatch.setattr(headless, "_triage_agent", _boom)
    assert headless.compose_brief(_customer(), _escalation(), Config()) is None
    assert any("could not be composed" in r.message for r in caplog.records)


def test_an_empty_brief_is_treated_as_no_brief(monkeypatch):
    """Better no brief than an empty section skeleton presented as a handoff."""
    monkeypatch.setattr(headless, "_triage_agent", lambda config: _StubAgent(text="   "))
    assert headless.compose_brief(_customer(), _escalation(), Config()) is None


def test_a_hanging_agent_is_cut_off_by_the_timeout(monkeypatch):
    """An agent can loop; a production request cannot hang waiting for one."""
    import asyncio

    class _Hanging(_StubAgent):
        async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
            await asyncio.sleep(30)
            yield Event(author=self.name)  # pragma: no cover - never reached

    monkeypatch.setattr(headless, "_triage_agent", lambda config: _Hanging())
    assert (
        headless.compose_brief(
            _customer(), _escalation(), Config(), timeout_seconds=0.25
        )
        is None
    )


def test_the_agent_is_built_once_per_model(monkeypatch):
    """Resolving the backend per brief would be wasteful; the cache key covers
    everything in Config that changes how the agent is built."""
    builds = {"n": 0}

    def _counting(config):
        builds["n"] += 1
        return _StubAgent()

    headless._AGENT_CACHE.clear()
    monkeypatch.setattr("smart_assignment.triage.agent.build_triage_agent", _counting)
    config = Config()
    headless._triage_agent(config)
    headless._triage_agent(config)
    headless._AGENT_CACHE.clear()

    assert builds["n"] == 1
