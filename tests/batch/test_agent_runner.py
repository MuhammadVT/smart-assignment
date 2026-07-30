"""
Tests for the AGENT batch runner (batch/agent_runner.py), fully offline.

The real ADK agent needs LLM credentials, so these drive the runner with a FAKE
ADK runner + session service (the same technique as tests/webapp/test_llm_chat.py):
the fake runner yields scripted events, and the fake session service returns the
state the agent's tool WOULD have written (a seeded profile + the decision
snapshot). The payload/outcome mapping then runs the REAL deterministic pipeline
over MockGeocoder + mock routes, so every assertion is reproducible with no key.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from smart_assignment.batch.agent_runner import AgentBatchRunner
from smart_assignment.batch.sink import (
    OUTCOME_ESCALATE,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_RECOMMEND,
    ListResultSink,
)
from smart_assignment.batch.source import MockProspectSource, Prospect
from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.models import CustomerProfile, Decision, SlotRecommendation
from smart_assignment.tools.slot_recommendation import (
    _STATE_LAST_DECISION_KEY,
    _STATE_LAST_RECOMMENDATION_KEY,
    _STATE_PROFILE_KEY,
)


# --- Fakes for the ADK runner / session service / events --------------------


class _FakeCall:
    def __init__(self, name, id="fc1", args=None):
        self.name = name
        self.id = id
        self.args = args or {}


class _FakePart:
    def __init__(self, text):
        self.text = text


class _FakeContent:
    def __init__(self, text):
        self.parts = [_FakePart(text)] if text is not None else []


class _FakeEvent:
    def __init__(self, calls=None, long_running=None, text=None, partial=False):
        self._calls = calls or []
        self.long_running_tool_ids = set(long_running) if long_running else None
        self.partial = partial
        self.content = _FakeContent(text) if text is not None else None

    def get_function_calls(self):
        return self._calls

    def get_function_responses(self):
        return []


class _FakeSession:
    def __init__(self, state):
        self.state = state


class _FakeSessionService:
    """Returns a pre-baked state per session_id -- what the agent's assign_prospect
    tool would have written (the tool itself never runs behind a fake runner)."""

    def __init__(self, states):
        self._states = states
        self.created = []

    async def create_session(self, *, app_name, user_id, session_id, state=None):
        self.created.append(session_id)
        return _FakeSession(self._states.get(session_id, state or {}))

    async def get_session(self, *, app_name, user_id, session_id):
        return _FakeSession(self._states.get(session_id))


class _FakeRunner:
    """Yields the scripted events for the session_id it is asked to run."""

    def __init__(self, events_by_session):
        self._events = events_by_session
        self.run_configs = []

    async def run_async(self, *, user_id, session_id, new_message, run_config=None):
        self.run_configs.append(run_config)
        for event in self._events.get(session_id, []):
            yield event


class _RaisingRunner:
    async def run_async(self, **kwargs):
        raise RuntimeError("model backend down")
        yield  # pragma: no cover - makes this an async generator


# --- helpers ----------------------------------------------------------------

_BAYOU_ADDR = "1200 McKinney St, Houston, TX 77010"
_GALLERIA_ADDR = "5085 Westheimer Rd, Houston, TX 77056"


def _config():
    """Deterministic: the payload rebuild reuses the snapshot decision, so no LLM."""
    return replace(
        DEFAULT_CONFIG,
        use_grounded_route_slot_escalation=False,
        use_grounded_route_slot_pick=False,
        use_escalation_triage=False,
    )


def _profile_dict(address, cases=90):
    return {
        "name": "Test Prospect",
        "address": address,
        "order_quantity_cases": cases,
        "customer_number": None,
        "preferred_day": None,
        "preferred_window_start": None,
        "preferred_window_end": None,
    }


def _state_with_decision(profile, recommendation):
    """Session state as it stands AFTER assign_prospect ran: the seeded profile,
    the agent-facing summary (truthy), and the decision snapshot bound to it."""
    return {
        _STATE_PROFILE_KEY: profile,
        _STATE_LAST_RECOMMENDATION_KEY: {"decision": recommendation.decision.value},
        _STATE_LAST_DECISION_KEY: {
            "profile": dict(profile),
            "recommendation": recommendation.to_state_dict(),
        },
    }


def _recommend_rec():
    return SlotRecommendation(
        customer_name="Test Prospect",
        decision=Decision.RECOMMENDED,
        total_score=0.79,
        reasoning="Deterministic reasoning.",
        recommended_route_id="RTE-4100",
        recommended_route_name="Central Houston",
        recommended_day="TUE",
        recommended_window="07:20-10:20",
    )


def _escalate_rec():
    return SlotRecommendation(
        customer_name="Test Prospect",
        decision=Decision.ESCALATED_LOW_SCORE,
        total_score=0.11,
        reasoning="Below the auto-assign bar.",
        recommended_route_id="RTE-4100",
        recommended_route_name="Central Houston",
        recommended_day="TUE",
        recommended_window="07:20-10:20",
        review_reason="Large order on a near-full route.",
    )


def _runner(source, events_by_session, states):
    return AgentBatchRunner(
        source,
        ListResultSink(),
        config=_config(),
        geocoder=MockGeocoder(),
        routes=fetch_candidate_routes(),
        clock=lambda: "t0",
        runner=_FakeRunner(events_by_session),
        session_service=_FakeSessionService(states),
    )


async def _run(runner):
    summary = await runner.run()
    return summary, {r.prospect_id: r for r in runner._sink.records}


# --- recommend / escalate happy paths ---------------------------------------


async def test_recommend_turn_maps_to_a_recommend_record_with_agent_narration():
    prospect = Prospect("P1", CustomerProfile(name="Test Prospect", address=_BAYOU_ADDR,
                                              order_quantity_cases=90, preferred_slot=None))
    events = {
        "P1": [
            _FakeEvent(calls=[_FakeCall("assign_prospect")]),
            _FakeEvent(text="I recommend RTE-4100 - Central Houston on TUE, 07:20-10:20."),
        ]
    }
    states = {"P1": _state_with_decision(_profile_dict(_BAYOU_ADDR), _recommend_rec())}
    summary, by_id = await _run(_runner(MockProspectSource([prospect]), events, states))

    assert summary.recommend == 1 and summary.escalate == 0 and summary.needs_attention == 0
    rec = by_id["P1"]
    assert rec.outcome == OUTCOME_RECOMMEND
    assert rec.payload and rec.payload.get("frontendHtml")  # Customer View renders unchanged
    # The agent's own narration is threaded into the result card's reasoning.
    assert "I recommend RTE-4100 - Central Houston" in rec.payload["resultHtml"]


async def test_escalation_turn_captures_the_triage_brief_from_request_input():
    prospect = Prospect("P2", CustomerProfile(name="Test Prospect", address=_GALLERIA_ADDR,
                                              order_quantity_cases=400, preferred_slot=None))
    brief = "SITUATION: large order.\nDECISION NEEDED: split or defer?"
    events = {
        "P2": [
            _FakeEvent(calls=[_FakeCall("assign_prospect")]),
            _FakeEvent(
                calls=[_FakeCall("adk_request_input", id="req-1", args={"message": brief})],
                long_running=["req-1"],
            ),
        ]
    }
    states = {"P2": _state_with_decision(_profile_dict(_GALLERIA_ADDR, 400), _escalate_rec())}
    summary, by_id = await _run(_runner(MockProspectSource([prospect]), events, states))

    assert summary.escalate == 1
    esc = by_id["P2"]
    assert esc.outcome == OUTCOME_ESCALATE
    assert esc.review_reason == "Large order on a near-full route."
    assert esc.triage_brief and "SITUATION" in esc.triage_brief  # the agent's captured brief
    assert esc.payload is not None


async def test_request_input_is_captured_not_resumed():
    """The driver must NOT send a resume FunctionResponse -- there is no human. It
    runs exactly one turn per prospect and stops at the escalation."""
    prospect = Prospect("P2", CustomerProfile(name="Test Prospect", address=_GALLERIA_ADDR,
                                              order_quantity_cases=400, preferred_slot=None))
    events = {
        "P2": [
            _FakeEvent(calls=[_FakeCall("assign_prospect")]),
            _FakeEvent(
                calls=[_FakeCall("adk_request_input", id="r", args={"message": "SITUATION"})],
                long_running=["r"],
            ),
        ]
    }
    states = {"P2": _state_with_decision(_profile_dict(_GALLERIA_ADDR, 400), _escalate_rec())}
    runner = _runner(MockProspectSource([prospect]), events, states)
    await _run(runner)
    # Exactly one turn was run for the prospect (no resume turn).
    assert len(runner._runner.run_configs) == 1


# --- fallbacks: never worse than the deterministic baseline ------------------


async def test_no_decision_in_state_falls_back_to_the_deterministic_outcome():
    """If the agent reported an error instead of deciding (nothing in state), the
    prospect degrades to the deterministic pipeline -- here an invalid intake, so a
    clean needs_attention."""
    bad = Prospect("BAD", CustomerProfile(name="Zero", address=_BAYOU_ADDR,
                                          order_quantity_cases=0, preferred_slot=None))
    events = {"BAD": [_FakeEvent(text="I couldn't proceed: the order quantity is missing.")]}
    # State carries only the seeded profile -- no decision was written.
    states = {"BAD": {_STATE_PROFILE_KEY: _profile_dict(_BAYOU_ADDR, 0)}}
    summary, by_id = await _run(_runner(MockProspectSource([bad]), events, states))

    assert summary.needs_attention == 1
    assert by_id["BAD"].outcome == OUTCOME_NEEDS_ATTENTION
    assert by_id["BAD"].payload is None and by_id["BAD"].error


async def test_agent_turn_raising_falls_back_to_deterministic_recommend():
    prospect = Prospect("P1", CustomerProfile(name="Test Prospect", address=_BAYOU_ADDR,
                                              order_quantity_cases=90, preferred_slot=None))
    runner = AgentBatchRunner(
        MockProspectSource([prospect]),
        ListResultSink(),
        config=_config(),
        geocoder=MockGeocoder(),
        routes=fetch_candidate_routes(),
        clock=lambda: "t0",
        runner=_RaisingRunner(),
        session_service=_FakeSessionService({}),
    )
    summary, by_id = await _run(runner)
    # The turn blew up, but the deterministic floor still produced a real decision.
    assert summary.recommend == 1
    assert by_id["P1"].outcome == OUTCOME_RECOMMEND


async def test_no_agent_available_runs_the_whole_batch_deterministically():
    """When the batch agent can't be built (e.g. no credentials), _get_runner
    returns None and every prospect uses the deterministic pipeline."""
    source = MockProspectSource.from_samples()
    runner = AgentBatchRunner(
        source,
        ListResultSink(),
        config=_config(),
        geocoder=MockGeocoder(),
        routes=fetch_candidate_routes(),
        clock=lambda: "t0",
    )
    with patch(
        "smart_assignment.agent.build_batch_agent", side_effect=RuntimeError("no creds")
    ):
        summary, by_id = await _run(runner)

    # Same classification the deterministic BatchRunner produces for SAMPLE_CUSTOMERS.
    assert summary.total == 4
    assert by_id["MOCK-001"].outcome == OUTCOME_RECOMMEND
    assert by_id["MOCK-002"].outcome == OUTCOME_ESCALATE
    assert summary.recommend == 2 and summary.escalate == 2


async def test_one_prospect_failure_does_not_abort_the_batch():
    """A bad prospect becomes needs_attention; the batch continues to the next."""
    bad = Prospect("BAD", CustomerProfile(name="Zero", address=_BAYOU_ADDR,
                                          order_quantity_cases=0, preferred_slot=None))
    good = Prospect("GOOD", CustomerProfile(name="Test Prospect", address=_BAYOU_ADDR,
                                            order_quantity_cases=90, preferred_slot=None))
    events = {
        "BAD": [_FakeEvent(text="order quantity missing")],
        "GOOD": [
            _FakeEvent(calls=[_FakeCall("assign_prospect")]),
            _FakeEvent(text="I recommend RTE-4100 - Central Houston."),
        ],
    }
    states = {
        "BAD": {_STATE_PROFILE_KEY: _profile_dict(_BAYOU_ADDR, 0)},
        "GOOD": _state_with_decision(_profile_dict(_BAYOU_ADDR), _recommend_rec()),
    }
    summary, by_id = await _run(_runner(MockProspectSource([bad, good]), events, states))

    assert summary.total == 2
    assert by_id["BAD"].outcome == OUTCOME_NEEDS_ATTENTION
    assert by_id["GOOD"].outcome == OUTCOME_RECOMMEND


async def test_turn_runs_in_non_streaming_mode():
    prospect = Prospect("P1", CustomerProfile(name="Test Prospect", address=_BAYOU_ADDR,
                                              order_quantity_cases=90, preferred_slot=None))
    events = {"P1": [_FakeEvent(calls=[_FakeCall("assign_prospect")]), _FakeEvent(text="ok")]}
    states = {"P1": _state_with_decision(_profile_dict(_BAYOU_ADDR), _recommend_rec())}
    runner = _runner(MockProspectSource([prospect]), events, states)
    await _run(runner)

    from google.adk.agents.run_config import StreamingMode

    assert runner._runner.run_configs[0].streaming_mode == StreamingMode.NONE


async def test_fresh_session_is_seeded_per_prospect():
    prospect = Prospect("P1", CustomerProfile(name="Test Prospect", address=_BAYOU_ADDR,
                                              order_quantity_cases=90, preferred_slot=None))
    events = {"P1": [_FakeEvent(calls=[_FakeCall("assign_prospect")]), _FakeEvent(text="ok")]}
    states = {"P1": _state_with_decision(_profile_dict(_BAYOU_ADDR), _recommend_rec())}
    runner = _runner(MockProspectSource([prospect]), events, states)
    await _run(runner)
    # A session named for the prospect was created (seeded with its profile).
    assert runner._session_service.created == ["P1"]
