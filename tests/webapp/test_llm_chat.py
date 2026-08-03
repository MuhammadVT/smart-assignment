"""
Tests for the Phase 2 LLM-conversational service (smart_assignment/webapp/llm_chat.py).

The real ADK agent needs LLM credentials, so these drive the streaming logic with
a FAKE runner + session service and an offline geocoder — exercising event→frame
mapping, the visualization rebuilt from session state, human-in-the-loop resume,
and the mode/credential resolution. No network, no key.
"""

from __future__ import annotations

import pytest

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision, SlotRecommendation
from smart_assignment.tools.slot_recommendation import (
    _STATE_LAST_DECISION_KEY,
    _STATE_PROFILE_KEY,
)
from smart_assignment.webapp.llm_chat import (
    LlmChatService,
    llm_credentials_available,
    resolve_mode,
    webapp_mode,
)

# --- Fakes for the ADK runner / session service / events --------------------


class _FakeCall:
    def __init__(self, name, id="fc1", args=None):
        self.name = name
        self.id = id
        self.args = args or {}


class _FakeResponse:
    """A tool's return value coming back on a FunctionResponse event. Defaults to
    the ``{"ok": True}`` shape every pipeline tool returns on success."""

    def __init__(self, name, id="fc1", response=None):
        self.name = name
        self.id = id
        self.response = {"ok": True} if response is None else response


class _FakePart:
    def __init__(self, text):
        self.text = text


class _FakeContent:
    def __init__(self, text):
        self.parts = [_FakePart(text)] if text is not None else []


class _FakeEvent:
    def __init__(self, calls=None, responses=None, long_running=None, text=None, partial=False):
        self._calls = calls or []
        self._responses = responses or []
        self.long_running_tool_ids = set(long_running) if long_running else None
        self.partial = partial
        self.content = _FakeContent(text) if text is not None else None

    def get_function_calls(self):
        return self._calls

    def get_function_responses(self):
        return self._responses


class _FakeSession:
    def __init__(self, state):
        self.state = state


class _FakeSessionService:
    def __init__(self, state=None):
        self._state = state or {}
        self.created = []

    async def create_session(self, *, app_name, user_id, session_id, state=None):
        self.created.append(session_id)
        return _FakeSession(self._state)

    async def get_session(self, *, app_name, user_id, session_id):
        return _FakeSession(self._state)


class _FakeRunner:
    """Yields a pre-scripted batch of events per run_async call, recording the
    new_message each call received (so a resume can be asserted)."""

    def __init__(self, batches):
        self._batches = [list(b) for b in batches]
        self.messages = []
        self.run_configs = []

    async def run_async(self, *, user_id, session_id, new_message, run_config=None):
        self.messages.append(new_message)
        self.run_configs.append(run_config)
        batch = self._batches.pop(0) if self._batches else []
        for event in batch:
            yield event


_SAMPLE_STATE = {
    _STATE_PROFILE_KEY: {
        "name": "Test Prospect",
        "address": "1200 McKinney St, Houston, TX 77010",
        "order_quantity_cases": 90,
        "customer_number": None,
        "preferred_day": "TUE",
        "preferred_window_start": "07:00",
        "preferred_window_end": "10:00",
    }
}


def _tool_pair(name, args=None, response=None, id=None):
    """The call + response event pair ADK always emits for one tool invocation.

    Breadcrumbs open as ``running`` on the call and are settled from the response,
    so a scripted turn must carry BOTH -- a call on its own means the tool was
    asked to do the work and hasn't reported back yet.
    """
    call_id = id or f"fc-{name}"
    return [
        _FakeEvent(calls=[_FakeCall(name, id=call_id, args=args)]),
        _FakeEvent(responses=[_FakeResponse(name, id=call_id, response=response)]),
    ]


def _steps(frames, status=None):
    """(name, status) for each tool frame, optionally filtered to one status."""
    return [
        (f["name"], f.get("status"))
        for f in frames
        if f["type"] == "tool" and (status is None or f.get("status") == status)
    ]


async def _collect(agen):
    return [frame async for frame in agen]


# --- Mode / credential resolution -------------------------------------------


def test_webapp_mode_defaults_to_llm(monkeypatch):
    monkeypatch.delenv("SMART_ASSIGNMENT_WEBAPP_MODE", raising=False)
    assert webapp_mode() == "llm"
    monkeypatch.setenv("SMART_ASSIGNMENT_WEBAPP_MODE", "deterministic")
    assert webapp_mode() == "deterministic"


def test_llm_credentials_available_standard(monkeypatch):
    cfg = Config(llm_backend="standard", model="gemini-2.5-flash")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    assert llm_credentials_available(cfg) is False
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert llm_credentials_available(cfg) is True


def test_llm_credentials_available_sage(monkeypatch):
    cfg = Config(llm_backend="sage")
    for v in ("SAGE_CLIENT_ID", "SAGE_CLIENT_SECRET", "SAGE_ENVIRONMENT"):
        monkeypatch.setenv(v, "x")
    assert llm_credentials_available(cfg) is True
    monkeypatch.delenv("SAGE_ENVIRONMENT", raising=False)
    assert llm_credentials_available(cfg) is False


def test_resolve_mode_stays_llm_without_a_credential_pre_check(monkeypatch):
    # The web app drives the real agent (like adk web) rather than pre-guessing
    # credentials and downgrading to the parser -- that heuristic gave false
    # negatives and stranded the chat on Phase 1. Genuine no-credential failures
    # are handled at runtime by the /api/chat deterministic fallback instead.
    monkeypatch.setenv("SMART_ASSIGNMENT_WEBAPP_MODE", "llm")
    cfg = Config(llm_backend="standard", model="gemini-2.5-flash")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    res = resolve_mode(cfg)
    assert res["mode"] == "llm"
    assert res["configured"] == "llm"


def test_resolve_mode_llm_when_available(monkeypatch):
    monkeypatch.setenv("SMART_ASSIGNMENT_WEBAPP_MODE", "llm")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    cfg = Config(llm_backend="standard", model="gemini-2.5-flash")
    assert resolve_mode(cfg)["mode"] == "llm"


def test_resolve_mode_explicit_deterministic(monkeypatch):
    monkeypatch.setenv("SMART_ASSIGNMENT_WEBAPP_MODE", "deterministic")
    assert resolve_mode(Config())["mode"] == "deterministic"


# --- Streaming a conversational turn ----------------------------------------


async def test_stream_turn_maps_tools_and_renders_visualization():
    events = [
        *_tool_pair("intake_customer"),
        *_tool_pair("find_candidate_routes"),
        *_tool_pair("evaluate_and_score_routes"),
        *_tool_pair("recommend_or_escalate"),
        _FakeEvent(text="Here is my recommendation."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "New prospect at 1200 McKinney St, 90 cases"))

    labels = [f["label"] for f in frames if f["type"] == "tool" and f.get("status") == "running"]
    assert labels == ["Intake", "Geo-Lookup", "Score & Rank", "Recommend / Decide"]
    # Each step opens running and is settled done by its own tool's response.
    assert _steps(frames, "done") == [
        ("intake_customer", "done"),
        ("find_candidate_routes", "done"),
        ("evaluate_and_score_routes", "done"),
        ("recommend_or_escalate", "done"),
    ]
    assert any(f["type"] == "message" and "recommendation" in f["text"] for f in frames)
    viz = [f for f in frames if f["type"] == "visualization"]
    assert len(viz) == 1
    assert len(viz[0]["payload"]["steps"]) == 5
    assert frames[-1] == {"type": "done"}


async def test_breadcrumbs_track_pipeline_steps_not_tool_calls():
    """The optimized default flow makes only TWO tool calls (intake ->
    recommend_or_escalate), but the live stepper must still show ALL four pipeline
    steps: recommend_or_escalate runs geo + score + decide internally, so its one
    call lights up Geo-Lookup, Score & Rank, and Recommend/Decide."""
    events = [
        *_tool_pair("intake_customer"),
        *_tool_pair("recommend_or_escalate"),
        _FakeEvent(text="Here is my recommendation."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "New prospect at 1200 McKinney St, 90 cases"))

    labels = [f["label"] for f in frames if f["type"] == "tool" and f.get("status") == "running"]
    # All four steps, in order, from just two tool calls.
    assert labels == ["Intake", "Geo-Lookup", "Score & Rank", "Recommend / Decide"]
    # And the full 5-step visualization still renders (state-derived, unchanged).
    viz = [f for f in frames if f["type"] == "visualization"]
    assert len(viz) == 1 and len(viz[0]["payload"]["steps"]) == 5


async def test_consolidated_tool_opens_its_steps_running_and_settles_them_together():
    """A step must never be shown as done off the back of a CALL. One
    recommend_or_escalate call opens Geo-Lookup + Score & Rank + Recommend/Decide
    as running; all three turn done only when that tool reports back."""
    events = [
        *_tool_pair("recommend_or_escalate"),
        _FakeEvent(text="Here is my recommendation."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "decide for 1200 McKinney St, 90 cases"))

    # Every running frame precedes every done frame: nothing is green while the
    # single tool call that does all three steps is still in flight.
    statuses = [s for _, s in _steps(frames)]
    assert statuses == ["running"] * 3 + ["done"] * 3
    assert _steps(frames, "running") == [
        ("find_candidate_routes", "running"),
        ("evaluate_and_score_routes", "running"),
        ("recommend_or_escalate", "running"),
    ]


async def test_failed_tool_marks_its_steps_failed_and_renders_no_visualization():
    """When the tool reports {"ok": false} -- e.g. the address can't be geocoded --
    none of its steps ran, so none may show as done. They are marked failed with
    the tool's OWN error text, and no result cards are rendered for a decision that
    was never made."""
    error = "I couldn't find a close match for '9999 Zzzqqx Nowhere Blvd'."
    events = [
        *_tool_pair("intake_customer"),
        *_tool_pair("recommend_or_escalate", response={"ok": False, "error": error}),
        _FakeEvent(text="I couldn't find that address -- could you double-check it?"),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "9999 Zzzqqx Nowhere Blvd, 60 cases"))

    assert _steps(frames, "done") == [("intake_customer", "done")]
    assert _steps(frames, "failed") == [
        ("find_candidate_routes", "failed"),
        ("evaluate_and_score_routes", "failed"),
        ("recommend_or_escalate", "failed"),
    ]
    # The failure breadcrumb relays the tool's own words, not a guessed cause.
    assert all(f["detail"] == error for f in frames if f.get("status") == "failed")
    # No decision was reached: no result cards, and the prospect isn't concluded.
    assert not [f for f in frames if f["type"] == "visualization"]
    assert "s1" not in service._concluded


async def test_escalation_shows_the_handoff_phase_while_the_brief_is_composed():
    """Composing the specialist brief (escalation_triage) is the LONGEST call in an
    escalation turn -- measured at ~14s against the real agent. Without a step of
    its own the panel sits fully ticked while it runs, so it gets a breadcrumb like
    any other tool, marked as the handoff phase rather than a fifth pipeline step.
    The decision step also reports that it escalated, so the handoff reads as a
    consequence."""
    events = [
        *_tool_pair(
            "recommend_or_escalate",
            response={"ok": True, "requires_human_review": True},
        ),
        # The AgentTool returns prose, not an {"ok": ...} dict.
        *_tool_pair("escalation_triage", response="SITUATION\nNew prospect, 90 cases..."),
        _FakeEvent(
            calls=[_FakeCall("adk_request_input", id="req-1", args={"message": "Confirm?"})],
            long_running=["req-1"],
        ),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "1200 McKinney St, 90 cases"))

    tools = [f for f in frames if f["type"] == "tool"]
    triage = [f for f in tools if f["name"] == "escalation_triage"]
    # Shown while it runs, then settled -- not a silent gap.
    assert [f["status"] for f in triage] == ["running", "done"]
    assert triage[0]["label"] == "Briefing a specialist"
    # Marked as the handoff phase on BOTH frames, so the row keeps its treatment
    # once it settles.
    assert all(f["phase"] == "handoff" for f in triage)
    # ...and no assignment step is mistaken for one.
    assert not any("phase" in f for f in tools if f["name"] != "escalation_triage")
    # The decision step closes by restating the tool's own requires_human_review.
    decided = [
        f for f in tools if f["name"] == "recommend_or_escalate" and f["status"] == "done"
    ]
    assert decided and decided[0]["detail"] == "Escalating for human review."
    assert any(f["type"] == "await_input" for f in frames)


async def test_a_plain_recommendation_shows_no_handoff_phase():
    """No escalation, no handoff: the breadcrumbs stay the four assignment steps and
    the decision step keeps its generic description."""
    events = [
        *_tool_pair(
            "recommend_or_escalate",
            response={"ok": True, "requires_human_review": False},
        ),
        _FakeEvent(text="I recommend RTE-A on TUE."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "1200 McKinney St, 90 cases"))

    tools = [f for f in frames if f["type"] == "tool"]
    assert not any("phase" in f for f in tools)
    assert not any(f["name"] == "escalation_triage" for f in tools)
    # A successful, non-escalating close adds no detail to overwrite the original.
    assert not any("detail" in f for f in tools if f["status"] == "done")


async def test_a_failed_visualization_rebuild_does_not_kill_the_turn(monkeypatch):
    """The visualization re-derives a decision the agent has ALREADY narrated. If
    that rebuild raises, the turn must still complete -- otherwise the agent's own
    correct reply is discarded and the caller falls back to the deterministic brain,
    contradicting what the user was just told."""
    events = [
        *_tool_pair("recommend_or_escalate"),
        _FakeEvent(text="I recommend RTE-A on TUE."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )

    async def _boom(*args, **kwargs):
        raise RuntimeError("geocoder unavailable on the re-run")

    monkeypatch.setattr(service, "_visualization_from_state", _boom)

    frames = await _collect(service.stream_turn("s1", "1200 McKinney St, 90 cases"))

    assert any(f["type"] == "message" and "RTE-A" in f["text"] for f in frames)
    assert not [f for f in frames if f["type"] == "visualization"]
    assert frames[-1] == {"type": "done"}


async def test_breadcrumbs_are_not_duplicated_across_tool_calls():
    """If the user asks for an on-demand find_candidate_routes before the decision,
    the Geo-Lookup step must not appear twice when recommend_or_escalate (which also
    covers geo) runs -- each step is shown at most once per turn."""
    events = [
        *_tool_pair("intake_customer"),
        *_tool_pair("find_candidate_routes"),
        *_tool_pair("recommend_or_escalate"),
        _FakeEvent(text="Here is my recommendation."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "show me the routes then decide"))

    labels = [f["label"] for f in frames if f["type"] == "tool" and f.get("status") == "running"]
    assert labels == ["Intake", "Geo-Lookup", "Score & Rank", "Recommend / Decide"]
    # Geo-Lookup was opened by find_candidate_routes, so ITS response settles it --
    # the later recommend_or_escalate must not re-open or re-settle it.
    assert _steps(frames, "done").count(("find_candidate_routes", "done")) == 1


async def test_stream_turn_tool_frames_carry_plain_language_detail():
    """Each tool frame carries a ``detail`` breadcrumb for the UI stepper, and
    Intake echoes the customer's own inputs back (grounded, not invented)."""
    events = [
        *_tool_pair("intake_customer", args={
            "order_quantity_cases": 90, "preferred_day": "TUE",
        }),
        *_tool_pair("find_candidate_routes"),
        *_tool_pair("evaluate_and_score_routes"),
        *_tool_pair("recommend_or_escalate"),
        _FakeEvent(text="Here is my recommendation."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "New prospect at 1200 McKinney St, 90 cases"))

    # The opening frame carries the wording; the closing one only what changed.
    tools = {f["name"]: f for f in frames if f["type"] == "tool" and f["status"] == "running"}
    # Every step has a detail line...
    assert all("detail" in f for f in tools.values())
    # ...and Intake reads back the stated order size + day.
    assert "90 cases" in tools["intake_customer"]["detail"]
    assert "TUE" in tools["intake_customer"]["detail"]
    # A successful close adds no new wording to overwrite it.
    done = [f for f in frames if f["type"] == "tool" and f["status"] == "done"]
    assert done and not any("detail" in f for f in done)


async def test_stream_turn_requests_non_streaming_mode():
    """The service must run the model in NON-streaming mode (StreamingMode.NONE),
    exactly like ``adk web``'s default. Forcing token streaming (SSE) tripped a
    LiteLLM async-streaming bug on some backends ("'coroutine' object is not an
    iterator") that made every turn fail and fall back to the deterministic brain.
    The browser SSE stream is emitted by ``stream_turn`` itself and does not need
    model token streaming, so NONE loses nothing."""
    from google.adk.agents.run_config import StreamingMode

    runner = _FakeRunner([[_FakeEvent(text="ok")]])
    service = LlmChatService(
        runner=runner, session_service=_FakeSessionService({}), geocoder=MockGeocoder()
    )
    await _collect(service.stream_turn("s1", "hello"))
    assert runner.run_configs and runner.run_configs[0] is not None
    assert runner.run_configs[0].streaming_mode == StreamingMode.NONE


async def test_stream_turn_partial_text_is_not_emitted():
    events = [_FakeEvent(text="partial chunk", partial=True), _FakeEvent(text="final answer")]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService({}),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "hello"))
    messages = [f["text"] for f in frames if f["type"] == "message"]
    assert messages == ["final answer"]


async def test_stream_turn_human_in_the_loop_then_resume():
    call = _FakeCall("adk_request_input", id="req-1", args={"message": "Please confirm this slot."})
    first = [_FakeEvent(calls=[call], long_running=["req-1"])]
    second = [_FakeEvent(text="Thanks, confirmed.")]
    runner = _FakeRunner([first, second])
    service = LlmChatService(
        runner=runner, session_service=_FakeSessionService({}), geocoder=MockGeocoder()
    )

    frames1 = await _collect(service.stream_turn("s1", "Assign a slot"))
    await_frames = [f for f in frames1 if f["type"] == "await_input"]
    assert await_frames and await_frames[0]["message"] == "Please confirm this slot."
    assert "s1" in service._pending_input

    # The next turn must resume via a FunctionResponse carrying the same call id.
    frames2 = await _collect(service.stream_turn("s1", "Yes, go ahead"))
    resume_msg = runner.messages[1]
    fr = resume_msg.parts[0].function_response
    assert fr.id == "req-1"
    assert fr.response == {"result": "Yes, go ahead"}
    assert "s1" not in service._pending_input
    assert any(f["type"] == "message" for f in frames2)


async def test_new_prospect_after_conclusion_rotates_to_a_fresh_session():
    """A second full prospect (one that carries a street address), entered after
    the first concluded, runs in a FRESH underlying ADK session -- so the previous
    prospect's history/state can't bleed into it."""
    turn1 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="First result.")]
    turn2 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="Second result.")]
    svc = LlmChatService(
        runner=_FakeRunner([turn1, turn2]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    await _collect(svc.stream_turn("s1", "5085 Westheimer Rd, Houston, TX 77056, 90 cases"))
    assert "s1" in svc._concluded  # first prospect reached a decision
    await _collect(svc.stream_turn("s1", "1200 McKinney St, Houston, TX 77010, 400 cases"))
    assert svc._generation["s1"] == 1  # rotated
    assert svc._session_service.created == ["s1", "s1#1"]  # a new ADK session


async def test_revision_after_conclusion_stays_in_the_same_session():
    """A revision (no new address, e.g. 'try 20 cases') keeps the same session so
    multi-turn context is preserved -- it must NOT rotate."""
    turn1 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="Result.")]
    turn2 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="Revised.")]
    svc = LlmChatService(
        runner=_FakeRunner([turn1, turn2]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    await _collect(svc.stream_turn("s1", "5085 Westheimer Rd, Houston, TX 77056, 90 cases"))
    await _collect(svc.stream_turn("s1", "try 20 cases"))
    assert svc._generation.get("s1", 0) == 0  # no rotation
    assert svc._session_service.created == ["s1"]  # same session reused


async def test_new_prospect_after_escalation_is_not_misrouted_as_a_resume():
    """After an escalation leaves a pending request_input, a NEW prospect must NOT
    be consumed as the specialist's reply -- it starts fresh, pending cleared."""
    call = _FakeCall("adk_request_input", id="req-1", args={"message": "Confirm?"})
    turn1 = [*_tool_pair("recommend_or_escalate"),
             _FakeEvent(calls=[call], long_running=["req-1"])]
    turn2 = [_FakeEvent(text="Second prospect handled.")]
    runner = _FakeRunner([turn1, turn2])
    svc = LlmChatService(
        runner=runner, session_service=_FakeSessionService(_SAMPLE_STATE), geocoder=MockGeocoder()
    )
    await _collect(svc.stream_turn("s1", "5085 Westheimer Rd, Houston, TX 77056, 90 cases"))
    assert "s1" in svc._pending_input  # escalation left a pending request_input
    await _collect(svc.stream_turn("s1", "1200 McKinney St, Houston, TX 77010, 400 cases"))
    second = runner.messages[1]
    assert second.parts[0].text  # a fresh text message, not a resume
    assert getattr(second.parts[0], "function_response", None) is None
    assert "s1" not in svc._pending_input  # pending cleared by the rotation


async def test_recommendation_narration_shown_in_visualization():
    """The agent's own recommendation narration (what it says AFTER calling
    recommend_or_escalate) is rendered in the result card's 'Why the agent chose
    this', so the panel matches the chat box word-for-word. Text emitted BEFORE
    the recommendation (e.g. a Score & Rank summary) must not leak into it."""
    events = [
        *_tool_pair("evaluate_and_score_routes"),
        _FakeEvent(text="All three routes look feasible with this order."),
        *_tool_pair("recommend_or_escalate"),
        _FakeEvent(text="I recommend RTE-A on TUE, 07:20-10:20 — tight fit and an open slot."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "what is your rec"))
    viz = [f for f in frames if f["type"] == "visualization"]
    assert len(viz) == 1
    html = viz[0]["payload"]["resultHtml"]
    # The exact narration the chat box showed is what the panel shows.
    assert "I recommend RTE-A on TUE, 07:20-10:20" in html
    # A pre-recommendation summary is NOT captured as the recommendation reasoning.
    assert "All three routes look feasible" not in html


async def test_visualization_none_when_profile_incomplete():
    service = LlmChatService(
        runner=_FakeRunner([[]]),
        session_service=_FakeSessionService({_STATE_PROFILE_KEY: {"address": ""}}),
        geocoder=MockGeocoder(),
    )
    assert await service._visualization_from_state("s1") is None


# --- the decision is made ONCE per turn --------------------------------------
#
# Step 5 can sample (grounded reasoning), so re-deciding for the visualization
# would show the user a second, possibly different outcome underneath the
# agent's narration of the first -- and record THAT one for feedback/tracing.


def _decision_state(recommendation, profile=None):
    """Session state carrying a profile plus a decision snapshot bound to it."""
    profile = profile if profile is not None else dict(_SAMPLE_STATE[_STATE_PROFILE_KEY])
    return {
        _STATE_PROFILE_KEY: profile,
        _STATE_LAST_DECISION_KEY: {
            "profile": dict(profile),
            "recommendation": recommendation.to_state_dict(),
        },
    }


def _counting_decider(monkeypatch):
    """Count how many times step 5 actually runs, keeping real behaviour."""
    import smart_assignment.routeslot as routeslot

    calls = []
    real = routeslot.decide_route_slot

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(routeslot, "decide_route_slot", _spy)
    return calls


async def test_cached_decision_is_reused_instead_of_re_deciding(monkeypatch):
    calls = _counting_decider(monkeypatch)
    # A decision the agent already made -- deliberately an ESCALATION, which a
    # fresh run of this healthy downtown prospect would NOT produce.
    cached = SlotRecommendation(
        customer_name="Test Prospect",
        decision=Decision.ESCALATED_LOW_SCORE,
        total_score=0.11,
        reasoning="Cached decision from the agent's own turn.",
        recommended_route_id="RTE-4100",
        recommended_route_name="Central Houston",
        recommended_day="TUE",
        recommended_window="07:20-10:20",
        review_reason="Cached escalation.",
    )
    service = LlmChatService(
        runner=_FakeRunner([[]]),
        session_service=_FakeSessionService(_decision_state(cached)),
        geocoder=MockGeocoder(),
    )
    payload = await service._visualization_from_state("s1")

    assert payload is not None
    assert not calls, "step 5 must not run again when a valid decision is cached"
    # The card -- and the feedback/trace context recorded for it -- carry the
    # CACHED outcome, not a freshly-sampled one.
    assert payload["_decision"]["outcome"] == "escalate"
    assert payload["_decision"]["review_reason"] == "Cached escalation."


async def test_stale_cached_decision_is_ignored_and_the_decision_is_recomputed(monkeypatch):
    calls = _counting_decider(monkeypatch)
    cached = SlotRecommendation(
        customer_name="Someone Else",
        decision=Decision.ESCALATED_LOW_SCORE,
        total_score=0.11,
        reasoning="Belongs to a different prospect.",
    )
    # Snapshot bound to a DIFFERENT profile than the one now in state.
    state = _decision_state(cached, profile=dict(_SAMPLE_STATE[_STATE_PROFILE_KEY]))
    state[_STATE_LAST_DECISION_KEY]["profile"]["order_quantity_cases"] = 400

    service = LlmChatService(
        runner=_FakeRunner([[]]),
        session_service=_FakeSessionService(state),
        geocoder=MockGeocoder(),
    )
    payload = await service._visualization_from_state("s1")

    assert payload is not None
    assert len(calls) == 1, "a stale snapshot must be ignored and the decision recomputed"
    # The recomputed decision wins, not the stale escalation.
    assert payload["_decision"]["outcome"] == "recommend"


async def test_decision_is_computed_once_when_nothing_is_cached(monkeypatch):
    calls = _counting_decider(monkeypatch)
    service = LlmChatService(
        runner=_FakeRunner([[]]),
        session_service=_FakeSessionService(dict(_SAMPLE_STATE)),
        geocoder=MockGeocoder(),
    )
    payload = await service._visualization_from_state("s1")

    assert payload is not None
    assert len(calls) == 1, "with no snapshot the decision runs exactly once"


@pytest.mark.parametrize(
    "pick,escalation",
    [(False, False), (True, False), (False, True), (True, True)],
)
async def test_cached_decision_is_reused_under_every_grounded_config(
    monkeypatch, pick, escalation
):
    """Reuse must not depend on how the decision was originally reached.

    run_slot_recommendation skips step 5 entirely when a decision is supplied, so
    this holds by construction -- pinned across all four flag combinations so a
    future change to the decision layer can't quietly reintroduce a second call
    (and, with grounded reasoning on, a second LLM round-trip) per turn."""
    import smart_assignment.webapp.llm_chat as llm_chat_module

    calls = _counting_decider(monkeypatch)
    monkeypatch.setattr(
        llm_chat_module,
        "DEFAULT_CONFIG",
        Config(
            use_grounded_route_slot_pick=pick,
            use_grounded_route_slot_escalation=escalation,
        ),
    )
    cached = SlotRecommendation(
        customer_name="Test Prospect",
        decision=Decision.RECOMMENDED,
        total_score=0.79,
        reasoning="Cached decision from the agent's own turn.",
        recommended_route_id="RTE-4100",
        recommended_window="07:20-10:20",
    )
    service = LlmChatService(
        runner=_FakeRunner([[]]),
        session_service=_FakeSessionService(_decision_state(cached)),
        geocoder=MockGeocoder(),
    )
    payload = await service._visualization_from_state("s1")

    assert payload is not None
    assert not calls, f"step 5 ran again with pick={pick}, escalation={escalation}"
    assert payload["_decision"]["outcome"] == "recommend"
