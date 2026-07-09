"""
Tests for the Phase 2 LLM-conversational service (smart_assignment/webapp/llm_chat.py).

The real ADK agent needs LLM credentials, so these drive the streaming logic with
a FAKE runner + session service and an offline geocoder — exercising event→frame
mapping, the visualization rebuilt from session state, human-in-the-loop resume,
and the mode/credential resolution. No network, no key.
"""

from __future__ import annotations

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.shared.config import Config
from smart_assignment.tools.slot_recommendation import _STATE_PROFILE_KEY
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

    async def run_async(self, *, user_id, session_id, new_message, run_config=None):
        self.messages.append(new_message)
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


def test_resolve_mode_downgrades_without_credentials(monkeypatch):
    monkeypatch.setenv("SMART_ASSIGNMENT_WEBAPP_MODE", "llm")
    cfg = Config(llm_backend="standard", model="gemini-2.5-flash")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    res = resolve_mode(cfg)
    assert res["mode"] == "deterministic"
    assert res["configured"] == "llm"
    assert "reason" in res


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
        _FakeEvent(calls=[_FakeCall("intake_customer")]),
        _FakeEvent(calls=[_FakeCall("find_candidate_routes")]),
        _FakeEvent(calls=[_FakeCall("evaluate_and_score_routes")]),
        _FakeEvent(calls=[_FakeCall("recommend_or_escalate")]),
        _FakeEvent(text="Here is my recommendation."),
    ]
    service = LlmChatService(
        runner=_FakeRunner([events]),
        session_service=_FakeSessionService(_SAMPLE_STATE),
        geocoder=MockGeocoder(),
    )
    frames = await _collect(service.stream_turn("s1", "New prospect at 1200 McKinney St, 90 cases"))

    tool_labels = [f["label"] for f in frames if f["type"] == "tool"]
    assert tool_labels == ["Intake", "Geo-Lookup", "Score & Rank", "Recommend / Decide"]
    assert any(f["type"] == "message" and "recommendation" in f["text"] for f in frames)
    viz = [f for f in frames if f["type"] == "visualization"]
    assert len(viz) == 1
    assert len(viz[0]["payload"]["steps"]) == 5
    assert frames[-1] == {"type": "done"}


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


async def test_visualization_none_when_profile_incomplete():
    service = LlmChatService(
        runner=_FakeRunner([[]]),
        session_service=_FakeSessionService({_STATE_PROFILE_KEY: {"address": ""}}),
        geocoder=MockGeocoder(),
    )
    assert await service._visualization_from_state("s1") is None
