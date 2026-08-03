"""
Tests for the opt-in cross-prospect *session memory* (Config.use_session_memory).

Covers the three seams the feature touches -- the config flag, the gated
``preload_memory`` tool on root_agent, and the memory wiring/scoping/ingest in
``webapp/llm_chat.py`` -- and, above all, pins the guarantee that with the flag
OFF (the default) nothing about today's behavior changes: no memory service is
built, no tool is added, and the fixed webapp user_id is used exactly as before.

Like ``test_llm_chat.py`` these stay offline: the real ADK agent needs LLM
credentials, so the streaming logic is driven with fakes (a fake runner, a fake
session service, a recording memory service) and an offline geocoder.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.shared.config import Config
import smart_assignment.webapp.llm_chat as llm_chat_module
from smart_assignment.webapp.llm_chat import _USER_ID, LlmChatService


# --- Fakes ------------------------------------------------------------------


class _FakeCall:
    def __init__(self, name, id="fc1", args=None):
        self.name = name
        self.id = id
        self.args = args or {}


class _FakeResponse:
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
    def __init__(self, calls=None, responses=None, text=None):
        self._calls = calls or []
        self._responses = responses or []
        self.long_running_tool_ids = None
        self.partial = False
        self.content = _FakeContent(text) if text is not None else None

    def get_function_calls(self):
        return self._calls

    def get_function_responses(self):
        return self._responses


def _tool_pair(name, id="fc1"):
    """The call + response pair ADK emits for one tool invocation. A prospect only
    counts as concluded once the tool REPORTS a decision, so both are needed."""
    return [
        _FakeEvent(calls=[_FakeCall(name, id=id)]),
        _FakeEvent(responses=[_FakeResponse(name, id=id)]),
    ]


class _FakeSession:
    """Carries the fields ``add_session_to_memory`` reads plus ``.state`` for the
    visualization rebuild. ``id``/``user_id`` are stamped by the service so a test
    can assert WHICH conversation was folded into memory and under which user."""

    def __init__(self, *, app_name, user_id, session_id, state, events=None):
        self.app_name = app_name
        self.user_id = user_id
        self.id = session_id
        self.state = state
        self.events = events or []


class _FakeSessionService:
    """Records create/get calls (with their user_id) and returns sessions that
    carry events, so the memory-ingest path can be exercised end to end."""

    def __init__(self, state=None):
        self._state = state or {}
        self.created = []  # list of (session_id, user_id)

    async def create_session(self, *, app_name, user_id, session_id, state=None):
        self.created.append((session_id, user_id))
        return _FakeSession(
            app_name=app_name, user_id=user_id, session_id=session_id, state=self._state
        )

    async def get_session(self, *, app_name, user_id, session_id):
        return _FakeSession(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=self._state,
            events=[_FakePart("earlier turn")],  # non-empty so the fold is meaningful
        )


class _RecordingMemoryService:
    def __init__(self):
        self.added = []  # sessions folded in via add_session_to_memory

    async def add_session_to_memory(self, session):
        self.added.append(session)


class _FakeRunner:
    """Yields a pre-scripted batch of events per run_async call and records the
    user_id each call ran under (so memory scoping can be asserted)."""

    def __init__(self, batches):
        self._batches = [list(b) for b in batches]
        self.user_ids = []

    async def run_async(self, *, user_id, session_id, new_message, run_config=None):
        self.user_ids.append(user_id)
        batch = self._batches.pop(0) if self._batches else []
        for event in batch:
            yield event


async def _collect(agen):
    return [frame async for frame in agen]


# --- The config flag --------------------------------------------------------


def test_session_memory_off_by_default():
    assert Config().use_session_memory is False


def test_session_memory_flag_reads_from_env():
    with patch.dict(os.environ, {"SMART_ASSIGNMENT_USE_SESSION_MEMORY": "true"}):
        assert Config.from_env().use_session_memory is True
    with patch.dict(os.environ, {"SMART_ASSIGNMENT_USE_SESSION_MEMORY": "false"}):
        assert Config.from_env().use_session_memory is False


# --- The gated preload_memory tool on root_agent ----------------------------


def _build_agent_with(monkeypatch, **flags):
    """Build root_agent offline: stub get_llm (so no backend/credentials) and pin
    a Config. Triage/address-resolution are off here so only the base tools plus
    (maybe) preload_memory are present, keeping the assertion about the flag clean."""
    import smart_assignment.agent as agent_mod

    monkeypatch.setattr(agent_mod, "get_llm", lambda cfg: "fake-model")
    monkeypatch.setattr(
        agent_mod,
        "DEFAULT_CONFIG",
        Config(use_escalation_triage=False, use_address_resolution=False, **flags),
    )
    return agent_mod._build_root_agent()


def test_preload_memory_tool_absent_when_flag_off(monkeypatch):
    agent = _build_agent_with(monkeypatch, use_session_memory=False)
    assert "preload_memory" not in {t.name for t in agent.tools}


def test_preload_memory_tool_present_when_flag_on(monkeypatch):
    agent = _build_agent_with(monkeypatch, use_session_memory=True)
    assert "preload_memory" in {t.name for t in agent.tools}


# --- Memory wiring / scoping (flag OFF = today's behavior, exactly) ----------


def test_memory_service_is_none_when_flag_off(monkeypatch):
    # Even an explicitly INJECTED memory service is ignored while the flag is off,
    # so flag-off can never wire memory onto the Runner.
    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=False))
    svc = LlmChatService(memory_service=_RecordingMemoryService(), geocoder=MockGeocoder())
    assert svc._get_memory_service() is None


def test_memory_service_built_when_flag_on(monkeypatch):
    from google.adk.memory import InMemoryMemoryService

    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=True))
    svc = LlmChatService(geocoder=MockGeocoder())
    assert isinstance(svc._get_memory_service(), InMemoryMemoryService)


def test_user_id_is_fixed_when_flag_off(monkeypatch):
    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=False))
    svc = LlmChatService(geocoder=MockGeocoder())
    assert svc._user_id_for("browser-123") == _USER_ID


def test_user_id_is_browser_session_when_flag_on(monkeypatch):
    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=True))
    svc = LlmChatService(geocoder=MockGeocoder())
    # The browser session id becomes the ADK user_id, scoping memory per browser.
    assert svc._user_id_for("browser-123") == "browser-123"


async def test_run_async_uses_fixed_user_id_when_flag_off(monkeypatch):
    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=False))
    runner = _FakeRunner([[_FakeEvent(text="ok")]])
    svc = LlmChatService(
        runner=runner, session_service=_FakeSessionService({}), geocoder=MockGeocoder()
    )
    await _collect(svc.stream_turn("s1", "hello"))
    assert runner.user_ids == [_USER_ID]
    assert svc._session_service.created == [("s1", _USER_ID)]


async def test_run_async_uses_browser_session_as_user_when_flag_on(monkeypatch):
    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=True))
    runner = _FakeRunner([[_FakeEvent(text="ok")]])
    svc = LlmChatService(
        runner=runner, session_service=_FakeSessionService({}), geocoder=MockGeocoder()
    )
    await _collect(svc.stream_turn("s1", "hello"))
    assert runner.user_ids == ["s1"]
    assert svc._session_service.created == [("s1", "s1")]


# --- The ingest-on-rotation behavior ----------------------------------------


async def test_concluding_prospect_is_folded_into_memory_on_rotation(monkeypatch):
    """With the flag on, when a NEW prospect (new address) follows a concluded one,
    the concluding ADK conversation is added to memory BEFORE the fresh session is
    minted -- so its facts survive the rotation. Memory is scoped to the browser
    session (user_id == 's1')."""
    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=True))
    turn1 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="First result.")]
    turn2 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="Second result.")]
    memory = _RecordingMemoryService()
    svc = LlmChatService(
        runner=_FakeRunner([turn1, turn2]),
        session_service=_FakeSessionService({}),  # empty state -> no pipeline run
        geocoder=MockGeocoder(),
        memory_service=memory,
    )

    await _collect(svc.stream_turn("s1", "5085 Westheimer Rd, Houston, TX 77056, 90 cases"))
    assert "s1" in svc._concluded
    assert memory.added == []  # nothing folded yet -- still the first prospect

    await _collect(svc.stream_turn("s1", "1200 McKinney St, Houston, TX 77010, 400 cases"))
    assert svc._generation["s1"] == 1  # rotated
    # Exactly the concluding conversation (the pre-rotation id 's1'), scoped to the
    # browser as user 's1', was folded into memory.
    assert len(memory.added) == 1
    folded = memory.added[0]
    assert folded.id == "s1"
    assert folded.user_id == "s1"
    # And both prospects' ADK sessions were created under the browser-scoped user.
    assert svc._session_service.created == [("s1", "s1"), ("s1#1", "s1")]


async def test_no_memory_ingest_when_flag_off(monkeypatch):
    """The same rotation with the flag OFF folds nothing into memory (even an
    injected service is never touched) and keeps the fixed user_id -- i.e. today's
    behavior is reproduced exactly."""
    monkeypatch.setattr(llm_chat_module, "DEFAULT_CONFIG", Config(use_session_memory=False))
    turn1 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="First result.")]
    turn2 = [*_tool_pair("recommend_or_escalate"), _FakeEvent(text="Second result.")]
    memory = _RecordingMemoryService()
    svc = LlmChatService(
        runner=_FakeRunner([turn1, turn2]),
        session_service=_FakeSessionService({}),
        geocoder=MockGeocoder(),
        memory_service=memory,
    )

    await _collect(svc.stream_turn("s1", "5085 Westheimer Rd, Houston, TX 77056, 90 cases"))
    await _collect(svc.stream_turn("s1", "1200 McKinney St, Houston, TX 77010, 400 cases"))

    assert svc._generation["s1"] == 1  # still rotates -- rotation itself is unchanged
    assert memory.added == []  # ...but nothing is ever folded into memory
    assert svc._session_service.created == [("s1", _USER_ID), ("s1#1", _USER_ID)]
