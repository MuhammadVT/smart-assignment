"""
Phase 2 — the LLM-conversational brain behind the chat box.

Drives the *real* ADK ``root_agent`` (smart_assignment/agent.py) via an ADK
``Runner``, so the chat is genuine multi-turn natural language: the model
collects intake conversationally, decides when to call each pipeline tool,
handles revisions, and escalates to a human via ADK's ``request_input`` tool.
Each turn is streamed to the browser as Server-Sent Events.

The step-by-step **visualization is not re-implemented**. Once the agent has a
complete profile in session state (``sa_profile``), we rebuild a
``CustomerProfile`` and re-run the deterministic pipeline to produce the exact
same payload the published Simulator uses (``build_workflow_payload``) — so what
the browser animates can never drift from the numbers the agent computed.

Mode + credentials
------------------
``SMART_ASSIGNMENT_WEBAPP_MODE`` selects the brain: ``llm`` (default — Phase 2)
or ``deterministic`` (Phase 1). When ``llm`` is configured but no LLM
credentials are detected for the active backend, the app transparently falls
back to deterministic mode so it still runs offline with no key — Phase 2 is
the default that activates the moment credentials are present.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.reporting.page import build_workflow_payload
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.geo import Geocoder
from smart_assignment.tools.slot_recommendation import (
    _GEOCODER,
    _STATE_PROFILE_KEY,
    _profile_from_state_dict,
)

_APP_NAME = "smart_assignment_webapp"
_USER_ID = "webapp_user"

# ADK's request_input long-running tool surfaces under this function name.
_REQUEST_INPUT_NAME = "adk_request_input"

# Map each pipeline tool the agent calls to the visualization step it drives, so
# the UI can show live progress breadcrumbs before the full cards animate.
_TOOL_STEPS = {
    "intake_customer": "Intake",
    "find_candidate_routes": "Geo-Lookup",
    "evaluate_and_score_routes": "Score & Rank",
    "recommend_or_escalate": "Recommend / Decide",
}


# ---------------------------------------------------------------------------
# Mode + credential resolution (cheap, no network, no runner build)
# ---------------------------------------------------------------------------


def webapp_mode() -> str:
    """Configured brain: 'llm' (default, Phase 2) or 'deterministic' (Phase 1)."""
    return os.environ.get("SMART_ASSIGNMENT_WEBAPP_MODE", "llm").strip().lower()


def llm_credentials_available(config: Config) -> bool:
    """True when the active LLM backend has the credentials it needs to run.

    Mirrors the checks in shared/llm.py without importing anything heavy:
    - sage      → the three SAGE_* vars must be set.
    - standard  → a litellm "<provider>/<model>" is assumed configured (its
                  provider key lives elsewhere); a bare Gemini name needs
                  GOOGLE_API_KEY or Vertex (GOOGLE_GENAI_USE_VERTEXAI).
    """
    if config.llm_backend == "sage":
        return all(
            os.environ.get(v) for v in ("SAGE_CLIENT_ID", "SAGE_CLIENT_SECRET", "SAGE_ENVIRONMENT")
        )
    if "/" in config.model:  # litellm-style provider/model
        return True
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))


def resolve_mode(config: Optional[Config] = None) -> dict:
    """Effective mode the app will serve.

    Returns ``{"mode", "configured"}``. Unless the operator explicitly asks for
    deterministic mode (``SMART_ASSIGNMENT_WEBAPP_MODE=deterministic``), the chat
    drives the **real ADK agent** -- exactly what ``adk web`` does -- so the
    web-app conversation behaves identically (free-form Q&A, the escalation-triage
    handoff, revisions, etc.).

    We deliberately do NOT pre-guess credential availability and downgrade to the
    Phase-1 parser here: that heuristic can disagree with what the agent actually
    needs (e.g. Vertex via a service account) and wrongly strand the app on the
    parser even when ``adk web`` works. If the agent genuinely can't run (no
    credentials, a model/network error), the ``/api/chat`` stream falls back to a
    deterministic result per turn (see ``app.chat``), so the chat never dead-ends.
    """
    config = config or DEFAULT_CONFIG
    configured = webapp_mode()
    if configured != "llm":
        return {"mode": "deterministic", "configured": configured}
    return {"mode": "llm", "configured": "llm"}


# ---------------------------------------------------------------------------
# The conversational service
# ---------------------------------------------------------------------------


class LlmChatService:
    """Runs the ADK agent for one browser conversation at a time (keyed by a
    client-supplied session_id) and yields SSE-ready frame dicts.

    ``runner``/``session_service``/``geocoder`` are injectable so tests can drive
    the streaming logic with a fake agent and an offline geocoder; in production
    they default to a real ADK ``Runner`` over ``root_agent`` and the same
    ``CensusGeocoder`` the tools use (so the visualization matches the agent).
    """

    def __init__(self, runner=None, session_service=None, geocoder: Optional[Geocoder] = None):
        self._runner = runner
        self._session_service = session_service
        self._geocoder = geocoder or _GEOCODER
        self._known_sessions: set[str] = set()
        # session_id -> {"id", "name"} of a pending request_input call to resume.
        self._pending_input: dict[str, dict] = {}

    # -- lazy ADK wiring (never built until a live turn actually needs it) --

    def _get_session_service(self):
        if self._session_service is None:
            from google.adk.sessions import InMemorySessionService

            self._session_service = InMemorySessionService()
        return self._session_service

    def _get_runner(self):
        if self._runner is None:
            from google.adk.runners import Runner

            from smart_assignment.agent import root_agent

            self._runner = Runner(
                agent=root_agent,
                app_name=_APP_NAME,
                session_service=self._get_session_service(),
            )
        return self._runner

    async def _ensure_session(self, session_id: str) -> None:
        if session_id in self._known_sessions:
            return
        await self._get_session_service().create_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
        )
        self._known_sessions.add(session_id)

    def _build_message(self, session_id: str, message: str):
        """A resume FunctionResponse if a request_input is pending, else text."""
        from google.genai import types

        pending = self._pending_input.pop(session_id, None)
        if pending:
            return types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=pending["id"],
                            name=pending["name"],
                            response={"result": message},
                        )
                    )
                ],
            )
        return types.Content(role="user", parts=[types.Part(text=message)])

    async def _visualization_from_state(self, session_id: str) -> Optional[dict]:
        """Rebuild the profile from session state and produce the Simulator
        payload by re-running the deterministic pipeline (drift-free)."""
        session = await self._get_session_service().get_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
        )
        state = (session.state if session else None) or {}
        profile = state.get(_STATE_PROFILE_KEY)
        if not profile or not profile.get("address") or not profile.get("order_quantity_cases"):
            return None
        customer = _profile_from_state_dict(profile)
        result = run_slot_recommendation(
            customer,
            config=DEFAULT_CONFIG,
            geocoder=self._geocoder,
            reasoner=DeterministicReasoner(),
        )
        return build_workflow_payload(result, DEFAULT_CONFIG)

    # -- the turn stream --

    async def stream_turn(self, session_id: str, message: str) -> AsyncGenerator[dict, None]:
        """Run one conversational turn, yielding frame dicts:

        ``{"type": "tool", "name", "label"}``      — a pipeline tool was called
        ``{"type": "message", "text"}``            — agent natural-language reply
        ``{"type": "await_input", "message"}``     — human-in-the-loop escalation
        ``{"type": "visualization", "payload"}``   — the 5 step cards + result
        ``{"type": "done"}``                       — turn finished
        """
        await self._ensure_session(session_id)
        runner = self._get_runner()
        new_message = self._build_message(session_id, message)

        from google.adk.agents.run_config import RunConfig, StreamingMode

        saw_recommendation = False
        async for event in runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=new_message,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            # Human-in-the-loop: request_input surfaces as a long-running call.
            if getattr(event, "long_running_tool_ids", None):
                for fc in event.get_function_calls():
                    if fc.id in event.long_running_tool_ids:
                        self._pending_input[session_id] = {"id": fc.id, "name": fc.name}
                        prompt = (fc.args or {}).get("message") or (
                            "A routing specialist needs to confirm this before it's final."
                        )
                        yield {"type": "await_input", "message": prompt}
                continue

            calls = event.get_function_calls()
            if calls:
                for fc in calls:
                    label = _TOOL_STEPS.get(fc.name)
                    if label:
                        yield {"type": "tool", "name": fc.name, "label": label}
                        if fc.name == "recommend_or_escalate":
                            saw_recommendation = True
                continue

            if event.get_function_responses():
                continue  # tool return values drive the pipeline; nothing to show

            # Natural-language text. Emit only the aggregated (non-partial) event
            # so the transcript gets each reply once, not per streamed chunk.
            if event.content and event.content.parts and not getattr(event, "partial", False):
                text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
                if text.strip():
                    yield {"type": "message", "text": text.strip()}

        if saw_recommendation:
            payload = await self._visualization_from_state(session_id)
            if payload:
                yield {"type": "visualization", "payload": payload}

        yield {"type": "done"}
