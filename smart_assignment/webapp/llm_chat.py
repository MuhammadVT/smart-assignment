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

import logging
import os
from typing import Any, AsyncGenerator, Optional

from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reporting.page import build_workflow_payload
from smart_assignment.webapp.decision import traced_decision
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.geo import Geocoder
from smart_assignment.shared.llm import offload_to_worker_thread
from smart_assignment.triage.formatting import normalize_brief
from smart_assignment.tools.slot_recommendation import (
    _GEOCODER,
    _STATE_PROFILE_KEY,
    _profile_from_state_dict,
    cached_decision_for,
)
from smart_assignment.webapp.narration import (
    ESCALATION_DETAIL,
    step_detail,
    step_label,
    step_phase,
    tool_steps,
)
from smart_assignment.webapp.parse import parse_intake

logger = logging.getLogger(__name__)

_APP_NAME = "smart_assignment_webapp"
_USER_ID = "webapp_user"

# ADK's request_input long-running tool surfaces under this function name.
_REQUEST_INPUT_NAME = "adk_request_input"

# Tools whose successful return IS the prospect's decision, so the turn should
# render the result visualization and mark the prospect concluded.
_DECISION_TOOLS = ("recommend_or_escalate", "assign_prospect")


class AgentTurnUnavailable(RuntimeError):
    """The agent's turn ended in a *recovered* model failure having produced no
    decision, so it has nothing to show.

    ``agent_callbacks`` deliberately stops such a failure from unwinding the
    Runner -- that is what keeps a turn whose pipeline already succeeded from
    being thrown away. But when nothing was produced, silence would be a
    regression: before recovery existed the exception reached ``app.chat``, which
    answered from the deterministic brain. Raising this preserves that floor, so
    recovery is never worse than the deterministic baseline -- only better when
    there is real work to protect.
    """


def _event_text(event: Any) -> str:
    """The aggregated text of an ADK event's parts, stripped ("" when there is
    none). Keeps the two text-reading branches below reading identically."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(p.text for p in parts if getattr(p, "text", None)).strip()


def _call_key(part: Any) -> str:
    """Correlation key pairing a FunctionCall with its FunctionResponse. ADK sets
    a matching ``id`` on both; fall back to the tool name so a backend that omits
    the id still pairs correctly for the one-call-at-a-time flow."""
    return getattr(part, "id", None) or part.name


def _requires_human_review(part: Any) -> bool:
    """Whether a decision tool reported that this prospect needs a human.

    Read straight off ``requires_human_review`` on the tool's own result, so the
    breadcrumb restates a fact the audited decision produced. False for any other
    tool, or a payload we can't read."""
    if part.name not in _DECISION_TOOLS:
        return False
    response = part.response
    return isinstance(response, dict) and bool(response.get("requires_human_review"))


def _tool_outcome(response: Any) -> tuple[bool, Optional[str]]:
    """Did a tool call succeed, and -- if not -- what did it say went wrong?

    The pipeline tools return a plain ``{"ok": bool, "error": str, ...}`` dict,
    which ADK hands back verbatim on the FunctionResponse, so this reads a real
    fact the tool reported rather than assuming an outcome from the call.

    Anything unreadable (a non-dict payload, or a tool we don't own such as the
    ``escalation_triage`` AgentTool) counts as success: this only drives
    breadcrumb wording, and treating a finished step as failed -- or leaving it
    spinning forever -- would be worse than the benign default.
    """
    if not isinstance(response, dict) or "ok" not in response:
        return True, None
    if response.get("ok"):
        return True, None
    error = response.get("error")
    return False, error.strip() if isinstance(error, str) and error.strip() else None


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

    def __init__(
        self,
        runner=None,
        session_service=None,
        geocoder: Optional[Geocoder] = None,
        memory_service=None,
    ):
        self._runner = runner
        self._session_service = session_service
        # The ADK memory service backing opt-in cross-prospect recall. Only built
        # (and only passed to the Runner) when Config.use_session_memory is on --
        # with the flag off it stays None and the Runner behaves exactly as before.
        self._memory_service = memory_service
        self._geocoder = geocoder or _GEOCODER
        self._known_sessions: set[str] = set()
        # browser session_id -> {"id", "name"} of a pending request_input to resume.
        self._pending_input: dict[str, dict] = {}
        # A browser session can hold many prospects one after another. When the
        # NEXT full prospect arrives after the current one concluded/escalated,
        # the underlying ADK conversation is rotated (a generation counter
        # suffixes the ADK session id; the browser session_id never changes) so
        # the transcript starts clean and any pending request_input can't
        # misroute the new prospect as the specialist's reply.
        #
        # Rotation is HYGIENE, not the correctness boundary. Cross-prospect
        # contamination is prevented one level down, in the tools every surface
        # shares: intake_customer resets the profile deterministically when a new
        # address arrives after a decision, and start_new_prospect lets the model
        # declare a switch rotation's parser can't see (see
        # tools/slot_recommendation.py -- adk web has no rotation at all and is
        # covered by the same guards). What rotation still buys here: a bounded,
        # per-prospect transcript (the sage backend folds recent history into its
        # system prompt), and clean pending-escalation bookkeeping.
        # ``_concluded`` marks a browser session whose current prospect already
        # reached a recommendation/escalation, so the NEXT full prospect triggers
        # a rotation (a mid-prospect revision does not).
        self._generation: dict[str, int] = {}
        self._concluded: set[str] = set()

    # -- lazy ADK wiring (never built until a live turn actually needs it) --

    def _get_session_service(self):
        if self._session_service is None:
            from google.adk.sessions import InMemorySessionService

            self._session_service = InMemorySessionService()
        return self._session_service

    def _get_memory_service(self):
        """The ADK memory service backing cross-prospect recall, built lazily only
        when session memory is enabled. Returns None when the flag is off -- the
        Runner then gets no memory service and behavior is byte-for-byte unchanged.
        Read at call time (not cached in __init__) so a test may flip the flag."""
        if not DEFAULT_CONFIG.use_session_memory:
            return None
        if self._memory_service is None:
            from google.adk.memory import InMemoryMemoryService

            self._memory_service = InMemoryMemoryService()
        return self._memory_service

    def _get_runner(self):
        if self._runner is None:
            from google.adk.runners import Runner

            from smart_assignment.agent import root_agent

            self._runner = Runner(
                agent=root_agent,
                app_name=_APP_NAME,
                session_service=self._get_session_service(),
                # None when session memory is off -> identical to the prior Runner.
                memory_service=self._get_memory_service(),
            )
        return self._runner

    def _user_id_for(self, session_id: str) -> str:
        """The ADK user_id for a browser session.

        With session memory ON, the browser session_id IS the user_id: ADK's
        per-user memory is keyed by (app_name, user_id), so using the browser id
        scopes recall to that one browser and its prospects -- never leaking one
        browser's facts into another's. With the flag OFF, the fixed ``_USER_ID``
        is used exactly as before, so nothing about the current behavior changes."""
        if DEFAULT_CONFIG.use_session_memory:
            return session_id
        return _USER_ID

    def _adk_session_id(self, session_id: str) -> str:
        """The underlying ADK session id for a browser session's CURRENT prospect.
        Generation 0 is the bare id (backward-compatible); later prospects get a
        ``#N`` suffix so each starts a fresh ADK conversation + state."""
        gen = self._generation.get(session_id, 0)
        return session_id if gen == 0 else f"{session_id}#{gen}"

    async def _maybe_rotate_prospect(self, session_id: str, message: str) -> None:
        """Start a new underlying ADK conversation when the user begins a NEW
        prospect after the current one already concluded/escalated. A new prospect
        is a message that carries a street address; a revision (e.g. "try 20
        cases", "make it Tuesday") carries none and stays in the same session so
        multi-turn context is preserved.

        Best-effort by design: the address regex misses many natural phrasings
        ("new customer at <address> - 260 cases" does not rotate), and that is
        acceptable because this is NOT what prevents cross-prospect
        contamination -- the intake-level guards do that on every surface (see
        the note on ``_generation`` in ``__init__``). Widening the trigger would
        only tidy transcripts sooner; failing to rotate must never leak data."""
        has_address = parse_intake(message).address is not None
        concluded = session_id in self._concluded or session_id in self._pending_input
        if has_address and concluded:
            # Before the fresh session wipes the concluding prospect's transcript,
            # fold it into memory so cross-prospect recall survives the rotation.
            # A no-op when session memory is off (no memory service). This is the
            # ONLY point that ingests: the current (still-active) prospect is never
            # in memory, so preload never double-counts what session replay already
            # shows the model.
            await self._ingest_current_into_memory(session_id)
            self._generation[session_id] = self._generation.get(session_id, 0) + 1
            # A rotated prospect is a fresh start: drop any pending escalation resume
            # (so the new prospect isn't misrouted as the specialist's reply) and
            # the concluded mark (the new prospect hasn't concluded yet).
            self._pending_input.pop(session_id, None)
            self._concluded.discard(session_id)

    async def _ingest_current_into_memory(self, session_id: str) -> None:
        """Fold the browser session's CURRENT (concluding) ADK conversation into
        the memory service, so its facts remain recallable after rotation minted a
        fresh, empty session. Called before the generation is bumped, so
        ``_adk_session_id`` still points at the prospect being left behind. A no-op
        when session memory is off, or if that session was never created."""
        memory_service = self._get_memory_service()
        if memory_service is None:
            return
        adk_session_id = self._adk_session_id(session_id)
        session = await self._get_session_service().get_session(
            app_name=_APP_NAME,
            user_id=self._user_id_for(session_id),
            session_id=adk_session_id,
        )
        if session is not None:
            await memory_service.add_session_to_memory(session)

    async def _ensure_session(self, adk_session_id: str, user_id: str) -> None:
        if adk_session_id in self._known_sessions:
            return
        await self._get_session_service().create_session(
            app_name=_APP_NAME, user_id=user_id, session_id=adk_session_id
        )
        self._known_sessions.add(adk_session_id)

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

    async def _visualization_from_state(
        self,
        adk_session_id: str,
        user_id: Optional[str] = None,
        reasoning_override: Optional[str] = None,
    ) -> Optional[dict]:
        """Rebuild the profile from session state and produce the Simulator
        payload. ``reasoning_override`` carries the agent's own recommendation
        narration so the result card's "Why the agent chose this" shows the same
        text the chat box did, not a separately-rendered one.

        Steps 1-4 are deterministic, so re-deriving the candidates here keeps the
        numbers drift-free. Step 5 is NOT deterministic once grounded reasoning is
        on -- it samples, and resamples for consensus -- so the decision the agent
        already made is REUSED rather than recomputed. Without that, the card
        could show a second, independently-sampled decision underneath the
        agent's narration of the first one, and the feedback/trace would record
        an outcome the user was never shown."""
        session = await self._get_session_service().get_session(
            app_name=_APP_NAME, user_id=user_id or _USER_ID, session_id=adk_session_id
        )
        state = (session.state if session else None) or {}
        profile = state.get(_STATE_PROFILE_KEY)
        if not profile or not profile.get("address") or not profile.get("order_quantity_cases"):
            return None
        customer = _profile_from_state_dict(profile)
        # None whenever the snapshot is missing, stale (the prospect was revised)
        # or unreadable -- in which case we decide once here, exactly as before.
        cached = cached_decision_for(state, profile)
        # The pipeline runs synchronously and, with no cached decision, may make a
        # grounded LLM call. We are on the server's event loop here, and for the
        # sage backend that call must run a coroutine on this loop -- impossible
        # if we block it. Offload to a worker thread so the loop stays free.
        with traced_decision(DEFAULT_CONFIG) as decision:
            result = await offload_to_worker_thread(
                run_slot_recommendation,
                customer,
                config=DEFAULT_CONFIG,
                geocoder=self._geocoder,
                recommendation=cached,
            )
            decision.record(result)
        payload = build_workflow_payload(
            result, DEFAULT_CONFIG, reasoning_override=reasoning_override
        )
        # Transient feedback hints (private ``_``-prefixed keys) consumed and
        # stripped by app._attach_feedback before the payload is serialized, so
        # they never reach the browser. They let the feedback stamp carry the
        # recommend/escalate outcome and a real trace link (see webapp/decision.py).
        payload["_decision"] = decision.context
        payload["_trace"] = dict(decision.coords) if decision.coords else None
        return payload

    # -- the turn stream --

    async def stream_turn(self, session_id: str, message: str) -> AsyncGenerator[dict, None]:
        """Run one conversational turn, yielding frame dicts:

        ``{"type": "tool", "name", "status", "label", "detail"}``
                                                   — a pipeline step changed state.
                                                   ``status`` is ``running`` (the
                                                   tool was just called), ``done``
                                                   or ``failed`` (the tool reported
                                                   back). ``label``/``detail`` come
                                                   with the ``running`` frame; the
                                                   closing frame carries only what
                                                   changed.
        ``{"type": "message", "text"}``            — agent natural-language reply
        ``{"type": "await_input", "message"}``     — human-in-the-loop escalation
        ``{"type": "visualization", "payload"}``   — the 5 step cards + result
        ``{"type": "done"}``                       — turn finished
        """
        # Start a fresh ADK conversation if this message begins a NEW prospect
        # after the current one concluded/escalated, so nothing bleeds across.
        # (When session memory is on, this also folds the concluding prospect into
        # memory first, so its facts survive the rotation.)
        await self._maybe_rotate_prospect(session_id, message)
        adk_session_id = self._adk_session_id(session_id)
        # With session memory on, the browser session_id is the ADK user_id, so
        # memory is scoped to this browser; off, it's the fixed _USER_ID as before.
        user_id = self._user_id_for(session_id)

        await self._ensure_session(adk_session_id, user_id)
        runner = self._get_runner()
        new_message = self._build_message(session_id, message)

        from google.adk.agents.run_config import RunConfig, StreamingMode

        saw_recommendation = False
        # Pipeline steps already shown as breadcrumbs this turn. Breadcrumbs track
        # the pipeline STEPS (Geo-Lookup, Score & Rank, ...), not the tool calls, so
        # one consolidated recommend_or_escalate call still lights up every step it
        # runs internally -- and a step is never shown twice (e.g. if the user asked
        # for an on-demand find_candidate_routes first). See narration.tool_steps.
        emitted_steps: set[str] = set()
        # Steps opened by an in-flight tool call, keyed by that call's id, so the
        # matching FunctionResponse can close them out with what ACTUALLY happened.
        # A call only tells us work was requested; only the response says whether
        # it succeeded, so no step is ever marked done off the back of a call.
        open_steps: dict[str, list[str]] = {}
        # The agent's own recommendation narration (everything it says AFTER it
        # calls recommend_or_escalate this turn), captured so the visualization's
        # "Why the agent chose this" can show the same words as the chat box.
        recommendation_reply: list[str] = []
        # Set when the turn ended on an error-recovery notice (agent_callbacks)
        # WITHOUT having reached a decision -- carries the reason so the raise
        # after the loop is diagnosable. See the branch below for why.
        unusable_turn: Optional[str] = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=new_message,
            # Run the model in NON-streaming mode -- exactly what ``adk web`` does
            # by default (its dev UI sends ``streaming=false`` unless "Token
            # Streaming" is toggled on, so ADK builds
            # ``RunConfig(streaming_mode=NONE)``). Two reasons this is the right
            # choice here, not a downgrade:
            #   1. Parity + robustness. Some LiteLLM-backed models (e.g. the Sage
            #      path) raise inside LiteLLM's async streaming handler
            #      ("'coroutine' object is not an iterator") when token streaming
            #      is requested. ``adk web`` avoids it by defaulting to NONE; when
            #      this service forced ``StreamingMode.SSE`` it hit that bug on
            #      every turn and silently fell back to the deterministic brain --
            #      the web-app-vs-``adk web`` divergence users saw.
            #   2. No lost behavior. The browser-facing SSE stream is emitted by
            #      THIS method (one frame per tool call / completed message); it
            #      does not depend on model token streaming. The loop below only
            #      emits aggregated, non-partial events anyway, so requesting
            #      token streaming bought nothing while adding a failure mode.
            run_config=RunConfig(streaming_mode=StreamingMode.NONE),
        ):
            # Human-in-the-loop: request_input surfaces as a long-running call.
            if getattr(event, "long_running_tool_ids", None):
                for fc in event.get_function_calls():
                    if fc.id in event.long_running_tool_ids:
                        self._pending_input[session_id] = {"id": fc.id, "name": fc.name}
                        prompt = (fc.args or {}).get("message") or (
                            "A routing specialist needs to confirm this before it's final."
                        )
                        # The escalation message is the triage brief. Normalize its
                        # layout at the display surface too, so the specialist always
                        # sees the same scannable structure even if an intervening
                        # agent reflowed it into a run-on line.
                        yield {"type": "await_input", "message": normalize_brief(prompt)}
                continue

            calls = event.get_function_calls()
            if calls:
                for fc in calls:
                    # Open one breadcrumb per pipeline STEP this tool runs internally,
                    # skipping any step already shown this turn -- so recommend_or_
                    # escalate lights up Geo-Lookup + Score & Rank + Recommend/Decide
                    # even though it is a single tool call (see narration.tool_steps).
                    # They open as RUNNING: the tool has been asked to do this work,
                    # and none of it has happened yet.
                    started: list[str] = []
                    for step in tool_steps(fc.name):
                        if step in emitted_steps:
                            continue
                        emitted_steps.add(step)
                        started.append(step)
                        frame = {
                            "type": "tool",
                            "name": step,
                            "label": step_label(step),
                            "status": "running",
                        }
                        # "handoff" for a step that passes the prospect to a person
                        # (the escalation brief), so the UI can style that phase
                        # apart from the assignment steps. Absent otherwise.
                        phase = step_phase(step)
                        if phase:
                            frame["phase"] = phase
                        # A plain-language line of what this step is doing (Intake
                        # echoes the customer's own inputs back); omit when there's
                        # nothing to add so the frame shape stays minimal.
                        detail = step_detail(step, fc.args or {})
                        if detail:
                            frame["detail"] = detail
                        yield frame
                    if started:
                        open_steps[_call_key(fc)] = started
                continue

            responses = event.get_function_responses()
            if responses:
                # The tool reported back -- the only point in the turn where we
                # learn what actually ran. Close its steps with that verdict, and
                # treat a decision as reached only when the tool says it succeeded.
                for fr in responses:
                    ok, error = _tool_outcome(fr.response)
                    if ok and fr.name in _DECISION_TOOLS:
                        saw_recommendation = True
                    # A decision that escalated says so on its own step, so the
                    # handoff breadcrumb that follows reads as a consequence rather
                    # than a surprise. Restates the tool's own
                    # ``requires_human_review`` flag -- the REASON is the audited
                    # brief's job, never a breadcrumb's.
                    escalated = ok and _requires_human_review(fr)
                    for step in open_steps.pop(_call_key(fr), []):
                        frame = {
                            "type": "tool",
                            "name": step,
                            "status": "done" if ok else "failed",
                        }
                        phase = step_phase(step)
                        if phase:
                            frame["phase"] = phase
                        # On a failure, relay the tool's OWN error text rather than
                        # narrating a cause we'd be guessing at.
                        if error:
                            frame["detail"] = error
                        elif escalated and step == "recommend_or_escalate":
                            frame["detail"] = ESCALATION_DETAIL
                        yield frame
                continue

            # An error-recovery notice from agent_callbacks: the model call failed
            # and the turn was ended gracefully instead of unwinding the Runner.
            # ADK's Event subclasses LlmResponse, so the stamped code is readable
            # here. What to do with it depends entirely on whether this turn had
            # already produced anything worth keeping:
            #
            #   decision reached -> the pipeline result is REAL and already shown.
            #       Keep it: surface the notice and let the visualization render
            #       below. Falling back now would replace a correct, audited answer
            #       with a second one that could contradict it.
            #   no decision      -> this turn produced nothing at all. Re-raise so
            #       ``app.chat`` runs the deterministic brain exactly as it did
            #       before recovery existed -- otherwise the user is told to try
            #       again where they used to get a real answer, which would be
            #       WORSE than the deterministic baseline this project guarantees.
            #
            # Either way the notice is never appended to ``recommendation_reply``:
            # an error is not the agent's reasoning for a decision it never made.
            if getattr(event, "error_code", None):
                if not saw_recommendation:
                    unusable_turn = (
                        getattr(event, "error_message", None) or event.error_code
                    )
                    continue
                text = _event_text(event)
                if text:
                    yield {"type": "message", "text": text}
                continue

            # Natural-language text. Emit only the aggregated (non-partial) event
            # so the transcript gets each reply once, not per streamed chunk.
            if event.content and event.content.parts and not getattr(event, "partial", False):
                text = _event_text(event)
                if text:
                    if saw_recommendation:
                        recommendation_reply.append(text)
                    yield {"type": "message", "text": text}

        if unusable_turn is not None:
            # Nothing was produced this turn, so hand the failure to the caller and
            # let the deterministic brain answer -- the floor this project promises.
            raise AgentTurnUnavailable(unusable_turn)

        if saw_recommendation:
            # This prospect reached a decision/escalation: the NEXT full prospect
            # in this browser session should start a fresh ADK conversation.
            self._concluded.add(session_id)
            override = "\n\n".join(recommendation_reply) or None
            try:
                payload = await self._visualization_from_state(
                    adk_session_id, user_id=user_id, reasoning_override=override
                )
            except Exception:  # noqa: BLE001 - the turn already has a real answer
                # The visualization is a re-derivation of a decision the agent has
                # ALREADY made and narrated to the user. If rebuilding it fails
                # (a geocoder hiccup on the re-run, unreadable state), that must
                # not take the turn down with it -- doing so would discard the
                # agent's own correct reply and hand the user a deterministic
                # fallback contradicting it. Log the reason and show no cards.
                logger.exception(
                    "Could not rebuild the visualization for session %s; the "
                    "agent's reply stands and no result cards are shown.",
                    session_id,
                )
                payload = None
            if payload:
                yield {"type": "visualization", "payload": payload}

        yield {"type": "done"}
