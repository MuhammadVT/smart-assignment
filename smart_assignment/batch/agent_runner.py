"""
The AGENT batch runner: drive the real conversational ``root_agent`` architecture
over a source of prospects, unattended, one prospect per turn.

``AgentBatchRunner`` is the batch orchestrator: it runs the actual ADK agent
(``agent.build_batch_agent``) once per prospect so the batch inherits everything
the agent adds -- its natural-language reasoning, the ``escalation_triage``
AgentTool brief, and any future agent behavior -- while staying non-interactive.
The deterministic per-prospect engine (``runner.run_one``) is its floor, not a
separate mode:

  * **Intake is seeded, not asked.** The Salesforce profile is written into the
    ADK session state before the turn, so the agent goes straight to the decision
    (one ``assign_prospect`` tool call) with no conversational back-and-forth.
  * **Escalation is captured, not paused.** The agent still composes its triage
    brief and calls ``request_input``; the driver INTERCEPTS that call, captures
    the brief, and terminates the turn (it never waits for a human reply) -- the
    brief becomes the escalation record the SC reviews asynchronously.
  * **Never worse than the deterministic baseline.** If the agent can't be built
    (no credentials / backend) the whole run degrades to the deterministic
    ``run_one``; if a single agent turn fails or produces no decision, THAT
    prospect degrades to ``run_one``. A per-prospect failure is recorded as
    ``needs_attention`` and never aborts the batch.

The output contract (``BatchRecord``), the source/sink protocols, and the payload
the Customer View renders are the SAME ones ``runner.py`` uses -- this runner only
swaps the per-prospect engine, so a result still renders with no frontend change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from smart_assignment.batch.runner import BatchSummary, _needs_attention, _utc_now_iso, run_one
from smart_assignment.batch.sink import (
    OUTCOME_ESCALATE,
    OUTCOME_NEEDS_ATTENTION,
    OUTCOME_RECOMMEND,
    BatchRecord,
    ResultSink,
)
from smart_assignment.batch.source import Prospect, ProspectSource
from smart_assignment.integrations.geocoding_client import resolve_geocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.shared.config import DEFAULT_CONFIG, Config
from smart_assignment.shared.geo import Geocoder
from smart_assignment.shared.llm import offload_to_worker_thread
from smart_assignment.shared.models import Route
from smart_assignment.tools.slot_recommendation import (
    _STATE_LAST_RECOMMENDATION_KEY,
    _STATE_PROFILE_KEY,
    _profile_to_state_dict,
    _profile_from_state_dict,
    cached_decision_for,
)

logger = logging.getLogger(__name__)

_APP_NAME = "smart_assignment_batch"
_USER_ID = "batch_user"
# The one-line prompt that kicks off a batch turn. The prospect is already seeded
# into session state, so this only tells the agent to proceed.
_TURN_MESSAGE = "Assign this prospect to a delivery route and slot."


class AgentBatchRunner:
    """Sequentially runs a ``ProspectSource`` through the batch AGENT into a
    ``ResultSink``.

    Collaborators are injected (source, sink, config, geocoder, routes, clock) so
    the runner is testable offline and points at real systems by argument, not by
    edit -- the same injection seam the deterministic engine uses.
    ``runner``/``session_service`` are
    additionally injectable so tests can drive the turn logic with a fake ADK
    runner (no credentials); in production they default to a real ADK ``Runner``
    over ``build_batch_agent``. The routes list, the geocoder, and the agent are
    resolved ONCE per run and reused for every prospect."""

    def __init__(
        self,
        source: ProspectSource,
        sink: ResultSink,
        config: Optional[Config] = None,
        geocoder: Optional[Geocoder] = None,
        routes: Optional[list[Route]] = None,
        clock: Optional[Callable[[], str]] = None,
        concurrency: int = 1,
        runner=None,
        session_service=None,
    ) -> None:
        self._source = source
        self._sink = sink
        self._config = config or DEFAULT_CONFIG
        self._geocoder = geocoder
        self._routes = routes
        self._clock = clock or _utc_now_iso
        # How many prospects may be in flight at once. 1 (default) is sequential --
        # exact prior behavior. Prospects are independent (each its own ADK session;
        # routes/geocoder resolved once and read-only), so raising this fans agent
        # turns out concurrently for throughput, bounded by a semaphore.
        self._concurrency = max(1, concurrency)
        self._runner = runner
        self._session_service = session_service
        # Guard so a failed one-time agent build isn't retried per prospect.
        self._runner_resolved = runner is not None

    # -- lazy ADK wiring (built once; None means "no agent -> deterministic") --

    def _get_session_service(self):
        if self._session_service is None:
            from google.adk.sessions import InMemorySessionService

            self._session_service = InMemorySessionService()
        return self._session_service

    def _get_runner(self):
        """The ADK Runner over the batch agent, or ``None`` if the agent can't be
        built (e.g. no credentials) -- in which case the run uses the deterministic
        pipeline throughout. Built at most once."""
        if self._runner_resolved:
            return self._runner
        self._runner_resolved = True
        try:
            from google.adk.runners import Runner

            from smart_assignment.agent import build_batch_agent

            agent = build_batch_agent(self._config)
            self._runner = Runner(
                agent=agent,
                app_name=_APP_NAME,
                session_service=self._get_session_service(),
            )
        except Exception as exc:  # noqa: BLE001 - never worse than the deterministic baseline
            logger.warning(
                "Batch agent unavailable (%s); running the deterministic pipeline for this "
                "batch instead. Check SMART_ASSIGNMENT_LLM_BACKEND and its credentials.",
                exc,
            )
            self._runner = None
        return self._runner

    # -- the run loop --

    async def run(self) -> BatchSummary:
        geocoder = self._geocoder or resolve_geocoder()
        routes = self._routes if self._routes is not None else fetch_candidate_routes()
        runner = self._get_runner()

        prospects = list(self._source.prospects())
        # Bound how many prospects are processed at once. With concurrency=1 this
        # awaits each in turn (sequential); higher values fan the independent turns
        # out, at most `_concurrency` in flight. Each prospect is fully isolated
        # (its own session), so a shared Runner/geocoder/routes is safe.
        semaphore = asyncio.Semaphore(self._concurrency)
        outcomes = await asyncio.gather(
            *(self._process_one(p, runner, geocoder, routes, semaphore) for p in prospects)
        )

        counts = {OUTCOME_RECOMMEND: 0, OUTCOME_ESCALATE: 0, OUTCOME_NEEDS_ATTENTION: 0}
        for outcome in outcomes:
            counts[outcome] += 1
        return BatchSummary(
            total=len(prospects),
            recommend=counts[OUTCOME_RECOMMEND],
            escalate=counts[OUTCOME_ESCALATE],
            needs_attention=counts[OUTCOME_NEEDS_ATTENTION],
        )

    async def _process_one(self, prospect, runner, geocoder, routes, semaphore) -> str:
        """Produce and emit one prospect's record, holding a concurrency slot only
        for the decision work. Returns the outcome for the summary tally. A failure
        here becomes ``needs_attention`` -- one prospect must never abort the batch."""
        generated_at = self._clock()
        async with semaphore:
            try:
                if runner is None:
                    # No agent this run -> the deterministic floor for every prospect.
                    record = await self._deterministic(prospect, geocoder, routes, generated_at)
                else:
                    record = await self._run_one_via_agent(
                        prospect, runner, geocoder, routes, generated_at
                    )
            except Exception as exc:  # noqa: BLE001 - one prospect must never abort the batch
                logger.exception(
                    "Batch entry %s failed unexpectedly; recording needs_attention.",
                    prospect.prospect_id,
                )
                record = _needs_attention(prospect, generated_at, f"unexpected error: {exc}")
        # Emit outside the slot: the write is trivial and freeing the slot sooner
        # lets the next agent turn start. emit() is synchronous, so concurrent
        # coroutines never interleave a partial record on the one event loop.
        self._sink.emit(record)
        return record.outcome

    # -- per-prospect: the agent turn, then map its outcome --

    async def _run_one_via_agent(
        self,
        prospect: Prospect,
        runner,
        geocoder: Geocoder,
        routes: list[Route],
        generated_at: str,
    ) -> BatchRecord:
        """Run one non-interactive agent turn and map its outcome to a BatchRecord.

        On any failure -- the turn raising, or completing without a stored decision
        (an intake/geocode problem the agent reported instead of deciding) -- the
        prospect degrades to the deterministic ``run_one``, so this is never worse
        than the baseline."""
        try:
            narration, brief = await self._drive_agent_turn(prospect)
            state = await self._session_state(prospect.prospect_id)
        except Exception as exc:  # noqa: BLE001 - degrade this prospect, keep the batch going
            logger.warning(
                "Agent turn for %s failed (%s); falling back to the deterministic pipeline.",
                prospect.prospect_id,
                exc,
            )
            return await self._deterministic(prospect, geocoder, routes, generated_at)

        profile = state.get(_STATE_PROFILE_KEY)
        decision = state.get(_STATE_LAST_RECOMMENDATION_KEY)
        if not profile or not decision:
            # The agent didn't reach a decision (e.g. assign_prospect returned
            # {"ok": false} on a bad/unresolvable address). Let the deterministic
            # path produce the typed outcome (a clean needs_attention, or a real
            # decision if the agent merely misbehaved) -- never worse than baseline.
            return await self._deterministic(prospect, geocoder, routes, generated_at)

        # Rebuild the exact decision the agent's tool already made (steps 1-4 are
        # re-derived deterministically; step 5 is REUSED from the snapshot, never
        # re-sampled) and render the same Customer-View payload. Offloaded so a
        # grounded re-derivation can hand its coroutine back to this event loop.
        customer = _profile_from_state_dict(profile)
        cached = cached_decision_for(state, profile)
        result = await offload_to_worker_thread(
            run_slot_recommendation,
            customer,
            routes=routes,
            config=self._config,
            geocoder=geocoder,
            recommendation=cached,
        )
        from smart_assignment.reporting.page import build_workflow_payload

        payload = build_workflow_payload(result, self._config, reasoning_override=narration or None)
        rec = result.recommendation
        if not rec.requires_human_review:
            return BatchRecord(
                prospect.prospect_id, generated_at, OUTCOME_RECOMMEND, payload=payload
            )
        # Escalation: the brief the agent composed via escalation_triage (captured
        # from the intercepted request_input), falling back to the decision's own
        # review reason if triage was off or produced nothing.
        return BatchRecord(
            prospect.prospect_id,
            generated_at,
            OUTCOME_ESCALATE,
            payload=payload,
            review_reason=rec.review_reason,
            triage_brief=(brief or rec.review_reason) or None,
        )

    async def _drive_agent_turn(self, prospect: Prospect) -> tuple[Optional[str], Optional[str]]:
        """Seed the prospect into a fresh ADK session, run one turn, and return
        ``(recommendation_narration, escalation_brief)``.

        The narration is the agent's own text AFTER it calls ``assign_prospect``
        (so it renders as the result card's reasoning, matching the chat surface).
        The brief is the message the agent passed to ``request_input`` on an
        escalation -- captured, not resumed (there is no human to reply)."""
        from google.adk.agents.run_config import RunConfig, StreamingMode
        from google.genai import types

        session_service = self._get_session_service()
        session_id = prospect.prospect_id
        # A fresh session per prospect, pre-seeded with the CRM profile so intake
        # needs no conversational turn. Nothing bleeds between prospects.
        await session_service.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
            session_id=session_id,
            state={_STATE_PROFILE_KEY: _profile_to_state_dict(prospect.profile)},
        )
        new_message = types.Content(role="user", parts=[types.Part(text=_TURN_MESSAGE)])

        saw_decision = False
        narration_parts: list[str] = []
        brief: Optional[str] = None
        async for event in self._runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=new_message,
            # Non-streaming, exactly like the web app -- the driver reads aggregated
            # events, not model token chunks (see webapp/llm_chat.py for why).
            run_config=RunConfig(streaming_mode=StreamingMode.NONE),
        ):
            # Escalation: request_input surfaces as a long-running call. Capture its
            # message (the triage brief) and do NOT resume -- there is no human.
            if getattr(event, "long_running_tool_ids", None):
                for fc in event.get_function_calls():
                    if fc.id in event.long_running_tool_ids:
                        brief = (fc.args or {}).get("message") or brief
                continue
            calls = event.get_function_calls()
            if calls:
                if any(fc.name == "assign_prospect" for fc in calls):
                    saw_decision = True
                continue
            if event.get_function_responses():
                continue
            # Natural-language text: keep only the aggregated (non-partial) reply,
            # and only what the agent says AFTER the decision, as the narration.
            if event.content and event.content.parts and not getattr(event, "partial", False):
                text = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
                if text.strip() and saw_decision:
                    narration_parts.append(text.strip())

        narration = "\n\n".join(narration_parts) or None
        if brief is not None:
            # Normalize the brief's layout at the sink too, so the SC always sees the
            # same scannable structure (matches the web app's display normalization).
            from smart_assignment.triage.formatting import normalize_brief

            brief = normalize_brief(brief)
        return narration, brief

    async def _session_state(self, session_id: str) -> dict:
        session = await self._get_session_service().get_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id
        )
        return (session.state if session else None) or {}

    async def _deterministic(
        self,
        prospect: Prospect,
        geocoder: Geocoder,
        routes: list[Route],
        generated_at: str,
    ) -> BatchRecord:
        """The deterministic floor for one prospect (the exact ``run_one`` the
        non-agent batch uses). Offloaded to a worker thread so a grounded decision
        can hand its coroutine back to this event loop."""
        return await offload_to_worker_thread(
            run_one, prospect, self._config, geocoder, routes, generated_at
        )
