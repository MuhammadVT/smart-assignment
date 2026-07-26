"""
Compose an escalation brief without a conversation.

`triage/agent.py` builds the specialist brief as an `LlmAgent` that `root_agent`
consults mid-conversation. That shape is right: writing the brief is genuinely
multi-step -- load the trace, draft prose, self-check that every figure is
grounded, revise -- which is what an agent is for, and unlike the route-slot
decision it produces free text rather than a choice from an enumerated set.

The headless service has no conversation and no ADK session, but it needs the
*same* brief. Rather than reimplement it (two prompts and two context builders
that would drift the first time the layout changed), this runs the **existing,
unmodified** agent in a throwaway session seeded with the two state keys
`get_escalation_context` reads. Nothing in `triage/` changes; the agent object is
shared, not copied.

Called on demand rather than inline with the decision, so the only unbounded,
model-driven step in the system sits behind a specialist actually opening the
escalation -- never on the critical path of a decision that may never be read.

Every failure returns ``None``. The caller then shows the structured escalation
facts it already has (reason, rejected routes, trade-off), which is the honest
fallback: a truncated or ungrounded brief presented as complete is worse for a
specialist than no brief at all.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from smart_assignment.shared.config import ROLE_TRIAGE, Config
from smart_assignment.shared.llm import _run_coro_blocking
from smart_assignment.shared.models import CustomerProfile, SlotRecommendation
from smart_assignment.triage.formatting import normalize_brief

logger = logging.getLogger(__name__)

_APP_NAME = "smart_assignment_headless_triage"
_USER_ID = "headless"
_SESSION_ID = "triage"

# The agent loads the context, drafts, self-checks the grounding, and revises
# until the check passes -- so the call count scales with how many revisions the
# brief needs, not with a fixed script. Measured against the mock world: a
# low-score escalation converged inside 8, but a no-feasible-slot one (where every
# figure is a constraint failure and the grounding check is fussier) needed more
# than 8 and settled by 16. The cap is a guard against a revise loop that never
# converges, not a budget, so it sits well above the worst case observed; the
# timeout is the backstop that actually bounds latency.
DEFAULT_MAX_LLM_CALLS = 24
DEFAULT_TIMEOUT_SECONDS = 120.0

_KICKOFF = (
    "Triage the escalated slot recommendation for the prospect in session state, "
    "and return the specialist brief."
)

# build_triage_agent resolves the LLM backend, which for sage constructs a
# registry -- worth doing once per model rather than once per brief. Keyed by
# everything in Config that changes how the agent is built; the instruction and
# tools are static.
_AGENT_CACHE: dict[tuple, Any] = {}


def _triage_agent(config: Config) -> Any:
    from smart_assignment.triage.agent import build_triage_agent

    key = (config.llm_backend, config.resolved_model(ROLE_TRIAGE), config.use_sage_gateway)
    agent = _AGENT_CACHE.get(key)
    if agent is None:
        agent = build_triage_agent(config)
        _AGENT_CACHE[key] = agent
    return agent


def _seed_state(customer: CustomerProfile, recommendation: SlotRecommendation) -> dict:
    """The session state `get_escalation_context` expects.

    ``sa_last_recommendation`` is built from `to_state_dict()` rather than by
    hand-picking the handful of keys the context builder happens to read today:
    that snapshot is pinned by a test asserting it covers every declared field, so
    a field added to `SlotRecommendation` arrives here automatically instead of
    silently going missing. ``requires_human_review`` is added because it is a
    property rather than a field, and the context builder gates on it.
    """
    from smart_assignment.tools.slot_recommendation import (
        _STATE_LAST_RECOMMENDATION_KEY,
        _STATE_PROFILE_KEY,
        _profile_to_state_dict,
    )

    return {
        _STATE_PROFILE_KEY: _profile_to_state_dict(customer),
        _STATE_LAST_RECOMMENDATION_KEY: {
            **recommendation.to_state_dict(),
            "requires_human_review": recommendation.requires_human_review,
        },
    }


def _final_text(event) -> str:
    """The natural-language text of one ADK event, ignoring tool traffic."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


async def _run_triage(
    customer: CustomerProfile,
    recommendation: SlotRecommendation,
    config: Config,
    max_llm_calls: int,
    timeout_seconds: float,
) -> str:
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=_APP_NAME,
        user_id=_USER_ID,
        session_id=_SESSION_ID,
        state=_seed_state(customer, recommendation),
    )
    runner = Runner(
        agent=_triage_agent(config),
        app_name=_APP_NAME,
        session_service=session_service,
    )

    async def _drive() -> str:
        chunks: list[str] = []
        async for event in runner.run_async(
            user_id=_USER_ID,
            session_id=_SESSION_ID,
            new_message=types.Content(role="user", parts=[types.Part(text=_KICKOFF)]),
            # Non-streaming, matching what `adk web` and the web app's chat
            # service do: some LiteLLM-backed models raise inside the async
            # streaming handler, and nothing here consumes partial tokens.
            run_config=RunConfig(
                streaming_mode=StreamingMode.NONE, max_llm_calls=max_llm_calls
            ),
        ):
            if event.get_function_calls() or event.get_function_responses():
                continue  # tool traffic; the brief is the final text turn
            if getattr(event, "partial", False):
                continue
            text = _final_text(event)
            if text.strip():
                chunks.append(text.strip())
        return "\n\n".join(chunks)

    return await asyncio.wait_for(_drive(), timeout=timeout_seconds)


def compose_brief(
    customer: CustomerProfile,
    recommendation: SlotRecommendation,
    config: Config,
    *,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """The specialist brief for an escalated recommendation, or ``None``.

    Synchronous, so an ordinary caller needs no async plumbing; the ADK run is
    driven on whichever event loop `shared.llm` has established (the service's
    process-wide loop, or a server's own), the same way a grounded call is.

    Returns ``None`` -- never raises, and never returns a partial brief -- when
    there is nothing to triage, when the model or its credentials are
    unavailable, when the agent exceeds its call cap or the timeout, or on any
    other failure. The caller shows the structured escalation facts instead.
    """
    if not recommendation.requires_human_review:
        logger.debug("Nothing to triage: this recommendation was auto-approved.")
        return None

    try:
        brief = _run_coro_blocking(
            _run_triage(customer, recommendation, config, max_llm_calls, timeout_seconds)
        )
    except Exception as exc:  # noqa: BLE001 - a brief is advisory; never break the caller
        logger.warning(
            "Escalation brief could not be composed (%s: %s); "
            "falling back to the structured escalation facts.",
            type(exc).__name__,
            exc,
        )
        return None

    if not brief or not brief.strip():
        logger.warning("Escalation triage returned no text; no brief for this escalation.")
        return None

    # Idempotent, and the agent's own after-model callback has normally already
    # applied it -- run it here too so a brief is laid out identically no matter
    # which surface produced it (the same reason the web app normalizes on
    # display).
    return normalize_brief(brief)
