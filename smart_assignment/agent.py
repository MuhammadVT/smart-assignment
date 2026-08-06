"""
ADK entry point. ADK's CLI (`adk run`, `adk web`) and deployment tooling
look for a `root_agent` here.

`root_agent` is a single conversational `LlmAgent`: it collects a prospect's
address, order quantity, and (optional) preferred slot over multiple turns,
then calls the deterministic pipeline (pipeline.py) as tools
(tools/slot_recommendation.py) rather than computing anything itself --
every distance, constraint check, score, and decision comes straight from
that same plain Python, so the outcome stays reproducible and auditable
even though the conversation is LLM-driven.

`root_agent` is built **lazily** (PEP 562 module ``__getattr__``): merely
importing this module -- which ``smart_assignment/__init__.py`` does for every
import of the package -- must not resolve the LLM backend, because under the
default ``sage`` backend ``get_llm`` requires credentials and would raise. The
offline paths (``scripts/run_local.py``, ``scripts/generate_page.py``, the test
suite, and the deterministic web app) import the package but never touch
``root_agent``, so they now work with no credentials. ADK discovers the agent by
attribute access (``smart_assignment.agent.root_agent`` /
``from smart_assignment.agent import root_agent``), which still triggers
construction with the configured backend -- identical behavior to before for
``adk run`` / ``adk web`` / ``adk deploy``.

The first multi-agent split is the **escalation-triage** sub-agent (see the
`triage` package), wired in below when `Config.use_escalation_triage` is on:
an `LlmAgent` exposed via `google.adk.tools.AgentTool` that root_agent consults
on an escalation to compose a specialist brief. It runs downstream of the
deterministic decision and only reads session state, so it never changes a
number or the decision. The same pattern (wrap a function in its own `LlmAgent`,
expose it via `AgentTool`) applies to any future sub-agent (a richer intake
agent, a Q&A agent over past recommendations), since each tool is already
independent and keyed only through session state (see tools/slot_recommendation.py).
"""

import functools

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, request_input

from smart_assignment.agent_callbacks import error_callbacks
from smart_assignment.prompts import build_batch_instruction, build_instruction
from smart_assignment.shared import tracing
from smart_assignment.shared.config import DEFAULT_CONFIG, ROLE_ROOT_AGENT, Config
from smart_assignment.shared.llm import get_llm, offload_to_worker_thread
from smart_assignment.tools import (
    assign_prospect,
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
    resolve_address,
    start_new_prospect,
)

# Cached after first access so repeated lookups return the same agent instance.
_root_agent: LlmAgent = None  # type: ignore[assignment]


def _offloaded_tool(func):
    """Wrap a synchronous pipeline tool so ADK runs it as an async tool whose
    blocking body executes OFF the event loop.

    ADK invokes a synchronous ``FunctionTool`` inline on the event-loop thread. On
    the server that thread is uvicorn's loop, and the pipeline these tools drive
    can make a synchronous grounded LLM call that (for the sage backend) must run a
    coroutine on that very loop -- which is impossible while the tool is blocking
    it. Offloading the body to a worker thread frees the loop, and
    ``offload_to_worker_thread`` records it so the grounded call can hand its
    coroutine back (see ``shared/llm.py``). ``functools.wraps`` preserves the
    name/signature/docstring, so ADK builds the identical function declaration and
    ``tool_context`` injection still works; the wrapper is just ``async``."""

    @functools.wraps(func)
    async def _async_tool(*args, **kwargs):
        return await offload_to_worker_thread(func, *args, **kwargs)

    return _async_tool


def _build_root_agent() -> LlmAgent:
    """Construct the conversational agent. Resolves the LLM backend (``get_llm``),
    so this needs credentials for the configured backend -- called only on first
    access to ``root_agent``, never at import."""
    # Install tracing (global provider + OTLP exporter + ADK instrumentor) BEFORE
    # the agent runs, so its conversational turns and tool calls are captured from
    # the first turn. A no-op when Config.use_tracing is off, so offline paths that
    # never build root_agent are entirely unaffected. This is the single entry
    # point that covers every agent-serving surface (adk web/deploy, the web app),
    # since they all build root_agent but none share a main() we own.
    tracing.configure_tracing(DEFAULT_CONFIG)

    triage_enabled = DEFAULT_CONFIG.use_escalation_triage
    address_resolution_enabled = DEFAULT_CONFIG.use_address_resolution
    session_memory_enabled = DEFAULT_CONFIG.use_session_memory

    # Every pipeline tool is offloaded to a worker thread (see _offloaded_tool):
    # its body is synchronous and may make a grounded LLM call that needs the
    # server's event loop free, so it must not run inline on that loop.
    tools = [
        FunctionTool(_offloaded_tool(intake_customer)),
        # The model-declared boundary between customers in one conversation: it
        # only discards state (worst case a spurious re-ask, never contamination),
        # and the deterministic address guard inside intake_customer still covers
        # the post-decision case when the model forgets to call it. Interactive
        # surfaces only -- batch seeds a fresh session per prospect and must keep
        # its tool surface byte-identical (see _batch_agent_tools).
        FunctionTool(_offloaded_tool(start_new_prospect)),
        FunctionTool(_offloaded_tool(find_candidate_routes)),
        FunctionTool(_offloaded_tool(evaluate_and_score_routes)),
        FunctionTool(_offloaded_tool(recommend_or_escalate)),
        request_input,
    ]
    if address_resolution_enabled:
        # A grounded typo/ambiguity corrector: on a geocode miss it picks the
        # closest of the geocoder's real candidate matches for the user to
        # confirm, instead of the agent inventing one (see the `address_resolve`
        # package). Only added when enabled, so the instruction never names a
        # tool that isn't present.
        tools.append(FunctionTool(_offloaded_tool(resolve_address)))
    if triage_enabled:
        # Imported lazily so the package import stays credential-free -- this
        # runs only while root_agent is being built, which already resolves the
        # backend via get_llm above.
        from smart_assignment.triage import build_triage_tool

        tools.append(build_triage_tool(DEFAULT_CONFIG))
    if session_memory_enabled:
        # Opt-in cross-prospect recall: ADK's preload_memory auto-runs on every
        # turn (the model never calls it), searching the memory service for facts
        # from earlier, rotated-away prospects and injecting the matches into the
        # instruction. The matching InMemoryMemoryService is wired onto the Runner
        # in webapp/llm_chat.py, which also folds a concluding prospect into memory
        # on rotation. Safe even without a memory service (e.g. a bare ``adk web``):
        # the tool swallows the lookup error and is a no-op, so the only effect of
        # the flag being on with no backend is nothing. Imported lazily to match
        # the other gated tools; credential-free.
        from google.adk.tools.preload_memory_tool import preload_memory_tool

        tools.append(preload_memory_tool)

    return LlmAgent(
        name="smart_assignment_agent",
        model=get_llm(DEFAULT_CONFIG.for_role(ROLE_ROOT_AGENT)),
        description=(
            "Collects a new prospect customer's delivery details conversationally "
            "and recommends -- or escalates -- a delivery route and slot."
        ),
        instruction=build_instruction(
            include_triage=triage_enabled,
            include_address_resolution=address_resolution_enabled,
        ),
        tools=tools,
        # A failed model call ends the turn with a plain reply, and a raised tool
        # becomes an ordinary {"ok": false} result, instead of ADK unwinding the
        # whole Runner and discarding a turn whose pipeline already succeeded.
        # Empty (agent built exactly as before) when the flag is off.
        **error_callbacks(DEFAULT_CONFIG),
    )


# --- Batch (non-interactive) agent ------------------------------------------
#
# The SAME conversational architecture, adapted for unattended runs over
# CRM-sourced prospects: one prospect per turn, driven to a final
# recommendation/escalation with no human in the loop (see the batch driver and
# build_batch_instruction). It is a distinct ENTRY POINT, not a global config
# switch -- like scripts/run_batch.py, nothing on the interactive path builds it,
# so root_agent's behavior is untouched with no flag to thread. The batch agent
# is built explicitly by its driver, never lazily at import, so importing this
# module stays credential-free.


def _batch_agent_tools(config: Config) -> list:
    """Assemble the batch agent's tool list: the consolidated one-shot decision
    tool, the human-handoff record, and -- when Config.use_escalation_triage is on
    -- the escalation_triage AgentTool. Kept separate from build_batch_agent so the
    tool wiring can be asserted offline without resolving the LLM backend.

    Every pipeline tool is offloaded to a worker thread (see _offloaded_tool): its
    body is synchronous and may make a grounded LLM call that needs the running
    event loop free, so it must not run inline on that loop (the batch driver runs
    the agent on an asyncio loop, same constraint as the web app's)."""
    tools = [
        FunctionTool(_offloaded_tool(assign_prospect)),
        request_input,
    ]
    if config.use_escalation_triage:
        # Imported lazily so the package import stays credential-free -- this runs
        # only while the batch agent is being built, which already resolves the
        # backend via get_llm in build_batch_agent.
        from smart_assignment.triage import build_triage_tool

        tools.append(build_triage_tool(config))
    return tools


def build_batch_agent(config: Config = DEFAULT_CONFIG) -> LlmAgent:
    """Construct the BATCH variant of the agent: same architecture, model role, and
    (when enabled) escalation-triage AgentTool as root_agent, but a consolidated
    single-tool flow (`assign_prospect`) and a non-interactive instruction for
    one-prospect-per-turn runs. Resolves the LLM backend (get_llm), so this needs
    credentials for the configured backend -- build it only from the batch entry
    point (its driver/script), never at import.

    ``config`` is injectable so a caller/test can, e.g., turn triage off; it
    defaults to DEFAULT_CONFIG so the batch entry point needs no wiring."""
    # Install tracing before the agent runs, exactly as _build_root_agent does, so
    # batch turns and tool calls are captured too. A no-op when tracing is off.
    tracing.configure_tracing(config)
    triage_enabled = config.use_escalation_triage

    return LlmAgent(
        name="smart_assignment_batch_agent",
        model=get_llm(config.for_role(ROLE_ROOT_AGENT)),
        description=(
            "Assigns one CRM-sourced prospect customer to a delivery route and slot "
            "non-interactively, in a single consolidated step."
        ),
        instruction=build_batch_instruction(include_triage=triage_enabled),
        tools=_batch_agent_tools(config),
        # Same error recovery as root_agent. Safe for an unattended run: the driver
        # keys its outcome off the DECISION stored in session state, so a turn that
        # ends early without one still degrades to the deterministic pipeline
        # exactly as it does today (see batch/agent_runner._run_one_via_agent).
        **error_callbacks(config),
    )


def __getattr__(name: str) -> object:
    # PEP 562: resolve `root_agent` on first attribute access rather than at
    # import time, so importing the package stays credential-free.
    if name == "root_agent":
        global _root_agent
        if _root_agent is None:
            _root_agent = _build_root_agent()
        return _root_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# `root_agent` is exposed via module __getattr__ above, not a static binding.
__all__ = ["root_agent"]  # noqa: F822
