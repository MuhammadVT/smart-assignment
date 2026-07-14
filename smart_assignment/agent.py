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

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, request_input

from smart_assignment.prompts import build_instruction
from smart_assignment.shared.config import DEFAULT_CONFIG, ROLE_ROOT_AGENT
from smart_assignment.shared.llm import get_llm
from smart_assignment.tools import (
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
)

# Cached after first access so repeated lookups return the same agent instance.
_root_agent: LlmAgent = None  # type: ignore[assignment]


def _build_root_agent() -> LlmAgent:
    """Construct the conversational agent. Resolves the LLM backend (``get_llm``),
    so this needs credentials for the configured backend -- called only on first
    access to ``root_agent``, never at import."""
    triage_enabled = DEFAULT_CONFIG.use_escalation_triage

    tools = [
        FunctionTool(intake_customer),
        FunctionTool(find_candidate_routes),
        FunctionTool(evaluate_and_score_routes),
        FunctionTool(recommend_or_escalate),
        request_input,
    ]
    if triage_enabled:
        # Imported lazily so the package import stays credential-free -- this
        # runs only while root_agent is being built, which already resolves the
        # backend via get_llm above.
        from smart_assignment.triage import build_triage_tool

        tools.append(build_triage_tool(DEFAULT_CONFIG))

    return LlmAgent(
        name="smart_assignment_agent",
        model=get_llm(DEFAULT_CONFIG.for_role(ROLE_ROOT_AGENT)),
        description=(
            "Collects a new prospect customer's delivery details conversationally "
            "and recommends -- or escalates -- a delivery route and slot."
        ),
        instruction=build_instruction(include_triage=triage_enabled),
        tools=tools,
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
