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

Currently one agent calling its tools in strict sequence -- there is no
multi-agent split yet. If a step later needs to become its own sub-agent
(e.g. a richer intake agent, or a separate Q&A agent over past
recommendations), wrap that tool's function in its own `LlmAgent` and
expose it here via `google.adk.tools.AgentTool` -- the tool functions
themselves don't need to change, since each is already independent and
keyed only through session state (see tools/slot_recommendation.py).
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, request_input

from smart_assignment.prompts import INSTRUCTION
from smart_assignment.shared.config import DEFAULT_CONFIG
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
    return LlmAgent(
        name="smart_assignment_agent",
        model=get_llm(DEFAULT_CONFIG),
        description=(
            "Collects a new prospect customer's delivery details conversationally "
            "and recommends -- or escalates -- a delivery route and slot."
        ),
        instruction=INSTRUCTION,
        tools=[
            FunctionTool(intake_customer),
            FunctionTool(find_candidate_routes),
            FunctionTool(evaluate_and_score_routes),
            FunctionTool(recommend_or_escalate),
            request_input,
        ],
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
