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
from smart_assignment.tools import (
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
)

root_agent = LlmAgent(
    name="smart_assignment_agent",
    model=DEFAULT_CONFIG.model,
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

__all__ = ["root_agent"]
