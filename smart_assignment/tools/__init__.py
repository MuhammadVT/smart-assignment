"""
ADK tools exposed to `root_agent` (see smart_assignment/agent.py).

Each tool wraps exactly one step of the deterministic slot-recommendation
pipeline (smart_assignment/pipeline.py) -- see slot_recommendation.py's
module docstring for why that split keeps the agent auditable and testable
without an LLM.
"""

from smart_assignment.tools.slot_recommendation import (
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
)

__all__ = [
    "intake_customer",
    "find_candidate_routes",
    "evaluate_and_score_routes",
    "recommend_or_escalate",
]
