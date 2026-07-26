"""
ADK tools exposed to `root_agent` (see smart_assignment/agent.py).

Each tool wraps exactly one step of the deterministic slot-recommendation
pipeline (smart_assignment/pipeline.py) -- see slot_recommendation.py's
module docstring for why that split keeps the agent auditable and testable
without an LLM.
"""

from smart_assignment.tools.slot_recommendation import (
    assign_delivery_slot,
    evaluate_and_score_routes,
    find_candidate_routes,
    intake_customer,
    recommend_or_escalate,
    resolve_address,
)

__all__ = [
    "intake_customer",
    "find_candidate_routes",
    "resolve_address",
    "evaluate_and_score_routes",
    "recommend_or_escalate",
    # The consolidated alternative to the four step tools above: one call runs the
    # whole deterministic chain (see Config.use_consolidated_pipeline_tool). Both
    # shapes are always exported and tested; agent.py registers one set or the
    # other.
    "assign_delivery_slot",
]
