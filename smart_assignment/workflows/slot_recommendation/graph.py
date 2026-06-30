"""
Graph assembly for the slot_recommendation workflow.

    START
      -> geocode_and_cluster_customer        (code)
      -> fetch_candidate_slots_node           (code)
      -> filter_feasible_slots_node           (code)
      -> route_on_feasibility                 (code, conditional router)
           "NO_OPTIONS"  -> escalate_no_feasible_slot   (human input)
           "HAS_OPTIONS" -> recommend_slot_agent        (LLM)
                              -> confidence_gate          (code, conditional router)
                                   "LOW_CONFIDENCE" -> escalate_low_confidence (human input)
                                   "HIGH_CONFIDENCE" -> format_output          (code)

Design rationale:
  - Constraint checking (capacity, hours, temp zone, overtime) is 100%
    deterministic code -- enforced structurally by the graph, not just by
    prompting. See shared/tools.py.
  - Exactly one LLM call happens, ONLY over the already-feasible options,
    to make the judgment call that genuinely benefits from reasoning.
  - Two distinct human escalation paths are modeled because they're
    operationally different problems: no feasible slot at all (likely
    needs a manual routing decision) vs. a feasible slot the model isn't
    confident about (needs a sanity check, not a redesign).

Data flow note [VERIFIED against ADK 2.0 graph data-handling docs]:
  - `Event.output` passes data ONLY to the immediately next node.
  - `Event.state` persists across the whole session/graph run -- used here
    to carry the customer profile across the LLM hop.
"""

from __future__ import annotations

from google.adk import Workflow

from smart_assignment.workflows.slot_recommendation.nodes import (
    confidence_gate,
    escalate_low_confidence,
    escalate_no_feasible_slot,
    fetch_candidate_slots_node,
    filter_feasible_slots_node,
    format_output,
    geocode_and_cluster_customer,
    prepare_recommendation_prompt,
    recommend_slot_agent,
    route_on_feasibility,
)

root_agent = Workflow(
    name="delivery_slot_recommendation_workflow",
    edges=[
        (
            "START",
            geocode_and_cluster_customer,
            fetch_candidate_slots_node,
            filter_feasible_slots_node,
            route_on_feasibility,
        ),
        (
            route_on_feasibility,
            {
                "NO_OPTIONS": escalate_no_feasible_slot,
                "HAS_OPTIONS": (
                    prepare_recommendation_prompt,
                    recommend_slot_agent,
                    confidence_gate,
                ),
            },
        ),
        (
            confidence_gate,
            {
                "LOW_CONFIDENCE": escalate_low_confidence,
                "HIGH_CONFIDENCE": format_output,
            },
        ),
    ],
)
