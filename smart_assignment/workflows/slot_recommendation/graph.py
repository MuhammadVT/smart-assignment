"""
ADK graph assembly for the slot_recommendation workflow — the deployable
(`adk run` / `adk web` / `adk deploy`) wrapper around the pipeline.

    START
      -> geo_lookup_node            (intake + geocode + Top-N nearest routes)
      -> constraint_and_score_node  (hard constraints, then weighted scoring)
      -> route_on_feasibility       (conditional router)
           "NO_OPTIONS"  -> escalate_no_feasible_slot    (human input)
           "HAS_OPTIONS" -> build_recommendation_node
                              -> confidence_gate           (conditional router)
                                   "LOW_CONFIDENCE"  -> escalate_low_confidence (human input)
                                   "HIGH_CONFIDENCE" -> format_output

All node logic delegates to pipeline.py, so this graph and the offline demo
share one implementation. The local demo (`scripts/run_local.py`) drives the
pipeline directly and needs no API key; this graph is the ADK deployment form.
"""

from __future__ import annotations

from google.adk import Workflow

from smart_assignment.workflows.slot_recommendation.nodes import (
    build_recommendation_node,
    confidence_gate,
    constraint_and_score_node,
    escalate_low_confidence,
    escalate_no_feasible_slot,
    format_output,
    geo_lookup_node,
    route_on_feasibility,
)

root_agent = Workflow(
    name="smart_assignment_slot_recommendation",
    edges=[
        (
            "START",
            geo_lookup_node,
            constraint_and_score_node,
            route_on_feasibility,
        ),
        (
            route_on_feasibility,
            {
                "NO_OPTIONS": escalate_no_feasible_slot,
                "HAS_OPTIONS": (
                    build_recommendation_node,
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
