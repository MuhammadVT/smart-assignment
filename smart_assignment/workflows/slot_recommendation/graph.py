"""
ADK graph assembly for the slot_recommendation workflow — the deployable
(`adk run` / `adk web` / `adk deploy`) wrapper around the pipeline.

    START
      -> intake_node                (resolve customer + geocode + Top-N nearest routes)
      -> constraint_and_score_node  (hard constraints, then weighted scoring)
      -> route_on_feasibility       (conditional router)
           "NO_OPTIONS"  -> escalate_no_feasible_slot    (human input)
           "HAS_OPTIONS" -> build_recommendation_node
                              -> total_score_gate          (conditional router)
                                   "LOW_SCORE"  -> escalate_low_score (human input)
                                   "HIGH_SCORE" -> format_output

All node logic delegates to pipeline.py, so this graph and the offline demo
share one implementation. The local demo (`scripts/run_local.py`) drives the
pipeline directly and needs no API key; this graph is the ADK deployment form.
"""

from __future__ import annotations

from google.adk import Workflow

from smart_assignment.workflows.slot_recommendation.nodes import (
    build_recommendation_node,
    constraint_and_score_node,
    escalate_low_score,
    escalate_no_feasible_slot,
    format_output,
    intake_node,
    route_on_feasibility,
    total_score_gate,
)

root_agent = Workflow(
    name="smart_assignment_slot_recommendation",
    edges=[
        (
            "START",
            intake_node,
            constraint_and_score_node,
            route_on_feasibility,
        ),
        # A tuple as a routing-map value is a fan-out, not a chain — so the
        # HAS_OPTIONS branch points at a single node, and the follow-on
        # build -> gate step is expressed as its own chain edge below.
        (
            route_on_feasibility,
            {
                "NO_OPTIONS": escalate_no_feasible_slot,
                "HAS_OPTIONS": build_recommendation_node,
            },
        ),
        (build_recommendation_node, total_score_gate),
        (
            total_score_gate,
            {
                "LOW_SCORE": escalate_low_score,
                "HIGH_SCORE": format_output,
            },
        ),
    ],
)
