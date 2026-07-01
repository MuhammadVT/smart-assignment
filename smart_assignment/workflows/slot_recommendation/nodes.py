"""
ADK graph nodes for the slot_recommendation workflow.

These are a thin adapter over the plain-Python pipeline in `pipeline.py`: each
node delegates to the same step functions the offline demo uses, so the ADK
deployment path and the local demo can never disagree on business logic. The
graph wiring itself lives in `graph.py`.

Node flow (mirrors pipeline.py):
  geo_lookup_node -> constraint_and_score_node -> route_on_feasibility
    NO_OPTIONS  -> escalate_no_feasible_slot            (human-in-the-loop)
    HAS_OPTIONS -> build_recommendation_node -> confidence_gate
                     LOW_CONFIDENCE  -> escalate_low_confidence  (human-in-the-loop)
                     HIGH_CONFIDENCE -> format_output
"""

from __future__ import annotations

from google.adk import Event
from google.adk.agents.invocation_context import InvocationContext as Context
from google.adk.events import RequestInput

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.models import CustomerProfile, Decision, SlotRecommendation
from smart_assignment.workflows.slot_recommendation.pipeline import (
    decide,
    evaluate_candidates,
    geo_lookup,
    intake,
)
from smart_assignment.workflows.slot_recommendation.reasoning import LLMReasoner

# --- Step 1 + 2: intake + geo-lookup (entry node) --------------------------


def geo_lookup_node(node_input: CustomerProfile) -> Event:
    customer = intake(node_input)
    candidates = geo_lookup(customer, fetch_candidate_routes(), MockGeocoder(), DEFAULT_CONFIG)
    return Event(output=candidates, state={"customer": customer})


# --- Step 3 + 4: hard constraints then weighted scoring --------------------


def constraint_and_score_node(node_input: list, ctx: Context) -> Event:
    customer: CustomerProfile = ctx.state["customer"]
    evaluations = evaluate_candidates(customer, node_input, DEFAULT_CONFIG)
    return Event(output=evaluations, state={"evaluations": evaluations})


# --- Conditional router: any feasible slot at all? -------------------------


def route_on_feasibility(node_input: list) -> Event:
    has_feasible = any(getattr(e, "feasible", False) for e in node_input)
    return Event(route=["HAS_OPTIONS" if has_feasible else "NO_OPTIONS"], output=node_input)


def escalate_no_feasible_slot(ctx: Context):
    customer: CustomerProfile = ctx.state["customer"]
    rec = decide(customer, ctx.state["evaluations"], LLMReasoner(DEFAULT_CONFIG), DEFAULT_CONFIG)
    yield RequestInput(message=rec.reasoning, response_schema=str)


# --- Step 5: build recommendation, then gate on confidence -----------------


def build_recommendation_node(node_input: list, ctx: Context) -> Event:
    customer: CustomerProfile = ctx.state["customer"]
    rec = decide(customer, node_input, LLMReasoner(DEFAULT_CONFIG), DEFAULT_CONFIG)
    return Event(output=rec, state={"recommendation": rec})


def confidence_gate(node_input: SlotRecommendation, ctx: Context) -> Event:
    route = "HIGH_CONFIDENCE" if node_input.decision == Decision.RECOMMENDED else "LOW_CONFIDENCE"
    return Event(route=[route], output=node_input)


def escalate_low_confidence(node_input: SlotRecommendation, ctx: Context):
    yield RequestInput(message=node_input.reasoning, response_schema=str)


def format_output(node_input: SlotRecommendation, ctx: Context) -> SlotRecommendation:
    return node_input
