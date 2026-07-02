"""
ADK graph nodes for the slot_recommendation workflow.

These are a thin adapter over the plain-Python pipeline in `pipeline.py`: each
node delegates to the same step functions the offline demo uses, so the ADK
deployment path and the local demo can never disagree on business logic. The
graph wiring itself lives in `graph.py`.

Input note: `adk run` / `adk web` hand the workflow a free-text user message.
The entry node therefore accepts `str` and resolves it to one of the mock
Sysco customers **by customer number only** (format ``NNN-NNNNNN``) — e.g.
type `067-100002`. Names are never accepted as input. Replace
`resolve_customer` with real intake parsing / a form when wiring this to a real
front end.

Node flow (mirrors pipeline.py):
  intake_node -> constraint_and_score_node -> route_on_feasibility
    NO_OPTIONS  -> escalate_no_feasible_slot            (terminal text)
    HAS_OPTIONS -> build_recommendation_node -> total_score_gate
                     LOW_SCORE  -> escalate_low_score  (terminal text)
                     HIGH_SCORE -> format_output        (terminal text)
"""

from __future__ import annotations

from google.adk import Event
from google.adk.agents.invocation_context import InvocationContext as Context
from google.genai import types

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.customer import normalize_customer_number
from smart_assignment.shared.models import (
    CustomerProfile,
    Decision,
    SlotRecommendation,
)
from smart_assignment.workflows.slot_recommendation.pipeline import (
    decide,
    evaluate_candidates,
    geo_lookup,
    intake,
)
from smart_assignment.workflows.slot_recommendation.reasoning import LLMReasoner

_DECISION_MARK = {
    Decision.RECOMMENDED: "RECOMMENDED",
    Decision.ESCALATED_LOW_SCORE: "ESCALATE -> human review (low total score)",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "ESCALATE -> human specialist (no feasible slot)",
}


def resolve_customer(user_text: str) -> CustomerProfile:
    """Map a typed customer number (``NNN-NNNNNN``) to a mock customer.

    Names are not accepted — Sysco identifies customers by customer number.
    Unknown/blank input falls back to the first sample so the demo still runs.
    """
    query = normalize_customer_number(user_text)
    if query:
        for customer in SAMPLE_CUSTOMERS:
            if query == customer.customer_number:
                return customer
    return SAMPLE_CUSTOMERS[0]


def _render(rec: SlotRecommendation) -> types.Content:
    lines = [
        f"Customer: {rec.customer_name} ({rec.customer_number})",
        f"Decision: {_DECISION_MARK[rec.decision]}  |  total score {rec.total_score:.0%}",
    ]
    if rec.recommended_route_id:
        lines.append(
            f"Proposed slot: {rec.recommended_route_id} ({rec.recommended_route_name}), "
            f"{rec.recommended_day}, window {rec.recommended_window}"
        )
    if rec.factor_breakdown:
        factors = "  ".join(
            f"{f.name}={f.value:.2f}(w{f.weight:.2f})" for f in rec.factor_breakdown
        )
        lines.append(f"Score factors: {factors}")
    lines.append(f"Reasoning: {rec.reasoning}")
    if rec.rejected_alternatives:
        lines.append("Alternatives considered:")
        lines.extend(f"  - {alt}" for alt in rec.rejected_alternatives)
    return types.Content(role="model", parts=[types.Part(text="\n".join(lines))])


# NOTE: ADK persists session *state* as JSON, so state may hold only
# JSON-serializable values. We stash the customer_number (a string) and
# re-resolve the (module-level, already-geocoded) CustomerProfile in later
# nodes; richer evaluation objects are passed node-to-node via Event.output.


# --- Step 1 + 2: intake + geo-lookup (entry node, receives user text) ------


def intake_node(node_input: str) -> Event:
    customer = intake(resolve_customer(node_input))
    candidates = geo_lookup(customer, fetch_candidate_routes(), MockGeocoder(), DEFAULT_CONFIG)
    return Event(output=candidates, state={"customer_number": customer.customer_number})


# --- Step 3 + 4: hard constraints then weighted scoring --------------------


def constraint_and_score_node(node_input: list, ctx: Context) -> Event:
    customer = resolve_customer(ctx.state["customer_number"])
    evaluations = evaluate_candidates(customer, node_input, DEFAULT_CONFIG)
    return Event(output=evaluations)


# --- Conditional router: any feasible slot at all? -------------------------


def route_on_feasibility(node_input: list) -> Event:
    has_feasible = any(getattr(e, "feasible", False) for e in node_input)
    return Event(route=["HAS_OPTIONS" if has_feasible else "NO_OPTIONS"], output=node_input)


def escalate_no_feasible_slot(node_input: list, ctx: Context) -> types.Content:
    """Terminal: no slot satisfied the hard constraints (would page a specialist)."""
    customer = resolve_customer(ctx.state["customer_number"])
    rec = decide(customer, node_input, LLMReasoner(DEFAULT_CONFIG), DEFAULT_CONFIG)
    return _render(rec)


# --- Step 5: build recommendation, then gate on total score ----------------


def build_recommendation_node(node_input: list, ctx: Context) -> Event:
    customer = resolve_customer(ctx.state["customer_number"])
    rec = decide(customer, node_input, LLMReasoner(DEFAULT_CONFIG), DEFAULT_CONFIG)
    return Event(output=rec)


def total_score_gate(node_input: SlotRecommendation, ctx: Context) -> Event:
    route = "HIGH_SCORE" if node_input.decision == Decision.RECOMMENDED else "LOW_SCORE"
    return Event(route=[route], output=node_input)


def escalate_low_score(node_input: SlotRecommendation, ctx: Context) -> types.Content:
    """Terminal: a slot is proposed but flagged for human review."""
    return _render(node_input)


def format_output(node_input: SlotRecommendation, ctx: Context) -> types.Content:
    return _render(node_input)
