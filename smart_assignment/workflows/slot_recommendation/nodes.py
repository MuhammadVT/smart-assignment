"""
Function nodes (deterministic) and the single LLM Agent node for the
slot_recommendation workflow. The graph wiring itself lives in graph.py.

Node responsibilities, matching the architecture documented in graph.py:
  geocode_and_cluster_customer  -> fetch_candidate_slots_node
  -> filter_feasible_slots_node -> route_on_feasibility (conditional)
  -> [escalate_no_feasible_slot] OR [prepare_recommendation_prompt
      -> recommend_slot_agent -> confidence_gate (conditional)
      -> escalate_low_confidence OR format_output]
"""

from __future__ import annotations

from google.adk import Agent, Event
from google.adk.agents.invocation_context import InvocationContext as Context
from google.adk.events import RequestInput

from smart_assignment.integrations.route_capacity_client import fetch_candidate_route_slots
from smart_assignment.shared.config import (
    DEFAULT_MODEL,
    SLOT_RECOMMENDATION_CONFIDENCE_THRESHOLD,
)
from smart_assignment.shared.models import (
    CustomerProfile,
    FeasibleSlotOption,
    SlotRecommendation,
)
from smart_assignment.shared.tools import (
    filter_feasible_slots,
)
from smart_assignment.shared.tools import (
    geocode_and_cluster_customer as _geocode_and_cluster_customer,
)
from smart_assignment.workflows.slot_recommendation.prompts import (
    RECOMMEND_SLOT_INSTRUCTION,
    SlotPromptInput,
    SlotRecommendationOutput,
)

# ---------------------------------------------------------------------------
# NODE 1: geocode + cluster (entry node -- receives the raw CustomerProfile)
# ---------------------------------------------------------------------------


def geocode_and_cluster_customer(node_input: CustomerProfile) -> Event:
    result = _geocode_and_cluster_customer(node_input)
    # Stash the customer + zone in session state so it survives the LLM hop;
    # also forward zone_id as the direct output for the next node.
    return Event(
        output=result["zone_id"],
        state={"customer": result["customer"], "zone_id": result["zone_id"]},
    )


# ---------------------------------------------------------------------------
# NODE 2: fetch candidate routes
# ---------------------------------------------------------------------------


def fetch_candidate_slots_node(node_input: str, ctx: Context) -> Event:
    zone_id = node_input
    customer: CustomerProfile = ctx.state["customer"]
    routes = fetch_candidate_route_slots(zone_id, customer)
    return Event(output=routes)


# ---------------------------------------------------------------------------
# NODE 3: hard constraint filtering
# ---------------------------------------------------------------------------


def filter_feasible_slots_node(node_input: list, ctx: Context) -> Event:
    candidate_routes = node_input
    customer: CustomerProfile = ctx.state["customer"]
    feasible = filter_feasible_slots(customer, candidate_routes)
    return Event(
        output=feasible,
        state={"feasible_options": feasible},
    )


# ---------------------------------------------------------------------------
# Conditional router: are there any feasible slots at all?
# ---------------------------------------------------------------------------


def route_on_feasibility(node_input: list) -> Event:
    feasible_options: list[FeasibleSlotOption] = node_input
    if not feasible_options:
        return Event(route=["NO_OPTIONS"])
    return Event(route=["HAS_OPTIONS"])


def escalate_no_feasible_slot(ctx: Context):
    """Human-in-the-loop: no slot satisfies hard constraints at all."""
    customer: CustomerProfile = ctx.state["customer"]
    message = (
        f"No feasible delivery slot found for new customer "
        f"{customer.name} ({customer.customer_id}) at {customer.address}.\n"
        f"Required volume: {customer.weekly_order_volume_cases} cases, "
        f"product zone: {customer.product_temp_zone}.\n"
        f"All candidate routes in this zone are at capacity, temperature-"
        f"incompatible, or would require driver overtime. This likely "
        f"requires a manual routing decision (e.g. new route, schedule "
        f"adjustment, or capacity reallocation)."
    )
    yield RequestInput(message=message, response_schema=str)


# ---------------------------------------------------------------------------
# LLM node: rank feasible options and recommend, with explicit reasoning
# ---------------------------------------------------------------------------


def _serialize_options_for_prompt(
    customer: CustomerProfile, options: list[FeasibleSlotOption]
) -> str:
    lines = [
        f"Customer: {customer.name} ({customer.customer_id})",
        f"Order volume: {customer.weekly_order_volume_cases} cases, "
        f"product zone: {customer.product_temp_zone}",
        f"Stated preference - days: {customer.requested_days}, "
        f"time window: {customer.requested_time_window}",
        "",
        "Feasible options (all have already passed hard constraint checks "
        "for capacity, temperature compatibility, and driver hours):",
    ]
    for i, opt in enumerate(options, start=1):
        rs = opt.route_slot
        lines.append(
            f"{i}. route_id={rs.route_id}, day={rs.day.value}, "
            f"window={opt.proposed_arrival_window[0].strftime('%H:%M')}-"
            f"{opt.proposed_arrival_window[1].strftime('%H:%M')}, "
            f"capacity_utilization_after={opt.capacity_utilization_after:.0%}, "
            f"geographic_fit_score={opt.geographic_fit_score:.2f}, "
            f"matches_customer_preference={opt.matches_customer_preference}, "
            f"remaining_capacity_after={opt.remaining_capacity_after_assignment} cases"
        )
    return "\n".join(lines)


def prepare_recommendation_prompt(node_input: list, ctx: Context) -> SlotPromptInput:
    customer: CustomerProfile = ctx.state["customer"]
    options: list[FeasibleSlotOption] = node_input
    return SlotPromptInput(prompt_text=_serialize_options_for_prompt(customer, options))


recommend_slot_agent = Agent(
    name="recommend_slot_agent",
    model=DEFAULT_MODEL,
    instruction=RECOMMEND_SLOT_INSTRUCTION,
    input_schema=SlotPromptInput,
    output_schema=SlotRecommendationOutput,
)


# ---------------------------------------------------------------------------
# Conditional router: confidence gate
# ---------------------------------------------------------------------------


def confidence_gate(node_input: SlotRecommendationOutput, ctx: Context) -> Event:
    if node_input.confidence < SLOT_RECOMMENDATION_CONFIDENCE_THRESHOLD:
        return Event(route=["LOW_CONFIDENCE"], output=node_input)
    return Event(route=["HIGH_CONFIDENCE"], output=node_input)


def escalate_low_confidence(node_input: SlotRecommendationOutput, ctx: Context):
    rec = node_input
    customer: CustomerProfile = ctx.state["customer"]
    message = (
        f"Low-confidence slot recommendation for new customer "
        f"{customer.name} ({customer.customer_id}) -- needs human review "
        f"before committing.\n\n"
        f"Proposed: route {rec.recommended_route_id}, {rec.recommended_day}, "
        f"{rec.recommended_window_start}-{rec.recommended_window_end} "
        f"(confidence {rec.confidence:.0%})\n\n"
        f"Model reasoning: {rec.reasoning}\n\n"
        f"Approve, override, or request a different slot."
    )
    yield RequestInput(message=message, response_schema=str)


# ---------------------------------------------------------------------------
# Final formatting node
# ---------------------------------------------------------------------------


def format_output(node_input: SlotRecommendationOutput, ctx: Context) -> SlotRecommendation:
    rec = node_input
    customer: CustomerProfile = ctx.state["customer"]
    return SlotRecommendation(
        customer_id=customer.customer_id,
        recommended_route_id=rec.recommended_route_id,
        recommended_day=rec.recommended_day,
        recommended_window=f"{rec.recommended_window_start}-{rec.recommended_window_end}",
        confidence=rec.confidence,
        reasoning=rec.reasoning,
        rejected_alternatives=rec.rejected_alternatives,
        requires_human_review=False,
    )
