"""
The route-slot decision: pick the best (route, slot) option and decide
recommend-vs-escalate. This supersedes the two-stage "judge the route, then pick
the slot" flow when `Config.use_route_slot_scoring` is on -- the slot choice is
absorbed into one grounded decision over route-slot options.

  - Deterministic (always the floor): the highest-total route-slot; escalate if
    its own total is below `route_slot_score_threshold`.
  - Grounded (`use_grounded_judgment` on): an LLM picks a route-slot from the
    enumerated options (constrained to the valid set, grounded + verified), with
    the deterministic best as reference and fallback. The recommend/escalate gate
    stays deterministic (a threshold on the chosen option's own total), so the
    high-stakes auto-assign decision remains auditable.

Any parse/verify/backend failure falls back to the deterministic best, logged --
so it is never worse than the flag being off.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from smart_assignment.routeslot.evidence import (
    RouteSlotOption,
    RouteSlotPacket,
    build_route_slot_packet,
)
from smart_assignment.routeslot.llm import generate_route_slot_choice
from smart_assignment.routeslot.prompts import (
    build_route_slot_prompt,
    build_route_slot_retry_prompt,
)
from smart_assignment.routeslot.schema import parse_route_slot_choice
from smart_assignment.routeslot.verifier import verify_choice
from smart_assignment.shared.config import Config
from smart_assignment.shared.constraints import CONSTRAINT_LABEL
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    Decision,
    SlotRecommendation,
)
from smart_assignment.shared.timeutils import fmt_window

logger = logging.getLogger(__name__)

# A choice_fn turns (config, prompt) into a raw route-slot-choice dict. Injectable
# so tests drive the grounded path with a fake and no network/credentials.
ChoiceFn = Callable[[Config, str], dict]


def decide_route_slot(
    customer: CustomerProfile,
    evaluations: list[CandidateEvaluation],
    config: Config,
    choice_fn: Optional[ChoiceFn] = None,
) -> SlotRecommendation:
    """Decide over route-slot options. `choice_fn` is only consulted when
    `config.use_grounded_judgment` is on; it defaults to the real backend call."""
    if not any(e.feasible for e in evaluations):
        return _no_feasible(customer, evaluations)

    packet = build_route_slot_packet(customer, evaluations, config)
    if packet.n == 0:  # feasible routes but no candidate slots -- defensive
        return _no_feasible(customer, evaluations)

    grounded_rationale: Optional[str] = None
    grounded_fallback_reason: Optional[str] = None
    index = 0  # deterministic best (packet is sorted by descending total)

    if config.use_grounded_judgment:
        picked, rationale, reason = _grounded_index(packet, config, choice_fn)
        if picked is not None:
            index, grounded_rationale = picked, rationale
        else:
            grounded_fallback_reason = reason

    chosen = packet.option_at(index)
    rec = _to_recommendation(customer, chosen, packet, config, grounded_rationale)
    if grounded_fallback_reason is not None:
        rec.grounded_fallback = True
        rec.grounded_fallback_reason = grounded_fallback_reason
    return rec


# --- grounded selection ------------------------------------------------------


def _grounded_index(
    packet: RouteSlotPacket, config: Config, choice_fn: Optional[ChoiceFn]
):
    """Return (index, rationale, None) on a verified grounded pick, or
    (None, None, reason) to signal a deterministic fallback."""
    fn = choice_fn or generate_route_slot_choice
    try:
        choice = parse_route_slot_choice(fn(config, build_route_slot_prompt(packet)))
        result = verify_choice(choice, packet)
        if not result.ok:
            retry = build_route_slot_retry_prompt(packet, result.as_feedback())
            choice = parse_route_slot_choice(fn(config, retry))
            result = verify_choice(choice, packet)
    except Exception as exc:  # noqa: BLE001 - any backend/parse failure -> fallback
        logger.warning(
            "Grounded route-slot decision failed (%s: %s); using the deterministic "
            "best route-slot. Check SMART_ASSIGNMENT_LLM_BACKEND and its credentials.",
            type(exc).__name__,
            exc,
        )
        return None, None, (
            "Grounded LLM reasoning was unavailable, so this shows the deterministic "
            "best route-slot. Check the LLM backend and its credentials."
        )
    if not result.ok:
        logger.warning(
            "Grounded route-slot choice ungrounded after one retry (%s); using the "
            "deterministic best route-slot.",
            result.as_feedback(),
        )
        return None, None, (
            "Grounded route-slot reasoning could not be verified; showing the "
            "deterministic best route-slot."
        )
    return choice.chosen_index, choice.rationale, None


# --- mapping to a SlotRecommendation -----------------------------------------


def _to_recommendation(
    customer: CustomerProfile,
    chosen: RouteSlotOption,
    packet: RouteSlotPacket,
    config: Config,
    grounded_rationale: Optional[str],
) -> SlotRecommendation:
    ev = chosen.evaluation
    route = ev.route
    scored = chosen.scored
    total = round(scored.total_score, 2)
    escalate = scored.total_score < config.route_slot_score_threshold
    decision = Decision.ESCALATED_LOW_SCORE if escalate else Decision.RECOMMENDED

    reasoning = grounded_rationale or _deterministic_reasoning(chosen)
    return SlotRecommendation(
        customer_number=customer.customer_number,
        customer_address=customer.address,
        customer_name=customer.name,
        decision=decision,
        total_score=total,
        reasoning=reasoning,
        recommended_route_id=route.route_id,
        recommended_route_name=route.name,
        recommended_day=route.day.value,
        recommended_window=fmt_window(scored.slot.window),
        recommended_window_basis=scored.slot.basis or None,
        recommended_window_rationale=grounded_rationale,
        factor_breakdown=scored.factor_scores,
        rejected_alternatives=_rejected(packet, chosen),
        review_reason=(
            f"Best route-slot scored {total:.0%}, below the "
            f"{config.route_slot_score_threshold:.0%} auto-assign bar."
            if escalate
            else None
        ),
    )


def _deterministic_reasoning(chosen: RouteSlotOption) -> str:
    route = chosen.evaluation.route
    top = max(chosen.scored.factor_scores, key=lambda fs: fs.weighted)
    return (
        f"{route.name} ({route.day.value}) at {fmt_window(chosen.scored.slot.window)} is the "
        f"strongest route-slot overall; {top.detail}."
    )


def _rejected(packet: RouteSlotPacket, chosen: RouteSlotOption) -> list[str]:
    out: list[str] = []
    for i, opt in enumerate(packet.options):
        p = packet.option_at(i)
        if p is chosen:
            continue
        out.append(
            f"{opt['route_id']} ({opt['day']}) {opt['window']}: route-slot scored "
            f"{opt['facts']['reference_weighted_score']:.2f}"
        )
    for c in packet.infeasible:
        failed = ", ".join(fc["name"] for fc in c.get("failed_constraints", []))
        out.append(f"{c['route_id']} ({c['day']}): infeasible — {failed}")
    return out


def _no_feasible(
    customer: CustomerProfile, evaluations: list[CandidateEvaluation]
) -> SlotRecommendation:
    rejected = []
    for ev in evaluations:
        if not ev.feasible:
            failed = ", ".join(
                CONSTRAINT_LABEL.get(c.name, c.name) for c in ev.failed_constraints
            )
            rejected.append(f"{ev.route.route_id} ({ev.route.day.value}): infeasible — {failed}")
    return SlotRecommendation(
        customer_number=customer.customer_number,
        customer_address=customer.address,
        customer_name=customer.name,
        decision=Decision.ESCALATED_NO_FEASIBLE_SLOT,
        total_score=0.0,
        reasoning="No candidate route satisfied all hard constraints.",
        rejected_alternatives=rejected,
        review_reason="No candidate route satisfied all hard constraints.",
    )
