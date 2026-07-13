"""
The route-slot decision: pick the best (route, slot) option and decide
recommend-vs-escalate. This supersedes the two-stage "judge the route, then pick
the slot" flow when `Config.use_route_slot_scoring` is on -- the slot choice is
absorbed into one grounded decision over route-slot options.

The recommend-vs-escalate boundary is a DETERMINISTIC threshold
(`route_slot_score_threshold`): a route-slot must clear it to auto-assign. The LLM
reasons only over the route-slots that already clear the bar -- so its pick is
always auto-assignable, and it can never *cause* an escalation. Escalation is
decided before/without the LLM:

  - no feasible route at all          -> ESCALATED_NO_FEASIBLE_SLOT
  - feasible route, but no slot built -> ESCALATED_NO_FEASIBLE_SLOT (distinct reason)
  - feasible route-slots, none clear  -> ESCALATED_LOW_SCORE (deterministic best
                                         proposed for the specialist; no LLM call)
  - >= 1 route-slot clears the bar     -> RECOMMENDED (LLM picks among the eligible
                                         ones when grounded; else the deterministic
                                         best), with verify + deterministic fallback.
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
from smart_assignment.routeslot.schema import (
    VERDICT_AGREE,
    RouteSlotChoice,
    parse_route_slot_choice,
)
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
    `config.use_grounded_judgment` is on AND at least one route-slot clears the
    auto-assign bar; it defaults to the real backend call."""
    if not any(e.feasible for e in evaluations):
        return _escalate_no_feasible(customer, evaluations)

    all_pairs = _all_route_slots(evaluations)
    if not all_pairs:
        # Serviceable route(s), but no delivery window could be constructed.
        return _escalate_no_slot(customer, evaluations)

    threshold = config.route_slot_score_threshold
    eligible = [p for p in all_pairs if p.scored.total_score >= threshold]
    if not eligible:
        # Nothing clears the bar -- escalate, proposing the deterministic best for
        # the specialist. The LLM is not consulted (it only reasons over recommendable
        # options).
        return _escalate_low_score(customer, all_pairs, evaluations, config)

    # >= 1 route-slot is auto-assignable. The LLM reasons over ONLY these.
    packet = build_route_slot_packet(customer, evaluations, config, min_score=threshold)
    index = 0  # deterministic best (packet is sorted by descending total)
    grounded_choice: Optional[RouteSlotChoice] = None
    grounded_fallback_reason: Optional[str] = None
    if config.use_grounded_judgment:
        picked, choice, reason = _grounded_index(packet, config, choice_fn)
        if picked is not None:
            index, grounded_choice = picked, choice
        else:
            grounded_fallback_reason = reason

    chosen = packet.option_at(index)
    rec = _recommend(customer, chosen, all_pairs, evaluations, packet, grounded_choice)
    if grounded_fallback_reason is not None:
        rec.grounded_fallback = True
        rec.grounded_fallback_reason = grounded_fallback_reason
    return rec


def _all_route_slots(evaluations: list[CandidateEvaluation]) -> list[RouteSlotOption]:
    pairs = [
        RouteSlotOption(evaluation=ev, scored=s)
        for ev in evaluations
        if ev.feasible
        for s in ev.scored_slots
    ]
    pairs.sort(key=lambda p: p.scored.total_score, reverse=True)
    return pairs


# --- grounded selection (over the eligible, above-threshold options) ---------


def _grounded_index(
    packet: RouteSlotPacket, config: Config, choice_fn: Optional[ChoiceFn]
):
    """Return (index, choice, None) on a verified grounded pick, or
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
    return choice.chosen_index, choice, None


# --- recommendation + escalations --------------------------------------------


def _recommend(
    customer: CustomerProfile,
    chosen: RouteSlotOption,
    all_pairs: list[RouteSlotOption],
    evaluations: list[CandidateEvaluation],
    packet: RouteSlotPacket,
    choice: Optional[RouteSlotChoice],
) -> SlotRecommendation:
    ev = chosen.evaluation
    route = ev.route
    scored = chosen.scored

    rec = SlotRecommendation(
        customer_number=customer.customer_number,
        customer_address=customer.address,
        customer_name=customer.name,
        decision=Decision.RECOMMENDED,
        total_score=round(scored.total_score, 2),
        reasoning=_deterministic_reasoning(chosen),
        recommended_route_id=route.route_id,
        recommended_route_name=route.name,
        recommended_day=route.day.value,
        recommended_window=fmt_window(scored.slot.window),
        recommended_window_basis=scored.slot.basis or None,
        factor_breakdown=scored.factor_scores,
        rejected_alternatives=_rejected(all_pairs, _key(chosen), evaluations),
        review_reason=None,
    )
    if choice is not None:
        _apply_grounded_narrative(rec, choice, packet)
    return rec


def _apply_grounded_narrative(
    rec: SlotRecommendation, choice: RouteSlotChoice, packet: RouteSlotPacket
) -> None:
    """Fold the verified grounded choice's structured explanation onto the
    recommendation, and compose the flat `reasoning` string from it so existing
    consumers keep working. Only reached on a successful grounded pick."""
    rec.decision_summary = choice.decision_summary
    rec.primary_reasons = list(choice.primary_reasons)
    rec.key_tradeoff = choice.key_tradeoff or None
    rec.runner_up = _render_runner_up(choice, packet)
    rec.default_comparison = _render_default_comparison(choice)

    parts = [choice.decision_summary, *choice.primary_reasons]
    if choice.key_tradeoff:
        parts.append(f"Trade-off: {choice.key_tradeoff}")
    reasoning = " ".join(p.strip() for p in parts if p and p.strip())
    rec.reasoning = reasoning
    rec.recommended_window_rationale = reasoning


def _render_runner_up(choice: RouteSlotChoice, packet: RouteSlotPacket) -> Optional[str]:
    ru = choice.runner_up
    if ru is None:
        return None
    opt = packet.options[ru.index] if 0 <= ru.index < packet.n else None
    if opt is None:
        return ru.why_not
    return f"{opt['route_name']} ({opt['day']}) {opt['window']} — {ru.why_not}"


def _render_default_comparison(choice: RouteSlotChoice) -> Optional[str]:
    cmp = choice.vs_deterministic_default
    if cmp is None:
        return None
    if cmp.verdict == VERDICT_AGREE:
        return "Agreed with the weighted-heuristic default."
    note = f" — {cmp.note}" if cmp.note else ""
    return f"Diverged from the weighted-heuristic default{note}"


def _escalate_low_score(
    customer: CustomerProfile,
    all_pairs: list[RouteSlotOption],
    evaluations: list[CandidateEvaluation],
    config: Config,
) -> SlotRecommendation:
    best = all_pairs[0]  # highest total, but below the bar
    ev = best.evaluation
    scored = best.scored
    bar = config.route_slot_score_threshold
    return SlotRecommendation(
        customer_number=customer.customer_number,
        customer_address=customer.address,
        customer_name=customer.name,
        decision=Decision.ESCALATED_LOW_SCORE,
        total_score=round(scored.total_score, 2),
        reasoning=_deterministic_reasoning(best),
        recommended_route_id=ev.route.route_id,
        recommended_route_name=ev.route.name,
        recommended_day=ev.route.day.value,
        recommended_window=fmt_window(scored.slot.window),
        recommended_window_basis=scored.slot.basis or None,
        factor_breakdown=scored.factor_scores,
        rejected_alternatives=_rejected(all_pairs, _key(best), evaluations),
        review_reason=(
            f"No route-slot cleared the {bar:.0%} auto-assign bar "
            f"(best {scored.total_score:.0%}). Surfacing the strongest option for a specialist."
        ),
    )


def _escalate_no_slot(
    customer: CustomerProfile, evaluations: list[CandidateEvaluation]
) -> SlotRecommendation:
    """Feasible route(s) exist, but none produced a candidate delivery slot -- its
    own escalation reason, distinct from the no-feasible-route case."""
    return SlotRecommendation(
        customer_number=customer.customer_number,
        customer_address=customer.address,
        customer_name=customer.name,
        decision=Decision.ESCALATED_NO_FEASIBLE_SLOT,
        total_score=0.0,
        reasoning=(
            "A serviceable route was found, but no delivery window could be constructed "
            "from its committed stops -- a routing specialist is needed to place a slot."
        ),
        rejected_alternatives=_infeasible_lines(evaluations),
        review_reason=(
            "Serviceable route(s) found, but no delivery window could be built from their "
            "committed stops."
        ),
    )


def _escalate_no_feasible(
    customer: CustomerProfile, evaluations: list[CandidateEvaluation]
) -> SlotRecommendation:
    return SlotRecommendation(
        customer_number=customer.customer_number,
        customer_address=customer.address,
        customer_name=customer.name,
        decision=Decision.ESCALATED_NO_FEASIBLE_SLOT,
        total_score=0.0,
        reasoning="No candidate route satisfied all hard constraints.",
        rejected_alternatives=_infeasible_lines(evaluations),
        review_reason="No candidate route satisfied all hard constraints.",
    )


# --- helpers -----------------------------------------------------------------


def _key(option: RouteSlotOption) -> tuple:
    return (option.evaluation.route.route_id, fmt_window(option.scored.slot.window))


def _deterministic_reasoning(option: RouteSlotOption) -> str:
    route = option.evaluation.route
    top = max(option.scored.factor_scores, key=lambda fs: fs.weighted)
    return (
        f"{route.name} ({route.day.value}) at {fmt_window(option.scored.slot.window)} is the "
        f"strongest route-slot overall; {top.detail}."
    )


def _rejected(
    all_pairs: list[RouteSlotOption],
    chosen_key: tuple,
    evaluations: list[CandidateEvaluation],
) -> list[str]:
    out: list[str] = []
    for p in all_pairs:
        if _key(p) == chosen_key:
            continue
        out.append(
            f"{p.evaluation.route.route_id} ({p.evaluation.route.day.value}) "
            f"{fmt_window(p.scored.slot.window)}: route-slot scored {p.scored.total_score:.2f}"
        )
    out.extend(_infeasible_lines(evaluations))
    return out


def _infeasible_lines(evaluations: list[CandidateEvaluation]) -> list[str]:
    lines: list[str] = []
    for ev in evaluations:
        if not ev.feasible:
            failed = ", ".join(
                CONSTRAINT_LABEL.get(c.name, c.name) for c in ev.failed_constraints
            )
            lines.append(f"{ev.route.route_id} ({ev.route.day.value}): infeasible — {failed}")
    return lines
