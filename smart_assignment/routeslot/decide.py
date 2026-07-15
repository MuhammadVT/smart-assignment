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

import json
import logging
from collections import Counter
from typing import Callable, Optional

from smart_assignment.routeslot.evidence import (
    RouteSlotOption,
    RouteSlotPacket,
    build_route_slot_packet,
)
from smart_assignment.routeslot.llm import generate_route_slot_choice
from smart_assignment.routeslot.prompts import (
    build_route_slot_decision_json_retry_prompt,
    build_route_slot_decision_prompt,
    build_route_slot_decision_retry_prompt,
    build_route_slot_json_retry_prompt,
    build_route_slot_prompt,
    build_route_slot_retry_prompt,
)
from smart_assignment.routeslot.schema import (
    VERDICT_AGREE,
    RouteSlotChoice,
    RSDecision,
    parse_route_slot_choice,
)
from smart_assignment.routeslot.verifier import verify_choice
from smart_assignment.shared.config import (
    FACTOR_CAPACITY_BUFFER,
    FACTOR_GEO_CLUSTERING,
    FACTOR_SLOT_AVAILABILITY,
    FACTOR_WINDOW_MATCH,
    Config,
)
from smart_assignment.shared.constraints import CONSTRAINT_LABEL
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    Decision,
    SlotRecommendation,
)
from smart_assignment.shared.timeutils import fmt_window

logger = logging.getLogger(__name__)

# Readable factor names for the deterministic narrative (kept local so routeslot
# doesn't depend on the reporting layer).
_FACTOR_LABEL = {
    FACTOR_GEO_CLUSTERING: "geographic fit",
    FACTOR_CAPACITY_BUFFER: "capacity headroom",
    FACTOR_WINDOW_MATCH: "preferred-window match",
    FACTOR_SLOT_AVAILABILITY: "slot openness",
}


def _factor_label(name: str) -> str:
    return _FACTOR_LABEL.get(name, name.replace("_", " "))


def _route_label(route) -> str:
    """How a route is named to the user everywhere in a recommendation: the stable
    route id AND the human-readable name together, as ``<route id> - <route name>``."""
    return f"{route.route_id} - {route.name}"


# A choice_fn turns (config, prompt) into a raw route-slot-choice dict. Injectable
# so tests drive the grounded path with a fake and no network/credentials.
ChoiceFn = Callable[[Config, str], dict]


def _choice_with_json_retry(fn, config, packet, prompt, json_retry_builder):
    """Get one parsed route-slot choice, retrying ONCE if the reply wasn't JSON.

    The sage generic agent is conversational: it often answers a decision prompt with
    prose or a tool call, so ``fn`` raises ``JSONDecodeError`` before there is anything
    to verify. A single corrective retry that demands JSON-only recovers the common
    case; a second non-JSON reply propagates and the caller falls back deterministically.
    Verification failures are handled separately (a parsed-but-wrong choice never
    reaches here as a JSONDecodeError). An injected test ``choice_fn`` returns a dict
    and never raises, so this is a straight pass-through under test."""
    try:
        raw = fn(config, prompt)
    except json.JSONDecodeError:
        logger.warning(
            "Grounded route-slot reply was not JSON (prose or tool call); retrying once "
            "demanding a JSON-only reply."
        )
        raw = fn(config, json_retry_builder(packet))
    return parse_route_slot_choice(raw)


def decide_route_slot(
    customer: CustomerProfile,
    evaluations: list[CandidateEvaluation],
    config: Config,
    choice_fn: Optional[ChoiceFn] = None,
) -> SlotRecommendation:
    """Decide over route-slot options.

    Non-feasible cases are ALWAYS a deterministic escalation. For the feasible
    ones the path depends on `config.use_grounded_route_slot_escalation`:

      - on (default): the LLM makes the recommend-vs-escalate call itself over all
        feasible route-slots (`_grounded_decide`), grounded + verified + resampled,
        with the deterministic threshold decision as the fallback.
      - off: the prior logic (`_threshold_decide`) -- the 0.55 bar gates
        recommend-vs-escalate and the LLM only picks among the above-bar options.
    """
    if not any(e.feasible for e in evaluations):
        return _escalate_no_feasible(customer, evaluations)

    all_pairs = _all_route_slots(evaluations)
    if not all_pairs:
        # Serviceable route(s), but no delivery window could be constructed.
        return _escalate_no_slot(customer, evaluations)

    if config.use_grounded_route_slot_escalation:
        return _grounded_decide(customer, all_pairs, evaluations, config, choice_fn)
    return _threshold_decide(customer, all_pairs, evaluations, config, choice_fn)


def _threshold_decide(
    customer: CustomerProfile,
    all_pairs: list[RouteSlotOption],
    evaluations: list[CandidateEvaluation],
    config: Config,
    choice_fn: Optional[ChoiceFn],
    allow_llm: bool = True,
) -> SlotRecommendation:
    """The threshold-gated path (rollback, and the fallback for the grounded path):
    the 0.55 bar decides recommend-vs-escalate, and the LLM -- when
    `use_grounded_judgment` is on and `allow_llm` -- only PICKS among the above-bar
    options. `allow_llm=False` forces a pure-deterministic decision (used as the
    grounded path's fallback so a failed LLM call isn't retried)."""
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
    if allow_llm and config.use_grounded_judgment:
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
        choice = _choice_with_json_retry(
            fn, config, packet, build_route_slot_prompt(packet),
            build_route_slot_json_retry_prompt,
        )
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


# --- grounded ESCALATION (the LLM decides recommend-vs-escalate) --------------


def _grounded_decide(
    customer: CustomerProfile,
    all_pairs: list[RouteSlotOption],
    evaluations: list[CandidateEvaluation],
    config: Config,
    choice_fn: Optional[ChoiceFn],
) -> SlotRecommendation:
    """The LLM reasons over ALL feasible route-slots and decides recommend-vs-escalate
    itself (the 0.55 bar is only a reference here). A confident recommendation ships
    on one verified call; an escalate -- or a low-confidence recommend -- is
    resampled `judgment_sample_count` times and combined by `judgment_consensus`
    before it may auto-assign. Any mechanical/verification failure falls back to the
    deterministic threshold decision, so it is never worse than the bar-gated path."""
    packet = build_route_slot_packet(
        customer, evaluations, config, auto_assign_threshold=config.route_slot_score_threshold
    )

    first = _verified_sample(packet, config, choice_fn)
    if first is None:
        # No verified sample (bad creds, malformed reply, persistent grounding
        # failure) -> the deterministic threshold decision, marked as a fallback.
        rec = _threshold_decide(
            customer, all_pairs, evaluations, config, choice_fn=None, allow_llm=False
        )
        rec.grounded_fallback = True
        rec.grounded_fallback_reason = (
            "Grounded LLM reasoning was unavailable, so this shows the deterministic "
            "threshold result. Check the LLM backend (SMART_ASSIGNMENT_LLM_BACKEND) and "
            "its credentials."
        )
        return rec

    if _ships_on_first_call(first, config):
        return _recommend_grounded(customer, first, packet, all_pairs, evaluations, [first])

    # Escalation-side (escalate, or low-confidence recommend): spend the budget.
    samples = [first]
    for _ in range(max(1, config.judgment_sample_count) - 1):
        extra = _verified_sample(packet, config, choice_fn)
        if extra is not None:
            samples.append(extra)
    return _resolve_escalation_side(customer, samples, packet, all_pairs, evaluations, config)


def _verified_sample(
    packet: RouteSlotPacket, config: Config, choice_fn: Optional[ChoiceFn]
) -> Optional[RouteSlotChoice]:
    """One decision sample: call, parse, verify; one corrective retry on a
    verification failure; None on any mechanical failure (all logged)."""
    fn = choice_fn or generate_route_slot_choice
    try:
        choice = _choice_with_json_retry(
            fn, config, packet, build_route_slot_decision_prompt(packet),
            build_route_slot_decision_json_retry_prompt,
        )
        result = verify_choice(choice, packet)
        if not result.ok:
            retry = build_route_slot_decision_retry_prompt(packet, result.as_feedback())
            choice = parse_route_slot_choice(fn(config, retry))
            result = verify_choice(choice, packet)
    except Exception as exc:  # noqa: BLE001 - any backend/parse failure -> fallback
        logger.warning(
            "Grounded route-slot decision failed (%s: %s); falling back to the "
            "deterministic threshold decision. Check SMART_ASSIGNMENT_LLM_BACKEND.",
            type(exc).__name__,
            exc,
        )
        return None
    if not result.ok:
        logger.warning(
            "Grounded route-slot decision ungrounded after one retry (%s); falling back.",
            result.as_feedback(),
        )
        return None
    return choice


def _ships_on_first_call(choice: RouteSlotChoice, config: Config) -> bool:
    """Only a confident recommendation ships on one call. An ESCALATE always
    resamples; a LOW-confidence recommend resamples too, unless the operator opted
    out via `judgment_retry_on_low_confidence_recommend`."""
    if choice.decision is RSDecision.ESCALATE:
        return False
    low_conf = choice.confidence.value == "LOW"
    if low_conf and config.judgment_retry_on_low_confidence_recommend:
        return False
    return True


def _resolve_escalation_side(
    customer: CustomerProfile,
    samples: list[RouteSlotChoice],
    packet: RouteSlotPacket,
    all_pairs: list[RouteSlotOption],
    evaluations: list[CandidateEvaluation],
    config: Config,
) -> SlotRecommendation:
    """Consensus over the k samples' recommend-vs-escalate DECISIONS (not their
    picks): clear back to a recommendation only if the consensus rule is met."""
    recommends = [s for s in samples if s.decision is RSDecision.RECOMMEND]
    n = len(samples)
    if config.judgment_consensus == "majority":
        cleared = len(recommends) * 2 > n
    else:  # "unanimous" (default, precautionary)
        cleared = len(recommends) == n

    if cleared and recommends:
        representative = _modal_recommend(recommends)
        return _recommend_grounded(
            customer, representative, packet, all_pairs, evaluations, samples
        )
    return _escalate_low_confidence(customer, samples, packet, all_pairs, evaluations, config)


def _modal_recommend(recommends: list[RouteSlotChoice]) -> RouteSlotChoice:
    """The sample whose picked option index is most common among recommenders (ties
    broken by sample order) -- so differing-but-good picks aren't disagreement; only
    the recommend/escalate decision is."""
    counts = Counter(s.chosen_index for s in recommends)
    modal_index, _ = counts.most_common(1)[0]
    for s in recommends:
        if s.chosen_index == modal_index:
            return s
    return recommends[0]


def _takes(samples: list[RouteSlotChoice], packet: RouteSlotPacket) -> list[str]:
    """One human-readable line per sample -- the divided reasoning a specialist
    should see, surfaced through `alternative_takes`."""
    lines: list[str] = []
    for s in samples:
        opt = packet.option_at(s.chosen_index)
        where = (
            f"{_route_label(opt.evaluation.route)} {fmt_window(opt.scored.slot.window)}"
            if opt is not None
            else "no valid option"
        )
        lines.append(f"[{s.decision.value}/{s.confidence.value}] {where}: {s.decision_summary}")
    return lines


def _recommend_grounded(
    customer: CustomerProfile,
    representative: RouteSlotChoice,
    packet: RouteSlotPacket,
    all_pairs: list[RouteSlotOption],
    evaluations: list[CandidateEvaluation],
    samples: list[RouteSlotChoice],
) -> SlotRecommendation:
    """A grounded RECOMMEND: map the representative choice to a recommendation and,
    when it came out of a resample, attach the divided takes."""
    chosen = packet.option_at(representative.chosen_index)
    rec = _recommend(customer, chosen, all_pairs, evaluations, packet, representative)
    if len(samples) > 1:
        rec.alternative_takes = _takes(samples, packet)
    return rec


def _escalate_low_confidence(
    customer: CustomerProfile,
    samples: list[RouteSlotChoice],
    packet: RouteSlotPacket,
    all_pairs: list[RouteSlotOption],
    evaluations: list[CandidateEvaluation],
    config: Config,
) -> SlotRecommendation:
    """The grounded reasoner judged no feasible route-slot strong enough to
    auto-assign. Propose the strongest option (the modal pick across samples, else
    the deterministic best) for the specialist, with all reasoned takes."""
    idxs = [s.chosen_index for s in samples if packet.option_at(s.chosen_index) is not None]
    proposed_index = Counter(idxs).most_common(1)[0][0] if idxs else 0
    best = packet.option_at(proposed_index) or all_pairs[0]
    recommends = sum(1 for s in samples if s.decision is RSDecision.RECOMMEND)

    ev = best.evaluation
    scored = best.scored
    rec = SlotRecommendation(
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
            f"Grounded reasoning judged no feasible route-slot strong enough to "
            f"auto-assign: {recommends}/{len(samples)} sample(s) recommended "
            f"(consensus rule: {config.judgment_consensus}). Surfacing the strongest "
            f"option and all reasoned takes for a specialist."
        ),
        alternative_takes=_takes(samples, packet),
    )
    # Structured floor so the escalation card isn't a one-liner either.
    _apply_deterministic_narrative(rec, best, all_pairs)
    return rec


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
    # Deterministic structured floor: always give the recommendation a summary,
    # the top reasons, the runner-up, and the trade-off -- grounded in the real
    # facts, no LLM. This is what a user sees even when grounded reasoning is off
    # or falls back, so the explanation is never just a one-liner.
    _apply_deterministic_narrative(rec, chosen, all_pairs)
    # Grounded enrichment: when the LLM produced a verified choice, its reasoned
    # prose (and AGREE/DIVERGE self-assessment) replaces the deterministic floor.
    if choice is not None:
        _apply_grounded_narrative(rec, choice, packet)
    return rec


def _apply_deterministic_narrative(
    rec: SlotRecommendation, chosen: RouteSlotOption, all_pairs: list[RouteSlotOption]
) -> None:
    """Fill the structured explanation fields from the score breakdown alone --
    the always-available floor beneath the grounded prose."""
    route = chosen.evaluation.route
    scored = chosen.scored
    rec.decision_summary = (
        f"Assign {_route_label(route)} · {route.day.value} · {fmt_window(scored.slot.window)}."
    )
    # One line PER scored factor, in the breakdown's canonical order (geographic
    # fit, capacity headroom, preferred-window match when a preference was stated,
    # slot openness) -- a comprehensive, consistently-structured read, not just the
    # top two. So slot openness and window match are always surfaced, never dropped.
    rec.primary_reasons = [
        f"{_factor_label(fs.name).capitalize()}: {fs.detail}." for fs in scored.factor_scores
    ]

    runner = _runner_up_option(chosen, all_pairs)
    if runner is not None:
        r_route = runner.evaluation.route
        rec.runner_up = (
            f"{_route_label(r_route)} ({r_route.day.value}) "
            f"{fmt_window(runner.scored.slot.window)} "
            f"— route-slot scored {runner.scored.total_score:.2f}"
        )
        rec.key_tradeoff = _deterministic_tradeoff(chosen, runner)


def _runner_up_option(
    chosen: RouteSlotOption, all_pairs: list[RouteSlotOption]
) -> Optional[RouteSlotOption]:
    """The next-best route-slot (all_pairs is score-ranked); None if the pick was
    the only option."""
    chosen_key = _key(chosen)
    for p in all_pairs:
        if _key(p) != chosen_key:
            return p
    return None


def _deterministic_tradeoff(chosen: RouteSlotOption, runner: RouteSlotOption) -> str:
    """One grounded sentence: the winner's score edge, and the one factor (if any)
    the runner-up actually leads on -- so 'why not the alternative' is explicit."""
    c_total = chosen.scored.total_score
    r_total = runner.scored.total_score
    lead = (
        f"Edges out the runner-up on overall route-slot score "
        f"({c_total:.2f} vs {r_total:.2f})"
    )
    chosen_values = {fs.name: fs.value for fs in chosen.scored.factor_scores}
    for fs in sorted(runner.scored.factor_scores, key=lambda f: f.weighted, reverse=True):
        chosen_value = chosen_values.get(fs.name)
        if chosen_value is not None and fs.value > chosen_value + 0.01:
            return (
                f"{lead}; the runner-up is stronger on {_factor_label(fs.name)} "
                f"({fs.value:.2f} vs {chosen_value:.2f}) but weaker overall."
            )
    return f"{lead}; it also matches or leads the runner-up on every scored factor."


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
    return f"{opt['route_id']} - {opt['route_name']} ({opt['day']}) {opt['window']} — {ru.why_not}"


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
        f"{_route_label(route)} ({route.day.value}) at "
        f"{fmt_window(option.scored.slot.window)} is the "
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
            f"{_route_label(p.evaluation.route)} ({p.evaluation.route.day.value}) "
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
            lines.append(f"{_route_label(ev.route)} ({ev.route.day.value}): infeasible — {failed}")
    return lines
