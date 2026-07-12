"""
Slot selection strategies and the post-decision refinement hook.

`SlotSelector.select` picks the final slot for a chosen route from that route's
candidate menu (`CandidateEvaluation.available_slots`):

  - `DeterministicSlotSelector` returns the slot the deterministic blend already
    chose (`evaluation.chosen_window`) -- the default.
  - `GroundedSlotSelector` lets an LLM pick a candidate by index (constrained to
    the enumerated menu, grounded + verified), and falls back to the
    deterministic pick on any failure -- so it is never worse than the default.

`refine_slot(...)` applies the configured selector to the winning route of an
already-built `SlotRecommendation`, updating only the presented slot (window,
basis, rationale) -- never the route or the score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from smart_assignment.shared.config import Config
from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    SlotRecommendation,
    Window,
)
from smart_assignment.shared.timeutils import fmt_window
from smart_assignment.slotpick.evidence import build_slot_packet
from smart_assignment.slotpick.llm import generate_slot_choice
from smart_assignment.slotpick.prompts import build_slot_prompt, build_slot_retry_prompt
from smart_assignment.slotpick.schema import parse_slot_choice
from smart_assignment.slotpick.verifier import verify_choice

logger = logging.getLogger(__name__)

# A choice_fn turns (config, prompt) into a raw slot-choice dict. Injectable so
# tests drive the grounded selector with a fake and no network/credentials.
ChoiceFn = Callable[[Config, str], dict]


@dataclass(frozen=True)
class SlotPick:
    window: Window
    basis: str
    rationale: Optional[str] = None


class SlotSelector(Protocol):
    def select(
        self, customer: CustomerProfile, evaluation: CandidateEvaluation, config: Config
    ) -> Optional[SlotPick]: ...


def _deterministic_pick(evaluation: CandidateEvaluation) -> Optional[SlotPick]:
    if evaluation.chosen_window is None:
        return None
    return SlotPick(evaluation.chosen_window, evaluation.window_basis, None)


class DeterministicSlotSelector:
    """Return the slot the deterministic blend already chose (no LLM)."""

    def select(
        self, customer: CustomerProfile, evaluation: CandidateEvaluation, config: Config
    ) -> Optional[SlotPick]:
        return _deterministic_pick(evaluation)


class GroundedSlotSelector:
    """Let an LLM pick a candidate slot by index, grounded + verified, with the
    deterministic pick as the fallback on any failure."""

    def __init__(self, choice_fn: Optional[ChoiceFn] = None):
        self._choice_fn = choice_fn or generate_slot_choice

    def select(
        self, customer: CustomerProfile, evaluation: CandidateEvaluation, config: Config
    ) -> Optional[SlotPick]:
        fallback = _deterministic_pick(evaluation)
        if not evaluation.available_slots:
            return fallback  # nothing to choose among

        packet = build_slot_packet(customer, evaluation, config)
        try:
            choice = parse_slot_choice(self._choice_fn(config, build_slot_prompt(packet)))
            result = verify_choice(choice, packet)
            if not result.ok:
                retry_prompt = build_slot_retry_prompt(packet, result.as_feedback())
                choice = parse_slot_choice(self._choice_fn(config, retry_prompt))
                result = verify_choice(choice, packet)
        except Exception as exc:  # noqa: BLE001 - any backend/parse failure -> fallback
            logger.warning(
                "Grounded slot selection failed (%s: %s); using the deterministic slot. "
                "Check SMART_ASSIGNMENT_LLM_BACKEND and its credentials.",
                type(exc).__name__,
                exc,
            )
            return fallback

        if not result.ok:
            logger.warning(
                "Grounded slot choice ungrounded after one retry (%s); using the "
                "deterministic slot.",
                result.as_feedback(),
            )
            return fallback

        option = packet.option_at(choice.chosen_index)
        if option is None:  # defensive; verifier already checked the range
            return fallback
        return SlotPick(option.window, option.basis, choice.rationale)


def default_slot_selector(config: Config) -> Optional[SlotSelector]:
    """The grounded selector when enabled, else None (keep the deterministic
    slot already on the recommendation -- refine_slot then no-ops)."""
    if config.use_grounded_slot_selection:
        return GroundedSlotSelector()
    return None


def refine_slot(
    recommendation: SlotRecommendation,
    evaluations: list[CandidateEvaluation],
    customer: CustomerProfile,
    config: Config,
    selector: Optional[SlotSelector] = None,
) -> SlotRecommendation:
    """Refine the winning route's presented slot in place using the configured
    selector. No-op when slot selection isn't enabled, there's no recommended
    route, or the selector returns nothing. Never touches the route or the score.
    """
    selector = selector if selector is not None else default_slot_selector(config)
    if selector is None or not recommendation.recommended_route_id:
        return recommendation

    winner = next(
        (e for e in evaluations if e.route.route_id == recommendation.recommended_route_id),
        None,
    )
    if winner is None:
        return recommendation

    pick = selector.select(customer, winner, config)
    if pick is not None and pick.window is not None:
        recommendation.recommended_window = fmt_window(pick.window)
        recommendation.recommended_window_basis = pick.basis or None
        recommendation.recommended_window_rationale = pick.rationale
    return recommendation
