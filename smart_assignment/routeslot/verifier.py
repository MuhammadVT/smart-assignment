"""
Deterministic verification of a parsed `RouteSlotChoice` against its packet. No
LLM. Three layers, all deterministic:

  1. Selection safety: the chosen index (and the runner-up index) must be real
     enumerated route-slot options.
  2. Structured citations: every {index, field, value} citation must resolve to a
     real option fact and match its value within tolerance.
  3. Grounded prose: the free-text explanation is scanned for numbers, and every
     one must be a real fact the packet actually contains -- so a figure the ops
     manager reads is never fabricated, even outside the citation list. This is
     the same tolerant approach as ``triage/verifier.py``'s brief scan, kept
     self-contained here so the two packages stay decoupled.

It also checks the model's self-assessment is honest: the AGREE/DIVERGE verdict
must match whether the pick actually equals the deterministic default, and the
trade-off / runner-up must be present when more than one option was offered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from smart_assignment.routeslot.evidence import NUMERIC_FACT_KEYS, RouteSlotPacket
from smart_assignment.routeslot.schema import (
    VERDICT_AGREE,
    VERDICT_DIVERGE,
    RouteSlotChoice,
)

# Facts are rounded to <=4dp; this absorbs rounding without admitting a
# genuinely different number.
_TOL = 0.02
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class RouteSlotVerification:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "; ".join(self.reasons)


def _values_close(a: float, b: float) -> bool:
    if abs(a - b) <= _TOL:
        return True
    return abs(a / 100.0 - b) <= _TOL or abs(a - b / 100.0) <= _TOL


def _packet_numbers(packet: RouteSlotPacket) -> list[float]:
    """Every number the model was actually shown -- factor values, weights,
    reference scores, and the customer's order size. A stated figure is grounded
    only if it matches one of these."""
    numbers: list[float] = []

    def walk(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            numbers.append(float(value))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(packet.as_dict())
    return numbers


def _packet_labels(packet: RouteSlotPacket) -> list[str]:
    """String tokens whose digits are NOT facts (route-ids, names, windows, days),
    scrubbed before the number scan so, e.g., a route-id ``BT149361`` or a window
    ``09:10-12:10`` is never mistaken for an ungrounded figure."""
    labels: list[str] = []
    customer = packet.customer or {}
    for key in ("name", "preferred_slot"):
        if customer.get(key):
            labels.append(str(customer[key]))
    for opt in packet.options:
        for key in ("route_id", "route_name", "day", "window", "anchor_time", "basis"):
            if opt.get(key):
                labels.append(str(opt[key]))
    return labels


def _ungrounded_numbers(choice: RouteSlotChoice, packet: RouteSlotPacket) -> list[str]:
    # Drop near-zero facts: matching a stated figure against ~0 is meaningless and
    # (with the percent tolerance) would ground almost anything small.
    numbers = [g for g in _packet_numbers(packet) if abs(g) > _TOL]
    labels = _packet_labels(packet)
    text = " ".join(choice.prose_fields())

    # Scrub labels (longest first, so a full route name goes before a bare id it
    # may contain) and HH:MM clock times, so their digits aren't read as facts.
    scrubbed = text
    for label in sorted(labels, key=len, reverse=True):
        if label:
            scrubbed = scrubbed.replace(label, " ")
    scrubbed = re.sub(r"\d{1,2}:\d{2}", " ", scrubbed)

    ungrounded: list[str] = []
    for token in _NUMBER_RE.findall(scrubbed):
        val = float(token)
        if "." not in token and val < 10:  # ignore small bare counts (e.g. "2 stops")
            continue
        # _values_close already tolerates percent-vs-fraction in both directions.
        if not any(_values_close(val, g) for g in numbers):
            ungrounded.append(token)
    return ungrounded


def _check_citations(choice: RouteSlotChoice, packet: RouteSlotPacket) -> list[str]:
    reasons: list[str] = []
    for c in choice.citations:
        facts = packet.option_facts(c.index)
        if facts is None:
            reasons.append(f"citation references unknown option index {c.index}")
            continue
        if c.field not in NUMERIC_FACT_KEYS:
            reasons.append(f"citation field {c.field!r} is not a citable route-slot fact")
            continue
        actual = facts.get(c.field)
        if actual is None or not _values_close(float(c.value), float(actual)):
            reasons.append(
                f"option[{c.index}].{c.field}={actual} but citation claims {c.value}"
            )
    return reasons


def _check_structure(choice: RouteSlotChoice, packet: RouteSlotPacket) -> list[str]:
    """Selection safety + honesty of the self-assessment and completeness of the
    trade-off narrative."""
    reasons: list[str] = []

    if not (0 <= choice.chosen_index < packet.n):
        reasons.append(
            f"chosen_index {choice.chosen_index} is not a valid route-slot option "
            f"(expected 0..{packet.n - 1})"
        )
        return reasons  # nothing else is meaningful without a valid pick

    multi = packet.n >= 2
    ru = choice.runner_up
    if multi:
        if not choice.key_tradeoff:
            reasons.append("key_tradeoff is required when more than one option is offered")
        if ru is None:
            reasons.append("runner_up is required when more than one option is offered")
    if ru is not None:
        if not (0 <= ru.index < packet.n):
            reasons.append(
                f"runner_up.index {ru.index} is not a valid route-slot option "
                f"(expected 0..{packet.n - 1})"
            )
        elif ru.index == choice.chosen_index:
            reasons.append("runner_up.index must differ from the chosen option")

    default_idx = packet.deterministic_best_index
    verdict = choice.vs_deterministic_default
    if verdict is not None and default_idx is not None:
        agrees = choice.chosen_index == default_idx
        if verdict.verdict == VERDICT_AGREE and not agrees:
            reasons.append(
                f"verdict AGREE but chosen_index {choice.chosen_index} != deterministic "
                f"default {default_idx}"
            )
        if verdict.verdict == VERDICT_DIVERGE and agrees:
            reasons.append(
                f"verdict DIVERGE but chosen_index {choice.chosen_index} equals the "
                f"deterministic default"
            )
        if verdict.verdict == VERDICT_DIVERGE and not verdict.note:
            reasons.append("verdict DIVERGE requires a justifying note")
    return reasons


def verify_choice(choice: RouteSlotChoice, packet: RouteSlotPacket) -> RouteSlotVerification:
    reasons = _check_structure(choice, packet)
    reasons += _check_citations(choice, packet)

    ungrounded = _ungrounded_numbers(choice, packet)
    if ungrounded:
        reasons.append(
            "rationale states figures not found in the evidence: " + ", ".join(ungrounded)
        )

    return RouteSlotVerification(ok=not reasons, reasons=reasons)
