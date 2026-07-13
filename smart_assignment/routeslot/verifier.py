"""
Deterministic verification of a parsed `RouteSlotChoice` against its packet. No
LLM. The chosen index must be one of the enumerated route-slot options, and every
structured citation must resolve to a real option fact and match its value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_assignment.routeslot.evidence import NUMERIC_FACT_KEYS, RouteSlotPacket
from smart_assignment.routeslot.schema import RouteSlotChoice

# Facts are rounded to <=4dp; this absorbs rounding without admitting a
# genuinely different number.
_TOL = 0.02


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


def verify_choice(choice: RouteSlotChoice, packet: RouteSlotPacket) -> RouteSlotVerification:
    reasons: list[str] = []

    if not (0 <= choice.chosen_index < packet.n):
        reasons.append(
            f"chosen_index {choice.chosen_index} is not a valid route-slot option "
            f"(expected 0..{packet.n - 1})"
        )

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

    return RouteSlotVerification(ok=not reasons, reasons=reasons)
