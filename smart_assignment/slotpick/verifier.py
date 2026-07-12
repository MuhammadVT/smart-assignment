"""
Deterministic verification of a parsed `SlotChoice` against its `SlotPacket`.
No LLM. The chosen index must be one of the enumerated candidates, and every
structured citation must resolve to a real candidate fact and match its value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_assignment.slotpick.evidence import NUMERIC_SLOT_FIELDS, SlotPacket
from smart_assignment.slotpick.schema import SlotChoice

# Facts are rounded to <=4dp; this absorbs rounding without admitting a
# genuinely different number.
_TOL = 0.02


@dataclass
class SlotVerification:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "; ".join(self.reasons)


def _values_close(a: float, b: float) -> bool:
    if abs(a - b) <= _TOL:
        return True
    return abs(a / 100.0 - b) <= _TOL or abs(a - b / 100.0) <= _TOL


def verify_choice(choice: SlotChoice, packet: SlotPacket) -> SlotVerification:
    reasons: list[str] = []

    # 1. the pick must be one of the enumerated candidates.
    if not (0 <= choice.chosen_index < packet.n):
        reasons.append(
            f"chosen_index {choice.chosen_index} is not a valid candidate "
            f"(expected 0..{packet.n - 1})"
        )

    # 2. every citation must resolve to a real candidate fact and match it.
    for c in choice.citations:
        facts = packet.candidate_facts(c.index)
        if facts is None:
            reasons.append(f"citation references unknown candidate index {c.index}")
            continue
        if c.field not in NUMERIC_SLOT_FIELDS:
            reasons.append(f"citation field {c.field!r} is not a citable slot fact")
            continue
        actual = facts.get(c.field)
        if actual is None or not _values_close(float(c.value), float(actual)):
            reasons.append(
                f"candidate[{c.index}].{c.field}={actual} but citation claims {c.value}"
            )

    return SlotVerification(ok=not reasons, reasons=reasons)
