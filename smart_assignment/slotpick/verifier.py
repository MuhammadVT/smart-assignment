"""
Deterministic verification of a parsed `SlotChoice` against its `SlotPacket`.
No LLM. The chosen index must be one of the enumerated candidates, and every
structured citation must resolve to a real candidate fact and match its value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_assignment.slotpick.evidence import NUMERIC_SLOT_FIELDS, SlotPacket
from smart_assignment.slotpick.schema import SlotChoice

# Facts are serialized at <=4dp and a whole-percent paraphrase of a fraction
# ("70%" for 0.7012) is off by at most 0.005 -- this absorbs faithful rounding
# without admitting a neighboring-but-different number.
_TOL = 0.005

# Fields whose values are fractions of 1 -- the only ones a percent phrasing
# may normalize against. Counts/minutes must match at face value, so a claim
# that is off by 100x (e.g. committed_overlap=200 against a stored 2) can't
# slip through the old unconditional /100 normalization.
_FRACTION_FIELDS = frozenset({"fit_score", "blended_score"})


@dataclass
class SlotVerification:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "; ".join(self.reasons)


def _values_close(field_name: str, cited: float, actual: float) -> bool:
    if abs(cited - actual) <= _TOL:
        return True
    # Percent phrasing of a fraction-valued fact: "70" (or "70%") for a stored
    # 0.7. Gated on the field being a fraction and the cited value actually
    # looking like a percent.
    if field_name in _FRACTION_FIELDS and cited > 1.5:
        return abs(cited / 100.0 - actual) <= _TOL
    return False


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
        if actual is None or not _values_close(c.field, float(c.value), float(actual)):
            reasons.append(
                f"candidate[{c.index}].{c.field}={actual} but citation claims {c.value}"
            )

    return SlotVerification(ok=not reasons, reasons=reasons)
