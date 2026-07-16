"""
Deterministic verification of a parsed `AddressChoice` against its
`AddressPacket`. No LLM. The chosen index must be one of the enumerated
candidates, and every structured citation must resolve to a real candidate fact
and match its value -- so the model can't pick a candidate that wasn't offered
or justify its pick with an invented number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_assignment.address_resolve.evidence import NUMERIC_ADDRESS_FIELDS, AddressPacket
from smart_assignment.address_resolve.schema import AddressChoice

# similarity is rounded to 4dp; this absorbs rounding without admitting a
# genuinely different number (0.02 would let a citation of one candidate's
# similarity pass against a neighboring candidate's).
_TOL = 0.005


@dataclass
class AddressVerification:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "; ".join(self.reasons)


def verify_choice(choice: AddressChoice, packet: AddressPacket) -> AddressVerification:
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
        if c.field not in NUMERIC_ADDRESS_FIELDS:
            reasons.append(f"citation field {c.field!r} is not a citable address fact")
            continue
        actual = facts.get(c.field)
        if actual is None or abs(float(c.value) - float(actual)) > _TOL:
            reasons.append(
                f"candidate[{c.index}].{c.field}={actual} but citation claims {c.value}"
            )

    return AddressVerification(ok=not reasons, reasons=reasons)
