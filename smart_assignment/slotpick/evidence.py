"""
The slot evidence packet: a single route's candidate slots enumerated for the
LLM to choose among, plus the customer/preference context and the per-candidate
facts it may reason over. Nothing here calls an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from smart_assignment.shared.models import (
    CandidateEvaluation,
    CustomerProfile,
    SlotOption,
)
from smart_assignment.shared.timeutils import (
    duration_minutes,
    fmt_time,
    fmt_window,
    overlap_minutes,
)

# The numeric fact keys a citation may reference on a candidate slot.
NUMERIC_SLOT_FIELDS = ("fit_score", "committed_overlap", "preference_overlap_minutes")


@dataclass
class SlotPacket:
    """JSON-safe view of one route's candidate slots for the LLM, plus the
    original `SlotOption`s kept for mapping a chosen index back to a window."""

    customer: dict
    route: dict
    preferred_window_minutes: Optional[int]
    candidates: list[dict]
    _options: list[SlotOption] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.candidates)

    def option_at(self, index: int) -> Optional[SlotOption]:
        if 0 <= index < len(self._options):
            return self._options[index]
        return None

    def candidate_facts(self, index: int) -> Optional[dict]:
        if 0 <= index < len(self.candidates):
            return self.candidates[index]["facts"]
        return None

    def as_dict(self) -> dict:
        return {
            "customer": self.customer,
            "route": self.route,
            "preferred_window_minutes": self.preferred_window_minutes,
            "candidates": self.candidates,
        }


def _slot_phrase(customer: CustomerProfile) -> Optional[str]:
    slot = customer.preferred_slot
    return f"{slot.day.value} {fmt_window(slot.window)}" if slot else None


def build_slot_packet(
    customer: CustomerProfile, evaluation: CandidateEvaluation
) -> SlotPacket:
    """Enumerate the chosen route's candidate slots (with per-candidate facts)
    for the grounded slot picker."""
    pref = customer.preferred_slot.window if customer.preferred_slot else None
    pref_minutes = duration_minutes(pref) if pref else None

    candidates: list[dict] = []
    for i, s in enumerate(evaluation.available_slots):
        pref_overlap = overlap_minutes(pref, s.window) if pref else 0
        candidates.append(
            {
                "index": i,
                "window": fmt_window(s.window),
                "anchor_time": fmt_time(s.anchor_time) if s.anchor_time else None,
                "basis": s.basis,
                "facts": {
                    "fit_score": round(s.fit_score, 4),
                    "committed_overlap": s.committed_overlap,
                    "preference_overlap_minutes": int(pref_overlap),
                },
            }
        )

    return SlotPacket(
        customer={
            "name": customer.name,
            "order_quantity_cases": int(customer.order_quantity_cases),
            "preferred_slot": _slot_phrase(customer),
        },
        route={
            "route_id": evaluation.route.route_id,
            "name": evaluation.route.name,
            "day": evaluation.route.day.value,
        },
        preferred_window_minutes=pref_minutes,
        candidates=candidates,
        _options=list(evaluation.available_slots),
    )
