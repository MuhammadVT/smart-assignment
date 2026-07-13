"""
The route-slot evidence packet: every feasible (route, slot) pair enumerated as
a flat, indexed menu the grounded decision reasons over. Each option carries its
per-slot factor values (geo/capacity shared from the route; window_match and
slot openness slot-specific) plus the deterministic weighted total as a
*reference*, and the packet names the index the deterministic blend would pick on
its own. Nothing here calls an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from smart_assignment.shared.config import Config
from smart_assignment.shared.constraints import CONSTRAINT_LABEL
from smart_assignment.shared.models import CandidateEvaluation, CustomerProfile, ScoredSlot
from smart_assignment.shared.timeutils import fmt_time, fmt_window

# Numeric fact keys a citation may reference on a route-slot option: the per-slot
# factor values (by canonical name) plus the reference weighted total.
NUMERIC_FACT_KEYS = (
    "geographic_clustering",
    "capacity_buffer",
    "window_match",
    "slot_availability",
    "reference_weighted_score",
)


@dataclass
class RouteSlotOption:
    """One (route, slot) candidate, kept alongside its JSON view so the decision
    can map a chosen index back to the route + slot without re-deriving."""

    evaluation: CandidateEvaluation
    scored: ScoredSlot


@dataclass
class RouteSlotPacket:
    customer: dict
    options: list[dict]
    infeasible: list[dict]
    deterministic_best_index: Optional[int]
    _options: list[RouteSlotOption] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.options)

    def option_at(self, index: int) -> Optional[RouteSlotOption]:
        if 0 <= index < len(self._options):
            return self._options[index]
        return None

    def option_facts(self, index: int) -> Optional[dict]:
        if 0 <= index < len(self.options):
            return self.options[index]["facts"]
        return None

    def as_dict(self) -> dict:
        return {
            "customer": self.customer,
            "route_slot_options": self.options,
            "infeasible_routes": self.infeasible,
            "deterministic_choice_index": self.deterministic_best_index,
        }


def _slot_phrase(customer: CustomerProfile) -> Optional[str]:
    slot = customer.preferred_slot
    return f"{slot.day.value} {fmt_window(slot.window)}" if slot else None


def _facts(scored: ScoredSlot) -> dict:
    facts = {fs.name: round(fs.value, 4) for fs in scored.factor_scores}
    facts["reference_weighted_score"] = round(scored.total_score, 4)
    return facts


def build_route_slot_packet(
    customer: CustomerProfile,
    evaluations: list[CandidateEvaluation],
    config: Config,
) -> RouteSlotPacket:
    """Enumerate every feasible route-slot as an indexed option for the grounded
    decision. Options are ordered by descending deterministic total so index 0 is
    typically the deterministic pick, but the pick is named explicitly."""
    pairs: list[RouteSlotOption] = [
        RouteSlotOption(evaluation=ev, scored=s)
        for ev in evaluations
        if ev.feasible
        for s in ev.scored_slots
    ]
    pairs.sort(key=lambda p: p.scored.total_score, reverse=True)

    options: list[dict] = []
    for i, p in enumerate(pairs):
        route = p.evaluation.route
        slot = p.scored.slot
        options.append(
            {
                "index": i,
                "route_id": route.route_id,
                "route_name": route.name,
                "day": route.day.value,
                "window": fmt_window(slot.window),
                "anchor_time": fmt_time(slot.anchor_time) if slot.anchor_time else None,
                "basis": slot.basis,
                "facts": _facts(p.scored),
                "factor_breakdown": [
                    {"name": fs.name, "value": round(fs.value, 4), "weight": fs.weight,
                     "detail": fs.detail}
                    for fs in p.scored.factor_scores
                ],
            }
        )

    infeasible = [
        {
            "route_id": ev.route.route_id,
            "day": ev.route.day.value,
            "failed_constraints": [
                {"name": CONSTRAINT_LABEL.get(c.name, c.name), "detail": c.detail}
                for c in ev.failed_constraints
            ],
        }
        for ev in evaluations
        if not ev.feasible
    ]

    best_index = 0 if options else None
    return RouteSlotPacket(
        customer={
            "name": customer.name,
            "order_quantity_cases": int(customer.order_quantity_cases),
            "preferred_slot": _slot_phrase(customer),
        },
        options=options,
        infeasible=infeasible,
        deterministic_best_index=best_index,
        _options=pairs,
    )
