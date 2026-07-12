"""
The *evidence packet* — the structured, raw-facts view of one customer's
candidate routes that the grounded-judgment LLM reasons over.

Design intent: expose every soft-decision-relevant number *raw and separately*,
rather than pre-collapsing them into a single weighted score. The old weighted
`total_score` is still included per feasible candidate, but explicitly labelled
`reference_weighted_score` and documented as NOT the deciding number — it exists
only so the model (and later audits) can see what the legacy formula would have
said.

Every candidate — feasible *and* infeasible — exposes the same flat `facts`
dict, so any citation the model makes (`route_id` + fact key) resolves through
one uniform lookup in `verifier.py`. Infeasible candidates additionally carry
their `failed_constraints`.

The raw facts are recomputed with `constraints.build_context` (a pure function),
so the packet can never drift from what the constraint/scoring code itself saw.
Nothing here calls an LLM or makes a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from smart_assignment.shared.config import Config
from smart_assignment.shared.constraints import CONSTRAINT_LABEL, build_context
from smart_assignment.shared.models import CandidateEvaluation, CustomerProfile
from smart_assignment.shared.timeutils import day_label, duration_minutes, fmt_window

# The numeric fact keys a citation may reference. Kept as a constant so the
# verifier and the prompt documentation stay in lock-step with what's actually
# serialized below.
NUMERIC_FACT_KEYS = (
    "distance_miles",
    "avg_stop_distance_miles",
    "cluster_reference_miles",
    "utilization_after",
    "capacity_ceiling",
    "remaining_capacity_after",
    "order_quantity_cases",
    "window_overlap_minutes",
    "preferred_window_minutes",
    "reference_weighted_score",
)


def _round(value: Optional[float], places: int) -> Optional[float]:
    return None if value is None else round(float(value), places)


def _candidate_facts(customer: CustomerProfile, ev: CandidateEvaluation, config: Config) -> dict:
    """The uniform, flat numeric facts for one candidate (feasible or not).

    Every value here is something the model may cite verbatim; the verifier
    resolves a citation by looking `facts[key]` up and comparing. Facts are
    recomputed from `build_context` so they match the constraint/scoring code.
    """
    ctx = build_context(customer, ev.route, config)
    pref = customer.preferred_slot
    pref_minutes = duration_minutes(pref.window) if pref is not None else None
    return {
        "distance_miles": _round(ctx.distance_miles, 1),
        "avg_stop_distance_miles": _round(ctx.avg_stop_distance_miles, 1),
        "cluster_reference_miles": _round(config.cluster_reference_miles, 1),
        "utilization_after": _round(ctx.utilization_after, 4),
        "capacity_ceiling": _round(config.max_utilization_after_assignment, 4),
        "remaining_capacity_after": int(round(ctx.remaining_capacity_after)),
        "order_quantity_cases": int(customer.order_quantity_cases),
        "window_overlap_minutes": int(ctx.window_overlap_minutes),
        "preferred_window_minutes": pref_minutes,
        "reference_weighted_score": _round(ev.total_score, 4) if ev.feasible else None,
    }


@dataclass
class EvidencePacket:
    """A JSON-safe, LLM-ready view of one customer's evaluated candidates.

    `as_dict()` is what actually goes into the prompt; the dataclass keeps the
    original `CandidateEvaluation`s around (`_evals_by_route_id`) so the judge
    can map a picked `route_id` back to a full evaluation without re-deriving
    anything.
    """

    customer: dict
    feasible_candidates: list[dict]
    infeasible_candidates: list[dict]
    _evals_by_route_id: dict[str, CandidateEvaluation] = field(default_factory=dict)

    @property
    def feasible_route_ids(self) -> list[str]:
        return [c["route_id"] for c in self.feasible_candidates]

    def evaluation_for(self, route_id: str) -> Optional[CandidateEvaluation]:
        return self._evals_by_route_id.get(route_id)

    def candidate_dict(self, route_id: str) -> Optional[dict]:
        for c in self.feasible_candidates + self.infeasible_candidates:
            if c["route_id"] == route_id:
                return c
        return None

    def as_dict(self) -> dict:
        """The plain dict embedded into the judgment prompt (no private fields)."""
        return {
            "customer": self.customer,
            "feasible_candidates": self.feasible_candidates,
            "infeasible_candidates": self.infeasible_candidates,
        }


def _slot_phrase(customer: CustomerProfile) -> Optional[str]:
    slot = customer.preferred_slot
    if slot is None:
        return None
    return f"{slot.day.value} {fmt_window(slot.window)}"


def _candidate_common(customer: CustomerProfile, ev: CandidateEvaluation, config: Config) -> dict:
    route = ev.route
    return {
        "route_id": route.route_id,
        "name": route.name,
        "day": route.day.value,
        "day_label": day_label(route.day),
        "window": fmt_window(ev.chosen_window) if ev.chosen_window else None,
        "facts": _candidate_facts(customer, ev, config),
    }


def build_evidence_packet(
    customer: CustomerProfile,
    evaluations: list[CandidateEvaluation],
    config: Config,
) -> EvidencePacket:
    """Serialize evaluated candidates into an `EvidencePacket` for the LLM."""
    feasible: list[dict] = []
    infeasible: list[dict] = []
    evals_by_id: dict[str, CandidateEvaluation] = {}

    for ev in evaluations:
        evals_by_id[ev.route.route_id] = ev
        entry = _candidate_common(customer, ev, config)
        if ev.feasible:
            entry["factor_breakdown"] = [
                {
                    "name": fs.name,
                    "value": round(fs.value, 4),
                    "weight": fs.weight,
                    "detail": fs.detail,
                }
                for fs in ev.factor_scores
            ]
            entry["reference_only_note"] = (
                "facts.reference_weighted_score is the legacy weighted formula's "
                "output, for context ONLY -- it is not a gate and you need not rank by it."
            )
            feasible.append(entry)
        else:
            entry["failed_constraints"] = [
                {"name": CONSTRAINT_LABEL.get(c.name, c.name), "detail": c.detail}
                for c in ev.failed_constraints
            ]
            infeasible.append(entry)

    customer_dict = {
        "name": customer.name,
        "order_quantity_cases": int(customer.order_quantity_cases),
        "preferred_slot": _slot_phrase(customer),
    }
    return EvidencePacket(
        customer=customer_dict,
        feasible_candidates=feasible,
        infeasible_candidates=infeasible,
        _evals_by_route_id=evals_by_id,
    )
