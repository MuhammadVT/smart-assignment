"""
Shared fixtures/helpers for the grounded-judgment tests.

Everything here is offline: real `CandidateEvaluation`s are produced by running
the deterministic front of the pipeline (intake -> geo_lookup -> evaluate) over
the mock customers, and the LLM itself is always a fake `judgment_fn` so no
network or credentials are ever touched.
"""

from __future__ import annotations

from datetime import time

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.integrations.route_capacity_client import fetch_candidate_routes
from smart_assignment.pipeline import evaluate_candidates, geo_lookup, intake
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import CustomerProfile, DayOfWeek, PreferredSlot

# The three canonical mock scenarios (mirrors tests/test_pipeline.py).
CLEAR_RECOMMEND = CustomerProfile(
    name="Bayou City Bistro",
    address="1200 McKinney St, Houston, TX 77010",
    order_quantity_cases=90,
    preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(7, 0), time(10, 0))),
)
LOW_SCORE = CustomerProfile(
    name="Galleria Grill & Catering",
    address="5085 Westheimer Rd, Houston, TX 77056",
    order_quantity_cases=400,
    preferred_slot=None,
)
NO_FEASIBLE = CustomerProfile(
    name="Katy Prairie Steakhouse",
    address="24600 Katy Fwy, Katy, TX 77494",
    order_quantity_cases=260,
    preferred_slot=PreferredSlot(DayOfWeek.TUE, (time(6, 0), time(8, 0))),
)


def evaluations_for(customer: CustomerProfile, config: Config | None = None):
    """Run the deterministic front of the pipeline; return (customer, evals)."""
    config = config or Config()
    customer = intake(customer)
    candidates = geo_lookup(customer, fetch_candidate_routes(), MockGeocoder(), config)
    return customer, evaluate_candidates(customer, candidates, config)


def feasible_ids(evaluations) -> list[str]:
    return [e.route.route_id for e in evaluations if e.feasible]


class FakeJudgmentFn:
    """A stand-in `judgment_fn` returning canned dicts and counting calls.

    Pass a single dict (repeated), a list of dicts (consumed in order, last one
    repeated), or include an `Exception` instance to simulate a backend error.
    """

    def __init__(self, outputs):
        self._outputs = outputs if isinstance(outputs, list) else [outputs]
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, config, prompt):
        self.prompts.append(prompt)
        out = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return out


def recommend(route_id, confidence="HIGH", rationale="This is the best available fit."):
    return {
        "decision": "RECOMMEND",
        "confidence": confidence,
        "recommended_route_id": route_id,
        "rationale": rationale,
        "citations": [],
    }


def escalate(route_id=None, confidence="LOW", rationale="Too marginal to auto-assign."):
    return {
        "decision": "ESCALATE",
        "confidence": confidence,
        "recommended_route_id": route_id,
        "rationale": rationale,
        "citations": [],
    }
