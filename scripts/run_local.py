"""
Local demo / smoke test for the Smart Assignment workflow.

Runs the pipeline over the mock Sysco customers in
`smart_assignment/mock_customers.py` and prints a full, auditable trace for
each: geocoding, the Top-N candidate routes, every hard-constraint outcome,
the weighted scoring breakdown, and the final recommendation or escalation.

Runs fully OFFLINE — no API key required. Reasoning defaults to the LLM
layer, which automatically falls back to a deterministic trace when no
GOOGLE_API_KEY / Vertex credentials are configured.

Run:
    python3 scripts/run_local.py                       # all sample customers
    python3 scripts/run_local.py --customer 067-100002 # one, by customer number
"""

from __future__ import annotations

import argparse

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.models import Decision, RecommendationResult
from smart_assignment.shared.timeutils import fmt_window
from smart_assignment.workflows.slot_recommendation.pipeline import run_slot_recommendation

_RULE = "=" * 78
_DECISION_MARK = {
    Decision.RECOMMENDED: "[RECOMMENDED]",
    Decision.ESCALATED_LOW_SCORE: "[ESCALATE - LOW TOTAL SCORE]",
    Decision.ESCALATED_NO_FEASIBLE_SLOT: "[ESCALATE - NO FEASIBLE SLOT]",
}


def _print_result(result: RecommendationResult) -> None:
    c = result.customer
    rec = result.recommendation
    slot = c.preferred_slot
    pref = f"{slot.day.value} {fmt_window(slot.window)}" if slot else "any"

    print(_RULE)
    print(f"CUSTOMER  {c.name} ({c.customer_number})")
    print(f"  intake    address={c.address!r}")
    print(f"            order={c.order_quantity_cases} cases, preferred_slot={pref}")
    loc = c.location
    print(f"  geocoded  ({loc.latitude:.4f}, {loc.longitude:.4f})")

    print(f"  Top-{len(result.candidates_considered)} candidate routes by proximity:")
    for cand in result.candidates_considered:
        status = "FEASIBLE" if cand.feasible else "infeasible"
        print(
            f"    - {cand.route.route_id} {cand.route.name} "
            f"[{cand.route.day.value}] {cand.distance_miles:.1f} mi -> {status}"
        )
        for oc in cand.constraint_outcomes:
            flag = "ok " if oc.passed else "FAIL"
            print(f"        {flag} {oc.name}: {oc.detail}")
        if cand.feasible:
            factors = "  ".join(
                f"{f.name}={f.value:.2f}(w{f.weight:.2f})" for f in cand.factor_scores
            )
            print(f"        score={cand.total_score:.2f}  [{factors}]")

    print(f"\n  DECISION  {_DECISION_MARK[rec.decision]}  total_score={rec.total_score:.0%}")
    if rec.recommended_route_id:
        print(
            f"            -> {rec.recommended_route_id} ({rec.recommended_route_name}), "
            f"{rec.recommended_day}, window {rec.recommended_window}"
        )
    if rec.review_reason:
        print(f"            review_reason: {rec.review_reason}")
    print(f"  REASONING {rec.reasoning}")
    if rec.rejected_alternatives:
        print("  ALTERNATIVES CONSIDERED:")
        for alt in rec.rejected_alternatives:
            print(f"    - {alt}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Smart Assignment workflow on mock data")
    parser.add_argument(
        "--customer",
        help="Only run for this customer number, e.g. 067-100002 (default: all)",
    )
    args = parser.parse_args()

    customers = SAMPLE_CUSTOMERS
    if args.customer:
        customers = [c for c in customers if c.customer_number == args.customer]
        if not customers:
            raise SystemExit(f"No sample customer with customer number {args.customer!r}")

    print(_RULE)
    print("SMART ASSIGNMENT - slot_recommendation workflow (mock Sysco data)")
    print(
        f"config: top_n={DEFAULT_CONFIG.top_n_candidate_routes} "
        f"max_util={DEFAULT_CONFIG.max_utilization_after_assignment:.0%} "
        f"total_score_threshold={DEFAULT_CONFIG.total_score_threshold:.0%} "
        f"weights={DEFAULT_CONFIG.factor_weights}"
    )

    for customer in customers:
        result = run_slot_recommendation(customer)
        _print_result(result)


if __name__ == "__main__":
    main()
