"""
Offline tests for the triage brief groundedness verifier (triage/verifier.py).
Pure functions -- no LLM, no ADK.
"""

from __future__ import annotations

from smart_assignment.triage.verifier import collect_grounding, verify_brief

# A context shaped like get_escalation_context's output, using real-world-ish
# ids: a numeric route-id ("3170") and a digit-containing route name.
_CONTEXT = {
    "customer": {"name": "New prospect", "order_quantity_cases": 90},
    "total_score": 0.77,
    "feasible_candidates": [
        {
            "route_id": "3170",
            "name": "BT149361-[D/T SKYLINES]",
            "facts": {
                "utilization_after": 0.81,
                "remaining_capacity_after": 110,
                "distance_miles": 0.5,
            },
        },
        {
            "route_id": "5174",
            "name": "RT77-A",
            "facts": {
                "utilization_after": 0.70,
                "remaining_capacity_after": 40,
                "distance_miles": 3.1,
            },
        },
    ],
    "infeasible_candidates": [],
}


def _grounding():
    return collect_grounding(_CONTEXT)


def test_collect_grounding_pulls_numbers_routes_and_labels():
    g = _grounding()
    assert 0.81 in g["numbers"] and 110.0 in g["numbers"] and 0.77 in g["numbers"]
    assert set(g["route_ids"]) == {"3170", "5174"}
    # Labels (scrubbed before the number scan) include names + the customer name.
    assert "BT149361-[D/T SKYLINES]" in g["labels"]
    assert "New prospect" in g["labels"]


def test_faithful_brief_is_grounded():
    brief = (
        "Route 3170 (BT149361-[D/T SKYLINES]) sits at 81% utilization with 110 cases "
        "of headroom; total score 77%. The next best option was 5174."
    )
    assert verify_brief(brief, _grounding()).ok


def test_percent_paraphrase_grounds_against_a_fraction():
    # 0.81 stored -> "roughly 81%" in prose must ground.
    assert verify_brief("Utilization is roughly 81% after this order.", _grounding()).ok


def test_numeric_route_id_is_not_flagged_as_an_ungrounded_number():
    # "3170" is a route-id, not a stray fact -- it must not trip the number scan.
    result = verify_brief("Route 3170 is the only feasible option.", _grounding())
    assert result.ok


def test_digit_containing_route_name_is_not_flagged():
    result = verify_brief("Route BT149361-[D/T SKYLINES] clusters tightly.", _grounding())
    assert result.ok


def test_hallucinated_number_is_flagged():
    result = verify_brief("This route is at 95% utilization with 250 cases spare.", _grounding())
    assert not result.ok
    assert "95" in result.ungrounded_numbers
    assert "250" in result.ungrounded_numbers


def test_hallucinated_route_is_flagged():
    result = verify_brief("RTE-9999 could also take the order.", _grounding())
    assert not result.ok
    assert "RTE-9999" in result.ungrounded_routes


def test_small_bare_counts_and_clock_times_are_ignored():
    # "the other 2 routes" and a window like "07:00-11:00" must not flag.
    brief = "Compared against the other 2 routes; the run is 07:00-11:00."
    assert verify_brief(brief, _grounding()).ok


def test_caveat_names_the_unverified_tokens():
    result = verify_brief("At 95% with RTE-9999 available.", _grounding())
    caveat = result.caveat()
    assert "95" in caveat and "RTE-9999" in caveat
    assert "caution" in caveat.lower()
