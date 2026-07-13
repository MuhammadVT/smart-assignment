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
            "day": "TUE",
            "day_label": "Tuesday",
            "window": "07:00-11:00",
            "facts": {
                "utilization_after": 0.81,
                "remaining_capacity_after": 110,
                "distance_miles": 0.5,
            },
        },
        {
            "route_id": "5174",
            "name": "RT77-A",
            "day": "THU",
            "day_label": "Thursday",
            "window": "12:00-15:00",
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


# ---------------------------------------------------------------------------
# Adversarial-hardening regressions: cases that slipped the original checks.
# ---------------------------------------------------------------------------


def test_collect_grounding_pulls_days_and_windows():
    g = _grounding()
    assert set(g["days"]) == {"TUE", "THU"}
    assert "07:00-11:00" in g["windows"] and "12:00-15:00" in g["windows"]


def test_unit_bearing_number_cannot_launder_through_percent_normalization():
    # Stored distance 0.5 must not ground "50 miles"; stored headroom 110 must
    # not ground "1.1 miles" or "11,000 cases".
    for brief in (
        "The stop is 50 miles from the route.",
        "The stop adds 1.1 miles of stem distance.",
        "Route 3170 has 11,000 cases of headroom.",
    ):
        assert not verify_brief(brief, _grounding()).ok, brief


def test_percent_paraphrase_still_grounds_but_not_against_counts():
    g = _grounding()
    # "81%" for the stored fraction 0.81 stays fine...
    assert verify_brief("Utilization lands at 81% after the order.", g).ok
    # ...but a percent can only normalize against fraction-scale values, so
    # "4000%"-style magnitude games against the 40-case headroom fail.
    result = verify_brief("Utilization lands at 4000% after the order.", g)
    assert not result.ok
    assert "4000" in result.ungrounded_numbers


def test_near_tolerance_but_different_percent_fails():
    # 82% is not a faithful paraphrase of a stored 0.81 -- the old 0.02
    # tolerance accepted it.
    result = verify_brief("Route 3170 sits at 82% utilization.", _grounding())
    assert not result.ok
    assert "82" in result.ungrounded_numbers


def test_comma_grouped_fabricated_number_fails():
    # "1,110" must be read as 1110, not split into "1" and a grounded "110".
    result = verify_brief("Freeing 1,110 cases would clear it.", _grounding())
    assert not result.ok
    assert "1,110" in result.ungrounded_numbers


def test_small_integer_with_unit_is_checked():
    result = verify_brief("Route 5174 has only 7 cases of headroom.", _grounding())
    assert not result.ok
    assert "7" in result.ungrounded_numbers


def test_route_mention_without_hyphen_fails_even_if_number_grounds():
    # "Route 40" names a nonexistent route; 40 happening to equal the stored
    # headroom must not save it.
    result = verify_brief("Route 40 can absorb this order.", _grounding())
    assert not result.ok
    assert "40" in result.ungrounded_routes


def test_wrong_day_is_flagged_and_real_day_passes():
    result = verify_brief("The Friday run has ample headroom.", _grounding())
    assert not result.ok
    assert "Friday" in result.ungrounded_days
    assert verify_brief("The Tuesday run has ample headroom.", _grounding()).ok


def test_invented_clock_window_is_flagged_and_real_window_passes():
    result = verify_brief("Deliver in the 13:00-17:00 window instead.", _grounding())
    assert not result.ok
    assert "17:00" in result.ungrounded_times
    assert verify_brief("The 07:00-11:00 window on 3170 works.", _grounding()).ok


def test_grounding_without_day_or_window_keys_skips_those_checks():
    # Grounding dicts stashed in session state before the day/time checks
    # existed must not start failing briefs retroactively.
    legacy = {k: v for k, v in _grounding().items() if k in ("numbers", "route_ids", "labels")}
    assert verify_brief("Deliver 13:00-17:00 on Friday.", legacy).ok


def test_caveat_names_days_and_times():
    result = verify_brief("Deliver 13:00-17:00 on Friday.", _grounding())
    caveat = result.caveat()
    assert "Friday" in caveat and "17:00" in caveat


def test_quantity_adjective_is_not_a_route_but_is_number_checked():
    # "90-case order" is prose, not a route id -- but its numeric part must
    # still ground (90 is the real order size; 999 is not).
    assert verify_brief("Headroom easily covers the 90-case order.", _grounding()).ok
    result = verify_brief("Headroom easily covers the 999-case order.", _grounding())
    assert not result.ok
    assert "999" in result.ungrounded_numbers
