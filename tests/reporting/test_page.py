"""
Tests for the overview-page generator (smart_assignment/reporting/page.py).

These guard the "page never drifts from the code" contract: the page is built
from live workflow results, so every sample customer, decision, and the
configured weights must appear in the output.
"""

from __future__ import annotations

import json
from html import escape as html_escape

from smart_assignment.mock_customers import SAMPLE_CUSTOMERS
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.reasoning import DeterministicReasoner
from smart_assignment.reporting.page import (
    build_map_data,
    build_page,
    build_workflow_payload,
    generate,
)
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision


def _results(config):
    reasoner = DeterministicReasoner()
    return [
        run_slot_recommendation(c, config=config, reasoner=reasoner) for c in SAMPLE_CUSTOMERS
    ]


def test_page_includes_every_sample_customer():
    config = Config()
    html = build_page(_results(config), config)
    assert html.startswith("<!DOCTYPE html>")
    for customer in SAMPLE_CUSTOMERS:
        assert html_escape(customer.address) in html
        assert html_escape(customer.name) in html


def test_page_reflects_live_decisions_and_reasoning():
    config = Config()
    results = _results(config)
    html = build_page(results, config)
    # Every decision's actual reasoning text must be present (no drift) --
    # HTML-escaped, since the natural-language text contains apostrophes.
    for result in results:
        assert html_escape(result.recommendation.reasoning) in html
    # All three outcome states are exercised by the sample set.
    decisions = {r.recommendation.decision for r in results}
    assert Decision.RECOMMENDED in decisions
    assert Decision.ESCALATED_LOW_SCORE in decisions
    assert Decision.ESCALATED_NO_FEASIBLE_SLOT in decisions


def test_page_reflects_configured_weights():
    config = Config()
    config.factor_weights["geographic_clustering"] = 0.50
    html = build_page(_results(config), config)
    assert "weight 0.50" in html  # picked up from config, not hard-coded


def test_page_has_architecture_and_agent_emphasis():
    config = Config()
    html = build_page(_results(config), config)
    assert "<svg" in html  # architecture diagram
    assert "AI AGENT" in html
    assert "executed autonomously by the agent" in html
    # A single agent runs all steps — no per-step "agent" badge implying many agents.
    assert "single AI agent performs all five stages" in html
    assert '<span class="abadge">' not in html


def test_delivery_window_is_soft_not_a_hard_constraint():
    config = Config()
    html = build_page(_results(config), config)
    assert "Delivery-window fit" not in html  # removed from hard constraints
    assert "soft preference" in html  # window is now a scoring-only preference


def test_page_states_number_sources():
    config = Config()
    html = build_page(_results(config), config)
    assert "Where the numbers come from" in html
    assert "cluster_reference_miles" in html
    assert "max_utilization_after_assignment" in html
    assert "shared/config.py" in html


def test_page_treats_customers_as_prospects_identified_by_address():
    config = Config()
    results = _results(config)
    html = build_page(results, config)
    # None of the sample customers have a Sysco number yet -- address is the
    # identifier, and the page says so rather than showing a blank/None.
    assert all(c.customer_number is None for c in SAMPLE_CUSTOMERS)
    assert "new prospect — no Sysco number yet" in html
    assert "Sysco customer number (optional)" in html
    # The simulator's run-by-address flow is documented, not run-by-number.
    assert "Enter a mock customer's address" in html


def test_page_has_three_tabs():
    config = Config()
    html = build_page(_results(config), config)
    for tab in ("overview", "architecture", "simulator"):
        assert f'data-tab="{tab}"' in html
        assert f'id="tab-{tab}"' in html


def test_scoring_section_shows_real_formulas():
    config = Config()
    html = build_page(_results(config), config)
    # The scoring dimensions and their code formulas are spelled out.
    assert "Exactly how each dimension is scored" in html
    assert "avg_miles_to_stops" in html
    assert "preferred_window_minutes" in html
    # Slot match gates on the day-of-week term -- wrong day (or no time
    # overlap at all) scores 0, it's not a source of partial credit.
    assert "route_day</b> ≠ <b>preferred_day" in html
    assert "Slot match (day + time)" in html
    # Capacity buffer is a flat-then-decay curve anchored on the safety margin,
    # not a straight "more headroom always wins" ratio.
    assert "capacity_buffer_safety_margin" in html
    assert "up to 75% full" in html  # default margin -> 90% ceiling - 15pp
    assert "15%-point safety margin" in html
    # total_score IS the gating number -- there's no separate confidence
    # formula, and a route's own score is never discounted for a close
    # runner-up (see reasoning.compute_total_score).
    assert "total_score = (" in html
    assert "no separate" in html and "confidence" in html
    assert "Total score threshold (auto-assign bar)" in html

    # The per-step simulator payload shows the scoring arithmetic for a scored route.
    marker = '<script type="application/json" id="workflow-data">'
    start = html.index(marker) + len(marker)
    payload = json.loads(html[start : html.index("</script>", start)])

    bayou_key = SAMPLE_CUSTOMERS[0].lookup_key
    galleria_key = SAMPLE_CUSTOMERS[1].lookup_key
    woodlands_key = SAMPLE_CUSTOMERS[3].lookup_key

    # Bayou stays comfortably under the safe line -> flat branch.
    bayou_joined = " ".join(payload[bayou_key]["steps"][3]["lines"])
    assert payload[bayou_key]["steps"][3]["title"] == "Score & Rank"
    assert "clustering = clamp(" in bayou_joined
    assert "total =" in bayou_joined
    assert "slot match = day(" in bayou_joined  # day-of-week gates the slot score
    assert "capacity buffer = 1.00 flat" in bayou_joined

    # Woodlands is mock-tuned to land in the 75-90% decay band, so the demo
    # actually exercises the decaying branch, not just the flat one.
    woodlands_joined = " ".join(payload[woodlands_key]["steps"][3]["lines"])
    assert "capacity buffer = clamp((" in woodlands_joined

    # Galleria is mock-tuned so her only feasible route's OWN score is
    # mediocre -- a genuine low-total-score escalation, not a tie-breaking
    # artifact between two good options.
    galleria_joined = " ".join(payload[galleria_key]["steps"][3]["lines"])
    assert "capacity buffer = clamp((" in galleria_joined

    # Intake step now surfaces the preferred slot (day + time).
    intake_lines = " ".join(payload[bayou_key]["steps"][0]["lines"])
    assert "Preferred slot (day + time)" in intake_lines
    assert "TUE" in intake_lines


def test_page_has_interactive_simulator_with_payload():
    config = Config()
    html = build_page(_results(config), config)
    # Simulator controls exist.
    assert 'id="run-btn"' in html
    assert 'id="cust-input"' in html
    assert 'id="workflow-data"' in html

    # The embedded JSON payload drives the simulator entirely client-side.
    marker = '<script type="application/json" id="workflow-data">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    payload = json.loads(html[start:end])

    for customer in SAMPLE_CUSTOMERS:
        entry = payload[customer.lookup_key]
        assert len(entry["steps"]) == 5  # the five workflow steps
        assert all(s["action"] and s["title"] for s in entry["steps"])
        assert entry["resultHtml"].strip().startswith("<article")


def test_build_map_data_includes_customer_and_route_geometry():
    config = Config()
    results = _results(config)
    bayou = results[0]  # downtown, clean recommend -- mix of feasible/infeasible routes
    map_data = build_map_data(bayou)

    assert map_data is not None
    assert map_data["customer"]["lat"] == bayou.customer.location.latitude
    assert map_data["customer"]["lng"] == bayou.customer.location.longitude
    assert len(map_data["routes"]) == len(bayou.candidates_considered)

    feasible = [r for r in map_data["routes"] if r["feasible"]]
    infeasible = [r for r in map_data["routes"] if not r["feasible"]]
    assert feasible and infeasible  # this sample exercises both branches

    for r in map_data["routes"]:
        assert "lat" in r["service_center"] and "lng" in r["service_center"]
        assert isinstance(r["stops"], list)
        assert all("lat" in s and "lng" in s for s in r["stops"])
    for r in feasible:
        assert r["total_score"] is not None
    for r in infeasible:
        assert r["total_score"] is None


def test_build_map_data_stops_carry_id_window_and_tier():
    """Each committed stop rides along with its customer number, its committed
    delivery window (open/close), and its customer tier -- which the frontend
    delivery-window timeline reads (windows/tier are optional: {open, close}/str
    or None)."""
    config = Config()
    map_data = build_map_data(_results(config)[0])

    stops = [s for r in map_data["routes"] for s in r["stops"]]
    assert stops  # the sample has committed stops to plot
    for s in stops:
        assert isinstance(s["id"], str) and s["id"]
        w = s["window"]
        assert w is None or (isinstance(w["open"], str) and isinstance(w["close"], str))
        assert s["tier"] is None or isinstance(s["tier"], str)
    # The mock route data populates windows and tiers, so at least one is present.
    assert any(s["window"] is not None for s in stops)
    assert any(s["tier"] for s in stops)


def test_build_map_data_carries_slot_rationale():
    """The map payload includes a "why this time slot" rationale for the
    recommended slot: how the window was placed (its basis) and the SLOT-level
    factors that pick it (slot match + availability, each with its figure). The
    route-level factors (geography, capacity) are excluded -- they explain the
    route, not the slot."""
    from smart_assignment.shared.timeutils import overlap_minutes as _overlap

    config = Config(use_route_slot_scoring=True)
    result = _results(config)[0]  # a clean recommend
    rec = result.recommendation
    winner = next(
        c
        for c in result.candidates_considered
        if c.route.route_id == rec.recommended_route_id
    )
    html = build_map_data(result, config)["rationaleHtml"]

    # Collapsible via a <details>/<summary> (triangle), collapsed by default.
    assert html and '<details class="slot-why">' in html
    assert '<summary class="slot-why-head">' in html
    assert "<details open" not in html  # hidden by default
    assert rec.recommended_route_id in html  # small route context
    # Slot-level factors, each with the concrete figure behind its score.
    assert "Slot availability (openness)" in html
    assert 'class="why-factor-detail"' in html
    # Openness calculation: contention named explicitly, then the 1 / (1 + contention) roll-up.
    assert "contention = Σ harm" in html
    assert "openness = 1 ÷" in html
    # ... and each contending committed stop is listed with its tier + harm.
    overlapping = [
        s for s in winner.route.committed_stops
        if s.delivery_time_window
        and _overlap(winner.chosen_window, s.delivery_time_window) > 0
    ]
    assert overlapping and all(s.customer_number in html for s in overlapping)
    assert "harm" in html
    # Proximity: the nearest committed stops the window was clustered around.
    assert "Proximity" in html and " mi" in html
    # Route-level factors are NOT in the slot rationale.
    assert "Geographic clustering" not in html
    assert "Capacity buffer" not in html


def test_build_map_data_routes_carry_ranked_order():
    """Every route carries a `rank` giving the agent's scored order (feasible
    recommended-first, then infeasible) -- the ranks are a 0..n-1 permutation and
    every feasible route ranks ahead of every infeasible one, so the frontend can
    show the delivery-window panels in the same order as the evaluated-routes list."""
    config = Config()
    map_data = build_map_data(_results(config)[0])
    routes = map_data["routes"]

    ranks = sorted(r["rank"] for r in routes)
    assert ranks == list(range(len(routes)))  # a clean permutation, no gaps/dupes

    feasible_ranks = [r["rank"] for r in routes if r["feasible"]]
    infeasible_ranks = [r["rank"] for r in routes if not r["feasible"]]
    if feasible_ranks and infeasible_ranks:
        assert max(feasible_ranks) < min(infeasible_ranks)


def test_build_map_data_none_without_a_geocoded_customer():
    config = Config()
    result = _results(config)[0]
    result.customer.location = None
    assert build_map_data(result) is None


def test_workflow_payload_embeds_map_data_for_every_sample():
    config = Config()
    for result in _results(config):
        payload = build_workflow_payload(result, config)
        assert payload["map"] is not None
        assert payload["map"]["customer"]["name"]
        assert len(payload["map"]["routes"]) == len(result.candidates_considered)


def test_workflow_payload_splits_result_card_and_routes_section():
    config = Config()
    bayou = _results(config)[0]
    payload = build_workflow_payload(bayou, config)
    # The recommended-route card no longer embeds the routes list -- that moves
    # to routesHtml (rendered separately, below the map in the web app).
    assert "Routes the agent evaluated" not in payload["resultHtml"]
    # routesHtml is a default-open <details> with one rich card per candidate.
    routes = payload["routesHtml"]
    assert routes.startswith('<details class="routes" open>')
    assert f"Routes the agent evaluated ({len(bayou.candidates_considered)})" in routes
    # Feasible routes get the weighted-factor bar chart; the recommended route is tagged.
    assert 'class="factors"' in routes
    assert "★ recommended" in routes
    assert bayou.recommendation.recommended_route_id in routes


def test_route_cards_show_bars_for_feasible_and_checks_for_infeasible():
    from smart_assignment.reporting.page import _route_cards

    config = Config()
    # Katy: all routes infeasible -> no factor bars (they never reach scoring),
    # but every route still shown with its constraint checks and a data-route-id
    # for the web app to colour-match to the map.
    katy = _results(config)[2]
    assert all(not e.feasible for e in katy.candidates_considered)
    html = _route_cards(katy, config)
    assert "INFEASIBLE" in html
    assert 'class="factors"' not in html  # nothing scored
    assert html.count('class="route routecard"') == len(katy.candidates_considered)
    assert html.count("data-route-id=") == len(katy.candidates_considered)


def test_route_slot_cards_and_payload_are_slot_level():
    """With route-slot scoring on, the evaluated section renders one card per
    ROUTE with its candidate slots listed inside (route info not repeated), the
    map payload carries each route's scored slots, and Step 4 shows each factor's
    formula so the math is checkable."""
    from smart_assignment.reporting.page import (
        _route_cards,
        _sim_steps,
        build_map_data,
    )

    config = Config(use_route_slot_scoring=True)
    bayou = _results(config)[0]  # a clean recommend with feasible routes + slots
    feasible = [e for e in bayou.candidates_considered if e.feasible]
    assert feasible and any(e.scored_slots for e in feasible)

    html = _route_cards(bayou, config)
    # One card per ROUTE (feasible + infeasible), not one per slot; each feasible
    # route's slots are listed inside (one .slot-head per candidate slot).
    assert "Routes the agent evaluated" in html
    assert html.count('class="route routecard"') == len(bayou.candidates_considered)
    assert html.count('class="slot-head"') == sum(len(e.scored_slots) for e in feasible)
    assert "★ recommended" in html  # exactly the winning (route, slot) is flagged

    # Map payload: each feasible route carries its scored slots (score-ranked),
    # with exactly one recommended slot overall.
    routes = build_map_data(bayou)["routes"]
    assert all("slots" in r for r in routes)
    feasible_routes = [r for r in routes if r["feasible"]]
    assert all(r["slots"] for r in feasible_routes)
    for r in feasible_routes:
        scores = [s["score"] for s in r["slots"]]
        assert scores == sorted(scores, reverse=True)  # highest score first
    assert sum(1 for r in routes for s in r["slots"] if s["recommended"]) == 1

    # Step 4 exposes the per-factor math (a formula + the cited inputs), each
    # route-slot in a collapsible "show the math" <details> (hidden by default).
    score_step = next(s for s in _sim_steps(bayou, config) if s["title"] == "Score & Rank")
    joined = " ".join(score_step["lines"])
    assert "clamp(1 − avg_mi" in joined  # geo formula
    assert "1 ÷ (1 + Σ tier-harm" in joined  # slot-availability formula
    assert "Route-slot score" in joined and ") ÷ " in joined  # weighted-sum breakdown
    # The toggle label itself is CSS-generated, so the summary span is empty markup.
    assert '<span class="rs-showmath"></span>' in joined
    assert joined.count('<details class="rs-details') == sum(len(e.scored_slots) for e in feasible)


def test_route_slot_card_shows_unscored_slot_match_without_preference():
    """When the prospect states no preferred slot, slot-match is dropped from the
    route-slot total -- but the card still LISTS the factor, marked 'not scored',
    so the user knows it exists and why it wasn't scored."""
    from smart_assignment.reporting.page import _example_card, _route_cards

    config = Config(use_route_slot_scoring=True)
    # Galleria: a sample with preferred_slot=None but a feasible route to score.
    galleria = _results(config)[1]
    assert galleria.customer.preferred_slot is None
    feasible = [e for e in galleria.candidates_considered if e.feasible]
    assert feasible and feasible[0].scored_slots
    # The dropped factor really is absent from the scored breakdown.
    names = {f.name for f in feasible[0].scored_slots[0].factor_scores}
    assert "window_match" not in names

    html = _route_cards(galleria, config)
    assert "Slot match (day + time)" in html  # still listed
    assert "not scored" in html and "no preferred slot" in html
    assert 'class="factor na"' in html

    # The recommendation card's own factor bars carry the same "not scored" pill.
    rec_card = _example_card(galleria, include_routes=False)
    assert 'class="factor na"' in rec_card
    assert "Slot match (day + time)" in rec_card and "not scored" in rec_card


def test_page_embeds_map_data_in_workflow_json():
    config = Config()
    html = build_page(_results(config), config)
    marker = '<script type="application/json" id="workflow-data">'
    start = html.index(marker) + len(marker)
    payload = json.loads(html[start : html.index("</script>", start)])
    for customer in SAMPLE_CUSTOMERS:
        entry = payload[customer.lookup_key]
        assert entry["map"] is not None
        assert "customer" in entry["map"] and "routes" in entry["map"]


def test_generate_writes_file(tmp_path):
    out = generate(output_path=tmp_path / "index.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Smart Assignment" in text
    assert "GENERATED by scripts/generate_page.py" in text


def test_route_slot_page_renders_availability_and_new_threshold():
    """With route-slot scoring on, the page explains the (route, slot) unit, the
    availability factor, and the route-slot auto-assign bar -- not the stale
    route-only narrative."""
    config = Config(use_route_slot_scoring=True)
    html = build_page(_results(config), config)
    assert html.startswith("<!DOCTYPE html>")
    # The route-slot explainer and factor are present...
    assert "Slot availability" in html
    assert "(route, slot)" in html or "route-slot" in html
    # ...the auto-assign bar reflects route_slot_score_threshold (0.55 default)...
    assert f"{config.route_slot_score_threshold:.0%}" in html
    # ...and the stale route-only "three dimensions" narrative is gone.
    assert "three dimensions" not in html
