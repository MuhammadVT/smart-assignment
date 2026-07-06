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
from smart_assignment.reporting.page import build_page, generate
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


def test_generate_writes_file(tmp_path):
    out = generate(output_path=tmp_path / "index.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Smart Assignment" in text
    assert "GENERATED by scripts/generate_page.py" in text
