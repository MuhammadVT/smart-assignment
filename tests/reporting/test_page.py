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
from smart_assignment.reporting.page import build_page, generate
from smart_assignment.shared.config import Config
from smart_assignment.shared.models import Decision
from smart_assignment.workflows.slot_recommendation.pipeline import run_slot_recommendation
from smart_assignment.workflows.slot_recommendation.reasoning import DeterministicReasoner


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
        assert customer.customer_number in html
        assert html_escape(customer.name) in html


def test_page_reflects_live_decisions_and_reasoning():
    config = Config()
    results = _results(config)
    html = build_page(results, config)
    # Every decision's actual reasoning text must be present verbatim (no drift).
    for result in results:
        assert result.recommendation.reasoning in html
    # All three outcome states are exercised by the sample set.
    decisions = {r.recommendation.decision for r in results}
    assert Decision.RECOMMENDED in decisions
    assert Decision.ESCALATED_LOW_CONFIDENCE in decisions
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
    assert "cases_remaining_after_add" in html
    assert "preferred_window_minutes" in html
    # The confidence formula from reasoning.compute_confidence is shown.
    assert "0.6·" in html and "0.5 + 0.5·" in html

    # The per-step simulator payload shows the scoring arithmetic for a scored route.
    marker = '<script type="application/json" id="workflow-data">'
    start = html.index(marker) + len(marker)
    payload = json.loads(html[start : html.index("</script>", start)])
    # 067-100001 (Bayou City Bistro) has a feasible, scored route.
    score_step = payload["067-100001"]["steps"][3]
    assert score_step["title"] == "Score & Rank"
    joined = " ".join(score_step["lines"])
    assert "clustering = clamp(" in joined
    assert "total =" in joined


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
        entry = payload[customer.customer_number]
        assert len(entry["steps"]) == 5  # the five workflow steps
        assert all(s["action"] and s["title"] for s in entry["steps"])
        assert entry["resultHtml"].strip().startswith("<article")


def test_generate_writes_file(tmp_path):
    out = generate(output_path=tmp_path / "index.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Smart Assignment" in text
    assert "GENERATED by scripts/generate_page.py" in text
