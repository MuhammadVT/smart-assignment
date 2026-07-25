"""Unit tests for the optional PII scrub."""

from __future__ import annotations

from smart_assignment.feedback.scrub import scrub_context, scrub_text


def test_scrubs_street_address():
    out = scrub_text("Wrong slot for 1200 McKinney St, Houston")
    assert "1200 McKinney St" not in out
    assert "[redacted]" in out
    assert "Wrong slot for" in out  # the quality signal survives


def test_scrubs_email_and_phone():
    out = scrub_text("contact jane.doe@example.com or 713-555-0142")
    assert "example.com" not in out
    assert "555-0142" not in out


def test_preserves_none_and_empty():
    assert scrub_text(None) is None
    assert scrub_text("") == ""


def test_idempotent():
    once = scrub_text("5000 Katy Mills Cir, Katy")
    assert scrub_text(once) == once


def test_scrub_context_only_free_text_values():
    ctx = {
        "name": "Bayou City Bistro",
        "address": "1200 McKinney St, Houston, TX",
        "outcome": "recommend",
        "recommended_route_id": "RTE-4100",
        "order_quantity_cases": 90,
    }
    out = scrub_context(ctx)
    # Free-text keys scrubbed...
    assert "McKinney" not in out["address"]
    # ...structured facts untouched, so curation still works.
    assert out["outcome"] == "recommend"
    assert out["recommended_route_id"] == "RTE-4100"
    assert out["order_quantity_cases"] == 90


def test_scrub_context_handles_empty():
    assert scrub_context(None) == {}
    assert scrub_context({}) == {}
