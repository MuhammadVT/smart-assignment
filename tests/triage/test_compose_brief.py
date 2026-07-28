"""
Tests for the non-agent brief composer (triage/compose.py) used by batch.

All offline. The grounded path is driven with a FAKE ``generate_text`` (patched at
its source module, since ``compose`` imports it lazily), so no backend or
credentials are needed and the deterministic verify/normalize/fallback machinery
is exercised directly.
"""

from __future__ import annotations

from dataclasses import replace

from smart_assignment.integrations.geocoding_client import MockGeocoder
from smart_assignment.pipeline import run_slot_recommendation
from smart_assignment.shared.config import DEFAULT_CONFIG
from smart_assignment.shared.models import CustomerProfile
from smart_assignment.triage.compose import compose_brief
from smart_assignment.triage.context import escalation_context_from_recommendation

_HEADERS = ("SITUATION", "ROOT CAUSE", "OPTIONS", "RECOMMENDATION", "DECISION NEEDED")


def _context(triage_on: bool):
    """Build a real escalation context for Galleria (a deterministic escalation),
    returning the (config, context) pair. ``triage_on`` toggles the compose flag
    only -- step 5 stays deterministic either way."""
    cfg = replace(
        DEFAULT_CONFIG,
        use_grounded_route_slot_escalation=False,
        use_grounded_route_slot_pick=False,
        use_escalation_triage=triage_on,
    )
    profile = CustomerProfile(
        name="Galleria Grill & Catering",
        address="5085 Westheimer Rd, Houston, TX 77056",
        order_quantity_cases=400,
        preferred_slot=None,
    )
    result = run_slot_recommendation(profile, config=cfg, geocoder=MockGeocoder())
    assert result.recommendation.requires_human_review
    context = escalation_context_from_recommendation(
        result.customer, result.candidates_considered, result.recommendation, cfg
    )
    return cfg, context


def _patch_generate_text(monkeypatch, reply):
    """Patch the source ``generate_text`` (compose imports it lazily). ``reply`` may
    be a string (returned for every call) or a callable ``() -> str``."""

    def fake(config, prompt, role=None):
        return reply() if callable(reply) else reply

    monkeypatch.setattr("smart_assignment.shared.llm.generate_text", fake)


def _faithful_brief(context: dict) -> str:
    """A brief citing ONLY grounded facts from ``context`` (order size + a feasible
    route's exact-integer headroom), so ``verify_brief`` passes with no caveat."""
    cand = context["feasible_candidates"][0]
    rid, name, day = cand["route_id"], cand["name"], cand["day"]
    rem = cand["facts"]["remaining_capacity_after"]
    cust = context["customer"]["name"]
    return (
        f"SITUATION\n{cust} — 400 cases — escalated for review.\n\n"
        f"ROOT CAUSE\n{rid} - {name} has only {rem} cases headroom for a 400 case order.\n\n"
        f"OPTIONS (most workable first)\n"
        f"1) {rid} - {name} · {day} — {rem} cases headroom\n\n"
        f"RECOMMENDATION\nStart with {rid} - {name}, the least-disruptive option.\n\n"
        f"DECISION NEEDED\nAssign to {rid} - {name} or hold for a bigger route?"
    )


# --- deterministic floor (flag off) -----------------------------------------


def test_flag_off_returns_deterministic_brief():
    cfg, context = _context(triage_on=False)
    brief = compose_brief(context, cfg)

    for header in _HEADERS:
        assert header in brief
    # A route is named with the "<id> - <name>" convention, and no LLM was called.
    assert "RTE-" in brief
    assert brief.startswith("SITUATION")


# --- grounded path (flag on, fake model) ------------------------------------


def test_grounded_brief_from_a_faithful_model_reply(monkeypatch):
    cfg, context = _context(triage_on=True)
    _patch_generate_text(monkeypatch, lambda: _faithful_brief(context))

    brief = compose_brief(context, cfg)

    for header in _HEADERS:
        assert header in brief
    assert "Unverified" not in brief  # everything grounded -> no caveat
    assert "400 case" in brief


def test_grounded_brief_appends_caveat_when_ungrounded(monkeypatch):
    cfg, context = _context(triage_on=True)
    # The model invents a headroom figure on both the draft and the retry.
    _patch_generate_text(
        monkeypatch,
        "SITUATION\nGalleria — 400 cases — escalated.\n\n"
        "ROOT CAUSE\nThe route has 9999 cases of headroom, plenty of room.\n\n"
        "OPTIONS (most workable first)\n1) plenty of options.\n\n"
        "RECOMMENDATION\nProceed.\n\nDECISION NEEDED\nAssign or hold?",
    )

    brief = compose_brief(context, cfg)

    assert "Unverified" in brief
    assert "9999" in brief


def test_grounded_falls_back_to_deterministic_on_error(monkeypatch):
    cfg, context = _context(triage_on=True)

    def boom(config, prompt, role=None):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr("smart_assignment.shared.llm.generate_text", boom)

    brief = compose_brief(context, cfg)

    # Fell back to the deterministic floor: full layout, no invented figure, no crash.
    for header in _HEADERS:
        assert header in brief
    assert "9999" not in brief
    assert "Unverified" not in brief


def test_empty_or_non_ok_context_yields_no_brief():
    cfg, _ = _context(triage_on=True)
    assert compose_brief({}, cfg) == ""
    assert compose_brief({"ok": False}, cfg) == ""
