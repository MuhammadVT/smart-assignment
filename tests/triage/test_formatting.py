"""Deterministic normalization of the triage brief's layout."""

from __future__ import annotations

from smart_assignment.triage.formatting import normalize_brief

# A brief the LLM emitted as one run-on line (the reported bug).
_RUN_ON = (
    "SITUATION: New prospect, 400 cases, escalated due to low auto-assign score "
    "(best 48%). ROOT CAUSE: No route-slot met the auto-assign threshold. The best "
    "option, route 3082, scored 48%. OPTIONS: 1) 3082 · WED — 87.75% utilization "
    "and 137 cases of headroom. Action: Secure customer approval for Wednesday "
    "delivery. Trade-off: Customer receives delivery on a non-preferred day. "
    "RECOMMENDATION: Start with Option 1, the only feasible route. DECISION NEEDED: "
    "Can New prospect accept delivery on Wednesday, 07:30-10:30, on route 3082?"
)


def test_run_on_brief_is_reflowed_into_sections():
    out = normalize_brief(_RUN_ON)
    lines = out.splitlines()
    # Every section header ends up on its own line.
    for header in ("SITUATION", "ROOT CAUSE", "OPTIONS", "RECOMMENDATION", "DECISION NEEDED"):
        assert header in lines, f"{header} should be on its own line"
    # The option marker and its labels start their own (indented) lines.
    assert any(line.startswith("1) 3082") for line in lines)
    assert any(line.strip().startswith("Action:") for line in lines)
    assert any(line.strip().startswith("Trade-off:") for line in lines)


def test_normalization_preserves_every_number_and_route():
    out = normalize_brief(_RUN_ON)
    for token in ("400", "48%", "3082", "87.75%", "137", "07:30-10:30"):
        assert token in out


def test_normalization_is_idempotent():
    once = normalize_brief(_RUN_ON)
    assert normalize_brief(once) == once


def test_already_formatted_brief_is_stable():
    brief = (
        "SITUATION\nNew prospect, 90 cases.\n\n"
        "OPTIONS (most workable first)\n"
        "1) 3170 · WED — 92.97% utilization, 40 cases of headroom\n"
        "   Action: Reduce load to clear the 90% ceiling.\n"
        "   Trade-off: offload existing orders.\n"
        "2) 1175 · MON — 93.46% utilization, 44 cases of headroom\n"
        "   Action: Reduce load to clear the 90% ceiling.\n"
        "   Trade-off: offload existing orders.\n\n"
        "RECOMMENDATION\nStart with option 1 (least over at 2.97%).\n\n"
        "DECISION NEEDED\nWhich route absorbs the order?"
    )
    out = normalize_brief(brief)
    # A well-formed brief is left materially unchanged (and stays idempotent).
    assert out == brief
    assert normalize_brief(out) == out


def test_prose_numbers_in_parens_are_not_split_as_options():
    # "(2.97%)" etc. must not be mistaken for an option marker "N)".
    text = "RECOMMENDATION\nStart with option 1 (2.97% over the 0.90 ceiling)."
    out = normalize_brief(text)
    assert "2.97% over the 0.90 ceiling" in out
    # No spurious newline injected mid-parenthetical.
    assert "(2.97% over the 0.90 ceiling)" in out


def test_empty_or_headerless_text_is_returned_stripped():
    assert normalize_brief("") == ""
    assert normalize_brief("   ") == "   "  # whitespace-only: unchanged
    assert normalize_brief("just a sentence.") == "just a sentence."


def test_finalize_callback_normalizes_the_final_brief():
    """The after-model callback reflows a run-on brief into the canonical layout
    (no grounding in state -> just the formatting pass)."""
    from google.adk.models import LlmResponse
    from google.genai import types

    from smart_assignment.triage.agent import _finalize_brief

    class _Ctx:
        state: dict = {}

    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=_RUN_ON)])
    )
    out = _finalize_brief(_Ctx(), resp)
    assert out is not None  # the run-on brief was rewritten
    text = "".join(getattr(p, "text", "") or "" for p in out.content.parts)
    assert "SITUATION" in text.splitlines()
    assert any(line.strip().startswith("Action:") for line in text.splitlines())
