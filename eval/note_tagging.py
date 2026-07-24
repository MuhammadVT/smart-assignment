"""
Phase 0 / Tier 2 -- tag a free-text feedback note with the judge dimension(s) it
concerns, so a holistic 👎 whose note says "the message was confusing" calibrates
``response_clarity`` specifically instead of the outcome-routed default.

Two taggers, composable:

* **Keyword tagger (default).** A transparent, auditable map from phrases to the
  four canonical dimensions. Deterministic, dependency-free, and readable in a PR
  -- you can see exactly why a note got a tag. This is the trustworthy baseline.
* **LLM suggestion (opt-in).** A single constrained classification call that
  *suggests* additional dimensions the keywords missed. It is advisory: it unions
  onto the keyword tags, degrades to nothing on any error (never worse than
  keyword-only), and is off unless explicitly requested -- because an LLM tagger
  is itself a judge, so the operator should sanity-check its suggestions.

Both return a ``Set[str]`` of dimension names, matching the ``NoteTagger`` shape
``judge_calibration`` expects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Set

from eval.judge_calibration import (
    DIM_BRIEF_QUALITY,
    DIM_DECISION_CORRECT,
    DIM_RESPONSE_CLARITY,
    DIM_SLOT_REASONABLE,
    DIMENSIONS,
    NoteTagger,
)

if TYPE_CHECKING:
    from smart_assignment.shared.config import Config

logger = logging.getLogger(__name__)

# Transparent phrase -> dimension map. Substring match on the lower-cased note;
# deliberately readable so a reviewer can audit every tag. Extend in the open.
KEYWORD_MAP = {
    DIM_RESPONSE_CLARITY: (
        "confus", "unclear", "not clear", "hard to read", "hard to understand",
        "jargon", "wording", "readab", "vague", "clarity", "clearly written",
    ),
    DIM_BRIEF_QUALITY: (
        "brief", "escalat", "handoff", "hand-off", "root cause", "missing info",
        "incomplete", "no options", "next step", "specialist",
    ),
    DIM_DECISION_CORRECT: (
        "wrong route", "wrong decision", "wrong call", "should have escalat",
        "should've escalat", "should escalate", "should recommend", "should've recommend",
        "misrouted", "shouldn't have", "should not have",
    ),
    DIM_SLOT_REASONABLE: (
        "slot", "window", "timing", "time was", "wrong time", "wrong day",
        "prefer", "preference", "morning", "afternoon",
    ),
}


def keyword_tags(note: str) -> Set[str]:
    """The dimensions whose keyword phrases appear in ``note`` (case-insensitive)."""
    low = (note or "").lower()
    return {dim for dim, phrases in KEYWORD_MAP.items() if any(p in low for p in phrases)}


def _parse_dimension_list(text: str) -> Set[str]:
    """Parse an LLM reply (a comma/space/line-separated list) into known dimensions.
    Anything not in ``DIMENSIONS`` (incl. 'none') is ignored -- constrained output."""
    tokens = {
        tok.strip().strip(".").lower()
        for chunk in (text or "").replace("\n", ",").split(",")
        for tok in chunk.split()
    }
    return {dim for dim in DIMENSIONS if dim in tokens}


def llm_suggest_tags(note: str, config: "Config") -> Set[str]:
    """A single constrained call suggesting which dimensions ``note`` concerns.
    Returns ``set()`` on any failure (no backend/credentials, parse error), so it
    is never worse than keyword-only. Imports the LLM seam lazily."""
    if not (note or "").strip():
        return set()
    try:
        from smart_assignment.shared.config import ROLE_QUALITY_JUDGE
        from smart_assignment.shared.llm import generate_text

        prompt = (
            "You are labeling a piece of feedback about a delivery-slot recommendation. "
            "Which of these dimensions does the note concern? Choose all that apply, or "
            "'none'.\n"
            f"Dimensions: {', '.join(DIMENSIONS)}\n"
            "- decision_correct: was recommend-vs-escalate right?\n"
            "- slot_reasonable: was the chosen delivery slot/day/time reasonable?\n"
            "- brief_quality: was the escalation/handoff brief useful?\n"
            "- response_clarity: was the customer-facing message clear?\n"
            f"Reply with a comma-separated list of dimension names only.\nNote: {note}"
        )
        text = generate_text(config.for_role(ROLE_QUALITY_JUDGE), prompt)
        return _parse_dimension_list(text)
    except Exception:  # noqa: BLE001 - suggestion is advisory; never break calibration
        logger.debug("LLM note tagging failed; falling back to keyword tags only.", exc_info=True)
        return set()


def make_note_tagger(
    config: Optional["Config"] = None, *, use_llm: bool = False
) -> NoteTagger:
    """A ``NoteTagger``: keyword tags always, plus LLM-suggested tags unioned on
    when ``use_llm`` is set and a config is given. The LLM path is opt-in and
    degrades to keyword-only on any failure."""

    def tagger(note: str) -> Set[str]:
        tags = keyword_tags(note)
        if use_llm and config is not None:
            tags |= llm_suggest_tags(note, config)
        return tags

    return tagger
