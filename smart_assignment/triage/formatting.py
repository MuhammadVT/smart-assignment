"""
Deterministic normalization of the triage brief's *layout*.

The brief is written by an LLM, so its formatting varies turn to turn -- one
escalation comes back as a clean multi-line brief, the next as a single run-on
line. This reflows any brief into the one canonical, scannable layout so the
specialist always sees the same structure.

It only moves whitespace and puts the known section headers / option markers /
Action|Trade-off labels on their own lines -- it never changes a word, a number,
or a route. So a brief that is already well-formed is left materially unchanged,
the grounding scan (``triage/verifier.py``) is unaffected (the figures are
identical), and applying it twice is a no-op. Purely mechanical and defensive:
callers wrap it so a formatting hiccup can never break the handoff.
"""

from __future__ import annotations

import re

# The canonical section headers, in order. Matched case-insensitively; an
# optional trailing "(...)" (e.g. "OPTIONS (most workable first)") is preserved,
# and a trailing ":" is dropped.
_SECTION_HEADERS = ("SITUATION", "ROOT CAUSE", "OPTIONS", "RECOMMENDATION", "DECISION NEEDED")

# An option marker: a single digit 1-9 followed by ")", not part of a longer
# number (so "0.90)" or "48%)" in prose is never mistaken for one). A preceding
# newline is absorbed so re-normalizing is a no-op (no accumulating blank lines).
_OPTION_RE = re.compile(r"[ \t]*\n?[ \t]*(?<![\d.])([1-9])\)[ \t]*")

# The per-option sub-labels (also absorbing a preceding newline for idempotency).
_LABEL_RE = re.compile(r"[ \t]*\n?[ \t]*\b(Action|Trade-?off)[ \t]*:[ \t]*", re.IGNORECASE)


def _header_pattern(header: str) -> re.Pattern:
    # Absorb surrounding blanks/newlines, an optional "(...)", and an optional
    # ":", but only when the header is a standalone token (followed by ":", "(",
    # whitespace, or end) -- never a substring of a longer word.
    return re.compile(
        r"[ \t]*\n?[ \t]*"
        + re.escape(header)
        + r"(?=[:\s(]|$)"
        + r"([ \t]*\([^)]*\))?"
        + r"[ \t]*:?[ \t]*\n?[ \t]*",
        re.IGNORECASE,
    )


def _header_repl(header: str):
    def repl(match: re.Match) -> str:
        paren = (match.group(1) or "").strip()
        label = f"{header} {paren}" if paren else header
        return f"\n\n{label}\n"

    return repl


def normalize_brief(text: str) -> str:
    """Reflow a triage brief into the canonical multi-line layout. Idempotent;
    returns the input unchanged when it is empty or has no recognizable sections."""
    if not text or not text.strip():
        return text
    s = text.strip()

    for header in _SECTION_HEADERS:
        s = _header_pattern(header).sub(_header_repl(header), s)

    # Each numbered option on its own line; its Action/Trade-off labels indented
    # beneath it.
    s = _OPTION_RE.sub(lambda m: f"\n{m.group(1)}) ", s)
    s = _LABEL_RE.sub(lambda m: f"\n   {m.group(1)}: ", s)

    # Collapse any run of 3+ newlines to a single blank line, and drop trailing
    # spaces left on a line.
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
