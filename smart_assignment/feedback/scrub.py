"""
Optional PII scrubbing for feedback notes and captured decision context.

This is a *toggle*, not a policy baked into the pipeline. On a trusted company
network the operator wants the real customer PII as part of the feedback (who the
account was, the actual street address), so scrubbing is turned OFF
(``feedback_scrub_pii=False``) and records are stored verbatim. On an
off-network / shared deployment scrubbing is ON (the default), so the durable log
never persists identifiers.

The scrub is deliberately conservative and dependency-free: it redacts the
patterns that most plausibly carry PII in this domain (email addresses, phone
numbers, and US street addresses) while leaving the *quality signal* -- the
label, the score, and the shape of the note -- intact. It is not a guarantee of
perfect de-identification (no regex scrub is); it is a pragmatic reduction of
casual PII leakage, and the honest place to strengthen it later (e.g. an NER
pass) without touching any call site.

Only the freeform ``note`` and the free-text *values* of a ``context`` dict are
scrubbed. Structured, non-PII fields (labels, scores, outcomes, route ids) pass
through untouched so curation and calibration still work on scrubbed data.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_REDACTED = "[redacted]"

# Email addresses.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone numbers: US-ish 10-digit forms with optional separators / country code.
_PHONE_RE = re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
# Street addresses: a house number followed by a street name and a common suffix.
# Case-insensitive; covers the abbreviations that appear in this repo's fixtures.
_STREET_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\- ]{2,40}?\s"
    r"(?:st|street|ave|avenue|rd|road|dr|drive|blvd|boulevard|ln|lane|"
    r"ct|court|cir|circle|way|pkwy|parkway|hwy|highway|ste|suite|apt)\b\.?",
    re.IGNORECASE,
)

# Context keys whose values are known to be free text that can carry PII, so they
# are scrubbed as prose. Everything else in a context dict is treated as a
# structured, non-PII fact and passed through.
_FREE_TEXT_CONTEXT_KEYS = frozenset({"name", "address", "note"})


def scrub_text(text: Optional[str]) -> Optional[str]:
    """Redact the PII patterns above from a string, preserving ``None``/empty.

    Order matters: address before phone so a street number isn't half-eaten by
    the phone matcher. Idempotent -- re-scrubbing already-redacted text is a
    no-op on the redactions themselves."""
    if not text:
        return text
    scrubbed = _STREET_RE.sub(_REDACTED, text)
    scrubbed = _EMAIL_RE.sub(_REDACTED, scrubbed)
    scrubbed = _PHONE_RE.sub(_REDACTED, scrubbed)
    return scrubbed


def scrub_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of ``context`` with only its free-text PII-bearing values
    scrubbed; structured facts (outcome, route id, score, ...) are left as-is."""
    if not context:
        return {}
    out: Dict[str, Any] = {}
    for key, value in context.items():
        if key in _FREE_TEXT_CONTEXT_KEYS and isinstance(value, str):
            out[key] = scrub_text(value)
        else:
            out[key] = value
    return out
