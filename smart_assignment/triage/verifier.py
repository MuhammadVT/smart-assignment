"""
Deterministic groundedness verification for the escalation-triage *brief*.

The triage sub-agent writes a free-text brief, so -- unlike the grounded-judgment
layer, which verifies structured citations -- there's no citation list to check.
Instead this scans the brief's prose for load-bearing tokens (numbers and
route-ids) and confirms each is grounded in the escalation context the brief was
built from. It's the same tolerant approach as ``judgment/verifier.py``'s prose
scan, kept self-contained here so the two packages stay decoupled.

Tolerant by design, to avoid false rejections of faithful prose:
  - route-ids, route NAMES, and the customer name (any of which may contain
    digits, e.g. a numeric route-id "3170" or a name "BT149361-[...]") are
    scrubbed first, so their digits are never mistaken for a fact;
  - HH:MM clock times are scrubbed;
  - percent-vs-fraction is normalized ("87%" grounds against a stored 0.87);
  - trivially small bare integers (< 10) are ignored (generic counts).
A number or route-id that still can't be grounded is reported -- the caller
(the self-check tool, and the after-model backstop) surfaces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches within the tolerance the evidence packet rounds facts to.
_TOL = 0.02
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# A route-id-shaped token: alphanumerics joined by hyphens, containing a digit.
_HYPHEN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")


@dataclass
class BriefVerification:
    ok: bool
    ungrounded_numbers: list[str] = field(default_factory=list)
    ungrounded_routes: list[str] = field(default_factory=list)

    def caveat(self) -> str:
        """A human-facing warning line naming the unverified tokens."""
        parts = []
        if self.ungrounded_numbers:
            parts.append(
                "figures not found in the evaluation trace: "
                + ", ".join(self.ungrounded_numbers)
            )
        if self.ungrounded_routes:
            parts.append(
                "routes not among the candidates: " + ", ".join(self.ungrounded_routes)
            )
        return "⚠ Unverified — " + "; ".join(parts) + ". Treat these with caution."


def collect_grounding(context: dict) -> dict:
    """Extract the groundable numbers, route-ids, and scrub-labels from an
    escalation context (the dict get_escalation_context returns). JSON-safe, so
    it can be stashed in session state for the self-check tool and the backstop.
    """
    numbers: list[float] = []
    route_ids: list[str] = []
    labels: list[str] = []  # everything to scrub out before the number scan

    customer = context.get("customer") or {}
    order = customer.get("order_quantity_cases")
    if isinstance(order, (int, float)) and not isinstance(order, bool):
        numbers.append(float(order))
    if customer.get("name"):
        labels.append(str(customer["name"]))

    total_score = context.get("total_score")
    if isinstance(total_score, (int, float)) and not isinstance(total_score, bool):
        numbers.append(float(total_score))

    candidates = (context.get("feasible_candidates") or []) + (
        context.get("infeasible_candidates") or []
    )
    for cand in candidates:
        rid = cand.get("route_id")
        if rid:
            route_ids.append(str(rid))
            labels.append(str(rid))
        if cand.get("name"):
            labels.append(str(cand["name"]))
        for value in (cand.get("facts") or {}).values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers.append(float(value))

    return {"numbers": numbers, "route_ids": route_ids, "labels": labels}


def _values_close(a: float, b: float) -> bool:
    if abs(a - b) <= _TOL:
        return True
    return abs(a / 100.0 - b) <= _TOL or abs(a - b / 100.0) <= _TOL


def verify_brief(brief: str, grounding: dict) -> BriefVerification:
    """Scan the brief; report numbers/route-ids not grounded in ``grounding``."""
    numbers = [float(n) for n in grounding.get("numbers", [])]
    route_ids = {str(r) for r in grounding.get("route_ids", [])}
    labels = [str(x) for x in grounding.get("labels", [])]
    text = brief or ""

    # Scrub labels (longest first, so a full route name goes before a bare id it
    # may contain) and clock times, so their digits aren't read as facts.
    scrubbed = text
    for label in sorted(labels, key=len, reverse=True):
        if label:
            scrubbed = scrubbed.replace(label, " ")
    scrubbed = re.sub(r"\d{1,2}:\d{2}", " ", scrubbed)

    ungrounded_numbers: list[str] = []
    for token in _NUMBER_RE.findall(scrubbed):
        val = float(token)
        if "." not in token and val < 10:  # ignore small bare counts
            continue
        candidates = [val, val / 100.0, val * 100.0]
        if not any(any(_values_close(cv, g) for g in numbers) for cv in candidates):
            ungrounded_numbers.append(token)

    # Scan the *scrubbed* text so real route-ids/names (already removed) and
    # clock times (e.g. "07:00-11:00" -> "00-11") can't be mistaken for a route.
    ungrounded_routes: list[str] = []
    for token in _HYPHEN_TOKEN_RE.findall(scrubbed):
        if any(ch.isdigit() for ch in token) and token not in route_ids:
            ungrounded_routes.append(token)

    ok = not ungrounded_numbers and not ungrounded_routes
    return BriefVerification(
        ok=ok,
        ungrounded_numbers=ungrounded_numbers,
        ungrounded_routes=ungrounded_routes,
    )
