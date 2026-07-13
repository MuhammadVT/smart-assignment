"""
Deterministic groundedness verification for the escalation-triage *brief*.

The triage sub-agent writes a free-text brief, so -- unlike the grounded-judgment
layer, which verifies structured citations -- there's no citation list to check.
Instead this scans the brief's prose for load-bearing tokens (numbers,
route-ids, day names, and HH:MM clock times) and confirms each is grounded in
the escalation context the brief was built from. It's the same tolerant
approach as ``judgment/verifier.py``'s prose scan, kept self-contained here so
the two packages stay decoupled.

Tolerant by design, to avoid false rejections of faithful prose:
  - route-ids, route NAMES, and the customer name (any of which may contain
    digits, e.g. a numeric route-id "3170" or a name "BT149361-[...]") are
    scrubbed first, so their digits are never mistaken for a fact;
  - percent-vs-fraction is normalized ("87%" grounds against a stored 0.87) --
    but only against fraction-scale context values, and never for a token that
    carries a concrete unit ("84 miles" cannot ground against a stored 0.84),
    so a figure can't silently shift magnitude by 100x;
  - "1,234"-style thousands groups are read whole, so a fabricated "1,110"
    can't pass by splitting into a discarded "1" and a grounded "110";
  - trivially small bare integers (< 10) are ignored (generic counts) UNLESS
    they carry a unit or percent sign ("only 5 cases of headroom" is a
    decision fact and must ground);
  - HH:MM clock times must match a candidate window or the preferred slot
    (contexts stashed before this check existed carry no window list, and are
    then given the old scrub-only treatment);
  - day names must belong to a candidate route or the preferred slot;
  - "route 40"-style mentions must name (or abbreviate) a real candidate.
A token that still can't be grounded is reported -- the caller (the self-check
tool, and the after-model backstop) surfaces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Absolute tolerance for matching a figure against a context value. Facts are
# serialized at <=4 dp and a whole-percent paraphrase of a fraction ("87%" for
# 0.8712) is off by at most 0.005, so this absorbs faithful rounding while
# rejecting a neighboring-but-different number (the old 0.02 let "82%" pass
# against a stored 0.81).
_TOL = 0.005
# Fraction-scale context values a percent phrasing may normalize against
# (utilization can exceed 1 for an overloaded route, e.g. 1.46 -> "146%").
_FRACTION_SCALE_MAX = 2.0

_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# A route-id-shaped token: alphanumerics joined by hyphens, containing a digit.
_HYPHEN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
# A "90-case"/"2-hour" style quantity adjective -- a number glued to lowercase
# words is prose, not a route id. Its numeric part is still verified by the
# number scan, so skipping it here loses nothing.
_QUANTITY_ADJECTIVE_RE = re.compile(r"^\d+(?:\.\d+)?(?:-[a-z]+)+$")
# "route 40" / "Rte #12" style mentions (a bare id after the word "route").
_ROUTE_MENTION_RE = re.compile(r"\b(?:route|rte)s?\s+#?([A-Za-z0-9][A-Za-z0-9-]*)", re.IGNORECASE)
# What immediately follows a number decides how it may be normalized.
_PERCENT_AFTER_RE = re.compile(r"\s*(?:%|percent\b|pct\b)", re.IGNORECASE)
_UNIT_AFTER_RE = re.compile(r"\s*(?:cases?\b|miles?\b|minutes?\b|mins?\b)", re.IGNORECASE)
# Only full day names case-insensitively; 3-letter codes only in upper case, so
# prose like "sat at 81%" can never false-positive.
_DAY_NAME_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", re.IGNORECASE
)
_DAY_CODE_RE = re.compile(r"\b(MON|TUE|WED|THU|FRI|SAT|SUN)\b")
_DAY_CODES = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
_DAY_NAME_TO_CODE = {
    "monday": "MON",
    "tuesday": "TUE",
    "wednesday": "WED",
    "thursday": "THU",
    "friday": "FRI",
    "saturday": "SAT",
    "sunday": "SUN",
}


@dataclass
class BriefVerification:
    ok: bool
    ungrounded_numbers: list[str] = field(default_factory=list)
    ungrounded_routes: list[str] = field(default_factory=list)
    ungrounded_days: list[str] = field(default_factory=list)
    ungrounded_times: list[str] = field(default_factory=list)

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
        if self.ungrounded_days:
            parts.append(
                "days not matching any candidate route or preference: "
                + ", ".join(self.ungrounded_days)
            )
        if self.ungrounded_times:
            parts.append(
                "times not matching any candidate window or preference: "
                + ", ".join(self.ungrounded_times)
            )
        return "⚠ Unverified — " + "; ".join(parts) + ". Treat these with caution."


def _day_code(raw: str) -> str | None:
    token = str(raw).strip()
    if token.upper()[:3] in _DAY_CODES and (len(token) == 3 or token.lower() in _DAY_NAME_TO_CODE):
        return token.upper()[:3]
    return _DAY_NAME_TO_CODE.get(token.lower())


def collect_grounding(context: dict) -> dict:
    """Extract the groundable numbers, route-ids, day codes, window times, and
    scrub-labels from an escalation context (the dict get_escalation_context
    returns). JSON-safe, so it can be stashed in session state for the
    self-check tool and the backstop.
    """
    numbers: list[float] = []
    route_ids: list[str] = []
    labels: list[str] = []  # everything to scrub out before the number scan
    days: list[str] = []  # canonical 3-letter day codes of candidates/preference
    windows: list[str] = []  # window strings whose HH:MM times a brief may quote

    customer = context.get("customer") or {}
    order = customer.get("order_quantity_cases")
    if isinstance(order, (int, float)) and not isinstance(order, bool):
        numbers.append(float(order))
    if customer.get("name"):
        labels.append(str(customer["name"]))
    preferred = customer.get("preferred_slot")
    if isinstance(preferred, str) and preferred:
        windows.append(preferred)
        first = preferred.split()[0]
        if _day_code(first):
            days.append(_day_code(first))

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
        for key in ("day", "day_label"):
            code = _day_code(cand[key]) if cand.get(key) else None
            if code and code not in days:
                days.append(code)
        if cand.get("window"):
            windows.append(str(cand["window"]))
        for value in (cand.get("facts") or {}).values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers.append(float(value))

    return {
        "numbers": numbers,
        "route_ids": route_ids,
        "labels": labels,
        "days": days,
        "windows": windows,
    }


def _number_grounded(
    val: float, numbers: list[float], is_percent: bool, has_unit: bool
) -> bool:
    if any(abs(val - g) <= _TOL for g in numbers):
        return True
    # Percent-vs-fraction, gated: the token must look like a percent (>1.5),
    # the context value must be fraction-scale, and the token must not carry a
    # concrete unit ("84 miles" may not ground against a stored 0.84).
    if val > 1.5 and (is_percent or not has_unit):
        return any(
            0.0 <= g <= _FRACTION_SCALE_MAX and abs(val / 100.0 - g) <= _TOL for g in numbers
        )
    return False


def verify_brief(brief: str, grounding: dict) -> BriefVerification:
    """Scan the brief; report tokens not grounded in ``grounding``."""
    numbers = [float(n) for n in grounding.get("numbers", [])]
    route_ids = {str(r) for r in grounding.get("route_ids", [])}
    labels = [str(x) for x in grounding.get("labels", [])]
    days = grounding.get("days")  # None -> grounding predates the day check
    windows = grounding.get("windows")  # None -> predates the time check
    text = brief or ""

    # Scrub labels (longest first, so a full route name goes before a bare id it
    # may contain), so their digits aren't read as facts.
    scrubbed = text
    for label in sorted(labels, key=len, reverse=True):
        if label:
            scrubbed = scrubbed.replace(label, " ")

    # Clock times: verify against the real windows (when the grounding carries
    # them), then scrub so their digits aren't re-read as facts.
    ungrounded_times: list[str] = []
    if windows is not None:
        allowed_times = {
            (int(m.group(1)), int(m.group(2)))
            for w in windows
            for m in _TIME_RE.finditer(str(w))
        }
        for m in _TIME_RE.finditer(scrubbed):
            if (int(m.group(1)), int(m.group(2))) not in allowed_times:
                ungrounded_times.append(m.group(0))
    scrubbed = _TIME_RE.sub(" ", scrubbed)

    ungrounded_numbers: list[str] = []
    for m in _NUMBER_RE.finditer(scrubbed):
        token = m.group(0)
        val = float(token.replace(",", ""))
        tail = scrubbed[m.end():]
        is_percent = bool(_PERCENT_AFTER_RE.match(tail))
        has_unit = bool(_UNIT_AFTER_RE.match(tail))
        if "." not in token and "," not in token and val < 10 and not is_percent and not has_unit:
            continue  # generic small count ("the other 2 routes")
        if not _number_grounded(val, numbers, is_percent, has_unit):
            ungrounded_numbers.append(token)

    ungrounded_days: list[str] = []
    if days is not None:
        allowed_days = {str(d) for d in days}
        for m in _DAY_NAME_RE.finditer(scrubbed):
            if _DAY_NAME_TO_CODE[m.group(1).lower()] not in allowed_days:
                ungrounded_days.append(m.group(0))
        for m in _DAY_CODE_RE.finditer(scrubbed):
            if m.group(1) not in allowed_days:
                ungrounded_days.append(m.group(1))

    # Scan the *scrubbed* text so real route-ids/names (already removed) and
    # clock times (e.g. "07:00-11:00" -> already scrubbed) can't be mistaken
    # for a route.
    ungrounded_routes: list[str] = []
    for token in _HYPHEN_TOKEN_RE.findall(scrubbed):
        if (
            any(ch.isdigit() for ch in token)
            and token not in route_ids
            and not _QUANTITY_ADJECTIVE_RE.match(token)
        ):
            ungrounded_routes.append(token)
    # "route 40"-style mentions: real ids were scrubbed above, so a digit-
    # bearing token still following "route"/"rte" must at least abbreviate a
    # real id -- otherwise it names a route that doesn't exist, even if the
    # bare number happens to match some unrelated fact.
    lowered_ids = [rid.lower() for rid in route_ids]
    for m in _ROUTE_MENTION_RE.finditer(scrubbed):
        token = m.group(1)
        if (
            any(ch.isdigit() for ch in token)
            and token not in ungrounded_routes
            and not any(token.lower() in rid for rid in lowered_ids)
        ):
            ungrounded_routes.append(token)

    ok = not (ungrounded_numbers or ungrounded_routes or ungrounded_days or ungrounded_times)
    return BriefVerification(
        ok=ok,
        ungrounded_numbers=ungrounded_numbers,
        ungrounded_routes=ungrounded_routes,
        ungrounded_days=ungrounded_days,
        ungrounded_times=ungrounded_times,
    )
