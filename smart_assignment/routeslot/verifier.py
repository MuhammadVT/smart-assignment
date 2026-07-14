"""
Deterministic verification of a parsed `RouteSlotChoice` against its packet. No
LLM. Three layers, all deterministic:

  1. Selection safety: the chosen index (and the runner-up index) must be real
     enumerated route-slot options.
  2. Structured citations: every {index, field, value} citation must resolve to a
     real option fact and match its value within tolerance.
  3. Grounded prose: the free-text explanation is scanned for numbers (including
     "1,234"-style thousands), route-ids and "route N" mentions, day names, and
     HH:MM clock times, and every one must be grounded in the packet -- so a
     figure the ops manager reads is never fabricated, even outside the citation
     list. Percent phrasings normalize only against fraction-scale values and
     never for unit-bearing tokens ("84 miles" can't launder through a stored
     0.84), and small integers carrying a unit or percent sign are checked. This
     is the same tolerant approach as ``triage/verifier.py``'s brief scan, kept
     self-contained here so the two packages stay decoupled.

It also checks the model's self-assessment is honest: the AGREE/DIVERGE verdict
must match whether the pick actually equals the deterministic default, and the
trade-off / runner-up must be present when more than one option was offered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from smart_assignment.routeslot.evidence import NUMERIC_FACT_KEYS, RouteSlotPacket
from smart_assignment.routeslot.schema import (
    VERDICT_AGREE,
    VERDICT_DIVERGE,
    RouteSlotChoice,
)

# Absolute tolerance for matching a stated number against a packet value. Facts
# are serialized at <=4dp and a whole-percent paraphrase of a fraction ("80%"
# for 0.8012) is off by at most 0.005, so this absorbs faithful rounding while
# rejecting a neighboring-but-different number (the old 0.02 let "82%" pass
# against a stored 0.80).
_TOL = 0.005
# Every citable route-slot fact is a fraction of 1 (factor values + the weighted
# total), so any of them may be phrased as a percent -- but only in the
# percent->fraction direction, and only when the cited value actually looks
# like a percent. The old bidirectional /100 let a citation 100x off pass.
_FRACTION_FIELDS = frozenset(NUMERIC_FACT_KEYS)
# Fraction-scale packet values a percent phrasing in prose may normalize against.
_FRACTION_SCALE_MAX = 2.0

# "1,234"-style thousands groups are captured whole, so a fabricated "1,150"
# can't pass by splitting into a discarded "1" and a coincidentally-grounded
# "150".
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
# A route-id-shaped token: alphanumerics joined by hyphens, containing a digit.
_HYPHEN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+")
# A "90-case"/"2-hour" style quantity adjective -- prose, not a route id; its
# numeric part is still verified by the number scan.
_QUANTITY_ADJECTIVE_RE = re.compile(r"^\d+(?:\.\d+)?(?:-[a-z]+)+$")
# "route 40" / "Rte #12" style mentions (a bare id after the word "route").
_ROUTE_MENTION_RE = re.compile(r"\b(?:route|rte)s?\s+#?([A-Za-z0-9][A-Za-z0-9-]*)", re.IGNORECASE)
# What immediately follows a number decides how it may be normalized.
_PERCENT_AFTER_RE = re.compile(r"\s*(?:%|percent\b|pct\b)", re.IGNORECASE)
_UNIT_AFTER_RE = re.compile(r"\s*(?:cases?\b|miles?\b|minutes?\b|mins?\b)", re.IGNORECASE)
# Only full day names case-insensitively; 3-letter codes only in upper case, so
# prose like "sat at 80%" can never false-positive.
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
class RouteSlotVerification:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "; ".join(self.reasons)


def _values_close(field_name: str, cited: float, actual: float) -> bool:
    if abs(cited - actual) <= _TOL:
        return True
    if field_name in _FRACTION_FIELDS and cited > 1.5:
        return abs(cited / 100.0 - actual) <= _TOL
    return False


def _packet_numbers(packet: RouteSlotPacket) -> list[float]:
    """Every number the model was actually shown -- factor values, weights,
    reference scores, the customer's order size, and figures embedded in the
    human-readable "detail" strings (factor details, failed-constraint details),
    which the model may faithfully quote. A stated figure is grounded only if it
    matches one of these."""
    numbers: list[float] = []

    def walk(value: object) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            numbers.append(float(value))
        elif isinstance(value, str):
            for tok in _NUMBER_RE.findall(value):
                numbers.append(float(tok.replace(",", "")))
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str) and k != "detail":
                    continue  # ids/names/windows are scrubbed labels, not facts
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(packet.as_dict())
    return numbers


def _packet_labels(packet: RouteSlotPacket) -> list[str]:
    """String tokens whose digits are NOT facts (route-ids, names, windows, days),
    scrubbed before the number scan so, e.g., a route-id ``BT149361`` or a window
    ``09:10-12:10`` is never mistaken for an ungrounded figure."""
    labels: list[str] = []
    customer = packet.customer or {}
    for key in ("name", "preferred_slot"):
        if customer.get(key):
            labels.append(str(customer[key]))
    for opt in packet.options:
        for key in ("route_id", "route_name", "day", "window", "anchor_time", "basis"):
            if opt.get(key):
                labels.append(str(opt[key]))
    for inf in packet.infeasible:
        for key in ("route_id", "day"):
            if inf.get(key):
                labels.append(str(inf[key]))
    return labels


def _allowed_route_ids(packet: RouteSlotPacket) -> set[str]:
    ids = {str(o["route_id"]) for o in packet.options if o.get("route_id")}
    ids |= {str(i["route_id"]) for i in packet.infeasible if i.get("route_id")}
    return ids


def _allowed_days(packet: RouteSlotPacket) -> set[str]:
    days = {str(o["day"]) for o in packet.options if o.get("day")}
    days |= {str(i["day"]) for i in packet.infeasible if i.get("day")}
    pref = (packet.customer or {}).get("preferred_slot")
    if isinstance(pref, str) and pref.split() and pref.split()[0] in _DAY_CODES:
        days.add(pref.split()[0])
    return days


def _allowed_times(packet: RouteSlotPacket) -> set[tuple[int, int]]:
    sources = []
    for opt in packet.options:
        sources.extend([opt.get("window"), opt.get("anchor_time")])
    pref = (packet.customer or {}).get("preferred_slot")
    if isinstance(pref, str):
        sources.append(pref)
    times: set[tuple[int, int]] = set()
    for text in sources:
        if isinstance(text, str):
            for m in _TIME_RE.finditer(text):
                times.add((int(m.group(1)), int(m.group(2))))
    return times


def _number_grounded(
    val: float, numbers: list[float], is_percent: bool, has_unit: bool
) -> bool:
    if any(abs(val - g) <= _TOL for g in numbers):
        return True
    # Percent-vs-fraction, gated: the token must look like a percent (>1.5),
    # the packet value must be fraction-scale, and the token must not carry a
    # concrete unit ("84 miles" may not ground against a stored 0.84).
    if val > 1.5 and (is_percent or not has_unit):
        return any(
            0.0 <= g <= _FRACTION_SCALE_MAX and abs(val / 100.0 - g) <= _TOL for g in numbers
        )
    return False


def _scan_prose(choice: RouteSlotChoice, packet: RouteSlotPacket) -> list[str]:
    """Tolerant prose scan (same approach as ``triage/verifier.py``): every
    load-bearing number, route-id, day name, and HH:MM time stated anywhere the
    ops manager reads it must be grounded in the packet."""
    reasons: list[str] = []
    # Drop near-zero facts: matching a stated figure against ~0 is meaningless and
    # (with the percent tolerance) would ground almost anything small.
    numbers = [g for g in _packet_numbers(packet) if abs(g) > _TOL]
    labels = _packet_labels(packet)
    route_ids = _allowed_route_ids(packet)
    allowed_days = _allowed_days(packet)
    allowed_times = _allowed_times(packet)
    text = " ".join(choice.prose_fields())

    # Scrub labels (longest first, so a full route name goes before a bare id it
    # may contain), so their digits aren't read as facts.
    scrubbed = text
    for label in sorted(labels, key=len, reverse=True):
        if label:
            scrubbed = scrubbed.replace(label, " ")

    # Clock times: verify against the real windows/anchors, then scrub them so
    # their digits aren't re-read as facts.
    for m in _TIME_RE.finditer(scrubbed):
        if (int(m.group(1)), int(m.group(2))) not in allowed_times:
            reasons.append(
                f"prose cites time {m.group(0)!r} that matches no option window, "
                f"anchor, or preferred slot"
            )
    scrubbed = _TIME_RE.sub(" ", scrubbed)

    for m in _NUMBER_RE.finditer(scrubbed):
        token = m.group(0)
        val = float(token.replace(",", ""))
        tail = scrubbed[m.end():]
        is_percent = bool(_PERCENT_AFTER_RE.match(tail))
        has_unit = bool(_UNIT_AFTER_RE.match(tail))
        if "." not in token and "," not in token and val < 10 and not is_percent and not has_unit:
            continue  # generic small count (e.g. "2 stops")
        if not _number_grounded(val, numbers, is_percent, has_unit):
            reasons.append(f"prose states figure {token!r} not found in the evidence")

    for m in _DAY_NAME_RE.finditer(scrubbed):
        if _DAY_NAME_TO_CODE[m.group(1).lower()] not in allowed_days:
            reasons.append(
                f"prose mentions {m.group(0)!r} but no route-slot option or preferred "
                f"slot falls on that day"
            )
    for m in _DAY_CODE_RE.finditer(scrubbed):
        if m.group(1) not in allowed_days:
            reasons.append(
                f"prose mentions {m.group(1)!r} but no route-slot option or preferred "
                f"slot falls on that day"
            )

    # Route-id-shaped tokens (real ids/names were scrubbed above) and "route 40"
    # style mentions must name (or abbreviate) a real candidate route.
    lowered_ids = [rid.lower() for rid in route_ids]
    flagged: set[str] = set()
    for token in _HYPHEN_TOKEN_RE.findall(scrubbed):
        if (
            any(ch.isdigit() for ch in token)
            and token not in route_ids
            and not _QUANTITY_ADJECTIVE_RE.match(token)
        ):
            flagged.add(token)
    for m in _ROUTE_MENTION_RE.finditer(scrubbed):
        token = m.group(1)
        if (
            any(ch.isdigit() for ch in token)
            and token not in flagged
            and not any(token.lower() in rid for rid in lowered_ids)
        ):
            flagged.add(token)
    for token in sorted(flagged):
        reasons.append(f"prose references route {token!r} that is not among the candidates")

    return reasons


def _check_citations(choice: RouteSlotChoice, packet: RouteSlotPacket) -> list[str]:
    reasons: list[str] = []
    for c in choice.citations:
        facts = packet.option_facts(c.index)
        if facts is None:
            reasons.append(f"citation references unknown option index {c.index}")
            continue
        if c.field not in NUMERIC_FACT_KEYS:
            reasons.append(f"citation field {c.field!r} is not a citable route-slot fact")
            continue
        actual = facts.get(c.field)
        if actual is None or not _values_close(c.field, float(c.value), float(actual)):
            reasons.append(
                f"option[{c.index}].{c.field}={actual} but citation claims {c.value}"
            )
    return reasons


def _check_structure(choice: RouteSlotChoice, packet: RouteSlotPacket) -> list[str]:
    """Selection safety + honesty of the self-assessment and completeness of the
    trade-off narrative."""
    reasons: list[str] = []

    if not (0 <= choice.chosen_index < packet.n):
        reasons.append(
            f"chosen_index {choice.chosen_index} is not a valid route-slot option "
            f"(expected 0..{packet.n - 1})"
        )
        return reasons  # nothing else is meaningful without a valid pick

    multi = packet.n >= 2
    ru = choice.runner_up
    if multi:
        if not choice.key_tradeoff:
            reasons.append("key_tradeoff is required when more than one option is offered")
        if ru is None:
            reasons.append("runner_up is required when more than one option is offered")
    if ru is not None:
        if not (0 <= ru.index < packet.n):
            reasons.append(
                f"runner_up.index {ru.index} is not a valid route-slot option "
                f"(expected 0..{packet.n - 1})"
            )
        elif ru.index == choice.chosen_index:
            reasons.append("runner_up.index must differ from the chosen option")

    default_idx = packet.deterministic_best_index
    verdict = choice.vs_deterministic_default
    if verdict is not None and default_idx is not None:
        agrees = choice.chosen_index == default_idx
        if verdict.verdict == VERDICT_AGREE and not agrees:
            reasons.append(
                f"verdict AGREE but chosen_index {choice.chosen_index} != deterministic "
                f"default {default_idx}"
            )
        if verdict.verdict == VERDICT_DIVERGE and agrees:
            reasons.append(
                f"verdict DIVERGE but chosen_index {choice.chosen_index} equals the "
                f"deterministic default"
            )
        if verdict.verdict == VERDICT_DIVERGE and not verdict.note:
            reasons.append("verdict DIVERGE requires a justifying note")
    return reasons


def verify_choice(choice: RouteSlotChoice, packet: RouteSlotPacket) -> RouteSlotVerification:
    reasons = _check_structure(choice, packet)
    reasons += _check_citations(choice, packet)
    reasons += _scan_prose(choice, packet)
    return RouteSlotVerification(ok=not reasons, reasons=reasons)
